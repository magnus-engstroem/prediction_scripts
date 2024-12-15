library(INLA)
library(inlabru)
library(ggplot2)
library(mgcv)
library(fmesher)



start_time <- Sys.time()

train <- read.csv("train.csv", header = FALSE, sep = ",")
names(train) <- c("long", "lat", "Y")


train_data <- sf::st_as_sf(
  data.frame(long = train$long, lat = train$lat),
  coords = c("long", "lat"))
  
train_data$Y <- train$Y



test <- read.csv("test.csv", header = FALSE, sep = ",")
names(test) <- c("long", "lat", "Y")

test_data <- sf::st_as_sf(
  data.frame(long = test$long, lat = test$lat),
  coords = c("long", "lat"))




bnd <- spoly(data.frame(long = c(0, 1, 1, 0), lat = c(0, 0, 1, 1)),)

mesh <- fm_mesh_2d_inla(boundary = bnd, max.edge = 0.1)


ggplot() +
  geom_fm(data = mesh)

matern <-
  inla.spde2.pcmatern(mesh,
                      prior.sigma = c(1, 0.5),
                      prior.range = c(0.2, 0.9)
  )

cmp <- Y ~ field(geometry, model = matern) + Intercept(1)

fit <- bru(cmp, data = train_data, family = "gaussian")

pred <- predict(fit, test_data, ~ field + Intercept)

error_variance <- 1/fit$summary.hyperpar[,4][1]
y_hat_variance <- (pred$sd)^2 + error_variance

l <- pred$mean - 1.96 * sqrt(y_hat_variance)
u <- pred$mean + 1.96 * sqrt(y_hat_variance)

output <- data.frame(y_hat = pred$mean, L = l, U = u)

run_time <- Sys.time() - start_time


write.csv(output, "inla1_test.csv", row.names = F, col.names = NA, sep = ",")
write.csv(run_time, "inla1_time.csv", row.names = F, col.names = NA, sep = ",")








