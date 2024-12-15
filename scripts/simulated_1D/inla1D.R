library(INLA)
library(inlabru)
library(ggplot2)
library(mgcv)
library(fmesher)



data(Poisson2_1D)

cd <- countdata2

cd <- cd[c("x", "count")]

train <- read.csv("1D_train_1.csv", header = TRUE, sep = ",")

test <- read.csv("1D_test_1.csv", header = TRUE, sep = ",")

cd <- data.frame(x = train$S, count = train$Y)

start_time <- Sys.time()

x <- seq(0, 1, by = 0.001) # this sets mesh points - try others if you like
mesh1D <- fm_mesh_1d(x, boundary = "free")


the_spde <- inla.spde2.pcmatern(mesh1D,
                                prior.range = c(0.2, 0.9),
                                prior.sigma = c(1, 0.01)
)

comp <- ~ field(x, model = the_spde) + Intercept(1, prec.li)

fit2.bru <- bru(
  comp,
  like(count ~ .,
       data = cd,
       family = "gaussian"
  )
)

xs <- c(train$S, test$S)


x4pred <- data.frame(x = xs)
pred2.bru <- predict(fit2.bru,
                     x4pred,
                     x ~ field + Intercept
)

plot(xs, pred2.bru$q0.975)
plot(cd$x, cd$count)


error_variance <- 1/fit2.bru$summary.hyperpar[,4][1]
y_hat_variance <- (pred2.bru$sd)^2 + error_variance

l <- pred2.bru$mean - 1.96 * sqrt(y_hat_variance)
u <- pred2.bru$mean + 1.96 * sqrt(y_hat_variance)

output <- data.frame(y_hat = pred2.bru$mean, L = l, U = u)

run_time <- Sys.time() - start_time


write.csv(output, "inla1_output.csv", row.names = F, col.names = NA, sep = ",")
write.csv(run_time, "inla1_time.csv", row.names = F, col.names = NA, sep = ",")






