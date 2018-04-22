##KPprototypes custer

The result of cluster can be shown by R plot:
```
library(grid)
library(ggplot2)

s1 <- read.csv('s1.csv', header=T)
s2<- read.csv('s2.csv', header=T)
p1 <- ggplot(s1, aes(X0,X1, color=as.factor(X4))) + geom_point()
p2 <- ggplot(s2, aes(X0,X1, color=as.factor(X4))) + geom_point()
grid.newpage()
vp1 <- viewport(x=0, y=0.5, width=1,height=0.5, just=c('left','bottom'))
vp2 <- viewport(x=0, y=0, width=1,height=0.5, just=c('left','bottom'))
print(p1, vp=vp1)
print(p2, vp=vp2)
```
![KPrototypes cluster](https://github.com/Alxe1/KPrototypes/blob/master/1.png)
