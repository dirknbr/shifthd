# Shift HD function

This is a Pyton implementation of the shift function which compares two distributions.

Orginal in R: https://github.com/GRousselet/blog/blob/master/shift_function/wilcox_modified.txt

It does this by taking deciles and then using the Harrell-Davis (1982) quantile estimator.

```
out = shifthd(x, y) # x and y are vectors
plot(out)
```

