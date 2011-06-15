


// -*- rust -*-
fn f[T, U](&T x, &U y) -> tup(T, U) { ret tup(x, y); }

fn main() { log f(tup(3, 4, 5), 4)._0._0; log f(5, 6)._0; }