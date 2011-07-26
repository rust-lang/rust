


// -*- rust -*-
fn f[T, U](&T x, &U y) -> rec(T a, U b) { ret rec(a=x, b=y); }

fn main() { log f(rec(x=3, y=4, z=5), 4).a.x; log f(5, 6).a; }