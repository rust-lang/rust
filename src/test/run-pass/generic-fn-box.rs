

fn f<T>(x: @T) -> @T { return x; }

fn main() { let x = f(@3); log(debug, *x); }
