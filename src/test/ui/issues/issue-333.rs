// run-pass

fn quux<T>(x: T) -> T { let f = id::<T>; return f(x); }

fn id<T>(x: T) -> T { return x; }

pub fn main() { assert_eq!(quux(10), 10); }
