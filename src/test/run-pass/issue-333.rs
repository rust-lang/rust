fn quux<T: Copy>(x: T) -> T { let f = id::<T>; return f(x); }

fn id<T: Copy>(x: T) -> T { return x; }

fn main() { assert (quux(10) == 10); }
