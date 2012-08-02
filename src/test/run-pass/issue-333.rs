fn quux<T: copy>(x: T) -> T { let f = id::<T>; return f(x); }

fn id<T: copy>(x: T) -> T { return x; }

fn main() { assert (quux(10) == 10); }
