fn quux<@T>(x: T) -> T { let f = id::<T>; ret f(x); }

fn id<@T>(x: T) -> T { ret x; }

fn main() { assert (quux(10) == 10); }
