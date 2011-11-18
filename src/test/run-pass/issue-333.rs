fn quux<copy T>(x: T) -> T { let f = bind id::<T>(_); ret f(x); }

fn id<copy T>(x: T) -> T { ret x; }

fn main() { assert (quux(10) == 10); }
