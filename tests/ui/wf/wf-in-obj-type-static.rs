// Check that we enforce WF conditions also for types in fns.


#![allow(dead_code)]

trait Object<T> { }

struct MustBeCopy<T:Copy> {
    t: T
}

struct Foo<T> {
    // needs T: 'static
    x: dyn Object<&'static T> //~ ERROR E0310
}


fn main() { }
