// Check that we enforce WF conditions also for types in fns.
//
// check-pass

#![allow(dead_code)]

trait Object<T> { }

struct MustBeCopy<T:Copy> {
    t: T
}

struct Foo<T> {
    x: dyn Object<&'static T>
}


fn main() { }
