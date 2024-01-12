// Check that we enforce WF conditions related to regions also for
// types in fns.

#![allow(dead_code)]

struct Foo<T> {
    x: fn() -> &'static T,
}

struct Bar<T> {
    x: fn(&'static T),
}

fn not_static<T>() {
    let _: Foo<T>; //~ ERROR E0310
    let _: Bar<T>; //~ ERROR E0310
}

fn main() { }
