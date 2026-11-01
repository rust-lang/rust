// Check that we enforce WF conditions also for types in fns.

struct MustBeCopy<T:Copy> {
    t: T
}

struct Foo<T> {
    // needs T: 'static
    x: fn() -> MustBeCopy<T> //~ ERROR E0277
}

fn main() { }
