// Check that we enforce WF conditions also for types in fns.

struct MustBeCopy<T:Copy> {
    t: T
}

struct Bar<T> {
    // needs T: Copy
    x: fn(MustBeCopy<T>) //~ ERROR E0277
}

fn main() { }
