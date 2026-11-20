//! This test checks that calling `.clone()` on a type that does
//! not implement the `Clone` trait results in a compilation error.
//! The `NotClone` and AlsoNotClone structs do not derive or
//! implement `Clone`, so attempting to clone them should fail.

struct NotClone {
    i: isize,
}

fn not_clone(i: isize) -> NotClone {
    NotClone { i }
}

struct AlsoNotClone {
    i: isize,
    j: NotClone,
}

fn also_not_clone(i: isize) -> AlsoNotClone {
    AlsoNotClone {
        i,
        j: NotClone { i: i },
    }
}

fn main() {
    let x = not_clone(10);
    let _y = x.clone();
    //~^ ERROR no method named `clone` found

    let x = also_not_clone(10);
    let _y = x.clone();
    //~^ ERROR no method named `clone` found
}
