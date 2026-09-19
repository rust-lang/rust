//@ check-pass
// Ensure that `size_of::<Thing>()` as a discriminant does not cause a cycle error.

enum Thing {
    Variant = size_of::<Thing>() as isize,
}

fn main() {}
