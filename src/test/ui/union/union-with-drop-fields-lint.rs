#![feature(untagged_unions)]
#![allow(dead_code)]
#![deny(unions_with_drop_fields)]

union U {
    a: u8, // OK
}

union W {
    a: String, //~ ERROR union contains a field with possibly non-trivial drop code
    b: String, // OK, only one field is reported
}

struct S(String);

// `S` doesn't implement `Drop` trait, but still has non-trivial destructor
union Y {
    a: S, //~ ERROR union contains a field with possibly non-trivial drop code
}

// We don't know if `T` is trivially-destructable or not until trans
union J<T> {
    a: T, //~ ERROR union contains a field with possibly non-trivial drop code
}

union H<T: Copy> {
    a: T, // OK, `T` is `Copy`, no destructor
}

fn main() {}
