#![feature(untagged_unions)]
#![allow(dead_code)]

union U {
    a: u8, // OK
}

union W {
    a: String, //~ ERROR unions may not contain fields that need dropping
    b: String, // OK, only one field is reported
}

struct S(String);

// `S` doesn't implement `Drop` trait, but still has non-trivial destructor
union Y {
    a: S, //~ ERROR unions may not contain fields that need dropping
}

// We don't know if `T` is trivially-destructable or not until trans
union J<T> {
    a: T, //~ ERROR unions may not contain fields that need dropping
}

union H<T: Copy> {
    a: T, // OK, `T` is `Copy`, no destructor
}

fn main() {}
