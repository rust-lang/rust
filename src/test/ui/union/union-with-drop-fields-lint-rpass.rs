// run-pass

#![feature(untagged_unions)]
#![allow(dead_code)]
#![allow(unions_with_drop_fields)]

union U {
    a: u8, // OK
}

union W {
    a: String, // OK
    b: String, // OK
}

struct S(String);

// `S` doesn't implement `Drop` trait, but still has non-trivial destructor
union Y {
    a: S, // OK
}

// We don't know if `T` is trivially-destructable or not until trans
union J<T> {
    a: T, // OK
}

union H<T: Copy> {
    a: T, // OK
}

fn main() {}
