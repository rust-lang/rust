// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![allow(dead_code)]

union U {
    a: u8, // OK
}

union W {
    a: String, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
    b: String, // OK, only one field is reported
}

struct S(String);

// `S` doesn't implement `Drop` trait, but still has non-trivial destructor
union Y {
    a: S, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

// We don't know if `T` is trivially-destructable or not until trans
union J<T> {
    a: T, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

union H<T: Copy> {
    a: T, // OK, `T` is `Copy`, no destructor
}

fn main() {}
