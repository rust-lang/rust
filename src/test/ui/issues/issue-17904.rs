// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Test that we can parse where clauses on various forms of tuple
// structs.

// pretty-expanded FIXME #23616

struct Bar<T>(T) where T: Copy;
struct Bleh<T, U>(T, U) where T: Copy, U: Sized;
struct Baz<T> where T: Copy {
    field: T
}

fn main() {}
