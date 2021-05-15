// Unused `pub` fields in non-`pub` structs should also trigger dead code warnings.
// check-pass

#![warn(dead_code)]

struct Foo {
    a: i32, //~ WARNING: field is never read
    pub b: i32, //~ WARNING: field is never read
}

struct Bar;

impl Bar {
    fn a(&self) -> i32 { 5 } //~ WARNING: associated function is never used
    pub fn b(&self) -> i32 { 6 } //~ WARNING: associated function is never used
}


fn main() {
    let _ = Foo { a: 1, b: 2 };
    let _ = Bar;
}
