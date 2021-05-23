// Unused `pub` fields in non-`pub` structs should also trigger dead code warnings.
// check-pass

#![warn(dead_code)]
#![feature(crate_visibility_modifier)]

struct Foo {
    a: i32, //~ WARNING: field is never read
    pub b: i32, //~ WARNING: field is never read
}

struct Bar;

impl Bar {
    fn a(&self) -> i32 { 5 } //~ WARNING: associated function is never used
    pub fn b(&self) -> i32 { 6 } //~ WARNING: associated function is never used
}

pub(crate) struct Foo1 {
    a: i32, //~ WARNING: field is never read
    pub b: i32, //~ WARNING: field is never read
}

pub(crate) struct Bar1;

impl Bar1 {
    fn a(&self) -> i32 { 5 } //~ WARNING: associated function is never used
    pub fn b(&self) -> i32 { 6 } //~ WARNING: associated function is never used
}

crate struct Foo2 {
    a: i32, //~ WARNING: field is never read
    pub b: i32, //~ WARNING: field is never read
}

crate struct Bar2;

impl Bar2 {
    fn a(&self) -> i32 { 5 } //~ WARNING: associated function is never used
    pub fn b(&self) -> i32 { 6 } //~ WARNING: associated function is never used
}


fn main() {
    let _ = Foo { a: 1, b: 2 };
    let _ = Bar;
    let _ = Foo1 { a: 1, b: 2 };
    let _ = Bar1;
    let _ = Foo2 { a: 1, b: 2 };
    let _ = Bar2;
}
