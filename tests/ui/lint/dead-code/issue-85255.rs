// Unused `pub` fields in non-`pub` structs should also trigger dead code warnings.
//@ check-pass

#![warn(dead_code)]

struct Foo {
    a: i32, //~ WARNING: fields `a` and `b` are never read
    pub b: i32,
}

struct Bar;

impl Bar {
    fn a(&self) -> i32 { 5 } //~ WARNING: methods `a` and `b` are never used [dead_code]
    pub fn b(&self) -> i32 { 6 }
}

pub(crate) struct Foo1 {
    a: i32, //~ WARNING: fields `a` and `b` are never read
    pub b: i32,
}

pub(crate) struct Bar1;

impl Bar1 {
    fn a(&self) -> i32 { 5 } //~ WARNING: methods `a` and `b` are never used [dead_code]
    pub fn b(&self) -> i32 { 6 }
}

pub(crate) struct Foo2 {
    a: i32, //~ WARNING: fields `a` and `b` are never read
    pub b: i32,
}

pub(crate) struct Bar2;

impl Bar2 {
    fn a(&self) -> i32 { 5 } //~ WARNING: methods `a` and `b` are never used [dead_code]
    pub fn b(&self) -> i32 { 6 }
}


fn main() {
    let _ = Foo { a: 1, b: 2 };
    let _ = Bar;
    let _ = Foo1 { a: 1, b: 2 };
    let _ = Bar1;
    let _ = Foo2 { a: 1, b: 2 };
    let _ = Bar2;
}
