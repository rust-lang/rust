//@ check-pass

#![allow(non_camel_case_types)]

pub fn main() {
    let a = 1;
    struct Foo<a> { field: a, };
}
