//@ build-fail
//@ compile-flags: -Zmir-opt-level=3
#![feature(inline_const)]

fn foo<T>() {
    if false {
        const { panic!() } //~ ERROR E0080
    }
}

fn main() {
    foo::<i32>();
}
