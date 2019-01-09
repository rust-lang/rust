// Test that the lifetime from the enclosing `&` is "inherited"
// through the `MyBox` struct.

#![allow(dead_code)]
#![feature(rustc_error)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a MyBox<Test>,
    u: &'a MyBox<Test+'a>,
}

struct MyBox<T:?Sized> {
    b: Box<T>
}

fn c<'a>(t: &'a MyBox<Test+'a>, mut ss: SomeStruct<'a>) {
    ss.t = t; //~ ERROR mismatched types
}

fn main() {
}
