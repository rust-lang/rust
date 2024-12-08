// Test that the lifetime from the enclosing `&` is "inherited"
// through the `MyBox` struct.

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a MyBox<dyn Test>,
    u: &'a MyBox<dyn Test + 'a>,
}

struct MyBox<T:?Sized> {
    b: Box<T>
}

fn c<'a>(t: &'a MyBox<dyn Test+'a>, mut ss: SomeStruct<'a>) {
    ss.t = t;
    //~^ ERROR lifetime may not live long enough
}

fn main() {
}
