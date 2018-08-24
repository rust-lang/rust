// Tests that items in subscopes can shadow type parameters and local variables (see issue #23880).

#![allow(unused)]
struct Foo<X> { x: Box<X> }
impl<Bar> Foo<Bar> {
    fn foo(&self) {
        type Bar = i32;
        let _: Bar = 42;
    }
}

fn main() {
    let f = 1;
    {
        fn f() {}
        f();
    }
}
