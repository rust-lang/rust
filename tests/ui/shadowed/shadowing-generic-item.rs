//! Test that generic parameters shadow structs and modules with the same name.

struct T { i: i32 }
fn f<T>() {
    let t = T { i: 0 }; //~ ERROR expected struct, variant or union type, found type parameter `T`
}

mod Foo {
    pub fn f() {}
}
fn g<Foo>() {
    Foo::f(); //~ ERROR no function or associated item named `f`
}

fn main() {}
