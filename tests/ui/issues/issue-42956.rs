//@ check-pass
#![allow(dead_code)]
#![allow(stable_features)]
#![feature(associated_consts)]

impl A for i32 {
    type Foo = u32;
}
impl B for u32 {
    const BAR: i32 = 0;
}

trait A {
    type Foo: B;
}

trait B {
    const BAR: i32;
}

fn generic<T: A>() {
    // This panics if the universal function call syntax is used as well
    println!("{}", T::Foo::BAR);
}

fn main() {}
