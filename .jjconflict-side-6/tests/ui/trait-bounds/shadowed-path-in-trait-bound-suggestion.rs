//@ edition: 2015
//@ run-rustfix
#![allow(non_snake_case)]
mod A {
    pub trait Trait {}
    impl Trait for i32 {}
}

mod B {
    pub struct A<H: A::Trait>(pub H); //~ ERROR cannot find trait
}

fn main() {
    let _ = B::A(42);
}
