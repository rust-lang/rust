// issue: 113314

#![feature(type_alias_impl_trait)]

type Op = impl std::fmt::Display;
#[define_opaque(Op)]
fn foo() -> Op {
    &"hello world"
}

fn transform<S>() -> impl std::fmt::Display {
    &0usize
}
#[define_opaque(Op)]
fn bad() -> Op {
    //~^ ERROR cannot resolve opaque type
    transform::<Op>()
}

fn main() {
    let mut x = foo();
    println!("{x}");
    x = bad();
    println!("{x}");
}
