//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

fn main() {
    type Tait = impl Sized;
    struct S {
        i: i32,
    }
    let x: Tait = S { i: 0 };
    println!("{}", x.i);
}
