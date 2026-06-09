//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

fn main() {
    type Tait = impl Copy;
    let foo: Tait = (1u32, 2u32);
    let x: (_, _) = foo;
    println!("{:?}", x);
}
