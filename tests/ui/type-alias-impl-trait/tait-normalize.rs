//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

fn enum_upvar() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    let x = move || match foo {
        None => (),
        Some((a, b)) => (),
    };
}

fn main(){}
