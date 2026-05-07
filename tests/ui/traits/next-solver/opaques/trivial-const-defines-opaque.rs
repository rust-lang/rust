//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

type Tait = impl Sized;

#[define_opaque(Tait)]
const FOO: Tait = 1;

fn main() {
    let _: Tait = FOO;
}
