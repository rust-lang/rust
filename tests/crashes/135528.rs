//@ known-bug: #135528
//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes
#![feature(type_alias_impl_trait)]
type Tait = impl Copy;

fn set(x: &isize) -> isize {
    *x
}

#[define_opaque(Tait)]
fn d(x: Tait) {
    set(x);
}

#[define_opaque(Tait)]
fn other_define() -> Tait {
    ()
}

fn main() {}
