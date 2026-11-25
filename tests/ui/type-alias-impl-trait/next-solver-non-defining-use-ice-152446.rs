//@ compile-flags: -Znext-solver=globally

#![feature(type_alias_impl_trait)]
type Tait = impl Iterator<Item = impl Sized>;

#[define_opaque(Tait)]
fn a() {
    let _: Tait = IntoIterator::into_iter([0i32; 0i32]); //~ ERROR mismatched types
}

#[define_opaque(Tait)]
fn b() {
    let _: Tait = IntoIterator::into_iter([0i32; 42 /* <- */]);
}

fn main() {}
