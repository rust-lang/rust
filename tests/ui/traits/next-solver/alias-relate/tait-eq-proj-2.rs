//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]

// Similar to tests/ui/traits/next-solver/alias-relate/tait-eq-proj.rs
// but check the alias-sub relation in the other direction.

type Tait = impl Iterator<Item = impl Sized>;

fn mk<T>() -> T {
    todo!()
}

#[define_opaque(Tait)]
fn a() {
    let x: Tait = mk();
    let mut array = mk();
    let mut z = IntoIterator::into_iter(array);
    z = x;
    array = [0i32; 32];
}

fn main() {}
