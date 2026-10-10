//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Previously we forgot to invalidate rigid aliases when typing mode is changed
// into `PostAnalysis`. Thus we failed to reveal opaques in const eval.

#![feature(generic_const_items, type_alias_impl_trait)]
#![allow(incomplete_features)]

type Opaque = impl Sized;

#[define_opaque(Opaque)]
fn value() -> Opaque {
    0u8
}

const DIV<T>: u8 = std::mem::size_of::<T>() as u8;

pub fn test() -> &'static u8 {
    &(42 / DIV::<Opaque>)
}

fn main() {
}
