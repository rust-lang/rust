//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Previously we forgot to invalidate rigid aliases when typing mode is changed
// into `PostAnalysis`. Thus we failed to reveal opaques in const eval.

#![feature(type_alias_impl_trait)]

type Opaque = impl Sized;

#[define_opaque(Opaque)]
fn value() -> Opaque {
    0u8
}

trait Trait {
    const N: usize;
}

impl<T> Trait for T {
    const N: usize = std::mem::size_of::<T>();
}

fn test(n: usize) {
    match n {
        <Opaque as Trait>::N => {}
        _ => {}
    }
}

fn main() {}
