//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![allow(warnings)]
trait Trait<U> {
    type Assoc;
}

impl<T> Trait<u64> for T {
    type Assoc = T;
}

fn lazy_init<T: Trait<U>, U>() -> (T, <T as Trait<U>>::Assoc) {
    todo!()
}

fn foo<T: Trait<u32, Assoc = T>>(x: T) {
    // When considering impl candidates to be equally valid as env candidates
    // this ends up being ambiguous as `U` can be both `u32´ and `u64` here.
    //
    // This is acceptable breakage but we should still note that it's
    // theoretically breaking.
    let (delayed, mut proj) = lazy_init::<_, _>();
    proj = x;
    let _: T = delayed;
}

fn main() {}
