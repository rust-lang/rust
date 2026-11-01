//@ known-bug: unknown

// If we want this to compile, then we'd need to do something like RPITs do,
// where nested associated constants have early-bound versions of their captured
// late-bound vars inserted into their generics. This gives us substitutable
// lifetimes to actually use when borrow-checking the associated const, which is
// lowered as a totally separate body from its parent. Since this doesn't exist,
// we should just error rather than resolving this late-bound var with no
// binder to actually attach it to, or worse, as a free region that can't even be
// substituted correctly, and ICEing. - @compiler-errors

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

const fn inner<'a>() -> usize where &'a (): Sized {
    3
}

fn test<'a>() {
    let _: [u8; inner::<'a>()];
    let _ = [0; inner::<'a>()];
}

fn main() {
    test();
}
