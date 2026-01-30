//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
// Reproduces https://github.com/rust-lang/rust/pull/151746#issuecomment-3822930803.
//
// The change we tried to make there caused relating a type variable with an alias inside lub,
// In 5bd20bbd0ba6c0285664e55a1ffc677d7487c98b, we moved around code
// that adds an alias-relate predicate to be earlier, from one shared codepath into several
// distinct code paths. However, we forgot one codepath, through lub, causing an ICE in serde.
// In the end we dropped said commit, but the reproducer is still a useful as test.

use std::marker::PhantomData;

pub trait Trait {
    type Error;
}

pub struct Struct<E>(PhantomData<E>);

impl Trait for () {
    type Error = ();
}

fn main() {
    let _: Struct<<() as Trait>::Error> = match loop {} {
        b => loop {},
        a => Struct(PhantomData),
    };
}
