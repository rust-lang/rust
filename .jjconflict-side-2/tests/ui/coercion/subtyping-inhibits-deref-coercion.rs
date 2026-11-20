// For some reason, subtyping due to higher ranked function pointers,
// even in an invariant position, causes deref coercion to not happen.
// This might be a compiler bug. Notably, the `pin!()` macro's soundness
// previously relied on code like this not compiling.
// See https://github.com/rust-lang/rust/issues/153438#issuecomment-4517609119

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

// `Sub` is a subtype of `Sup`
type Sup = fn(&'static ());
type Sub = for<'a> fn(&'a ());

struct Thing<S>(PhantomData<S>);

impl Deref for Thing<Sub> {
    type Target = Thing<Sup>;
    fn deref(&self) -> &Thing<Sup> {
        unimplemented!()
    }
}

impl DerefMut for Thing<Sub> {
    fn deref_mut(&mut self) -> &mut Thing<Sup> {
        unimplemented!()
    }
}

// This could, in theory, compile due to a deref coercion. However, it does not.
fn foo(x: &mut Thing<Sub>) -> &mut Thing<Sup> {
    x
    //~^ ERROR mismatched types
}

fn main() {}
