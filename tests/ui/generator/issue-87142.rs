// compile-flags: -Cdebuginfo=2
// build-pass

// Regression test for #87142
// This test needs the above flags and the "lib" crate type.

#![feature(impl_trait_in_assoc_type, generator_trait, generators)]
#![crate_type = "lib"]

use std::ops::Coroutine;

pub trait CoroutineProviderAlt: Sized {
    type Gen: Coroutine<(), Return = (), Yield = ()>;

    fn start(ctx: Context<Self>) -> Self::Gen;
}

pub struct Context<G: 'static + CoroutineProviderAlt> {
    pub link: Box<G::Gen>,
}

impl CoroutineProviderAlt for () {
    type Gen = impl Coroutine<(), Return = (), Yield = ()>;
    fn start(ctx: Context<Self>) -> Self::Gen {
        move || {
            match ctx {
                _ => (),
            }
            yield ();
        }
    }
}
