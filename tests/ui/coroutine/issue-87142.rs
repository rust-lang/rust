//@ compile-flags: -Cdebuginfo=2
//@ build-pass

// Regression test for #87142
// This test needs the above flags and the "lib" crate type.

#![feature(impl_trait_in_assoc_type, coroutine_trait, coroutines)]
#![crate_type = "lib"]

use std::ops::Coroutine;

pub trait CoroutineProviderAlt: Sized {
    type Coro: Coroutine<(), Return = (), Yield = ()>;

    fn start(ctx: Context<Self>) -> Self::Coro;
}

pub struct Context<G: 'static + CoroutineProviderAlt> {
    pub link: Box<G::Coro>,
}

impl CoroutineProviderAlt for () {
    type Coro = impl Coroutine<(), Return = (), Yield = ()>;
    fn start(ctx: Context<Self>) -> Self::Coro {
        #[coroutine]
        move || {
            match ctx {
                _ => (),
            }
            yield ();
        }
    }
}
