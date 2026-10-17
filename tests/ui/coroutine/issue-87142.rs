//@ compile-flags: -Cdebuginfo=2
//@ build-pass

// Regression test for #87142
// This test needs the above flags and the "lib" crate type.

#![feature(impl_trait_in_assoc_type, coroutine_trait, coroutines)]
#![crate_type = "lib"]

use std::ops::Coroutine;

pub trait CoroutineProviderAlt: Sized {
    type Coro<'a>: Coroutine<(), Return = (), Yield<'a> = ()>
    where
        Self: 'a;

    fn start<'a>(ctx: Context<Self>) -> Self::Coro<'a>;
}

pub struct Context<G: 'static + CoroutineProviderAlt> {
    pub link: Box<G::Coro<'static>>,
}

impl CoroutineProviderAlt for () {
    type Coro<'a> = impl Coroutine<(), Return = (), Yield<'a> = ()>;
    fn start<'a>(ctx: Context<Self>) -> Self::Coro<'a> {
        #[coroutine]
        move || {
            match ctx {
                _ => (),
            }
            yield ();
        }
    }
}
