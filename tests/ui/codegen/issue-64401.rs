//@ build-pass
// The ICE didn't happen with `cargo check` but `cargo build`.

use std::marker::PhantomData;

trait Owned<'a> {
    type Reader;
}

impl<'a> Owned<'a> for () {
    type Reader = ();
}

trait Handler {
    fn handle(&self);
}

struct CtxHandlerWithoutState<M, F> {
    message_type: PhantomData<M>,
    _function: F,
}

impl<M, F> CtxHandlerWithoutState<M, F> {
    pub fn new(_function: F) -> Self {
        Self {
            message_type: PhantomData,
            _function,
        }
    }
}

impl<'a, M, F> Handler for CtxHandlerWithoutState<M, F>
where
    F: Fn(<M as Owned<'a>>::Reader),
    M: Owned<'a>,
{
    fn handle(&self) {}
}

fn e_to_i<M: for<'a> Owned<'a>>(_: <M as Owned<'_>>::Reader) {}

fn send_external_to_internal<M>()
where
    M: for<'a> Owned<'a>,
{
    let _: Box<dyn Handler> = Box::new(CtxHandlerWithoutState::<M, _>::new(e_to_i::<M>));
}

fn main() {
    send_external_to_internal::<()>()
}
