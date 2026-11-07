//@ known-bug: #149162
use std::marker::PhantomData;
pub trait ViewArgument {
    type Params<'a>;
}

impl ViewArgument for () {
    type Params<'a> = ();
}

pub trait View {}

pub fn buttons() -> Option<impl View> {
    Some(()).map(|()| text_button(|()| {}))
}
pub fn text_button<State: ViewArgument>(
    _: impl Fn(<State as ViewArgument>::Params<'_>),
) -> Button<State, impl Fn()> {
    Button {
        callback: || (),
        phantom: PhantomData,
    }
}
pub struct Button<State, F> {
    pub callback: F,
    pub phantom: PhantomData<State>,
}

impl<F> View for Button<(), F> {}
fn main() {}
