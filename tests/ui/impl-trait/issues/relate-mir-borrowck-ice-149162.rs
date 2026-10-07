//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] known-bug: #149162
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr

// This test previously caused an ICE in `fn buttons` when relating opaque
// types. This has been fixed by the opaque type rework in the new solver.
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
