// This doesn't work, because we don't flow information from opaque types
// into function arguments via the function's generic parameters
// FIXME(oli-obk): make `expected_inputs_for_expected_output` support this
//@ revisions: default next
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

fn reify_as() -> Thunk<impl FnOnce(Continuation) -> Continuation> {
    Thunk::new(|mut cont| {
        //~^ ERROR type annotations needed
        cont.reify_as();
        cont
    })
}

type Tait = impl FnOnce(Continuation) -> Continuation;

#[define_opaque(Tait)]
fn reify_as_tait() -> Thunk<Tait> {
    Thunk::new(|mut cont| {
        //~^ ERROR type annotations needed
        cont.reify_as();
        cont
    })
}

#[must_use]
struct Thunk<F>(F);

impl<F> Thunk<F> {
    fn new(f: F) -> Self
    where
        F: ContFn,
    {
        Thunk(f)
    }
}

trait ContFn {}

impl<F: FnOnce(Continuation) -> Continuation> ContFn for F {}

struct Continuation;

impl Continuation {
    fn reify_as(&mut self) {}
}

fn main() {}
