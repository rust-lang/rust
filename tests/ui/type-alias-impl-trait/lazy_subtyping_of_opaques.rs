#![feature(type_alias_impl_trait)]

//! This test used to ICE rust-lang/rust#124891
//! because we added an assertion for catching cases where opaque types get
//! registered during the processing of subtyping predicates.

type Tait = impl FnOnce() -> ();

#[define_opaque(Tait)]
fn reify_as_tait() -> Thunk<Tait> {
    //~^ ERROR: expected a `FnOnce()` closure, found `()`
    Thunk::new(|cont| cont)
    //~^ ERROR: mismatched types
    //~| ERROR: expected a `FnOnce()` closure, found `()`
}

struct Thunk<F>(F);

impl<F> Thunk<F> {
    fn new(f: F)
    where
        F: ContFn,
    {
        todo!();
    }
}

trait ContFn {}

impl<F: FnOnce(Tait) -> ()> ContFn for F {}

fn main() {}
