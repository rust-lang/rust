//! Regression test for <https://github.com/rust-lang/rust/issues/164449>.
//@ aux-build: dependent-crate-no-gce-next-solver-aux.rs
//@ compile-flags: -Znext-solver=globally
//@ build-pass

extern crate dependent_crate_no_gce_next_solver_aux as aux;

use std::marker::PhantomData;

trait Loader {
    fn load(&self);
}

struct Parameter<T>(PhantomData<T>);

impl<T: Send> Loader for Parameter<T> {
    fn load(&self) {}
}

fn first(x: aux::Arr) -> u8 {
    x[0]
}

fn takes_tr<T: aux::Tr<7>>(_: T) {}

fn main() {
    let loader: &dyn Loader = &Parameter::<aux::Data>(PhantomData);
    loader.load();
    let _ = first([0; 4]);
    takes_tr(aux::A);
}
