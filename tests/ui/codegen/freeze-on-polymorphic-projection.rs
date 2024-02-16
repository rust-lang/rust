//@ build-pass
//@ compile-flags: -Copt-level=1 --crate-type=lib

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

pub unsafe trait Storage {
    type Handle;
}

pub unsafe trait MultipleStorage: Storage {}

default unsafe impl<S> Storage for S where S: MultipleStorage {}

// Make sure that we call is_freeze on `(S::Handle,)` in the param-env of `ice`,
// instead of in an empty, reveal-all param-env.
pub fn ice<S: Storage>(boxed: (S::Handle,)) -> (S::Handle,) {
    boxed
}
