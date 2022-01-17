// this file solely exists to test constants defined in foreign crates.
// As the most common case is the `http` crate, it replicates `http::HeadewrName`'s structure.

#![allow(clippy::declare_interior_mutable_const)]

use std::sync::atomic::AtomicUsize;

enum Private<T> {
    ToBeUnfrozen(#[allow(dead_code)] T),
    Frozen(#[allow(dead_code)] usize),
}

pub struct Wrapper(#[allow(dead_code)] Private<AtomicUsize>);

pub const WRAPPED_PRIVATE_UNFROZEN_VARIANT: Wrapper = Wrapper(Private::ToBeUnfrozen(AtomicUsize::new(6)));
pub const WRAPPED_PRIVATE_FROZEN_VARIANT: Wrapper = Wrapper(Private::Frozen(7));
