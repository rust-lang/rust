//! This module provides a way to deal with self-referential data.
//!
//! The main idea is to allocate such data in a generator frame and then
//! give access to it by executing user-provided closures inside that generator.
//! The module provides a safe abstraction for the latter task.
//!
//! The interface consists of two exported macros meant to be used together:
//! * `declare_box_region_type` wraps a generator inside a struct with `access`
//!   method which accepts closures.
//! * `box_region_allow_access` is a helper which should be called inside
//!   a generator to actually execute those closures.

use std::marker::PhantomData;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

#[derive(Copy, Clone)]
pub struct AccessAction(*mut dyn FnMut());

impl AccessAction {
    pub fn get(self) -> *mut dyn FnMut() {
        self.0
    }
}

#[derive(Copy, Clone)]
pub enum Action {
    Initial,
    Access(AccessAction),
    Complete,
}

pub struct PinnedGenerator<I, A, R> {
    generator: Pin<Box<dyn Generator<Action, Yield = YieldType<I, A>, Return = R>>>,
}

impl<I, A, R> PinnedGenerator<I, A, R> {
    pub fn new<T: Generator<Action, Yield = YieldType<I, A>, Return = R> + 'static>(
        generator: T,
    ) -> (I, Self) {
        let mut result = PinnedGenerator { generator: Box::pin(generator) };

        // Run it to the first yield to set it up
        let init = match Pin::new(&mut result.generator).resume(Action::Initial) {
            GeneratorState::Yielded(YieldType::Initial(y)) => y,
            _ => panic!(),
        };

        (init, result)
    }

    pub unsafe fn access(&mut self, closure: *mut dyn FnMut()) {
        // Call the generator, which in turn will call the closure
        if let GeneratorState::Complete(_) =
            Pin::new(&mut self.generator).resume(Action::Access(AccessAction(closure)))
        {
            panic!()
        }
    }

    pub fn complete(&mut self) -> R {
        // Tell the generator we want it to complete, consuming it and yielding a result
        let result = Pin::new(&mut self.generator).resume(Action::Complete);
        if let GeneratorState::Complete(r) = result { r } else { panic!() }
    }
}

#[derive(PartialEq)]
pub struct Marker<T>(PhantomData<T>);

impl<T> Marker<T> {
    pub unsafe fn new() -> Self {
        Marker(PhantomData)
    }
}

pub enum YieldType<I, A> {
    Initial(I),
    Accessor(Marker<A>),
}
