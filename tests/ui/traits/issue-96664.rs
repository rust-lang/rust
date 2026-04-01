//@ check-pass

#![feature(trait_alias)]

pub trait State = Clone + Send + Sync + PartialOrd + PartialEq + std::fmt::Display;
pub trait RandState<S: State> = FnMut() -> S + Send;

pub trait Evaluator {
    type State;
}

pub struct Evolver<E: Evaluator> {
    rand_state: Box<dyn RandState<E::State>>,
}

fn main() {}
