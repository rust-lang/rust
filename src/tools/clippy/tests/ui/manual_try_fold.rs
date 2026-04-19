//@aux-build:proc_macros.rs
#![allow(clippy::unnecessary_fold, unused)]
#![warn(clippy::manual_try_fold)]
#![feature(try_trait_v2)]
#![feature(try_trait_v2_residual)]
//@no-rustfix
use std::ops::{ControlFlow, FromResidual, Residual, Try, FromOutput, Branch};

#[macro_use]
extern crate proc_macros;

// Test custom `Try` with more than 1 argument
struct NotOption(i32, i32);

struct NotOptionResidual;

impl<R> FromResidual<R> for NotOption {
    fn from_residual(_: R) -> Self {
        todo!()
    }
}

impl Residual<()> for NotOptionResidual {
    type TryType = NotOption;
}

impl Branch for NotOption {
    type Output = ();
    type Residual = NotOptionResidual;

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        todo!()
    }
}

impl FromOutput for NotOption {
    fn from_output(_: Self::Output) -> Self {
        todo!()
    }
}
// Test custom `Try` with only 1 argument
#[derive(Default)]
struct NotOptionButWorse(i32);

struct NotOptionButWorseResidual;

impl<R> FromResidual<R> for NotOptionButWorse {
    fn from_residual(_: R) -> Self {
        todo!()
    }
}

impl Residual<()> for NotOptionButWorseResidual {
    type TryType = NotOptionButWorse;
}

impl Branch for NotOptionButWorse {
    type Output = ();
    type Residual = NotOptionButWorseResidual;

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        todo!()
    }
}

impl FromOutput for NotOptionButWorse {
    fn from_output(_: Self::Output) -> Self {
        todo!()
    }
}

fn main() {
    [1, 2, 3]
        .iter()
        .fold(Some(0i32), |sum, i| sum?.checked_add(*i))
        //~^ manual_try_fold
        .unwrap();
    [1, 2, 3]
        .iter()
        .fold(NotOption(0i32, 0i32), |sum, i| NotOption(0i32, 0i32));
    //~^ manual_try_fold
    [1, 2, 3]
        .iter()
        .fold(NotOptionButWorse(0i32), |sum, i| NotOptionButWorse(0i32));
    //~^ manual_try_fold
    // Do not lint
    [1, 2, 3].iter().try_fold(0i32, |sum, i| sum.checked_add(*i)).unwrap();
    [1, 2, 3].iter().fold(0i32, |sum, i| sum + i);
    [1, 2, 3]
        .iter()
        .fold(NotOptionButWorse::default(), |sum, i| NotOptionButWorse::default());
    external! {
        [1, 2, 3].iter().fold(Some(0i32), |sum, i| sum?.checked_add(*i)).unwrap();
        [1, 2, 3].iter().try_fold(0i32, |sum, i| sum.checked_add(*i)).unwrap();
    }
    with_span! {
        span
        [1, 2, 3].iter().fold(Some(0i32), |sum, i| sum?.checked_add(*i)).unwrap();
        [1, 2, 3].iter().try_fold(0i32, |sum, i| sum.checked_add(*i)).unwrap();
    }
}

#[clippy::msrv = "1.26.0"]
fn msrv_too_low() {
    [1, 2, 3]
        .iter()
        .fold(Some(0i32), |sum, i| sum?.checked_add(*i))
        .unwrap();
}

#[clippy::msrv = "1.27.0"]
fn msrv_juust_right() {
    [1, 2, 3]
        .iter()
        .fold(Some(0i32), |sum, i| sum?.checked_add(*i))
        //~^ manual_try_fold
        .unwrap();
}

mod issue11876 {
    struct Foo;

    impl Bar for Foo {
        type Output = u32;
    }

    trait Bar: Sized {
        type Output;
        fn fold<A, F>(self, init: A, func: F) -> Fold<Self, A, F>
        where
            A: Clone,
            F: Fn(A, Self::Output) -> A,
        {
            Fold { this: self, init, func }
        }
    }

    #[allow(dead_code)]
    struct Fold<S, A, F> {
        this: S,
        init: A,
        func: F,
    }

    fn main() {
        Foo.fold(Some(0), |acc, entry| Some(acc? + entry));
    }
}
