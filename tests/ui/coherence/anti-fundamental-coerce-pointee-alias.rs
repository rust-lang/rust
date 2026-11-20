//@ check-fail

// Regression test for #148899: downstream crates must not be able to implement `CoerceUnsized`
// or `DispatchFromDyn` for `Pin`, including through a type alias as `derive(CoercePointee)`
// would generate when applied to `type Thing<T> = Pin<Wrap<T>>`.

#![feature(coerce_unsized, dispatch_from_dyn, unsize)]

use std::marker::Unsize;
use std::ops::{CoerceUnsized, DispatchFromDyn};
use std::pin::Pin;

pub struct Wrap<T: ?Sized>(Box<T>);

type Thing<T> = Pin<Wrap<T>>;

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Thing<U>> for Thing<T> {}
//~^ ERROR cannot implement `CoerceUnsized` for the fundamental type
//~| ERROR the trait bound `Wrap<T>: CoerceUnsized<Wrap<U>>` is not satisfied

impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Thing<U>> for Thing<T> {}
//~^ ERROR cannot implement `DispatchFromDyn` for the fundamental type

fn main() {}
