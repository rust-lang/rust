//@aux-build:proc_macros.rs

#![allow(unused, clippy::multiple_bound_locations)]
#![warn(clippy::needless_maybe_sized)]

extern crate proc_macros;
use proc_macros::external;

fn directly<T: Sized + ?Sized>(t: &T) {}
//~^ needless_maybe_sized

trait A: Sized {}
trait B: A {}

fn depth_1<T: A + ?Sized>(t: &T) {}
//~^ needless_maybe_sized
fn depth_2<T: B + ?Sized>(t: &T) {}
//~^ needless_maybe_sized

// We only need to show one
fn multiple_paths<T: A + B + ?Sized>(t: &T) {}
//~^ needless_maybe_sized

fn in_where<T>(t: &T)
where
    T: Sized + ?Sized,
    //~^ needless_maybe_sized
{
}

fn mixed_1<T: Sized>(t: &T)
where
    T: ?Sized,
    //~^ needless_maybe_sized
{
}

fn mixed_2<T: ?Sized>(t: &T)
//~^ needless_maybe_sized
where
    T: Sized,
{
}

fn mixed_3<T>(t: &T)
where
    T: Sized,
    T: ?Sized,
    //~^ needless_maybe_sized
{
}

struct Struct<T: Sized + ?Sized>(T);
//~^ needless_maybe_sized

impl<T: Sized + ?Sized> Struct<T> {
    //~^ needless_maybe_sized
    fn method<U: Sized + ?Sized>(&self) {}
    //~^ needless_maybe_sized
}

enum Enum<T: Sized + ?Sized + 'static> {
    //~^ needless_maybe_sized
    Variant(&'static T),
}

union Union<'a, T: Sized + ?Sized> {
    //~^ needless_maybe_sized
    a: &'a T,
}

trait Trait<T: Sized + ?Sized> {
    //~^ needless_maybe_sized
    fn trait_method<U: Sized + ?Sized>() {}
    //~^ needless_maybe_sized

    type GAT<U: Sized + ?Sized>;
    //~^ needless_maybe_sized

    type Assoc: Sized + ?Sized; // False negative
}

trait SecondInTrait: Send + Sized {}
fn second_in_trait<T: ?Sized + SecondInTrait>() {}
//~^ needless_maybe_sized

fn impl_trait(_: &(impl Sized + ?Sized)) {}
//~^ needless_maybe_sized

trait GenericTrait<T>: Sized {}
fn in_generic_trait<T: GenericTrait<U> + ?Sized, U>() {}
//~^ needless_maybe_sized

mod larger_graph {
    // C1  C2  Sized
    //  \  /\  /
    //   B1  B2
    //    \  /
    //     A1

    trait C1 {}
    trait C2 {}
    trait B1: C1 + C2 {}
    trait B2: C2 + Sized {}
    trait A1: B1 + B2 {}

    fn larger_graph<T: A1 + ?Sized>() {}
    //~^ needless_maybe_sized
}

// Should not lint

fn sized<T: Sized>() {}
fn maybe_sized<T: ?Sized>() {}

struct SeparateBounds<T: ?Sized>(T);
impl<T: Sized> SeparateBounds<T> {}

trait P {}
trait Q: P {}

fn ok_depth_1<T: P + ?Sized>() {}
fn ok_depth_2<T: Q + ?Sized>() {}

external! {
    fn in_macro<T: Clone + ?Sized>(t: &T) {}

    fn with_local_clone<T: $Clone + ?Sized>(t: &T) {}
}

#[derive(Clone)]
struct InDerive<T: ?Sized> {
    t: T,
}

struct Refined<T: ?Sized>(T);
impl<T: Sized> Refined<T> {}

fn main() {}
