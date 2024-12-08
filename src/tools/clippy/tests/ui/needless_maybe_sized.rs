//@aux-build:proc_macros.rs

#![allow(unused, clippy::multiple_bound_locations)]
#![warn(clippy::needless_maybe_sized)]

extern crate proc_macros;
use proc_macros::external;

fn directly<T: Sized + ?Sized>(t: &T) {}

trait A: Sized {}
trait B: A {}

fn depth_1<T: A + ?Sized>(t: &T) {}
fn depth_2<T: B + ?Sized>(t: &T) {}

// We only need to show one
fn multiple_paths<T: A + B + ?Sized>(t: &T) {}

fn in_where<T>(t: &T)
where
    T: Sized + ?Sized,
{
}

fn mixed_1<T: Sized>(t: &T)
where
    T: ?Sized,
{
}

fn mixed_2<T: ?Sized>(t: &T)
where
    T: Sized,
{
}

fn mixed_3<T>(t: &T)
where
    T: Sized,
    T: ?Sized,
{
}

struct Struct<T: Sized + ?Sized>(T);

impl<T: Sized + ?Sized> Struct<T> {
    fn method<U: Sized + ?Sized>(&self) {}
}

enum Enum<T: Sized + ?Sized + 'static> {
    Variant(&'static T),
}

union Union<'a, T: Sized + ?Sized> {
    a: &'a T,
}

trait Trait<T: Sized + ?Sized> {
    fn trait_method<U: Sized + ?Sized>() {}

    type GAT<U: Sized + ?Sized>;

    type Assoc: Sized + ?Sized; // False negative
}

trait SecondInTrait: Send + Sized {}
fn second_in_trait<T: ?Sized + SecondInTrait>() {}

fn impl_trait(_: &(impl Sized + ?Sized)) {}

trait GenericTrait<T>: Sized {}
fn in_generic_trait<T: GenericTrait<U> + ?Sized, U>() {}

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
