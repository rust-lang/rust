#![deny(clippy::trait_duplication_in_bounds)]
#![allow(unused)]
#![feature(associated_const_equality, const_trait_impl)]

use std::any::Any;

fn bad_foo<T: Clone + Clone + Clone + Copy, U: Clone + Copy>(arg0: T, argo1: U) {
    //~^ trait_duplication_in_bounds
    unimplemented!();
}

fn bad_bar<T, U>(arg0: T, arg1: U)
where
    T: Clone + Clone + Clone + Copy,
    //~^ trait_duplication_in_bounds
    U: Clone + Copy,
{
    unimplemented!();
}

fn good_bar<T: Clone + Copy, U: Clone + Copy>(arg0: T, arg1: U) {
    unimplemented!();
}

fn good_foo<T, U>(arg0: T, arg1: U)
where
    T: Clone + Copy,
    U: Clone + Copy,
{
    unimplemented!();
}

trait GoodSelfTraitBound: Clone + Copy {
    fn f();
}

trait GoodSelfWhereClause {
    fn f()
    where
        Self: Clone + Copy;
}

trait BadSelfTraitBound: Clone + Clone + Clone {
    //~^ trait_duplication_in_bounds
    fn f();
}

trait BadSelfWhereClause {
    fn f()
    where
        Self: Clone + Clone + Clone;
    //~^ trait_duplication_in_bounds
}

trait GoodTraitBound<T: Clone + Copy, U: Clone + Copy> {
    fn f();
}

trait GoodWhereClause<T, U> {
    fn f()
    where
        T: Clone + Copy,
        U: Clone + Copy;
}

trait BadTraitBound<T: Clone + Clone + Clone + Copy, U: Clone + Copy> {
    //~^ trait_duplication_in_bounds
    fn f();
}

trait BadWhereClause<T, U> {
    fn f()
    where
        T: Clone + Clone + Clone + Copy,
        //~^ trait_duplication_in_bounds
        U: Clone + Copy;
}

struct GoodStructBound<T: Clone + Copy, U: Clone + Copy> {
    t: T,
    u: U,
}

impl<T: Clone + Copy, U: Clone + Copy> GoodTraitBound<T, U> for GoodStructBound<T, U> {
    // this should not warn
    fn f() {}
}

struct GoodStructWhereClause;

impl<T, U> GoodTraitBound<T, U> for GoodStructWhereClause
where
    T: Clone + Copy,
    U: Clone + Copy,
{
    // this should not warn
    fn f() {}
}

fn no_error_separate_arg_bounds(program: impl AsRef<()>, dir: impl AsRef<()>, args: &[impl AsRef<()>]) {}

trait GenericTrait<T> {}

fn good_generic<T: GenericTrait<u64> + GenericTrait<u32>>(arg0: T) {
    unimplemented!();
}

fn bad_generic<T: GenericTrait<u64> + GenericTrait<u32> + GenericTrait<u64>>(arg0: T) {
    //~^ trait_duplication_in_bounds
    unimplemented!();
}

mod foo {
    pub trait Clone {}
}

fn qualified_path<T: std::clone::Clone + Clone + foo::Clone>(arg0: T) {
    //~^ trait_duplication_in_bounds
    unimplemented!();
}

fn good_trait_object(arg0: &(dyn Any + Send)) {
    unimplemented!();
}

fn bad_trait_object(arg0: &(dyn Any + Send + Send)) {
    //~^ trait_duplication_in_bounds
    unimplemented!();
}

trait Proj {
    type S;
}

impl Proj for () {
    type S = ();
}

impl Proj for i32 {
    type S = i32;
}

trait Base<T> {
    fn is_base(&self);
}

trait Derived<B: Proj>: Base<B::S> + Base<()> {
    fn is_derived(&self);
}

fn f<P: Proj>(obj: &dyn Derived<P>) {
    obj.is_derived();
    Base::<P::S>::is_base(obj);
    Base::<()>::is_base(obj);
}

// #13476
trait Value<const N: usize> {}
fn const_generic<T: Value<0> + Value<1>>() {}

// #11067 and #9626
fn assoc_tys_generics<'a, 'b, T, U>()
where
    T: IntoIterator<Item = ()> + IntoIterator<Item = i32>,
    U: From<&'a str> + From<&'b [u16]>,
{
}

// #13476
const trait ConstTrait {}
const fn const_trait_bounds_good<T: ConstTrait + [const] ConstTrait>() {}

const fn const_trait_bounds_bad<T: [const] ConstTrait + [const] ConstTrait>() {}
//~^ trait_duplication_in_bounds

fn projections<T, U, V>()
where
    U: ToOwned,
    V: ToOwned,
    T: IntoIterator<Item = U::Owned> + IntoIterator<Item = U::Owned>,
    //~^ trait_duplication_in_bounds
    V: IntoIterator<Item = U::Owned> + IntoIterator<Item = V::Owned>,
{
}

fn main() {
    let _x: fn(_) = f::<()>;
    let _x: fn(_) = f::<i32>;
}

// #13706
fn assoc_tys_bounds<T>()
where
    T: Iterator<Item: Clone> + Iterator<Item: Clone>,
{
}
trait AssocConstTrait {
    const ASSOC: usize;
}
fn assoc_const_args<T>()
where
    T: AssocConstTrait<ASSOC = 0> + AssocConstTrait<ASSOC = 0>,
    //~^ trait_duplication_in_bounds
{
}
