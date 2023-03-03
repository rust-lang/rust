// Test `type_alias_bounds` lint warning about bounds in type-alias-impl-trait.

// check-pass
#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;

type T1<T: Debug> = (impl Debug, T); //~ WARN not enforced in type aliases
fn f1<U: Debug>(x: U) -> T1<U> {
    (Vec::<U>::new(), x)
}

type T2<T: Debug> = (impl Debug, usize); // no warning here!
fn f2<U: Debug>() -> T2<U> {
    (Vec::<U>::new(), 0)
}

trait Bar<T: Debug> {}
impl<T: Debug> Bar<T> for Vec<T> {}

type T3<T: Debug> = (impl Bar<T>, T); //~ WARN not enforced in type aliases
fn f3<U: Debug>(v: U) -> T3<U> {
    (Vec::<U>::new(), v)
}

type T4<T: Debug> = (impl Bar<T>, usize); // no warning here!
fn f4<U: Debug>() -> T4<U> {
    (Vec::<U>::new(), 0)
}

type T5<'a: 'a, T: Debug> = (impl Debug, &'a usize); //~ WARN not enforced in type aliases
fn f5<'a, T: Debug>(x: &'a usize) -> T5<'a, T> {
    (Vec::<T>::new(), x)
}

fn main() {}
