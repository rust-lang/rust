// run-pass
#![feature(type_equality_constraints)]

pub fn foo1<T: IntoIterator>() where T = Vec<T::Item> {}
pub fn foo2<T: IntoIterator>() where T::Item = u32 {}
pub fn foo3<T: IntoIterator>() where T::Item = u32, T = Vec<u32> {}
pub fn foo4<T: IntoIterator>() where T::Item = u32, T = Vec<T::Item> {}

pub trait Bar {}
pub fn foo5<T: IntoIterator>() where T::Item = u32, T::Item: Bar {}
pub fn foo6<T: IntoIterator>() where T::Item: Bar, T::Item = u32 {}

fn main() {}
