//! Regression test for <https://github.com/rust-lang/rust/issues/22471>.

//@ check-pass
#![allow(type_alias_bounds)]

type Foo<T> where T: Copy = Box<T>;

fn main(){}
