// edition:2018
// check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::future::Future;
use std::fmt::Debug;

type Foo = impl Debug;

fn f() -> impl Future<Output = Foo> {
    async move { 22_u32 }
}

fn main() {}
