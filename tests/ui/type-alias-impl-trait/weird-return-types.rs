//@ edition:2018
//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::fmt::Debug;
use std::future::Future;

type Foo = impl Debug;

#[define_opaque(Foo)]
fn f() -> impl Future<Output = Foo> {
    async move { 22_u32 }
}

fn main() {}
