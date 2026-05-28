#![feature(optimize_attribute)]
#![feature(stmt_expr_attributes)]
#![deny(unused_attributes)]
#![allow(dead_code)]

//@ edition: 2018

#[optimize(speed)] //~ ERROR attribute cannot be used on
struct F;

fn invalid() {
    #[optimize(speed)] //~ ERROR attribute cannot be used on
    {
        1
    };
}

#[optimize(speed)]
fn valid() {}

#[optimize(speed)] //~ ERROR attribute cannot be used on
mod valid_module {}

#[optimize(speed)] //~ ERROR attribute cannot be used on
impl F {}

fn main() {
    let _ = #[optimize(speed)]
    (|| 1);
}

use std::future::Future;

fn async_block() -> impl Future<Output = ()> {
    #[optimize(speed)]
    async { }
}

#[optimize(speed)]
async fn async_fn() {
    ()
}

trait Foo {
    #[optimize(speed)] //~ ERROR attribute cannot be used on
    fn invalid();
    #[optimize(speed)]
    fn valid() {}
}

impl Foo for () {
    #[optimize(speed)]
    fn invalid() {}
    #[optimize(size)]
    fn valid() {}
}

#[optimize(speed)]
#[optimize(speed)] //~ ERROR multiple `optimize` attributes
fn duplicate_same() {}

#[optimize(speed)]
#[optimize(size)] //~ ERROR multiple `optimize` attributes
fn duplicate_different() {}

#[optimize(none)] //~ ERROR `#[optimize(none)]` cannot be used with `#[inline]` attributes
#[inline]
fn inline_conflict_a() {}

#[inline(always)]
#[optimize(none)] //~ ERROR `#[optimize(none)]` cannot be used with `#[inline]` attributes
fn inline_conflict_b() {}

#[inline(never)]
#[optimize(none)]
fn inline_conflict_c() {}
