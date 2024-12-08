//@aux-build: proc_macros.rs

#![allow(unused)]
#![warn(clippy::let_underscore_untyped)]

extern crate proc_macros;
use proc_macros::with_span;

use clippy_utils::is_from_proc_macro;
use std::boxed::Box;
use std::fmt::Display;
use std::future::Future;

fn a() -> u32 {
    1
}

fn b<T>(x: T) -> T {
    x
}

fn c() -> impl Display {
    1
}

fn d(x: &u32) -> &u32 {
    x
}

fn e() -> Result<u32, ()> {
    Ok(1)
}

fn f() -> Box<dyn Display> {
    Box::new(1)
}

fn g() -> impl Fn() {
    || {}
}

with_span!(
    span

    fn dont_lint_proc_macro() {
        let _ = a();
    }
);

fn main() {
    let _ = a();
    let _ = b(1);
    let _ = c();
    let _ = d(&1);
    let _ = e();
    let _ = f();
    let _ = g();
    let closure = || {};

    _ = a();
    _ = b(1);
    _ = c();
    _ = d(&1);
    _ = e();
    _ = f();

    let _: u32 = a();
    let _: u32 = b(1);
    let _: &u32 = d(&1);
    let _: Result<_, _> = e();
    let _: Box<_> = f();

    #[allow(clippy::let_underscore_untyped)]
    let _ = a();
}

async fn dont_lint_async_prototype(_: u8) {}
