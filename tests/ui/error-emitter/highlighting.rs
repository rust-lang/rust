// Make sure "highlighted" code is colored purple

//@ compile-flags: --error-format=human --color=always
//@ error-pattern:[35mfor<'a> [0m
//@ edition:2018

//@ revisions: windows not-windows
//@ [windows]only-windows
//@ [not-windows]ignore-windows

use core::pin::Pin;
use core::future::Future;
use core::any::Any;

fn query(_: fn(Box<(dyn Any + Send + '_)>) -> Pin<Box<(
    dyn Future<Output = Result<Box<(dyn Any + 'static)>, String>> + Send + 'static
)>>) {}

fn wrapped_fn<'a>(_: Box<(dyn Any + Send)>) -> Pin<Box<(
    dyn Future<Output = Result<Box<(dyn Any + 'static)>, String>> + Send + 'static
)>> {
    Box::pin(async { Err("nope".into()) })
}

fn main() {
    query(wrapped_fn);
}
