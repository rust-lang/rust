//@ known-bug: rust-lang/rust#142155
//@ needs-rustc-debug-assertions
//@ edition:2021

#![warn(tail_expr_drop_order)]
use core::future::Future;
use core::pin::Pin;

fn wrapped_fn<'a>() -> Pin<Box<(dyn Future<Output = Result<Box<()>, String>> + Send + 'static)>> {
    Box::pin(async { Err("nope".into()) })
}

pub fn main() {}
