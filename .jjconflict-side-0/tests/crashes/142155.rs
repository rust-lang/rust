//@ known-bug: #142155
//@ needs-rustc-debug-assertions
//@ edition: 2021

#![warn(tail_expr_drop_order)]
use core::future::Future;

fn f() -> impl Future<Output = Option<String>> {
    async { Some("nope".into()) }
}

fn main() {}
