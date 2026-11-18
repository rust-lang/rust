//@ edition: 2021
//@ check-pass

#![warn(tail_expr_drop_order)]
use core::future::Future;

fn f() -> impl Future<Output = Option<String>> {
    async { Some("nope".into()) }
}

fn main() {}
