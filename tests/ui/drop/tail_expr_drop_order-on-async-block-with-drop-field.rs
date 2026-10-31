//@ edition: 2021
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/142155>.
// The tail expr drop order lint used to ICE on this async block. It stopped
// once types with only a sync `Drop` were no longer treated as needing async drop.

#![warn(tail_expr_drop_order)]
use core::future::Future;

fn f() -> impl Future<Output = Option<String>> {
    async { Some("nope".into()) }
}

fn main() {}
