//@ check-pass
//@ compile-flags: -Zinline-mir -Zvalidate-mir
//@ edition: 2024

// See comment below.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Waker};

fn call_once<T>(f: impl FnOnce() -> T) -> T { f() }

fn main() {
    let x = async || {};
    // We first inline `call_once<{async closure}>`.
    //
    // This gives us a future whose type is the "FnOnce" flavor of the async closure's
    // child coroutine. The body of this coroutine is synthetic, which we synthesize in
    // the by-move body query.
    let fut = pin!(call_once(x));
    // We then try to inline that body in this poll call.
    //
    // The inliner does some inlinability checks; one of these checks involves checking
    // the body for the `#[rustc_no_mir_inline]` attribute. Since the synthetic body had
    // no HIR synthesized, but it's still a local def id, we end up ICEing in the
    // `local_def_id_to_hir_id` call when trying to read its attrs.
    fut.poll(&mut Context::from_waker(Waker::noop()));
}
