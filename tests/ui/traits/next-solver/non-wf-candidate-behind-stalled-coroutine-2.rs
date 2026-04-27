//@ check-pass
//@ compile-flags: -Znext-solver -Zvalidate-mir
//@ edition: 2021

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/251>.
//
// This previously caused an ICE due to a non–well-formed coroutine
// hidden type failing the leak check in the next-solver.
//
// In `TypingMode::Analysis`, the problematic type is hidden behind a
// stalled coroutine candidate. However, in later passes (e.g. MIR
// validation), we eagerly normalize it. The candidate that was
// previously accepted as a solution then fails the leak check, resulting
// in broken MIR and ultimately an ICE.

use std::future::Future;

trait Access {
    // has to have an associated type, but can be anything
    type Reader;

    fn read(&self) -> impl Future<Output = Self::Reader> + Send {
        async { loop {} }
    }
}

trait AccessDyn: Sync {}
impl Access for dyn AccessDyn {
    type Reader = ();
}

trait Stream {
    fn poll_next(s: &'static dyn AccessDyn);
}

// has to be a function in a trait impl, can't be a normal impl block or standalone fn
impl Stream for () {
    fn poll_next(s: &'static dyn AccessDyn) {
        // new async block is important
        is_dyn_send(&async {
            s.read().await;
        });
    }
}

fn is_dyn_send(_: &dyn Send) {}

fn main() {}
