//@ check-pass
//@ compile-flags: -Znext-solver=globally
//@ edition:2021

// Regression test for <https://github.com/rust-lang/rust/issues/140577>.
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
use std::pin::Pin;
trait Acquire {
    type Connection;
}
impl Acquire for &'static () {
    type Connection = ();
}
fn b<T: Acquire>() -> impl Future + Send {
    let x: Pin<Box<dyn Future<Output = T::Connection> + Send>> = todo!();
    x
}
fn main() {
    async {
        b::<&()>().await;
    }
    .aa();
}

impl<F> Filter for F where F: Send {}

trait Filter {
    fn aa(self)
    where
        Self: Sized,
    {
    }
}
