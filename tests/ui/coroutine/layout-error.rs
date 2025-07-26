// Verifies that computing a layout of a coroutine tainted by type errors
// doesn't ICE. Regression test for #80998.
//
//@ edition:2018

#![feature(type_alias_impl_trait)]
use std::future::Future;

pub struct Task<F: Future>(F);
impl<F: Future> Task<F> {
    const fn new() -> Self {
        todo!()
    }
    fn spawn(&self, _: impl FnOnce() -> F) {
        todo!()
    }
}

pub type F = impl Future;

#[define_opaque(F)]
fn foo()
where
    F:,
{
    async fn cb() {
        let a = Foo; //~ ERROR cannot find value `Foo` in this scope
    }

    Task::spawn(&POOL, || cb());
}

// Check that statics are inhabited computes they layout.
static POOL: Task<F> = Task::new();

fn main() {}
