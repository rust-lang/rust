//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] build-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #119095
//@[old] build-fail
//@ edition: 2021

// Regression test for #119095. No clue why this does not fail
// to compile with the new solver. Likely because we normalize
// `impl Unit<&'placeholder ()>` in a separate nested goal, so
// its constraints on placeholder are ignored by the leak check.

fn any<T>() -> T {
    loop {}
}

trait Acquire {
    type Connection;
}

impl Acquire for &'static () {
    type Connection = ();
}

trait Unit {}
impl Unit for () {}

fn get_connection<T>() -> impl Unit
where
    T: Acquire,
    T::Connection: Unit,
{
    any::<T::Connection>()
}

fn main() {
    let future = async { async { get_connection::<&'static ()>() }.await };

    future.resolve_me();
}

trait ResolveMe {
    fn resolve_me(self);
}

impl<S> ResolveMe for S
where
    (): CheckSend<S>,
{
    fn resolve_me(self) {}
}

trait CheckSend<F> {}
impl<F> CheckSend<F> for () where F: Send {}

trait NeverImplemented {}
impl<E, F> CheckSend<F> for E where E: NeverImplemented {}
