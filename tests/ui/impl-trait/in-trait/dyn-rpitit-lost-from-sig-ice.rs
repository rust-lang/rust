//! Regression test for <https://github.com/rust-lang/rust/issues/151964>.
//! The RPITIT is dropped from `foo`'s signature because `Out` doesn't exist,
//! so `Bar` is considered dyn-compatible even though it has an RPITIT.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Foo {}
trait Bar {
    fn foo(&self) -> &dyn Foo<Out = impl Sized>;
    //~^ ERROR associated type `Out` not found for `Foo`
}

fn main() {
    let x: &dyn Bar = todo!();
    x.foo();
    //[current]~^ ERROR the trait `Bar` is not dyn compatible
}
