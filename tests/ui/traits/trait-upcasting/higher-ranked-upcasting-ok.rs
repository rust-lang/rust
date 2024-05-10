//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// We should be able to instantiate a binder during trait upcasting.
// This test could be `check-pass`, but we should make sure that we
// do so in both trait solvers.
#![feature(trait_upcasting)]
#![crate_type = "rlib"]
trait Supertrait<'a, 'b> {}

trait Subtrait<'a, 'b>: Supertrait<'a, 'b> {}

impl<'a> Supertrait<'a, 'a> for () {}
impl<'a> Subtrait<'a, 'a> for () {}
fn ok(x: &dyn for<'a, 'b> Subtrait<'a, 'b>) -> &dyn for<'a> Supertrait<'a, 'a> {
    x //~ ERROR mismatched types
    //[current]~^ ERROR mismatched types
}
