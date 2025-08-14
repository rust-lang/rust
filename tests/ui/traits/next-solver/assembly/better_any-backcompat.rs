//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// A regression test for trait-system-refactor-initiative#183. While
// this concrete instance is likely not practically unsound, the general
// pattern is, see #57893.

use std::any::TypeId;

unsafe trait TidAble<'a>: Tid<'a> {}
trait TidExt<'a>: Tid<'a> {
    fn downcast_box(self: Box<Self>) {
        loop {}
    }
}

impl<'a, X: ?Sized + Tid<'a>> TidExt<'a> for X {}

unsafe trait Tid<'a>: 'a {}

unsafe impl<'a, T: ?Sized + TidAble<'a>> Tid<'a> for T {}

impl<'a> dyn Tid<'a> + 'a {
    fn downcast_any_box(self: Box<Self>) {
        self.downcast_box();
    }
}

unsafe impl<'a> TidAble<'a> for dyn Tid<'a> + 'a {}

fn main() {}
