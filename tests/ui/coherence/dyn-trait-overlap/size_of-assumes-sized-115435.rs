//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #115435
//@[old] build-fail
//@ edition:2021
//@ compile-flags: -Copt-level=0

// Regression test for the ICE in #115435. This bug is caused by #57893.
trait MyTrait {
    type Target: ?Sized;
}

impl<A: ?Sized> MyTrait for A {
    type Target = A;
}

fn main() {
    bug_run::<dyn MyTrait<Target = u8>>();
    //[next]~^ ERROR: type annotations needed
}

fn bug_run<T: ?Sized>()
where
    <T as MyTrait>::Target: Sized,
{
    bug::<T>();
}

fn bug<T>() {
    std::mem::size_of::<T>();
}
