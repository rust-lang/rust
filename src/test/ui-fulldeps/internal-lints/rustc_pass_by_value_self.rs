// compile-flags: -Z unstable-options
// NOTE: This test doesn't actually require `fulldeps`
// so we could instead use it as a `ui` test.
//
// Considering that all other `internal-lints` are tested here
// this seems like the cleaner solution though.
#![feature(rustc_attrs)]
#![deny(rustc::pass_by_value)]
#![allow(unused)]

#[rustc_pass_by_value]
struct TyCtxt<'tcx> {
    inner: &'tcx (),
}

impl<'tcx> TyCtxt<'tcx> {
    fn by_value(self) {} // OK
    fn by_ref(&self) {} //~ ERROR passing `TyCtxt<'tcx>` by reference
}

struct TyS<'tcx> {
    inner: &'tcx (),
}

#[rustc_pass_by_value]
type Ty<'tcx> = &'tcx TyS<'tcx>;

impl<'tcx> TyS<'tcx> {
    fn by_value(self: Ty<'tcx>) {}
    fn by_ref(self: &Ty<'tcx>) {} //~ ERROR passing `Ty<'tcx>` by reference
}

#[rustc_pass_by_value]
struct Foo;

impl Foo {
    fn with_ref(&self) {} //~ ERROR passing `Foo` by reference
}

#[rustc_pass_by_value]
struct WithParameters<T, const N: usize, M = u32> {
    slice: [T; N],
    m: M,
}

impl<T> WithParameters<T, 1> {
    fn with_ref(&self) {} //~ ERROR passing `WithParameters<T, 1_usize>` by reference
}

impl<T> WithParameters<T, 1, u8> {
    fn with_ref(&self) {} //~ ERROR passing `WithParameters<T, 1_usize, u8>` by reference
}

fn main() {}
