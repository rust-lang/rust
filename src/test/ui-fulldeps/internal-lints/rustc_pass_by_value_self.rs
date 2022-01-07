// compile-flags: -Z unstable-options
// NOTE: This test doesn't actually require `fulldeps`
// so we could instead use it as a `ui` test.
//
// Considering that all other `internal-lints` are tested here
// this seems like the cleaner solution though.
#![feature(rustc_attrs)]
#![deny(rustc::pass_by_value)]
#![allow(unused)]

#[rustc_diagnostic_item = "TyCtxt"]
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

#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
type Ty<'tcx> = &'tcx TyS<'tcx>;

impl<'tcx> TyS<'tcx> {
    fn by_value(self: Ty<'tcx>) {}
    fn by_ref(self: &Ty<'tcx>) {} //~ ERROR passing `Ty<'tcx>` by reference
}

fn main() {}
