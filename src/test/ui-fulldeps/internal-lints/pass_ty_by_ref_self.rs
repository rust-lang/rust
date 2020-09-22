// NOTE: This test doesn't actually require `fulldeps`
// so we could instead use it as an `ui` test.
//
// Considering that all other `internal-lints` are tested here
// this seems like the cleaner solution though.
#![feature(rustc_attrs)]
#![deny(rustc::ty_pass_by_reference)]
#![allow(unused)]

#[rustc_diagnostic_item = "TyCtxt"]
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
type Ty<'tcx> = &'tcx TyS<'tcx>;

impl<'tcx> TyS<'tcx> {
    fn by_value(self: Ty<'tcx>) {}
    fn by_ref(self: &Ty<'tcx>) {} //~ ERROR passing `Ty<'tcx>` by reference
}

fn main() {}
