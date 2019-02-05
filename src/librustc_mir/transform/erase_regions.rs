//! This pass erases all early-bound regions from the types occurring in the MIR.
//! We want to do this once just before codegen, so codegen does not have to take
//! care erasing regions all over the place.
//! NOTE:  We do NOT erase regions of statements that are relevant for
//! "types-as-contracts"-validation, namely, AcquireValid, ReleaseValid

use rustc::ty::TyCtxt;
use rustc::mir::*;
use transform::{MirPass, MirSource};

pub struct EraseRegions;

impl MirPass for EraseRegions {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource,
                          mir: &mut Mir<'tcx>) {
        *mir = tcx.erase_regions(mir);
    }
}
