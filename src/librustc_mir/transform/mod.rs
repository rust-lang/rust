// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::Mir;
use rustc::mir::transform::{MirPassIndex, MirSuite, MirSource};
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;

pub mod simplify_branches;
pub mod simplify;
pub mod erase_regions;
pub mod no_landing_pads;
pub mod type_check;
pub mod add_call_guards;
pub mod promote_consts;
pub mod qualify_consts;
pub mod dump_mir;
pub mod deaggregator;
pub mod instcombine;
pub mod copy_prop;
pub mod inline;

pub(crate) fn provide(providers: &mut Providers) {
    self::qualify_consts::provide(providers);
    *providers = Providers {
        ..*providers
    };
}

pub(crate) fn run_suite<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  source: MirSource,
                                  suite: MirSuite,
                                  mir: &mut Mir<'tcx>)
{
    let passes = tcx.mir_passes.passes(suite);

    for (pass, index) in passes.iter().zip(0..) {
        let pass_num = MirPassIndex(index);

        for hook in tcx.mir_passes.hooks() {
            hook.on_mir_pass(tcx, suite, pass_num, &pass.name(), source, &mir, false);
        }

        pass.run_pass(tcx, source, mir);

        for hook in tcx.mir_passes.hooks() {
            hook.on_mir_pass(tcx, suite, pass_num, &pass.name(), source, &mir, true);
        }
    }
}
