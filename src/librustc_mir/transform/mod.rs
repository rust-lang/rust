// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::mir::Mir;
use rustc::mir::transform::{MirPassIndex, MirSuite, MirSource, MIR_VALIDATED, MIR_OPTIMIZED};
use rustc::ty::{self, TyCtxt};
use rustc::ty::steal::Steal;
use rustc::ty::maps::Providers;
use syntax_pos::DUMMY_SP;

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
pub mod interprocedural;

pub fn provide(providers: &mut Providers) {
    self::qualify_consts::provide(providers);
    *providers = Providers {
        optimized_mir,
        mir_suite,
        mir_pass,
        ..*providers
    };
}

fn optimized_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Mir<'tcx> {
    let mir = tcx.mir_suite((MIR_OPTIMIZED, def_id)).steal();
    tcx.alloc_mir(mir)
}

fn mir_suite<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       (suite, def_id): (MirSuite, DefId))
                       -> &'tcx Steal<Mir<'tcx>>
{
    let passes = &tcx.mir_passes;

    if suite == MIR_VALIDATED {
        let id = tcx.hir.as_local_node_id(def_id).expect("mir source requires local def-id");
        let source = MirSource::from_node(tcx, id);
        if let MirSource::Const(_) = source {
            // Ensure that we compute the `mir_const_qualif` for
            // constants at this point, before we do any further
            // optimization (and before we steal the previous
            // MIR). We don't directly need the result, so we can
            // just force it.
            ty::queries::mir_const_qualif::force(tcx, DUMMY_SP, def_id);
        }
    }

    let len = passes.len_passes(suite);
    assert!(len > 0, "no passes in {:?}", suite);
    tcx.mir_pass((suite, MirPassIndex(len - 1), def_id))
}

fn mir_pass<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                      (suite, pass_num, def_id): (MirSuite, MirPassIndex, DefId))
                      -> &'tcx Steal<Mir<'tcx>>
{
    let passes = &tcx.mir_passes;
    let pass = passes.pass(suite, pass_num);

    let id = tcx.hir.as_local_node_id(def_id).expect("mir source requires local def-id");
    let source = MirSource::from_node(tcx, id);

    let mut mir = {
        let MirSuite(suite) = suite;
        let MirPassIndex(pass_num) = pass_num;
        if pass_num > 0 {
            tcx.mir_pass((MirSuite(suite), MirPassIndex(pass_num - 1), def_id)).steal()
        } else if suite > 0 {
            tcx.mir_suite((MirSuite(suite - 1), def_id)).steal()
        } else {
            tcx.mir_build(def_id).steal()
        }
    };

    for hook in passes.hooks() {
        hook.on_mir_pass(tcx, suite, pass_num, &pass.name(), source, &mir, false);
    }

    pass.run_pass(tcx, source, &mut mir);

    for hook in passes.hooks() {
        hook.on_mir_pass(tcx, suite, pass_num, &pass.name(), source, &mir, true);
    }

    tcx.alloc_steal_mir(mir)
}

