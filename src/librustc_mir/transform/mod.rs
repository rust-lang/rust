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
use rustc::mir::transform::{MirCtxt, MirPassIndex, MirSuite, MirSource, MIR_OPTIMIZED, PassId};
use rustc::ty::steal::Steal;
use rustc::ty::TyCtxt;
use rustc::ty::maps::{Multi, Providers};
use std::cell::Ref;

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
    let len = passes.len_passes(suite);
    assert!(len > 0, "no passes in {:?}", suite);
    tcx.mir_pass((suite, MirPassIndex(len - 1), def_id))
}

fn mir_pass<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                      (suite, pass_num, def_id): (MirSuite, MirPassIndex, DefId))
                      -> Multi<PassId, &'tcx Steal<Mir<'tcx>>>
{
    let passes = &tcx.mir_passes;
    let pass = passes.pass(suite, pass_num);
    let mir_ctxt = MirCtxtImpl { tcx, pass_num, suite, def_id };

    for hook in passes.hooks() {
        hook.on_mir_pass(&mir_ctxt, None);
    }

    let mir = pass.run_pass(&mir_ctxt);

    let key = &(suite, pass_num, def_id);
    for hook in passes.hooks() {
        for (&(_, _, k), v) in mir.iter(key) {
            let v = &v.borrow();
            hook.on_mir_pass(&mir_ctxt, Some((k, v)));
        }
    }

    mir
}

struct MirCtxtImpl<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pass_num: MirPassIndex,
    suite: MirSuite,
    def_id: DefId
}

impl<'a, 'tcx> MirCtxt<'a, 'tcx> for MirCtxtImpl<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    fn suite(&self) -> MirSuite {
        self.suite
    }

    fn pass_num(&self) -> MirPassIndex {
        self.pass_num
    }

    fn def_id(&self) -> DefId {
        self.def_id
    }

    fn source(&self) -> MirSource {
        let id = self.tcx.hir.as_local_node_id(self.def_id)
                             .expect("mir source requires local def-id");
        MirSource::from_node(self.tcx, id)
    }

    fn read_previous_mir(&self) -> Ref<'tcx, Mir<'tcx>> {
        self.previous_mir(self.def_id).borrow()
    }

    fn steal_previous_mir(&self) -> Mir<'tcx> {
        self.previous_mir(self.def_id).steal()
    }

    fn read_previous_mir_of(&self, def_id: DefId) -> Ref<'tcx, Mir<'tcx>> {
        self.previous_mir(def_id).borrow()
    }

    fn steal_previous_mir_of(&self, def_id: DefId) -> Mir<'tcx> {
        self.previous_mir(def_id).steal()
    }
}

impl<'a, 'tcx> MirCtxtImpl<'a, 'tcx> {
    fn previous_mir(&self, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
        let MirSuite(suite) = self.suite;
        let MirPassIndex(pass_num) = self.pass_num;
        if pass_num > 0 {
            self.tcx.mir_pass((MirSuite(suite), MirPassIndex(pass_num - 1), def_id))
        } else if suite > 0 {
            self.tcx.mir_suite((MirSuite(suite - 1), def_id))
        } else {
            self.tcx.mir_build(def_id)
        }
    }
}
