// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TypeFoldable;
use rustc::ty::subst::{Kind, Substs};
use rustc::ty::{Ty, TyCtxt, ClosureSubsts, RegionVid, RegionKind};
use rustc::mir::{Mir, Location, Rvalue, BasicBlock, Statement, StatementKind};
use rustc::mir::visit::{MutVisitor, Lookup};
use rustc::mir::transform::{MirPass, MirSource};
use rustc::infer::{self, InferCtxt};
use syntax_pos::DUMMY_SP;
use std::collections::HashMap;

#[allow(dead_code)]
struct NLLVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    lookup_map: HashMap<RegionVid, Lookup>,
    infcx: InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> NLLVisitor<'a, 'gcx, 'tcx> {
    pub fn new(infcx: InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        NLLVisitor {
            infcx: infcx,
            lookup_map: HashMap::new(),
        }
    }

    pub fn into_results(self) -> HashMap<RegionVid, Lookup> {
        self.lookup_map
    }

    fn renumber_regions<T>(&self, value: &T) -> T where T: TypeFoldable<'tcx> {
        self.infcx.tcx.fold_regions(value, &mut false, |_region, _depth| {
            self.infcx.next_region_var(infer::MiscVariable(DUMMY_SP))
        })
    }

    fn store_region(&mut self, region: &RegionKind, lookup: Lookup) {
        if let RegionKind::ReVar(rid) = *region {
            self.lookup_map.entry(rid).or_insert(lookup);
        }
    }

    fn store_ty_regions(&mut self, ty: &Ty<'tcx>, lookup: Lookup) {
        for region in ty.regions() {
            self.store_region(region, lookup);
        }
    }

    fn store_kind_regions(&mut self, kind: &'tcx Kind, lookup: Lookup) {
        if let Some(ty) = kind.as_type() {
            self.store_ty_regions(&ty, lookup);
        } else if let Some(region) = kind.as_region() {
            self.store_region(region, lookup);
        }
    }
}

impl<'a, 'gcx, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'gcx, 'tcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, lookup: Lookup) {
        let old_ty = *ty;
        *ty = self.renumber_regions(&old_ty);
        self.store_ty_regions(ty, lookup);
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>, location: Location) {
        *substs = self.renumber_regions(&{*substs});
        let lookup = Lookup::Loc(location);
        for kind in *substs {
            self.store_kind_regions(kind, lookup);
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        match *rvalue {
            Rvalue::Ref(ref mut r, _, _) => {
                let old_r = *r;
                *r = self.renumber_regions(&old_r);
                let lookup = Lookup::Loc(location);
                self.store_region(r, lookup);
            }
            Rvalue::Use(..) |
            Rvalue::Repeat(..) |
            Rvalue::Len(..) |
            Rvalue::Cast(..) |
            Rvalue::BinaryOp(..) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::UnaryOp(..) |
            Rvalue::Discriminant(..) |
            Rvalue::NullaryOp(..) |
            Rvalue::Aggregate(..) => {
                // These variants don't contain regions.
            }
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_closure_substs(&mut self,
                            substs: &mut ClosureSubsts<'tcx>,
                            location: Location) {
        *substs = self.renumber_regions(substs);
        let lookup = Lookup::Loc(location);
        for kind in substs.substs {
            self.store_kind_regions(kind, lookup);
        }
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        if let StatementKind::EndRegion(_) = statement.kind {
            statement.kind = StatementKind::Nop;
        }
        self.super_statement(block, statement, location);
    }
}

// MIR Pass for non-lexical lifetimes
pub struct NLL;

impl MirPass for NLL {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource,
                          mir: &mut Mir<'tcx>) {
        if !tcx.sess.opts.debugging_opts.nll {
            return;
        }

        tcx.infer_ctxt().enter(|infcx| {
            // Clone mir so we can mutate it without disturbing the rest of the compiler
            let mut renumbered_mir = mir.clone();
            let mut visitor = NLLVisitor::new(infcx);
            visitor.visit_mir(&mut renumbered_mir);
            let _results = visitor.into_results();
        })
    }
}