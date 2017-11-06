// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::Idx;
use rustc::ty::subst::{Kind, Substs};
use rustc::ty::{self, ClosureSubsts, RegionKind, RegionVid, Ty, TypeFoldable};
use rustc::mir::{BasicBlock, Local, Location, Mir, Rvalue, Statement, StatementKind};
use rustc::mir::visit::{MutVisitor, TyContext};
use rustc::infer::{self as rustc_infer, InferCtxt};
use syntax_pos::DUMMY_SP;
use std::collections::HashMap;

use super::free_regions::FreeRegions;

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
pub fn renumber_mir<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    free_regions: &FreeRegions<'tcx>,
    mir: &mut Mir<'tcx>,
) -> usize {
    // Create inference variables for each of the free regions
    // declared on the function signature.
    let free_region_inference_vars = (0..free_regions.indices.len())
        .map(|_| {
            infcx.next_region_var(rustc_infer::MiscVariable(DUMMY_SP))
        })
        .collect();

    let mut visitor = NLLVisitor {
        infcx,
        lookup_map: HashMap::new(),
        num_region_variables: free_regions.indices.len(),
        free_regions,
        free_region_inference_vars,
        arg_count: mir.arg_count,
    };
    visitor.visit_mir(mir);
    visitor.num_region_variables
}

struct NLLVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    lookup_map: HashMap<RegionVid, TyContext>,
    num_region_variables: usize,
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    free_regions: &'a FreeRegions<'tcx>,
    free_region_inference_vars: Vec<ty::Region<'tcx>>,
    arg_count: usize,
}

impl<'a, 'gcx, 'tcx> NLLVisitor<'a, 'gcx, 'tcx> {
    /// Replaces all regions appearing in `value` with fresh inference
    /// variables. This is what we do for almost the entire MIR, with
    /// the exception of the declared types of our arguments.
    fn renumber_regions<T>(&mut self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.infcx
            .tcx
            .fold_regions(value, &mut false, |_region, _depth| {
                self.num_region_variables += 1;
                self.infcx
                    .next_region_var(rustc_infer::MiscVariable(DUMMY_SP))
            })
    }

    /// Renumbers the regions appearing in `value`, but those regions
    /// are expected to be free regions from the function signature.
    fn renumber_free_regions<T>(&mut self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.infcx
            .tcx
            .fold_regions(value, &mut false, |region, _depth| {
                let index = self.free_regions.indices[&region];
                self.free_region_inference_vars[index]
            })
    }

    fn store_region(&mut self, region: &RegionKind, lookup: TyContext) {
        if let RegionKind::ReVar(rid) = *region {
            self.lookup_map.entry(rid).or_insert(lookup);
        }
    }

    fn store_ty_regions(&mut self, ty: &Ty<'tcx>, ty_context: TyContext) {
        for region in ty.regions() {
            self.store_region(region, ty_context);
        }
    }

    fn store_kind_regions(&mut self, kind: &'tcx Kind, ty_context: TyContext) {
        if let Some(ty) = kind.as_type() {
            self.store_ty_regions(&ty, ty_context);
        } else if let Some(region) = kind.as_region() {
            self.store_region(region, ty_context);
        }
    }

    fn is_argument_or_return_slot(&self, local: Local) -> bool {
        // The first argument is return slot, next N are arguments.
        local.index() <= self.arg_count
    }
}

impl<'a, 'gcx, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'gcx, 'tcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        let is_arg = match ty_context {
            TyContext::LocalDecl { local, .. } => self.is_argument_or_return_slot(local),
            _ => false,
        };

        let old_ty = *ty;
        *ty = if is_arg {
            self.renumber_free_regions(&old_ty)
        } else {
            self.renumber_regions(&old_ty)
        };
        self.store_ty_regions(ty, ty_context);
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>, location: Location) {
        *substs = self.renumber_regions(&{ *substs });
        let ty_context = TyContext::Location(location);
        for kind in *substs {
            self.store_kind_regions(kind, ty_context);
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        match *rvalue {
            Rvalue::Ref(ref mut r, _, _) => {
                let old_r = *r;
                *r = self.renumber_regions(&old_r);
                let ty_context = TyContext::Location(location);
                self.store_region(r, ty_context);
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

    fn visit_closure_substs(&mut self, substs: &mut ClosureSubsts<'tcx>, location: Location) {
        *substs = self.renumber_regions(substs);
        let ty_context = TyContext::Location(location);
        for kind in substs.substs {
            self.store_kind_regions(kind, ty_context);
        }
    }

    fn visit_statement(
        &mut self,
        block: BasicBlock,
        statement: &mut Statement<'tcx>,
        location: Location,
    ) {
        if let StatementKind::EndRegion(_) = statement.kind {
            statement.kind = StatementKind::Nop;
        }
        self.super_statement(block, statement, location);
    }
}
