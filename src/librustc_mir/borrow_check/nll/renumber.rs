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
use rustc::ty::subst::Substs;
use rustc::ty::{self, ClosureSubsts, Ty, TypeFoldable};
use rustc::mir::{BasicBlock, Local, Location, Mir, Statement, StatementKind};
use rustc::mir::RETURN_PLACE;
use rustc::mir::visit::{MutVisitor, TyContext};
use rustc::infer::{InferCtxt, NLLRegionVariableOrigin};

use super::ToRegionVid;
use super::universal_regions::UniversalRegions;

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
pub fn renumber_mir<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    universal_regions: &UniversalRegions<'tcx>,
    mir: &mut Mir<'tcx>,
) {
    debug!("renumber_mir()");
    debug!("renumber_mir: mir.arg_count={:?}", mir.arg_count);

    // Update the return type and types of the arguments based on the
    // `universal_regions` computation.
    debug!("renumber_mir: output_ty={:?}", universal_regions.output_ty);
    mir.local_decls[RETURN_PLACE].ty = universal_regions.output_ty;
    for (&input_ty, local) in universal_regions
        .input_tys
        .iter()
        .zip((1..).map(Local::new))
    {
        debug!("renumber_mir: input_ty={:?} local={:?}", input_ty, local);
        mir.local_decls[local].ty = input_ty;
    }

    let mut visitor = NLLVisitor {
        infcx,
        arg_count: mir.arg_count,
    };
    visitor.visit_mir(mir);
}

struct NLLVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    arg_count: usize,
}

impl<'a, 'gcx, 'tcx> NLLVisitor<'a, 'gcx, 'tcx> {
    /// Replaces all regions appearing in `value` with fresh inference
    /// variables. This is what we do for almost the entire MIR, with
    /// the exception of the declared types of our arguments.
    fn renumber_regions<T>(&mut self, ty_context: TyContext, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        debug!("renumber_regions(value={:?})", value);

        self.infcx
            .tcx
            .fold_regions(value, &mut false, |_region, _depth| {
                let origin = NLLRegionVariableOrigin::Inferred(ty_context);
                self.infcx.next_nll_region_var(origin)
            })
    }

    /// Checks that all the regions appearing in `value` have already
    /// been renumbered. `FreeRegions` code should have done this.
    fn assert_free_regions_are_renumbered<T>(&self, value: &T)
    where
        T: TypeFoldable<'tcx>,
    {
        debug!("assert_free_regions_are_renumbered(value={:?})", value);

        self.infcx.tcx.for_each_free_region(value, |region| {
            region.to_region_vid(); // will panic if `region` is not renumbered
        });
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
            TyContext::ReturnTy(..) => true,
            TyContext::Location(..) => false,
        };
        debug!(
            "visit_ty(ty={:?}, is_arg={:?}, ty_context={:?})",
            ty,
            is_arg,
            ty_context
        );

        if is_arg {
            self.assert_free_regions_are_renumbered(ty);
        } else {
            *ty = self.renumber_regions(ty_context, ty);
        }

        debug!("visit_ty: ty={:?}", ty);
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>, location: Location) {
        debug!("visit_substs(substs={:?}, location={:?})", substs, location);

        let ty_context = TyContext::Location(location);
        *substs = self.renumber_regions(ty_context, &{ *substs });

        debug!("visit_substs: substs={:?}", substs);
    }

    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        debug!("visit_region(region={:?}, location={:?})", region, location);

        let old_region = *region;
        let ty_context = TyContext::Location(location);
        *region = self.renumber_regions(ty_context, &old_region);

        debug!("visit_region: region={:?}", region);
    }

    fn visit_const(&mut self, constant: &mut &'tcx ty::Const<'tcx>, location: Location) {
        let ty_context = TyContext::Location(location);
        *constant = self.renumber_regions(ty_context, &*constant);
    }

    fn visit_closure_substs(&mut self, substs: &mut ClosureSubsts<'tcx>, location: Location) {
        debug!(
            "visit_closure_substs(substs={:?}, location={:?})",
            substs,
            location
        );

        let ty_context = TyContext::Location(location);
        *substs = self.renumber_regions(ty_context, substs);

        debug!("visit_closure_substs: substs={:?}", substs);
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
