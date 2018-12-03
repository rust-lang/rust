// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::subst::Substs;
use rustc::ty::{self, ClosureSubsts, GeneratorSubsts, Ty, TypeFoldable};
use rustc::mir::{Location, Mir, UserTypeAnnotation};
use rustc::mir::visit::{MutVisitor, TyContext};
use rustc::infer::{InferCtxt, NLLRegionVariableOrigin};

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
pub fn renumber_mir<'tcx>(infcx: &InferCtxt<'_, '_, 'tcx>, mir: &mut Mir<'tcx>) {
    debug!("renumber_mir()");
    debug!("renumber_mir: mir.arg_count={:?}", mir.arg_count);

    let mut visitor = NLLVisitor { infcx };
    visitor.visit_mir(mir);
}

/// Replaces all regions appearing in `value` with fresh inference
/// variables.
pub fn renumber_regions<'tcx, T>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    value: &T,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!("renumber_regions(value={:?})", value);

    infcx
        .tcx
        .fold_regions(value, &mut false, |_region, _depth| {
            let origin = NLLRegionVariableOrigin::Existential;
            infcx.next_nll_region_var(origin)
        })
}

struct NLLVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> NLLVisitor<'a, 'gcx, 'tcx> {
    fn renumber_regions<T>(&mut self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        renumber_regions(self.infcx, value)
    }
}

impl<'a, 'gcx, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'gcx, 'tcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        debug!("visit_ty(ty={:?}, ty_context={:?})", ty, ty_context);

        *ty = self.renumber_regions(ty);

        debug!("visit_ty: ty={:?}", ty);
    }

    fn visit_user_type_annotation(&mut self, _ty: &mut UserTypeAnnotation<'tcx>) {
        // User type annotations represent the types that the user
        // wrote in the progarm. We don't want to erase the regions
        // from these types: rather, we want to add them as
        // constraints at type-check time.
        debug!("visit_user_type_annotation: skipping renumber");
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>, location: Location) {
        debug!("visit_substs(substs={:?}, location={:?})", substs, location);

        *substs = self.renumber_regions(&{ *substs });

        debug!("visit_substs: substs={:?}", substs);
    }

    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        debug!("visit_region(region={:?}, location={:?})", region, location);

        let old_region = *region;
        *region = self.renumber_regions(&old_region);

        debug!("visit_region: region={:?}", region);
    }

    fn visit_const(&mut self, constant: &mut &'tcx ty::Const<'tcx>, _location: Location) {
        *constant = self.renumber_regions(&*constant);
    }

    fn visit_generator_substs(&mut self,
                              substs: &mut GeneratorSubsts<'tcx>,
                              location: Location) {
        debug!(
            "visit_generator_substs(substs={:?}, location={:?})",
            substs,
            location,
        );

        *substs = self.renumber_regions(substs);

        debug!("visit_generator_substs: substs={:?}", substs);
    }

    fn visit_closure_substs(&mut self, substs: &mut ClosureSubsts<'tcx>, location: Location) {
        debug!(
            "visit_closure_substs(substs={:?}, location={:?})",
            substs,
            location
        );

        *substs = self.renumber_regions(substs);

        debug!("visit_closure_substs: substs={:?}", substs);
    }
}
