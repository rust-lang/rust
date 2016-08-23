// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt};
use common::*;
use rustc::util::ppaux;

use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Instance<'tcx> {
    pub def: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

impl<'tcx> fmt::Display for Instance<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ppaux::parameterized(f, &self.substs, self.def, ppaux::Ns::Value, &[])
    }
}

impl<'tcx> Instance<'tcx> {
    pub fn new(def_id: DefId, substs: &'tcx Substs<'tcx>)
               -> Instance<'tcx> {
        assert!(substs.regions.iter().all(|&r| r == ty::ReErased));
        Instance { def: def_id, substs: substs }
    }
    pub fn mono<'a>(scx: &SharedCrateContext<'a, 'tcx>, def_id: DefId) -> Instance<'tcx> {
        Instance::new(def_id, scx.empty_substs_for_def_id(def_id))
    }
}

/// Monomorphizes a type from the AST by first applying the in-scope
/// substitutions and then normalizing any associated types.
pub fn apply_param_substs<'a, 'tcx, T>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       param_substs: &Substs<'tcx>,
                                       value: &T)
                                       -> T
    where T: TransNormalize<'tcx>
{
    let substituted = value.subst(tcx, param_substs);
    tcx.normalize_associated_type(&substituted)
}


/// Returns the normalized type of a struct field
pub fn field_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_substs: &Substs<'tcx>,
                          f: ty::FieldDef<'tcx>)
                          -> Ty<'tcx>
{
    tcx.normalize_associated_type(&f.ty(tcx, param_substs))
}
