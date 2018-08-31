// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::mir::ProjectionElem;
use rustc::mir::{Local, Mir, Place, PlaceBase};
use rustc::mir::tcx::PlaceTy;
use rustc::ty::{self, TyCtxt};

/// Extension methods for the `Place` type.
crate trait PlaceExt<'tcx> {
    /// True if this is a deref of a raw pointer.
    fn is_unsafe_place(&self, tcx: TyCtxt<'_, '_, 'tcx>, mir: &Mir<'tcx>) -> bool;

    /// If this is a place like `x.f.g`, returns the local
    /// `x`. Returns `None` if this is based in a static.
    fn root_local(&self) -> Option<Local>;
}

impl<'tcx> PlaceExt<'tcx> for Place<'tcx> {
    fn is_unsafe_place(&self, tcx: TyCtxt<'_, '_, 'tcx>, mir: &Mir<'tcx>) -> bool {
        let mut is_unsafe_place = match self.base {
            PlaceBase::Promoted(_) |
            PlaceBase::Local(_) => false,
            PlaceBase::Static(ref static_) => {
                tcx.is_static(static_.def_id) == Some(hir::Mutability::MutMutable)
            }
        };

        let mut base_ty = self.base.ty(mir);
        for elem in self.elems.iter() {
            is_unsafe_place = match elem {
                ProjectionElem::Field(..)
                | ProjectionElem::Downcast(..)
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Index(_) => continue,
                ProjectionElem::Deref => {
                    match base_ty.sty {
                        ty::TyRawPtr(..) => true,
                        _ => continue,
                    }
                }
            };
            base_ty = PlaceTy::from(base_ty).projection_ty(tcx, elem).to_ty(tcx);
        }

        is_unsafe_place
    }

    fn root_local(&self) -> Option<Local> {
        match self.base {
            PlaceBase::Promoted(_) |
            PlaceBase::Static(_) => return None,
            PlaceBase::Local(local) => return Some(local),
        }
    }
}
