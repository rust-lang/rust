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
use rustc::mir::{Local, Mir, Place};
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
        match self {
            Place::Local(_) => false,
            Place::Static(static_) => {
                tcx.is_static(static_.def_id) == Some(hir::Mutability::MutMutable)
            }
            Place::Projection(proj) => match proj.elem {
                ProjectionElem::Field(..)
                | ProjectionElem::Downcast(..)
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Index(_) => proj.base.is_unsafe_place(tcx, mir),
                ProjectionElem::Deref => {
                    let ty = proj.base.ty(mir, tcx).to_ty(tcx);
                    match ty.sty {
                        ty::TyRawPtr(..) => true,
                        _ => proj.base.is_unsafe_place(tcx, mir),
                    }
                }
            },
        }
    }

    fn root_local(&self) -> Option<Local> {
        let mut p = self;
        loop {
            match p {
                Place::Projection(pi) => p = &pi.base,
                Place::Static(_) => return None,
                Place::Local(l) => return Some(*l),
            }
        }
    }
}
