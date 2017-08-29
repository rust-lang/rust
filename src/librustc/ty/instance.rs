// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use ty::{self, Ty, TypeFoldable, Substs};
use util::ppaux;

use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Instance<'tcx> {
    pub def: InstanceDef<'tcx>,
    pub substs: &'tcx Substs<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum InstanceDef<'tcx> {
    Item(DefId),
    Intrinsic(DefId),

    /// <fn() as FnTrait>::call_*
    /// def-id is FnTrait::call_*
    FnPtrShim(DefId, Ty<'tcx>),

    /// <Trait as Trait>::fn
    Virtual(DefId, usize),

    /// <[mut closure] as FnOnce>::call_once
    ClosureOnceShim { call_once: DefId },

    /// drop_in_place::<T>; None for empty drop glue.
    DropGlue(DefId, Option<Ty<'tcx>>),

    /// Builtin method implementation, e.g. `Clone::clone`.
    CloneShim(DefId, Ty<'tcx>),
}

impl<'tcx> InstanceDef<'tcx> {
    #[inline]
    pub fn def_id(&self) -> DefId {
        match *self {
            InstanceDef::Item(def_id) |
            InstanceDef::FnPtrShim(def_id, _) |
            InstanceDef::Virtual(def_id, _) |
            InstanceDef::Intrinsic(def_id, ) |
            InstanceDef::ClosureOnceShim { call_once: def_id } |
            InstanceDef::DropGlue(def_id, _) |
            InstanceDef::CloneShim(def_id, _) => def_id
        }
    }

    #[inline]
    pub fn def_ty<'a>(&self, tcx: ty::TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        tcx.type_of(self.def_id())
    }

    #[inline]
    pub fn attrs<'a>(&self, tcx: ty::TyCtxt<'a, 'tcx, 'tcx>) -> ty::Attributes<'tcx> {
        tcx.get_attrs(self.def_id())
    }
}

impl<'tcx> fmt::Display for Instance<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        ppaux::parameterized(f, self.substs, self.def_id(), &[])?;
        match self.def {
            InstanceDef::Item(_) => Ok(()),
            InstanceDef::Intrinsic(_) => {
                write!(f, " - intrinsic")
            }
            InstanceDef::Virtual(_, num) => {
                write!(f, " - shim(#{})", num)
            }
            InstanceDef::FnPtrShim(_, ty) => {
                write!(f, " - shim({:?})", ty)
            }
            InstanceDef::ClosureOnceShim { .. } => {
                write!(f, " - shim")
            }
            InstanceDef::DropGlue(_, ty) => {
                write!(f, " - shim({:?})", ty)
            }
            InstanceDef::CloneShim(_, ty) => {
                write!(f, " - shim({:?})", ty)
            }
        }
    }
}

impl<'a, 'b, 'tcx> Instance<'tcx> {
    pub fn new(def_id: DefId, substs: &'tcx Substs<'tcx>)
               -> Instance<'tcx> {
        assert!(substs.is_normalized_for_trans() && !substs.has_escaping_regions(),
                "substs of instance {:?} not normalized for trans: {:?}",
                def_id, substs);
        Instance { def: InstanceDef::Item(def_id), substs: substs }
    }

    pub fn mono(tcx: ty::TyCtxt<'a, 'tcx, 'b>, def_id: DefId) -> Instance<'tcx> {
        Instance::new(def_id, tcx.global_tcx().empty_substs_for_def_id(def_id))
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.def.def_id()
    }
}
