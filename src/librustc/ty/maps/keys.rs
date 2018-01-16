// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Defines the set of legal keys that can be used in queries.

use hir::def_id::{CrateNum, DefId, LOCAL_CRATE, DefIndex};
use ty::{self, Ty, TyCtxt};
use ty::subst::Substs;
use ty::fast_reject::SimplifiedType;
use mir;

use std::fmt::Debug;
use std::hash::Hash;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::symbol::InternedString;

/// The `Key` trait controls what types can legally be used as the key
/// for a query.
pub trait Key: Clone + Hash + Eq + Debug {
    /// Given an instance of this key, what crate is it referring to?
    /// This is used to find the provider.
    fn map_crate(&self) -> CrateNum;

    /// In the event that a cycle occurs, if no explicit span has been
    /// given for a query with key `self`, what span should we use?
    fn default_span(&self, tcx: TyCtxt) -> Span;
}

impl<'tcx> Key for ty::InstanceDef<'tcx> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl Key for CrateNum {
    fn map_crate(&self) -> CrateNum {
        *self
    }
    fn default_span(&self, _: TyCtxt) -> Span {
        DUMMY_SP
    }
}

impl Key for DefIndex {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _tcx: TyCtxt) -> Span {
        DUMMY_SP
    }
}

impl Key for DefId {
    fn map_crate(&self) -> CrateNum {
        self.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(*self)
    }
}

impl Key for (DefId, DefId) {
    fn map_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, DefId) {
    fn map_crate(&self) -> CrateNum {
        self.0
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (DefId, SimplifiedType) {
    fn map_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (DefId, &'tcx Substs<'tcx>) {
    fn map_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>) {
    fn map_crate(&self) -> CrateNum {
        self.1.def_id().krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(self.1.def_id())
    }
}

impl<'tcx> Key for ty::PolyTraitRef<'tcx>{
    fn map_crate(&self) -> CrateNum {
        self.def_id().krate
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for Ty<'tcx> {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::ParamEnvAnd<'tcx, T> {
    fn map_crate(&self) -> CrateNum {
        self.value.map_crate()
    }
    fn default_span(&self, tcx: TyCtxt) -> Span {
        self.value.default_span(tcx)
    }
}

impl Key for InternedString {
    fn map_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _tcx: TyCtxt) -> Span {
        DUMMY_SP
    }
}
