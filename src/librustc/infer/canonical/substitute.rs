// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains code to substitute new values into a
//! `Canonical<'tcx, T>`.
//!
//! For an overview of what canonicaliation is and how it fits into
//! rustc, check out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html

use infer::canonical::{Canonical, CanonicalVarValues};
use ty::fold::{TypeFoldable, TypeFolder};
use ty::subst::UnpackedKind;
use ty::{self, Ty, TyCtxt, TypeFlags};

impl<'tcx, V> Canonical<'tcx, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    pub fn substitute(&self, tcx: TyCtxt<'_, '_, 'tcx>, var_values: &CanonicalVarValues<'tcx>) -> V
    where
        V: TypeFoldable<'tcx>,
    {
        self.substitute_projected(tcx, var_values, |value| value)
    }

    /// Allows one to apply a substitute to some subset of
    /// `self.value`. Invoke `projection_fn` with `self.value` to get
    /// a value V that is expressed in terms of the same canonical
    /// variables bound in `self` (usually this extracts from subset
    /// of `self`). Apply the substitution `var_values` to this value
    /// V, replacing each of the canonical variables.
    pub fn substitute_projected<T>(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        var_values: &CanonicalVarValues<'tcx>,
        projection_fn: impl FnOnce(&V) -> &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        assert_eq!(self.variables.len(), var_values.var_values.len());
        let value = projection_fn(&self.value);
        substitute_value(tcx, var_values, value)
    }
}

/// Substitute the values from `var_values` into `value`. `var_values`
/// must be values for the set of canonical variables that appear in
/// `value`.
pub(super) fn substitute_value<'a, 'tcx, T>(
    tcx: TyCtxt<'_, '_, 'tcx>,
    var_values: &CanonicalVarValues<'tcx>,
    value: &'a T,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    if var_values.var_values.is_empty() {
        debug_assert!(!value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS));
        value.clone()
    } else if !value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS) {
        value.clone()
    } else {
        value.fold_with(&mut CanonicalVarValuesSubst { tcx, var_values })
    }
}

struct CanonicalVarValuesSubst<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    var_values: &'cx CanonicalVarValues<'tcx>,
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for CanonicalVarValuesSubst<'cx, 'gcx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'_, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::TyInfer(ty::InferTy::CanonicalTy(c)) => {
                match self.var_values.var_values[c].unpack() {
                    UnpackedKind::Type(ty) => ty,
                    r => bug!("{:?} is a type but value is {:?}", c, r),
                }
            }
            _ => {
                if !t.has_type_flags(TypeFlags::HAS_CANONICAL_VARS) {
                    t
                } else {
                    t.super_fold_with(self)
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r {
            ty::RegionKind::ReCanonical(c) => match self.var_values.var_values[*c].unpack() {
                UnpackedKind::Lifetime(l) => l,
                r => bug!("{:?} is a region but value is {:?}", c, r),
            },
            _ => r.super_fold_with(self),
        }
    }
}
