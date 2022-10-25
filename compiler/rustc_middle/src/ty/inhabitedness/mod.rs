//! This module contains logic for determining whether a type is inhabited or
//! uninhabited. The [`InhabitedPredicate`] type captures the minimum
//! information needed to determine whether a type is inhabited given a
//! `ParamEnv` and module ID.
//!
//! # Example
//! ```rust
//! enum Void {}
//! mod a {
//!     pub mod b {
//!         pub struct SecretlyUninhabited {
//!             _priv: !,
//!         }
//!     }
//! }
//!
//! mod c {
//!     pub struct AlsoSecretlyUninhabited {
//!         _priv: Void,
//!     }
//!     mod d {
//!     }
//! }
//!
//! struct Foo {
//!     x: a::b::SecretlyUninhabited,
//!     y: c::AlsoSecretlyUninhabited,
//! }
//! ```
//! In this code, the type `Foo` will only be visibly uninhabited inside the
//! modules `b`, `c` and `d`. Calling `uninhabited_predicate` on `Foo` will
//! return `NotInModule(b) AND NotInModule(c)`.
//!
//! We need this information for pattern-matching on `Foo` or types that contain
//! `Foo`.
//!
//! # Example
//! ```rust
//! let foo_result: Result<T, Foo> = ... ;
//! let Ok(t) = foo_result;
//! ```
//! This code should only compile in modules where the uninhabitedness of `Foo`
//! is visible.

use crate::ty::context::TyCtxt;
use crate::ty::{self, DefId, Ty, VariantDef, Visibility};

use rustc_type_ir::sty::TyKind::*;

pub mod inhabited_predicate;

pub use inhabited_predicate::InhabitedPredicate;

pub(crate) fn provide(providers: &mut ty::query::Providers) {
    *providers =
        ty::query::Providers { inhabited_predicate_adt, inhabited_predicate_type, ..*providers };
}

impl<'tcx> TyCtxt<'tcx> {
    /// Checks whether a type is visibly uninhabited from a particular module.
    ///
    /// # Example
    /// ```
    /// #![feature(never_type)]
    /// # fn main() {}
    /// enum Void {}
    /// mod a {
    ///     pub mod b {
    ///         pub struct SecretlyUninhabited {
    ///             _priv: !,
    ///         }
    ///     }
    /// }
    ///
    /// mod c {
    ///     use super::Void;
    ///     pub struct AlsoSecretlyUninhabited {
    ///         _priv: Void,
    ///     }
    ///     mod d {
    ///     }
    /// }
    ///
    /// struct Foo {
    ///     x: a::b::SecretlyUninhabited,
    ///     y: c::AlsoSecretlyUninhabited,
    /// }
    /// ```
    /// In this code, the type `Foo` will only be visibly uninhabited inside the
    /// modules b, c and d. This effects pattern-matching on `Foo` or types that
    /// contain `Foo`.
    ///
    /// # Example
    /// ```ignore (illustrative)
    /// let foo_result: Result<T, Foo> = ... ;
    /// let Ok(t) = foo_result;
    /// ```
    /// This code should only compile in modules where the uninhabitedness of Foo is
    /// visible.
    pub fn is_ty_uninhabited_from(
        self,
        module: DefId,
        ty: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> bool {
        !ty.inhabited_predicate(self).apply(self, param_env, module)
    }
}

/// Returns an `InhabitedPredicate` that is generic over type parameters and
/// requires calling [`InhabitedPredicate::subst`]
fn inhabited_predicate_adt(tcx: TyCtxt<'_>, def_id: DefId) -> InhabitedPredicate<'_> {
    if let Some(def_id) = def_id.as_local() {
        if matches!(tcx.representability(def_id), ty::Representability::Infinite) {
            return InhabitedPredicate::True;
        }
    }
    let adt = tcx.adt_def(def_id);
    InhabitedPredicate::any(
        tcx,
        adt.variants().iter().map(|variant| variant.inhabited_predicate(tcx, adt)),
    )
}

impl<'tcx> VariantDef {
    /// Calculates the forest of `DefId`s from which this variant is visibly uninhabited.
    pub fn inhabited_predicate(
        &self,
        tcx: TyCtxt<'tcx>,
        adt: ty::AdtDef<'_>,
    ) -> InhabitedPredicate<'tcx> {
        debug_assert!(!adt.is_union());
        if self.is_field_list_non_exhaustive() && !self.def_id.is_local() {
            // Non-exhaustive variants from other crates are always considered inhabited.
            return InhabitedPredicate::True;
        }
        InhabitedPredicate::all(
            tcx,
            self.fields.iter().map(|field| {
                let pred = tcx.type_of(field.did).inhabited_predicate(tcx);
                if adt.is_enum() {
                    return pred;
                }
                match field.vis {
                    Visibility::Public => pred,
                    Visibility::Restricted(from) => {
                        pred.or(tcx, InhabitedPredicate::NotInModule(from))
                    }
                }
            }),
        )
    }
}

impl<'tcx> Ty<'tcx> {
    pub fn inhabited_predicate(self, tcx: TyCtxt<'tcx>) -> InhabitedPredicate<'tcx> {
        match self.kind() {
            // For now, union`s are always considered inhabited
            Adt(adt, _) if adt.is_union() => InhabitedPredicate::True,
            // Non-exhaustive ADTs from other crates are always considered inhabited
            Adt(adt, _) if adt.is_variant_list_non_exhaustive() && !adt.did().is_local() => {
                InhabitedPredicate::True
            }
            Never => InhabitedPredicate::False,
            Param(_) | Projection(_) => InhabitedPredicate::GenericType(self),
            Tuple(tys) if tys.is_empty() => InhabitedPredicate::True,
            // use a query for more complex cases
            Adt(..) | Array(..) | Tuple(_) => tcx.inhabited_predicate_type(self),
            // references and other types are inhabited
            _ => InhabitedPredicate::True,
        }
    }
}

/// N.B. this query should only be called through `Ty::inhabited_predicate`
fn inhabited_predicate_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> InhabitedPredicate<'tcx> {
    match *ty.kind() {
        Adt(adt, substs) => tcx.inhabited_predicate_adt(adt.did()).subst(tcx, substs),

        Tuple(tys) => {
            InhabitedPredicate::all(tcx, tys.iter().map(|ty| ty.inhabited_predicate(tcx)))
        }

        // If we can evaluate the array length before having a `ParamEnv`, then
        // we can simplify the predicate. This is an optimization.
        Array(ty, len) => match len.kind().try_to_machine_usize(tcx) {
            Some(0) => InhabitedPredicate::True,
            Some(1..) => ty.inhabited_predicate(tcx),
            None => ty.inhabited_predicate(tcx).or(tcx, InhabitedPredicate::ConstIsZero(len)),
        },

        _ => bug!("unexpected TyKind, use `Ty::inhabited_predicate`"),
    }
}
