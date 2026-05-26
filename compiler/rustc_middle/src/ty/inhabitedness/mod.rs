//! This module contains logic for determining whether a type is inhabited or
//! uninhabited. The [`InhabitedPredicate`] type captures the minimum
//! information needed to determine whether a type is inhabited given a
//! `ParamEnv` and module ID.
//!
//! # Example
//! ```rust
//! #![feature(never_type)]
//! mod a {
//!     pub mod b {
//!         pub struct SecretlyUninhabited {
//!             _priv: !,
//!         }
//!     }
//! }
//!
//! mod c {
//!     enum Void {}
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
//! modules `b`, `c` and `d`. Calling `inhabited_predicate` on `Foo` will
//! return `NotInModule(b) AND NotInModule(c)`.
//!
//! We need this information for pattern-matching on `Foo` or types that contain
//! `Foo`.
//!
//! # Example
//! ```ignore(illustrative)
//! let foo_result: Result<T, Foo> = ... ;
//! let Ok(t) = foo_result;
//! ```
//! This code should only compile in modules where the uninhabitedness of `Foo`
//! is visible.

use std::assert_matches;

use rustc_data_structures::fx::FxHashSet;
use rustc_type_ir::TyKind::*;
use tracing::instrument;

use crate::query::Providers;
use crate::ty::{self, DefId, Ty, TyCtxt, TypeVisitableExt, VariantDef, Visibility};

pub mod inhabited_predicate;

pub use inhabited_predicate::InhabitedPredicate;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        inhabited_predicate_adt,
        inhabited_predicate_type,
        is_opsem_inhabited_raw,
        ..*providers
    };
}

/// Returns an `InhabitedPredicate` that is generic over type parameters and
/// requires calling [`InhabitedPredicate::instantiate`]
fn inhabited_predicate_adt(tcx: TyCtxt<'_>, def_id: DefId) -> InhabitedPredicate<'_> {
    if let Some(def_id) = def_id.as_local() {
        tcx.ensure_ok().check_representability(def_id);
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
        InhabitedPredicate::all(
            tcx,
            self.fields.iter().map(|field| {
                let pred = tcx
                    .type_of(field.did)
                    .instantiate_identity()
                    .skip_norm_wip()
                    .inhabited_predicate(tcx);
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
    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn inhabited_predicate(self, tcx: TyCtxt<'tcx>) -> InhabitedPredicate<'tcx> {
        debug_assert!(!self.has_infer());
        match self.kind() {
            // For now, unions are always considered inhabited
            Adt(adt, _) if adt.is_union() => InhabitedPredicate::True,
            // Non-exhaustive ADTs from other crates are always considered inhabited
            Adt(adt, _) if adt.variant_list_has_applicable_non_exhaustive() => {
                InhabitedPredicate::True
            }
            Never => InhabitedPredicate::False,
            Param(_)
            | Alias(ty::AliasTy {
                kind: ty::Inherent { .. } | ty::Projection { .. } | ty::Free { .. },
                ..
            }) => InhabitedPredicate::GenericType(self),
            &Alias(ty::AliasTy { kind: ty::Opaque { def_id }, args, .. }) => {
                match def_id.as_local() {
                    // Foreign opaque is considered inhabited.
                    None => InhabitedPredicate::True,
                    // Local opaque type may possibly be revealed.
                    Some(local_def_id) => {
                        let key = ty::OpaqueTypeKey { def_id: local_def_id, args };
                        InhabitedPredicate::OpaqueType(key)
                    }
                }
            }
            Tuple(tys) if tys.is_empty() => InhabitedPredicate::True,
            // use a query for more complex cases
            Adt(..) | Array(..) | Tuple(_) => tcx.inhabited_predicate_type(self),
            // references and other types are inhabited
            _ => InhabitedPredicate::True,
        }
    }

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
    pub fn is_inhabited_from(
        self,
        tcx: TyCtxt<'tcx>,
        module: DefId,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> bool {
        self.inhabited_predicate(tcx).apply(tcx, typing_env, module)
    }

    /// Returns true if the type is uninhabited without regard to visibility.
    ///
    /// This is still conservative; for instance, a `#[non_exhaustive]` enum *in another crate*
    /// is always considered inhabited.
    pub fn is_privately_uninhabited(
        self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> bool {
        !self.inhabited_predicate(tcx).apply_ignore_module(tcx, typing_env)
    }

    /// Returns whether `self` is considered inhabited on the opsem level, i.e., its validity
    /// invariant might be satisfiable. `self` is expected to be monomorphic and normalized.
    pub fn is_opsem_inhabited(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        // Handle simple cases directly, use the query with its cache for the rest.
        is_opsem_inhabited_recursor(self, tcx, &mut (), /* stop_at_ref */ false, &|ty, _, _| {
            // ADT handler: stop recursing, invoke the query.
            tcx.is_opsem_inhabited_raw(typing_env.as_query_input(ty))
        })
    }
}

/// N.B. this query should only be called through `Ty::inhabited_predicate`
fn inhabited_predicate_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> InhabitedPredicate<'tcx> {
    match *ty.kind() {
        Adt(adt, args) => tcx.inhabited_predicate_adt(adt.did()).instantiate(tcx, args),

        Tuple(tys) => {
            InhabitedPredicate::all(tcx, tys.iter().map(|ty| ty.inhabited_predicate(tcx)))
        }

        // If we can evaluate the array length before having a `ParamEnv`, then
        // we can simplify the predicate. This is an optimization.
        Array(ty, len) => match len.try_to_target_usize(tcx) {
            Some(0) => InhabitedPredicate::True,
            Some(1..) => ty.inhabited_predicate(tcx),
            None => ty.inhabited_predicate(tcx).or(tcx, InhabitedPredicate::ConstIsZero(len)),
        },

        _ => bug!("unexpected TyKind, use `Ty::inhabited_predicate`"),
    }
}

/// Recurse over a type to determine whether it is inhabited on the opsem level.
/// Key constraints are:
/// - if a type's validity invariant is satisfiable, it must be opsem-inhabited.
/// - if a type's layout is marked uninhabited, it must be opsem-uninhabited.
///
/// Beyond that, the value returned by this function is not a stable guarantee.
///
/// When we encounter an ADT, we call `adt_handler`, giving it as its last argument a closure that
/// it can invoke to continue the recursion. This lets us share the logic for "simple" cases
/// (i.e., everything except for ADTs) between `Ty::is_opsem_inhabited` and the query.
///
/// `seen` is used to detect infinite recursion: the set contains all ADTs that we encountered
/// on our path to the current type.
/// If `stop_at_ref` is true, we stop recursing at the next reference we encounter.
fn is_opsem_inhabited_recursor<'tcx, SEEN>(
    ty: Ty<'tcx>,
    tcx: TyCtxt<'tcx>,
    seen: &mut SEEN,
    stop_at_ref: bool,
    adt_handler: &impl Fn(
        Ty<'tcx>,
        &mut SEEN,
        &dyn Fn(Ty<'tcx>, &mut SEEN, /* stop_at_ref */ bool) -> bool,
    ) -> bool,
) -> bool {
    match *ty.kind() {
        // Trivially (un)inhabited types
        ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Bool
        | ty::Char
        | ty::Str
        | ty::Foreign(..)
        | ty::RawPtr(..)
        | ty::FnPtr(..)
        | ty::FnDef(..) => true,
        ty::Dynamic(..) => true, // We can't reason about traits, assume they are inhabited
        ty::Slice(..) => true,   // Slices can always be empty
        ty::Never => false,

        // Types where we recurse
        ty::Ref(_, pointee, _) => {
            if stop_at_ref {
                // Bailing out here is safe as the layout code always considers references
                // inhabited, so the implication ("layout uninhabited => opsem uninhabited")
                // is upheld.
                return true;
            }
            is_opsem_inhabited_recursor(pointee, tcx, seen, stop_at_ref, adt_handler)
        }
        ty::Tuple(tys) => tys
            .iter()
            .all(|ty| is_opsem_inhabited_recursor(ty, tcx, seen, stop_at_ref, adt_handler)),
        ty::Array(elem, len) => {
            len.try_to_target_usize(tcx).unwrap() == 0
                || is_opsem_inhabited_recursor(elem, tcx, seen, stop_at_ref, adt_handler)
        }
        ty::Pat(inner, _pat) => {
            is_opsem_inhabited_recursor(inner, tcx, seen, stop_at_ref, adt_handler)
        }
        ty::Closure(_def, args) => {
            let args = args.as_closure();
            args.upvar_tys()
                .iter()
                .all(|ty| is_opsem_inhabited_recursor(ty, tcx, seen, stop_at_ref, adt_handler))
        }
        ty::Coroutine(_def, args) => {
            let args = args.as_coroutine();
            args.upvar_tys()
                .iter()
                .all(|ty| is_opsem_inhabited_recursor(ty, tcx, seen, stop_at_ref, adt_handler))
        }
        ty::CoroutineClosure(_def, args) => {
            let args = args.as_coroutine_closure();
            args.upvar_tys()
                .iter()
                .all(|ty| is_opsem_inhabited_recursor(ty, tcx, seen, stop_at_ref, adt_handler))
        }
        ty::UnsafeBinder(base) => {
            let base = tcx.instantiate_bound_regions_with_erased((*base).into());
            is_opsem_inhabited_recursor(base, tcx, seen, stop_at_ref, adt_handler)
        }
        ty::Adt(..) => {
            // ADTs need a special handler to avoid infinite recursion. That handler is meant to
            // call back into the recursor. Ideally it'd just call `is_opsem_inhabited_recursor` but
            // then it would have to pass itself as the adt_handler argument which is not possible
            // in Rust... so we provide the handler with a callback that it can use to continue the
            // recursion with the same `adt_handler`.
            adt_handler(ty, seen, &|ty, seen, stop_at_ref| {
                is_opsem_inhabited_recursor(ty, tcx, seen, stop_at_ref, adt_handler)
            })
        }

        ty::Error(_)
        | ty::Infer(..)
        | ty::Placeholder(..)
        | ty::Bound(..)
        | ty::Param(..)
        | ty::Alias(..)
        | ty::CoroutineWitness(..) => {
            bug!("non-normalized type in `is_opsem_uninhabited`: `{ty}`")
        }
    }
}

fn is_opsem_inhabited_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    env: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
) -> bool {
    let (ty, typing_env) = (env.value, env.typing_env);
    assert_matches!(
        ty.kind(),
        ty::Adt(..),
        "the query should only be invoked by `Ty::is_opsem_inhabited`"
    );

    is_opsem_inhabited_recursor(
        ty,
        tcx,
        &mut FxHashSet::<DefId>::default(),
        /* stop_at_ref */ false,
        &|ty, seen, rec| {
            let ty::Adt(adt_def, adt_args) = *ty.kind() else {
                unreachable! {}
            };
            if adt_def.is_union() {
                // Unions are always inhabited.
                return true;
            }

            let new_adt = seen.insert(adt_def.did());
            // If we have seen this ADT before, stop at the next reference to avoid infinite
            // recursion. We can't stop here since we have to ensure that "layout inhabited"
            // implies "opsem inhabited".
            let stop_at_ref = !new_adt;

            // We are inhabited if in some variant all fields are inhabited.
            let inhabited = adt_def.variants().iter().any(|variant| {
                variant.fields.iter().all(|field| {
                    let ty = field.ty(tcx, adt_args);
                    let ty = tcx.normalize_erasing_regions(typing_env, ty);
                    rec(ty, seen, stop_at_ref)
                })
            });

            // Remove the type again so that we allow it to appear on other branches.
            if new_adt {
                seen.remove(&adt_def.did());
            }

            inhabited
        },
    )
}
