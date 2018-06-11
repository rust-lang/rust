// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains the "canonicalizer" itself.
//!
//! For an overview of what canonicaliation is and how it fits into
//! rustc, check out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html

use infer::canonical::{
    Canonical, CanonicalTyVarKind, CanonicalVarInfo, CanonicalVarKind, CanonicalVarValues,
    Canonicalized,
};
use infer::InferCtxt;
use std::sync::atomic::Ordering;
use ty::fold::{TypeFoldable, TypeFolder};
use ty::subst::Kind;
use ty::{self, CanonicalVar, Lift, Slice, Ty, TyCtxt, TypeFlags};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::IndexVec;

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
    /// Canonicalizes a query value `V`. When we canonicalize a query,
    /// we not only canonicalize unbound inference variables, but we
    /// *also* replace all free regions whatsoever. So for example a
    /// query like `T: Trait<'static>` would be canonicalized to
    ///
    /// ```text
    /// T: Trait<'?0>
    /// ```
    ///
    /// with a mapping M that maps `'?0` to `'static`.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc guide][c].
    ///
    /// [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html#canonicalizing-the-query
    pub fn canonicalize_query<V>(
        &self,
        value: &V,
    ) -> (Canonicalized<'gcx, V>, CanonicalVarValues<'tcx>)
    where
        V: TypeFoldable<'tcx> + Lift<'gcx>,
    {
        self.tcx
            .sess
            .perf_stats
            .queries_canonicalized
            .fetch_add(1, Ordering::Relaxed);

        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            CanonicalizeAllFreeRegions(true),
        )
    }

    /// Canonicalizes a query *response* `V`. When we canonicalize a
    /// query response, we only canonicalize unbound inference
    /// variables, and we leave other free regions alone. So,
    /// continuing with the example from `canonicalize_query`, if
    /// there was an input query `T: Trait<'static>`, it would have
    /// been canonicalized to
    ///
    /// ```text
    /// T: Trait<'?0>
    /// ```
    ///
    /// with a mapping M that maps `'?0` to `'static`. But if we found that there
    /// exists only one possible impl of `Trait`, and it looks like
    ///
    ///     impl<T> Trait<'static> for T { .. }
    ///
    /// then we would prepare a query result R that (among other
    /// things) includes a mapping to `'?0 := 'static`. When
    /// canonicalizing this query result R, we would leave this
    /// reference to `'static` alone.
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc guide][c].
    ///
    /// [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html#canonicalizing-the-query-result
    pub fn canonicalize_response<V>(
        &self,
        value: &V,
    ) -> (Canonicalized<'gcx, V>, CanonicalVarValues<'tcx>)
    where
        V: TypeFoldable<'tcx> + Lift<'gcx>,
    {
        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            CanonicalizeAllFreeRegions(false),
        )
    }
}

/// If this flag is true, then all free regions will be replaced with
/// a canonical var. This is used to make queries as generic as
/// possible. For example, the query `F: Foo<'static>` would be
/// canonicalized to `F: Foo<'0>`.
struct CanonicalizeAllFreeRegions(pub bool);

struct Canonicalizer<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: Option<&'cx InferCtxt<'cx, 'gcx, 'tcx>>,
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    variables: IndexVec<CanonicalVar, CanonicalVarInfo>,
    indices: FxHashMap<Kind<'tcx>, CanonicalVar>,
    var_values: IndexVec<CanonicalVar, Kind<'tcx>>,
    canonicalize_all_free_regions: CanonicalizeAllFreeRegions,
    needs_canonical_flags: TypeFlags,
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for Canonicalizer<'cx, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(..) => {
                // leave bound regions alone
                r
            }

            ty::ReVar(vid) => {
                let r = self
                    .infcx
                    .unwrap()
                    .borrow_region_constraints()
                    .opportunistic_resolve_var(self.tcx, vid);
                let info = CanonicalVarInfo {
                    kind: CanonicalVarKind::Region,
                };
                debug!(
                    "canonical: region var found with vid {:?}, \
                     opportunistically resolved to {:?}",
                    vid, r
                );
                let cvar = self.canonical_var(info, r.into());
                self.tcx().mk_region(ty::ReCanonical(cvar))
            }

            ty::ReStatic
            | ty::ReEarlyBound(..)
            | ty::ReFree(_)
            | ty::ReScope(_)
            | ty::ReSkolemized(..)
            | ty::ReEmpty
            | ty::ReErased => {
                if self.canonicalize_all_free_regions.0 {
                    let info = CanonicalVarInfo {
                        kind: CanonicalVarKind::Region,
                    };
                    let cvar = self.canonical_var(info, r.into());
                    self.tcx().mk_region(ty::ReCanonical(cvar))
                } else {
                    r
                }
            }

            ty::ReClosureBound(..) | ty::ReCanonical(_) => {
                bug!("canonical region encountered during canonicalization")
            }
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::TyInfer(ty::TyVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::General, t),

            ty::TyInfer(ty::IntVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::Int, t),

            ty::TyInfer(ty::FloatVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::Float, t),

            ty::TyInfer(ty::FreshTy(_))
            | ty::TyInfer(ty::FreshIntTy(_))
            | ty::TyInfer(ty::FreshFloatTy(_)) => {
                bug!("encountered a fresh type during canonicalization")
            }

            ty::TyInfer(ty::CanonicalTy(_)) => {
                bug!("encountered a canonical type during canonicalization")
            }

            ty::TyClosure(..)
            | ty::TyGenerator(..)
            | ty::TyGeneratorWitness(..)
            | ty::TyBool
            | ty::TyChar
            | ty::TyInt(..)
            | ty::TyUint(..)
            | ty::TyFloat(..)
            | ty::TyAdt(..)
            | ty::TyStr
            | ty::TyError
            | ty::TyArray(..)
            | ty::TySlice(..)
            | ty::TyRawPtr(..)
            | ty::TyRef(..)
            | ty::TyFnDef(..)
            | ty::TyFnPtr(_)
            | ty::TyDynamic(..)
            | ty::TyNever
            | ty::TyTuple(..)
            | ty::TyProjection(..)
            | ty::TyForeign(..)
            | ty::TyParam(..)
            | ty::TyAnon(..) => {
                if t.flags.intersects(self.needs_canonical_flags) {
                    t.super_fold_with(self)
                } else {
                    t
                }
            }
        }
    }
}

impl<'cx, 'gcx, 'tcx> Canonicalizer<'cx, 'gcx, 'tcx> {
    /// The main `canonicalize` method, shared impl of
    /// `canonicalize_query` and `canonicalize_response`.
    fn canonicalize<V>(
        value: &V,
        infcx: Option<&'cx InferCtxt<'cx, 'gcx, 'tcx>>,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        canonicalize_all_free_regions: CanonicalizeAllFreeRegions,
    ) -> (Canonicalized<'gcx, V>, CanonicalVarValues<'tcx>)
    where
        V: TypeFoldable<'tcx> + Lift<'gcx>,
    {
        debug_assert!(
            !value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS),
            "canonicalizing a canonical value: {:?}",
            value,
        );

        let needs_canonical_flags = if canonicalize_all_free_regions.0 {
            TypeFlags::HAS_FREE_REGIONS | TypeFlags::KEEP_IN_LOCAL_TCX
        } else {
            TypeFlags::KEEP_IN_LOCAL_TCX
        };

        let gcx = tcx.global_tcx();

        // Fast path: nothing that needs to be canonicalized.
        if !value.has_type_flags(needs_canonical_flags) {
            let out_value = gcx.lift(value).unwrap();
            let canon_value = Canonical {
                variables: Slice::empty(),
                value: out_value,
            };
            let values = CanonicalVarValues {
                var_values: IndexVec::default(),
            };
            return (canon_value, values);
        }

        let mut canonicalizer = Canonicalizer {
            infcx,
            tcx,
            canonicalize_all_free_regions,
            needs_canonical_flags,
            variables: IndexVec::default(),
            indices: FxHashMap::default(),
            var_values: IndexVec::default(),
        };
        let out_value = value.fold_with(&mut canonicalizer);

        // Once we have canonicalized `out_value`, it should not
        // contain anything that ties it to this inference context
        // anymore, so it should live in the global arena.
        let out_value = gcx.lift(&out_value).unwrap_or_else(|| {
            bug!(
                "failed to lift `{:?}`, canonicalized from `{:?}`",
                out_value,
                value
            )
        });

        let canonical_variables = tcx.intern_canonical_var_infos(&canonicalizer.variables.raw);

        let canonical_value = Canonical {
            variables: canonical_variables,
            value: out_value,
        };
        let canonical_var_values = CanonicalVarValues {
            var_values: canonicalizer.var_values,
        };
        (canonical_value, canonical_var_values)
    }

    /// Creates a canonical variable replacing `kind` from the input,
    /// or returns an existing variable if `kind` has already been
    /// seen. `kind` is expected to be an unbound variable (or
    /// potentially a free region).
    fn canonical_var(&mut self, info: CanonicalVarInfo, kind: Kind<'tcx>) -> CanonicalVar {
        let Canonicalizer {
            indices,
            variables,
            var_values,
            ..
        } = self;

        indices
            .entry(kind)
            .or_insert_with(|| {
                let cvar1 = variables.push(info);
                let cvar2 = var_values.push(kind);
                assert_eq!(cvar1, cvar2);
                cvar1
            })
            .clone()
    }

    /// Given a type variable `ty_var` of the given kind, first check
    /// if `ty_var` is bound to anything; if so, canonicalize
    /// *that*. Otherwise, create a new canonical variable for
    /// `ty_var`.
    fn canonicalize_ty_var(&mut self, ty_kind: CanonicalTyVarKind, ty_var: Ty<'tcx>) -> Ty<'tcx> {
        let infcx = self.infcx.expect("encountered ty-var without infcx");
        let bound_to = infcx.shallow_resolve(ty_var);
        if bound_to != ty_var {
            self.fold_ty(bound_to)
        } else {
            let info = CanonicalVarInfo {
                kind: CanonicalVarKind::Ty(ty_kind),
            };
            let cvar = self.canonical_var(info, ty_var.into());
            self.tcx().mk_infer(ty::InferTy::CanonicalTy(cvar))
        }
    }
}
