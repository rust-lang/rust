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
    Canonical, CanonicalTyVarKind, CanonicalVarInfo, CanonicalVarKind, Canonicalized,
    SmallCanonicalVarValues,
};
use infer::InferCtxt;
use std::sync::atomic::Ordering;
use ty::fold::{TypeFoldable, TypeFolder};
use ty::subst::Kind;
use ty::{self, CanonicalVar, Lift, List, Ty, TyCtxt, TypeFlags};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use smallvec::SmallVec;

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
        var_values: &mut SmallCanonicalVarValues<'tcx>
    ) -> Canonicalized<'gcx, V>
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
            CanonicalizeRegionMode {
                static_region: true,
                other_free_regions: true,
            },
            var_values,
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
    ) -> Canonicalized<'gcx, V>
    where
        V: TypeFoldable<'tcx> + Lift<'gcx>,
    {
        let mut var_values = SmallVec::new();
        Canonicalizer::canonicalize(
            value,
            Some(self),
            self.tcx,
            CanonicalizeRegionMode {
                static_region: false,
                other_free_regions: false,
            },
            &mut var_values
        )
    }

    /// A hacky variant of `canonicalize_query` that does not
    /// canonicalize `'static`.  Unfortunately, the existing leak
    /// check treaks `'static` differently in some cases (see also
    /// #33684), so if we are performing an operation that may need to
    /// prove "leak-check" related things, we leave `'static`
    /// alone.
    ///
    /// FIXME(#48536) -- once we have universes, we can remove this and just use
    /// `canonicalize_query`.
    pub fn canonicalize_hr_query_hack<V>(
        &self,
        value: &V,
        var_values: &mut SmallCanonicalVarValues<'tcx>
    ) -> Canonicalized<'gcx, V>
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
            CanonicalizeRegionMode {
                static_region: false,
                other_free_regions: true,
            },
            var_values
        )
    }
}

/// If this flag is true, then all free regions will be replaced with
/// a canonical var. This is used to make queries as generic as
/// possible. For example, the query `F: Foo<'static>` would be
/// canonicalized to `F: Foo<'0>`.
struct CanonicalizeRegionMode {
    static_region: bool,
    other_free_regions: bool,
}

impl CanonicalizeRegionMode {
    fn any(&self) -> bool {
        self.static_region || self.other_free_regions
    }
}

struct Canonicalizer<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: Option<&'cx InferCtxt<'cx, 'gcx, 'tcx>>,
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    variables: SmallVec<[CanonicalVarInfo; 8]>,
    var_values: &'cx mut SmallCanonicalVarValues<'tcx>,
    // Note that indices is only used once `var_values` is big enough to be
    // heap-allocated.
    indices: FxHashMap<Kind<'tcx>, CanonicalVar>,
    canonicalize_region_mode: CanonicalizeRegionMode,
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

            ty::ReStatic => {
                if self.canonicalize_region_mode.static_region {
                    let info = CanonicalVarInfo {
                        kind: CanonicalVarKind::Region,
                    };
                    let cvar = self.canonical_var(info, r.into());
                    self.tcx().mk_region(ty::ReCanonical(cvar))
                } else {
                    r
                }
            }

            ty::ReEarlyBound(..)
            | ty::ReFree(_)
            | ty::ReScope(_)
            | ty::ReSkolemized(..)
            | ty::ReEmpty
            | ty::ReErased => {
                if self.canonicalize_region_mode.other_free_regions {
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
            ty::Infer(ty::TyVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::General, t),

            ty::Infer(ty::IntVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::Int, t),

            ty::Infer(ty::FloatVar(_)) => self.canonicalize_ty_var(CanonicalTyVarKind::Float, t),

            ty::Infer(ty::FreshTy(_))
            | ty::Infer(ty::FreshIntTy(_))
            | ty::Infer(ty::FreshFloatTy(_)) => {
                bug!("encountered a fresh type during canonicalization")
            }

            ty::Infer(ty::CanonicalTy(_)) => {
                bug!("encountered a canonical type during canonicalization")
            }

            ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Adt(..)
            | ty::Str
            | ty::Error
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Dynamic(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Projection(..)
            | ty::Foreign(..)
            | ty::Param(..)
            | ty::Opaque(..) => {
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
        canonicalize_region_mode: CanonicalizeRegionMode,
        var_values: &'cx mut SmallCanonicalVarValues<'tcx>
    ) -> Canonicalized<'gcx, V>
    where
        V: TypeFoldable<'tcx> + Lift<'gcx>,
    {
        debug_assert!(
            !value.has_type_flags(TypeFlags::HAS_CANONICAL_VARS),
            "canonicalizing a canonical value: {:?}",
            value,
        );

        let needs_canonical_flags = if canonicalize_region_mode.any() {
            TypeFlags::HAS_FREE_REGIONS | TypeFlags::KEEP_IN_LOCAL_TCX
        } else {
            TypeFlags::KEEP_IN_LOCAL_TCX
        };

        let gcx = tcx.global_tcx();

        // Fast path: nothing that needs to be canonicalized.
        if !value.has_type_flags(needs_canonical_flags) {
            let out_value = gcx.lift(value).unwrap();
            let canon_value = Canonical {
                variables: List::empty(),
                value: out_value,
            };
            return canon_value;
        }

        let mut canonicalizer = Canonicalizer {
            infcx,
            tcx,
            canonicalize_region_mode,
            needs_canonical_flags,
            variables: SmallVec::new(),
            var_values,
            indices: FxHashMap::default(),
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

        let canonical_variables = tcx.intern_canonical_var_infos(&canonicalizer.variables);

        Canonical {
            variables: canonical_variables,
            value: out_value,
        }
    }

    /// Creates a canonical variable replacing `kind` from the input,
    /// or returns an existing variable if `kind` has already been
    /// seen. `kind` is expected to be an unbound variable (or
    /// potentially a free region).
    fn canonical_var(&mut self, info: CanonicalVarInfo, kind: Kind<'tcx>) -> CanonicalVar {
        let Canonicalizer {
            variables,
            var_values,
            indices,
            ..
        } = self;

        // This code is hot. `variables` and `var_values` are usually small
        // (fewer than 8 elements ~95% of the time). They are SmallVec's to
        // avoid allocations in those cases. We also don't use `indices` to
        // determine if a kind has been seen before until the limit of 8 has
        // been exceeded, to also avoid allocations for `indices`.
        if !var_values.spilled() {
            // `var_values` is stack-allocated. `indices` isn't used yet. Do a
            // direct linear search of `var_values`.
            if let Some(idx) = var_values.iter().position(|&k| k == kind) {
                // `kind` is already present in `var_values`.
                CanonicalVar::new(idx)
            } else {
                // `kind` isn't present in `var_values`. Append it. Likewise
                // for `info` and `variables`.
                variables.push(info);
                var_values.push(kind);
                assert_eq!(variables.len(), var_values.len());

                // If `var_values` has become big enough to be heap-allocated,
                // fill up `indices` to facilitate subsequent lookups.
                if var_values.spilled() {
                    assert!(indices.is_empty());
                    *indices =
                        var_values.iter()
                            .enumerate()
                            .map(|(i, &kind)| (kind, CanonicalVar::new(i)))
                            .collect();
                }
                // The cv is the index of the appended element.
                CanonicalVar::new(var_values.len() - 1)
            }
        } else {
            // `var_values` is large. Do a hashmap search via `indices`.
            *indices
                .entry(kind)
                .or_insert_with(|| {
                    variables.push(info);
                    var_values.push(kind);
                    assert_eq!(variables.len(), var_values.len());
                    CanonicalVar::new(variables.len() - 1)
                })
        }
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
