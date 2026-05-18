//! Type inference for patterns.

use std::{
    cmp,
    collections::hash_map::Entry::{Occupied, Vacant},
    iter,
};

use hir_def::{
    AdtId, LocalFieldId, VariantId,
    expr_store::path::Path,
    hir::{
        BindingAnnotation, BindingId, Expr, ExprId, ExprOrPatId, Literal, Pat, PatId,
        RecordFieldPat,
    },
    resolver::ValueNs,
    signatures::VariantFields,
};
use rustc_ast_ir::Mutability;
use rustc_hash::FxHashMap;
use rustc_type_ir::{
    TypeVisitableExt as _,
    inherent::{IntoKind as _, Ty as _},
};
use span::Edition;
use tracing::{debug, instrument, trace};

use crate::{
    BindingMode, InferenceDiagnostic, Span,
    infer::{
        AllowTwoPhase, ByRef, Expectation, InferenceContext, PatAdjust, PatAdjustment,
        expr::ExprIsRead,
    },
    next_solver::{
        Const, TraitRef, Ty, TyKind, Tys,
        infer::{
            InferOk,
            traits::{Obligation, ObligationCause},
        },
    },
    utils::EnumerateAndAdjustIterator,
};

impl ByRef {
    #[must_use]
    fn cap_ref_mutability(mut self, mutbl: Mutability) -> Self {
        if let ByRef::Yes(old_mutbl) = &mut self {
            *old_mutbl = cmp::min(*old_mutbl, mutbl);
        }
        self
    }
}

impl BindingMode {
    fn from_annotation(annotation: BindingAnnotation) -> BindingMode {
        match annotation {
            BindingAnnotation::Unannotated => BindingMode(ByRef::No, Mutability::Not),
            BindingAnnotation::Mutable => BindingMode(ByRef::No, Mutability::Mut),
            BindingAnnotation::Ref => BindingMode(ByRef::Yes(Mutability::Not), Mutability::Not),
            BindingAnnotation::RefMut => BindingMode(ByRef::Yes(Mutability::Mut), Mutability::Not),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PatOrigin {
    LetExpr,
    LetStmt { has_else: bool },
    Param,
    MatchArm,
    DestructuringAssignment,
}

impl PatOrigin {
    fn default_binding_modes(self) -> bool {
        self != PatOrigin::DestructuringAssignment
    }
}

#[derive(Copy, Clone)]
struct PatInfo {
    binding_mode: ByRef,
    max_ref_mutbl: MutblCap,
    pat_origin: PatOrigin,
}

/// Mode for adjusting the expected type and binding mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdjustMode {
    /// Peel off all immediate reference types. If the `deref_patterns` feature is enabled, this
    /// also peels smart pointer ADTs.
    Peel { kind: PeelKind },
    /// Pass on the input binding mode and expected type.
    Pass,
}

/// Restrictions on what types to peel when adjusting the expected type and binding mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PeelKind {
    /// Only peel reference types. This is used for explicit `deref!(_)` patterns, which dereference
    /// any number of `&`/`&mut` references, plus a single smart pointer.
    ExplicitDerefPat,
    /// Implicitly peel references, and if `deref_patterns` is enabled, smart pointer ADTs.
    Implicit {
        /// The ADT the pattern is a constructor for, if applicable, so that we don't peel it. See
        /// [`ResolvedPat`] for more information.
        until_adt: Option<AdtId>,
        /// The number of references at the head of the pattern's type, so we can leave that many
        /// untouched. This is `1` for string literals, and `0` for most patterns.
        pat_ref_layers: usize,
    },
}

impl AdjustMode {
    const fn peel_until_adt(opt_adt_def: Option<AdtId>) -> AdjustMode {
        AdjustMode::Peel { kind: PeelKind::Implicit { until_adt: opt_adt_def, pat_ref_layers: 0 } }
    }
    const fn peel_all() -> AdjustMode {
        AdjustMode::peel_until_adt(None)
    }
}

/// `ref mut` bindings (explicit or match-ergonomics) are not allowed behind an `&` reference.
/// Normally, the borrow checker enforces this, but for (currently experimental) match ergonomics,
/// we track this when typing patterns for two purposes:
///
/// - For RFC 3627's Rule 3, when this would prevent us from binding with `ref mut`, we limit the
///   default binding mode to be by shared `ref` when it would otherwise be `ref mut`.
///
/// - For RFC 3627's Rule 5, we allow `&` patterns to match against `&mut` references, treating them
///   as if they were shared references. Since the scrutinee is mutable in this case, the borrow
///   checker won't catch if we bind with `ref mut`, so we need to throw an error ourselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MutblCap {
    /// Mutability restricted to immutable.
    Not,

    /// Mutability restricted to immutable, but only because of the pattern
    /// (not the scrutinee type).
    ///
    /// The contained span, if present, points to an `&` pattern
    /// that is the reason for the restriction,
    /// and which will be reported in a diagnostic.
    WeaklyNot,

    /// No restriction on mutability
    Mut,
}

impl MutblCap {
    #[must_use]
    fn cap_to_weakly_not(self) -> Self {
        match self {
            MutblCap::Not => MutblCap::Not,
            _ => MutblCap::WeaklyNot,
        }
    }

    #[must_use]
    fn as_mutbl(self) -> Mutability {
        match self {
            MutblCap::Not | MutblCap::WeaklyNot => Mutability::Not,
            MutblCap::Mut => Mutability::Mut,
        }
    }
}

/// Variations on RFC 3627's Rule 4: when do reference patterns match against inherited references?
///
/// "Inherited reference" designates the `&`/`&mut` types that arise from using match ergonomics, i.e.
/// from matching a reference type with a non-reference pattern. E.g. when `Some(x)` matches on
/// `&mut Option<&T>`, `x` gets type `&mut &T` and the outer `&mut` is considered "inherited".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InheritedRefMatchRule {
    /// Reference patterns consume only the inherited reference if possible, regardless of whether
    /// the underlying type being matched against is a reference type. If there is no inherited
    /// reference, a reference will be consumed from the underlying type.
    EatOuter,
    /// Reference patterns consume only a reference from the underlying type if possible. If the
    /// underlying type is not a reference type, the inherited reference will be consumed.
    EatInner,
    /// When the underlying type is a reference type, reference patterns consume both layers of
    /// reference, i.e. they both reset the binding mode and consume the reference type.
    EatBoth {
        /// If `true`, an inherited reference will be considered when determining whether a reference
        /// pattern matches a given type:
        /// - If the underlying type is not a reference, a reference pattern may eat the inherited reference;
        /// - If the underlying type is a reference, a reference pattern matches if it can eat either one
        ///   of the underlying and inherited references. E.g. a `&mut` pattern is allowed if either the
        ///   underlying type is `&mut` or the inherited reference is `&mut`.
        ///
        /// If `false`, a reference pattern is only matched against the underlying type.
        /// This is `false` for stable Rust and `true` for both the `ref_pat_eat_one_layer_2024` and
        /// `ref_pat_eat_one_layer_2024_structural` feature gates.
        consider_inherited_ref: bool,
    },
}

/// When checking patterns containing paths, we need to know the path's resolution to determine
/// whether to apply match ergonomics and implicitly dereference the scrutinee. For instance, when
/// the `deref_patterns` feature is enabled and we're matching against a scrutinee of type
/// `Cow<'a, Option<u8>>`, we insert an implicit dereference to allow the pattern `Some(_)` to type,
/// but we must not dereference it when checking the pattern `Cow::Borrowed(_)`.
///
/// `ResolvedPat` contains the information from resolution needed to determine match ergonomics
/// adjustments, and to finish checking the pattern once we know its adjusted type.
#[derive(Clone, Copy, Debug)]
struct ResolvedPat<'db> {
    /// The type of the pattern, to be checked against the type of the scrutinee after peeling. This
    /// is also used to avoid peeling the scrutinee's constructors (see the `Cow` example above).
    ty: Ty<'db>,
    kind: ResolvedPatKind,
}

#[derive(Clone, Copy, Debug)]
enum ResolvedPatKind {
    Path { res: ValueNs },
    Struct { variant: VariantId },
    TupleStruct { variant: VariantId },
}

impl<'db> ResolvedPat<'db> {
    fn adjust_mode(&self) -> AdjustMode {
        if let ResolvedPatKind::Path { res, .. } = self.kind
            && matches!(res, ValueNs::ConstId(_))
        {
            // These constants can be of a reference type, e.g. `const X: &u8 = &0;`.
            // Peeling the reference types too early will cause type checking failures.
            // Although it would be possible to *also* peel the types of the constants too.
            AdjustMode::Pass
        } else {
            // The remaining possible resolutions for path, struct, and tuple struct patterns are
            // ADT constructors. As such, we may peel references freely, but we must not peel the
            // ADT itself from the scrutinee if it's a smart pointer.
            AdjustMode::peel_until_adt(self.ty.as_adt().map(|(adt, _)| adt))
        }
    }
}

impl<'a, 'db> InferenceContext<'a, 'db> {
    /// Experimental pattern feature: after matching against a shared reference, do we limit the
    /// default binding mode in subpatterns to be `ref` when it would otherwise be `ref mut`?
    /// This corresponds to Rule 3 of RFC 3627.
    fn downgrade_mut_inside_shared(&self) -> bool {
        // NB: RFC 3627 proposes stabilizing Rule 3 in all editions. If we adopt the same behavior
        // across all editions, this may be removed.
        self.features.ref_pat_eat_one_layer_2024_structural
    }

    /// Experimental pattern feature: when do reference patterns match against inherited references?
    /// This corresponds to variations on Rule 4 of RFC 3627.
    fn ref_pat_matches_inherited_ref(&self, edition: Edition) -> InheritedRefMatchRule {
        // NB: The particular rule used here is likely to differ across editions, so calls to this
        // may need to become edition checks after match ergonomics stabilize.
        if edition.at_least_2024() {
            if self.features.ref_pat_eat_one_layer_2024 {
                InheritedRefMatchRule::EatOuter
            } else if self.features.ref_pat_eat_one_layer_2024_structural {
                InheritedRefMatchRule::EatInner
            } else {
                // Currently, matching against an inherited ref on edition 2024 is an error.
                // Use `EatBoth` as a fallback to be similar to stable Rust.
                InheritedRefMatchRule::EatBoth { consider_inherited_ref: false }
            }
        } else {
            InheritedRefMatchRule::EatBoth {
                consider_inherited_ref: self.features.ref_pat_eat_one_layer_2024
                    || self.features.ref_pat_eat_one_layer_2024_structural,
            }
        }
    }

    /// Experimental pattern feature: do `&` patterns match against `&mut` references, treating them
    /// as if they were shared references? This corresponds to Rule 5 of RFC 3627.
    fn ref_pat_matches_mut_ref(&self) -> bool {
        // NB: RFC 3627 proposes stabilizing Rule 5 in all editions. If we adopt the same behavior
        // across all editions, this may be removed.
        self.features.ref_pat_eat_one_layer_2024
            || self.features.ref_pat_eat_one_layer_2024_structural
    }

    /// Type check the given top level pattern against the `expected` type.
    ///
    /// If a `Some(span)` is provided and `origin_expr` holds,
    /// then the `span` represents the scrutinee's span.
    /// The scrutinee is found in e.g. `match scrutinee { ... }` and `let pat = scrutinee;`.
    ///
    /// Otherwise, `Some(span)` represents the span of a type expression
    /// which originated the `expected` type.
    pub(super) fn infer_top_pat(&mut self, pat: PatId, expected: Ty<'db>, pat_origin: PatOrigin) {
        let pat_info =
            PatInfo { binding_mode: ByRef::No, max_ref_mutbl: MutblCap::Mut, pat_origin };
        self.infer_pat(pat, expected, pat_info);
    }

    /// Type check the given `pat` against the `expected` type
    /// with the provided `binding_mode` (default binding mode).
    ///
    /// Outside of this module, `check_pat_top` should always be used.
    /// Conversely, inside this module, `check_pat_top` should never be used.
    #[instrument(level = "debug", skip(self, pat_info))]
    fn infer_pat(&mut self, pat_id: PatId, expected: Ty<'db>, pat_info: PatInfo) {
        // For patterns containing paths, we need the path's resolution to determine whether to
        // implicitly dereference the scrutinee before matching.
        let pat = &self.store[pat_id];
        let opt_path_res = match pat {
            Pat::Path(path) => Some(self.resolve_pat_path(pat_id, path)),
            Pat::Record { path, .. } => Some(self.resolve_record_pat(pat_id, path)),
            Pat::TupleStruct { path, .. } => Some(self.resolve_tuple_struct_pat(pat_id, path)),
            _ => None,
        };
        let adjust_mode = self.calc_adjust_mode(pat_id, pat, opt_path_res);
        let ty = self.infer_pat_inner(pat_id, opt_path_res, adjust_mode, expected, pat_info);
        let ty = self.insert_type_vars_shallow(ty);
        self.write_pat_ty(pat_id, ty);

        // If we implicitly inserted overloaded dereferences before matching check the pattern to
        // see if the dereferenced types need `DerefMut` bounds.
        if let Some(derefed_tys) = self.result.pat_adjustment(pat_id)
            && derefed_tys.iter().any(|adjust| adjust.kind == PatAdjust::OverloadedDeref)
        {
            let infer_ok = self.register_deref_mut_bounds_if_needed(
                pat_id,
                pat_id,
                derefed_tys.iter().filter_map(|adjust| match adjust.kind {
                    PatAdjust::OverloadedDeref => Some(adjust.source.as_ref()),
                    PatAdjust::BuiltinDeref => None,
                }),
            );
            self.table.register_infer_ok(infer_ok);
        }

        // (note_1): In most of the cases where (note_1) is referenced
        // (literals and constants being the exception), we relate types
        // using strict equality, even though subtyping would be sufficient.
        // There are a few reasons for this, some of which are fairly subtle
        // and which cost me (nmatsakis) an hour or two debugging to remember,
        // so I thought I'd write them down this time.
        //
        // 1. There is no loss of expressiveness here, though it does
        // cause some inconvenience. What we are saying is that the type
        // of `x` becomes *exactly* what is expected. This can cause unnecessary
        // errors in some cases, such as this one:
        //
        // ```
        // fn foo<'x>(x: &'x i32) {
        //    let a = 1;
        //    let mut z = x;
        //    z = &a;
        // }
        // ```
        //
        // The reason we might get an error is that `z` might be
        // assigned a type like `&'x i32`, and then we would have
        // a problem when we try to assign `&a` to `z`, because
        // the lifetime of `&a` (i.e., the enclosing block) is
        // shorter than `'x`.
        //
        // HOWEVER, this code works fine. The reason is that the
        // expected type here is whatever type the user wrote, not
        // the initializer's type. In this case the user wrote
        // nothing, so we are going to create a type variable `Z`.
        // Then we will assign the type of the initializer (`&'x i32`)
        // as a subtype of `Z`: `&'x i32 <: Z`. And hence we
        // will instantiate `Z` as a type `&'0 i32` where `'0` is
        // a fresh region variable, with the constraint that `'x : '0`.
        // So basically we're all set.
        //
        // Note that there are two tests to check that this remains true
        // (`regions-reassign-{match,let}-bound-pointer.rs`).
        //
        // 2. An outdated issue related to the old HIR borrowck. See the test
        // `regions-relate-bound-regions-on-closures-to-inference-variables.rs`,
    }

    // Helper to avoid resolving the same path pattern several times.
    fn infer_pat_inner(
        &mut self,
        pat: PatId,
        opt_path_res: Option<Result<ResolvedPat<'db>, ()>>,
        adjust_mode: AdjustMode,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        #[cfg(debug_assertions)]
        if matches!(pat_info.binding_mode, ByRef::Yes(Mutability::Mut))
            && pat_info.max_ref_mutbl != MutblCap::Mut
            && self.downgrade_mut_inside_shared()
        {
            panic!("Pattern mutability cap violated!");
        }

        // Resolve type if needed.
        let expected = if let AdjustMode::Peel { .. } = adjust_mode
            && pat_info.pat_origin.default_binding_modes()
        {
            self.table.try_structurally_resolve_type(pat.into(), expected)
        } else {
            expected
        };

        match self.store[pat] {
            // Peel off a `&` or `&mut`from the scrutinee type. See the examples in
            // `tests/ui/rfcs/rfc-2005-default-binding-mode`.
            _ if let AdjustMode::Peel { kind: peel_kind } = adjust_mode
                && pat_info.pat_origin.default_binding_modes()
                && let TyKind::Ref(_, inner_ty, inner_mutability) = expected.kind()
                && self.should_peel_ref(peel_kind, expected) =>
            {
                debug!("inspecting {:?}", expected);

                debug!("current discriminant is Ref, inserting implicit deref");
                // Preserve the reference type. We'll need it later during THIR lowering.
                self.result.pat_adjustments.entry(pat).or_default().push(PatAdjustment {
                    kind: PatAdjust::BuiltinDeref,
                    source: expected.store(),
                });

                // Use the old pat info to keep `current_depth` to its old value.
                let new_pat_info = self.adjust_pat_info(inner_mutability, pat_info);

                // Recurse with the new expected type.
                self.infer_pat_inner(pat, opt_path_res, adjust_mode, inner_ty, new_pat_info)
            }
            // If `deref_patterns` is enabled, peel a smart pointer from the scrutinee type. See the
            // examples in `tests/ui/pattern/deref_patterns/`.
            _ if self.features.deref_patterns
                && let AdjustMode::Peel { kind: peel_kind } = adjust_mode
                && pat_info.pat_origin.default_binding_modes()
                && self.should_peel_smart_pointer(peel_kind, expected) =>
            {
                debug!("scrutinee ty {expected:?} is a smart pointer, inserting pin deref");

                // The scrutinee is a smart pointer; implicitly dereference it. This adds a
                // requirement that `expected: DerefPure`.
                let inner_ty = self.deref_pat_target(pat, expected);
                // Once we've checked `pat`, we'll add a `DerefMut` bound if it contains any
                // `ref mut` bindings. See `Self::register_deref_mut_bounds_if_needed`.

                self.check_deref_pattern(
                    pat,
                    opt_path_res,
                    adjust_mode,
                    expected,
                    inner_ty,
                    PatAdjust::OverloadedDeref,
                    pat_info,
                )
            }
            Pat::Missing => self.types.types.error,
            Pat::Wild | Pat::Rest => expected,
            // We allow any type here; we ensure that the type is uninhabited during match checking.
            // Pat::Never => expected,
            Pat::Path(_) => {
                let ty = match opt_path_res.unwrap() {
                    Ok(ref pr) => self.infer_pat_path(pat, pr, expected),
                    Err(()) => self.types.types.error,
                };
                self.write_pat_ty(pat, ty);
                ty
            }
            Pat::Lit(expr) => self.infer_lit_pat(expr, expected),
            Pat::Range { start: lhs, end: rhs, .. } => {
                self.infer_range_pat(pat, lhs, rhs, expected)
            }
            Pat::Bind { id: var_id, subpat } => {
                self.infer_bind_pat(pat, var_id, subpat, expected, pat_info)
            }
            Pat::TupleStruct { args: ref subpats, ellipsis: ddpos, .. } => match opt_path_res
                .unwrap()
            {
                Ok(ResolvedPat { ty, kind: ResolvedPatKind::TupleStruct { variant } }) => self
                    .infer_tuple_struct_pat(pat, subpats, ddpos, ty, variant, expected, pat_info),
                Err(()) => {
                    let ty_err = self.types.types.error;
                    for &subpat in subpats {
                        self.infer_pat(subpat, ty_err, pat_info);
                    }
                    ty_err
                }
                Ok(pr) => panic!("tuple struct pattern resolved to {pr:?}"),
            },
            Pat::Record { args: ref fields, ellipsis: has_rest_pat, .. } => {
                match opt_path_res.unwrap() {
                    Ok(ResolvedPat { ty, kind: ResolvedPatKind::Struct { variant } }) => self
                        .infer_record_pat(
                            pat,
                            fields,
                            has_rest_pat,
                            ty,
                            variant,
                            expected,
                            pat_info,
                        ),
                    Err(()) => {
                        let ty_err = self.types.types.error;
                        for field in fields {
                            self.infer_pat(field.pat, ty_err, pat_info);
                        }
                        ty_err
                    }
                    Ok(pr) => panic!("struct pattern resolved to {pr:?}"),
                }
            }
            // Pat::Guard(pat, cond) => {
            //     self.infer_pat(pat, expected, pat_info);
            //     self.check_expr_has_type_or_error(cond, self.tcx.types.bool, |_| {});
            //     expected
            // }
            Pat::Or(ref pats) => {
                for &pat in pats {
                    self.infer_pat(pat, expected, pat_info);
                }
                expected
            }
            Pat::Tuple { args: ref elements, ellipsis: ddpos } => {
                self.infer_tuple_pat(pat, elements, ddpos, expected, pat_info)
            }
            Pat::Box { inner } => self.infer_box_pat(pat, inner, expected, pat_info),
            Pat::Deref { inner } => self.infer_deref_pat(pat, inner, expected, pat_info),
            // Pat::Deref(inner) => self.infer_deref_pat(pat.span, inner, expected, pat_info),
            Pat::Ref { pat: inner, mutability: mutbl } => self.infer_ref_pat(
                pat,
                inner,
                if mutbl.is_mut() { Mutability::Mut } else { Mutability::Not },
                expected,
                pat_info,
            ),
            Pat::Slice { prefix: ref before, slice, suffix: ref after } => {
                self.infer_slice_pat(pat, before, slice, after, expected, pat_info)
            }
            Pat::Expr(expr) => self.infer_destructuring_assignment_expr(expr, expected),
            Pat::ConstBlock(expr) => {
                self.infer_expr(expr, &Expectation::has_type(expected), ExprIsRead::Yes)
            }
        }
    }

    fn adjust_pat_info(&self, inner_mutability: Mutability, pat_info: PatInfo) -> PatInfo {
        let mut binding_mode = match pat_info.binding_mode {
            // If default binding mode is by value, make it `ref`, `ref mut`, `ref pin const`
            // or `ref pin mut` (depending on whether we observe `&`, `&mut`, `&pin const` or
            // `&pin mut`).
            ByRef::No => ByRef::Yes(inner_mutability),
            ByRef::Yes(mutability) => {
                let mutability = match mutability {
                    // When `ref mut`, stay a `ref mut` (on `&mut`) or downgrade to `ref` (on `&`).
                    Mutability::Mut => inner_mutability,
                    // Once a `ref`, always a `ref`.
                    // This is because a `& &mut` cannot mutate the underlying value.
                    Mutability::Not => Mutability::Not,
                };
                ByRef::Yes(mutability)
            }
        };

        let PatInfo { mut max_ref_mutbl, .. } = pat_info;
        if self.downgrade_mut_inside_shared() {
            binding_mode = binding_mode.cap_ref_mutability(max_ref_mutbl.as_mutbl());
        }
        match binding_mode {
            ByRef::Yes(Mutability::Not) => max_ref_mutbl = MutblCap::Not,
            _ => {}
        }
        debug!("default binding mode is now {:?}", binding_mode);
        PatInfo { binding_mode, max_ref_mutbl, ..pat_info }
    }

    fn check_deref_pattern(
        &mut self,
        pat: PatId,
        opt_path_res: Option<Result<ResolvedPat<'db>, ()>>,
        adjust_mode: AdjustMode,
        expected: Ty<'db>,
        mut inner_ty: Ty<'db>,
        pat_adjust_kind: PatAdjust,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        debug_assert!(
            !matches!(pat_adjust_kind, PatAdjust::BuiltinDeref),
            "unexpected deref pattern for builtin reference type {expected:?}",
        );

        let pat_adjustments = self.result.pat_adjustments.entry(pat).or_default();
        // We may reach the recursion limit if a user matches on a type `T` satisfying
        // `T: Deref<Target = T>`; error gracefully in this case.
        // FIXME(deref_patterns): If `deref_patterns` stabilizes, it may make sense to move
        // this check out of this branch. Alternatively, this loop could be implemented with
        // autoderef and this check removed. For now though, don't break code compiling on
        // stable with lots of `&`s and a low recursion limit, if anyone's done that.
        if pat_adjustments.len() < self.resolver.top_level_def_map().recursion_limit() as usize {
            // Preserve the smart pointer type for THIR lowering and closure upvar analysis.
            pat_adjustments.push(PatAdjustment { kind: pat_adjust_kind, source: expected.store() });
        } else {
            // FIXME: Emit an error.
            inner_ty = self.types.types.error;
        }

        // Recurse, using the old pat info to keep `current_depth` to its old value.
        // Peeling smart pointers does not update the default binding mode.
        self.infer_pat_inner(pat, opt_path_res, adjust_mode, inner_ty, pat_info)
    }

    /// How should the binding mode and expected type be adjusted?
    ///
    /// When the pattern contains a path, `opt_path_res` must be `Some(path_res)`.
    fn calc_adjust_mode(
        &mut self,
        pat_id: PatId,
        pat: &Pat,
        opt_path_res: Option<Result<ResolvedPat<'db>, ()>>,
    ) -> AdjustMode {
        match pat {
            // Type checking these product-like types successfully always require
            // that the expected type be of those types and not reference types.
            Pat::Tuple { .. } | Pat::Range { .. } | Pat::Slice { .. } => AdjustMode::peel_all(),
            // When checking an explicit deref pattern, only peel reference types.
            // FIXME(deref_patterns): If box patterns and deref patterns need to coexist, box
            // patterns may want `PeelKind::Implicit`, stopping on encountering a box.
            Pat::Box { .. } | Pat::Deref { .. } => {
                AdjustMode::Peel { kind: PeelKind::ExplicitDerefPat }
            }
            // A never pattern behaves somewhat like a literal or unit variant.
            // Pat::Never => AdjustMode::peel_all(),
            // For patterns with paths, how we peel the scrutinee depends on the path's resolution.
            Pat::Record { .. }
            | Pat::TupleStruct { .. }
            | Pat::Path(_) => {
                // If there was an error resolving the path, default to peeling everything.
                opt_path_res.unwrap().map_or(AdjustMode::peel_all(), |pr| pr.adjust_mode())
            }

            // String and byte-string literals result in types `&str` and `&[u8]` respectively.
            // All other literals result in non-reference types.
            // As a result, we allow `if let 0 = &&0 {}` but not `if let "foo" = &&"foo" {}` unless
            // `deref_patterns` is enabled.
            &Pat::Lit(expr) | &Pat::ConstBlock(expr) => {
                let lit_ty = self.infer_expr_pat_unadjusted(expr);
                // Call `resolve_vars_if_possible` here for inline const blocks.
                let lit_ty = self.infcx().resolve_vars_if_possible(lit_ty);
                // If `deref_patterns` is enabled, allow `if let "foo" = &&"foo" {}`.
                if self.features.deref_patterns {
                    let mut peeled_ty = lit_ty;
                    let mut pat_ref_layers = 0;
                    while let TyKind::Ref(_, inner_ty, mutbl) =
                        self.table.try_structurally_resolve_type(pat_id.into(), peeled_ty).kind()
                    {
                        // We rely on references at the head of constants being immutable.
                        debug_assert!(mutbl.is_not());
                        pat_ref_layers += 1;
                        peeled_ty = inner_ty;
                    }
                    AdjustMode::Peel {
                        kind: PeelKind::Implicit { until_adt: None, pat_ref_layers },
                    }
                } else {
                    if lit_ty.is_ref() { AdjustMode::Pass } else { AdjustMode::peel_all() }
                }
            }

            // Ref patterns are complicated, we handle them in `check_pat_ref`.
            Pat::Ref { .. }
            // No need to do anything on a missing pattern.
            | Pat::Missing
            // A `_`/`..` pattern works with any expected type, so there's no need to do anything.
            | Pat::Wild | Pat::Rest
            // Bindings also work with whatever the expected type is,
            // and moreover if we peel references off, that will give us the wrong binding type.
            // Also, we can have a subpattern `binding @ pat`.
            // Each side of the `@` should be treated independently (like with OR-patterns).
            | Pat::Bind { .. }
            // `Pat::Expr(_)` inside assignments becomes a binding in rustc, therefore should be
            // the same as `Pat::Bind`.
            | Pat::Expr(_)
            // An OR-pattern just propagates to each individual alternative.
            // This is maximally flexible, allowing e.g., `Some(mut x) | &Some(mut x)`.
            // In that example, `Some(mut x)` results in `Peel` whereas `&Some(mut x)` in `Reset`.
            | Pat::Or(_)
            // Like or-patterns, guard patterns just propagate to their subpatterns.
            /* | Pat::Guard(..) */ => AdjustMode::Pass,
        }
    }

    /// Assuming `expected` is a reference type, determine whether to peel it before matching.
    fn should_peel_ref(&self, peel_kind: PeelKind, mut expected: Ty<'db>) -> bool {
        debug_assert!(expected.is_ref());
        let pat_ref_layers = match peel_kind {
            PeelKind::ExplicitDerefPat => 0,
            PeelKind::Implicit { pat_ref_layers, .. } => pat_ref_layers,
        };

        // Most patterns don't have reference types, so we'll want to peel all references from the
        // scrutinee before matching. To optimize for the common case, return early.
        if pat_ref_layers == 0 {
            return true;
        }
        debug_assert!(
            self.features.deref_patterns,
            "Peeling for patterns with reference types is gated by `deref_patterns`."
        );

        // If the pattern has as many or more layers of reference as the expected type, we can match
        // without peeling more, unless we find a smart pointer or `&mut` that we also need to peel.
        // We don't treat `&` and `&mut` as interchangeable, but by peeling `&mut`s before matching,
        // we can still, e.g., match on a `&mut str` with a string literal pattern. This is because
        // string literal patterns may be used where `str` is expected.
        let mut expected_ref_layers = 0;
        while let TyKind::Ref(_, inner_ty, mutbl) = expected.kind() {
            if mutbl.is_mut() {
                // Mutable references can't be in the final value of constants, thus they can't be
                // at the head of their types, thus we should always peel `&mut`.
                return true;
            }
            expected_ref_layers += 1;
            expected = inner_ty;
        }
        pat_ref_layers < expected_ref_layers || self.should_peel_smart_pointer(peel_kind, expected)
    }

    /// Determine whether `expected` is a smart pointer type that should be peeled before matching.
    fn should_peel_smart_pointer(&self, peel_kind: PeelKind, expected: Ty<'db>) -> bool {
        // Explicit `deref!(_)` patterns match against smart pointers; don't peel in that case.
        if let PeelKind::Implicit { until_adt, .. } = peel_kind
            // For simplicity, only apply overloaded derefs if `expected` is a known ADT.
            // FIXME(deref_patterns): we'll get better diagnostics for users trying to
            // implicitly deref generics if we allow them here, but primitives, tuples, and
            // inference vars definitely should be stopped. Figure out what makes most sense.
            && let TyKind::Adt(scrutinee_adt, _) = expected.kind()
            // Don't peel if the pattern type already matches the scrutinee. E.g., stop here if
            // matching on a `Cow<'a, T>` scrutinee with a `Cow::Owned(_)` pattern.
            && until_adt != Some(scrutinee_adt.def_id())
            // At this point, the pattern isn't able to match `expected` without peeling. Check
            // that it implements `Deref` before assuming it's a smart pointer, to get a normal
            // type error instead of a missing impl error if not. This only checks for `Deref`,
            // not `DerefPure`: we require that too, but we want a trait error if it's missing.
            && let Some(deref_trait) = self.lang_items.Deref
            && self.infcx().type_implements_trait(deref_trait, [expected], self.table.param_env).may_apply()
        {
            true
        } else {
            false
        }
    }

    fn infer_expr_pat_unadjusted(&mut self, expr: ExprId) -> Ty<'db> {
        self.infer_expr_no_expect(expr, ExprIsRead::Yes)
    }

    fn infer_lit_pat(&mut self, expr: ExprId, expected: Ty<'db>) -> Ty<'db> {
        let literal = match &self.store[expr] {
            Expr::Literal(literal) => literal,
            _ => panic!("expected a literal"),
        };

        // We've already computed the type above (when checking for a non-ref pat),
        // so avoid computing it again.
        let ty = self.expr_ty(expr);

        // Byte string patterns behave the same way as array patterns
        // They can denote both statically and dynamically-sized byte arrays.
        // Additionally, when `deref_patterns` is enabled, byte string literal patterns may have
        // types `[u8]` or `[u8; N]`, in order to type, e.g., `deref!(b"..."): Vec<u8>`.
        let mut pat_ty = ty;
        if matches!(literal, Literal::ByteString(_)) {
            let expected = self.structurally_resolve_type(expr.into(), expected);
            match expected.kind() {
                // Allow `b"...": &[u8]`
                TyKind::Ref(_, inner_ty, _)
                    if self
                        .table
                        .try_structurally_resolve_type(expr.into(), inner_ty)
                        .is_slice() =>
                {
                    trace!(?expr, "polymorphic byte string lit");
                    pat_ty = self.types.types.static_u8_slice;
                }
                // Allow `b"...": [u8; 3]` for `deref_patterns`
                TyKind::Array(..) if self.features.deref_patterns => {
                    pat_ty = match ty.kind() {
                        TyKind::Ref(_, inner_ty, _) => inner_ty,
                        _ => panic!("found byte string literal with non-ref type {ty:?}"),
                    }
                }
                // Allow `b"...": [u8]` for `deref_patterns`
                TyKind::Slice(..) if self.features.deref_patterns => {
                    pat_ty = self.types.types.u8_slice;
                }
                // Otherwise, `b"...": &[u8; 3]`
                _ => {}
            }
        }

        // When `deref_patterns` is enabled, in order to allow `deref!("..."): String`, we allow
        // string literal patterns to have type `str`. This is accounted for when lowering to MIR.
        if self.features.deref_patterns
            && matches!(literal, Literal::String(_))
            && self.table.try_structurally_resolve_type(expr.into(), expected).is_str()
        {
            pat_ty = self.types.types.str;
        }

        // Somewhat surprising: in this case, the subtyping relation goes the
        // opposite way as the other cases. Actually what we really want is not
        // a subtyping relation at all but rather that there exists a LUB
        // (so that they can be compared). However, in practice, constants are
        // always scalars or strings. For scalars subtyping is irrelevant,
        // and for strings `ty` is type is `&'static str`, so if we say that
        //
        //     &'static str <: expected
        //
        // then that's equivalent to there existing a LUB.
        _ = self.demand_suptype(expr.into(), expected, pat_ty);

        pat_ty
    }

    fn infer_range_pat(
        &mut self,
        pat: PatId,
        lhs_expr: Option<ExprId>,
        rhs_expr: Option<ExprId>,
        expected: Ty<'db>,
    ) -> Ty<'db> {
        let mut calc_side = |opt_expr: Option<ExprId>| match opt_expr {
            None => None,
            Some(expr) => {
                let ty = self.infer_expr_pat_unadjusted(expr);
                // Check that the end-point is possibly of numeric or char type.
                // The early check here is not for correctness, but rather better
                // diagnostics (e.g. when `&str` is being matched, `expected` will
                // be peeled to `str` while ty here is still `&str`, if we don't
                // err early here, a rather confusing unification error will be
                // emitted instead).
                let ty = self.table.try_structurally_resolve_type(expr.into(), ty);
                let fail =
                    !(ty.is_numeric() || ty.is_char() || ty.is_ty_var() || ty.references_error());
                Some((fail, ty, expr))
            }
        };
        let mut lhs = calc_side(lhs_expr);
        let mut rhs = calc_side(rhs_expr);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            // There exists a side that didn't meet our criteria that the end-point
            // be of a numeric or char type, as checked in `calc_side` above.
            // FIXME: Emit an error.
            return self.types.types.error;
        }

        // Unify each side with `expected`.
        // Subtyping doesn't matter here, as the value is some kind of scalar.
        let mut demand_eqtype = |x: &mut _| {
            if let Some((_, x_ty, x_expr)) = *x {
                _ = self.demand_eqtype(ExprOrPatId::from(x_expr), expected, x_ty);
            }
        };
        demand_eqtype(&mut lhs);
        demand_eqtype(&mut rhs);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            return self.types.types.error;
        }

        // Find the unified type and check if it's of numeric or char type again.
        // This check is needed if both sides are inference variables.
        // We require types to be resolved here so that we emit inference failure
        // rather than "_ is not a char or numeric".
        let ty = self.structurally_resolve_type(
            lhs_expr.or(rhs_expr).map(ExprOrPatId::ExprId).unwrap_or(pat.into()),
            expected,
        );
        if !(ty.is_numeric() || ty.is_char() || ty.references_error()) {
            // FIXME: Emit an error.
            return self.types.types.error;
        }
        ty
    }

    fn infer_bind_pat(
        &mut self,
        pat: PatId,
        var_id: BindingId,
        sub: Option<PatId>,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        let PatInfo { binding_mode: def_br, .. } = pat_info;
        let binding_data = &self.store[var_id];

        // Determine the binding mode...
        let user_bind_annot = BindingMode::from_annotation(binding_data.mode);
        let bm = match user_bind_annot {
            BindingMode(ByRef::No, Mutability::Mut) if let ByRef::Yes(_) = def_br => {
                // Only mention the experimental `mut_ref` feature if if we're in edition 2024 and
                // using other experimental matching features compatible with it.
                if self.edition.at_least_2024()
                    && (self.features.ref_pat_eat_one_layer_2024
                        || self.features.ref_pat_eat_one_layer_2024_structural)
                {
                    if !self.features.mut_ref {
                        // FIXME: Emit an error: binding cannot be both mutable and by-reference.
                    }

                    BindingMode(def_br, Mutability::Mut)
                } else {
                    // `mut` resets the binding mode on edition <= 2021
                    BindingMode(ByRef::No, Mutability::Mut)
                }
            }
            BindingMode(ByRef::No, mutbl) => BindingMode(def_br, mutbl),
            BindingMode(ByRef::Yes(_), _) => user_bind_annot,
        };

        if matches!(bm.0, ByRef::Yes(Mutability::Mut))
            && let MutblCap::WeaklyNot = pat_info.max_ref_mutbl
        {
            // FIXME: Emit an error: cannot borrow as mutable inside an `&` pattern.
        }

        // ...and store it in a side table:
        self.result.binding_modes.insert(pat, bm);

        debug!("check_pat_ident: pat.hir_id={:?} bm={:?}", pat, bm);

        let local_ty = match bm.0 {
            ByRef::Yes(mutbl) => {
                // If the binding is like `ref x | ref mut x`,
                // then `x` is assigned a value of type `&M T` where M is the
                // mutability and T is the expected type.
                //
                // Under pin ergonomics, if the binding is like `ref pin const|mut x`,
                // then `x` is assigned a value of type `&pin M T` where M is the
                // mutability and T is the expected type.
                //
                // `x` is assigned a value of type `&M T`, hence `&M T <: typeof(x)`
                // is required. However, we use equality, which is stronger.
                // See (note_1) for an explanation.
                self.new_ref_ty(pat.into(), mutbl, expected)
            }
            // Otherwise, the type of x is the expected type `T`.
            ByRef::No => expected, // As above, `T <: typeof(x)` is required, but we use equality, see (note_1).
        };

        // We have a concrete type for the local, so we do not need to taint it and hide follow up errors *using* the local.
        if let Some(existing_local_ty) = self.result.type_of_binding.get(var_id) {
            // If there are multiple arms, make sure they all agree on
            // what the type of the binding `x` ought to be.
            _ = self.demand_eqtype(pat.into(), existing_local_ty.as_ref(), local_ty);
        } else {
            self.write_binding_ty(var_id, local_ty);
        }

        if let Some(p) = sub {
            self.infer_pat(p, expected, pat_info);
        }

        local_ty
    }

    fn check_dereferenceable(&self, expected: Ty<'db>, inner: PatId) -> Result<(), ()> {
        if let Pat::Bind { .. } = self.store[inner]
            && let Some(pointee_ty) = self.shallow_resolve(expected).builtin_deref(true)
            && let TyKind::Dynamic(..) = pointee_ty.kind()
        {
            // This is "x = dyn SomeTrait" being reduced from
            // "let &x = &dyn SomeTrait" or "let box x = Box<dyn SomeTrait>", an error.
            // FIXME: Emit an error. rustc emits this message:
            const _CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ: &str = "\
This error indicates that a pointer to a trait type cannot be implicitly dereferenced by a \
pattern. Every trait defines a type, but because the size of trait implementors isn't fixed, \
this type has no compile-time size. Therefore, all accesses to trait types must be through \
pointers. If you encounter this error you should try to avoid dereferencing the pointer.

You can read more about trait objects in the Trait Objects section of the Reference: \
https://doc.rust-lang.org/reference/types.html#trait-objects";
        }
        Ok(())
    }

    fn resolve_record_pat(&mut self, pat: PatId, path: &Path) -> Result<ResolvedPat<'db>, ()> {
        // Resolve the path and check the definition for errors.
        let (pat_ty, Some(variant)) = self.resolve_variant(pat.into(), path, false) else {
            return Err(());
        };
        self.write_variant_resolution(pat.into(), variant);
        Ok(ResolvedPat { ty: pat_ty, kind: ResolvedPatKind::Struct { variant } })
    }

    fn infer_record_pat(
        &mut self,
        pat: PatId,
        fields: &[RecordFieldPat],
        has_rest_pat: bool,
        pat_ty: Ty<'db>,
        variant: VariantId,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        // Type-check the path.
        let _ = self.demand_eqtype(pat.into(), expected, pat_ty);

        // Type-check subpatterns.
        self.check_record_pat_fields(pat_ty, pat, variant, fields, has_rest_pat, pat_info);
        pat_ty
    }

    fn resolve_pat_path(&mut self, pat: PatId, path: &Path) -> Result<ResolvedPat<'db>, ()> {
        let (res, pat_ty) = self.infer_path(path, pat.into()).ok_or(())?;
        match res {
            ValueNs::FunctionId(_)
            | ValueNs::GenericParam(_)
            | ValueNs::ImplSelf(_)
            | ValueNs::LocalBinding(_)
            | ValueNs::StaticId(_) => {
                // FIXME: Emit an error.
                return Err(());
            }
            ValueNs::ConstId(_) | ValueNs::EnumVariantId(_) | ValueNs::StructId(_) => {} // OK
        }

        Ok(ResolvedPat { ty: pat_ty, kind: ResolvedPatKind::Path { res } })
    }

    fn infer_pat_path(
        &mut self,
        pat: PatId,
        resolved: &ResolvedPat<'db>,
        expected: Ty<'db>,
    ) -> Ty<'db> {
        _ = self.demand_suptype(pat.into(), expected, resolved.ty);
        resolved.ty
    }

    fn resolve_tuple_struct_pat(
        &mut self,
        pat: PatId,
        path: &Path,
    ) -> Result<ResolvedPat<'db>, ()> {
        // Resolve the path and check the definition for errors.
        let (pat_ty, Some(variant)) = self.resolve_variant(pat.into(), path, true) else {
            return Err(());
        };
        self.write_variant_resolution(pat.into(), variant);
        Ok(ResolvedPat { ty: pat_ty, kind: ResolvedPatKind::TupleStruct { variant } })
    }

    fn infer_tuple_struct_pat(
        &mut self,
        pat: PatId,
        subpats: &[PatId],
        ddpos: Option<u32>,
        pat_ty: Ty<'db>,
        variant: VariantId,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        let interner = self.interner();

        // Type-check the tuple struct pattern against the expected type.
        let had_err = self.demand_eqtype(pat.into(), expected, pat_ty);

        let variant_fields = variant.fields(self.db);
        let variant_field_tys = self.db.field_types(variant);
        let TyKind::Adt(_, args) = pat_ty.kind() else {
            panic!("unexpected pattern type {:?}", pat_ty);
        };
        // Type-check subpatterns.
        if subpats.len() == variant_fields.len()
            || subpats.len() < variant_fields.len() && ddpos.is_some()
        {
            for (i, &subpat) in subpats.iter().enumerate_and_adjust(variant_fields.len(), ddpos) {
                let field_id = LocalFieldId::from_raw(la_arena::RawIdx::from_u32(i as u32));
                let field_ty =
                    variant_field_tys[field_id].get().instantiate(interner, args).skip_norm_wip();
                self.infer_pat(subpat, field_ty, pat_info);
            }
            if let Err(()) = had_err {
                for &pat in subpats {
                    self.infer_pat(pat, self.types.types.error, pat_info);
                }
                return self.types.types.error;
            }
        } else {
            self.push_diagnostic(InferenceDiagnostic::MismatchedTupleStructPatArgCount {
                pat,
                expected: variant_fields.len(),
                found: subpats.len(),
            });

            for (i, &pat) in subpats.iter().enumerate() {
                let field_id = LocalFieldId::from_raw(la_arena::RawIdx::from_u32(i as u32));
                let expected = match variant_field_tys.get(field_id) {
                    Some(field_ty) => field_ty.get().instantiate(interner, args).skip_norm_wip(),
                    None => self.types.types.error,
                };
                self.infer_pat(pat, expected, pat_info);
            }
        }
        pat_ty
    }

    fn infer_tuple_pat(
        &mut self,
        pat: PatId,
        elements: &[PatId],
        ddpos: Option<u32>,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        let interner = self.interner();
        let mut expected_len = elements.len();
        if ddpos.is_some() {
            // Require known type only when `..` is present.
            if let TyKind::Tuple(tys) = self.structurally_resolve_type(pat.into(), expected).kind()
            {
                expected_len = tys.len();
            }
        }
        let max_len = cmp::max(expected_len, elements.len());

        let element_tys_iter = (0..max_len).map(|i| {
            self.table.next_ty_var(elements.get(i).copied().map(Span::PatId).unwrap_or(Span::Dummy))
        });
        let element_tys = Tys::new_from_iter(interner, element_tys_iter);
        let pat_ty = Ty::new(interner, TyKind::Tuple(element_tys));
        if self.demand_eqtype(pat.into(), expected, pat_ty).is_err() {
            let expected = if let TyKind::Tuple(tys) =
                self.table.try_structurally_resolve_type(Span::Dummy, expected).kind()
            {
                for (expected_var, found) in iter::zip(element_tys, tys) {
                    // Constrain the infer var so that the type mismatch error message, which contains it,
                    // will be better.
                    _ = self.demand_eqtype(pat.into(), expected_var, found);
                }
                tys
            } else {
                self.types.empty.tys
            };
            let expected = expected.iter().chain(iter::repeat(self.types.types.error));
            Ty::new_tup_from_iter(
                interner,
                iter::zip(expected, elements).map(|(expected, &elem)| {
                    self.infer_pat(elem, expected, pat_info);
                    self.result.type_of_pat_with_adjust(elem)
                }),
            )
        } else {
            for (i, &elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.infer_pat(elem, element_tys[i], pat_info);
            }
            pat_ty
        }
    }

    fn check_record_pat_fields(
        &mut self,
        adt_ty: Ty<'db>,
        _pat: PatId,
        variant: VariantId,
        fields: &[RecordFieldPat],
        has_rest_pat: bool,
        pat_info: PatInfo,
    ) {
        let interner = self.interner();

        let TyKind::Adt(_, args) = adt_ty.kind() else {
            panic!("struct pattern is not an ADT");
        };

        // Index the struct fields' types.
        let variant_fields = variant.fields(self.db);
        let field_map = variant_fields
            .fields()
            .iter()
            .map(|(i, field)| (field.name.clone(), i))
            .collect::<FxHashMap<_, _>>();
        let variant_field_tys = self.db.field_types(variant);
        let variant_fields_vis = VariantFields::field_visibilities(self.db, variant);

        // Keep track of which fields have already appeared in the pattern.
        let mut used_fields = FxHashMap::default();

        let mut inexistent_fields = vec![];
        // Typecheck each field.
        for (field_idx, field) in fields.iter().enumerate() {
            match used_fields.entry(field.name.clone()) {
                Occupied(_occupied) => {
                    self.push_diagnostic(InferenceDiagnostic::DuplicateField {
                        field: field.pat.into(),
                        variant,
                    });
                }
                Vacant(vacant) => {
                    vacant.insert(field_idx);
                }
            };
            let field_idx = field_map.get(&field.name).copied();
            let field_ty = match field_idx {
                Some(field_idx) => {
                    if !self.resolver.is_visible(self.db, variant_fields_vis[field_idx]) {
                        self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                            field: field.pat.into(),
                            private: Some(field_idx),
                            variant,
                        });
                    }

                    variant_field_tys[field_idx].get().instantiate(interner, args).skip_norm_wip()
                }
                None => {
                    inexistent_fields.push(field);
                    self.types.types.error
                }
            };

            self.infer_pat(field.pat, field_ty, pat_info);
        }

        let unmentioned_fields = variant_fields
            .fields()
            .iter()
            .filter(|(_, field)| !used_fields.contains_key(&field.name))
            .collect::<Vec<_>>();

        for inexistent_field in inexistent_fields {
            self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                field: inexistent_field.pat.into(),
                private: None,
                variant,
            });
        }

        // Require `..` if struct has non_exhaustive attribute.
        let non_exhaustive = self.has_applicable_non_exhaustive(variant.into());
        if non_exhaustive && !has_rest_pat {
            // FIXME: Emit an error.
        }

        // Report an error if an incorrect number of fields was specified.
        if matches!(variant, VariantId::UnionId(_)) {
            if fields.len() != 1 {
                // FIXME: Emit an error, unions can't have more than one field.
            }
            if has_rest_pat {
                // FIXME: Emit an error, unions can't have a rest pat.
            }
        } else if !unmentioned_fields.is_empty() && !has_rest_pat {
            // FIXME: Emit an error.
        }
    }

    fn infer_box_pat(
        &mut self,
        pat: PatId,
        inner: PatId,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        let interner = self.interner();
        let (box_ty, inner_ty) = self
            .check_dereferenceable(expected, inner)
            .map(|()| {
                // Here, `demand::subtype` is good enough, but I don't
                // think any errors can be introduced by using `demand::eqtype`.
                let inner_ty = self.table.next_ty_var(inner.into());
                let box_ty = Ty::new_box(interner, inner_ty);
                _ = self.demand_eqtype(pat.into(), expected, box_ty);
                (box_ty, inner_ty)
            })
            .unwrap_or_else(|()| {
                let err = self.types.types.error;
                (err, err)
            });
        self.infer_pat(inner, inner_ty, pat_info);
        box_ty
    }

    fn infer_deref_pat(
        &mut self,
        pat: PatId,
        inner: PatId,
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        let target_ty = self.deref_pat_target(pat, expected);
        self.infer_pat(inner, target_ty, pat_info);
        let infer_ok = self.register_deref_mut_bounds_if_needed(pat, inner, [expected]);
        self.table.register_infer_ok(infer_ok);
        expected
    }

    fn deref_pat_target(&mut self, pat: PatId, source_ty: Ty<'db>) -> Ty<'db> {
        let (Some(deref_pure), Some(deref_target)) =
            (self.lang_items.DerefPure, self.lang_items.DerefTarget)
        else {
            return self.types.types.error;
        };
        // Register a `DerefPure` bound, which is required by all `deref!()` pats.
        let interner = self.interner();
        self.table.register_bound(source_ty, deref_pure, ObligationCause::new(pat));
        // The expected type for the deref pat's inner pattern is `<expected as Deref>::Target`.
        let target_ty = Ty::new_projection(interner, deref_target.into(), [source_ty]);
        self.table.try_structurally_resolve_type(pat.into(), target_ty)
    }

    /// Check if the interior of a deref pattern (either explicit or implicit) has any `ref mut`
    /// bindings, which would require `DerefMut` to be emitted in MIR building instead of just
    /// `Deref`. We do this *after* checking the inner pattern, since we want to make sure to
    /// account for `ref mut` binding modes inherited from implicitly dereferencing `&mut` refs.
    fn register_deref_mut_bounds_if_needed(
        &self,
        pat: PatId,
        inner: PatId,
        derefed_tys: impl IntoIterator<Item = Ty<'db>>,
    ) -> InferOk<'db, ()> {
        let mut infer_ok = InferOk { value: (), obligations: Vec::new() };
        if self.pat_has_ref_mut_binding(inner) {
            let Some(deref_mut) = self.lang_items.DerefMut else { return infer_ok };
            let interner = self.interner();
            for mutably_derefed_ty in derefed_tys {
                infer_ok.obligations.push(Obligation::new(
                    interner,
                    ObligationCause::new(pat),
                    self.table.param_env,
                    TraitRef::new(interner, deref_mut.into(), [mutably_derefed_ty]),
                ));
            }
        }
        infer_ok
    }

    /// Does the pattern recursively contain a `ref mut` binding in it?
    ///
    /// This is used to determined whether a `deref` pattern should emit a `Deref`
    /// or `DerefMut` call for its pattern scrutinee.
    ///
    /// This is computed from the typeck results since we want to make
    /// sure to apply any match-ergonomics adjustments, which we cannot
    /// determine from the HIR alone.
    pub(super) fn pat_has_ref_mut_binding(&self, pat: PatId) -> bool {
        let mut has_ref_mut = false;
        self.store.walk_pats(pat, &mut |pat| {
            if let Some(BindingMode(ByRef::Yes(Mutability::Mut), _)) =
                self.result.binding_modes.get(pat)
            {
                has_ref_mut = true;
            }
        });
        has_ref_mut
    }

    // Precondition: Pat is Ref(inner)
    fn infer_ref_pat(
        &mut self,
        pat: PatId,
        inner: PatId,
        pat_mutbl: Mutability,
        mut expected: Ty<'db>,
        mut pat_info: PatInfo,
    ) -> Ty<'db> {
        let ref_pat_matches_mut_ref = self.ref_pat_matches_mut_ref();
        if ref_pat_matches_mut_ref && pat_mutbl == Mutability::Not {
            // If `&` patterns can match against mutable reference types (RFC 3627, Rule 5), we need
            // to prevent subpatterns from binding with `ref mut`. Subpatterns of a shared reference
            // pattern should have read-only access to the scrutinee, and the borrow checker won't
            // catch it in this case.
            pat_info.max_ref_mutbl = pat_info.max_ref_mutbl.cap_to_weakly_not();
        }

        expected = self.table.try_structurally_resolve_type(pat.into(), expected);
        // Determine whether we're consuming an inherited reference and resetting the default
        // binding mode, based on edition and enabled experimental features.
        if let ByRef::Yes(inh_mut) = pat_info.binding_mode {
            match self.ref_pat_matches_inherited_ref(self.edition) {
                InheritedRefMatchRule::EatOuter => {
                    // ref pattern attempts to consume inherited reference
                    if pat_mutbl > inh_mut {
                        // Tried to match inherited `ref` with `&mut`
                        // NB: This assumes that `&` patterns can match against mutable references
                        // (RFC 3627, Rule 5). If we implement a pattern typing ruleset with Rule 4E
                        // but not Rule 5, we'll need to check that here.
                        debug_assert!(ref_pat_matches_mut_ref);
                        // FIXME: Emit an error.
                    }

                    pat_info.binding_mode = ByRef::No;
                    self.result.skipped_ref_pats.insert(pat);
                    self.infer_pat(inner, expected, pat_info);
                    return expected;
                }
                InheritedRefMatchRule::EatInner => {
                    if let TyKind::Ref(_, _, r_mutbl) = expected.kind()
                        && pat_mutbl <= r_mutbl
                    {
                        // Match against the reference type; don't consume the inherited ref.
                        // NB: The check for compatible pattern and ref type mutability assumes that
                        // `&` patterns can match against mutable references (RFC 3627, Rule 5). If
                        // we implement a pattern typing ruleset with Rule 4 (including the fallback
                        // to matching the inherited ref when the inner ref can't match) but not
                        // Rule 5, we'll need to check that here.
                        debug_assert!(ref_pat_matches_mut_ref);
                        // NB: For RFC 3627's Rule 3, we limit the default binding mode's ref
                        // mutability to `pat_info.max_ref_mutbl`. If we implement a pattern typing
                        // ruleset with Rule 4 but not Rule 3, we'll need to check that here.
                        debug_assert!(self.downgrade_mut_inside_shared());
                        let mutbl_cap = cmp::min(r_mutbl, pat_info.max_ref_mutbl.as_mutbl());
                        pat_info.binding_mode = pat_info.binding_mode.cap_ref_mutability(mutbl_cap);
                    } else {
                        // The reference pattern can't match against the expected type, so try
                        // matching against the inherited ref instead.
                        if pat_mutbl > inh_mut {
                            // We can't match an inherited shared reference with `&mut`.
                            // NB: This assumes that `&` patterns can match against mutable
                            // references (RFC 3627, Rule 5). If we implement a pattern typing
                            // ruleset with Rule 4 but not Rule 5, we'll need to check that here.
                            // FIXME(ref_pat_eat_one_layer_2024_structural): If we already tried
                            // matching the real reference, the error message should explain that
                            // falling back to the inherited reference didn't work. This should be
                            // the same error as the old-Edition version below.
                            debug_assert!(ref_pat_matches_mut_ref);
                            // FIXME: Emit an error.
                        }

                        pat_info.binding_mode = ByRef::No;
                        self.result.skipped_ref_pats.insert(pat);
                        self.infer_pat(inner, expected, pat_info);
                        return expected;
                    }
                }
                InheritedRefMatchRule::EatBoth { consider_inherited_ref: true } => {
                    // Reset binding mode on old editions
                    pat_info.binding_mode = ByRef::No;

                    if let TyKind::Ref(_, inner_ty, _) = expected.kind() {
                        // Consume both the inherited and inner references.
                        if pat_mutbl.is_mut() && inh_mut.is_mut() {
                            // As a special case, a `&mut` reference pattern will be able to match
                            // against a reference type of any mutability if the inherited ref is
                            // mutable. Since this allows us to match against a shared reference
                            // type, we refer to this as "falling back" to matching the inherited
                            // reference, though we consume the real reference as well. We handle
                            // this here to avoid adding this case to the common logic below.
                            self.infer_pat(inner, inner_ty, pat_info);
                            return expected;
                        } else {
                            // Otherwise, use the common logic below for matching the inner
                            // reference type.
                            // FIXME(ref_pat_eat_one_layer_2024_structural): If this results in a
                            // mutability mismatch, the error message should explain that falling
                            // back to the inherited reference didn't work. This should be the same
                            // error as the Edition 2024 version above.
                        }
                    } else {
                        // The expected type isn't a reference type, so only match against the
                        // inherited reference.
                        if pat_mutbl > inh_mut {
                            // We can't match a lone inherited shared reference with `&mut`.
                            // FIXME: Emit an error.
                        }

                        self.result.skipped_ref_pats.insert(pat);
                        self.infer_pat(inner, expected, pat_info);
                        return expected;
                    }
                }
                InheritedRefMatchRule::EatBoth { consider_inherited_ref: false } => {
                    // Reset binding mode on stable Rust. This will be a type error below if
                    // `expected` is not a reference type.
                    pat_info.binding_mode = ByRef::No;
                }
            }
        }

        let (ref_ty, inner_ty) = match self.check_dereferenceable(expected, inner) {
            Ok(()) => {
                // `demand::subtype` would be good enough, but using `eqtype` turns
                // out to be equally general. See (note_1) for details.

                // Take region, inner-type from expected type if we can,
                // to avoid creating needless variables. This also helps with
                // the bad interactions of the given hack detailed in (note_1).
                debug!("check_pat_ref: expected={:?}", expected);
                match expected.as_reference() {
                    Some((r_ty, _, r_mutbl))
                        if ((ref_pat_matches_mut_ref && r_mutbl >= pat_mutbl)
                            || r_mutbl == pat_mutbl) =>
                    {
                        if r_mutbl == Mutability::Not {
                            pat_info.max_ref_mutbl = MutblCap::Not;
                        }

                        (expected, r_ty)
                    }
                    _ => {
                        let inner_ty = self.table.next_ty_var(inner.into());
                        let ref_ty = self.new_ref_ty(inner.into(), pat_mutbl, inner_ty);
                        debug!("check_pat_ref: demanding {:?} = {:?}", expected, ref_ty);
                        _ = self.demand_eqtype(pat.into(), expected, ref_ty);

                        (ref_ty, inner_ty)
                    }
                }
            }
            Err(()) => {
                let err = self.types.types.error;
                (err, err)
            }
        };

        self.infer_pat(inner, inner_ty, pat_info);
        ref_ty
    }

    /// Create a reference or pinned reference type with a fresh region variable.
    fn new_ref_ty(&self, span: Span, mutbl: Mutability, ty: Ty<'db>) -> Ty<'db> {
        let region = self.table.next_region_var(span);
        Ty::new_ref(self.interner(), region, ty, mutbl)
    }

    fn try_resolve_slice_ty_to_array_ty(
        &self,
        before: &[PatId],
        slice: Option<PatId>,
        pat: PatId,
    ) -> Option<Ty<'db>> {
        if slice.is_some() {
            return None;
        }

        let interner = self.interner();
        let len = before.len();
        let inner_ty = self.table.next_ty_var(pat.into());

        Some(Ty::new_array(interner, inner_ty, len.try_into().unwrap()))
    }

    /// Used to determines whether we can infer the expected type in the slice pattern to be of type array.
    /// This is only possible if we're in an irrefutable pattern. If we were to allow this in refutable
    /// patterns we wouldn't e.g. report ambiguity in the following situation:
    ///
    /// ```ignore(rust)
    /// struct Zeroes;
    ///    const ARR: [usize; 2] = [0; 2];
    ///    const ARR2: [usize; 2] = [2; 2];
    ///
    ///    impl Into<&'static [usize; 2]> for Zeroes {
    ///        fn into(self) -> &'static [usize; 2] {
    ///            &ARR
    ///        }
    ///    }
    ///
    ///    impl Into<&'static [usize]> for Zeroes {
    ///        fn into(self) -> &'static [usize] {
    ///            &ARR2
    ///        }
    ///    }
    ///
    ///    fn main() {
    ///        let &[a, b]: &[usize] = Zeroes.into() else {
    ///           ..
    ///        };
    ///    }
    /// ```
    ///
    /// If we're in an irrefutable pattern we prefer the array impl candidate given that
    /// the slice impl candidate would be rejected anyway (if no ambiguity existed).
    fn pat_is_irrefutable(&self, pat_origin: PatOrigin) -> bool {
        match pat_origin {
            PatOrigin::LetExpr | PatOrigin::MatchArm => false,
            PatOrigin::LetStmt { has_else } => !has_else,
            PatOrigin::DestructuringAssignment | PatOrigin::Param => true,
        }
    }

    /// Type check a slice pattern.
    ///
    /// Syntactically, these look like `[pat_0, ..., pat_n]`.
    /// Semantically, we are type checking a pattern with structure:
    /// ```ignore (not-rust)
    /// [before_0, ..., before_n, (slice, after_0, ... after_n)?]
    /// ```
    /// The type of `slice`, if it is present, depends on the `expected` type.
    /// If `slice` is missing, then so is `after_i`.
    /// If `slice` is present, it can still represent 0 elements.
    fn infer_slice_pat(
        &mut self,
        pat: PatId,
        before: &[PatId],
        slice: Option<PatId>,
        after: &[PatId],
        expected: Ty<'db>,
        pat_info: PatInfo,
    ) -> Ty<'db> {
        let expected = self.table.try_structurally_resolve_type(pat.into(), expected);

        // If the pattern is irrefutable and `expected` is an infer ty, we try to equate it
        // to an array if the given pattern allows it. See issue #76342
        if self.pat_is_irrefutable(pat_info.pat_origin)
            && expected.is_ty_var()
            && let Some(resolved_arr_ty) = self.try_resolve_slice_ty_to_array_ty(before, slice, pat)
        {
            debug!(?resolved_arr_ty);
            let _ = self.demand_eqtype(pat.into(), expected, resolved_arr_ty);
        }

        let expected = self.structurally_resolve_type(pat.into(), expected);
        debug!(?expected);

        let (element_ty, opt_slice_ty, inferred) = match expected.kind() {
            // An array, so we might have something like `let [a, b, c] = [0, 1, 2];`.
            TyKind::Array(element_ty, len) => {
                let min = before.len() as u64 + after.len() as u64;
                let (opt_slice_ty, expected) =
                    self.check_array_pat_len(pat, element_ty, expected, slice, len, min.into());
                // `opt_slice_ty.is_none()` => `slice.is_none()`.
                // Note, though, that opt_slice_ty could be `Some(error_ty)`.
                assert!(opt_slice_ty.is_some() || slice.is_none());
                (element_ty, opt_slice_ty, expected)
            }
            TyKind::Slice(element_ty) => (element_ty, Some(expected), expected),
            // The expected type must be an array or slice, but was neither, so error.
            _ => {
                self.push_diagnostic(InferenceDiagnostic::ExpectedArrayOrSlicePat {
                    pat,
                    found: expected.store(),
                });
                let err = self.types.types.error;
                (err, Some(err), err)
            }
        };

        // Type check all the patterns before `slice`.
        for &elt in before {
            self.infer_pat(elt, element_ty, pat_info);
        }
        // Type check the `slice`, if present, against its expected type.
        if let Some(slice) = slice {
            self.infer_pat(slice, opt_slice_ty.unwrap(), pat_info);
        }
        // Type check the elements after `slice`, if present.
        for &elt in after {
            self.infer_pat(elt, element_ty, pat_info);
        }
        inferred
    }

    /// Type check the length of an array pattern.
    ///
    /// Returns both the type of the variable length pattern (or `None`), and the potentially
    /// inferred array type. We only return `None` for the slice type if `slice.is_none()`.
    fn check_array_pat_len(
        &mut self,
        pat: PatId,
        element_ty: Ty<'db>,
        arr_ty: Ty<'db>,
        slice: Option<PatId>,
        len: Const<'db>,
        min_len: u128,
    ) -> (Option<Ty<'db>>, Ty<'db>) {
        let len = crate::consteval::try_const_usize(self.db, len);

        if let Some(len) = len {
            // Now we know the length...
            if slice.is_none() {
                // ...and since there is no variable-length pattern,
                // we require an exact match between the number of elements
                // in the array pattern and as provided by the matched type.
                if min_len == len {
                    return (None, arr_ty);
                }

                self.push_diagnostic(InferenceDiagnostic::MismatchedArrayPatLen {
                    pat,
                    expected: len,
                    found: min_len,
                    has_rest: false,
                });
            } else if let Some(pat_len) = len.checked_sub(min_len) {
                // The variable-length pattern was there,
                // so it has an array type with the remaining elements left as its size...
                return (Some(Ty::new_array(self.interner(), element_ty, pat_len)), arr_ty);
            } else {
                // ...however, in this case, there were no remaining elements.
                // That is, the slice pattern requires more than the array type offers.
                self.push_diagnostic(InferenceDiagnostic::MismatchedArrayPatLen {
                    pat,
                    expected: len,
                    found: min_len,
                    has_rest: true,
                });
            }
        } else if slice.is_none() {
            // We have a pattern with a fixed length,
            // which we can use to infer the length of the array.
            let updated_arr_ty = Ty::new_array(self.interner(), element_ty, min_len);
            _ = self.demand_eqtype(pat.into(), updated_arr_ty, arr_ty);
            return (None, updated_arr_ty);
        } else {
            // We have a variable-length pattern and don't know the array length.
            // This happens if we have e.g.,
            // `let [a, b, ..] = arr` where `arr: [T; N]` where `const N: usize`.
            // FIXME: Emit an error: cannot pattern-match on an array without a fixed length.
        };

        // If we get here, we must have emitted an error.
        (Some(self.types.types.error), arr_ty)
    }

    fn infer_destructuring_assignment_expr(&mut self, expr: ExprId, expected: Ty<'db>) -> Ty<'db> {
        // LHS of assignment doesn't constitute reads.
        let expr_is_read = ExprIsRead::No;
        let lhs_ty = self.infer_expr_inner(expr, &Expectation::has_type(expected), expr_is_read);
        match self.coerce(expr, expected, lhs_ty, AllowTwoPhase::No, expr_is_read) {
            Ok(ty) => ty,
            Err(_) => {
                self.emit_type_mismatch(expr.into(), expected, lhs_ty);
                // `rhs_ty` is returned so no further type mismatches are
                // reported because of this mismatch.
                expected
            }
        }
    }
}
