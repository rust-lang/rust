use std::cmp;
use std::collections::hash_map::Entry::{Occupied, Vacant};

use rustc_abi::FieldIdx;
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, ErrorGuaranteed, MultiSpan, pluralize, struct_span_code_err,
};
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::{
    self as hir, BindingMode, ByRef, ExprKind, HirId, LangItem, Mutability, Pat, PatExpr,
    PatExprKind, PatKind, expr_needs_parens,
};
use rustc_hir_analysis::autoderef::report_autoderef_recursion_limit_error;
use rustc_infer::infer;
use rustc_middle::traits::PatternOriginExpr;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_session::lint::builtin::NON_EXHAUSTIVE_OMITTED_PATTERNS;
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::edition::Edition;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::source_map::Spanned;
use rustc_span::{BytePos, DUMMY_SP, Ident, Span, kw, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{ObligationCause, ObligationCauseCode};
use tracing::{debug, instrument, trace};
use ty::VariantDef;
use ty::adjustment::{PatAdjust, PatAdjustment};

use super::report_unexpected_variant_res;
use crate::expectation::Expectation;
use crate::gather_locals::DeclOrigin;
use crate::{FnCtxt, errors};

const CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ: &str = "\
This error indicates that a pointer to a trait type cannot be implicitly dereferenced by a \
pattern. Every trait defines a type, but because the size of trait implementors isn't fixed, \
this type has no compile-time size. Therefore, all accesses to trait types must be through \
pointers. If you encounter this error you should try to avoid dereferencing the pointer.

You can read more about trait objects in the Trait Objects section of the Reference: \
https://doc.rust-lang.org/reference/types.html#trait-objects";

fn is_number(text: &str) -> bool {
    text.chars().all(|c: char| c.is_digit(10))
}

/// Information about the expected type at the top level of type checking a pattern.
///
/// **NOTE:** This is only for use by diagnostics. Do NOT use for type checking logic!
#[derive(Copy, Clone)]
struct TopInfo<'tcx> {
    /// The `expected` type at the top level of type checking a pattern.
    expected: Ty<'tcx>,
    /// Was the origin of the `span` from a scrutinee expression?
    ///
    /// Otherwise there is no scrutinee and it could be e.g. from the type of a formal parameter.
    origin_expr: Option<&'tcx hir::Expr<'tcx>>,
    /// The span giving rise to the `expected` type, if one could be provided.
    ///
    /// If `origin_expr` is `true`, then this is the span of the scrutinee as in:
    ///
    /// - `match scrutinee { ... }`
    /// - `let _ = scrutinee;`
    ///
    /// This is used to point to add context in type errors.
    /// In the following example, `span` corresponds to the `a + b` expression:
    ///
    /// ```text
    /// error[E0308]: mismatched types
    ///  --> src/main.rs:L:C
    ///   |
    /// L |    let temp: usize = match a + b {
    ///   |                            ----- this expression has type `usize`
    /// L |         Ok(num) => num,
    ///   |         ^^^^^^^ expected `usize`, found enum `std::result::Result`
    ///   |
    ///   = note: expected type `usize`
    ///              found type `std::result::Result<_, _>`
    /// ```
    span: Option<Span>,
    /// The [`HirId`] of the top-level pattern.
    hir_id: HirId,
}

#[derive(Copy, Clone)]
struct PatInfo<'tcx> {
    binding_mode: ByRef,
    max_ref_mutbl: MutblCap,
    top_info: TopInfo<'tcx>,
    decl_origin: Option<DeclOrigin<'tcx>>,

    /// The depth of current pattern
    current_depth: u32,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn pattern_cause(&self, ti: &TopInfo<'tcx>, cause_span: Span) -> ObligationCause<'tcx> {
        // If origin_expr exists, then expected represents the type of origin_expr.
        // If span also exists, then span == origin_expr.span (although it doesn't need to exist).
        // In that case, we can peel away references from both and treat them
        // as the same.
        let origin_expr_info = ti.origin_expr.map(|mut cur_expr| {
            let mut count = 0;

            // cur_ty may have more layers of references than cur_expr.
            // We can only make suggestions about cur_expr, however, so we'll
            // use that as our condition for stopping.
            while let ExprKind::AddrOf(.., inner) = &cur_expr.kind {
                cur_expr = inner;
                count += 1;
            }

            PatternOriginExpr {
                peeled_span: cur_expr.span,
                peeled_count: count,
                peeled_prefix_suggestion_parentheses: expr_needs_parens(cur_expr),
            }
        });

        let code = ObligationCauseCode::Pattern {
            span: ti.span,
            root_ty: ti.expected,
            origin_expr: origin_expr_info,
        };
        self.cause(cause_span, code)
    }

    fn demand_eqtype_pat_diag(
        &'a self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ti: &TopInfo<'tcx>,
    ) -> Result<(), Diag<'a>> {
        self.demand_eqtype_with_origin(&self.pattern_cause(ti, cause_span), expected, actual)
            .map_err(|mut diag| {
                if let Some(expr) = ti.origin_expr {
                    self.suggest_fn_call(&mut diag, expr, expected, |output| {
                        self.can_eq(self.param_env, output, actual)
                    });
                }
                diag
            })
    }

    fn demand_eqtype_pat(
        &self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ti: &TopInfo<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        self.demand_eqtype_pat_diag(cause_span, expected, actual, ti).map_err(|err| err.emit())
    }
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
        until_adt: Option<DefId>,
        /// The number of references at the head of the pattern's type, so we can leave that many
        /// untouched. This is `1` for string literals, and `0` for most patterns.
        pat_ref_layers: usize,
    },
}

impl AdjustMode {
    const fn peel_until_adt(opt_adt_def: Option<DefId>) -> AdjustMode {
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
    WeaklyNot(Option<Span>),

    /// No restriction on mutability
    Mut,
}

impl MutblCap {
    #[must_use]
    fn cap_to_weakly_not(self, span: Option<Span>) -> Self {
        match self {
            MutblCap::Not => MutblCap::Not,
            _ => MutblCap::WeaklyNot(span),
        }
    }

    #[must_use]
    fn as_mutbl(self) -> Mutability {
        match self {
            MutblCap::Not | MutblCap::WeaklyNot(_) => Mutability::Not,
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
        ///    of the underlying and inherited references. E.g. a `&mut` pattern is allowed if either the
        ///    underlying type is `&mut` or the inherited reference is `&mut`.
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
struct ResolvedPat<'tcx> {
    /// The type of the pattern, to be checked against the type of the scrutinee after peeling. This
    /// is also used to avoid peeling the scrutinee's constructors (see the `Cow` example above).
    ty: Ty<'tcx>,
    kind: ResolvedPatKind<'tcx>,
}

#[derive(Clone, Copy, Debug)]
enum ResolvedPatKind<'tcx> {
    Path { res: Res, pat_res: Res, segments: &'tcx [hir::PathSegment<'tcx>] },
    Struct { variant: &'tcx VariantDef },
    TupleStruct { res: Res, variant: &'tcx VariantDef },
}

impl<'tcx> ResolvedPat<'tcx> {
    fn adjust_mode(&self) -> AdjustMode {
        if let ResolvedPatKind::Path { res, .. } = self.kind
            && matches!(res, Res::Def(DefKind::Const | DefKind::AssocConst, _))
        {
            // These constants can be of a reference type, e.g. `const X: &u8 = &0;`.
            // Peeling the reference types too early will cause type checking failures.
            // Although it would be possible to *also* peel the types of the constants too.
            AdjustMode::Pass
        } else {
            // The remaining possible resolutions for path, struct, and tuple struct patterns are
            // ADT constructors. As such, we may peel references freely, but we must not peel the
            // ADT itself from the scrutinee if it's a smart pointer.
            AdjustMode::peel_until_adt(self.ty.ty_adt_def().map(|adt| adt.did()))
        }
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Experimental pattern feature: after matching against a shared reference, do we limit the
    /// default binding mode in subpatterns to be `ref` when it would otherwise be `ref mut`?
    /// This corresponds to Rule 3 of RFC 3627.
    fn downgrade_mut_inside_shared(&self) -> bool {
        // NB: RFC 3627 proposes stabilizing Rule 3 in all editions. If we adopt the same behavior
        // across all editions, this may be removed.
        self.tcx.features().ref_pat_eat_one_layer_2024_structural()
    }

    /// Experimental pattern feature: when do reference patterns match against inherited references?
    /// This corresponds to variations on Rule 4 of RFC 3627.
    fn ref_pat_matches_inherited_ref(&self, edition: Edition) -> InheritedRefMatchRule {
        // NB: The particular rule used here is likely to differ across editions, so calls to this
        // may need to become edition checks after match ergonomics stabilize.
        if edition.at_least_rust_2024() {
            if self.tcx.features().ref_pat_eat_one_layer_2024() {
                InheritedRefMatchRule::EatOuter
            } else if self.tcx.features().ref_pat_eat_one_layer_2024_structural() {
                InheritedRefMatchRule::EatInner
            } else {
                // Currently, matching against an inherited ref on edition 2024 is an error.
                // Use `EatBoth` as a fallback to be similar to stable Rust.
                InheritedRefMatchRule::EatBoth { consider_inherited_ref: false }
            }
        } else {
            InheritedRefMatchRule::EatBoth {
                consider_inherited_ref: self.tcx.features().ref_pat_eat_one_layer_2024()
                    || self.tcx.features().ref_pat_eat_one_layer_2024_structural(),
            }
        }
    }

    /// Experimental pattern feature: do `&` patterns match against `&mut` references, treating them
    /// as if they were shared references? This corresponds to Rule 5 of RFC 3627.
    fn ref_pat_matches_mut_ref(&self) -> bool {
        // NB: RFC 3627 proposes stabilizing Rule 5 in all editions. If we adopt the same behavior
        // across all editions, this may be removed.
        self.tcx.features().ref_pat_eat_one_layer_2024()
            || self.tcx.features().ref_pat_eat_one_layer_2024_structural()
    }

    /// Type check the given top level pattern against the `expected` type.
    ///
    /// If a `Some(span)` is provided and `origin_expr` holds,
    /// then the `span` represents the scrutinee's span.
    /// The scrutinee is found in e.g. `match scrutinee { ... }` and `let pat = scrutinee;`.
    ///
    /// Otherwise, `Some(span)` represents the span of a type expression
    /// which originated the `expected` type.
    pub(crate) fn check_pat_top(
        &self,
        pat: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        span: Option<Span>,
        origin_expr: Option<&'tcx hir::Expr<'tcx>>,
        decl_origin: Option<DeclOrigin<'tcx>>,
    ) {
        let top_info = TopInfo { expected, origin_expr, span, hir_id: pat.hir_id };
        let pat_info = PatInfo {
            binding_mode: ByRef::No,
            max_ref_mutbl: MutblCap::Mut,
            top_info,
            decl_origin,
            current_depth: 0,
        };
        self.check_pat(pat, expected, pat_info);
    }

    /// Type check the given `pat` against the `expected` type
    /// with the provided `binding_mode` (default binding mode).
    ///
    /// Outside of this module, `check_pat_top` should always be used.
    /// Conversely, inside this module, `check_pat_top` should never be used.
    #[instrument(level = "debug", skip(self, pat_info))]
    fn check_pat(&self, pat: &'tcx Pat<'tcx>, expected: Ty<'tcx>, pat_info: PatInfo<'tcx>) {
        // For patterns containing paths, we need the path's resolution to determine whether to
        // implicitly dereference the scrutinee before matching.
        let opt_path_res = match pat.kind {
            PatKind::Expr(PatExpr { kind: PatExprKind::Path(qpath), hir_id, span }) => {
                Some(self.resolve_pat_path(*hir_id, *span, qpath))
            }
            PatKind::Struct(ref qpath, ..) => Some(self.resolve_pat_struct(pat, qpath)),
            PatKind::TupleStruct(ref qpath, ..) => Some(self.resolve_pat_tuple_struct(pat, qpath)),
            _ => None,
        };
        let adjust_mode = self.calc_adjust_mode(pat, opt_path_res);
        let ty = self.check_pat_inner(pat, opt_path_res, adjust_mode, expected, pat_info);
        self.write_ty(pat.hir_id, ty);

        // If we implicitly inserted overloaded dereferences before matching, check the pattern to
        // see if the dereferenced types need `DerefMut` bounds.
        if let Some(derefed_tys) = self.typeck_results.borrow().pat_adjustments().get(pat.hir_id)
            && derefed_tys.iter().any(|adjust| adjust.kind == PatAdjust::OverloadedDeref)
        {
            self.register_deref_mut_bounds_if_needed(
                pat.span,
                pat,
                derefed_tys.iter().filter_map(|adjust| match adjust.kind {
                    PatAdjust::OverloadedDeref => Some(adjust.source),
                    PatAdjust::BuiltinDeref => None,
                }),
            );
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
    fn check_pat_inner(
        &self,
        pat: &'tcx Pat<'tcx>,
        opt_path_res: Option<Result<ResolvedPat<'tcx>, ErrorGuaranteed>>,
        adjust_mode: AdjustMode,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        #[cfg(debug_assertions)]
        if pat_info.binding_mode == ByRef::Yes(Mutability::Mut)
            && pat_info.max_ref_mutbl != MutblCap::Mut
            && self.downgrade_mut_inside_shared()
        {
            span_bug!(pat.span, "Pattern mutability cap violated!");
        }

        // Resolve type if needed.
        let expected = if let AdjustMode::Peel { .. } = adjust_mode
            && pat.default_binding_modes
        {
            self.try_structurally_resolve_type(pat.span, expected)
        } else {
            expected
        };
        let old_pat_info = pat_info;
        let pat_info = PatInfo { current_depth: old_pat_info.current_depth + 1, ..old_pat_info };

        match pat.kind {
            // Peel off a `&` or `&mut` from the scrutinee type. See the examples in
            // `tests/ui/rfcs/rfc-2005-default-binding-mode`.
            _ if let AdjustMode::Peel { kind: peel_kind } = adjust_mode
                && pat.default_binding_modes
                && let ty::Ref(_, inner_ty, inner_mutability) = *expected.kind()
                && self.should_peel_ref(peel_kind, expected) =>
            {
                debug!("inspecting {:?}", expected);

                debug!("current discriminant is Ref, inserting implicit deref");
                // Preserve the reference type. We'll need it later during THIR lowering.
                self.typeck_results
                    .borrow_mut()
                    .pat_adjustments_mut()
                    .entry(pat.hir_id)
                    .or_default()
                    .push(PatAdjustment { kind: PatAdjust::BuiltinDeref, source: expected });

                let mut binding_mode = ByRef::Yes(match pat_info.binding_mode {
                    // If default binding mode is by value, make it `ref` or `ref mut`
                    // (depending on whether we observe `&` or `&mut`).
                    ByRef::No |
                    // When `ref mut`, stay a `ref mut` (on `&mut`) or downgrade to `ref` (on `&`).
                    ByRef::Yes(Mutability::Mut) => inner_mutability,
                    // Once a `ref`, always a `ref`.
                    // This is because a `& &mut` cannot mutate the underlying value.
                    ByRef::Yes(Mutability::Not) => Mutability::Not,
                });

                let mut max_ref_mutbl = pat_info.max_ref_mutbl;
                if self.downgrade_mut_inside_shared() {
                    binding_mode = binding_mode.cap_ref_mutability(max_ref_mutbl.as_mutbl());
                }
                if binding_mode == ByRef::Yes(Mutability::Not) {
                    max_ref_mutbl = MutblCap::Not;
                }
                debug!("default binding mode is now {:?}", binding_mode);

                // Use the old pat info to keep `current_depth` to its old value.
                let new_pat_info = PatInfo { binding_mode, max_ref_mutbl, ..old_pat_info };
                // Recurse with the new expected type.
                self.check_pat_inner(pat, opt_path_res, adjust_mode, inner_ty, new_pat_info)
            }
            // If `deref_patterns` is enabled, peel a smart pointer from the scrutinee type. See the
            // examples in `tests/ui/pattern/deref_patterns/`.
            _ if self.tcx.features().deref_patterns()
                && let AdjustMode::Peel { kind: peel_kind } = adjust_mode
                && pat.default_binding_modes
                && self.should_peel_smart_pointer(peel_kind, expected) =>
            {
                debug!("scrutinee ty {expected:?} is a smart pointer, inserting overloaded deref");
                // The scrutinee is a smart pointer; implicitly dereference it. This adds a
                // requirement that `expected: DerefPure`.
                let mut inner_ty = self.deref_pat_target(pat.span, expected);
                // Once we've checked `pat`, we'll add a `DerefMut` bound if it contains any
                // `ref mut` bindings. See `Self::register_deref_mut_bounds_if_needed`.

                let mut typeck_results = self.typeck_results.borrow_mut();
                let mut pat_adjustments_table = typeck_results.pat_adjustments_mut();
                let pat_adjustments = pat_adjustments_table.entry(pat.hir_id).or_default();
                // We may reach the recursion limit if a user matches on a type `T` satisfying
                // `T: Deref<Target = T>`; error gracefully in this case.
                // FIXME(deref_patterns): If `deref_patterns` stabilizes, it may make sense to move
                // this check out of this branch. Alternatively, this loop could be implemented with
                // autoderef and this check removed. For now though, don't break code compiling on
                // stable with lots of `&`s and a low recursion limit, if anyone's done that.
                if self.tcx.recursion_limit().value_within_limit(pat_adjustments.len()) {
                    // Preserve the smart pointer type for THIR lowering and closure upvar analysis.
                    pat_adjustments
                        .push(PatAdjustment { kind: PatAdjust::OverloadedDeref, source: expected });
                } else {
                    let guar = report_autoderef_recursion_limit_error(self.tcx, pat.span, expected);
                    inner_ty = Ty::new_error(self.tcx, guar);
                }
                drop(typeck_results);

                // Recurse, using the old pat info to keep `current_depth` to its old value.
                // Peeling smart pointers does not update the default binding mode.
                self.check_pat_inner(pat, opt_path_res, adjust_mode, inner_ty, old_pat_info)
            }
            PatKind::Missing | PatKind::Wild | PatKind::Err(_) => expected,
            // We allow any type here; we ensure that the type is uninhabited during match checking.
            PatKind::Never => expected,
            PatKind::Expr(PatExpr { kind: PatExprKind::Path(_), hir_id, .. }) => {
                let ty = match opt_path_res.unwrap() {
                    Ok(ref pr) => {
                        self.check_pat_path(pat.hir_id, pat.span, pr, expected, &pat_info.top_info)
                    }
                    Err(guar) => Ty::new_error(self.tcx, guar),
                };
                self.write_ty(*hir_id, ty);
                ty
            }
            PatKind::Expr(lt) => self.check_pat_lit(pat.span, lt, expected, &pat_info.top_info),
            PatKind::Range(lhs, rhs, _) => {
                self.check_pat_range(pat.span, lhs, rhs, expected, &pat_info.top_info)
            }
            PatKind::Binding(ba, var_id, ident, sub) => {
                self.check_pat_ident(pat, ba, var_id, ident, sub, expected, pat_info)
            }
            PatKind::TupleStruct(ref qpath, subpats, ddpos) => match opt_path_res.unwrap() {
                Ok(ResolvedPat { ty, kind: ResolvedPatKind::TupleStruct { res, variant } }) => self
                    .check_pat_tuple_struct(
                        pat, qpath, subpats, ddpos, res, ty, variant, expected, pat_info,
                    ),
                Err(guar) => {
                    let ty_err = Ty::new_error(self.tcx, guar);
                    for subpat in subpats {
                        self.check_pat(subpat, ty_err, pat_info);
                    }
                    ty_err
                }
                Ok(pr) => span_bug!(pat.span, "tuple struct pattern resolved to {pr:?}"),
            },
            PatKind::Struct(_, fields, has_rest_pat) => match opt_path_res.unwrap() {
                Ok(ResolvedPat { ty, kind: ResolvedPatKind::Struct { variant } }) => self
                    .check_pat_struct(pat, fields, has_rest_pat, ty, variant, expected, pat_info),
                Err(guar) => {
                    let ty_err = Ty::new_error(self.tcx, guar);
                    for field in fields {
                        self.check_pat(field.pat, ty_err, pat_info);
                    }
                    ty_err
                }
                Ok(pr) => span_bug!(pat.span, "struct pattern resolved to {pr:?}"),
            },
            PatKind::Guard(pat, cond) => {
                self.check_pat(pat, expected, pat_info);
                self.check_expr_has_type_or_error(cond, self.tcx.types.bool, |_| {});
                expected
            }
            PatKind::Or(pats) => {
                for pat in pats {
                    self.check_pat(pat, expected, pat_info);
                }
                expected
            }
            PatKind::Tuple(elements, ddpos) => {
                self.check_pat_tuple(pat.span, elements, ddpos, expected, pat_info)
            }
            PatKind::Box(inner) => self.check_pat_box(pat.span, inner, expected, pat_info),
            PatKind::Deref(inner) => self.check_pat_deref(pat.span, inner, expected, pat_info),
            PatKind::Ref(inner, mutbl) => self.check_pat_ref(pat, inner, mutbl, expected, pat_info),
            PatKind::Slice(before, slice, after) => {
                self.check_pat_slice(pat.span, before, slice, after, expected, pat_info)
            }
        }
    }

    /// How should the binding mode and expected type be adjusted?
    ///
    /// When the pattern contains a path, `opt_path_res` must be `Some(path_res)`.
    fn calc_adjust_mode(
        &self,
        pat: &'tcx Pat<'tcx>,
        opt_path_res: Option<Result<ResolvedPat<'tcx>, ErrorGuaranteed>>,
    ) -> AdjustMode {
        match &pat.kind {
            // Type checking these product-like types successfully always require
            // that the expected type be of those types and not reference types.
            PatKind::Tuple(..) | PatKind::Range(..) | PatKind::Slice(..) => AdjustMode::peel_all(),
            // When checking an explicit deref pattern, only peel reference types.
            // FIXME(deref_patterns): If box patterns and deref patterns need to coexist, box
            // patterns may want `PeelKind::Implicit`, stopping on encountering a box.
            PatKind::Box(_) | PatKind::Deref(_) => {
                AdjustMode::Peel { kind: PeelKind::ExplicitDerefPat }
            }
            // A never pattern behaves somewhat like a literal or unit variant.
            PatKind::Never => AdjustMode::peel_all(),
            // For patterns with paths, how we peel the scrutinee depends on the path's resolution.
            PatKind::Struct(..)
            | PatKind::TupleStruct(..)
            | PatKind::Expr(PatExpr { kind: PatExprKind::Path(_), .. }) => {
                // If there was an error resolving the path, default to peeling everything.
                opt_path_res.unwrap().map_or(AdjustMode::peel_all(), |pr| pr.adjust_mode())
            }

            // String and byte-string literals result in types `&str` and `&[u8]` respectively.
            // All other literals result in non-reference types.
            // As a result, we allow `if let 0 = &&0 {}` but not `if let "foo" = &&"foo" {}` unless
            // `deref_patterns` is enabled.
            PatKind::Expr(lt) => {
                // Path patterns have already been handled, and inline const blocks currently
                // aren't possible to write, so any handling for them would be untested.
                if cfg!(debug_assertions)
                    && self.tcx.features().deref_patterns()
                    && !matches!(lt.kind, PatExprKind::Lit { .. })
                {
                    span_bug!(
                        lt.span,
                        "FIXME(deref_patterns): adjust mode unimplemented for {:?}",
                        lt.kind
                    );
                }
                // Call `resolve_vars_if_possible` here for inline const blocks.
                let lit_ty = self.resolve_vars_if_possible(self.check_pat_expr_unadjusted(lt));
                // If `deref_patterns` is enabled, allow `if let "foo" = &&"foo" {}`.
                if self.tcx.features().deref_patterns() {
                    let mut peeled_ty = lit_ty;
                    let mut pat_ref_layers = 0;
                    while let ty::Ref(_, inner_ty, mutbl) =
                        *self.try_structurally_resolve_type(pat.span, peeled_ty).kind()
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
            PatKind::Ref(..)
            // No need to do anything on a missing pattern.
            | PatKind::Missing
            // A `_` pattern works with any expected type, so there's no need to do anything.
            | PatKind::Wild
            // A malformed pattern doesn't have an expected type, so let's just accept any type.
            | PatKind::Err(_)
            // Bindings also work with whatever the expected type is,
            // and moreover if we peel references off, that will give us the wrong binding type.
            // Also, we can have a subpattern `binding @ pat`.
            // Each side of the `@` should be treated independently (like with OR-patterns).
            | PatKind::Binding(..)
            // An OR-pattern just propagates to each individual alternative.
            // This is maximally flexible, allowing e.g., `Some(mut x) | &Some(mut x)`.
            // In that example, `Some(mut x)` results in `Peel` whereas `&Some(mut x)` in `Reset`.
            | PatKind::Or(_)
            // Like or-patterns, guard patterns just propogate to their subpatterns.
            | PatKind::Guard(..) => AdjustMode::Pass,
        }
    }

    /// Assuming `expected` is a reference type, determine whether to peel it before matching.
    fn should_peel_ref(&self, peel_kind: PeelKind, mut expected: Ty<'tcx>) -> bool {
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
            self.tcx.features().deref_patterns(),
            "Peeling for patterns with reference types is gated by `deref_patterns`."
        );

        // If the pattern has as many or more layers of reference as the expected type, we can match
        // without peeling more, unless we find a smart pointer or `&mut` that we also need to peel.
        // We don't treat `&` and `&mut` as interchangeable, but by peeling `&mut`s before matching,
        // we can still, e.g., match on a `&mut str` with a string literal pattern. This is because
        // string literal patterns may be used where `str` is expected.
        let mut expected_ref_layers = 0;
        while let ty::Ref(_, inner_ty, mutbl) = *expected.kind() {
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
    fn should_peel_smart_pointer(&self, peel_kind: PeelKind, expected: Ty<'tcx>) -> bool {
        // Explicit `deref!(_)` patterns match against smart pointers; don't peel in that case.
        if let PeelKind::Implicit { until_adt, .. } = peel_kind
            // For simplicity, only apply overloaded derefs if `expected` is a known ADT.
            // FIXME(deref_patterns): we'll get better diagnostics for users trying to
            // implicitly deref generics if we allow them here, but primitives, tuples, and
            // inference vars definitely should be stopped. Figure out what makes most sense.
            && let ty::Adt(scrutinee_adt, _) = *expected.kind()
            // Don't peel if the pattern type already matches the scrutinee. E.g., stop here if
            // matching on a `Cow<'a, T>` scrutinee with a `Cow::Owned(_)` pattern.
            && until_adt != Some(scrutinee_adt.did())
            // At this point, the pattern isn't able to match `expected` without peeling. Check
            // that it implements `Deref` before assuming it's a smart pointer, to get a normal
            // type error instead of a missing impl error if not. This only checks for `Deref`,
            // not `DerefPure`: we require that too, but we want a trait error if it's missing.
            && let Some(deref_trait) = self.tcx.lang_items().deref_trait()
            && self.type_implements_trait(deref_trait, [expected], self.param_env).may_apply()
        {
            true
        } else {
            false
        }
    }

    fn check_pat_expr_unadjusted(&self, lt: &'tcx hir::PatExpr<'tcx>) -> Ty<'tcx> {
        let ty = match &lt.kind {
            rustc_hir::PatExprKind::Lit { lit, negated } => {
                let ty = self.check_expr_lit(lit, Expectation::NoExpectation);
                if *negated {
                    self.register_bound(
                        ty,
                        self.tcx.require_lang_item(LangItem::Neg, lt.span),
                        ObligationCause::dummy_with_span(lt.span),
                    );
                }
                ty
            }
            rustc_hir::PatExprKind::ConstBlock(c) => {
                self.check_expr_const_block(c, Expectation::NoExpectation)
            }
            rustc_hir::PatExprKind::Path(qpath) => {
                let (res, opt_ty, segments) =
                    self.resolve_ty_and_res_fully_qualified_call(qpath, lt.hir_id, lt.span);
                self.instantiate_value_path(segments, opt_ty, res, lt.span, lt.span, lt.hir_id).0
            }
        };
        self.write_ty(lt.hir_id, ty);
        ty
    }

    fn check_pat_lit(
        &self,
        span: Span,
        lt: &hir::PatExpr<'tcx>,
        expected: Ty<'tcx>,
        ti: &TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        // We've already computed the type above (when checking for a non-ref pat),
        // so avoid computing it again.
        let ty = self.node_ty(lt.hir_id);

        // Byte string patterns behave the same way as array patterns
        // They can denote both statically and dynamically-sized byte arrays.
        // Additionally, when `deref_patterns` is enabled, byte string literal patterns may have
        // types `[u8]` or `[u8; N]`, in order to type, e.g., `deref!(b"..."): Vec<u8>`.
        let mut pat_ty = ty;
        if let hir::PatExprKind::Lit {
            lit: Spanned { node: ast::LitKind::ByteStr(..), .. }, ..
        } = lt.kind
        {
            let tcx = self.tcx;
            let expected = self.structurally_resolve_type(span, expected);
            match *expected.kind() {
                // Allow `b"...": &[u8]`
                ty::Ref(_, inner_ty, _)
                    if self.try_structurally_resolve_type(span, inner_ty).is_slice() =>
                {
                    trace!(?lt.hir_id.local_id, "polymorphic byte string lit");
                    pat_ty = Ty::new_imm_ref(
                        tcx,
                        tcx.lifetimes.re_static,
                        Ty::new_slice(tcx, tcx.types.u8),
                    );
                }
                // Allow `b"...": [u8; 3]` for `deref_patterns`
                ty::Array(..) if tcx.features().deref_patterns() => {
                    pat_ty = match *ty.kind() {
                        ty::Ref(_, inner_ty, _) => inner_ty,
                        _ => span_bug!(span, "found byte string literal with non-ref type {ty:?}"),
                    }
                }
                // Allow `b"...": [u8]` for `deref_patterns`
                ty::Slice(..) if tcx.features().deref_patterns() => {
                    pat_ty = Ty::new_slice(tcx, tcx.types.u8);
                }
                // Otherwise, `b"...": &[u8; 3]`
                _ => {}
            }
        }

        // When `deref_patterns` is enabled, in order to allow `deref!("..."): String`, we allow
        // string literal patterns to have type `str`. This is accounted for when lowering to MIR.
        if self.tcx.features().deref_patterns()
            && let hir::PatExprKind::Lit {
                lit: Spanned { node: ast::LitKind::Str(..), .. }, ..
            } = lt.kind
            && self.try_structurally_resolve_type(span, expected).is_str()
        {
            pat_ty = self.tcx.types.str_;
        }

        if self.tcx.features().string_deref_patterns()
            && let hir::PatExprKind::Lit {
                lit: Spanned { node: ast::LitKind::Str(..), .. }, ..
            } = lt.kind
        {
            let tcx = self.tcx;
            let expected = self.resolve_vars_if_possible(expected);
            pat_ty = match expected.kind() {
                ty::Adt(def, _) if tcx.is_lang_item(def.did(), LangItem::String) => expected,
                ty::Str => Ty::new_static_str(tcx),
                _ => pat_ty,
            };
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
        let cause = self.pattern_cause(ti, span);
        if let Err(err) = self.demand_suptype_with_origin(&cause, expected, pat_ty) {
            err.emit_unless(
                ti.span
                    .filter(|&s| {
                        // In the case of `if`- and `while`-expressions we've already checked
                        // that `scrutinee: bool`. We know that the pattern is `true`,
                        // so an error here would be a duplicate and from the wrong POV.
                        s.is_desugaring(DesugaringKind::CondTemporary)
                    })
                    .is_some(),
            );
        }

        pat_ty
    }

    fn check_pat_range(
        &self,
        span: Span,
        lhs: Option<&'tcx hir::PatExpr<'tcx>>,
        rhs: Option<&'tcx hir::PatExpr<'tcx>>,
        expected: Ty<'tcx>,
        ti: &TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        let calc_side = |opt_expr: Option<&'tcx hir::PatExpr<'tcx>>| match opt_expr {
            None => None,
            Some(expr) => {
                let ty = self.check_pat_expr_unadjusted(expr);
                // Check that the end-point is possibly of numeric or char type.
                // The early check here is not for correctness, but rather better
                // diagnostics (e.g. when `&str` is being matched, `expected` will
                // be peeled to `str` while ty here is still `&str`, if we don't
                // err early here, a rather confusing unification error will be
                // emitted instead).
                let ty = self.try_structurally_resolve_type(expr.span, ty);
                let fail =
                    !(ty.is_numeric() || ty.is_char() || ty.is_ty_var() || ty.references_error());
                Some((fail, ty, expr.span))
            }
        };
        let mut lhs = calc_side(lhs);
        let mut rhs = calc_side(rhs);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            // There exists a side that didn't meet our criteria that the end-point
            // be of a numeric or char type, as checked in `calc_side` above.
            let guar = self.emit_err_pat_range(span, lhs, rhs);
            return Ty::new_error(self.tcx, guar);
        }

        // Unify each side with `expected`.
        // Subtyping doesn't matter here, as the value is some kind of scalar.
        let demand_eqtype = |x: &mut _, y| {
            if let Some((ref mut fail, x_ty, x_span)) = *x
                && let Err(mut err) = self.demand_eqtype_pat_diag(x_span, expected, x_ty, ti)
            {
                if let Some((_, y_ty, y_span)) = y {
                    self.endpoint_has_type(&mut err, y_span, y_ty);
                }
                err.emit();
                *fail = true;
            }
        };
        demand_eqtype(&mut lhs, rhs);
        demand_eqtype(&mut rhs, lhs);

        if let (Some((true, ..)), _) | (_, Some((true, ..))) = (lhs, rhs) {
            return Ty::new_misc_error(self.tcx);
        }

        // Find the unified type and check if it's of numeric or char type again.
        // This check is needed if both sides are inference variables.
        // We require types to be resolved here so that we emit inference failure
        // rather than "_ is not a char or numeric".
        let ty = self.structurally_resolve_type(span, expected);
        if !(ty.is_numeric() || ty.is_char() || ty.references_error()) {
            if let Some((ref mut fail, _, _)) = lhs {
                *fail = true;
            }
            if let Some((ref mut fail, _, _)) = rhs {
                *fail = true;
            }
            let guar = self.emit_err_pat_range(span, lhs, rhs);
            return Ty::new_error(self.tcx, guar);
        }
        ty
    }

    fn endpoint_has_type(&self, err: &mut Diag<'_>, span: Span, ty: Ty<'_>) {
        if !ty.references_error() {
            err.span_label(span, format!("this is of type `{ty}`"));
        }
    }

    fn emit_err_pat_range(
        &self,
        span: Span,
        lhs: Option<(bool, Ty<'tcx>, Span)>,
        rhs: Option<(bool, Ty<'tcx>, Span)>,
    ) -> ErrorGuaranteed {
        let span = match (lhs, rhs) {
            (Some((true, ..)), Some((true, ..))) => span,
            (Some((true, _, sp)), _) => sp,
            (_, Some((true, _, sp))) => sp,
            _ => span_bug!(span, "emit_err_pat_range: no side failed or exists but still error?"),
        };
        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0029,
            "only `char` and numeric types are allowed in range patterns"
        );
        let msg = |ty| {
            let ty = self.resolve_vars_if_possible(ty);
            format!("this is of type `{ty}` but it should be `char` or numeric")
        };
        let mut one_side_err = |first_span, first_ty, second: Option<(bool, Ty<'tcx>, Span)>| {
            err.span_label(first_span, msg(first_ty));
            if let Some((_, ty, sp)) = second {
                let ty = self.resolve_vars_if_possible(ty);
                self.endpoint_has_type(&mut err, sp, ty);
            }
        };
        match (lhs, rhs) {
            (Some((true, lhs_ty, lhs_sp)), Some((true, rhs_ty, rhs_sp))) => {
                err.span_label(lhs_sp, msg(lhs_ty));
                err.span_label(rhs_sp, msg(rhs_ty));
            }
            (Some((true, lhs_ty, lhs_sp)), rhs) => one_side_err(lhs_sp, lhs_ty, rhs),
            (lhs, Some((true, rhs_ty, rhs_sp))) => one_side_err(rhs_sp, rhs_ty, lhs),
            _ => span_bug!(span, "Impossible, verified above."),
        }
        if (lhs, rhs).references_error() {
            err.downgrade_to_delayed_bug();
        }
        if self.tcx.sess.teach(err.code.unwrap()) {
            err.note(
                "In a match expression, only numbers and characters can be matched \
                    against a range. This is because the compiler checks that the range \
                    is non-empty at compile-time, and is unable to evaluate arbitrary \
                    comparison functions. If you want to capture values of an orderable \
                    type between two end-points, you can use a guard.",
            );
        }
        err.emit()
    }

    fn check_pat_ident(
        &self,
        pat: &'tcx Pat<'tcx>,
        user_bind_annot: BindingMode,
        var_id: HirId,
        ident: Ident,
        sub: Option<&'tcx Pat<'tcx>>,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let PatInfo { binding_mode: def_br, top_info: ti, .. } = pat_info;

        // Determine the binding mode...
        let bm = match user_bind_annot {
            BindingMode(ByRef::No, Mutability::Mut) if let ByRef::Yes(def_br_mutbl) = def_br => {
                // Only mention the experimental `mut_ref` feature if if we're in edition 2024 and
                // using other experimental matching features compatible with it.
                if pat.span.at_least_rust_2024()
                    && (self.tcx.features().ref_pat_eat_one_layer_2024()
                        || self.tcx.features().ref_pat_eat_one_layer_2024_structural())
                {
                    if !self.tcx.features().mut_ref() {
                        feature_err(
                            &self.tcx.sess,
                            sym::mut_ref,
                            pat.span.until(ident.span),
                            "binding cannot be both mutable and by-reference",
                        )
                        .emit();
                    }

                    BindingMode(def_br, Mutability::Mut)
                } else {
                    // `mut` resets the binding mode on edition <= 2021
                    self.add_rust_2024_migration_desugared_pat(
                        pat_info.top_info.hir_id,
                        pat,
                        't', // last char of `mut`
                        def_br_mutbl,
                    );
                    BindingMode(ByRef::No, Mutability::Mut)
                }
            }
            BindingMode(ByRef::No, mutbl) => BindingMode(def_br, mutbl),
            BindingMode(ByRef::Yes(user_br_mutbl), _) => {
                if let ByRef::Yes(def_br_mutbl) = def_br {
                    // `ref`/`ref mut` overrides the binding mode on edition <= 2021
                    self.add_rust_2024_migration_desugared_pat(
                        pat_info.top_info.hir_id,
                        pat,
                        match user_br_mutbl {
                            Mutability::Not => 'f', // last char of `ref`
                            Mutability::Mut => 't', // last char of `ref mut`
                        },
                        def_br_mutbl,
                    );
                }
                user_bind_annot
            }
        };

        if bm.0 == ByRef::Yes(Mutability::Mut)
            && let MutblCap::WeaklyNot(and_pat_span) = pat_info.max_ref_mutbl
        {
            let mut err = struct_span_code_err!(
                self.dcx(),
                ident.span,
                E0596,
                "cannot borrow as mutable inside an `&` pattern"
            );

            if let Some(span) = and_pat_span {
                err.span_suggestion(
                    span,
                    "replace this `&` with `&mut`",
                    "&mut ",
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        }

        // ...and store it in a side table:
        self.typeck_results.borrow_mut().pat_binding_modes_mut().insert(pat.hir_id, bm);

        debug!("check_pat_ident: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);

        let local_ty = self.local_ty(pat.span, pat.hir_id);
        let eq_ty = match bm.0 {
            ByRef::Yes(mutbl) => {
                // If the binding is like `ref x | ref mut x`,
                // then `x` is assigned a value of type `&M T` where M is the
                // mutability and T is the expected type.
                //
                // `x` is assigned a value of type `&M T`, hence `&M T <: typeof(x)`
                // is required. However, we use equality, which is stronger.
                // See (note_1) for an explanation.
                self.new_ref_ty(pat.span, mutbl, expected)
            }
            // Otherwise, the type of x is the expected type `T`.
            ByRef::No => expected, // As above, `T <: typeof(x)` is required, but we use equality, see (note_1).
        };

        // We have a concrete type for the local, so we do not need to taint it and hide follow up errors *using* the local.
        let _ = self.demand_eqtype_pat(pat.span, eq_ty, local_ty, &ti);

        // If there are multiple arms, make sure they all agree on
        // what the type of the binding `x` ought to be.
        if var_id != pat.hir_id {
            self.check_binding_alt_eq_ty(user_bind_annot, pat.span, var_id, local_ty, &ti);
        }

        if let Some(p) = sub {
            self.check_pat(p, expected, pat_info);
        }

        local_ty
    }

    /// When a variable is bound several times in a `PatKind::Or`, it'll resolve all of the
    /// subsequent bindings of the same name to the first usage. Verify that all of these
    /// bindings have the same type by comparing them all against the type of that first pat.
    fn check_binding_alt_eq_ty(
        &self,
        ba: BindingMode,
        span: Span,
        var_id: HirId,
        ty: Ty<'tcx>,
        ti: &TopInfo<'tcx>,
    ) {
        let var_ty = self.local_ty(span, var_id);
        if let Err(mut err) = self.demand_eqtype_pat_diag(span, var_ty, ty, ti) {
            let var_ty = self.resolve_vars_if_possible(var_ty);
            let msg = format!("first introduced with type `{var_ty}` here");
            err.span_label(self.tcx.hir_span(var_id), msg);
            let in_match = self.tcx.hir_parent_iter(var_id).any(|(_, n)| {
                matches!(
                    n,
                    hir::Node::Expr(hir::Expr {
                        kind: hir::ExprKind::Match(.., hir::MatchSource::Normal),
                        ..
                    })
                )
            });
            let pre = if in_match { "in the same arm, " } else { "" };
            err.note(format!("{pre}a binding must have the same type in all alternatives"));
            self.suggest_adding_missing_ref_or_removing_ref(
                &mut err,
                span,
                var_ty,
                self.resolve_vars_if_possible(ty),
                ba,
            );
            err.emit();
        }
    }

    fn suggest_adding_missing_ref_or_removing_ref(
        &self,
        err: &mut Diag<'_>,
        span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        ba: BindingMode,
    ) {
        match (expected.kind(), actual.kind(), ba) {
            (ty::Ref(_, inner_ty, _), _, BindingMode::NONE)
                if self.can_eq(self.param_env, *inner_ty, actual) =>
            {
                err.span_suggestion_verbose(
                    span.shrink_to_lo(),
                    "consider adding `ref`",
                    "ref ",
                    Applicability::MaybeIncorrect,
                );
            }
            (_, ty::Ref(_, inner_ty, _), BindingMode::REF)
                if self.can_eq(self.param_env, expected, *inner_ty) =>
            {
                err.span_suggestion_verbose(
                    span.with_hi(span.lo() + BytePos(4)),
                    "consider removing `ref`",
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
            _ => (),
        }
    }

    /// Precondition: pat is a `Ref(_)` pattern
    fn borrow_pat_suggestion(&self, err: &mut Diag<'_>, pat: &Pat<'_>) {
        let tcx = self.tcx;
        if let PatKind::Ref(inner, mutbl) = pat.kind
            && let PatKind::Binding(_, _, binding, ..) = inner.kind
        {
            let binding_parent = tcx.parent_hir_node(pat.hir_id);
            debug!(?inner, ?pat, ?binding_parent);

            let mutability = match mutbl {
                ast::Mutability::Mut => "mut",
                ast::Mutability::Not => "",
            };

            let mut_var_suggestion = 'block: {
                if mutbl.is_not() {
                    break 'block None;
                }

                let ident_kind = match binding_parent {
                    hir::Node::Param(_) => "parameter",
                    hir::Node::LetStmt(_) => "variable",
                    hir::Node::Arm(_) => "binding",

                    // Provide diagnostics only if the parent pattern is struct-like,
                    // i.e. where `mut binding` makes sense
                    hir::Node::Pat(Pat { kind, .. }) => match kind {
                        PatKind::Struct(..)
                        | PatKind::TupleStruct(..)
                        | PatKind::Or(..)
                        | PatKind::Guard(..)
                        | PatKind::Tuple(..)
                        | PatKind::Slice(..) => "binding",

                        PatKind::Missing
                        | PatKind::Wild
                        | PatKind::Never
                        | PatKind::Binding(..)
                        | PatKind::Box(..)
                        | PatKind::Deref(_)
                        | PatKind::Ref(..)
                        | PatKind::Expr(..)
                        | PatKind::Range(..)
                        | PatKind::Err(_) => break 'block None,
                    },

                    // Don't provide suggestions in other cases
                    _ => break 'block None,
                };

                Some((
                    pat.span,
                    format!("to declare a mutable {ident_kind} use"),
                    format!("mut {binding}"),
                ))
            };

            match binding_parent {
                // Check that there is explicit type (ie this is not a closure param with inferred type)
                // so we don't suggest moving something to the type that does not exist
                hir::Node::Param(hir::Param { ty_span, pat, .. }) if pat.span != *ty_span => {
                    err.multipart_suggestion_verbose(
                        format!("to take parameter `{binding}` by reference, move `&{mutability}` to the type"),
                        vec![
                            (pat.span.until(inner.span), "".to_owned()),
                            (ty_span.shrink_to_lo(), mutbl.ref_prefix_str().to_owned()),
                        ],
                        Applicability::MachineApplicable
                    );

                    if let Some((sp, msg, sugg)) = mut_var_suggestion {
                        err.span_note(sp, format!("{msg}: `{sugg}`"));
                    }
                }
                hir::Node::Pat(pt) if let PatKind::TupleStruct(_, pat_arr, _) = pt.kind => {
                    for i in pat_arr.iter() {
                        if let PatKind::Ref(the_ref, _) = i.kind
                            && let PatKind::Binding(mt, _, ident, _) = the_ref.kind
                        {
                            let BindingMode(_, mtblty) = mt;
                            err.span_suggestion_verbose(
                                i.span,
                                format!("consider removing `&{mutability}` from the pattern"),
                                mtblty.prefix_str().to_string() + &ident.name.to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    if let Some((sp, msg, sugg)) = mut_var_suggestion {
                        err.span_note(sp, format!("{msg}: `{sugg}`"));
                    }
                }
                hir::Node::Param(_) | hir::Node::Arm(_) | hir::Node::Pat(_) => {
                    // rely on match ergonomics or it might be nested `&&pat`
                    err.span_suggestion_verbose(
                        pat.span.until(inner.span),
                        format!("consider removing `&{mutability}` from the pattern"),
                        "",
                        Applicability::MaybeIncorrect,
                    );

                    if let Some((sp, msg, sugg)) = mut_var_suggestion {
                        err.span_note(sp, format!("{msg}: `{sugg}`"));
                    }
                }
                _ if let Some((sp, msg, sugg)) = mut_var_suggestion => {
                    err.span_suggestion(sp, msg, sugg, Applicability::MachineApplicable);
                }
                _ => {} // don't provide suggestions in other cases #55175
            }
        }
    }

    fn check_dereferenceable(
        &self,
        span: Span,
        expected: Ty<'tcx>,
        inner: &Pat<'_>,
    ) -> Result<(), ErrorGuaranteed> {
        if let PatKind::Binding(..) = inner.kind
            && let Some(pointee_ty) = self.shallow_resolve(expected).builtin_deref(true)
            && let ty::Dynamic(..) = pointee_ty.kind()
        {
            // This is "x = dyn SomeTrait" being reduced from
            // "let &x = &dyn SomeTrait" or "let box x = Box<dyn SomeTrait>", an error.
            let type_str = self.ty_to_string(expected);
            let mut err = struct_span_code_err!(
                self.dcx(),
                span,
                E0033,
                "type `{}` cannot be dereferenced",
                type_str
            );
            err.span_label(span, format!("type `{type_str}` cannot be dereferenced"));
            if self.tcx.sess.teach(err.code.unwrap()) {
                err.note(CANNOT_IMPLICITLY_DEREF_POINTER_TRAIT_OBJ);
            }
            return Err(err.emit());
        }
        Ok(())
    }

    fn resolve_pat_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &hir::QPath<'tcx>,
    ) -> Result<ResolvedPat<'tcx>, ErrorGuaranteed> {
        // Resolve the path and check the definition for errors.
        let (variant, pat_ty) = self.check_struct_path(qpath, pat.hir_id)?;
        Ok(ResolvedPat { ty: pat_ty, kind: ResolvedPatKind::Struct { variant } })
    }

    fn check_pat_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        fields: &'tcx [hir::PatField<'tcx>],
        has_rest_pat: bool,
        pat_ty: Ty<'tcx>,
        variant: &'tcx VariantDef,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        // Type-check the path.
        let _ = self.demand_eqtype_pat(pat.span, expected, pat_ty, &pat_info.top_info);

        // Type-check subpatterns.
        match self.check_struct_pat_fields(pat_ty, pat, variant, fields, has_rest_pat, pat_info) {
            Ok(()) => pat_ty,
            Err(guar) => Ty::new_error(self.tcx, guar),
        }
    }

    fn resolve_pat_path(
        &self,
        path_id: HirId,
        span: Span,
        qpath: &'tcx hir::QPath<'_>,
    ) -> Result<ResolvedPat<'tcx>, ErrorGuaranteed> {
        let tcx = self.tcx;

        let (res, opt_ty, segments) =
            self.resolve_ty_and_res_fully_qualified_call(qpath, path_id, span);
        match res {
            Res::Err => {
                let e =
                    self.dcx().span_delayed_bug(qpath.span(), "`Res::Err` but no error emitted");
                self.set_tainted_by_errors(e);
                return Err(e);
            }
            Res::Def(DefKind::AssocFn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::Variant, _) => {
                let expected = "unit struct, unit variant or constant";
                let e = report_unexpected_variant_res(tcx, res, None, qpath, span, E0533, expected);
                return Err(e);
            }
            Res::SelfCtor(def_id) => {
                if let ty::Adt(adt_def, _) = *tcx.type_of(def_id).skip_binder().kind()
                    && adt_def.is_struct()
                    && let Some((CtorKind::Const, _)) = adt_def.non_enum_variant().ctor
                {
                    // Ok, we allow unit struct ctors in patterns only.
                } else {
                    let e = report_unexpected_variant_res(
                        tcx,
                        res,
                        None,
                        qpath,
                        span,
                        E0533,
                        "unit struct",
                    );
                    return Err(e);
                }
            }
            Res::Def(
                DefKind::Ctor(_, CtorKind::Const)
                | DefKind::Const
                | DefKind::AssocConst
                | DefKind::ConstParam,
                _,
            ) => {} // OK
            _ => bug!("unexpected pattern resolution: {:?}", res),
        }

        // Find the type of the path pattern, for later checking.
        let (pat_ty, pat_res) =
            self.instantiate_value_path(segments, opt_ty, res, span, span, path_id);
        Ok(ResolvedPat { ty: pat_ty, kind: ResolvedPatKind::Path { res, pat_res, segments } })
    }

    fn check_pat_path(
        &self,
        pat_id_for_diag: HirId,
        span: Span,
        resolved: &ResolvedPat<'tcx>,
        expected: Ty<'tcx>,
        ti: &TopInfo<'tcx>,
    ) -> Ty<'tcx> {
        if let Err(err) =
            self.demand_suptype_with_origin(&self.pattern_cause(ti, span), expected, resolved.ty)
        {
            self.emit_bad_pat_path(err, pat_id_for_diag, span, resolved);
        }
        resolved.ty
    }

    fn maybe_suggest_range_literal(
        &self,
        e: &mut Diag<'_>,
        opt_def_id: Option<hir::def_id::DefId>,
        ident: Ident,
    ) -> bool {
        match opt_def_id {
            Some(def_id) => match self.tcx.hir_get_if_local(def_id) {
                Some(hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Const(_, _, _, body_id),
                    ..
                })) => match self.tcx.hir_node(body_id.hir_id) {
                    hir::Node::Expr(expr) => {
                        if hir::is_range_literal(expr) {
                            let span = self.tcx.hir_span(body_id.hir_id);
                            if let Ok(snip) = self.tcx.sess.source_map().span_to_snippet(span) {
                                e.span_suggestion_verbose(
                                    ident.span,
                                    "you may want to move the range into the match block",
                                    snip,
                                    Applicability::MachineApplicable,
                                );
                                return true;
                            }
                        }
                    }
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        }
        false
    }

    fn emit_bad_pat_path(
        &self,
        mut e: Diag<'_>,
        hir_id: HirId,
        pat_span: Span,
        resolved_pat: &ResolvedPat<'tcx>,
    ) {
        let ResolvedPatKind::Path { res, pat_res, segments } = resolved_pat.kind else {
            span_bug!(pat_span, "unexpected resolution for path pattern: {resolved_pat:?}");
        };

        if let Some(span) = self.tcx.hir_res_span(pat_res) {
            e.span_label(span, format!("{} defined here", res.descr()));
            if let [hir::PathSegment { ident, .. }] = &*segments {
                e.span_label(
                    pat_span,
                    format!(
                        "`{}` is interpreted as {} {}, not a new binding",
                        ident,
                        res.article(),
                        res.descr(),
                    ),
                );
                match self.tcx.parent_hir_node(hir_id) {
                    hir::Node::PatField(..) => {
                        e.span_suggestion_verbose(
                            ident.span.shrink_to_hi(),
                            "bind the struct field to a different name instead",
                            format!(": other_{}", ident.as_str().to_lowercase()),
                            Applicability::HasPlaceholders,
                        );
                    }
                    _ => {
                        let (type_def_id, item_def_id) = match resolved_pat.ty.kind() {
                            ty::Adt(def, _) => match res {
                                Res::Def(DefKind::Const, def_id) => (Some(def.did()), Some(def_id)),
                                _ => (None, None),
                            },
                            _ => (None, None),
                        };

                        let is_range = match type_def_id.and_then(|id| self.tcx.as_lang_item(id)) {
                            Some(
                                LangItem::Range
                                | LangItem::RangeFrom
                                | LangItem::RangeTo
                                | LangItem::RangeFull
                                | LangItem::RangeInclusiveStruct
                                | LangItem::RangeToInclusive,
                            ) => true,
                            _ => false,
                        };
                        if is_range {
                            if !self.maybe_suggest_range_literal(&mut e, item_def_id, *ident) {
                                let msg = "constants only support matching by type, \
                                    if you meant to match against a range of values, \
                                    consider using a range pattern like `min ..= max` in the match block";
                                e.note(msg);
                            }
                        } else {
                            let msg = "introduce a new binding instead";
                            let sugg = format!("other_{}", ident.as_str().to_lowercase());
                            e.span_suggestion(
                                ident.span,
                                msg,
                                sugg,
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                };
            }
        }
        e.emit();
    }

    fn resolve_pat_tuple_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &'tcx hir::QPath<'tcx>,
    ) -> Result<ResolvedPat<'tcx>, ErrorGuaranteed> {
        let tcx = self.tcx;
        let report_unexpected_res = |res: Res| {
            let expected = "tuple struct or tuple variant";
            let e = report_unexpected_variant_res(tcx, res, None, qpath, pat.span, E0164, expected);
            Err(e)
        };

        // Resolve the path and check the definition for errors.
        let (res, opt_ty, segments) =
            self.resolve_ty_and_res_fully_qualified_call(qpath, pat.hir_id, pat.span);
        if res == Res::Err {
            let e = self.dcx().span_delayed_bug(pat.span, "`Res::Err` but no error emitted");
            self.set_tainted_by_errors(e);
            return Err(e);
        }

        // Type-check the path.
        let (pat_ty, res) =
            self.instantiate_value_path(segments, opt_ty, res, pat.span, pat.span, pat.hir_id);
        if !pat_ty.is_fn() {
            return report_unexpected_res(res);
        }

        let variant = match res {
            Res::Err => {
                self.dcx().span_bug(pat.span, "`Res::Err` but no error emitted");
            }
            Res::Def(DefKind::AssocConst | DefKind::AssocFn, _) => {
                return report_unexpected_res(res);
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) => tcx.expect_variant_res(res),
            _ => bug!("unexpected pattern resolution: {:?}", res),
        };

        // Replace constructor type with constructed type for tuple struct patterns.
        let pat_ty = pat_ty.fn_sig(tcx).output();
        let pat_ty = pat_ty.no_bound_vars().expect("expected fn type");

        Ok(ResolvedPat { ty: pat_ty, kind: ResolvedPatKind::TupleStruct { res, variant } })
    }

    fn check_pat_tuple_struct(
        &self,
        pat: &'tcx Pat<'tcx>,
        qpath: &'tcx hir::QPath<'tcx>,
        subpats: &'tcx [Pat<'tcx>],
        ddpos: hir::DotDotPos,
        res: Res,
        pat_ty: Ty<'tcx>,
        variant: &'tcx VariantDef,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let on_error = |e| {
            for pat in subpats {
                self.check_pat(pat, Ty::new_error(tcx, e), pat_info);
            }
        };

        // Type-check the tuple struct pattern against the expected type.
        let diag = self.demand_eqtype_pat_diag(pat.span, expected, pat_ty, &pat_info.top_info);
        let had_err = diag.map_err(|diag| diag.emit());

        // Type-check subpatterns.
        if subpats.len() == variant.fields.len()
            || subpats.len() < variant.fields.len() && ddpos.as_opt_usize().is_some()
        {
            let ty::Adt(_, args) = pat_ty.kind() else {
                bug!("unexpected pattern type {:?}", pat_ty);
            };
            for (i, subpat) in subpats.iter().enumerate_and_adjust(variant.fields.len(), ddpos) {
                let field = &variant.fields[FieldIdx::from_usize(i)];
                let field_ty = self.field_ty(subpat.span, field, args);
                self.check_pat(subpat, field_ty, pat_info);

                self.tcx.check_stability(
                    variant.fields[FieldIdx::from_usize(i)].did,
                    Some(subpat.hir_id),
                    subpat.span,
                    None,
                );
            }
            if let Err(e) = had_err {
                on_error(e);
                return Ty::new_error(tcx, e);
            }
        } else {
            let e = self.emit_err_pat_wrong_number_of_fields(
                pat.span,
                res,
                qpath,
                subpats,
                &variant.fields.raw,
                expected,
                had_err,
            );
            on_error(e);
            return Ty::new_error(tcx, e);
        }
        pat_ty
    }

    fn emit_err_pat_wrong_number_of_fields(
        &self,
        pat_span: Span,
        res: Res,
        qpath: &hir::QPath<'_>,
        subpats: &'tcx [Pat<'tcx>],
        fields: &'tcx [ty::FieldDef],
        expected: Ty<'tcx>,
        had_err: Result<(), ErrorGuaranteed>,
    ) -> ErrorGuaranteed {
        let subpats_ending = pluralize!(subpats.len());
        let fields_ending = pluralize!(fields.len());

        let subpat_spans = if subpats.is_empty() {
            vec![pat_span]
        } else {
            subpats.iter().map(|p| p.span).collect()
        };
        let last_subpat_span = *subpat_spans.last().unwrap();
        let res_span = self.tcx.def_span(res.def_id());
        let def_ident_span = self.tcx.def_ident_span(res.def_id()).unwrap_or(res_span);
        let field_def_spans = if fields.is_empty() {
            vec![res_span]
        } else {
            fields.iter().map(|f| f.ident(self.tcx).span).collect()
        };
        let last_field_def_span = *field_def_spans.last().unwrap();

        let mut err = struct_span_code_err!(
            self.dcx(),
            MultiSpan::from_spans(subpat_spans),
            E0023,
            "this pattern has {} field{}, but the corresponding {} has {} field{}",
            subpats.len(),
            subpats_ending,
            res.descr(),
            fields.len(),
            fields_ending,
        );
        err.span_label(
            last_subpat_span,
            format!("expected {} field{}, found {}", fields.len(), fields_ending, subpats.len()),
        );
        if self.tcx.sess.source_map().is_multiline(qpath.span().between(last_subpat_span)) {
            err.span_label(qpath.span(), "");
        }
        if self.tcx.sess.source_map().is_multiline(def_ident_span.between(last_field_def_span)) {
            err.span_label(def_ident_span, format!("{} defined here", res.descr()));
        }
        for span in &field_def_spans[..field_def_spans.len() - 1] {
            err.span_label(*span, "");
        }
        err.span_label(
            last_field_def_span,
            format!("{} has {} field{}", res.descr(), fields.len(), fields_ending),
        );

        // Identify the case `Some(x, y)` where the expected type is e.g. `Option<(T, U)>`.
        // More generally, the expected type wants a tuple variant with one field of an
        // N-arity-tuple, e.g., `V_i((p_0, .., p_N))`. Meanwhile, the user supplied a pattern
        // with the subpatterns directly in the tuple variant pattern, e.g., `V_i(p_0, .., p_N)`.
        let missing_parentheses = match (expected.kind(), fields, had_err) {
            // #67037: only do this if we could successfully type-check the expected type against
            // the tuple struct pattern. Otherwise the args could get out of range on e.g.,
            // `let P() = U;` where `P != U` with `struct P<T>(T);`.
            (ty::Adt(_, args), [field], Ok(())) => {
                let field_ty = self.field_ty(pat_span, field, args);
                match field_ty.kind() {
                    ty::Tuple(fields) => fields.len() == subpats.len(),
                    _ => false,
                }
            }
            _ => false,
        };
        if missing_parentheses {
            let (left, right) = match subpats {
                // This is the zero case; we aim to get the "hi" part of the `QPath`'s
                // span as the "lo" and then the "hi" part of the pattern's span as the "hi".
                // This looks like:
                //
                // help: missing parentheses
                //   |
                // L |     let A(()) = A(());
                //   |          ^  ^
                [] => (qpath.span().shrink_to_hi(), pat_span),
                // Easy case. Just take the "lo" of the first sub-pattern and the "hi" of the
                // last sub-pattern. In the case of `A(x)` the first and last may coincide.
                // This looks like:
                //
                // help: missing parentheses
                //   |
                // L |     let A((x, y)) = A((1, 2));
                //   |           ^    ^
                [first, ..] => (first.span.shrink_to_lo(), subpats.last().unwrap().span),
            };
            err.multipart_suggestion(
                "missing parentheses",
                vec![(left, "(".to_string()), (right.shrink_to_hi(), ")".to_string())],
                Applicability::MachineApplicable,
            );
        } else if fields.len() > subpats.len() && pat_span != DUMMY_SP {
            let after_fields_span = pat_span.with_hi(pat_span.hi() - BytePos(1)).shrink_to_hi();
            let all_fields_span = match subpats {
                [] => after_fields_span,
                [field] => field.span,
                [first, .., last] => first.span.to(last.span),
            };

            // Check if all the fields in the pattern are wildcards.
            let all_wildcards = subpats.iter().all(|pat| matches!(pat.kind, PatKind::Wild));
            let first_tail_wildcard =
                subpats.iter().enumerate().fold(None, |acc, (pos, pat)| match (acc, &pat.kind) {
                    (None, PatKind::Wild) => Some(pos),
                    (Some(_), PatKind::Wild) => acc,
                    _ => None,
                });
            let tail_span = match first_tail_wildcard {
                None => after_fields_span,
                Some(0) => subpats[0].span.to(after_fields_span),
                Some(pos) => subpats[pos - 1].span.shrink_to_hi().to(after_fields_span),
            };

            // FIXME: heuristic-based suggestion to check current types for where to add `_`.
            let mut wildcard_sugg = vec!["_"; fields.len() - subpats.len()].join(", ");
            if !subpats.is_empty() {
                wildcard_sugg = String::from(", ") + &wildcard_sugg;
            }

            err.span_suggestion_verbose(
                after_fields_span,
                "use `_` to explicitly ignore each field",
                wildcard_sugg,
                Applicability::MaybeIncorrect,
            );

            // Only suggest `..` if more than one field is missing
            // or the pattern consists of all wildcards.
            if fields.len() - subpats.len() > 1 || all_wildcards {
                if subpats.is_empty() || all_wildcards {
                    err.span_suggestion_verbose(
                        all_fields_span,
                        "use `..` to ignore all fields",
                        "..",
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_suggestion_verbose(
                        tail_span,
                        "use `..` to ignore the rest of the fields",
                        ", ..",
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }

        err.emit()
    }

    fn check_pat_tuple(
        &self,
        span: Span,
        elements: &'tcx [Pat<'tcx>],
        ddpos: hir::DotDotPos,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let mut expected_len = elements.len();
        if ddpos.as_opt_usize().is_some() {
            // Require known type only when `..` is present.
            if let ty::Tuple(tys) = self.structurally_resolve_type(span, expected).kind() {
                expected_len = tys.len();
            }
        }
        let max_len = cmp::max(expected_len, elements.len());

        let element_tys_iter = (0..max_len).map(|_| self.next_ty_var(span));
        let element_tys = tcx.mk_type_list_from_iter(element_tys_iter);
        let pat_ty = Ty::new_tup(tcx, element_tys);
        if let Err(reported) = self.demand_eqtype_pat(span, expected, pat_ty, &pat_info.top_info) {
            // Walk subpatterns with an expected type of `err` in this case to silence
            // further errors being emitted when using the bindings. #50333
            let element_tys_iter = (0..max_len).map(|_| Ty::new_error(tcx, reported));
            for (_, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.check_pat(elem, Ty::new_error(tcx, reported), pat_info);
            }
            Ty::new_tup_from_iter(tcx, element_tys_iter)
        } else {
            for (i, elem) in elements.iter().enumerate_and_adjust(max_len, ddpos) {
                self.check_pat(elem, element_tys[i], pat_info);
            }
            pat_ty
        }
    }

    fn check_struct_pat_fields(
        &self,
        adt_ty: Ty<'tcx>,
        pat: &'tcx Pat<'tcx>,
        variant: &'tcx ty::VariantDef,
        fields: &'tcx [hir::PatField<'tcx>],
        has_rest_pat: bool,
        pat_info: PatInfo<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        let tcx = self.tcx;

        let ty::Adt(adt, args) = adt_ty.kind() else {
            span_bug!(pat.span, "struct pattern is not an ADT");
        };

        // Index the struct fields' types.
        let field_map = variant
            .fields
            .iter_enumerated()
            .map(|(i, field)| (field.ident(self.tcx).normalize_to_macros_2_0(), (i, field)))
            .collect::<FxHashMap<_, _>>();

        // Keep track of which fields have already appeared in the pattern.
        let mut used_fields = FxHashMap::default();
        let mut result = Ok(());

        let mut inexistent_fields = vec![];
        // Typecheck each field.
        for field in fields {
            let span = field.span;
            let ident = tcx.adjust_ident(field.ident, variant.def_id);
            let field_ty = match used_fields.entry(ident) {
                Occupied(occupied) => {
                    let guar = self.error_field_already_bound(span, field.ident, *occupied.get());
                    result = Err(guar);
                    Ty::new_error(tcx, guar)
                }
                Vacant(vacant) => {
                    vacant.insert(span);
                    field_map
                        .get(&ident)
                        .map(|(i, f)| {
                            self.write_field_index(field.hir_id, *i);
                            self.tcx.check_stability(f.did, Some(field.hir_id), span, None);
                            self.field_ty(span, f, args)
                        })
                        .unwrap_or_else(|| {
                            inexistent_fields.push(field);
                            Ty::new_misc_error(tcx)
                        })
                }
            };

            self.check_pat(field.pat, field_ty, pat_info);
        }

        let mut unmentioned_fields = variant
            .fields
            .iter()
            .map(|field| (field, field.ident(self.tcx).normalize_to_macros_2_0()))
            .filter(|(_, ident)| !used_fields.contains_key(ident))
            .collect::<Vec<_>>();

        let inexistent_fields_err = if !inexistent_fields.is_empty()
            && !inexistent_fields.iter().any(|field| field.ident.name == kw::Underscore)
        {
            // we don't care to report errors for a struct if the struct itself is tainted
            variant.has_errors()?;
            Some(self.error_inexistent_fields(
                adt.variant_descr(),
                &inexistent_fields,
                &mut unmentioned_fields,
                pat,
                variant,
                args,
            ))
        } else {
            None
        };

        // Require `..` if struct has non_exhaustive attribute.
        let non_exhaustive = variant.field_list_has_applicable_non_exhaustive();
        if non_exhaustive && !has_rest_pat {
            self.error_foreign_non_exhaustive_spat(pat, adt.variant_descr(), fields.is_empty());
        }

        let mut unmentioned_err = None;
        // Report an error if an incorrect number of fields was specified.
        if adt.is_union() {
            if fields.len() != 1 {
                self.dcx().emit_err(errors::UnionPatMultipleFields { span: pat.span });
            }
            if has_rest_pat {
                self.dcx().emit_err(errors::UnionPatDotDot { span: pat.span });
            }
        } else if !unmentioned_fields.is_empty() {
            let accessible_unmentioned_fields: Vec<_> = unmentioned_fields
                .iter()
                .copied()
                .filter(|(field, _)| self.is_field_suggestable(field, pat.hir_id, pat.span))
                .collect();

            if !has_rest_pat {
                if accessible_unmentioned_fields.is_empty() {
                    unmentioned_err = Some(self.error_no_accessible_fields(pat, fields));
                } else {
                    unmentioned_err = Some(self.error_unmentioned_fields(
                        pat,
                        &accessible_unmentioned_fields,
                        accessible_unmentioned_fields.len() != unmentioned_fields.len(),
                        fields,
                    ));
                }
            } else if non_exhaustive && !accessible_unmentioned_fields.is_empty() {
                self.lint_non_exhaustive_omitted_patterns(
                    pat,
                    &accessible_unmentioned_fields,
                    adt_ty,
                )
            }
        }
        match (inexistent_fields_err, unmentioned_err) {
            (Some(i), Some(u)) => {
                if let Err(e) = self.error_tuple_variant_as_struct_pat(pat, fields, variant) {
                    // We don't want to show the nonexistent fields error when this was
                    // `Foo { a, b }` when it should have been `Foo(a, b)`.
                    i.delay_as_bug();
                    u.delay_as_bug();
                    Err(e)
                } else {
                    i.emit();
                    Err(u.emit())
                }
            }
            (None, Some(u)) => {
                if let Err(e) = self.error_tuple_variant_as_struct_pat(pat, fields, variant) {
                    u.delay_as_bug();
                    Err(e)
                } else {
                    Err(u.emit())
                }
            }
            (Some(err), None) => Err(err.emit()),
            (None, None) => {
                self.error_tuple_variant_index_shorthand(variant, pat, fields)?;
                result
            }
        }
    }

    fn error_tuple_variant_index_shorthand(
        &self,
        variant: &VariantDef,
        pat: &'_ Pat<'_>,
        fields: &[hir::PatField<'_>],
    ) -> Result<(), ErrorGuaranteed> {
        // if this is a tuple struct, then all field names will be numbers
        // so if any fields in a struct pattern use shorthand syntax, they will
        // be invalid identifiers (for example, Foo { 0, 1 }).
        if let (Some(CtorKind::Fn), PatKind::Struct(qpath, field_patterns, ..)) =
            (variant.ctor_kind(), &pat.kind)
        {
            let has_shorthand_field_name = field_patterns.iter().any(|field| field.is_shorthand);
            if has_shorthand_field_name {
                let path = rustc_hir_pretty::qpath_to_string(&self.tcx, qpath);
                let mut err = struct_span_code_err!(
                    self.dcx(),
                    pat.span,
                    E0769,
                    "tuple variant `{path}` written as struct variant",
                );
                err.span_suggestion_verbose(
                    qpath.span().shrink_to_hi().to(pat.span.shrink_to_hi()),
                    "use the tuple variant pattern syntax instead",
                    format!("({})", self.get_suggested_tuple_struct_pattern(fields, variant)),
                    Applicability::MaybeIncorrect,
                );
                return Err(err.emit());
            }
        }
        Ok(())
    }

    fn error_foreign_non_exhaustive_spat(&self, pat: &Pat<'_>, descr: &str, no_fields: bool) {
        let sess = self.tcx.sess;
        let sm = sess.source_map();
        let sp_brace = sm.end_point(pat.span);
        let sp_comma = sm.end_point(pat.span.with_hi(sp_brace.hi()));
        let sugg = if no_fields || sp_brace != sp_comma { ".. }" } else { ", .. }" };

        struct_span_code_err!(
            self.dcx(),
            pat.span,
            E0638,
            "`..` required with {descr} marked as non-exhaustive",
        )
        .with_span_suggestion_verbose(
            sp_comma,
            "add `..` at the end of the field list to ignore all other fields",
            sugg,
            Applicability::MachineApplicable,
        )
        .emit();
    }

    fn error_field_already_bound(
        &self,
        span: Span,
        ident: Ident,
        other_field: Span,
    ) -> ErrorGuaranteed {
        struct_span_code_err!(
            self.dcx(),
            span,
            E0025,
            "field `{}` bound multiple times in the pattern",
            ident
        )
        .with_span_label(span, format!("multiple uses of `{ident}` in pattern"))
        .with_span_label(other_field, format!("first use of `{ident}`"))
        .emit()
    }

    fn error_inexistent_fields(
        &self,
        kind_name: &str,
        inexistent_fields: &[&hir::PatField<'tcx>],
        unmentioned_fields: &mut Vec<(&'tcx ty::FieldDef, Ident)>,
        pat: &'tcx Pat<'tcx>,
        variant: &ty::VariantDef,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Diag<'a> {
        let tcx = self.tcx;
        let (field_names, t, plural) = if let [field] = inexistent_fields {
            (format!("a field named `{}`", field.ident), "this", "")
        } else {
            (
                format!(
                    "fields named {}",
                    inexistent_fields
                        .iter()
                        .map(|field| format!("`{}`", field.ident))
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
                "these",
                "s",
            )
        };
        let spans = inexistent_fields.iter().map(|field| field.ident.span).collect::<Vec<_>>();
        let mut err = struct_span_code_err!(
            self.dcx(),
            spans,
            E0026,
            "{} `{}` does not have {}",
            kind_name,
            tcx.def_path_str(variant.def_id),
            field_names
        );
        if let Some(pat_field) = inexistent_fields.last() {
            err.span_label(
                pat_field.ident.span,
                format!(
                    "{} `{}` does not have {} field{}",
                    kind_name,
                    tcx.def_path_str(variant.def_id),
                    t,
                    plural
                ),
            );

            if let [(field_def, field)] = unmentioned_fields.as_slice()
                && self.is_field_suggestable(field_def, pat.hir_id, pat.span)
            {
                let suggested_name =
                    find_best_match_for_name(&[field.name], pat_field.ident.name, None);
                if let Some(suggested_name) = suggested_name {
                    err.span_suggestion(
                        pat_field.ident.span,
                        "a field with a similar name exists",
                        suggested_name,
                        Applicability::MaybeIncorrect,
                    );

                    // When we have a tuple struct used with struct we don't want to suggest using
                    // the (valid) struct syntax with numeric field names. Instead we want to
                    // suggest the expected syntax. We infer that this is the case by parsing the
                    // `Ident` into an unsized integer. The suggestion will be emitted elsewhere in
                    // `smart_resolve_context_dependent_help`.
                    if suggested_name.to_ident_string().parse::<usize>().is_err() {
                        // We don't want to throw `E0027` in case we have thrown `E0026` for them.
                        unmentioned_fields.retain(|&(_, x)| x.name != suggested_name);
                    }
                } else if inexistent_fields.len() == 1 {
                    match pat_field.pat.kind {
                        PatKind::Expr(_)
                            if !self.may_coerce(
                                self.typeck_results.borrow().node_type(pat_field.pat.hir_id),
                                self.field_ty(field.span, field_def, args),
                            ) => {}
                        _ => {
                            err.span_suggestion_short(
                                pat_field.ident.span,
                                format!(
                                    "`{}` has a field named `{}`",
                                    tcx.def_path_str(variant.def_id),
                                    field.name,
                                ),
                                field.name,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                }
            }
        }
        if tcx.sess.teach(err.code.unwrap()) {
            err.note(
                "This error indicates that a struct pattern attempted to \
                 extract a nonexistent field from a struct. Struct fields \
                 are identified by the name used before the colon : so struct \
                 patterns should resemble the declaration of the struct type \
                 being matched.\n\n\
                 If you are using shorthand field patterns but want to refer \
                 to the struct field by a different name, you should rename \
                 it explicitly.",
            );
        }
        err
    }

    fn error_tuple_variant_as_struct_pat(
        &self,
        pat: &Pat<'_>,
        fields: &'tcx [hir::PatField<'tcx>],
        variant: &ty::VariantDef,
    ) -> Result<(), ErrorGuaranteed> {
        if let (Some(CtorKind::Fn), PatKind::Struct(qpath, pattern_fields, ..)) =
            (variant.ctor_kind(), &pat.kind)
        {
            let is_tuple_struct_match = !pattern_fields.is_empty()
                && pattern_fields.iter().map(|field| field.ident.name.as_str()).all(is_number);
            if is_tuple_struct_match {
                return Ok(());
            }

            // we don't care to report errors for a struct if the struct itself is tainted
            variant.has_errors()?;

            let path = rustc_hir_pretty::qpath_to_string(&self.tcx, qpath);
            let mut err = struct_span_code_err!(
                self.dcx(),
                pat.span,
                E0769,
                "tuple variant `{}` written as struct variant",
                path
            );
            let (sugg, appl) = if fields.len() == variant.fields.len() {
                (
                    self.get_suggested_tuple_struct_pattern(fields, variant),
                    Applicability::MachineApplicable,
                )
            } else {
                (
                    variant.fields.iter().map(|_| "_").collect::<Vec<&str>>().join(", "),
                    Applicability::MaybeIncorrect,
                )
            };
            err.span_suggestion_verbose(
                qpath.span().shrink_to_hi().to(pat.span.shrink_to_hi()),
                "use the tuple variant pattern syntax instead",
                format!("({sugg})"),
                appl,
            );
            return Err(err.emit());
        }
        Ok(())
    }

    fn get_suggested_tuple_struct_pattern(
        &self,
        fields: &[hir::PatField<'_>],
        variant: &VariantDef,
    ) -> String {
        let variant_field_idents =
            variant.fields.iter().map(|f| f.ident(self.tcx)).collect::<Vec<Ident>>();
        fields
            .iter()
            .map(|field| {
                match self.tcx.sess.source_map().span_to_snippet(field.pat.span) {
                    Ok(f) => {
                        // Field names are numbers, but numbers
                        // are not valid identifiers
                        if variant_field_idents.contains(&field.ident) {
                            String::from("_")
                        } else {
                            f
                        }
                    }
                    Err(_) => rustc_hir_pretty::pat_to_string(&self.tcx, field.pat),
                }
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Returns a diagnostic reporting a struct pattern which is missing an `..` due to
    /// inaccessible fields.
    ///
    /// ```text
    /// error: pattern requires `..` due to inaccessible fields
    ///   --> src/main.rs:10:9
    ///    |
    /// LL |     let foo::Foo {} = foo::Foo::default();
    ///    |         ^^^^^^^^^^^
    ///    |
    /// help: add a `..`
    ///    |
    /// LL |     let foo::Foo { .. } = foo::Foo::default();
    ///    |                  ^^^^^^
    /// ```
    fn error_no_accessible_fields(
        &self,
        pat: &Pat<'_>,
        fields: &'tcx [hir::PatField<'tcx>],
    ) -> Diag<'a> {
        let mut err = self
            .dcx()
            .struct_span_err(pat.span, "pattern requires `..` due to inaccessible fields");

        if let Some(field) = fields.last() {
            err.span_suggestion_verbose(
                field.span.shrink_to_hi(),
                "ignore the inaccessible and unused fields",
                ", ..",
                Applicability::MachineApplicable,
            );
        } else {
            let qpath_span = if let PatKind::Struct(qpath, ..) = &pat.kind {
                qpath.span()
            } else {
                bug!("`error_no_accessible_fields` called on non-struct pattern");
            };

            // Shrink the span to exclude the `foo:Foo` in `foo::Foo { }`.
            let span = pat.span.with_lo(qpath_span.shrink_to_hi().hi());
            err.span_suggestion_verbose(
                span,
                "ignore the inaccessible and unused fields",
                " { .. }",
                Applicability::MachineApplicable,
            );
        }
        err
    }

    /// Report that a pattern for a `#[non_exhaustive]` struct marked with `non_exhaustive_omitted_patterns`
    /// is not exhaustive enough.
    ///
    /// Nb: the partner lint for enums lives in `compiler/rustc_mir_build/src/thir/pattern/usefulness.rs`.
    fn lint_non_exhaustive_omitted_patterns(
        &self,
        pat: &Pat<'_>,
        unmentioned_fields: &[(&ty::FieldDef, Ident)],
        ty: Ty<'tcx>,
    ) {
        fn joined_uncovered_patterns(witnesses: &[&Ident]) -> String {
            const LIMIT: usize = 3;
            match witnesses {
                [] => {
                    unreachable!(
                        "expected an uncovered pattern, otherwise why are we emitting an error?"
                    )
                }
                [witness] => format!("`{witness}`"),
                [head @ .., tail] if head.len() < LIMIT => {
                    let head: Vec<_> = head.iter().map(<_>::to_string).collect();
                    format!("`{}` and `{}`", head.join("`, `"), tail)
                }
                _ => {
                    let (head, tail) = witnesses.split_at(LIMIT);
                    let head: Vec<_> = head.iter().map(<_>::to_string).collect();
                    format!("`{}` and {} more", head.join("`, `"), tail.len())
                }
            }
        }
        let joined_patterns = joined_uncovered_patterns(
            &unmentioned_fields.iter().map(|(_, i)| i).collect::<Vec<_>>(),
        );

        self.tcx.node_span_lint(NON_EXHAUSTIVE_OMITTED_PATTERNS, pat.hir_id, pat.span, |lint| {
            lint.primary_message("some fields are not explicitly listed");
            lint.span_label(pat.span, format!("field{} {} not listed", rustc_errors::pluralize!(unmentioned_fields.len()), joined_patterns));
            lint.help(
                "ensure that all fields are mentioned explicitly by adding the suggested fields",
            );
            lint.note(format!(
                "the pattern is of type `{ty}` and the `non_exhaustive_omitted_patterns` attribute was found",
            ));
        });
    }

    /// Returns a diagnostic reporting a struct pattern which does not mention some fields.
    ///
    /// ```text
    /// error[E0027]: pattern does not mention field `bar`
    ///   --> src/main.rs:15:9
    ///    |
    /// LL |     let foo::Foo {} = foo::Foo::new();
    ///    |         ^^^^^^^^^^^ missing field `bar`
    /// ```
    fn error_unmentioned_fields(
        &self,
        pat: &Pat<'_>,
        unmentioned_fields: &[(&ty::FieldDef, Ident)],
        have_inaccessible_fields: bool,
        fields: &'tcx [hir::PatField<'tcx>],
    ) -> Diag<'a> {
        let inaccessible = if have_inaccessible_fields { " and inaccessible fields" } else { "" };
        let field_names = if let [(_, field)] = unmentioned_fields {
            format!("field `{field}`{inaccessible}")
        } else {
            let fields = unmentioned_fields
                .iter()
                .map(|(_, name)| format!("`{name}`"))
                .collect::<Vec<String>>()
                .join(", ");
            format!("fields {fields}{inaccessible}")
        };
        let mut err = struct_span_code_err!(
            self.dcx(),
            pat.span,
            E0027,
            "pattern does not mention {}",
            field_names
        );
        err.span_label(pat.span, format!("missing {field_names}"));
        let len = unmentioned_fields.len();
        let (prefix, postfix, sp) = match fields {
            [] => match &pat.kind {
                PatKind::Struct(path, [], false) => {
                    (" { ", " }", path.span().shrink_to_hi().until(pat.span.shrink_to_hi()))
                }
                _ => return err,
            },
            [.., field] => {
                // Account for last field having a trailing comma or parse recovery at the tail of
                // the pattern to avoid invalid suggestion (#78511).
                let tail = field.span.shrink_to_hi().with_hi(pat.span.hi());
                match &pat.kind {
                    PatKind::Struct(..) => (", ", " }", tail),
                    _ => return err,
                }
            }
        };
        err.span_suggestion(
            sp,
            format!(
                "include the missing field{} in the pattern{}",
                pluralize!(len),
                if have_inaccessible_fields { " and ignore the inaccessible fields" } else { "" }
            ),
            format!(
                "{}{}{}{}",
                prefix,
                unmentioned_fields
                    .iter()
                    .map(|(_, name)| {
                        let field_name = name.to_string();
                        if is_number(&field_name) { format!("{field_name}: _") } else { field_name }
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                if have_inaccessible_fields { ", .." } else { "" },
                postfix,
            ),
            Applicability::MachineApplicable,
        );
        err.span_suggestion(
            sp,
            format!(
                "if you don't care about {these} missing field{s}, you can explicitly ignore {them}",
                these = pluralize!("this", len),
                s = pluralize!(len),
                them = if len == 1 { "it" } else { "them" },
            ),
            format!(
                "{}{}{}{}",
                prefix,
                unmentioned_fields
                    .iter()
                    .map(|(_, name)| {
                        let field_name = name.to_string();
                        format!("{field_name}: _")
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                if have_inaccessible_fields { ", .." } else { "" },
                postfix,
            ),
            Applicability::MachineApplicable,
        );
        err.span_suggestion(
            sp,
            "or always ignore missing fields here",
            format!("{prefix}..{postfix}"),
            Applicability::MachineApplicable,
        );
        err
    }

    fn check_pat_box(
        &self,
        span: Span,
        inner: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let (box_ty, inner_ty) = self
            .check_dereferenceable(span, expected, inner)
            .and_then(|()| {
                // Here, `demand::subtype` is good enough, but I don't
                // think any errors can be introduced by using `demand::eqtype`.
                let inner_ty = self.next_ty_var(inner.span);
                let box_ty = Ty::new_box(tcx, inner_ty);
                self.demand_eqtype_pat(span, expected, box_ty, &pat_info.top_info)?;
                Ok((box_ty, inner_ty))
            })
            .unwrap_or_else(|guar| {
                let err = Ty::new_error(tcx, guar);
                (err, err)
            });
        self.check_pat(inner, inner_ty, pat_info);
        box_ty
    }

    fn check_pat_deref(
        &self,
        span: Span,
        inner: &'tcx Pat<'tcx>,
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let target_ty = self.deref_pat_target(span, expected);
        self.check_pat(inner, target_ty, pat_info);
        self.register_deref_mut_bounds_if_needed(span, inner, [expected]);
        expected
    }

    fn deref_pat_target(&self, span: Span, source_ty: Ty<'tcx>) -> Ty<'tcx> {
        // Register a `DerefPure` bound, which is required by all `deref!()` pats.
        let tcx = self.tcx;
        self.register_bound(
            source_ty,
            tcx.require_lang_item(hir::LangItem::DerefPure, span),
            self.misc(span),
        );
        // The expected type for the deref pat's inner pattern is `<expected as Deref>::Target`.
        let target_ty = Ty::new_projection(
            tcx,
            tcx.require_lang_item(hir::LangItem::DerefTarget, span),
            [source_ty],
        );
        let target_ty = self.normalize(span, target_ty);
        self.try_structurally_resolve_type(span, target_ty)
    }

    /// Check if the interior of a deref pattern (either explicit or implicit) has any `ref mut`
    /// bindings, which would require `DerefMut` to be emitted in MIR building instead of just
    /// `Deref`. We do this *after* checking the inner pattern, since we want to make sure to
    /// account for `ref mut` binding modes inherited from implicitly dereferencing `&mut` refs.
    fn register_deref_mut_bounds_if_needed(
        &self,
        span: Span,
        inner: &'tcx Pat<'tcx>,
        derefed_tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) {
        if self.typeck_results.borrow().pat_has_ref_mut_binding(inner) {
            for mutably_derefed_ty in derefed_tys {
                self.register_bound(
                    mutably_derefed_ty,
                    self.tcx.require_lang_item(hir::LangItem::DerefMut, span),
                    self.misc(span),
                );
            }
        }
    }

    // Precondition: Pat is Ref(inner)
    fn check_pat_ref(
        &self,
        pat: &'tcx Pat<'tcx>,
        inner: &'tcx Pat<'tcx>,
        pat_mutbl: Mutability,
        mut expected: Ty<'tcx>,
        mut pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        let pat_prefix_span =
            inner.span.find_ancestor_inside(pat.span).map(|end| pat.span.until(end));

        let ref_pat_matches_mut_ref = self.ref_pat_matches_mut_ref();
        if ref_pat_matches_mut_ref && pat_mutbl == Mutability::Not {
            // If `&` patterns can match against mutable reference types (RFC 3627, Rule 5), we need
            // to prevent subpatterns from binding with `ref mut`. Subpatterns of a shared reference
            // pattern should have read-only access to the scrutinee, and the borrow checker won't
            // catch it in this case.
            pat_info.max_ref_mutbl = pat_info.max_ref_mutbl.cap_to_weakly_not(pat_prefix_span);
        }

        expected = self.try_structurally_resolve_type(pat.span, expected);
        // Determine whether we're consuming an inherited reference and resetting the default
        // binding mode, based on edition and enabled experimental features.
        if let ByRef::Yes(inh_mut) = pat_info.binding_mode {
            match self.ref_pat_matches_inherited_ref(pat.span.edition()) {
                InheritedRefMatchRule::EatOuter => {
                    // ref pattern attempts to consume inherited reference
                    if pat_mutbl > inh_mut {
                        // Tried to match inherited `ref` with `&mut`
                        // NB: This assumes that `&` patterns can match against mutable references
                        // (RFC 3627, Rule 5). If we implement a pattern typing ruleset with Rule 4E
                        // but not Rule 5, we'll need to check that here.
                        debug_assert!(ref_pat_matches_mut_ref);
                        self.error_inherited_ref_mutability_mismatch(pat, pat_prefix_span);
                    }

                    pat_info.binding_mode = ByRef::No;
                    self.typeck_results.borrow_mut().skipped_ref_pats_mut().insert(pat.hir_id);
                    self.check_pat(inner, expected, pat_info);
                    return expected;
                }
                InheritedRefMatchRule::EatInner => {
                    if let ty::Ref(_, _, r_mutbl) = *expected.kind()
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
                            self.error_inherited_ref_mutability_mismatch(pat, pat_prefix_span);
                        }

                        pat_info.binding_mode = ByRef::No;
                        self.typeck_results.borrow_mut().skipped_ref_pats_mut().insert(pat.hir_id);
                        self.check_pat(inner, expected, pat_info);
                        return expected;
                    }
                }
                InheritedRefMatchRule::EatBoth { consider_inherited_ref: true } => {
                    // Reset binding mode on old editions
                    pat_info.binding_mode = ByRef::No;

                    if let ty::Ref(_, inner_ty, _) = *expected.kind() {
                        // Consume both the inherited and inner references.
                        if pat_mutbl.is_mut() && inh_mut.is_mut() {
                            // As a special case, a `&mut` reference pattern will be able to match
                            // against a reference type of any mutability if the inherited ref is
                            // mutable. Since this allows us to match against a shared reference
                            // type, we refer to this as "falling back" to matching the inherited
                            // reference, though we consume the real reference as well. We handle
                            // this here to avoid adding this case to the common logic below.
                            self.check_pat(inner, inner_ty, pat_info);
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
                            self.error_inherited_ref_mutability_mismatch(pat, pat_prefix_span);
                        }

                        self.typeck_results.borrow_mut().skipped_ref_pats_mut().insert(pat.hir_id);
                        self.check_pat(inner, expected, pat_info);
                        return expected;
                    }
                }
                InheritedRefMatchRule::EatBoth { consider_inherited_ref: false } => {
                    // Reset binding mode on stable Rust. This will be a type error below if
                    // `expected` is not a reference type.
                    pat_info.binding_mode = ByRef::No;
                    self.add_rust_2024_migration_desugared_pat(
                        pat_info.top_info.hir_id,
                        pat,
                        match pat_mutbl {
                            Mutability::Not => '&', // last char of `&`
                            Mutability::Mut => 't', // last char of `&mut`
                        },
                        inh_mut,
                    )
                }
            }
        }

        let (ref_ty, inner_ty) = match self.check_dereferenceable(pat.span, expected, inner) {
            Ok(()) => {
                // `demand::subtype` would be good enough, but using `eqtype` turns
                // out to be equally general. See (note_1) for details.

                // Take region, inner-type from expected type if we can,
                // to avoid creating needless variables. This also helps with
                // the bad interactions of the given hack detailed in (note_1).
                debug!("check_pat_ref: expected={:?}", expected);
                match *expected.kind() {
                    ty::Ref(_, r_ty, r_mutbl)
                        if (ref_pat_matches_mut_ref && r_mutbl >= pat_mutbl)
                            || r_mutbl == pat_mutbl =>
                    {
                        if r_mutbl == Mutability::Not {
                            pat_info.max_ref_mutbl = MutblCap::Not;
                        }

                        (expected, r_ty)
                    }

                    _ => {
                        let inner_ty = self.next_ty_var(inner.span);
                        let ref_ty = self.new_ref_ty(pat.span, pat_mutbl, inner_ty);
                        debug!("check_pat_ref: demanding {:?} = {:?}", expected, ref_ty);
                        let err = self.demand_eqtype_pat_diag(
                            pat.span,
                            expected,
                            ref_ty,
                            &pat_info.top_info,
                        );

                        // Look for a case like `fn foo(&foo: u32)` and suggest
                        // `fn foo(foo: &u32)`
                        if let Err(mut err) = err {
                            self.borrow_pat_suggestion(&mut err, pat);
                            err.emit();
                        }
                        (ref_ty, inner_ty)
                    }
                }
            }
            Err(guar) => {
                let err = Ty::new_error(tcx, guar);
                (err, err)
            }
        };

        self.check_pat(inner, inner_ty, pat_info);
        ref_ty
    }

    /// Create a reference type with a fresh region variable.
    fn new_ref_ty(&self, span: Span, mutbl: Mutability, ty: Ty<'tcx>) -> Ty<'tcx> {
        let region = self.next_region_var(infer::PatternRegion(span));
        Ty::new_ref(self.tcx, region, ty, mutbl)
    }

    fn error_inherited_ref_mutability_mismatch(
        &self,
        pat: &'tcx Pat<'tcx>,
        pat_prefix_span: Option<Span>,
    ) -> ErrorGuaranteed {
        let err_msg = "mismatched types";
        let err = if let Some(span) = pat_prefix_span {
            let mut err = self.dcx().struct_span_err(span, err_msg);
            err.code(E0308);
            err.note("cannot match inherited `&` with `&mut` pattern");
            err.span_suggestion_verbose(
                span,
                "replace this `&mut` pattern with `&`",
                "&",
                Applicability::MachineApplicable,
            );
            err
        } else {
            self.dcx().struct_span_err(pat.span, err_msg)
        };
        err.emit()
    }

    fn try_resolve_slice_ty_to_array_ty(
        &self,
        before: &'tcx [Pat<'tcx>],
        slice: Option<&'tcx Pat<'tcx>>,
        span: Span,
    ) -> Option<Ty<'tcx>> {
        if slice.is_some() {
            return None;
        }

        let tcx = self.tcx;
        let len = before.len();
        let inner_ty = self.next_ty_var(span);

        Some(Ty::new_array(tcx, inner_ty, len.try_into().unwrap()))
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
    fn pat_is_irrefutable(&self, decl_origin: Option<DeclOrigin<'_>>) -> bool {
        match decl_origin {
            Some(DeclOrigin::LocalDecl { els: None }) => true,
            Some(DeclOrigin::LocalDecl { els: Some(_) } | DeclOrigin::LetExpr) | None => false,
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
    fn check_pat_slice(
        &self,
        span: Span,
        before: &'tcx [Pat<'tcx>],
        slice: Option<&'tcx Pat<'tcx>>,
        after: &'tcx [Pat<'tcx>],
        expected: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> Ty<'tcx> {
        let expected = self.try_structurally_resolve_type(span, expected);

        // If the pattern is irrefutable and `expected` is an infer ty, we try to equate it
        // to an array if the given pattern allows it. See issue #76342
        if self.pat_is_irrefutable(pat_info.decl_origin) && expected.is_ty_var() {
            if let Some(resolved_arr_ty) =
                self.try_resolve_slice_ty_to_array_ty(before, slice, span)
            {
                debug!(?resolved_arr_ty);
                let _ = self.demand_eqtype(span, expected, resolved_arr_ty);
            }
        }

        let expected = self.structurally_resolve_type(span, expected);
        debug!(?expected);

        let (element_ty, opt_slice_ty, inferred) = match *expected.kind() {
            // An array, so we might have something like `let [a, b, c] = [0, 1, 2];`.
            ty::Array(element_ty, len) => {
                let min = before.len() as u64 + after.len() as u64;
                let (opt_slice_ty, expected) =
                    self.check_array_pat_len(span, element_ty, expected, slice, len, min);
                // `opt_slice_ty.is_none()` => `slice.is_none()`.
                // Note, though, that opt_slice_ty could be `Some(error_ty)`.
                assert!(opt_slice_ty.is_some() || slice.is_none());
                (element_ty, opt_slice_ty, expected)
            }
            ty::Slice(element_ty) => (element_ty, Some(expected), expected),
            // The expected type must be an array or slice, but was neither, so error.
            _ => {
                let guar = expected.error_reported().err().unwrap_or_else(|| {
                    self.error_expected_array_or_slice(span, expected, pat_info)
                });
                let err = Ty::new_error(self.tcx, guar);
                (err, Some(err), err)
            }
        };

        // Type check all the patterns before `slice`.
        for elt in before {
            self.check_pat(elt, element_ty, pat_info);
        }
        // Type check the `slice`, if present, against its expected type.
        if let Some(slice) = slice {
            self.check_pat(slice, opt_slice_ty.unwrap(), pat_info);
        }
        // Type check the elements after `slice`, if present.
        for elt in after {
            self.check_pat(elt, element_ty, pat_info);
        }
        inferred
    }

    /// Type check the length of an array pattern.
    ///
    /// Returns both the type of the variable length pattern (or `None`), and the potentially
    /// inferred array type. We only return `None` for the slice type if `slice.is_none()`.
    fn check_array_pat_len(
        &self,
        span: Span,
        element_ty: Ty<'tcx>,
        arr_ty: Ty<'tcx>,
        slice: Option<&'tcx Pat<'tcx>>,
        len: ty::Const<'tcx>,
        min_len: u64,
    ) -> (Option<Ty<'tcx>>, Ty<'tcx>) {
        let len = self.try_structurally_resolve_const(span, len).try_to_target_usize(self.tcx);

        let guar = if let Some(len) = len {
            // Now we know the length...
            if slice.is_none() {
                // ...and since there is no variable-length pattern,
                // we require an exact match between the number of elements
                // in the array pattern and as provided by the matched type.
                if min_len == len {
                    return (None, arr_ty);
                }

                self.error_scrutinee_inconsistent_length(span, min_len, len)
            } else if let Some(pat_len) = len.checked_sub(min_len) {
                // The variable-length pattern was there,
                // so it has an array type with the remaining elements left as its size...
                return (Some(Ty::new_array(self.tcx, element_ty, pat_len)), arr_ty);
            } else {
                // ...however, in this case, there were no remaining elements.
                // That is, the slice pattern requires more than the array type offers.
                self.error_scrutinee_with_rest_inconsistent_length(span, min_len, len)
            }
        } else if slice.is_none() {
            // We have a pattern with a fixed length,
            // which we can use to infer the length of the array.
            let updated_arr_ty = Ty::new_array(self.tcx, element_ty, min_len);
            self.demand_eqtype(span, updated_arr_ty, arr_ty);
            return (None, updated_arr_ty);
        } else {
            // We have a variable-length pattern and don't know the array length.
            // This happens if we have e.g.,
            // `let [a, b, ..] = arr` where `arr: [T; N]` where `const N: usize`.
            self.error_scrutinee_unfixed_length(span)
        };

        // If we get here, we must have emitted an error.
        (Some(Ty::new_error(self.tcx, guar)), arr_ty)
    }

    fn error_scrutinee_inconsistent_length(
        &self,
        span: Span,
        min_len: u64,
        size: u64,
    ) -> ErrorGuaranteed {
        struct_span_code_err!(
            self.dcx(),
            span,
            E0527,
            "pattern requires {} element{} but array has {}",
            min_len,
            pluralize!(min_len),
            size,
        )
        .with_span_label(span, format!("expected {} element{}", size, pluralize!(size)))
        .emit()
    }

    fn error_scrutinee_with_rest_inconsistent_length(
        &self,
        span: Span,
        min_len: u64,
        size: u64,
    ) -> ErrorGuaranteed {
        struct_span_code_err!(
            self.dcx(),
            span,
            E0528,
            "pattern requires at least {} element{} but array has {}",
            min_len,
            pluralize!(min_len),
            size,
        )
        .with_span_label(
            span,
            format!("pattern cannot match array of {} element{}", size, pluralize!(size),),
        )
        .emit()
    }

    fn error_scrutinee_unfixed_length(&self, span: Span) -> ErrorGuaranteed {
        struct_span_code_err!(
            self.dcx(),
            span,
            E0730,
            "cannot pattern-match on an array without a fixed length",
        )
        .emit()
    }

    fn error_expected_array_or_slice(
        &self,
        span: Span,
        expected_ty: Ty<'tcx>,
        pat_info: PatInfo<'tcx>,
    ) -> ErrorGuaranteed {
        let PatInfo { top_info: ti, current_depth, .. } = pat_info;

        let mut slice_pat_semantics = false;
        let mut as_deref = None;
        let mut slicing = None;
        if let ty::Ref(_, ty, _) = expected_ty.kind()
            && let ty::Array(..) | ty::Slice(..) = ty.kind()
        {
            slice_pat_semantics = true;
        } else if self
            .autoderef(span, expected_ty)
            .silence_errors()
            .any(|(ty, _)| matches!(ty.kind(), ty::Slice(..) | ty::Array(..)))
            && let Some(span) = ti.span
            && let Some(_) = ti.origin_expr
        {
            let resolved_ty = self.resolve_vars_if_possible(ti.expected);
            let (is_slice_or_array_or_vector, resolved_ty) =
                self.is_slice_or_array_or_vector(resolved_ty);
            match resolved_ty.kind() {
                ty::Adt(adt_def, _)
                    if self.tcx.is_diagnostic_item(sym::Option, adt_def.did())
                        || self.tcx.is_diagnostic_item(sym::Result, adt_def.did()) =>
                {
                    // Slicing won't work here, but `.as_deref()` might (issue #91328).
                    as_deref = Some(errors::AsDerefSuggestion { span: span.shrink_to_hi() });
                }
                _ => (),
            }

            let is_top_level = current_depth <= 1;
            if is_slice_or_array_or_vector && is_top_level {
                slicing = Some(errors::SlicingSuggestion { span: span.shrink_to_hi() });
            }
        }
        self.dcx().emit_err(errors::ExpectedArrayOrSlice {
            span,
            ty: expected_ty,
            slice_pat_semantics,
            as_deref,
            slicing,
        })
    }

    fn is_slice_or_array_or_vector(&self, ty: Ty<'tcx>) -> (bool, Ty<'tcx>) {
        match ty.kind() {
            ty::Adt(adt_def, _) if self.tcx.is_diagnostic_item(sym::Vec, adt_def.did()) => {
                (true, ty)
            }
            ty::Ref(_, ty, _) => self.is_slice_or_array_or_vector(*ty),
            ty::Slice(..) | ty::Array(..) => (true, ty),
            _ => (false, ty),
        }
    }

    /// Record a pattern that's invalid under Rust 2024 match ergonomics, along with a problematic
    /// span, so that the pattern migration lint can desugar it during THIR construction.
    fn add_rust_2024_migration_desugared_pat(
        &self,
        pat_id: HirId,
        subpat: &'tcx Pat<'tcx>,
        final_char: char,
        def_br_mutbl: Mutability,
    ) {
        // Try to trim the span we're labeling to just the `&` or binding mode that's an issue.
        let from_expansion = subpat.span.from_expansion();
        let trimmed_span = if from_expansion {
            // If the subpattern is from an expansion, highlight the whole macro call instead.
            subpat.span
        } else {
            let trimmed = self.tcx.sess.source_map().span_through_char(subpat.span, final_char);
            // The edition of the trimmed span should be the same as `subpat.span`; this will be a
            // a hard error if the subpattern is of edition >= 2024. We set it manually to be sure:
            trimmed.with_ctxt(subpat.span.ctxt())
        };

        let mut typeck_results = self.typeck_results.borrow_mut();
        let mut table = typeck_results.rust_2024_migration_desugared_pats_mut();
        // FIXME(ref_pat_eat_one_layer_2024): The migration diagnostic doesn't know how to track the
        // default binding mode in the presence of Rule 3 or Rule 5. As a consequence, the labels it
        // gives for default binding modes are wrong, as well as suggestions based on the default
        // binding mode. This keeps it from making those suggestions, as doing so could panic.
        let info = table.entry(pat_id).or_insert_with(|| ty::Rust2024IncompatiblePatInfo {
            primary_labels: Vec::new(),
            bad_modifiers: false,
            bad_ref_pats: false,
            suggest_eliding_modes: !self.tcx.features().ref_pat_eat_one_layer_2024()
                && !self.tcx.features().ref_pat_eat_one_layer_2024_structural(),
        });

        let pat_kind = if let PatKind::Binding(user_bind_annot, _, _, _) = subpat.kind {
            info.bad_modifiers = true;
            // If the user-provided binding modifier doesn't match the default binding mode, we'll
            // need to suggest reference patterns, which can affect other bindings.
            // For simplicity, we opt to suggest making the pattern fully explicit.
            info.suggest_eliding_modes &=
                user_bind_annot == BindingMode(ByRef::Yes(def_br_mutbl), Mutability::Not);
            "binding modifier"
        } else {
            info.bad_ref_pats = true;
            // For simplicity, we don't try to suggest eliding reference patterns. Thus, we'll
            // suggest adding them instead, which can affect the types assigned to bindings.
            // As such, we opt to suggest making the pattern fully explicit.
            info.suggest_eliding_modes = false;
            "reference pattern"
        };
        // Only provide a detailed label if the problematic subpattern isn't from an expansion.
        // In the case that it's from a macro, we'll add a more detailed note in the emitter.
        let primary_label = if from_expansion {
            // We can't suggest eliding modifiers within expansions.
            info.suggest_eliding_modes = false;
            // NB: This wording assumes the only expansions that can produce problematic reference
            // patterns and bindings are macros. If a desugaring or AST pass is added that can do
            // so, we may want to inspect the span's source callee or macro backtrace.
            "occurs within macro expansion".to_owned()
        } else {
            let dbm_str = match def_br_mutbl {
                Mutability::Not => "ref",
                Mutability::Mut => "ref mut",
            };
            format!("{pat_kind} not allowed under `{dbm_str}` default binding mode")
        };
        info.primary_labels.push((trimmed_span, primary_label));
    }
}
