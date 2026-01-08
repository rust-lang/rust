//! Validation of patterns/matches.

mod check_match;
mod const_to_pat;
mod migration;

use std::assert_matches::assert_matches;
use std::cmp::Ordering;
use std::sync::Arc;

use rustc_abi::{FieldIdx, Integer};
use rustc_errors::codes::*;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::{self as hir, RangeEnd};
use rustc_index::Idx;
use rustc_middle::mir::interpret::LitToConstInput;
use rustc_middle::thir::{
    Ascription, DerefPatBorrowMode, FieldPat, LocalVarId, Pat, PatKind, PatRange, PatRangeBoundary,
};
use rustc_middle::ty::adjustment::{PatAdjust, PatAdjustment};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::ErrorGuaranteed;
use tracing::{debug, instrument};

pub(crate) use self::check_match::check_match;
use self::migration::PatMigration;
use crate::errors::*;

/// Context for lowering HIR patterns to THIR patterns.
struct PatCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,

    /// Used by the Rust 2024 migration lint.
    rust_2024_migration: Option<PatMigration<'tcx>>,
}

#[instrument(level = "debug", skip(tcx, typing_env, typeck_results), ret)]
pub(super) fn pat_from_hir<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
    pat: &'tcx hir::Pat<'tcx>,
    // Present if `pat` came from a let statement with an explicit type annotation
    let_stmt_type: Option<&hir::Ty<'tcx>>,
) -> Box<Pat<'tcx>> {
    let mut pcx = PatCtxt {
        tcx,
        typing_env,
        typeck_results,
        rust_2024_migration: typeck_results
            .rust_2024_migration_desugared_pats()
            .get(pat.hir_id)
            .map(PatMigration::new),
    };

    let mut thir_pat = pcx.lower_pattern(pat);

    // If this pattern came from a let statement with an explicit type annotation
    // (e.g. `let x: Foo = ...`), retain that user type information in the THIR pattern.
    if let Some(let_stmt_type) = let_stmt_type
        && let Some(&user_ty) = typeck_results.user_provided_types().get(let_stmt_type.hir_id)
    {
        debug!(?user_ty);
        let annotation = CanonicalUserTypeAnnotation {
            user_ty: Box::new(user_ty),
            span: let_stmt_type.span,
            inferred_ty: typeck_results.node_type(let_stmt_type.hir_id),
        };
        thir_pat = Box::new(Pat {
            ty: thir_pat.ty,
            span: thir_pat.span,
            kind: PatKind::AscribeUserType {
                ascription: Ascription { annotation, variance: ty::Covariant },
                subpattern: thir_pat,
            },
        });
    }

    if let Some(m) = pcx.rust_2024_migration {
        m.emit(tcx, pat.hir_id);
    }

    thir_pat
}

impl<'tcx> PatCtxt<'tcx> {
    fn lower_pattern(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Box<Pat<'tcx>> {
        let adjustments: &[PatAdjustment<'tcx>] =
            self.typeck_results.pat_adjustments().get(pat.hir_id).map_or(&[], |v| &**v);

        // Track the default binding mode for the Rust 2024 migration suggestion.
        // Implicitly dereferencing references changes the default binding mode, but implicit deref
        // patterns do not. Only track binding mode changes if a ref type is in the adjustments.
        let mut opt_old_mode_span = None;
        if let Some(s) = &mut self.rust_2024_migration
            && adjustments.iter().any(|adjust| adjust.kind == PatAdjust::BuiltinDeref)
        {
            opt_old_mode_span = s.visit_implicit_derefs(pat.span, adjustments);
        }

        // When implicit dereferences have been inserted in this pattern, the unadjusted lowered
        // pattern has the type that results *after* dereferencing. For example, in this code:
        //
        // ```
        // match &&Some(0i32) {
        //     Some(n) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // the type assigned to `Some(n)` in `unadjusted_pat` would be `Option<i32>` (this is
        // determined in rustc_hir_analysis::check::match). The adjustments would be
        //
        // `vec![&&Option<i32>, &Option<i32>]`.
        //
        // Applying the adjustments, we want to instead output `&&Some(n)` (as a THIR pattern). So
        // we wrap the unadjusted pattern in `PatKind::Deref` repeatedly, consuming the
        // adjustments in *reverse order* (last-in-first-out, so that the last `Deref` inserted
        // gets the least-dereferenced type).
        let unadjusted_pat = match pat.kind {
            hir::PatKind::Ref(inner, _, _)
                if self.typeck_results.skipped_ref_pats().contains(pat.hir_id) =>
            {
                self.lower_pattern(inner)
            }
            _ => self.lower_pattern_unadjusted(pat),
        };

        let adjusted_pat = adjustments.iter().rev().fold(unadjusted_pat, |thir_pat, adjust| {
            debug!("{:?}: wrapping pattern with adjustment {:?}", thir_pat, adjust);
            let span = thir_pat.span;
            let kind = match adjust.kind {
                PatAdjust::BuiltinDeref => PatKind::Deref { subpattern: thir_pat },
                PatAdjust::OverloadedDeref => {
                    let borrow = self.typeck_results.deref_pat_borrow_mode(adjust.source, pat);
                    PatKind::DerefPattern { subpattern: thir_pat, borrow }
                }
                PatAdjust::PinDeref => PatKind::Deref { subpattern: thir_pat },
            };
            Box::new(Pat { span, ty: adjust.source, kind })
        });

        if let Some(s) = &mut self.rust_2024_migration
            && adjustments.iter().any(|adjust| adjust.kind == PatAdjust::BuiltinDeref)
        {
            s.leave_ref(opt_old_mode_span);
        }

        adjusted_pat
    }

    fn lower_pattern_range_endpoint(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>, // Range pattern containing the endpoint
        expr: Option<&'tcx hir::PatExpr<'tcx>>,
        // Out-parameter collecting extra data to be reapplied by the caller
        ascriptions: &mut Vec<Ascription<'tcx>>,
    ) -> Result<Option<PatRangeBoundary<'tcx>>, ErrorGuaranteed> {
        assert_matches!(pat.kind, hir::PatKind::Range(..));

        // For partly-bounded ranges like `X..` or `..X`, an endpoint will be absent.
        // Return None in that case; the caller will use NegInfinity or PosInfinity instead.
        let Some(expr) = expr else { return Ok(None) };

        // Lower the endpoint into a temporary `PatKind` that will then be
        // deconstructed to obtain the constant value and other data.
        let mut kind: PatKind<'tcx> = self.lower_pat_expr(pat, expr);

        // Unpeel any ascription or inline-const wrapper nodes.
        loop {
            match kind {
                PatKind::AscribeUserType { ascription, subpattern } => {
                    ascriptions.push(ascription);
                    kind = subpattern.kind;
                }
                PatKind::ExpandedConstant { def_id: _, subpattern } => {
                    // Expanded-constant nodes are currently only needed by
                    // diagnostics that don't apply to range patterns, so we
                    // can just discard them here.
                    kind = subpattern.kind;
                }
                _ => break,
            }
        }

        // The unpeeled kind should now be a constant, giving us the endpoint value.
        let PatKind::Constant { value } = kind else {
            let msg =
                format!("found bad range pattern endpoint `{expr:?}` outside of error recovery");
            return Err(self.tcx.dcx().span_delayed_bug(expr.span, msg));
        };
        Ok(Some(PatRangeBoundary::Finite(value.valtree)))
    }

    /// Overflowing literals are linted against in a late pass. This is mostly fine, except when we
    /// encounter a range pattern like `-130i8..2`: if we believe `eval_bits`, this looks like a
    /// range where the endpoints are in the wrong order. To avoid a confusing error message, we
    /// check for overflow then.
    /// This is only called when the range is already known to be malformed.
    fn error_on_literal_overflow(
        &self,
        expr: Option<&'tcx hir::PatExpr<'tcx>>,
        ty: Ty<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        use rustc_ast::ast::LitKind;

        let Some(expr) = expr else {
            return Ok(());
        };
        let span = expr.span;

        // We need to inspect the original expression, because if we only inspect the output of
        // `eval_bits`, an overflowed value has already been wrapped around.
        // We mostly copy the logic from the `rustc_lint::OVERFLOWING_LITERALS` lint.
        let hir::PatExprKind::Lit { lit, negated } = expr.kind else {
            return Ok(());
        };
        let LitKind::Int(lit_val, _) = lit.node else {
            return Ok(());
        };
        let (min, max): (i128, u128) = match ty.kind() {
            ty::Int(ity) => {
                let size = Integer::from_int_ty(&self.tcx, *ity).size();
                (size.signed_int_min(), size.signed_int_max() as u128)
            }
            ty::Uint(uty) => {
                let size = Integer::from_uint_ty(&self.tcx, *uty).size();
                (0, size.unsigned_int_max())
            }
            _ => {
                return Ok(());
            }
        };
        // Detect literal value out of range `[min, max]` inclusive, avoiding use of `-min` to
        // prevent overflow/panic.
        if (negated && lit_val > max + 1) || (!negated && lit_val > max) {
            return Err(self.tcx.dcx().emit_err(LiteralOutOfRange { span, ty, min, max }));
        }
        Ok(())
    }

    fn lower_pattern_range(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>,
        lo_expr: Option<&'tcx hir::PatExpr<'tcx>>,
        hi_expr: Option<&'tcx hir::PatExpr<'tcx>>,
        end: RangeEnd,
    ) -> Result<PatKind<'tcx>, ErrorGuaranteed> {
        let ty = self.typeck_results.node_type(pat.hir_id);
        let span = pat.span;

        if lo_expr.is_none() && hi_expr.is_none() {
            let msg = "found twice-open range pattern (`..`) outside of error recovery";
            self.tcx.dcx().span_bug(span, msg);
        }

        // Collect extra data while lowering the endpoints, to be reapplied later.
        let mut ascriptions = vec![];
        let mut lower_endpoint =
            |expr| self.lower_pattern_range_endpoint(pat, expr, &mut ascriptions);

        let lo = lower_endpoint(lo_expr)?.unwrap_or(PatRangeBoundary::NegInfinity);
        let hi = lower_endpoint(hi_expr)?.unwrap_or(PatRangeBoundary::PosInfinity);

        let cmp = lo.compare_with(hi, ty, self.tcx);
        let mut kind = PatKind::Range(Arc::new(PatRange { lo, hi, end, ty }));
        match (end, cmp) {
            // `x..y` where `x < y`.
            (RangeEnd::Excluded, Some(Ordering::Less)) => {}
            // `x..=y` where `x < y`.
            (RangeEnd::Included, Some(Ordering::Less)) => {}
            // `x..=y` where `x == y` and `x` and `y` are finite.
            (RangeEnd::Included, Some(Ordering::Equal)) if lo.is_finite() && hi.is_finite() => {
                let value = ty::Value { ty, valtree: lo.as_finite().unwrap() };
                kind = PatKind::Constant { value };
            }
            // `..=x` where `x == ty::MIN`.
            (RangeEnd::Included, Some(Ordering::Equal)) if !lo.is_finite() => {}
            // `x..` where `x == ty::MAX` (yes, `x..` gives `RangeEnd::Included` since it is meant
            // to include `ty::MAX`).
            (RangeEnd::Included, Some(Ordering::Equal)) if !hi.is_finite() => {}
            // `x..y` where `x >= y`, or `x..=y` where `x > y`. The range is empty => error.
            _ => {
                // Emit a more appropriate message if there was overflow.
                self.error_on_literal_overflow(lo_expr, ty)?;
                self.error_on_literal_overflow(hi_expr, ty)?;
                let e = match end {
                    RangeEnd::Included => {
                        self.tcx.dcx().emit_err(LowerRangeBoundMustBeLessThanOrEqualToUpper {
                            span,
                            teach: self.tcx.sess.teach(E0030),
                        })
                    }
                    RangeEnd::Excluded if lo_expr.is_none() => {
                        self.tcx.dcx().emit_err(UpperRangeBoundCannotBeMin { span })
                    }
                    RangeEnd::Excluded => {
                        self.tcx.dcx().emit_err(LowerRangeBoundMustBeLessThanUpper { span })
                    }
                };
                return Err(e);
            }
        }

        // If we are handling a range with associated constants (e.g.
        // `Foo::<'a>::A..=Foo::B`), we need to put the ascriptions for the associated
        // constants somewhere. Have them on the range pattern.
        for ascription in ascriptions {
            let subpattern = Box::new(Pat { span, ty, kind });
            kind = PatKind::AscribeUserType { ascription, subpattern };
        }
        // `PatKind::ExpandedConstant` wrappers from range endpoints used to
        // also be preserved here, but that was only needed for unsafeck of
        // inline `const { .. }` patterns, which were removed by
        // <https://github.com/rust-lang/rust/pull/138492>.

        Ok(kind)
    }

    #[instrument(skip(self), level = "debug")]
    fn lower_pattern_unadjusted(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Box<Pat<'tcx>> {
        let mut ty = self.typeck_results.node_type(pat.hir_id);
        let mut span = pat.span;

        let kind = match pat.kind {
            hir::PatKind::Missing => PatKind::Missing,

            hir::PatKind::Wild => PatKind::Wild,

            hir::PatKind::Never => PatKind::Never,

            hir::PatKind::Expr(value) => self.lower_pat_expr(pat, value),

            hir::PatKind::Range(lo_expr, hi_expr, end) => {
                self.lower_pattern_range(pat, lo_expr, hi_expr, end).unwrap_or_else(PatKind::Error)
            }

            hir::PatKind::Deref(subpattern) => {
                let borrow = self.typeck_results.deref_pat_borrow_mode(ty, subpattern);
                PatKind::DerefPattern { subpattern: self.lower_pattern(subpattern), borrow }
            }
            hir::PatKind::Ref(subpattern, _, _) => {
                // Track the default binding mode for the Rust 2024 migration suggestion.
                let opt_old_mode_span =
                    self.rust_2024_migration.as_mut().and_then(|s| s.visit_explicit_deref());
                let subpattern = self.lower_pattern(subpattern);
                if let Some(s) = &mut self.rust_2024_migration {
                    s.leave_ref(opt_old_mode_span);
                }
                PatKind::Deref { subpattern }
            }
            hir::PatKind::Box(subpattern) => PatKind::DerefPattern {
                subpattern: self.lower_pattern(subpattern),
                borrow: DerefPatBorrowMode::Box,
            },

            hir::PatKind::Slice(prefix, slice, suffix) => {
                self.slice_or_array_pattern(pat, prefix, slice, suffix)
            }

            hir::PatKind::Tuple(pats, ddpos) => {
                let ty::Tuple(tys) = ty.kind() else {
                    span_bug!(pat.span, "unexpected type for tuple pattern: {:?}", ty);
                };
                let subpatterns = self.lower_tuple_subpats(pats, tys.len(), ddpos);
                PatKind::Leaf { subpatterns }
            }

            hir::PatKind::Binding(explicit_ba, id, ident, sub) => {
                if let Some(ident_span) = ident.span.find_ancestor_inside(span) {
                    span = span.with_hi(ident_span.hi());
                }

                let mode = *self
                    .typeck_results
                    .pat_binding_modes()
                    .get(pat.hir_id)
                    .expect("missing binding mode");

                if let Some(s) = &mut self.rust_2024_migration {
                    s.visit_binding(pat.span, mode, explicit_ba, ident);
                }

                // A ref x pattern is the same node used for x, and as such it has
                // x's type, which is &T, where we want T (the type being matched).
                let var_ty = ty;
                if let hir::ByRef::Yes(pinnedness, _) = mode.0 {
                    match pinnedness {
                        hir::Pinnedness::Pinned
                            if let Some(pty) = ty.pinned_ty()
                                && let &ty::Ref(_, rty, _) = pty.kind() =>
                        {
                            ty = rty;
                        }
                        hir::Pinnedness::Not if let &ty::Ref(_, rty, _) = ty.kind() => {
                            ty = rty;
                        }
                        _ => bug!("`ref {}` has wrong type {}", ident, ty),
                    }
                };

                PatKind::Binding {
                    mode,
                    name: ident.name,
                    var: LocalVarId(id),
                    ty: var_ty,
                    subpattern: self.lower_opt_pattern(sub),
                    is_primary: id == pat.hir_id,
                    is_shorthand: false,
                }
            }

            hir::PatKind::TupleStruct(ref qpath, pats, ddpos) => {
                let res = self.typeck_results.qpath_res(qpath, pat.hir_id);
                let ty::Adt(adt_def, _) = ty.kind() else {
                    span_bug!(pat.span, "tuple struct pattern not applied to an ADT {:?}", ty);
                };
                let variant_def = adt_def.variant_of_res(res);
                let subpatterns = self.lower_tuple_subpats(pats, variant_def.fields.len(), ddpos);
                self.lower_variant_or_leaf(pat, None, res, subpatterns)
            }

            hir::PatKind::Struct(ref qpath, fields, _) => {
                let res = self.typeck_results.qpath_res(qpath, pat.hir_id);
                let subpatterns = fields
                    .iter()
                    .map(|field| {
                        let mut pattern = *self.lower_pattern(field.pat);
                        if let PatKind::Binding { ref mut is_shorthand, .. } = pattern.kind {
                            *is_shorthand = field.is_shorthand;
                        }
                        let field = self.typeck_results.field_index(field.hir_id);
                        FieldPat { field, pattern }
                    })
                    .collect();

                self.lower_variant_or_leaf(pat, None, res, subpatterns)
            }

            hir::PatKind::Or(pats) => PatKind::Or { pats: self.lower_patterns(pats) },

            // FIXME(guard_patterns): implement guard pattern lowering
            hir::PatKind::Guard(pat, _) => self.lower_pattern(pat).kind,

            hir::PatKind::Err(guar) => PatKind::Error(guar),
        };

        Box::new(Pat { span, ty, kind })
    }

    fn lower_tuple_subpats(
        &mut self,
        pats: &'tcx [hir::Pat<'tcx>],
        expected_len: usize,
        gap_pos: hir::DotDotPos,
    ) -> Vec<FieldPat<'tcx>> {
        pats.iter()
            .enumerate_and_adjust(expected_len, gap_pos)
            .map(|(i, subpattern)| FieldPat {
                field: FieldIdx::new(i),
                pattern: *self.lower_pattern(subpattern),
            })
            .collect()
    }

    fn lower_patterns(&mut self, pats: &'tcx [hir::Pat<'tcx>]) -> Box<[Pat<'tcx>]> {
        pats.iter().map(|p| *self.lower_pattern(p)).collect()
    }

    fn lower_opt_pattern(&mut self, pat: Option<&'tcx hir::Pat<'tcx>>) -> Option<Box<Pat<'tcx>>> {
        pat.map(|p| self.lower_pattern(p))
    }

    fn slice_or_array_pattern(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>,
        prefix: &'tcx [hir::Pat<'tcx>],
        slice: Option<&'tcx hir::Pat<'tcx>>,
        suffix: &'tcx [hir::Pat<'tcx>],
    ) -> PatKind<'tcx> {
        let ty = self.typeck_results.node_type(pat.hir_id);

        let prefix = self.lower_patterns(prefix);
        let slice = self.lower_opt_pattern(slice);
        let suffix = self.lower_patterns(suffix);
        match ty.kind() {
            // Matching a slice, `[T]`.
            ty::Slice(..) => PatKind::Slice { prefix, slice, suffix },
            // Fixed-length array, `[T; len]`.
            ty::Array(_, len) => {
                let len = len
                    .try_to_target_usize(self.tcx)
                    .expect("expected len of array pat to be definite");
                assert!(len >= prefix.len() as u64 + suffix.len() as u64);
                PatKind::Array { prefix, slice, suffix }
            }
            _ => span_bug!(pat.span, "bad slice pattern type {ty:?}"),
        }
    }

    fn lower_variant_or_leaf(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>,
        expr: Option<&'tcx hir::PatExpr<'tcx>>,
        res: Res,
        subpatterns: Vec<FieldPat<'tcx>>,
    ) -> PatKind<'tcx> {
        // Check whether the caller should have provided an `expr` for this pattern kind.
        assert_matches!(
            (pat.kind, expr),
            (hir::PatKind::Expr(..) | hir::PatKind::Range(..), Some(_))
                | (hir::PatKind::Struct(..) | hir::PatKind::TupleStruct(..), None)
        );

        // Use the id/span of the `hir::PatExpr`, if provided.
        // Otherwise, use the id/span of the `hir::Pat`.
        let (hir_id, span) = match expr {
            Some(expr) => (expr.hir_id, expr.span),
            None => (pat.hir_id, pat.span),
        };
        let ty = self.typeck_results.node_type(hir_id);

        let res = match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), variant_ctor_id) => {
                let variant_id = self.tcx.parent(variant_ctor_id);
                Res::Def(DefKind::Variant, variant_id)
            }
            res => res,
        };

        let mut kind = match res {
            Res::Def(DefKind::Variant, variant_id) => {
                let enum_id = self.tcx.parent(variant_id);
                let adt_def = self.tcx.adt_def(enum_id);
                if adt_def.is_enum() {
                    let args = match ty.kind() {
                        ty::Adt(_, args) | ty::FnDef(_, args) => args,
                        ty::Error(e) => {
                            // Avoid ICE (#50585)
                            return PatKind::Error(*e);
                        }
                        _ => bug!("inappropriate type for def: {:?}", ty),
                    };
                    PatKind::Variant {
                        adt_def,
                        args,
                        variant_index: adt_def.variant_index_with_id(variant_id),
                        subpatterns,
                    }
                } else {
                    PatKind::Leaf { subpatterns }
                }
            }

            Res::Def(
                DefKind::Struct
                | DefKind::Ctor(CtorOf::Struct, ..)
                | DefKind::Union
                | DefKind::TyAlias
                | DefKind::AssocTy,
                _,
            )
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. }
            | Res::SelfCtor(..) => PatKind::Leaf { subpatterns },
            _ => {
                let e = match res {
                    Res::Def(DefKind::ConstParam, def_id) => {
                        let const_span = self.tcx.def_span(def_id);
                        self.tcx.dcx().emit_err(ConstParamInPattern { span, const_span })
                    }
                    Res::Def(DefKind::Static { .. }, def_id) => {
                        let static_span = self.tcx.def_span(def_id);
                        self.tcx.dcx().emit_err(StaticInPattern { span, static_span })
                    }
                    _ => self.tcx.dcx().emit_err(NonConstPath { span }),
                };
                PatKind::Error(e)
            }
        };

        if let Some(user_ty) = self.user_args_applied_to_ty_of_hir_id(hir_id) {
            debug!("lower_variant_or_leaf: kind={:?} user_ty={:?} span={:?}", kind, user_ty, span);
            let annotation = CanonicalUserTypeAnnotation {
                user_ty: Box::new(user_ty),
                span,
                inferred_ty: self.typeck_results.node_type(hir_id),
            };
            kind = PatKind::AscribeUserType {
                subpattern: Box::new(Pat { span, ty, kind }),
                ascription: Ascription { annotation, variance: ty::Covariant },
            };
        }

        kind
    }

    fn user_args_applied_to_ty_of_hir_id(
        &self,
        hir_id: hir::HirId,
    ) -> Option<ty::CanonicalUserType<'tcx>> {
        crate::thir::util::user_args_applied_to_ty_of_hir_id(self.tcx, self.typeck_results, hir_id)
    }

    /// Takes a HIR Path. If the path is a constant, evaluates it and feeds
    /// it to `const_to_pat`. Any other path (like enum variants without fields)
    /// is converted to the corresponding pattern via `lower_variant_or_leaf`.
    #[instrument(skip(self), level = "debug")]
    fn lower_path(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>, // Pattern that directly contains `expr`
        expr: &'tcx hir::PatExpr<'tcx>,
        qpath: &hir::QPath<'_>,
    ) -> Box<Pat<'tcx>> {
        assert_matches!(pat.kind, hir::PatKind::Expr(..) | hir::PatKind::Range(..));

        let id = expr.hir_id;
        let span = expr.span;
        let ty = self.typeck_results.node_type(id);
        let res = self.typeck_results.qpath_res(qpath, id);

        let (def_id, user_ty) = match res {
            Res::Def(DefKind::Const, def_id) | Res::Def(DefKind::AssocConst, def_id) => {
                (def_id, self.typeck_results.user_provided_types().get(id))
            }

            _ => {
                // The path isn't the name of a constant, so it must actually
                // be a unit struct or unit variant (e.g. `Option::None`).
                let kind = self.lower_variant_or_leaf(pat, Some(expr), res, vec![]);
                return Box::new(Pat { span, ty, kind });
            }
        };

        // Lower the named constant to a THIR pattern.
        let args = self.typeck_results.node_args(id);
        // FIXME(mgca): we will need to special case IACs here to have type system compatible
        // generic args, instead of how we represent them in body expressions.
        let c = ty::Const::new_unevaluated(self.tcx, ty::UnevaluatedConst { def: def_id, args });
        let mut pattern = self.const_to_pat(c, ty, id, span);

        // If this is an associated constant with an explicit user-written
        // type, add an ascription node (e.g. `<Foo<'a> as MyTrait>::CONST`).
        if let Some(&user_ty) = user_ty {
            let annotation = CanonicalUserTypeAnnotation {
                user_ty: Box::new(user_ty),
                span,
                inferred_ty: self.typeck_results.node_type(id),
            };
            let kind = PatKind::AscribeUserType {
                subpattern: pattern,
                ascription: Ascription {
                    annotation,
                    // Note that we use `Contravariant` here. See the
                    // `variance` field documentation for details.
                    variance: ty::Contravariant,
                },
            };
            pattern = Box::new(Pat { span, kind, ty });
        }

        pattern
    }

    /// Lowers the kinds of "expression" that can appear in a HIR pattern:
    /// - Paths (e.g. `FOO`, `foo::BAR`, `Option::None`)
    /// - Literals, possibly negated (e.g. `-128u8`, `"hello"`)
    fn lower_pat_expr(
        &mut self,
        pat: &'tcx hir::Pat<'tcx>, // Pattern that directly contains `expr`
        expr: &'tcx hir::PatExpr<'tcx>,
    ) -> PatKind<'tcx> {
        assert_matches!(pat.kind, hir::PatKind::Expr(..) | hir::PatKind::Range(..));
        match &expr.kind {
            hir::PatExprKind::Path(qpath) => self.lower_path(pat, expr, qpath).kind,
            hir::PatExprKind::Lit { lit, negated } => {
                // We handle byte string literal patterns by using the pattern's type instead of the
                // literal's type in `const_to_pat`: if the literal `b"..."` matches on a slice reference,
                // the pattern's type will be `&[u8]` whereas the literal's type is `&[u8; 3]`; using the
                // pattern's type means we'll properly translate it to a slice reference pattern. This works
                // because slices and arrays have the same valtree representation.
                //
                // Under `feature(deref_patterns)`, this adjustment can also convert string literal
                // patterns to `str`, and byte-string literal patterns to `[u8; N]` or `[u8]`.

                let pat_ty = self.typeck_results.node_type(pat.hir_id);
                let lit_input = LitToConstInput { lit: lit.node, ty: pat_ty, neg: *negated };
                let constant = self.tcx.at(expr.span).lit_to_const(lit_input);
                self.const_to_pat(constant, pat_ty, expr.hir_id, lit.span).kind
            }
        }
    }
}
