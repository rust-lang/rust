//! Validation of patterns/matches.

mod check_match;
mod const_to_pat;
mod migration;

use std::cmp::Ordering;
use std::sync::Arc;

use rustc_abi::{FieldIdx, Integer};
use rustc_errors::codes::*;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::{self as hir, LangItem, RangeEnd};
use rustc_index::Idx;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::interpret::LitToConstInput;
use rustc_middle::thir::{
    Ascription, FieldPat, LocalVarId, Pat, PatKind, PatRange, PatRangeBoundary,
};
use rustc_middle::ty::adjustment::{PatAdjust, PatAdjustment};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation, Ty, TyCtxt, TypingMode};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::DefId;
use rustc_span::{ErrorGuaranteed, Span};
use tracing::{debug, instrument};

pub(crate) use self::check_match::check_match;
use self::migration::PatMigration;
use crate::errors::*;

struct PatCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,

    /// Used by the Rust 2024 migration lint.
    rust_2024_migration: Option<PatMigration<'a>>,
}

pub(super) fn pat_from_hir<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,
    pat: &'tcx hir::Pat<'tcx>,
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
    let result = pcx.lower_pattern(pat);
    debug!("pat_from_hir({:?}) = {:?}", pat, result);
    if let Some(m) = pcx.rust_2024_migration {
        m.emit(tcx, pat.hir_id);
    }
    result
}

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
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
            hir::PatKind::Ref(inner, _)
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
        expr: Option<&'tcx hir::PatExpr<'tcx>>,
        // Out-parameters collecting extra data to be reapplied by the caller
        ascriptions: &mut Vec<Ascription<'tcx>>,
        expanded_consts: &mut Vec<DefId>,
    ) -> Result<Option<PatRangeBoundary<'tcx>>, ErrorGuaranteed> {
        let Some(expr) = expr else { return Ok(None) };

        // Lower the endpoint into a temporary `PatKind` that will then be
        // deconstructed to obtain the constant value and other data.
        let mut kind: PatKind<'tcx> = self.lower_pat_expr(expr, None);

        // Unpeel any ascription or inline-const wrapper nodes.
        loop {
            match kind {
                PatKind::AscribeUserType { ascription, subpattern } => {
                    ascriptions.push(ascription);
                    kind = subpattern.kind;
                }
                PatKind::ExpandedConstant { def_id, subpattern } => {
                    expanded_consts.push(def_id);
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

        Ok(Some(PatRangeBoundary::Finite(value)))
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
        lo_expr: Option<&'tcx hir::PatExpr<'tcx>>,
        hi_expr: Option<&'tcx hir::PatExpr<'tcx>>,
        end: RangeEnd,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Result<PatKind<'tcx>, ErrorGuaranteed> {
        if lo_expr.is_none() && hi_expr.is_none() {
            let msg = "found twice-open range pattern (`..`) outside of error recovery";
            self.tcx.dcx().span_bug(span, msg);
        }

        // Collect extra data while lowering the endpoints, to be reapplied later.
        let mut ascriptions = vec![];
        let mut expanded_consts = vec![];

        let mut lower_endpoint =
            |expr| self.lower_pattern_range_endpoint(expr, &mut ascriptions, &mut expanded_consts);

        let lo = lower_endpoint(lo_expr)?.unwrap_or(PatRangeBoundary::NegInfinity);
        let hi = lower_endpoint(hi_expr)?.unwrap_or(PatRangeBoundary::PosInfinity);

        let cmp = lo.compare_with(hi, ty, self.tcx, self.typing_env);
        let mut kind = PatKind::Range(Arc::new(PatRange { lo, hi, end, ty }));
        match (end, cmp) {
            // `x..y` where `x < y`.
            (RangeEnd::Excluded, Some(Ordering::Less)) => {}
            // `x..=y` where `x < y`.
            (RangeEnd::Included, Some(Ordering::Less)) => {}
            // `x..=y` where `x == y` and `x` and `y` are finite.
            (RangeEnd::Included, Some(Ordering::Equal)) if lo.is_finite() && hi.is_finite() => {
                kind = PatKind::Constant { value: lo.as_finite().unwrap() };
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
        for def_id in expanded_consts {
            let subpattern = Box::new(Pat { span, ty, kind });
            kind = PatKind::ExpandedConstant { def_id, subpattern };
        }
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

            hir::PatKind::Expr(value) => self.lower_pat_expr(value, Some(ty)),

            hir::PatKind::Range(ref lo_expr, ref hi_expr, end) => {
                let (lo_expr, hi_expr) = (lo_expr.as_deref(), hi_expr.as_deref());
                self.lower_pattern_range(lo_expr, hi_expr, end, ty, span)
                    .unwrap_or_else(PatKind::Error)
            }

            hir::PatKind::Deref(subpattern) => {
                let borrow = self.typeck_results.deref_pat_borrow_mode(ty, subpattern);
                PatKind::DerefPattern { subpattern: self.lower_pattern(subpattern), borrow }
            }
            hir::PatKind::Ref(subpattern, _) => {
                // Track the default binding mode for the Rust 2024 migration suggestion.
                let opt_old_mode_span =
                    self.rust_2024_migration.as_mut().and_then(|s| s.visit_explicit_deref());
                let subpattern = self.lower_pattern(subpattern);
                if let Some(s) = &mut self.rust_2024_migration {
                    s.leave_ref(opt_old_mode_span);
                }
                PatKind::Deref { subpattern }
            }
            hir::PatKind::Box(subpattern) => {
                PatKind::Deref { subpattern: self.lower_pattern(subpattern) }
            }

            hir::PatKind::Slice(prefix, slice, suffix) => {
                self.slice_or_array_pattern(pat.span, ty, prefix, slice, suffix)
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
                if let hir::ByRef::Yes(_) = mode.0 {
                    if let ty::Ref(_, rty, _) = ty.kind() {
                        ty = *rty;
                    } else {
                        bug!("`ref {}` has wrong type {}", ident, ty);
                    }
                };

                PatKind::Binding {
                    mode,
                    name: ident.name,
                    var: LocalVarId(id),
                    ty: var_ty,
                    subpattern: self.lower_opt_pattern(sub),
                    is_primary: id == pat.hir_id,
                }
            }

            hir::PatKind::TupleStruct(ref qpath, pats, ddpos) => {
                let res = self.typeck_results.qpath_res(qpath, pat.hir_id);
                let ty::Adt(adt_def, _) = ty.kind() else {
                    span_bug!(pat.span, "tuple struct pattern not applied to an ADT {:?}", ty);
                };
                let variant_def = adt_def.variant_of_res(res);
                let subpatterns = self.lower_tuple_subpats(pats, variant_def.fields.len(), ddpos);
                self.lower_variant_or_leaf(res, pat.hir_id, pat.span, ty, subpatterns)
            }

            hir::PatKind::Struct(ref qpath, fields, _) => {
                let res = self.typeck_results.qpath_res(qpath, pat.hir_id);
                let subpatterns = fields
                    .iter()
                    .map(|field| FieldPat {
                        field: self.typeck_results.field_index(field.hir_id),
                        pattern: *self.lower_pattern(field.pat),
                    })
                    .collect();

                self.lower_variant_or_leaf(res, pat.hir_id, pat.span, ty, subpatterns)
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
        span: Span,
        ty: Ty<'tcx>,
        prefix: &'tcx [hir::Pat<'tcx>],
        slice: Option<&'tcx hir::Pat<'tcx>>,
        suffix: &'tcx [hir::Pat<'tcx>],
    ) -> PatKind<'tcx> {
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
            _ => span_bug!(span, "bad slice pattern type {:?}", ty),
        }
    }

    fn lower_variant_or_leaf(
        &mut self,
        res: Res,
        hir_id: hir::HirId,
        span: Span,
        ty: Ty<'tcx>,
        subpatterns: Vec<FieldPat<'tcx>>,
    ) -> PatKind<'tcx> {
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
    fn lower_path(&mut self, qpath: &hir::QPath<'_>, id: hir::HirId, span: Span) -> Box<Pat<'tcx>> {
        let ty = self.typeck_results.node_type(id);
        let res = self.typeck_results.qpath_res(qpath, id);

        let (def_id, user_ty) = match res {
            Res::Def(DefKind::Const, def_id) | Res::Def(DefKind::AssocConst, def_id) => {
                (def_id, self.typeck_results.user_provided_types().get(id))
            }

            _ => {
                // The path isn't the name of a constant, so it must actually
                // be a unit struct or unit variant (e.g. `Option::None`).
                let kind = self.lower_variant_or_leaf(res, id, span, ty, vec![]);
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

    /// Lowers an inline const block (e.g. `const { 1 + 1 }`) to a pattern.
    fn lower_inline_const(
        &mut self,
        block: &'tcx hir::ConstBlock,
        id: hir::HirId,
        span: Span,
    ) -> PatKind<'tcx> {
        let tcx = self.tcx;
        let def_id = block.def_id;
        let ty = tcx.typeck(def_id).node_type(block.hir_id);

        let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id());
        let parent_args = ty::GenericArgs::identity_for_item(tcx, typeck_root_def_id);
        let args = ty::InlineConstArgs::new(tcx, ty::InlineConstArgsParts { parent_args, ty }).args;

        let ct = ty::UnevaluatedConst { def: def_id.to_def_id(), args };
        let c = ty::Const::new_unevaluated(self.tcx, ct);
        let pattern = self.const_to_pat(c, ty, id, span);

        // Apply a type ascription for the inline constant.
        let annotation = {
            let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
            let args = ty::InlineConstArgs::new(
                tcx,
                ty::InlineConstArgsParts { parent_args, ty: infcx.next_ty_var(span) },
            )
            .args;
            infcx.canonicalize_user_type_annotation(ty::UserType::new(ty::UserTypeKind::TypeOf(
                def_id.to_def_id(),
                ty::UserArgs { args, user_self_ty: None },
            )))
        };
        let annotation =
            CanonicalUserTypeAnnotation { user_ty: Box::new(annotation), span, inferred_ty: ty };
        PatKind::AscribeUserType {
            subpattern: pattern,
            ascription: Ascription {
                annotation,
                // Note that we use `Contravariant` here. See the `variance` field documentation
                // for details.
                variance: ty::Contravariant,
            },
        }
    }

    /// Lowers the kinds of "expression" that can appear in a HIR pattern:
    /// - Paths (e.g. `FOO`, `foo::BAR`, `Option::None`)
    /// - Inline const blocks (e.g. `const { 1 + 1 }`)
    /// - Literals, possibly negated (e.g. `-128u8`, `"hello"`)
    fn lower_pat_expr(
        &mut self,
        expr: &'tcx hir::PatExpr<'tcx>,
        pat_ty: Option<Ty<'tcx>>,
    ) -> PatKind<'tcx> {
        match &expr.kind {
            hir::PatExprKind::Path(qpath) => self.lower_path(qpath, expr.hir_id, expr.span).kind,
            hir::PatExprKind::ConstBlock(anon_const) => {
                self.lower_inline_const(anon_const, expr.hir_id, expr.span)
            }
            hir::PatExprKind::Lit { lit, negated } => {
                // We handle byte string literal patterns by using the pattern's type instead of the
                // literal's type in `const_to_pat`: if the literal `b"..."` matches on a slice reference,
                // the pattern's type will be `&[u8]` whereas the literal's type is `&[u8; 3]`; using the
                // pattern's type means we'll properly translate it to a slice reference pattern. This works
                // because slices and arrays have the same valtree representation.
                // HACK: As an exception, use the literal's type if `pat_ty` is `String`; this can happen if
                // `string_deref_patterns` is enabled. There's a special case for that when lowering to MIR.
                // FIXME(deref_patterns): This hack won't be necessary once `string_deref_patterns` is
                // superseded by a more general implementation of deref patterns.
                let ct_ty = match pat_ty {
                    Some(pat_ty)
                        if let ty::Adt(def, _) = *pat_ty.kind()
                            && self.tcx.is_lang_item(def.did(), LangItem::String) =>
                    {
                        if !self.tcx.features().string_deref_patterns() {
                            span_bug!(
                                expr.span,
                                "matching on `String` went through without enabling string_deref_patterns"
                            );
                        }
                        self.typeck_results.node_type(expr.hir_id)
                    }
                    Some(pat_ty) => pat_ty,
                    None => self.typeck_results.node_type(expr.hir_id),
                };
                let lit_input = LitToConstInput { lit: lit.node, ty: ct_ty, neg: *negated };
                let constant = self.tcx.at(expr.span).lit_to_const(lit_input);
                self.const_to_pat(constant, ct_ty, expr.hir_id, lit.span).kind
            }
        }
    }
}
