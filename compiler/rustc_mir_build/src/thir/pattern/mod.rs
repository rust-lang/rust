//! Validation of patterns/matches.

mod check_match;
mod const_to_pat;

pub(crate) use self::check_match::check_match;

use crate::errors::*;
use crate::thir::util::UserAnnotatedTyHelpers;

use rustc_errors::codes::*;
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::RangeEnd;
use rustc_index::Idx;
use rustc_middle::mir::interpret::{ErrorHandled, GlobalId, LitToConstError, LitToConstInput};
use rustc_middle::mir::{self, BorrowKind, Const, Mutability};
use rustc_middle::thir::{
    Ascription, BindingMode, FieldPat, LocalVarId, Pat, PatKind, PatRange, PatRangeBoundary,
};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::def_id::LocalDefId;
use rustc_span::{ErrorGuaranteed, Span};
use rustc_target::abi::{FieldIdx, Integer};

use std::cmp::Ordering;

struct PatCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,
}

pub(super) fn pat_from_hir<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,
    pat: &'tcx hir::Pat<'tcx>,
) -> Box<Pat<'tcx>> {
    let mut pcx = PatCtxt { tcx, param_env, typeck_results };
    let result = pcx.lower_pattern(pat);
    debug!("pat_from_hir({:?}) = {:?}", pat, result);
    result
}

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
    fn lower_pattern(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Box<Pat<'tcx>> {
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
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        self.typeck_results.pat_adjustments().get(pat.hir_id).unwrap_or(&vec![]).iter().rev().fold(
            unadjusted_pat,
            |pat: Box<_>, ref_ty| {
                debug!("{:?}: wrapping pattern with type {:?}", pat, ref_ty);
                Box::new(Pat {
                    span: pat.span,
                    ty: *ref_ty,
                    kind: PatKind::Deref { subpattern: pat },
                })
            },
        )
    }

    fn lower_pattern_range_endpoint(
        &mut self,
        expr: Option<&'tcx hir::Expr<'tcx>>,
    ) -> Result<
        (Option<PatRangeBoundary<'tcx>>, Option<Ascription<'tcx>>, Option<LocalDefId>),
        ErrorGuaranteed,
    > {
        match expr {
            None => Ok((None, None, None)),
            Some(expr) => {
                let (kind, ascr, inline_const) = match self.lower_lit(expr) {
                    PatKind::InlineConstant { subpattern, def } => {
                        (subpattern.kind, None, Some(def))
                    }
                    PatKind::AscribeUserType { ascription, subpattern: box Pat { kind, .. } } => {
                        (kind, Some(ascription), None)
                    }
                    kind => (kind, None, None),
                };
                let value = if let PatKind::Constant { value } = kind {
                    value
                } else {
                    let msg = format!(
                        "found bad range pattern endpoint `{expr:?}` outside of error recovery"
                    );
                    return Err(self.tcx.dcx().span_delayed_bug(expr.span, msg));
                };
                Ok((Some(PatRangeBoundary::Finite(value)), ascr, inline_const))
            }
        }
    }

    /// Overflowing literals are linted against in a late pass. This is mostly fine, except when we
    /// encounter a range pattern like `-130i8..2`: if we believe `eval_bits`, this looks like a
    /// range where the endpoints are in the wrong order. To avoid a confusing error message, we
    /// check for overflow then.
    /// This is only called when the range is already known to be malformed.
    fn error_on_literal_overflow(
        &self,
        expr: Option<&'tcx hir::Expr<'tcx>>,
        ty: Ty<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        use hir::{ExprKind, UnOp};
        use rustc_ast::ast::LitKind;

        let Some(mut expr) = expr else {
            return Ok(());
        };
        let span = expr.span;

        // We need to inspect the original expression, because if we only inspect the output of
        // `eval_bits`, an overflowed value has already been wrapped around.
        // We mostly copy the logic from the `rustc_lint::OVERFLOWING_LITERALS` lint.
        let mut negated = false;
        if let ExprKind::Unary(UnOp::Neg, sub_expr) = expr.kind {
            negated = true;
            expr = sub_expr;
        }
        let ExprKind::Lit(lit) = expr.kind else {
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
        lo_expr: Option<&'tcx hir::Expr<'tcx>>,
        hi_expr: Option<&'tcx hir::Expr<'tcx>>,
        end: RangeEnd,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Result<PatKind<'tcx>, ErrorGuaranteed> {
        if lo_expr.is_none() && hi_expr.is_none() {
            let msg = "found twice-open range pattern (`..`) outside of error recovery";
            self.tcx.dcx().span_bug(span, msg);
        }

        let (lo, lo_ascr, lo_inline) = self.lower_pattern_range_endpoint(lo_expr)?;
        let (hi, hi_ascr, hi_inline) = self.lower_pattern_range_endpoint(hi_expr)?;

        let lo = lo.unwrap_or(PatRangeBoundary::NegInfinity);
        let hi = hi.unwrap_or(PatRangeBoundary::PosInfinity);

        let cmp = lo.compare_with(hi, ty, self.tcx, self.param_env);
        let mut kind = PatKind::Range(Box::new(PatRange { lo, hi, end, ty }));
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
                            teach: self.tcx.sess.teach(E0030).then_some(()),
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
        for ascr in [lo_ascr, hi_ascr] {
            if let Some(ascription) = ascr {
                kind = PatKind::AscribeUserType {
                    ascription,
                    subpattern: Box::new(Pat { span, ty, kind }),
                };
            }
        }
        for inline_const in [lo_inline, hi_inline] {
            if let Some(def) = inline_const {
                kind =
                    PatKind::InlineConstant { def, subpattern: Box::new(Pat { span, ty, kind }) };
            }
        }
        Ok(kind)
    }

    #[instrument(skip(self), level = "debug")]
    fn lower_pattern_unadjusted(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Box<Pat<'tcx>> {
        let mut ty = self.typeck_results.node_type(pat.hir_id);
        let mut span = pat.span;

        let kind = match pat.kind {
            hir::PatKind::Wild => PatKind::Wild,

            hir::PatKind::Never => PatKind::Never,

            hir::PatKind::Lit(value) => self.lower_lit(value),

            hir::PatKind::Range(ref lo_expr, ref hi_expr, end) => {
                let (lo_expr, hi_expr) = (lo_expr.as_deref(), hi_expr.as_deref());
                self.lower_pattern_range(lo_expr, hi_expr, end, ty, span)
                    .unwrap_or_else(PatKind::Error)
            }

            hir::PatKind::Path(ref qpath) => {
                return self.lower_path(qpath, pat.hir_id, pat.span);
            }

            hir::PatKind::Ref(subpattern, _) | hir::PatKind::Box(subpattern) => {
                PatKind::Deref { subpattern: self.lower_pattern(subpattern) }
            }

            hir::PatKind::Slice(prefix, ref slice, suffix) => {
                self.slice_or_array_pattern(pat.span, ty, prefix, slice, suffix)
            }

            hir::PatKind::Tuple(pats, ddpos) => {
                let ty::Tuple(tys) = ty.kind() else {
                    span_bug!(pat.span, "unexpected type for tuple pattern: {:?}", ty);
                };
                let subpatterns = self.lower_tuple_subpats(pats, tys.len(), ddpos);
                PatKind::Leaf { subpatterns }
            }

            hir::PatKind::Binding(_, id, ident, ref sub) => {
                if let Some(ident_span) = ident.span.find_ancestor_inside(span) {
                    span = span.with_hi(ident_span.hi());
                }

                let bm = *self
                    .typeck_results
                    .pat_binding_modes()
                    .get(pat.hir_id)
                    .expect("missing binding mode");
                let (mutability, mode) = match bm {
                    ty::BindByValue(mutbl) => (mutbl, BindingMode::ByValue),
                    ty::BindByReference(hir::Mutability::Mut) => (
                        Mutability::Not,
                        BindingMode::ByRef(BorrowKind::Mut { kind: mir::MutBorrowKind::Default }),
                    ),
                    ty::BindByReference(hir::Mutability::Not) => {
                        (Mutability::Not, BindingMode::ByRef(BorrowKind::Shared))
                    }
                };

                // A ref x pattern is the same node used for x, and as such it has
                // x's type, which is &T, where we want T (the type being matched).
                let var_ty = ty;
                if let ty::BindByReference(_) = bm {
                    if let ty::Ref(_, rty, _) = ty.kind() {
                        ty = *rty;
                    } else {
                        bug!("`ref {}` has wrong type {}", ident, ty);
                    }
                };

                PatKind::Binding {
                    mutability,
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
                        pattern: self.lower_pattern(field.pat),
                    })
                    .collect();

                self.lower_variant_or_leaf(res, pat.hir_id, pat.span, ty, subpatterns)
            }

            hir::PatKind::Or(pats) => PatKind::Or { pats: self.lower_patterns(pats) },

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
                pattern: self.lower_pattern(subpattern),
            })
            .collect()
    }

    fn lower_patterns(&mut self, pats: &'tcx [hir::Pat<'tcx>]) -> Box<[Box<Pat<'tcx>>]> {
        pats.iter().map(|p| self.lower_pattern(p)).collect()
    }

    fn lower_opt_pattern(
        &mut self,
        pat: &'tcx Option<&'tcx hir::Pat<'tcx>>,
    ) -> Option<Box<Pat<'tcx>>> {
        pat.map(|p| self.lower_pattern(p))
    }

    fn slice_or_array_pattern(
        &mut self,
        span: Span,
        ty: Ty<'tcx>,
        prefix: &'tcx [hir::Pat<'tcx>],
        slice: &'tcx Option<&'tcx hir::Pat<'tcx>>,
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
                let len = len.eval_target_usize(self.tcx, self.param_env);
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
                    Res::Def(DefKind::ConstParam, _) => {
                        self.tcx.dcx().emit_err(ConstParamInPattern { span })
                    }
                    Res::Def(DefKind::Static(_), _) => {
                        self.tcx.dcx().emit_err(StaticInPattern { span })
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
                ascription: Ascription { annotation, variance: ty::Variance::Covariant },
            };
        }

        kind
    }

    /// Takes a HIR Path. If the path is a constant, evaluates it and feeds
    /// it to `const_to_pat`. Any other path (like enum variants without fields)
    /// is converted to the corresponding pattern via `lower_variant_or_leaf`.
    #[instrument(skip(self), level = "debug")]
    fn lower_path(&mut self, qpath: &hir::QPath<'_>, id: hir::HirId, span: Span) -> Box<Pat<'tcx>> {
        let ty = self.typeck_results.node_type(id);
        let res = self.typeck_results.qpath_res(qpath, id);

        let pat_from_kind = |kind| Box::new(Pat { span, ty, kind });

        let (def_id, is_associated_const) = match res {
            Res::Def(DefKind::Const, def_id) => (def_id, false),
            Res::Def(DefKind::AssocConst, def_id) => (def_id, true),

            _ => return pat_from_kind(self.lower_variant_or_leaf(res, id, span, ty, vec![])),
        };

        // Use `Reveal::All` here because patterns are always monomorphic even if their function
        // isn't.
        let param_env_reveal_all = self.param_env.with_reveal_all_normalized(self.tcx);
        // N.B. There is no guarantee that args collected in typeck results are fully normalized,
        // so they need to be normalized in order to pass to `Instance::resolve`, which will ICE
        // if given unnormalized types.
        let args = self
            .tcx
            .normalize_erasing_regions(param_env_reveal_all, self.typeck_results.node_args(id));
        let instance = match ty::Instance::resolve(self.tcx, param_env_reveal_all, def_id, args) {
            Ok(Some(i)) => i,
            Ok(None) => {
                // It should be assoc consts if there's no error but we cannot resolve it.
                debug_assert!(is_associated_const);

                let e = self.tcx.dcx().emit_err(AssocConstInPattern { span });
                return pat_from_kind(PatKind::Error(e));
            }

            Err(_) => {
                let e = self.tcx.dcx().emit_err(CouldNotEvalConstPattern { span });
                return pat_from_kind(PatKind::Error(e));
            }
        };

        let cid = GlobalId { instance, promoted: None };
        // Prefer valtrees over opaque constants.
        let const_value = self
            .tcx
            .const_eval_global_id_for_typeck(param_env_reveal_all, cid, Some(span))
            .map(|val| match val {
                Some(valtree) => mir::Const::Ty(ty::Const::new_value(self.tcx, valtree, ty)),
                None => mir::Const::Val(
                    self.tcx
                        .const_eval_global_id(param_env_reveal_all, cid, Some(span))
                        .expect("const_eval_global_id_for_typeck should have already failed"),
                    ty,
                ),
            });

        match const_value {
            Ok(const_) => {
                let pattern = self.const_to_pat(const_, id, span);

                if !is_associated_const {
                    return pattern;
                }

                let user_provided_types = self.typeck_results().user_provided_types();
                if let Some(&user_ty) = user_provided_types.get(id) {
                    let annotation = CanonicalUserTypeAnnotation {
                        user_ty: Box::new(user_ty),
                        span,
                        inferred_ty: self.typeck_results().node_type(id),
                    };
                    Box::new(Pat {
                        span,
                        kind: PatKind::AscribeUserType {
                            subpattern: pattern,
                            ascription: Ascription {
                                annotation,
                                // Note that use `Contravariant` here. See the
                                // `variance` field documentation for details.
                                variance: ty::Variance::Contravariant,
                            },
                        },
                        ty: const_.ty(),
                    })
                } else {
                    pattern
                }
            }
            Err(ErrorHandled::TooGeneric(_)) => {
                // While `Reported | Linted` cases will have diagnostics emitted already
                // it is not true for TooGeneric case, so we need to give user more information.
                let e = self.tcx.dcx().emit_err(ConstPatternDependsOnGenericParameter { span });
                pat_from_kind(PatKind::Error(e))
            }
            Err(_) => {
                let e = self.tcx.dcx().emit_err(CouldNotEvalConstPattern { span });
                pat_from_kind(PatKind::Error(e))
            }
        }
    }

    /// Converts inline const patterns.
    fn lower_inline_const(
        &mut self,
        block: &'tcx hir::ConstBlock,
        id: hir::HirId,
        span: Span,
    ) -> PatKind<'tcx> {
        let tcx = self.tcx;
        let def_id = block.def_id;
        let body_id = block.body;
        let expr = &tcx.hir().body(body_id).value;
        let ty = tcx.typeck(def_id).node_type(block.hir_id);

        // Special case inline consts that are just literals. This is solely
        // a performance optimization, as we could also just go through the regular
        // const eval path below.
        // FIXME: investigate the performance impact of removing this.
        let lit_input = match expr.kind {
            hir::ExprKind::Lit(lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: false }),
            hir::ExprKind::Unary(hir::UnOp::Neg, expr) => match expr.kind {
                hir::ExprKind::Lit(lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: true }),
                _ => None,
            },
            _ => None,
        };
        if let Some(lit_input) = lit_input {
            match tcx.at(expr.span).lit_to_const(lit_input) {
                Ok(c) => return self.const_to_pat(Const::Ty(c), id, span).kind,
                // If an error occurred, ignore that it's a literal
                // and leave reporting the error up to const eval of
                // the unevaluated constant below.
                Err(_) => {}
            }
        }

        let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id());
        let parent_args =
            tcx.erase_regions(ty::GenericArgs::identity_for_item(tcx, typeck_root_def_id));
        let args = ty::InlineConstArgs::new(tcx, ty::InlineConstArgsParts { parent_args, ty }).args;

        let uneval = mir::UnevaluatedConst { def: def_id.to_def_id(), args, promoted: None };
        debug_assert!(!args.has_free_regions());

        let ct = ty::UnevaluatedConst { def: def_id.to_def_id(), args };
        // First try using a valtree in order to destructure the constant into a pattern.
        // FIXME: replace "try to do a thing, then fall back to another thing"
        // but something more principled, like a trait query checking whether this can be turned into a valtree.
        if let Ok(Some(valtree)) =
            self.tcx.const_eval_resolve_for_typeck(self.param_env, ct, Some(span))
        {
            let subpattern =
                self.const_to_pat(Const::Ty(ty::Const::new_value(self.tcx, valtree, ty)), id, span);
            PatKind::InlineConstant { subpattern, def: def_id }
        } else {
            // If that fails, convert it to an opaque constant pattern.
            match tcx.const_eval_resolve(self.param_env, uneval, Some(span)) {
                Ok(val) => self.const_to_pat(mir::Const::Val(val, ty), id, span).kind,
                Err(ErrorHandled::TooGeneric(_)) => {
                    // If we land here it means the const can't be evaluated because it's `TooGeneric`.
                    let e = self.tcx.dcx().emit_err(ConstPatternDependsOnGenericParameter { span });
                    PatKind::Error(e)
                }
                Err(ErrorHandled::Reported(err, ..)) => PatKind::Error(err.into()),
            }
        }
    }

    /// Converts literals, paths and negation of literals to patterns.
    /// The special case for negation exists to allow things like `-128_i8`
    /// which would overflow if we tried to evaluate `128_i8` and then negate
    /// afterwards.
    fn lower_lit(&mut self, expr: &'tcx hir::Expr<'tcx>) -> PatKind<'tcx> {
        let (lit, neg) = match expr.kind {
            hir::ExprKind::Path(ref qpath) => {
                return self.lower_path(qpath, expr.hir_id, expr.span).kind;
            }
            hir::ExprKind::ConstBlock(ref anon_const) => {
                return self.lower_inline_const(anon_const, expr.hir_id, expr.span);
            }
            hir::ExprKind::Lit(ref lit) => (lit, false),
            hir::ExprKind::Unary(hir::UnOp::Neg, ref expr) => {
                let hir::ExprKind::Lit(ref lit) = expr.kind else {
                    span_bug!(expr.span, "not a literal: {:?}", expr);
                };
                (lit, true)
            }
            _ => span_bug!(expr.span, "not a literal: {:?}", expr),
        };

        let lit_input =
            LitToConstInput { lit: &lit.node, ty: self.typeck_results.expr_ty(expr), neg };
        match self.tcx.at(expr.span).lit_to_const(lit_input) {
            Ok(constant) => self.const_to_pat(Const::Ty(constant), expr.hir_id, lit.span).kind,
            Err(LitToConstError::Reported(e)) => PatKind::Error(e),
            Err(LitToConstError::TypeError) => bug!("lower_lit: had type error"),
        }
    }
}

impl<'tcx> UserAnnotatedTyHelpers<'tcx> for PatCtxt<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn typeck_results(&self) -> &ty::TypeckResults<'tcx> {
        self.typeck_results
    }
}
