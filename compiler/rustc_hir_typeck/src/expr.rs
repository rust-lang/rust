// ignore-tidy-filelength
// FIXME: we should move the field error reporting code somewhere else.

//! Type checking expressions.
//!
//! See [`rustc_hir_analysis::check`] for more context on type checking in general.

use rustc_abi::{ExternAbi, FIRST_VARIANT, FieldIdx};
use rustc_ast::util::parser::ExprPrecedence;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, ErrorGuaranteed, MultiSpan, StashKey, Subdiagnostic, listify, pluralize,
    struct_span_code_err,
};
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{Attribute, ExprKind, HirId, QPath};
use rustc_hir_analysis::NoVariantNamed;
use rustc_hir_analysis::hir_ty_lowering::{FeedConstTy, HirTyLowerer as _};
use rustc_infer::infer;
use rustc_infer::infer::{DefineOpaqueTypes, InferOk};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AllowTwoPhase};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, AdtKind, GenericArgsRef, Ty, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span, Symbol, kw, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{self, ObligationCauseCode, ObligationCtxt};
use tracing::{debug, instrument, trace};
use {rustc_ast as ast, rustc_hir as hir};

use crate::Expectation::{self, ExpectCastableToType, ExpectHasType, NoExpectation};
use crate::coercion::{CoerceMany, DynamicCoerceMany};
use crate::errors::{
    AddressOfTemporaryTaken, BaseExpressionDoubleDot, BaseExpressionDoubleDotAddExpr,
    BaseExpressionDoubleDotEnableDefaultFieldValues, BaseExpressionDoubleDotRemove,
    CantDereference, FieldMultiplySpecifiedInInitializer, FunctionalRecordUpdateOnNonStruct,
    HelpUseLatestEdition, NakedAsmOutsideNakedFn, NoFieldOnType, NoFieldOnVariant,
    ReturnLikeStatementKind, ReturnStmtOutsideOfFnBody, StructExprNonExhaustive,
    TypeMismatchFruTypo, YieldExprOutsideOfCoroutine,
};
use crate::{
    BreakableCtxt, CoroutineTypes, Diverges, FnCtxt, GatherLocalsVisitor, Needs,
    TupleArgumentsFlag, cast, fatally_break_rust, report_unexpected_variant_res, type_error_struct,
};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn precedence(&self, expr: &hir::Expr<'_>) -> ExprPrecedence {
        let for_each_attr = |id: HirId, callback: &mut dyn FnMut(&Attribute)| {
            for attr in self.tcx.hir_attrs(id) {
                // For the purpose of rendering suggestions, disregard attributes
                // that originate from desugaring of any kind. For example, `x?`
                // desugars to `#[allow(unreachable_code)] match ...`. Failing to
                // ignore the prefix attribute in the desugaring would cause this
                // suggestion:
                //
                //     let y: u32 = x?.try_into().unwrap();
                //                    ++++++++++++++++++++
                //
                // to be rendered as:
                //
                //     let y: u32 = (x?).try_into().unwrap();
                //                  +  +++++++++++++++++++++
                if attr.span().desugaring_kind().is_none() {
                    callback(attr);
                }
            }
        };
        expr.precedence(&for_each_attr)
    }

    /// Check an expr with an expectation type, and also demand that the expr's
    /// evaluated type is a subtype of the expectation at the end. This is a
    /// *hard* requirement.
    pub(crate) fn check_expr_has_type_or_error(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected_ty: Ty<'tcx>,
        extend_err: impl FnOnce(&mut Diag<'_>),
    ) -> Ty<'tcx> {
        let mut ty = self.check_expr_with_expectation(expr, ExpectHasType(expected_ty));

        // While we don't allow *arbitrary* coercions here, we *do* allow
        // coercions from ! to `expected`.
        if self.try_structurally_resolve_type(expr.span, ty).is_never()
            && self.expr_guaranteed_to_constitute_read_for_never(expr)
        {
            if let Some(adjustments) = self.typeck_results.borrow().adjustments().get(expr.hir_id) {
                let reported = self.dcx().span_delayed_bug(
                    expr.span,
                    "expression with never type wound up being adjusted",
                );

                return if let [Adjustment { kind: Adjust::NeverToAny, target }] = &adjustments[..] {
                    target.to_owned()
                } else {
                    Ty::new_error(self.tcx(), reported)
                };
            }

            let adj_ty = self.next_ty_var(expr.span);
            self.apply_adjustments(
                expr,
                vec![Adjustment { kind: Adjust::NeverToAny, target: adj_ty }],
            );
            ty = adj_ty;
        }

        if let Err(mut err) = self.demand_suptype_diag(expr.span, expected_ty, ty) {
            let _ = self.emit_type_mismatch_suggestions(
                &mut err,
                expr.peel_drop_temps(),
                ty,
                expected_ty,
                None,
                None,
            );
            extend_err(&mut err);
            err.emit();
        }
        ty
    }

    /// Check an expr with an expectation type, and also demand that the expr's
    /// evaluated type is a coercible to the expectation at the end. This is a
    /// *hard* requirement.
    pub(super) fn check_expr_coercible_to_type(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) -> Ty<'tcx> {
        self.check_expr_coercible_to_type_or_error(expr, expected, expected_ty_expr, |_, _| {})
    }

    pub(crate) fn check_expr_coercible_to_type_or_error(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        extend_err: impl FnOnce(&mut Diag<'_>, Ty<'tcx>),
    ) -> Ty<'tcx> {
        let ty = self.check_expr_with_hint(expr, expected);
        // checks don't need two phase
        match self.demand_coerce_diag(expr, ty, expected, expected_ty_expr, AllowTwoPhase::No) {
            Ok(ty) => ty,
            Err(mut err) => {
                extend_err(&mut err, ty);
                err.emit();
                // Return the original type instead of an error type here, otherwise the type of `x` in
                // `let x: u32 = ();` will be a type error, causing all subsequent usages of `x` to not
                // report errors, even though `x` is definitely `u32`.
                expected
            }
        }
    }

    /// Check an expr with an expectation type. Don't actually enforce that expectation
    /// is related to the expr's evaluated type via subtyping or coercion. This is
    /// usually called because we want to do that subtype/coerce call manually for better
    /// diagnostics.
    pub(super) fn check_expr_with_hint(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, ExpectHasType(expected))
    }

    /// Check an expr with an expectation type, and also [`Needs`] which will
    /// prompt typeck to convert any implicit immutable derefs to mutable derefs.
    fn check_expr_with_expectation_and_needs(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        needs: Needs,
    ) -> Ty<'tcx> {
        let ty = self.check_expr_with_expectation(expr, expected);

        // If the expression is used in a place whether mutable place is required
        // e.g. LHS of assignment, perform the conversion.
        if let Needs::MutPlace = needs {
            self.convert_place_derefs_to_mutable(expr);
        }

        ty
    }

    /// Check an expr with no expectations.
    pub(super) fn check_expr(&self, expr: &'tcx hir::Expr<'tcx>) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, NoExpectation)
    }

    /// Check an expr with no expectations, but with [`Needs`] which will
    /// prompt typeck to convert any implicit immutable derefs to mutable derefs.
    pub(super) fn check_expr_with_needs(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        needs: Needs,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_needs(expr, NoExpectation, needs)
    }

    /// Check an expr with an expectation type which may be used to eagerly
    /// guide inference when evaluating that expr.
    #[instrument(skip(self, expr), level = "debug")]
    pub(super) fn check_expr_with_expectation(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_args(expr, expected, None)
    }

    /// Same as [`Self::check_expr_with_expectation`], but allows us to pass in
    /// the arguments of a [`ExprKind::Call`] when evaluating its callee that
    /// is an [`ExprKind::Path`]. We use this to refine the spans for certain
    /// well-formedness guarantees for the path expr.
    pub(super) fn check_expr_with_expectation_and_args(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        call_expr_and_args: Option<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
    ) -> Ty<'tcx> {
        if self.tcx().sess.verbose_internals() {
            // make this code only run with -Zverbose-internals because it is probably slow
            if let Ok(lint_str) = self.tcx.sess.source_map().span_to_snippet(expr.span) {
                if !lint_str.contains('\n') {
                    debug!("expr text: {lint_str}");
                } else {
                    let mut lines = lint_str.lines();
                    if let Some(line0) = lines.next() {
                        let remaining_lines = lines.count();
                        debug!("expr text: {line0}");
                        debug!("expr text: ...(and {remaining_lines} more lines)");
                    }
                }
            }
        }

        // True if `expr` is a `Try::from_ok(())` that is a result of desugaring a try block
        // without the final expr (e.g. `try { return; }`). We don't want to generate an
        // unreachable_code lint for it since warnings for autogenerated code are confusing.
        let is_try_block_generated_unit_expr = match expr.kind {
            ExprKind::Call(_, [arg]) => {
                expr.span.is_desugaring(DesugaringKind::TryBlock)
                    && arg.span.is_desugaring(DesugaringKind::TryBlock)
            }
            _ => false,
        };

        // Warn for expressions after diverging siblings.
        if !is_try_block_generated_unit_expr {
            self.warn_if_unreachable(expr.hir_id, expr.span, "expression");
        }

        // Whether a past expression diverges doesn't affect typechecking of this expression, so we
        // reset `diverges` while checking `expr`.
        let old_diverges = self.diverges.replace(Diverges::Maybe);

        if self.is_whole_body.replace(false) {
            // If this expression is the whole body and the function diverges because of its
            // arguments, we check this here to ensure the body is considered to diverge.
            self.diverges.set(self.function_diverges_because_of_empty_arguments.get())
        };

        let ty = ensure_sufficient_stack(|| match &expr.kind {
            // Intercept the callee path expr and give it better spans.
            hir::ExprKind::Path(
                qpath @ (hir::QPath::Resolved(..) | hir::QPath::TypeRelative(..)),
            ) => self.check_expr_path(qpath, expr, call_expr_and_args),
            _ => self.check_expr_kind(expr, expected),
        });
        let ty = self.resolve_vars_if_possible(ty);

        // Warn for non-block expressions with diverging children.
        match expr.kind {
            ExprKind::Block(..)
            | ExprKind::If(..)
            | ExprKind::Let(..)
            | ExprKind::Loop(..)
            | ExprKind::Match(..) => {}
            // If `expr` is a result of desugaring the try block and is an ok-wrapped
            // diverging expression (e.g. it arose from desugaring of `try { return }`),
            // we skip issuing a warning because it is autogenerated code.
            ExprKind::Call(..) if expr.span.is_desugaring(DesugaringKind::TryBlock) => {}
            // Likewise, do not lint unreachable code injected via contracts desugaring.
            ExprKind::Call(..) if expr.span.is_desugaring(DesugaringKind::Contract) => {}
            ExprKind::Call(callee, _) => self.warn_if_unreachable(expr.hir_id, callee.span, "call"),
            ExprKind::MethodCall(segment, ..) => {
                self.warn_if_unreachable(expr.hir_id, segment.ident.span, "call")
            }
            _ => self.warn_if_unreachable(expr.hir_id, expr.span, "expression"),
        }

        // Any expression that produces a value of type `!` must have diverged,
        // unless it's a place expression that isn't being read from, in which case
        // diverging would be unsound since we may never actually read the `!`.
        // e.g. `let _ = *never_ptr;` with `never_ptr: *const !`.
        if self.try_structurally_resolve_type(expr.span, ty).is_never()
            && self.expr_guaranteed_to_constitute_read_for_never(expr)
        {
            self.diverges.set(self.diverges.get() | Diverges::always(expr.span));
        }

        // Record the type, which applies it effects.
        // We need to do this after the warning above, so that
        // we don't warn for the diverging expression itself.
        self.write_ty(expr.hir_id, ty);

        // Combine the diverging and has_error flags.
        self.diverges.set(self.diverges.get() | old_diverges);

        debug!("type of {} is...", self.tcx.hir_id_to_string(expr.hir_id));
        debug!("... {:?}, expected is {:?}", ty, expected);

        ty
    }

    /// Whether this expression constitutes a read of value of the type that
    /// it evaluates to.
    ///
    /// This is used to determine if we should consider the block to diverge
    /// if the expression evaluates to `!`, and if we should insert a `NeverToAny`
    /// coercion for values of type `!`.
    ///
    /// This function generally returns `false` if the expression is a place
    /// expression and the *parent* expression is the scrutinee of a match or
    /// the pointee of an `&` addr-of expression, since both of those parent
    /// expressions take a *place* and not a value.
    pub(super) fn expr_guaranteed_to_constitute_read_for_never(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> bool {
        // We only care about place exprs. Anything else returns an immediate
        // which would constitute a read. We don't care about distinguishing
        // "syntactic" place exprs since if the base of a field projection is
        // not a place then it would've been UB to read from it anyways since
        // that constitutes a read.
        if !expr.is_syntactic_place_expr() {
            return true;
        }

        let parent_node = self.tcx.parent_hir_node(expr.hir_id);
        match parent_node {
            hir::Node::Expr(parent_expr) => {
                match parent_expr.kind {
                    // Addr-of, field projections, and LHS of assignment don't constitute reads.
                    // Assignment does call `drop_in_place`, though, but its safety
                    // requirements are not the same.
                    ExprKind::AddrOf(..) | hir::ExprKind::Field(..) => false,

                    // Place-preserving expressions only constitute reads if their
                    // parent expression constitutes a read.
                    ExprKind::Type(..) | ExprKind::UnsafeBinderCast(..) => {
                        self.expr_guaranteed_to_constitute_read_for_never(expr)
                    }

                    ExprKind::Assign(lhs, _, _) => {
                        // Only the LHS does not constitute a read
                        expr.hir_id != lhs.hir_id
                    }

                    // See note on `PatKind::Or` below for why this is `all`.
                    ExprKind::Match(scrutinee, arms, _) => {
                        assert_eq!(scrutinee.hir_id, expr.hir_id);
                        arms.iter()
                            .all(|arm| self.pat_guaranteed_to_constitute_read_for_never(arm.pat))
                    }
                    ExprKind::Let(hir::LetExpr { init, pat, .. }) => {
                        assert_eq!(init.hir_id, expr.hir_id);
                        self.pat_guaranteed_to_constitute_read_for_never(*pat)
                    }

                    // Any expression child of these expressions constitute reads.
                    ExprKind::Array(_)
                    | ExprKind::Call(_, _)
                    | ExprKind::Use(_, _)
                    | ExprKind::MethodCall(_, _, _, _)
                    | ExprKind::Tup(_)
                    | ExprKind::Binary(_, _, _)
                    | ExprKind::Unary(_, _)
                    | ExprKind::Cast(_, _)
                    | ExprKind::DropTemps(_)
                    | ExprKind::If(_, _, _)
                    | ExprKind::Closure(_)
                    | ExprKind::Block(_, _)
                    | ExprKind::AssignOp(_, _, _)
                    | ExprKind::Index(_, _, _)
                    | ExprKind::Break(_, _)
                    | ExprKind::Ret(_)
                    | ExprKind::Become(_)
                    | ExprKind::InlineAsm(_)
                    | ExprKind::Struct(_, _, _)
                    | ExprKind::Repeat(_, _)
                    | ExprKind::Yield(_, _) => true,

                    // These expressions have no (direct) sub-exprs.
                    ExprKind::ConstBlock(_)
                    | ExprKind::Loop(_, _, _, _)
                    | ExprKind::Lit(_)
                    | ExprKind::Path(_)
                    | ExprKind::Continue(_)
                    | ExprKind::OffsetOf(_, _)
                    | ExprKind::Err(_) => unreachable!("no sub-expr expected for {:?}", expr.kind),
                }
            }

            // If we have a subpattern that performs a read, we want to consider this
            // to diverge for compatibility to support something like `let x: () = *never_ptr;`.
            hir::Node::LetStmt(hir::LetStmt { init: Some(target), pat, .. }) => {
                assert_eq!(target.hir_id, expr.hir_id);
                self.pat_guaranteed_to_constitute_read_for_never(*pat)
            }

            // These nodes (if they have a sub-expr) do constitute a read.
            hir::Node::Block(_)
            | hir::Node::Arm(_)
            | hir::Node::ExprField(_)
            | hir::Node::AnonConst(_)
            | hir::Node::ConstBlock(_)
            | hir::Node::ConstArg(_)
            | hir::Node::Stmt(_)
            | hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Const(..) | hir::ItemKind::Static(..),
                ..
            })
            | hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Const(..), ..
            })
            | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Const(..), .. }) => true,

            hir::Node::TyPat(_) | hir::Node::Pat(_) => {
                self.dcx().span_delayed_bug(expr.span, "place expr not allowed in pattern");
                true
            }

            // These nodes do not have direct sub-exprs.
            hir::Node::Param(_)
            | hir::Node::Item(_)
            | hir::Node::ForeignItem(_)
            | hir::Node::TraitItem(_)
            | hir::Node::ImplItem(_)
            | hir::Node::Variant(_)
            | hir::Node::Field(_)
            | hir::Node::PathSegment(_)
            | hir::Node::Ty(_)
            | hir::Node::AssocItemConstraint(_)
            | hir::Node::TraitRef(_)
            | hir::Node::PatField(_)
            | hir::Node::PatExpr(_)
            | hir::Node::LetStmt(_)
            | hir::Node::Synthetic
            | hir::Node::Err(_)
            | hir::Node::Ctor(_)
            | hir::Node::Lifetime(_)
            | hir::Node::GenericParam(_)
            | hir::Node::Crate(_)
            | hir::Node::Infer(_)
            | hir::Node::WherePredicate(_)
            | hir::Node::PreciseCapturingNonLifetimeArg(_)
            | hir::Node::OpaqueTy(_) => {
                unreachable!("no sub-expr expected for {parent_node:?}")
            }
        }
    }

    /// Whether this pattern constitutes a read of value of the scrutinee that
    /// it is matching against. This is used to determine whether we should
    /// perform `NeverToAny` coercions.
    ///
    /// See above for the nuances of what happens when this returns true.
    pub(super) fn pat_guaranteed_to_constitute_read_for_never(&self, pat: &hir::Pat<'_>) -> bool {
        match pat.kind {
            // Does not constitute a read.
            hir::PatKind::Wild => false,

            // Might not constitute a read, since the condition might be false.
            hir::PatKind::Guard(_, _) => true,

            // This is unnecessarily restrictive when the pattern that doesn't
            // constitute a read is unreachable.
            //
            // For example `match *never_ptr { value => {}, _ => {} }` or
            // `match *never_ptr { _ if false => {}, value => {} }`.
            //
            // It is however fine to be restrictive here; only returning `true`
            // can lead to unsoundness.
            hir::PatKind::Or(subpats) => {
                subpats.iter().all(|pat| self.pat_guaranteed_to_constitute_read_for_never(pat))
            }

            // Does constitute a read, since it is equivalent to a discriminant read.
            hir::PatKind::Never => true,

            // All of these constitute a read, or match on something that isn't `!`,
            // which would require a `NeverToAny` coercion.
            hir::PatKind::Missing
            | hir::PatKind::Binding(_, _, _, _)
            | hir::PatKind::Struct(_, _, _)
            | hir::PatKind::TupleStruct(_, _, _)
            | hir::PatKind::Tuple(_, _)
            | hir::PatKind::Box(_)
            | hir::PatKind::Ref(_, _)
            | hir::PatKind::Deref(_)
            | hir::PatKind::Expr(_)
            | hir::PatKind::Range(_, _, _)
            | hir::PatKind::Slice(_, _, _)
            | hir::PatKind::Err(_) => true,
        }
    }

    #[instrument(skip(self, expr), level = "debug")]
    fn check_expr_kind(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        trace!("expr={:#?}", expr);

        let tcx = self.tcx;
        match expr.kind {
            ExprKind::Lit(ref lit) => self.check_expr_lit(lit, expected),
            ExprKind::Binary(op, lhs, rhs) => self.check_expr_binop(expr, op, lhs, rhs, expected),
            ExprKind::Assign(lhs, rhs, span) => {
                self.check_expr_assign(expr, expected, lhs, rhs, span)
            }
            ExprKind::AssignOp(op, lhs, rhs) => {
                self.check_expr_assign_op(expr, op, lhs, rhs, expected)
            }
            ExprKind::Unary(unop, oprnd) => self.check_expr_unop(unop, oprnd, expected, expr),
            ExprKind::AddrOf(kind, mutbl, oprnd) => {
                self.check_expr_addr_of(kind, mutbl, oprnd, expected, expr)
            }
            ExprKind::Path(QPath::LangItem(lang_item, _)) => {
                self.check_lang_item_path(lang_item, expr)
            }
            ExprKind::Path(ref qpath) => self.check_expr_path(qpath, expr, None),
            ExprKind::InlineAsm(asm) => {
                // We defer some asm checks as we may not have resolved the input and output types yet (they may still be infer vars).
                self.deferred_asm_checks.borrow_mut().push((asm, expr.hir_id));
                self.check_expr_asm(asm, expr.span)
            }
            ExprKind::OffsetOf(container, fields) => {
                self.check_expr_offset_of(container, fields, expr)
            }
            ExprKind::Break(destination, ref expr_opt) => {
                self.check_expr_break(destination, expr_opt.as_deref(), expr)
            }
            ExprKind::Continue(destination) => self.check_expr_continue(destination, expr),
            ExprKind::Ret(ref expr_opt) => self.check_expr_return(expr_opt.as_deref(), expr),
            ExprKind::Become(call) => self.check_expr_become(call, expr),
            ExprKind::Let(let_expr) => self.check_expr_let(let_expr, expr.hir_id),
            ExprKind::Loop(body, _, source, _) => {
                self.check_expr_loop(body, source, expected, expr)
            }
            ExprKind::Match(discrim, arms, match_src) => {
                self.check_expr_match(expr, discrim, arms, expected, match_src)
            }
            ExprKind::Closure(closure) => self.check_expr_closure(closure, expr.span, expected),
            ExprKind::Block(body, _) => self.check_expr_block(body, expected),
            ExprKind::Call(callee, args) => self.check_expr_call(expr, callee, args, expected),
            ExprKind::Use(used_expr, _) => self.check_expr_use(used_expr, expected),
            ExprKind::MethodCall(segment, receiver, args, _) => {
                self.check_expr_method_call(expr, segment, receiver, args, expected)
            }
            ExprKind::Cast(e, t) => self.check_expr_cast(e, t, expr),
            ExprKind::Type(e, t) => {
                let ascribed_ty = self.lower_ty_saving_user_provided_ty(t);
                let ty = self.check_expr_with_hint(e, ascribed_ty);
                self.demand_eqtype(e.span, ascribed_ty, ty);
                ascribed_ty
            }
            ExprKind::If(cond, then_expr, opt_else_expr) => {
                self.check_expr_if(cond, then_expr, opt_else_expr, expr.span, expected)
            }
            ExprKind::DropTemps(e) => self.check_expr_with_expectation(e, expected),
            ExprKind::Array(args) => self.check_expr_array(args, expected, expr),
            ExprKind::ConstBlock(ref block) => self.check_expr_const_block(block, expected),
            ExprKind::Repeat(element, ref count) => {
                self.check_expr_repeat(element, count, expected, expr)
            }
            ExprKind::Tup(elts) => self.check_expr_tuple(elts, expected, expr),
            ExprKind::Struct(qpath, fields, ref base_expr) => {
                self.check_expr_struct(expr, expected, qpath, fields, base_expr)
            }
            ExprKind::Field(base, field) => self.check_expr_field(expr, base, field, expected),
            ExprKind::Index(base, idx, brackets_span) => {
                self.check_expr_index(base, idx, expr, brackets_span)
            }
            ExprKind::Yield(value, _) => self.check_expr_yield(value, expr),
            ExprKind::UnsafeBinderCast(kind, inner_expr, ty) => {
                self.check_expr_unsafe_binder_cast(expr.span, kind, inner_expr, ty, expected)
            }
            ExprKind::Err(guar) => Ty::new_error(tcx, guar),
        }
    }

    fn check_expr_unop(
        &self,
        unop: hir::UnOp,
        oprnd: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let expected_inner = match unop {
            hir::UnOp::Not | hir::UnOp::Neg => expected,
            hir::UnOp::Deref => NoExpectation,
        };
        let mut oprnd_t = self.check_expr_with_expectation(oprnd, expected_inner);

        if !oprnd_t.references_error() {
            oprnd_t = self.structurally_resolve_type(expr.span, oprnd_t);
            match unop {
                hir::UnOp::Deref => {
                    if let Some(ty) = self.lookup_derefing(expr, oprnd, oprnd_t) {
                        oprnd_t = ty;
                    } else {
                        let mut err =
                            self.dcx().create_err(CantDereference { span: expr.span, ty: oprnd_t });
                        let sp = tcx.sess.source_map().start_point(expr.span).with_parent(None);
                        if let Some(sp) =
                            tcx.sess.psess.ambiguous_block_expr_parse.borrow().get(&sp)
                        {
                            err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
                        }
                        oprnd_t = Ty::new_error(tcx, err.emit());
                    }
                }
                hir::UnOp::Not => {
                    let result = self.check_user_unop(expr, oprnd_t, unop, expected_inner);
                    // If it's builtin, we can reuse the type, this helps inference.
                    if !(oprnd_t.is_integral() || *oprnd_t.kind() == ty::Bool) {
                        oprnd_t = result;
                    }
                }
                hir::UnOp::Neg => {
                    let result = self.check_user_unop(expr, oprnd_t, unop, expected_inner);
                    // If it's builtin, we can reuse the type, this helps inference.
                    if !oprnd_t.is_numeric() {
                        oprnd_t = result;
                    }
                }
            }
        }
        oprnd_t
    }

    fn check_expr_addr_of(
        &self,
        kind: hir::BorrowKind,
        mutbl: hir::Mutability,
        oprnd: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let hint = expected.only_has_type(self).map_or(NoExpectation, |ty| {
            match self.try_structurally_resolve_type(expr.span, ty).kind() {
                ty::Ref(_, ty, _) | ty::RawPtr(ty, _) => {
                    if oprnd.is_syntactic_place_expr() {
                        // Places may legitimately have unsized types.
                        // For example, dereferences of a wide pointer and
                        // the last field of a struct can be unsized.
                        ExpectHasType(*ty)
                    } else {
                        Expectation::rvalue_hint(self, *ty)
                    }
                }
                _ => NoExpectation,
            }
        });
        let ty =
            self.check_expr_with_expectation_and_needs(oprnd, hint, Needs::maybe_mut_place(mutbl));

        match kind {
            _ if ty.references_error() => Ty::new_misc_error(self.tcx),
            hir::BorrowKind::Raw => {
                self.check_named_place_expr(oprnd);
                Ty::new_ptr(self.tcx, ty, mutbl)
            }
            hir::BorrowKind::Ref => {
                // Note: at this point, we cannot say what the best lifetime
                // is to use for resulting pointer. We want to use the
                // shortest lifetime possible so as to avoid spurious borrowck
                // errors. Moreover, the longest lifetime will depend on the
                // precise details of the value whose address is being taken
                // (and how long it is valid), which we don't know yet until
                // type inference is complete.
                //
                // Therefore, here we simply generate a region variable. The
                // region inferencer will then select a suitable value.
                // Finally, borrowck will infer the value of the region again,
                // this time with enough precision to check that the value
                // whose address was taken can actually be made to live as long
                // as it needs to live.
                let region = self.next_region_var(infer::BorrowRegion(expr.span));
                Ty::new_ref(self.tcx, region, ty, mutbl)
            }
        }
    }

    /// Does this expression refer to a place that either:
    /// * Is based on a local or static.
    /// * Contains a dereference
    /// Note that the adjustments for the children of `expr` should already
    /// have been resolved.
    fn check_named_place_expr(&self, oprnd: &'tcx hir::Expr<'tcx>) {
        let is_named = oprnd.is_place_expr(|base| {
            // Allow raw borrows if there are any deref adjustments.
            //
            // const VAL: (i32,) = (0,);
            // const REF: &(i32,) = &(0,);
            //
            // &raw const VAL.0;            // ERROR
            // &raw const REF.0;            // OK, same as &raw const (*REF).0;
            //
            // This is maybe too permissive, since it allows
            // `let u = &raw const Box::new((1,)).0`, which creates an
            // immediately dangling raw pointer.
            self.typeck_results
                .borrow()
                .adjustments()
                .get(base.hir_id)
                .is_some_and(|x| x.iter().any(|adj| matches!(adj.kind, Adjust::Deref(_))))
        });
        if !is_named {
            self.dcx().emit_err(AddressOfTemporaryTaken { span: oprnd.span });
        }
    }

    fn check_lang_item_path(
        &self,
        lang_item: hir::LangItem,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        self.resolve_lang_item_path(lang_item, expr.span, expr.hir_id).1
    }

    pub(crate) fn check_expr_path(
        &self,
        qpath: &'tcx hir::QPath<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        call_expr_and_args: Option<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let (res, opt_ty, segs) =
            self.resolve_ty_and_res_fully_qualified_call(qpath, expr.hir_id, expr.span);
        let ty = match res {
            Res::Err => {
                self.suggest_assoc_method_call(segs);
                let e =
                    self.dcx().span_delayed_bug(qpath.span(), "`Res::Err` but no error emitted");
                Ty::new_error(tcx, e)
            }
            Res::Def(DefKind::Variant, _) => {
                let e = report_unexpected_variant_res(
                    tcx,
                    res,
                    Some(expr),
                    qpath,
                    expr.span,
                    E0533,
                    "value",
                );
                Ty::new_error(tcx, e)
            }
            _ => {
                self.instantiate_value_path(
                    segs,
                    opt_ty,
                    res,
                    call_expr_and_args.map_or(expr.span, |(e, _)| e.span),
                    expr.span,
                    expr.hir_id,
                )
                .0
            }
        };

        if let ty::FnDef(did, _) = *ty.kind() {
            let fn_sig = ty.fn_sig(tcx);

            if tcx.is_intrinsic(did, sym::transmute) {
                let Some(from) = fn_sig.inputs().skip_binder().get(0) else {
                    span_bug!(
                        tcx.def_span(did),
                        "intrinsic fn `transmute` defined with no parameters"
                    );
                };
                let to = fn_sig.output().skip_binder();
                // We defer the transmute to the end of typeck, once all inference vars have
                // been resolved or we errored. This is important as we can only check transmute
                // on concrete types, but the output type may not be known yet (it would only
                // be known if explicitly specified via turbofish).
                self.deferred_transmute_checks.borrow_mut().push((*from, to, expr.hir_id));
            }
            if !tcx.features().unsized_fn_params() {
                // We want to remove some Sized bounds from std functions,
                // but don't want to expose the removal to stable Rust.
                // i.e., we don't want to allow
                //
                // ```rust
                // drop as fn(str);
                // ```
                //
                // to work in stable even if the Sized bound on `drop` is relaxed.
                for i in 0..fn_sig.inputs().skip_binder().len() {
                    // We just want to check sizedness, so instead of introducing
                    // placeholder lifetimes with probing, we just replace higher lifetimes
                    // with fresh vars.
                    let span = call_expr_and_args
                        .and_then(|(_, args)| args.get(i))
                        .map_or(expr.span, |arg| arg.span);
                    let input = self.instantiate_binder_with_fresh_vars(
                        span,
                        infer::BoundRegionConversionTime::FnCall,
                        fn_sig.input(i),
                    );
                    self.require_type_is_sized_deferred(
                        input,
                        span,
                        ObligationCauseCode::SizedArgumentType(None),
                    );
                }
            }
            // Here we want to prevent struct constructors from returning unsized types,
            // which can happen with fn pointer coercion on stable.
            // Also, as we just want to check sizedness, instead of introducing
            // placeholder lifetimes with probing, we just replace higher lifetimes
            // with fresh vars.
            let output = self.instantiate_binder_with_fresh_vars(
                expr.span,
                infer::BoundRegionConversionTime::FnCall,
                fn_sig.output(),
            );
            self.require_type_is_sized_deferred(
                output,
                call_expr_and_args.map_or(expr.span, |(e, _)| e.span),
                ObligationCauseCode::SizedCallReturnType,
            );
        }

        // We always require that the type provided as the value for
        // a type parameter outlives the moment of instantiation.
        let args = self.typeck_results.borrow().node_args(expr.hir_id);
        self.add_wf_bounds(args, expr.span);

        ty
    }

    fn check_expr_break(
        &self,
        destination: hir::Destination,
        expr_opt: Option<&'tcx hir::Expr<'tcx>>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        if let Ok(target_id) = destination.target_id {
            let (e_ty, cause);
            if let Some(e) = expr_opt {
                // If this is a break with a value, we need to type-check
                // the expression. Get an expected type from the loop context.
                let opt_coerce_to = {
                    // We should release `enclosing_breakables` before the `check_expr_with_hint`
                    // below, so can't move this block of code to the enclosing scope and share
                    // `ctxt` with the second `enclosing_breakables` borrow below.
                    let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
                    match enclosing_breakables.opt_find_breakable(target_id) {
                        Some(ctxt) => ctxt.coerce.as_ref().map(|coerce| coerce.expected_ty()),
                        None => {
                            // Avoid ICE when `break` is inside a closure (#65383).
                            return Ty::new_error_with_message(
                                tcx,
                                expr.span,
                                "break was outside loop, but no error was emitted",
                            );
                        }
                    }
                };

                // If the loop context is not a `loop { }`, then break with
                // a value is illegal, and `opt_coerce_to` will be `None`.
                // Set expectation to error in that case and set tainted
                // by error (#114529)
                let coerce_to = opt_coerce_to.unwrap_or_else(|| {
                    let guar = self.dcx().span_delayed_bug(
                        expr.span,
                        "illegal break with value found but no error reported",
                    );
                    self.set_tainted_by_errors(guar);
                    Ty::new_error(tcx, guar)
                });

                // Recurse without `enclosing_breakables` borrowed.
                e_ty = self.check_expr_with_hint(e, coerce_to);
                cause = self.misc(e.span);
            } else {
                // Otherwise, this is a break *without* a value. That's
                // always legal, and is equivalent to `break ()`.
                e_ty = tcx.types.unit;
                cause = self.misc(expr.span);
            }

            // Now that we have type-checked `expr_opt`, borrow
            // the `enclosing_loops` field and let's coerce the
            // type of `expr_opt` into what is expected.
            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            let Some(ctxt) = enclosing_breakables.opt_find_breakable(target_id) else {
                // Avoid ICE when `break` is inside a closure (#65383).
                return Ty::new_error_with_message(
                    tcx,
                    expr.span,
                    "break was outside loop, but no error was emitted",
                );
            };

            if let Some(ref mut coerce) = ctxt.coerce {
                if let Some(e) = expr_opt {
                    coerce.coerce(self, &cause, e, e_ty);
                } else {
                    assert!(e_ty.is_unit());
                    let ty = coerce.expected_ty();
                    coerce.coerce_forced_unit(
                        self,
                        &cause,
                        |mut err| {
                            self.suggest_missing_semicolon(&mut err, expr, e_ty, false, false);
                            self.suggest_mismatched_types_on_tail(
                                &mut err, expr, ty, e_ty, target_id,
                            );
                            let error =
                                Some(TypeError::Sorts(ExpectedFound { expected: ty, found: e_ty }));
                            self.annotate_loop_expected_due_to_inference(err, expr, error);
                            if let Some(val) =
                                self.err_ctxt().ty_kind_suggestion(self.param_env, ty)
                            {
                                err.span_suggestion_verbose(
                                    expr.span.shrink_to_hi(),
                                    "give the `break` a value of the expected type",
                                    format!(" {val}"),
                                    Applicability::HasPlaceholders,
                                );
                            }
                        },
                        false,
                    );
                }
            } else {
                // If `ctxt.coerce` is `None`, we can just ignore
                // the type of the expression. This is because
                // either this was a break *without* a value, in
                // which case it is always a legal type (`()`), or
                // else an error would have been flagged by the
                // `loops` pass for using break with an expression
                // where you are not supposed to.
                assert!(expr_opt.is_none() || self.tainted_by_errors().is_some());
            }

            // If we encountered a `break`, then (no surprise) it may be possible to break from the
            // loop... unless the value being returned from the loop diverges itself, e.g.
            // `break return 5` or `break loop {}`.
            ctxt.may_break |= !self.diverges.get().is_always();

            // the type of a `break` is always `!`, since it diverges
            tcx.types.never
        } else {
            // Otherwise, we failed to find the enclosing loop;
            // this can only happen if the `break` was not
            // inside a loop at all, which is caught by the
            // loop-checking pass.
            let err = Ty::new_error_with_message(
                self.tcx,
                expr.span,
                "break was outside loop, but no error was emitted",
            );

            // We still need to assign a type to the inner expression to
            // prevent the ICE in #43162.
            if let Some(e) = expr_opt {
                self.check_expr_with_hint(e, err);

                // ... except when we try to 'break rust;'.
                // ICE this expression in particular (see #43162).
                if let ExprKind::Path(QPath::Resolved(_, path)) = e.kind {
                    if let [segment] = path.segments
                        && segment.ident.name == sym::rust
                    {
                        fatally_break_rust(self.tcx, expr.span);
                    }
                }
            }

            // There was an error; make type-check fail.
            err
        }
    }

    fn check_expr_continue(
        &self,
        destination: hir::Destination,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        if let Ok(target_id) = destination.target_id {
            if let hir::Node::Expr(hir::Expr { kind: ExprKind::Loop(..), .. }) =
                self.tcx.hir_node(target_id)
            {
                self.tcx.types.never
            } else {
                // Liveness linting assumes `continue`s all point to loops. We'll report an error
                // in `check_mod_loops`, but make sure we don't run liveness (#113379, #121623).
                let guar = self.dcx().span_delayed_bug(
                    expr.span,
                    "found `continue` not pointing to loop, but no error reported",
                );
                Ty::new_error(self.tcx, guar)
            }
        } else {
            // There was an error; make type-check fail.
            Ty::new_misc_error(self.tcx)
        }
    }

    fn check_expr_return(
        &self,
        expr_opt: Option<&'tcx hir::Expr<'tcx>>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        if self.ret_coercion.is_none() {
            self.emit_return_outside_of_fn_body(expr, ReturnLikeStatementKind::Return);

            if let Some(e) = expr_opt {
                // We still have to type-check `e` (issue #86188), but calling
                // `check_return_expr` only works inside fn bodies.
                self.check_expr(e);
            }
        } else if let Some(e) = expr_opt {
            if self.ret_coercion_span.get().is_none() {
                self.ret_coercion_span.set(Some(e.span));
            }
            self.check_return_or_body_tail(e, true);
        } else {
            let mut coercion = self.ret_coercion.as_ref().unwrap().borrow_mut();
            if self.ret_coercion_span.get().is_none() {
                self.ret_coercion_span.set(Some(expr.span));
            }
            let cause = self.cause(expr.span, ObligationCauseCode::ReturnNoExpression);
            if let Some((_, fn_decl)) = self.get_fn_decl(expr.hir_id) {
                coercion.coerce_forced_unit(
                    self,
                    &cause,
                    |db| {
                        let span = fn_decl.output.span();
                        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                            db.span_label(
                                span,
                                format!("expected `{snippet}` because of this return type"),
                            );
                        }
                    },
                    true,
                );
            } else {
                coercion.coerce_forced_unit(self, &cause, |_| (), true);
            }
        }
        self.tcx.types.never
    }

    fn check_expr_become(
        &self,
        call: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        match &self.ret_coercion {
            Some(ret_coercion) => {
                let ret_ty = ret_coercion.borrow().expected_ty();
                let call_expr_ty = self.check_expr_with_hint(call, ret_ty);

                // N.B. don't coerce here, as tail calls can't support most/all coercions
                // FIXME(explicit_tail_calls): add a diagnostic note that `become` doesn't allow coercions
                self.demand_suptype(expr.span, ret_ty, call_expr_ty);
            }
            None => {
                self.emit_return_outside_of_fn_body(expr, ReturnLikeStatementKind::Become);

                // Fallback to simply type checking `call` without hint/demanding the right types.
                // Best effort to highlight more errors.
                self.check_expr(call);
            }
        }

        self.tcx.types.never
    }

    /// Check an expression that _is being returned_.
    /// For example, this is called with `return_expr: $expr` when `return $expr`
    /// is encountered.
    ///
    /// Note that this function must only be called in function bodies.
    ///
    /// `explicit_return` is `true` if we're checking an explicit `return expr`,
    /// and `false` if we're checking a trailing expression.
    pub(super) fn check_return_or_body_tail(
        &self,
        return_expr: &'tcx hir::Expr<'tcx>,
        explicit_return: bool,
    ) {
        let ret_coercion = self.ret_coercion.as_ref().unwrap_or_else(|| {
            span_bug!(return_expr.span, "check_return_expr called outside fn body")
        });

        let ret_ty = ret_coercion.borrow().expected_ty();
        let return_expr_ty = self.check_expr_with_hint(return_expr, ret_ty);
        let mut span = return_expr.span;
        let mut hir_id = return_expr.hir_id;
        // Use the span of the trailing expression for our cause,
        // not the span of the entire function
        if !explicit_return
            && let ExprKind::Block(body, _) = return_expr.kind
            && let Some(last_expr) = body.expr
        {
            span = last_expr.span;
            hir_id = last_expr.hir_id;
        }
        ret_coercion.borrow_mut().coerce(
            self,
            &self.cause(span, ObligationCauseCode::ReturnValue(return_expr.hir_id)),
            return_expr,
            return_expr_ty,
        );

        if let Some(fn_sig) = self.body_fn_sig()
            && fn_sig.output().has_opaque_types()
        {
            // Point any obligations that were registered due to opaque type
            // inference at the return expression.
            self.select_obligations_where_possible(|errors| {
                self.point_at_return_for_opaque_ty_error(
                    errors,
                    hir_id,
                    span,
                    return_expr_ty,
                    return_expr.span,
                );
            });
        }
    }

    /// Emit an error because `return` or `become` is used outside of a function body.
    ///
    /// `expr` is the `return` (`become`) "statement", `kind` is the kind of the statement
    /// either `Return` or `Become`.
    fn emit_return_outside_of_fn_body(&self, expr: &hir::Expr<'_>, kind: ReturnLikeStatementKind) {
        let mut err = ReturnStmtOutsideOfFnBody {
            span: expr.span,
            encl_body_span: None,
            encl_fn_span: None,
            statement_kind: kind,
        };

        let encl_item_id = self.tcx.hir_get_parent_item(expr.hir_id);

        if let hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn { .. },
            span: encl_fn_span,
            ..
        })
        | hir::Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(_)),
            span: encl_fn_span,
            ..
        })
        | hir::Node::ImplItem(hir::ImplItem {
            kind: hir::ImplItemKind::Fn(..),
            span: encl_fn_span,
            ..
        }) = self.tcx.hir_node_by_def_id(encl_item_id.def_id)
        {
            // We are inside a function body, so reporting "return statement
            // outside of function body" needs an explanation.

            let encl_body_owner_id = self.tcx.hir_enclosing_body_owner(expr.hir_id);

            // If this didn't hold, we would not have to report an error in
            // the first place.
            assert_ne!(encl_item_id.def_id, encl_body_owner_id);

            let encl_body = self.tcx.hir_body_owned_by(encl_body_owner_id);

            err.encl_body_span = Some(encl_body.value.span);
            err.encl_fn_span = Some(*encl_fn_span);
        }

        self.dcx().emit_err(err);
    }

    fn point_at_return_for_opaque_ty_error(
        &self,
        errors: &mut Vec<traits::FulfillmentError<'tcx>>,
        hir_id: HirId,
        span: Span,
        return_expr_ty: Ty<'tcx>,
        return_span: Span,
    ) {
        // Don't point at the whole block if it's empty
        if span == return_span {
            return;
        }
        for err in errors {
            let cause = &mut err.obligation.cause;
            if let ObligationCauseCode::OpaqueReturnType(None) = cause.code() {
                let new_cause = self.cause(
                    cause.span,
                    ObligationCauseCode::OpaqueReturnType(Some((return_expr_ty, hir_id))),
                );
                *cause = new_cause;
            }
        }
    }

    pub(crate) fn check_lhs_assignable(
        &self,
        lhs: &'tcx hir::Expr<'tcx>,
        code: ErrCode,
        op_span: Span,
        adjust_err: impl FnOnce(&mut Diag<'_>),
    ) {
        if lhs.is_syntactic_place_expr() {
            return;
        }

        let mut err = self.dcx().struct_span_err(op_span, "invalid left-hand side of assignment");
        err.code(code);
        err.span_label(lhs.span, "cannot assign to this expression");

        self.comes_from_while_condition(lhs.hir_id, |expr| {
            err.span_suggestion_verbose(
                expr.span.shrink_to_lo(),
                "you might have meant to use pattern destructuring",
                "let ",
                Applicability::MachineApplicable,
            );
        });
        self.check_for_missing_semi(lhs, &mut err);

        adjust_err(&mut err);

        err.emit();
    }

    /// Check if the expression that could not be assigned to was a typoed expression that
    pub(crate) fn check_for_missing_semi(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        err: &mut Diag<'_>,
    ) -> bool {
        if let hir::ExprKind::Binary(binop, lhs, rhs) = expr.kind
            && let hir::BinOpKind::Mul = binop.node
            && self.tcx.sess.source_map().is_multiline(lhs.span.between(rhs.span))
            && rhs.is_syntactic_place_expr()
        {
            //      v missing semicolon here
            // foo()
            // *bar = baz;
            // (#80446).
            err.span_suggestion_verbose(
                lhs.span.shrink_to_hi(),
                "you might have meant to write a semicolon here",
                ";",
                Applicability::MachineApplicable,
            );
            return true;
        }
        false
    }

    // Check if an expression `original_expr_id` comes from the condition of a while loop,
    /// as opposed from the body of a while loop, which we can naively check by iterating
    /// parents until we find a loop...
    pub(super) fn comes_from_while_condition(
        &self,
        original_expr_id: HirId,
        then: impl FnOnce(&hir::Expr<'_>),
    ) {
        let mut parent = self.tcx.parent_hir_id(original_expr_id);
        loop {
            let node = self.tcx.hir_node(parent);
            match node {
                hir::Node::Expr(hir::Expr {
                    kind:
                        hir::ExprKind::Loop(
                            hir::Block {
                                expr:
                                    Some(hir::Expr {
                                        kind:
                                            hir::ExprKind::Match(expr, ..) | hir::ExprKind::If(expr, ..),
                                        ..
                                    }),
                                ..
                            },
                            _,
                            hir::LoopSource::While,
                            _,
                        ),
                    ..
                }) => {
                    // Check if our original expression is a child of the condition of a while loop.
                    // If it is, then we have a situation like `while Some(0) = value.get(0) {`,
                    // where `while let` was more likely intended.
                    if self.tcx.hir_parent_id_iter(original_expr_id).any(|id| id == expr.hir_id) {
                        then(expr);
                    }
                    break;
                }
                hir::Node::Item(_)
                | hir::Node::ImplItem(_)
                | hir::Node::TraitItem(_)
                | hir::Node::Crate(_) => break,
                _ => {
                    parent = self.tcx.parent_hir_id(parent);
                }
            }
        }
    }

    // A generic function for checking the 'then' and 'else' clauses in an 'if'
    // or 'if-else' expression.
    fn check_expr_if(
        &self,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        opt_else_expr: Option<&'tcx hir::Expr<'tcx>>,
        sp: Span,
        orig_expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let cond_ty = self.check_expr_has_type_or_error(cond_expr, self.tcx.types.bool, |_| {});

        self.warn_if_unreachable(
            cond_expr.hir_id,
            then_expr.span,
            "block in `if` or `while` expression",
        );

        let cond_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        let expected = orig_expected.try_structurally_resolve_and_adjust_for_branches(self, sp);
        let then_ty = self.check_expr_with_expectation(then_expr, expected);
        let then_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        // We've already taken the expected type's preferences
        // into account when typing the `then` branch. To figure
        // out the initial shot at a LUB, we thus only consider
        // `expected` if it represents a *hard* constraint
        // (`only_has_type`); otherwise, we just go with a
        // fresh type variable.
        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce: DynamicCoerceMany<'_> = CoerceMany::new(coerce_to_ty);

        coerce.coerce(self, &self.misc(sp), then_expr, then_ty);

        if let Some(else_expr) = opt_else_expr {
            let else_ty = self.check_expr_with_expectation(else_expr, expected);
            let else_diverges = self.diverges.get();

            let tail_defines_return_position_impl_trait =
                self.return_position_impl_trait_from_match_expectation(orig_expected);
            let if_cause = self.if_cause(
                sp,
                cond_expr.span,
                then_expr,
                else_expr,
                then_ty,
                else_ty,
                tail_defines_return_position_impl_trait,
            );

            coerce.coerce(self, &if_cause, else_expr, else_ty);

            // We won't diverge unless both branches do (or the condition does).
            self.diverges.set(cond_diverges | then_diverges & else_diverges);
        } else {
            self.if_fallback_coercion(sp, cond_expr, then_expr, &mut coerce);

            // If the condition is false we can't diverge.
            self.diverges.set(cond_diverges);
        }

        let result_ty = coerce.complete(self);
        if let Err(guar) = cond_ty.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            result_ty
        }
    }

    /// Type check assignment expression `expr` of form `lhs = rhs`.
    /// The expected type is `()` and is passed to the function for the purposes of diagnostics.
    fn check_expr_assign(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        lhs: &'tcx hir::Expr<'tcx>,
        rhs: &'tcx hir::Expr<'tcx>,
        span: Span,
    ) -> Ty<'tcx> {
        let expected_ty = expected.only_has_type(self);
        if expected_ty == Some(self.tcx.types.bool) {
            let guar = self.expr_assign_expected_bool_error(expr, lhs, rhs, span);
            return Ty::new_error(self.tcx, guar);
        }

        let lhs_ty = self.check_expr_with_needs(lhs, Needs::MutPlace);

        let suggest_deref_binop = |err: &mut Diag<'_>, rhs_ty: Ty<'tcx>| {
            if let Some(lhs_deref_ty) = self.deref_once_mutably_for_diagnostic(lhs_ty) {
                // Can only assign if the type is sized, so if `DerefMut` yields a type that is
                // unsized, do not suggest dereferencing it.
                let lhs_deref_ty_is_sized = self
                    .infcx
                    .type_implements_trait(
                        self.tcx.require_lang_item(LangItem::Sized, span),
                        [lhs_deref_ty],
                        self.param_env,
                    )
                    .may_apply();
                if lhs_deref_ty_is_sized && self.may_coerce(rhs_ty, lhs_deref_ty) {
                    err.span_suggestion_verbose(
                        lhs.span.shrink_to_lo(),
                        "consider dereferencing here to assign to the mutably borrowed value",
                        "*",
                        Applicability::MachineApplicable,
                    );
                }
            }
        };

        // This is (basically) inlined `check_expr_coercible_to_type`, but we want
        // to suggest an additional fixup here in `suggest_deref_binop`.
        let rhs_ty = self.check_expr_with_hint(rhs, lhs_ty);
        if let Err(mut diag) =
            self.demand_coerce_diag(rhs, rhs_ty, lhs_ty, Some(lhs), AllowTwoPhase::No)
        {
            suggest_deref_binop(&mut diag, rhs_ty);
            diag.emit();
        }

        self.check_lhs_assignable(lhs, E0070, span, |err| {
            if let Some(rhs_ty) = self.typeck_results.borrow().expr_ty_opt(rhs) {
                suggest_deref_binop(err, rhs_ty);
            }
        });

        self.require_type_is_sized(lhs_ty, lhs.span, ObligationCauseCode::AssignmentLhsSized);

        if let Err(guar) = (lhs_ty, rhs_ty).error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            self.tcx.types.unit
        }
    }

    /// The expected type is `bool` but this will result in `()` so we can reasonably
    /// say that the user intended to write `lhs == rhs` instead of `lhs = rhs`.
    /// The likely cause of this is `if foo = bar { .. }`.
    fn expr_assign_expected_bool_error(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        lhs: &'tcx hir::Expr<'tcx>,
        rhs: &'tcx hir::Expr<'tcx>,
        span: Span,
    ) -> ErrorGuaranteed {
        let actual_ty = self.tcx.types.unit;
        let expected_ty = self.tcx.types.bool;
        let mut err = self.demand_suptype_diag(expr.span, expected_ty, actual_ty).unwrap_err();
        let lhs_ty = self.check_expr(lhs);
        let rhs_ty = self.check_expr(rhs);
        let refs_can_coerce = |lhs: Ty<'tcx>, rhs: Ty<'tcx>| {
            let lhs = Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, lhs.peel_refs());
            let rhs = Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, rhs.peel_refs());
            self.may_coerce(rhs, lhs)
        };
        let (applicability, eq) = if self.may_coerce(rhs_ty, lhs_ty) {
            (Applicability::MachineApplicable, true)
        } else if refs_can_coerce(rhs_ty, lhs_ty) {
            // The lhs and rhs are likely missing some references in either side. Subsequent
            // suggestions will show up.
            (Applicability::MaybeIncorrect, true)
        } else if let ExprKind::Binary(
            Spanned { node: hir::BinOpKind::And | hir::BinOpKind::Or, .. },
            _,
            rhs_expr,
        ) = lhs.kind
        {
            // if x == 1 && y == 2 { .. }
            //                 +
            let actual_lhs = self.check_expr(rhs_expr);
            let may_eq = self.may_coerce(rhs_ty, actual_lhs) || refs_can_coerce(rhs_ty, actual_lhs);
            (Applicability::MaybeIncorrect, may_eq)
        } else if let ExprKind::Binary(
            Spanned { node: hir::BinOpKind::And | hir::BinOpKind::Or, .. },
            lhs_expr,
            _,
        ) = rhs.kind
        {
            // if x == 1 && y == 2 { .. }
            //       +
            let actual_rhs = self.check_expr(lhs_expr);
            let may_eq = self.may_coerce(actual_rhs, lhs_ty) || refs_can_coerce(actual_rhs, lhs_ty);
            (Applicability::MaybeIncorrect, may_eq)
        } else {
            (Applicability::MaybeIncorrect, false)
        };

        if !lhs.is_syntactic_place_expr()
            && lhs.is_approximately_pattern()
            && !matches!(lhs.kind, hir::ExprKind::Lit(_))
        {
            // Do not suggest `if let x = y` as `==` is way more likely to be the intention.
            if let hir::Node::Expr(hir::Expr { kind: ExprKind::If { .. }, .. }) =
                self.tcx.parent_hir_node(expr.hir_id)
            {
                err.span_suggestion_verbose(
                    expr.span.shrink_to_lo(),
                    "you might have meant to use pattern matching",
                    "let ",
                    applicability,
                );
            };
        }
        if eq {
            err.span_suggestion_verbose(
                span.shrink_to_hi(),
                "you might have meant to compare for equality",
                '=',
                applicability,
            );
        }

        // If the assignment expression itself is ill-formed, don't
        // bother emitting another error
        err.emit_unless(lhs_ty.references_error() || rhs_ty.references_error())
    }

    pub(super) fn check_expr_let(
        &self,
        let_expr: &'tcx hir::LetExpr<'tcx>,
        hir_id: HirId,
    ) -> Ty<'tcx> {
        GatherLocalsVisitor::gather_from_let_expr(self, let_expr, hir_id);

        // for let statements, this is done in check_stmt
        let init = let_expr.init;
        self.warn_if_unreachable(init.hir_id, init.span, "block in `let` expression");

        // otherwise check exactly as a let statement
        self.check_decl((let_expr, hir_id).into());

        // but return a bool, for this is a boolean expression
        if let ast::Recovered::Yes(error_guaranteed) = let_expr.recovered {
            self.set_tainted_by_errors(error_guaranteed);
            Ty::new_error(self.tcx, error_guaranteed)
        } else {
            self.tcx.types.bool
        }
    }

    fn check_expr_loop(
        &self,
        body: &'tcx hir::Block<'tcx>,
        source: hir::LoopSource,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let coerce = match source {
            // you can only use break with a value from a normal `loop { }`
            hir::LoopSource::Loop => {
                let coerce_to = expected.coercion_target_type(self, body.span);
                Some(CoerceMany::new(coerce_to))
            }

            hir::LoopSource::While | hir::LoopSource::ForLoop => None,
        };

        let ctxt = BreakableCtxt {
            coerce,
            may_break: false, // Will get updated if/when we find a `break`.
        };

        let (ctxt, ()) = self.with_breakable_ctxt(expr.hir_id, ctxt, || {
            self.check_block_no_value(body);
        });

        if ctxt.may_break {
            // No way to know whether it's diverging because
            // of a `break` or an outer `break` or `return`.
            self.diverges.set(Diverges::Maybe);
        } else {
            self.diverges.set(self.diverges.get() | Diverges::always(expr.span));
        }

        // If we permit break with a value, then result type is
        // the LUB of the breaks (possibly ! if none); else, it
        // is nil. This makes sense because infinite loops
        // (which would have type !) are only possible iff we
        // permit break with a value.
        if ctxt.coerce.is_none() && !ctxt.may_break {
            self.dcx().span_bug(body.span, "no coercion, but loop may not break");
        }
        ctxt.coerce.map(|c| c.complete(self)).unwrap_or_else(|| self.tcx.types.unit)
    }

    /// Checks a method call.
    fn check_expr_method_call(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        segment: &'tcx hir::PathSegment<'tcx>,
        rcvr: &'tcx hir::Expr<'tcx>,
        args: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let rcvr_t = self.check_expr(rcvr);
        // no need to check for bot/err -- callee does that
        let rcvr_t = self.structurally_resolve_type(rcvr.span, rcvr_t);

        match self.lookup_method(rcvr_t, segment, segment.ident.span, expr, rcvr, args) {
            Ok(method) => {
                self.write_method_call_and_enforce_effects(expr.hir_id, expr.span, method);

                self.check_argument_types(
                    segment.ident.span,
                    expr,
                    &method.sig.inputs()[1..],
                    method.sig.output(),
                    expected,
                    args,
                    method.sig.c_variadic,
                    TupleArgumentsFlag::DontTupleArguments,
                    Some(method.def_id),
                );

                // Functions of type `extern "custom" fn(/* ... */)` cannot be called using
                // `ExprKind::MethodCall`. These functions have a calling convention that is
                // unknown to rust, hence it cannot generate code for the call. The only way
                // to execute such a function is via inline assembly.
                if let ExternAbi::Custom = method.sig.abi {
                    self.tcx.dcx().emit_err(crate::errors::AbiCustomCall { span: expr.span });
                }

                method.sig.output()
            }
            Err(error) => {
                let guar = self.report_method_error(expr.hir_id, rcvr_t, error, expected, false);

                let err_inputs = self.err_args(args.len(), guar);
                let err_output = Ty::new_error(self.tcx, guar);

                self.check_argument_types(
                    segment.ident.span,
                    expr,
                    &err_inputs,
                    err_output,
                    NoExpectation,
                    args,
                    false,
                    TupleArgumentsFlag::DontTupleArguments,
                    None,
                );

                err_output
            }
        }
    }

    /// Checks use `x.use`.
    fn check_expr_use(
        &self,
        used_expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation(used_expr, expected)
    }

    fn check_expr_cast(
        &self,
        e: &'tcx hir::Expr<'tcx>,
        t: &'tcx hir::Ty<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // Find the type of `e`. Supply hints based on the type we are casting to,
        // if appropriate.
        let t_cast = self.lower_ty_saving_user_provided_ty(t);
        let t_cast = self.resolve_vars_if_possible(t_cast);
        let t_expr = self.check_expr_with_expectation(e, ExpectCastableToType(t_cast));
        let t_expr = self.resolve_vars_if_possible(t_expr);

        // Eagerly check for some obvious errors.
        if let Err(guar) = (t_expr, t_cast).error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            // Defer other checks until we're done type checking.
            let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
            match cast::CastCheck::new(self, e, t_expr, t_cast, t.span, expr.span) {
                Ok(cast_check) => {
                    debug!(
                        "check_expr_cast: deferring cast from {:?} to {:?}: {:?}",
                        t_cast, t_expr, cast_check,
                    );
                    deferred_cast_checks.push(cast_check);
                    t_cast
                }
                Err(guar) => Ty::new_error(self.tcx, guar),
            }
        }
    }

    fn check_expr_unsafe_binder_cast(
        &self,
        span: Span,
        kind: ast::UnsafeBinderCastKind,
        inner_expr: &'tcx hir::Expr<'tcx>,
        hir_ty: Option<&'tcx hir::Ty<'tcx>>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        match kind {
            ast::UnsafeBinderCastKind::Wrap => {
                let ascribed_ty =
                    hir_ty.map(|hir_ty| self.lower_ty_saving_user_provided_ty(hir_ty));
                let expected_ty = expected.only_has_type(self);
                let binder_ty = match (ascribed_ty, expected_ty) {
                    (Some(ascribed_ty), Some(expected_ty)) => {
                        self.demand_eqtype(inner_expr.span, expected_ty, ascribed_ty);
                        expected_ty
                    }
                    (Some(ty), None) | (None, Some(ty)) => ty,
                    // This will always cause a structural resolve error, but we do it
                    // so we don't need to manually report an E0282 both on this codepath
                    // and in the others; it all happens in `structurally_resolve_type`.
                    (None, None) => self.next_ty_var(inner_expr.span),
                };

                let binder_ty = self.structurally_resolve_type(inner_expr.span, binder_ty);
                let hint_ty = match *binder_ty.kind() {
                    ty::UnsafeBinder(binder) => self.instantiate_binder_with_fresh_vars(
                        inner_expr.span,
                        infer::BoundRegionConversionTime::HigherRankedType,
                        binder.into(),
                    ),
                    ty::Error(e) => Ty::new_error(self.tcx, e),
                    _ => {
                        let guar = self
                            .dcx()
                            .struct_span_err(
                                hir_ty.map_or(span, |hir_ty| hir_ty.span),
                                format!(
                                    "`wrap_binder!()` can only wrap into unsafe binder, not {}",
                                    binder_ty.sort_string(self.tcx)
                                ),
                            )
                            .with_note("unsafe binders are the only valid output of wrap")
                            .emit();
                        Ty::new_error(self.tcx, guar)
                    }
                };

                self.check_expr_has_type_or_error(inner_expr, hint_ty, |_| {});

                binder_ty
            }
            ast::UnsafeBinderCastKind::Unwrap => {
                let ascribed_ty =
                    hir_ty.map(|hir_ty| self.lower_ty_saving_user_provided_ty(hir_ty));
                let hint_ty = ascribed_ty.unwrap_or_else(|| self.next_ty_var(inner_expr.span));
                // FIXME(unsafe_binders): coerce here if needed?
                let binder_ty = self.check_expr_has_type_or_error(inner_expr, hint_ty, |_| {});

                // Unwrap the binder. This will be ambiguous if it's an infer var, and will error
                // if it's not an unsafe binder.
                let binder_ty = self.structurally_resolve_type(inner_expr.span, binder_ty);
                match *binder_ty.kind() {
                    ty::UnsafeBinder(binder) => self.instantiate_binder_with_fresh_vars(
                        inner_expr.span,
                        infer::BoundRegionConversionTime::HigherRankedType,
                        binder.into(),
                    ),
                    ty::Error(e) => Ty::new_error(self.tcx, e),
                    _ => {
                        let guar = self
                            .dcx()
                            .struct_span_err(
                                hir_ty.map_or(inner_expr.span, |hir_ty| hir_ty.span),
                                format!(
                                    "expected unsafe binder, found {} as input of \
                                    `unwrap_binder!()`",
                                    binder_ty.sort_string(self.tcx)
                                ),
                            )
                            .with_note("only an unsafe binder type can be unwrapped")
                            .emit();
                        Ty::new_error(self.tcx, guar)
                    }
                }
            }
        }
    }

    fn check_expr_array(
        &self,
        args: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let element_ty = if !args.is_empty() {
            let coerce_to = expected
                .to_option(self)
                .and_then(|uty| self.try_structurally_resolve_type(expr.span, uty).builtin_index())
                .unwrap_or_else(|| self.next_ty_var(expr.span));
            let mut coerce = CoerceMany::with_coercion_sites(coerce_to, args);
            assert_eq!(self.diverges.get(), Diverges::Maybe);
            for e in args {
                let e_ty = self.check_expr_with_hint(e, coerce_to);
                let cause = self.misc(e.span);
                coerce.coerce(self, &cause, e, e_ty);
            }
            coerce.complete(self)
        } else {
            self.next_ty_var(expr.span)
        };
        let array_len = args.len() as u64;
        self.suggest_array_len(expr, array_len);
        Ty::new_array(self.tcx, element_ty, array_len)
    }

    fn suggest_array_len(&self, expr: &'tcx hir::Expr<'tcx>, array_len: u64) {
        let parent_node = self.tcx.hir_parent_iter(expr.hir_id).find(|(_, node)| {
            !matches!(node, hir::Node::Expr(hir::Expr { kind: hir::ExprKind::AddrOf(..), .. }))
        });
        let Some((_, hir::Node::LetStmt(hir::LetStmt { ty: Some(ty), .. }))) = parent_node else {
            return;
        };
        if let hir::TyKind::Array(_, ct) = ty.peel_refs().kind {
            let span = ct.span();
            self.dcx().try_steal_modify_and_emit_err(
                span,
                StashKey::UnderscoreForArrayLengths,
                |err| {
                    err.span_suggestion(
                        span,
                        "consider specifying the array length",
                        array_len,
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }

    pub(super) fn check_expr_const_block(
        &self,
        block: &'tcx hir::ConstBlock,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let body = self.tcx.hir_body(block.body);

        // Create a new function context.
        let def_id = block.def_id;
        let fcx = FnCtxt::new(self, self.param_env, def_id);

        let ty = fcx.check_expr_with_expectation(body.value, expected);
        fcx.require_type_is_sized(ty, body.value.span, ObligationCauseCode::SizedConstOrStatic);
        fcx.write_ty(block.hir_id, ty);
        ty
    }

    fn check_expr_repeat(
        &self,
        element: &'tcx hir::Expr<'tcx>,
        count: &'tcx hir::ConstArg<'tcx>,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let count_span = count.span();
        let count = self.try_structurally_resolve_const(
            count_span,
            self.normalize(count_span, self.lower_const_arg(count, FeedConstTy::No)),
        );

        if let Some(count) = count.try_to_target_usize(tcx) {
            self.suggest_array_len(expr, count);
        }

        let uty = match expected {
            ExpectHasType(uty) => uty.builtin_index(),
            _ => None,
        };

        let (element_ty, t) = match uty {
            Some(uty) => {
                self.check_expr_coercible_to_type(element, uty, None);
                (uty, uty)
            }
            None => {
                let ty = self.next_ty_var(element.span);
                let element_ty = self.check_expr_has_type_or_error(element, ty, |_| {});
                (element_ty, ty)
            }
        };

        if let Err(guar) = element_ty.error_reported() {
            return Ty::new_error(tcx, guar);
        }

        // We defer checking whether the element type is `Copy` as it is possible to have
        // an inference variable as a repeat count and it seems unlikely that `Copy` would
        // have inference side effects required for type checking to succeed.
        self.deferred_repeat_expr_checks.borrow_mut().push((element, element_ty, count));

        let ty = Ty::new_array_with_const_len(tcx, t, count);
        self.register_wf_obligation(ty.into(), expr.span, ObligationCauseCode::WellFormed(None));
        ty
    }

    fn check_expr_tuple(
        &self,
        elts: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let flds = expected.only_has_type(self).and_then(|ty| {
            let ty = self.try_structurally_resolve_type(expr.span, ty);
            match ty.kind() {
                ty::Tuple(flds) => Some(&flds[..]),
                _ => None,
            }
        });

        let elt_ts_iter = elts.iter().enumerate().map(|(i, e)| match flds {
            Some(fs) if i < fs.len() => {
                let ety = fs[i];
                self.check_expr_coercible_to_type(e, ety, None);
                ety
            }
            _ => self.check_expr_with_expectation(e, NoExpectation),
        });
        let tuple = Ty::new_tup_from_iter(self.tcx, elt_ts_iter);
        if let Err(guar) = tuple.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            self.require_type_is_sized(
                tuple,
                expr.span,
                ObligationCauseCode::TupleInitializerSized,
            );
            tuple
        }
    }

    fn check_expr_struct(
        &self,
        expr: &hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        qpath: &'tcx QPath<'tcx>,
        fields: &'tcx [hir::ExprField<'tcx>],
        base_expr: &'tcx hir::StructTailExpr<'tcx>,
    ) -> Ty<'tcx> {
        // Find the relevant variant
        let (variant, adt_ty) = match self.check_struct_path(qpath, expr.hir_id) {
            Ok(data) => data,
            Err(guar) => {
                self.check_struct_fields_on_error(fields, base_expr);
                return Ty::new_error(self.tcx, guar);
            }
        };

        // Prohibit struct expressions when non-exhaustive flag is set.
        let adt = adt_ty.ty_adt_def().expect("`check_struct_path` returned non-ADT type");
        if variant.field_list_has_applicable_non_exhaustive() {
            self.dcx()
                .emit_err(StructExprNonExhaustive { span: expr.span, what: adt.variant_descr() });
        }

        self.check_expr_struct_fields(
            adt_ty,
            expected,
            expr,
            qpath.span(),
            variant,
            fields,
            base_expr,
        );

        self.require_type_is_sized(adt_ty, expr.span, ObligationCauseCode::StructInitializerSized);
        adt_ty
    }

    fn check_expr_struct_fields(
        &self,
        adt_ty: Ty<'tcx>,
        expected: Expectation<'tcx>,
        expr: &hir::Expr<'_>,
        path_span: Span,
        variant: &'tcx ty::VariantDef,
        hir_fields: &'tcx [hir::ExprField<'tcx>],
        base_expr: &'tcx hir::StructTailExpr<'tcx>,
    ) {
        let tcx = self.tcx;

        let adt_ty = self.try_structurally_resolve_type(path_span, adt_ty);
        let adt_ty_hint = expected.only_has_type(self).and_then(|expected| {
            self.fudge_inference_if_ok(|| {
                let ocx = ObligationCtxt::new(self);
                ocx.sup(&self.misc(path_span), self.param_env, expected, adt_ty)?;
                if !ocx.select_where_possible().is_empty() {
                    return Err(TypeError::Mismatch);
                }
                Ok(self.resolve_vars_if_possible(adt_ty))
            })
            .ok()
        });
        if let Some(adt_ty_hint) = adt_ty_hint {
            // re-link the variables that the fudging above can create.
            self.demand_eqtype(path_span, adt_ty_hint, adt_ty);
        }

        let ty::Adt(adt, args) = adt_ty.kind() else {
            span_bug!(path_span, "non-ADT passed to check_expr_struct_fields");
        };
        let adt_kind = adt.adt_kind();

        let mut remaining_fields = variant
            .fields
            .iter_enumerated()
            .map(|(i, field)| (field.ident(tcx).normalize_to_macros_2_0(), (i, field)))
            .collect::<UnordMap<_, _>>();

        let mut seen_fields = FxHashMap::default();

        let mut error_happened = false;

        if variant.fields.len() != remaining_fields.len() {
            // Some field is defined more than once. Make sure we don't try to
            // instantiate this struct in static/const context.
            let guar =
                self.dcx().span_delayed_bug(expr.span, "struct fields have non-unique names");
            self.set_tainted_by_errors(guar);
            error_happened = true;
        }

        // Type-check each field.
        for (idx, field) in hir_fields.iter().enumerate() {
            let ident = tcx.adjust_ident(field.ident, variant.def_id);
            let field_type = if let Some((i, v_field)) = remaining_fields.remove(&ident) {
                seen_fields.insert(ident, field.span);
                self.write_field_index(field.hir_id, i);

                // We don't look at stability attributes on
                // struct-like enums (yet...), but it's definitely not
                // a bug to have constructed one.
                if adt_kind != AdtKind::Enum {
                    tcx.check_stability(v_field.did, Some(field.hir_id), field.span, None);
                }

                self.field_ty(field.span, v_field, args)
            } else {
                error_happened = true;
                let guar = if let Some(prev_span) = seen_fields.get(&ident) {
                    self.dcx().emit_err(FieldMultiplySpecifiedInInitializer {
                        span: field.ident.span,
                        prev_span: *prev_span,
                        ident,
                    })
                } else {
                    self.report_unknown_field(
                        adt_ty,
                        variant,
                        expr,
                        field,
                        hir_fields,
                        adt.variant_descr(),
                    )
                };

                Ty::new_error(tcx, guar)
            };

            // Check that the expected field type is WF. Otherwise, we emit no use-site error
            // in the case of coercions for non-WF fields, which leads to incorrect error
            // tainting. See issue #126272.
            self.register_wf_obligation(
                field_type.into(),
                field.expr.span,
                ObligationCauseCode::WellFormed(None),
            );

            // Make sure to give a type to the field even if there's
            // an error, so we can continue type-checking.
            let ty = self.check_expr_with_hint(field.expr, field_type);
            let diag = self.demand_coerce_diag(field.expr, ty, field_type, None, AllowTwoPhase::No);

            if let Err(diag) = diag {
                if idx == hir_fields.len() - 1 {
                    if remaining_fields.is_empty() {
                        self.suggest_fru_from_range_and_emit(field, variant, args, diag);
                    } else {
                        diag.stash(field.span, StashKey::MaybeFruTypo);
                    }
                } else {
                    diag.emit();
                }
            }
        }

        // Make sure the programmer specified correct number of fields.
        if adt_kind == AdtKind::Union && hir_fields.len() != 1 {
            struct_span_code_err!(
                self.dcx(),
                path_span,
                E0784,
                "union expressions should have exactly one field",
            )
            .emit();
        }

        // If check_expr_struct_fields hit an error, do not attempt to populate
        // the fields with the base_expr. This could cause us to hit errors later
        // when certain fields are assumed to exist that in fact do not.
        if error_happened {
            if let hir::StructTailExpr::Base(base_expr) = base_expr {
                self.check_expr(base_expr);
            }
            return;
        }

        if let hir::StructTailExpr::DefaultFields(span) = *base_expr {
            let mut missing_mandatory_fields = Vec::new();
            let mut missing_optional_fields = Vec::new();
            for f in &variant.fields {
                let ident = self.tcx.adjust_ident(f.ident(self.tcx), variant.def_id);
                if let Some(_) = remaining_fields.remove(&ident) {
                    if f.value.is_none() {
                        missing_mandatory_fields.push(ident);
                    } else {
                        missing_optional_fields.push(ident);
                    }
                }
            }
            if !self.tcx.features().default_field_values() {
                let sugg = self.tcx.crate_level_attribute_injection_span(expr.hir_id);
                self.dcx().emit_err(BaseExpressionDoubleDot {
                    span: span.shrink_to_hi(),
                    // We only mention enabling the feature if this is a nightly rustc *and* the
                    // expression would make sense with the feature enabled.
                    default_field_values_suggestion: if self.tcx.sess.is_nightly_build()
                        && missing_mandatory_fields.is_empty()
                        && !missing_optional_fields.is_empty()
                        && sugg.is_some()
                    {
                        sugg
                    } else {
                        None
                    },
                    default_field_values_help: if self.tcx.sess.is_nightly_build()
                        && missing_mandatory_fields.is_empty()
                        && !missing_optional_fields.is_empty()
                        && sugg.is_none()
                    {
                        Some(BaseExpressionDoubleDotEnableDefaultFieldValues)
                    } else {
                        None
                    },
                    add_expr: if !missing_mandatory_fields.is_empty()
                        || !missing_optional_fields.is_empty()
                    {
                        Some(BaseExpressionDoubleDotAddExpr { span: span.shrink_to_hi() })
                    } else {
                        None
                    },
                    remove_dots: if missing_mandatory_fields.is_empty()
                        && missing_optional_fields.is_empty()
                    {
                        Some(BaseExpressionDoubleDotRemove { span })
                    } else {
                        None
                    },
                });
                return;
            }
            if variant.fields.is_empty() {
                let mut err = self.dcx().struct_span_err(
                    span,
                    format!(
                        "`{adt_ty}` has no fields, `..` needs at least one default field in the \
                         struct definition",
                    ),
                );
                err.span_label(path_span, "this type has no fields");
                err.emit();
            }
            if !missing_mandatory_fields.is_empty() {
                let s = pluralize!(missing_mandatory_fields.len());
                let fields = listify(&missing_mandatory_fields, |f| format!("`{f}`")).unwrap();
                self.dcx()
                    .struct_span_err(
                        span.shrink_to_lo(),
                        format!("missing field{s} {fields} in initializer"),
                    )
                    .with_span_label(
                        span.shrink_to_lo(),
                        "fields that do not have a defaulted value must be provided explicitly",
                    )
                    .emit();
                return;
            }
            let fru_tys = match adt_ty.kind() {
                ty::Adt(adt, args) if adt.is_struct() => variant
                    .fields
                    .iter()
                    .map(|f| self.normalize(span, f.ty(self.tcx, args)))
                    .collect(),
                ty::Adt(adt, args) if adt.is_enum() => variant
                    .fields
                    .iter()
                    .map(|f| self.normalize(span, f.ty(self.tcx, args)))
                    .collect(),
                _ => {
                    self.dcx().emit_err(FunctionalRecordUpdateOnNonStruct { span });
                    return;
                }
            };
            self.typeck_results.borrow_mut().fru_field_types_mut().insert(expr.hir_id, fru_tys);
        } else if let hir::StructTailExpr::Base(base_expr) = base_expr {
            // FIXME: We are currently creating two branches here in order to maintain
            // consistency. But they should be merged as much as possible.
            let fru_tys = if self.tcx.features().type_changing_struct_update() {
                if adt.is_struct() {
                    // Make some fresh generic parameters for our ADT type.
                    let fresh_args = self.fresh_args_for_item(base_expr.span, adt.did());
                    // We do subtyping on the FRU fields first, so we can
                    // learn exactly what types we expect the base expr
                    // needs constrained to be compatible with the struct
                    // type we expect from the expectation value.
                    let fru_tys = variant
                        .fields
                        .iter()
                        .map(|f| {
                            let fru_ty = self
                                .normalize(expr.span, self.field_ty(base_expr.span, f, fresh_args));
                            let ident = self.tcx.adjust_ident(f.ident(self.tcx), variant.def_id);
                            if let Some(_) = remaining_fields.remove(&ident) {
                                let target_ty = self.field_ty(base_expr.span, f, args);
                                let cause = self.misc(base_expr.span);
                                match self.at(&cause, self.param_env).sup(
                                    // We're already using inference variables for any params, and don't allow converting
                                    // between different structs, so there is no way this ever actually defines an opaque type.
                                    // Thus choosing `Yes` is fine.
                                    DefineOpaqueTypes::Yes,
                                    target_ty,
                                    fru_ty,
                                ) {
                                    Ok(InferOk { obligations, value: () }) => {
                                        self.register_predicates(obligations)
                                    }
                                    Err(_) => {
                                        span_bug!(
                                            cause.span,
                                            "subtyping remaining fields of type changing FRU failed: {target_ty} != {fru_ty}: {}::{}",
                                            variant.name,
                                            ident.name,
                                        );
                                    }
                                }
                            }
                            self.resolve_vars_if_possible(fru_ty)
                        })
                        .collect();
                    // The use of fresh args that we have subtyped against
                    // our base ADT type's fields allows us to guide inference
                    // along so that, e.g.
                    // ```
                    // MyStruct<'a, F1, F2, const C: usize> {
                    //     f: F1,
                    //     // Other fields that reference `'a`, `F2`, and `C`
                    // }
                    //
                    // let x = MyStruct {
                    //    f: 1usize,
                    //    ..other_struct
                    // };
                    // ```
                    // will have the `other_struct` expression constrained to
                    // `MyStruct<'a, _, F2, C>`, as opposed to just `_`...
                    // This is important to allow coercions to happen in
                    // `other_struct` itself. See `coerce-in-base-expr.rs`.
                    let fresh_base_ty = Ty::new_adt(self.tcx, *adt, fresh_args);
                    self.check_expr_has_type_or_error(
                        base_expr,
                        self.resolve_vars_if_possible(fresh_base_ty),
                        |_| {},
                    );
                    fru_tys
                } else {
                    // Check the base_expr, regardless of a bad expected adt_ty, so we can get
                    // type errors on that expression, too.
                    self.check_expr(base_expr);
                    self.dcx().emit_err(FunctionalRecordUpdateOnNonStruct { span: base_expr.span });
                    return;
                }
            } else {
                self.check_expr_has_type_or_error(base_expr, adt_ty, |_| {
                    let base_ty = self.typeck_results.borrow().expr_ty(*base_expr);
                    let same_adt = matches!((adt_ty.kind(), base_ty.kind()),
                        (ty::Adt(adt, _), ty::Adt(base_adt, _)) if adt == base_adt);
                    if self.tcx.sess.is_nightly_build() && same_adt {
                        feature_err(
                            &self.tcx.sess,
                            sym::type_changing_struct_update,
                            base_expr.span,
                            "type changing struct updating is experimental",
                        )
                        .emit();
                    }
                });
                match adt_ty.kind() {
                    ty::Adt(adt, args) if adt.is_struct() => variant
                        .fields
                        .iter()
                        .map(|f| self.normalize(expr.span, f.ty(self.tcx, args)))
                        .collect(),
                    _ => {
                        self.dcx()
                            .emit_err(FunctionalRecordUpdateOnNonStruct { span: base_expr.span });
                        return;
                    }
                }
            };
            self.typeck_results.borrow_mut().fru_field_types_mut().insert(expr.hir_id, fru_tys);
        } else if adt_kind != AdtKind::Union && !remaining_fields.is_empty() {
            debug!(?remaining_fields);
            let private_fields: Vec<&ty::FieldDef> = variant
                .fields
                .iter()
                .filter(|field| !field.vis.is_accessible_from(tcx.parent_module(expr.hir_id), tcx))
                .collect();

            if !private_fields.is_empty() {
                self.report_private_fields(
                    adt_ty,
                    path_span,
                    expr.span,
                    private_fields,
                    hir_fields,
                );
            } else {
                self.report_missing_fields(
                    adt_ty,
                    path_span,
                    expr.span,
                    remaining_fields,
                    variant,
                    hir_fields,
                    args,
                );
            }
        }
    }

    fn check_struct_fields_on_error(
        &self,
        fields: &'tcx [hir::ExprField<'tcx>],
        base_expr: &'tcx hir::StructTailExpr<'tcx>,
    ) {
        for field in fields {
            self.check_expr(field.expr);
        }
        if let hir::StructTailExpr::Base(base) = *base_expr {
            self.check_expr(base);
        }
    }

    /// Report an error for a struct field expression when there are fields which aren't provided.
    ///
    /// ```text
    /// error: missing field `you_can_use_this_field` in initializer of `foo::Foo`
    ///  --> src/main.rs:8:5
    ///   |
    /// 8 |     foo::Foo {};
    ///   |     ^^^^^^^^ missing `you_can_use_this_field`
    ///
    /// error: aborting due to 1 previous error
    /// ```
    fn report_missing_fields(
        &self,
        adt_ty: Ty<'tcx>,
        span: Span,
        full_span: Span,
        remaining_fields: UnordMap<Ident, (FieldIdx, &ty::FieldDef)>,
        variant: &'tcx ty::VariantDef,
        hir_fields: &'tcx [hir::ExprField<'tcx>],
        args: GenericArgsRef<'tcx>,
    ) {
        let len = remaining_fields.len();

        let displayable_field_names: Vec<&str> =
            remaining_fields.items().map(|(ident, _)| ident.as_str()).into_sorted_stable_ord();

        let mut truncated_fields_error = String::new();
        let remaining_fields_names = match &displayable_field_names[..] {
            [field1] => format!("`{field1}`"),
            [field1, field2] => format!("`{field1}` and `{field2}`"),
            [field1, field2, field3] => format!("`{field1}`, `{field2}` and `{field3}`"),
            _ => {
                truncated_fields_error =
                    format!(" and {} other field{}", len - 3, pluralize!(len - 3));
                displayable_field_names
                    .iter()
                    .take(3)
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };

        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0063,
            "missing field{} {}{} in initializer of `{}`",
            pluralize!(len),
            remaining_fields_names,
            truncated_fields_error,
            adt_ty
        );
        err.span_label(span, format!("missing {remaining_fields_names}{truncated_fields_error}"));

        if remaining_fields.items().all(|(_, (_, field))| field.value.is_some())
            && self.tcx.sess.is_nightly_build()
        {
            let msg = format!(
                "all remaining fields have default values, {you_can} use those values with `..`",
                you_can = if self.tcx.features().default_field_values() {
                    "you can"
                } else {
                    "if you added `#![feature(default_field_values)]` to your crate you could"
                },
            );
            if let Some(hir_field) = hir_fields.last() {
                err.span_suggestion_verbose(
                    hir_field.span.shrink_to_hi(),
                    msg,
                    ", ..".to_string(),
                    Applicability::MachineApplicable,
                );
            } else if hir_fields.is_empty() {
                err.span_suggestion_verbose(
                    span.shrink_to_hi().with_hi(full_span.hi()),
                    msg,
                    " { .. }".to_string(),
                    Applicability::MachineApplicable,
                );
            }
        }

        if let Some(hir_field) = hir_fields.last() {
            self.suggest_fru_from_range_and_emit(hir_field, variant, args, err);
        } else {
            err.emit();
        }
    }

    /// If the last field is a range literal, but it isn't supposed to be, then they probably
    /// meant to use functional update syntax.
    fn suggest_fru_from_range_and_emit(
        &self,
        last_expr_field: &hir::ExprField<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgsRef<'tcx>,
        mut err: Diag<'_>,
    ) {
        // I don't use 'is_range_literal' because only double-sided, half-open ranges count.
        if let ExprKind::Struct(QPath::LangItem(LangItem::Range, ..), [range_start, range_end], _) =
            last_expr_field.expr.kind
            && let variant_field =
                variant.fields.iter().find(|field| field.ident(self.tcx) == last_expr_field.ident)
            && let range_def_id = self.tcx.lang_items().range_struct()
            && variant_field
                .and_then(|field| field.ty(self.tcx, args).ty_adt_def())
                .map(|adt| adt.did())
                != range_def_id
        {
            // Use a (somewhat arbitrary) filtering heuristic to avoid printing
            // expressions that are either too long, or have control character
            // such as newlines in them.
            let expr = self
                .tcx
                .sess
                .source_map()
                .span_to_snippet(range_end.expr.span)
                .ok()
                .filter(|s| s.len() < 25 && !s.contains(|c: char| c.is_control()));

            let fru_span = self
                .tcx
                .sess
                .source_map()
                .span_extend_while_whitespace(range_start.span)
                .shrink_to_hi()
                .to(range_end.span);

            err.subdiagnostic(TypeMismatchFruTypo { expr_span: range_start.span, fru_span, expr });

            // Suppress any range expr type mismatches
            self.dcx().try_steal_replace_and_emit_err(
                last_expr_field.span,
                StashKey::MaybeFruTypo,
                err,
            );
        } else {
            err.emit();
        }
    }

    /// Report an error for a struct field expression when there are invisible fields.
    ///
    /// ```text
    /// error: cannot construct `Foo` with struct literal syntax due to private fields
    ///  --> src/main.rs:8:5
    ///   |
    /// 8 |     foo::Foo {};
    ///   |     ^^^^^^^^
    ///
    /// error: aborting due to 1 previous error
    /// ```
    fn report_private_fields(
        &self,
        adt_ty: Ty<'tcx>,
        span: Span,
        expr_span: Span,
        private_fields: Vec<&ty::FieldDef>,
        used_fields: &'tcx [hir::ExprField<'tcx>],
    ) {
        let mut err =
            self.dcx().struct_span_err(
                span,
                format!(
                    "cannot construct `{adt_ty}` with struct literal syntax due to private fields",
                ),
            );
        let (used_private_fields, remaining_private_fields): (
            Vec<(Symbol, Span, bool)>,
            Vec<(Symbol, Span, bool)>,
        ) = private_fields
            .iter()
            .map(|field| {
                match used_fields.iter().find(|used_field| field.name == used_field.ident.name) {
                    Some(used_field) => (field.name, used_field.span, true),
                    None => (field.name, self.tcx.def_span(field.did), false),
                }
            })
            .partition(|field| field.2);
        err.span_labels(used_private_fields.iter().map(|(_, span, _)| *span), "private field");
        if !remaining_private_fields.is_empty() {
            let names = if remaining_private_fields.len() > 6 {
                String::new()
            } else {
                format!(
                    "{} ",
                    listify(&remaining_private_fields, |(name, _, _)| format!("`{name}`"))
                        .expect("expected at least one private field to report")
                )
            };
            err.note(format!(
                "{}private field{s} {names}that {were} not provided",
                if used_fields.is_empty() { "" } else { "...and other " },
                s = pluralize!(remaining_private_fields.len()),
                were = pluralize!("was", remaining_private_fields.len()),
            ));
        }

        if let ty::Adt(def, _) = adt_ty.kind() {
            let def_id = def.did();
            let mut items = self
                .tcx
                .inherent_impls(def_id)
                .into_iter()
                .flat_map(|i| self.tcx.associated_items(i).in_definition_order())
                // Only assoc fn with no receivers.
                .filter(|item| item.is_fn() && !item.is_method())
                .filter_map(|item| {
                    // Only assoc fns that return `Self`
                    let fn_sig = self.tcx.fn_sig(item.def_id).skip_binder();
                    let ret_ty = fn_sig.output();
                    let ret_ty = self.tcx.normalize_erasing_late_bound_regions(
                        self.typing_env(self.param_env),
                        ret_ty,
                    );
                    if !self.can_eq(self.param_env, ret_ty, adt_ty) {
                        return None;
                    }
                    let input_len = fn_sig.inputs().skip_binder().len();
                    let name = item.name();
                    let order = !name.as_str().starts_with("new");
                    Some((order, name, input_len))
                })
                .collect::<Vec<_>>();
            items.sort_by_key(|(order, _, _)| *order);
            let suggestion = |name, args| {
                format!(
                    "::{name}({})",
                    std::iter::repeat("_").take(args).collect::<Vec<_>>().join(", ")
                )
            };
            match &items[..] {
                [] => {}
                [(_, name, args)] => {
                    err.span_suggestion_verbose(
                        span.shrink_to_hi().with_hi(expr_span.hi()),
                        format!("you might have meant to use the `{name}` associated function"),
                        suggestion(name, *args),
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => {
                    err.span_suggestions(
                        span.shrink_to_hi().with_hi(expr_span.hi()),
                        "you might have meant to use an associated function to build this type",
                        items.iter().map(|(_, name, args)| suggestion(name, *args)),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            if let Some(default_trait) = self.tcx.get_diagnostic_item(sym::Default)
                && self
                    .infcx
                    .type_implements_trait(default_trait, [adt_ty], self.param_env)
                    .may_apply()
            {
                err.multipart_suggestion(
                    "consider using the `Default` trait",
                    vec![
                        (span.shrink_to_lo(), "<".to_string()),
                        (
                            span.shrink_to_hi().with_hi(expr_span.hi()),
                            " as std::default::Default>::default()".to_string(),
                        ),
                    ],
                    Applicability::MaybeIncorrect,
                );
            }
        }

        err.emit();
    }

    fn report_unknown_field(
        &self,
        ty: Ty<'tcx>,
        variant: &'tcx ty::VariantDef,
        expr: &hir::Expr<'_>,
        field: &hir::ExprField<'_>,
        skip_fields: &[hir::ExprField<'_>],
        kind_name: &str,
    ) -> ErrorGuaranteed {
        // we don't care to report errors for a struct if the struct itself is tainted
        if let Err(guar) = variant.has_errors() {
            return guar;
        }
        let mut err = self.err_ctxt().type_error_struct_with_diag(
            field.ident.span,
            |actual| match ty.kind() {
                ty::Adt(adt, ..) if adt.is_enum() => struct_span_code_err!(
                    self.dcx(),
                    field.ident.span,
                    E0559,
                    "{} `{}::{}` has no field named `{}`",
                    kind_name,
                    actual,
                    variant.name,
                    field.ident
                ),
                _ => struct_span_code_err!(
                    self.dcx(),
                    field.ident.span,
                    E0560,
                    "{} `{}` has no field named `{}`",
                    kind_name,
                    actual,
                    field.ident
                ),
            },
            ty,
        );

        let variant_ident_span = self.tcx.def_ident_span(variant.def_id).unwrap();
        match variant.ctor {
            Some((CtorKind::Fn, def_id)) => match ty.kind() {
                ty::Adt(adt, ..) if adt.is_enum() => {
                    err.span_label(
                        variant_ident_span,
                        format!(
                            "`{adt}::{variant}` defined here",
                            adt = ty,
                            variant = variant.name,
                        ),
                    );
                    err.span_label(field.ident.span, "field does not exist");
                    let fn_sig = self.tcx.fn_sig(def_id).instantiate_identity();
                    let inputs = fn_sig.inputs().skip_binder();
                    let fields = format!(
                        "({})",
                        inputs.iter().map(|i| format!("/* {i} */")).collect::<Vec<_>>().join(", ")
                    );
                    let (replace_span, sugg) = match expr.kind {
                        hir::ExprKind::Struct(qpath, ..) => {
                            (qpath.span().shrink_to_hi().with_hi(expr.span.hi()), fields)
                        }
                        _ => {
                            (expr.span, format!("{ty}::{variant}{fields}", variant = variant.name))
                        }
                    };
                    err.span_suggestion_verbose(
                        replace_span,
                        format!(
                            "`{adt}::{variant}` is a tuple {kind_name}, use the appropriate syntax",
                            adt = ty,
                            variant = variant.name,
                        ),
                        sugg,
                        Applicability::HasPlaceholders,
                    );
                }
                _ => {
                    err.span_label(variant_ident_span, format!("`{ty}` defined here"));
                    err.span_label(field.ident.span, "field does not exist");
                    let fn_sig = self.tcx.fn_sig(def_id).instantiate_identity();
                    let inputs = fn_sig.inputs().skip_binder();
                    let fields = format!(
                        "({})",
                        inputs.iter().map(|i| format!("/* {i} */")).collect::<Vec<_>>().join(", ")
                    );
                    err.span_suggestion_verbose(
                        expr.span,
                        format!("`{ty}` is a tuple {kind_name}, use the appropriate syntax",),
                        format!("{ty}{fields}"),
                        Applicability::HasPlaceholders,
                    );
                }
            },
            _ => {
                // prevent all specified fields from being suggested
                let available_field_names = self.available_field_names(variant, expr, skip_fields);
                if let Some(field_name) =
                    find_best_match_for_name(&available_field_names, field.ident.name, None)
                {
                    err.span_label(field.ident.span, "unknown field");
                    err.span_suggestion_verbose(
                        field.ident.span,
                        "a field with a similar name exists",
                        field_name,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    match ty.kind() {
                        ty::Adt(adt, ..) => {
                            if adt.is_enum() {
                                err.span_label(
                                    field.ident.span,
                                    format!("`{}::{}` does not have this field", ty, variant.name),
                                );
                            } else {
                                err.span_label(
                                    field.ident.span,
                                    format!("`{ty}` does not have this field"),
                                );
                            }
                            if available_field_names.is_empty() {
                                err.note("all struct fields are already assigned");
                            } else {
                                err.note(format!(
                                    "available fields are: {}",
                                    self.name_series_display(available_field_names)
                                ));
                            }
                        }
                        _ => bug!("non-ADT passed to report_unknown_field"),
                    }
                };
            }
        }
        err.emit()
    }

    fn available_field_names(
        &self,
        variant: &'tcx ty::VariantDef,
        expr: &hir::Expr<'_>,
        skip_fields: &[hir::ExprField<'_>],
    ) -> Vec<Symbol> {
        variant
            .fields
            .iter()
            .filter(|field| {
                skip_fields.iter().all(|&skip| skip.ident.name != field.name)
                    && self.is_field_suggestable(field, expr.hir_id, expr.span)
            })
            .map(|field| field.name)
            .collect()
    }

    fn name_series_display(&self, names: Vec<Symbol>) -> String {
        // dynamic limit, to never omit just one field
        let limit = if names.len() == 6 { 6 } else { 5 };
        let mut display =
            names.iter().take(limit).map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ");
        if names.len() > limit {
            display = format!("{} ... and {} others", display, names.len() - limit);
        }
        display
    }

    /// Find the position of a field named `ident` in `base_def`, accounting for unnammed fields.
    /// Return whether such a field has been found. The path to it is stored in `nested_fields`.
    /// `ident` must have been adjusted beforehand.
    fn find_adt_field(
        &self,
        base_def: ty::AdtDef<'tcx>,
        ident: Ident,
    ) -> Option<(FieldIdx, &'tcx ty::FieldDef)> {
        // No way to find a field in an enum.
        if base_def.is_enum() {
            return None;
        }

        for (field_idx, field) in base_def.non_enum_variant().fields.iter_enumerated() {
            if field.ident(self.tcx).normalize_to_macros_2_0() == ident {
                // We found the field we wanted.
                return Some((field_idx, field));
            }
        }

        None
    }

    /// Check field access expressions, this works for both structs and tuples.
    /// Returns the Ty of the field.
    ///
    /// ```ignore (illustrative)
    /// base.field
    /// ^^^^^^^^^^ expr
    /// ^^^^       base
    ///      ^^^^^ field
    /// ```
    fn check_expr_field(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        base: &'tcx hir::Expr<'tcx>,
        field: Ident,
        // The expected type hint of the field.
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        debug!("check_field(expr: {:?}, base: {:?}, field: {:?})", expr, base, field);
        let base_ty = self.check_expr(base);
        let base_ty = self.structurally_resolve_type(base.span, base_ty);

        // Whether we are trying to access a private field. Used for error reporting.
        let mut private_candidate = None;

        // Field expressions automatically deref
        let mut autoderef = self.autoderef(expr.span, base_ty);
        while let Some((deref_base_ty, _)) = autoderef.next() {
            debug!("deref_base_ty: {:?}", deref_base_ty);
            match deref_base_ty.kind() {
                ty::Adt(base_def, args) if !base_def.is_enum() => {
                    debug!("struct named {:?}", deref_base_ty);
                    // we don't care to report errors for a struct if the struct itself is tainted
                    if let Err(guar) = base_def.non_enum_variant().has_errors() {
                        return Ty::new_error(self.tcx(), guar);
                    }

                    let fn_body_hir_id = self.tcx.local_def_id_to_hir_id(self.body_id);
                    let (ident, def_scope) =
                        self.tcx.adjust_ident_and_get_scope(field, base_def.did(), fn_body_hir_id);

                    if let Some((idx, field)) = self.find_adt_field(*base_def, ident) {
                        self.write_field_index(expr.hir_id, idx);

                        let adjustments = self.adjust_steps(&autoderef);
                        if field.vis.is_accessible_from(def_scope, self.tcx) {
                            self.apply_adjustments(base, adjustments);
                            self.register_predicates(autoderef.into_obligations());

                            self.tcx.check_stability(field.did, Some(expr.hir_id), expr.span, None);
                            return self.field_ty(expr.span, field, args);
                        }

                        // The field is not accessible, fall through to error reporting.
                        private_candidate = Some((adjustments, base_def.did()));
                    }
                }
                ty::Tuple(tys) => {
                    if let Ok(index) = field.as_str().parse::<usize>() {
                        if field.name == sym::integer(index) {
                            if let Some(&field_ty) = tys.get(index) {
                                let adjustments = self.adjust_steps(&autoderef);
                                self.apply_adjustments(base, adjustments);
                                self.register_predicates(autoderef.into_obligations());

                                self.write_field_index(expr.hir_id, FieldIdx::from_usize(index));
                                return field_ty;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        // We failed to check the expression, report an error.

        // Emits an error if we deref an infer variable, like calling `.field` on a base type
        // of `&_`. We can also use this to suppress unnecessary "missing field" errors that
        // will follow ambiguity errors.
        let final_ty = self.structurally_resolve_type(autoderef.span(), autoderef.final_ty(false));
        if let ty::Error(_) = final_ty.kind() {
            return final_ty;
        }

        if let Some((adjustments, did)) = private_candidate {
            // (#90483) apply adjustments to avoid ExprUseVisitor from
            // creating erroneous projection.
            self.apply_adjustments(base, adjustments);
            let guar = self.ban_private_field_access(
                expr,
                base_ty,
                field,
                did,
                expected.only_has_type(self),
            );
            return Ty::new_error(self.tcx(), guar);
        }

        let guar = if self.method_exists_for_diagnostic(
            field,
            base_ty,
            expr.hir_id,
            expected.only_has_type(self),
        ) {
            // If taking a method instead of calling it
            self.ban_take_value_of_method(expr, base_ty, field)
        } else if !base_ty.is_primitive_ty() {
            self.ban_nonexisting_field(field, base, expr, base_ty)
        } else {
            let field_name = field.to_string();
            let mut err = type_error_struct!(
                self.dcx(),
                field.span,
                base_ty,
                E0610,
                "`{base_ty}` is a primitive type and therefore doesn't have fields",
            );
            let is_valid_suffix = |field: &str| {
                if field == "f32" || field == "f64" {
                    return true;
                }
                let mut chars = field.chars().peekable();
                match chars.peek() {
                    Some('e') | Some('E') => {
                        chars.next();
                        if let Some(c) = chars.peek()
                            && !c.is_numeric()
                            && *c != '-'
                            && *c != '+'
                        {
                            return false;
                        }
                        while let Some(c) = chars.peek() {
                            if !c.is_numeric() {
                                break;
                            }
                            chars.next();
                        }
                    }
                    _ => (),
                }
                let suffix = chars.collect::<String>();
                suffix.is_empty() || suffix == "f32" || suffix == "f64"
            };
            let maybe_partial_suffix = |field: &str| -> Option<&str> {
                let first_chars = ['f', 'l'];
                if field.len() >= 1
                    && field.to_lowercase().starts_with(first_chars)
                    && field[1..].chars().all(|c| c.is_ascii_digit())
                {
                    if field.to_lowercase().starts_with(['f']) { Some("f32") } else { Some("f64") }
                } else {
                    None
                }
            };
            if let ty::Infer(ty::IntVar(_)) = base_ty.kind()
                && let ExprKind::Lit(Spanned {
                    node: ast::LitKind::Int(_, ast::LitIntType::Unsuffixed),
                    ..
                }) = base.kind
                && !base.span.from_expansion()
            {
                if is_valid_suffix(&field_name) {
                    err.span_suggestion_verbose(
                        field.span.shrink_to_lo(),
                        "if intended to be a floating point literal, consider adding a `0` after the period",
                        '0',
                        Applicability::MaybeIncorrect,
                    );
                } else if let Some(correct_suffix) = maybe_partial_suffix(&field_name) {
                    err.span_suggestion_verbose(
                        field.span,
                        format!("if intended to be a floating point literal, consider adding a `0` after the period and a `{correct_suffix}` suffix"),
                        format!("0{correct_suffix}"),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            err.emit()
        };

        Ty::new_error(self.tcx(), guar)
    }

    fn suggest_await_on_field_access(
        &self,
        err: &mut Diag<'_>,
        field_ident: Ident,
        base: &'tcx hir::Expr<'tcx>,
        ty: Ty<'tcx>,
    ) {
        let Some(output_ty) = self.err_ctxt().get_impl_future_output_ty(ty) else {
            err.span_label(field_ident.span, "unknown field");
            return;
        };
        let ty::Adt(def, _) = output_ty.kind() else {
            err.span_label(field_ident.span, "unknown field");
            return;
        };
        // no field access on enum type
        if def.is_enum() {
            err.span_label(field_ident.span, "unknown field");
            return;
        }
        if !def.non_enum_variant().fields.iter().any(|field| field.ident(self.tcx) == field_ident) {
            err.span_label(field_ident.span, "unknown field");
            return;
        }
        err.span_label(
            field_ident.span,
            "field not available in `impl Future`, but it is available in its `Output`",
        );
        match self.tcx.coroutine_kind(self.body_id) {
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)) => {
                err.span_suggestion_verbose(
                    base.span.shrink_to_hi(),
                    "consider `await`ing on the `Future` to access the field",
                    ".await",
                    Applicability::MaybeIncorrect,
                );
            }
            _ => {
                let mut span: MultiSpan = base.span.into();
                span.push_span_label(self.tcx.def_span(self.body_id), "this is not `async`");
                err.span_note(
                    span,
                    "this implements `Future` and its output type has the field, \
                    but the future cannot be awaited in a synchronous function",
                );
            }
        }
    }

    fn ban_nonexisting_field(
        &self,
        ident: Ident,
        base: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        base_ty: Ty<'tcx>,
    ) -> ErrorGuaranteed {
        debug!(
            "ban_nonexisting_field: field={:?}, base={:?}, expr={:?}, base_ty={:?}",
            ident, base, expr, base_ty
        );
        let mut err = self.no_such_field_err(ident, base_ty, expr);

        match *base_ty.peel_refs().kind() {
            ty::Array(_, len) => {
                self.maybe_suggest_array_indexing(&mut err, base, ident, len);
            }
            ty::RawPtr(..) => {
                self.suggest_first_deref_field(&mut err, base, ident);
            }
            ty::Param(param_ty) => {
                err.span_label(ident.span, "unknown field");
                self.point_at_param_definition(&mut err, param_ty);
            }
            ty::Alias(ty::Opaque, _) => {
                self.suggest_await_on_field_access(&mut err, ident, base, base_ty.peel_refs());
            }
            _ => {
                err.span_label(ident.span, "unknown field");
            }
        }

        self.suggest_fn_call(&mut err, base, base_ty, |output_ty| {
            if let ty::Adt(def, _) = output_ty.kind()
                && !def.is_enum()
            {
                def.non_enum_variant().fields.iter().any(|field| {
                    field.ident(self.tcx) == ident
                        && field.vis.is_accessible_from(expr.hir_id.owner.def_id, self.tcx)
                })
            } else if let ty::Tuple(tys) = output_ty.kind()
                && let Ok(idx) = ident.as_str().parse::<usize>()
            {
                idx < tys.len()
            } else {
                false
            }
        });

        if ident.name == kw::Await {
            // We know by construction that `<expr>.await` is either on Rust 2015
            // or results in `ExprKind::Await`. Suggest switching the edition to 2018.
            err.note("to `.await` a `Future`, switch to Rust 2018 or later");
            HelpUseLatestEdition::new().add_to_diag(&mut err);
        }

        err.emit()
    }

    fn ban_private_field_access(
        &self,
        expr: &hir::Expr<'tcx>,
        expr_t: Ty<'tcx>,
        field: Ident,
        base_did: DefId,
        return_ty: Option<Ty<'tcx>>,
    ) -> ErrorGuaranteed {
        let mut err = self.private_field_err(field, base_did);

        // Also check if an accessible method exists, which is often what is meant.
        if self.method_exists_for_diagnostic(field, expr_t, expr.hir_id, return_ty)
            && !self.expr_in_place(expr.hir_id)
        {
            self.suggest_method_call(
                &mut err,
                format!("a method `{field}` also exists, call it with parentheses"),
                field,
                expr_t,
                expr,
                None,
            );
        }
        err.emit()
    }

    fn ban_take_value_of_method(
        &self,
        expr: &hir::Expr<'tcx>,
        expr_t: Ty<'tcx>,
        field: Ident,
    ) -> ErrorGuaranteed {
        let mut err = type_error_struct!(
            self.dcx(),
            field.span,
            expr_t,
            E0615,
            "attempted to take value of method `{field}` on type `{expr_t}`",
        );
        err.span_label(field.span, "method, not a field");
        let expr_is_call =
            if let hir::Node::Expr(hir::Expr { kind: ExprKind::Call(callee, _args), .. }) =
                self.tcx.parent_hir_node(expr.hir_id)
            {
                expr.hir_id == callee.hir_id
            } else {
                false
            };
        let expr_snippet =
            self.tcx.sess.source_map().span_to_snippet(expr.span).unwrap_or_default();
        let is_wrapped = expr_snippet.starts_with('(') && expr_snippet.ends_with(')');
        let after_open = expr.span.lo() + rustc_span::BytePos(1);
        let before_close = expr.span.hi() - rustc_span::BytePos(1);

        if expr_is_call && is_wrapped {
            err.multipart_suggestion(
                "remove wrapping parentheses to call the method",
                vec![
                    (expr.span.with_hi(after_open), String::new()),
                    (expr.span.with_lo(before_close), String::new()),
                ],
                Applicability::MachineApplicable,
            );
        } else if !self.expr_in_place(expr.hir_id) {
            // Suggest call parentheses inside the wrapping parentheses
            let span = if is_wrapped {
                expr.span.with_lo(after_open).with_hi(before_close)
            } else {
                expr.span
            };
            self.suggest_method_call(
                &mut err,
                "use parentheses to call the method",
                field,
                expr_t,
                expr,
                Some(span),
            );
        } else if let ty::RawPtr(ptr_ty, _) = expr_t.kind()
            && let ty::Adt(adt_def, _) = ptr_ty.kind()
            && let ExprKind::Field(base_expr, _) = expr.kind
            && let [variant] = &adt_def.variants().raw
            && variant.fields.iter().any(|f| f.ident(self.tcx) == field)
        {
            err.multipart_suggestion(
                "to access the field, dereference first",
                vec![
                    (base_expr.span.shrink_to_lo(), "(*".to_string()),
                    (base_expr.span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MaybeIncorrect,
            );
        } else {
            err.help("methods are immutable and cannot be assigned to");
        }

        // See `StashKey::GenericInFieldExpr` for more info
        self.dcx().try_steal_replace_and_emit_err(field.span, StashKey::GenericInFieldExpr, err)
    }

    fn point_at_param_definition(&self, err: &mut Diag<'_>, param: ty::ParamTy) {
        let generics = self.tcx.generics_of(self.body_id);
        let generic_param = generics.type_param(param, self.tcx);
        if let ty::GenericParamDefKind::Type { synthetic: true, .. } = generic_param.kind {
            return;
        }
        let param_def_id = generic_param.def_id;
        let param_hir_id = match param_def_id.as_local() {
            Some(x) => self.tcx.local_def_id_to_hir_id(x),
            None => return,
        };
        let param_span = self.tcx.hir_span(param_hir_id);
        let param_name = self.tcx.hir_ty_param_name(param_def_id.expect_local());

        err.span_label(param_span, format!("type parameter '{param_name}' declared here"));
    }

    fn maybe_suggest_array_indexing(
        &self,
        err: &mut Diag<'_>,
        base: &hir::Expr<'_>,
        field: Ident,
        len: ty::Const<'tcx>,
    ) {
        err.span_label(field.span, "unknown field");
        if let (Some(len), Ok(user_index)) = (
            self.try_structurally_resolve_const(base.span, len).try_to_target_usize(self.tcx),
            field.as_str().parse::<u64>(),
        ) {
            let help = "instead of using tuple indexing, use array indexing";
            let applicability = if len < user_index {
                Applicability::MachineApplicable
            } else {
                Applicability::MaybeIncorrect
            };
            err.multipart_suggestion(
                help,
                vec![
                    (base.span.between(field.span), "[".to_string()),
                    (field.span.shrink_to_hi(), "]".to_string()),
                ],
                applicability,
            );
        }
    }

    fn suggest_first_deref_field(&self, err: &mut Diag<'_>, base: &hir::Expr<'_>, field: Ident) {
        err.span_label(field.span, "unknown field");
        let val = if let Ok(base) = self.tcx.sess.source_map().span_to_snippet(base.span)
            && base.len() < 20
        {
            format!("`{base}`")
        } else {
            "the value".to_string()
        };
        err.multipart_suggestion(
            format!("{val} is a raw pointer; try dereferencing it"),
            vec![
                (base.span.shrink_to_lo(), "(*".into()),
                (base.span.between(field.span), format!(").")),
            ],
            Applicability::MaybeIncorrect,
        );
    }

    fn no_such_field_err(
        &self,
        field: Ident,
        base_ty: Ty<'tcx>,
        expr: &hir::Expr<'tcx>,
    ) -> Diag<'_> {
        let span = field.span;
        debug!("no_such_field_err(span: {:?}, field: {:?}, expr_t: {:?})", span, field, base_ty);

        let mut err = self.dcx().create_err(NoFieldOnType { span, ty: base_ty, field });
        if base_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        if let Some(within_macro_span) = span.within_macro(expr.span, self.tcx.sess.source_map()) {
            err.span_label(within_macro_span, "due to this macro variable");
        }

        // try to add a suggestion in case the field is a nested field of a field of the Adt
        let mod_id = self.tcx.parent_module(expr.hir_id).to_def_id();
        let (ty, unwrap) = if let ty::Adt(def, args) = base_ty.kind()
            && (self.tcx.is_diagnostic_item(sym::Result, def.did())
                || self.tcx.is_diagnostic_item(sym::Option, def.did()))
            && let Some(arg) = args.get(0)
            && let Some(ty) = arg.as_type()
        {
            (ty, "unwrap().")
        } else {
            (base_ty, "")
        };
        for (found_fields, args) in
            self.get_field_candidates_considering_privacy_for_diag(span, ty, mod_id, expr.hir_id)
        {
            let field_names = found_fields.iter().map(|field| field.name).collect::<Vec<_>>();
            let mut candidate_fields: Vec<_> = found_fields
                .into_iter()
                .filter_map(|candidate_field| {
                    self.check_for_nested_field_satisfying_condition_for_diag(
                        span,
                        &|candidate_field, _| candidate_field.ident(self.tcx()) == field,
                        candidate_field,
                        args,
                        vec![],
                        mod_id,
                        expr.hir_id,
                    )
                })
                .map(|mut field_path| {
                    field_path.pop();
                    field_path.iter().map(|id| format!("{}.", id)).collect::<String>()
                })
                .collect::<Vec<_>>();
            candidate_fields.sort();

            let len = candidate_fields.len();
            // Don't suggest `.field` if the base expr is from a different
            // syntax context than the field.
            if len > 0 && expr.span.eq_ctxt(field.span) {
                err.span_suggestions(
                    field.span.shrink_to_lo(),
                    format!(
                        "{} of the expressions' fields {} a field of the same name",
                        if len > 1 { "some" } else { "one" },
                        if len > 1 { "have" } else { "has" },
                    ),
                    candidate_fields.iter().map(|path| format!("{unwrap}{path}")),
                    Applicability::MaybeIncorrect,
                );
            } else if let Some(field_name) =
                find_best_match_for_name(&field_names, field.name, None)
            {
                err.span_suggestion_verbose(
                    field.span,
                    "a field with a similar name exists",
                    format!("{unwrap}{}", field_name),
                    Applicability::MaybeIncorrect,
                );
            } else if !field_names.is_empty() {
                let is = if field_names.len() == 1 { " is" } else { "s are" };
                err.note(
                    format!("available field{is}: {}", self.name_series_display(field_names),),
                );
            }
        }
        err
    }

    fn private_field_err(&self, field: Ident, base_did: DefId) -> Diag<'_> {
        let struct_path = self.tcx().def_path_str(base_did);
        let kind_name = self.tcx().def_descr(base_did);
        struct_span_code_err!(
            self.dcx(),
            field.span,
            E0616,
            "field `{field}` of {kind_name} `{struct_path}` is private",
        )
        .with_span_label(field.span, "private field")
    }

    pub(crate) fn get_field_candidates_considering_privacy_for_diag(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
        mod_id: DefId,
        hir_id: HirId,
    ) -> Vec<(Vec<&'tcx ty::FieldDef>, GenericArgsRef<'tcx>)> {
        debug!("get_field_candidates(span: {:?}, base_t: {:?}", span, base_ty);

        let mut autoderef = self.autoderef(span, base_ty).silence_errors();
        let deref_chain: Vec<_> = autoderef.by_ref().collect();

        // Don't probe if we hit the recursion limit, since it may result in
        // quadratic blowup if we then try to further deref the results of this
        // function. This is a best-effort method, after all.
        if autoderef.reached_recursion_limit() {
            return vec![];
        }

        deref_chain
            .into_iter()
            .filter_map(move |(base_t, _)| {
                match base_t.kind() {
                    ty::Adt(base_def, args) if !base_def.is_enum() => {
                        let tcx = self.tcx;
                        let fields = &base_def.non_enum_variant().fields;
                        // Some struct, e.g. some that impl `Deref`, have all private fields
                        // because you're expected to deref them to access the _real_ fields.
                        // This, for example, will help us suggest accessing a field through a `Box<T>`.
                        if fields.iter().all(|field| !field.vis.is_accessible_from(mod_id, tcx)) {
                            return None;
                        }
                        return Some((
                            fields
                                .iter()
                                .filter(move |field| {
                                    field.vis.is_accessible_from(mod_id, tcx)
                                        && self.is_field_suggestable(field, hir_id, span)
                                })
                                // For compile-time reasons put a limit on number of fields we search
                                .take(100)
                                .collect::<Vec<_>>(),
                            *args,
                        ));
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// This method is called after we have encountered a missing field error to recursively
    /// search for the field
    pub(crate) fn check_for_nested_field_satisfying_condition_for_diag(
        &self,
        span: Span,
        matches: &impl Fn(&ty::FieldDef, Ty<'tcx>) -> bool,
        candidate_field: &ty::FieldDef,
        subst: GenericArgsRef<'tcx>,
        mut field_path: Vec<Ident>,
        mod_id: DefId,
        hir_id: HirId,
    ) -> Option<Vec<Ident>> {
        debug!(
            "check_for_nested_field_satisfying(span: {:?}, candidate_field: {:?}, field_path: {:?}",
            span, candidate_field, field_path
        );

        if field_path.len() > 3 {
            // For compile-time reasons and to avoid infinite recursion we only check for fields
            // up to a depth of three
            None
        } else {
            field_path.push(candidate_field.ident(self.tcx).normalize_to_macros_2_0());
            let field_ty = candidate_field.ty(self.tcx, subst);
            if matches(candidate_field, field_ty) {
                return Some(field_path);
            } else {
                for (nested_fields, subst) in self
                    .get_field_candidates_considering_privacy_for_diag(
                        span, field_ty, mod_id, hir_id,
                    )
                {
                    // recursively search fields of `candidate_field` if it's a ty::Adt
                    for field in nested_fields {
                        if let Some(field_path) = self
                            .check_for_nested_field_satisfying_condition_for_diag(
                                span,
                                matches,
                                field,
                                subst,
                                field_path.clone(),
                                mod_id,
                                hir_id,
                            )
                        {
                            return Some(field_path);
                        }
                    }
                }
            }
            None
        }
    }

    fn check_expr_index(
        &self,
        base: &'tcx hir::Expr<'tcx>,
        idx: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        brackets_span: Span,
    ) -> Ty<'tcx> {
        let base_t = self.check_expr(base);
        let idx_t = self.check_expr(idx);

        if base_t.references_error() {
            base_t
        } else if idx_t.references_error() {
            idx_t
        } else {
            let base_t = self.structurally_resolve_type(base.span, base_t);
            match self.lookup_indexing(expr, base, base_t, idx, idx_t) {
                Some((index_ty, element_ty)) => {
                    // two-phase not needed because index_ty is never mutable
                    self.demand_coerce(idx, idx_t, index_ty, None, AllowTwoPhase::No);
                    self.select_obligations_where_possible(|errors| {
                        self.point_at_index(errors, idx.span);
                    });
                    element_ty
                }
                None => {
                    // Attempt to *shallowly* search for an impl which matches,
                    // but has nested obligations which are unsatisfied.
                    for (base_t, _) in self.autoderef(base.span, base_t).silence_errors() {
                        if let Some((_, index_ty, element_ty)) =
                            self.find_and_report_unsatisfied_index_impl(base, base_t)
                        {
                            self.demand_coerce(idx, idx_t, index_ty, None, AllowTwoPhase::No);
                            return element_ty;
                        }
                    }

                    let mut err = type_error_struct!(
                        self.dcx(),
                        brackets_span,
                        base_t,
                        E0608,
                        "cannot index into a value of type `{base_t}`",
                    );
                    // Try to give some advice about indexing tuples.
                    if let ty::Tuple(types) = base_t.kind() {
                        let mut needs_note = true;
                        // If the index is an integer, we can show the actual
                        // fixed expression:
                        if let ExprKind::Lit(lit) = idx.kind
                            && let ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) = lit.node
                            && i.get()
                                < types
                                    .len()
                                    .try_into()
                                    .expect("expected tuple index to be < usize length")
                        {
                            err.span_suggestion(
                                brackets_span,
                                "to access tuple elements, use",
                                format!(".{i}"),
                                Applicability::MachineApplicable,
                            );
                            needs_note = false;
                        } else if let ExprKind::Path(..) = idx.peel_borrows().kind {
                            err.span_label(
                                idx.span,
                                "cannot access tuple elements at a variable index",
                            );
                        }
                        if needs_note {
                            err.help(
                                "to access tuple elements, use tuple indexing \
                                        syntax (e.g., `tuple.0`)",
                            );
                        }
                    }

                    if base_t.is_raw_ptr() && idx_t.is_integral() {
                        err.multipart_suggestion(
                            "consider using `wrapping_add` or `add` for indexing into raw pointer",
                            vec![
                                (base.span.between(idx.span), ".wrapping_add(".to_owned()),
                                (
                                    idx.span.shrink_to_hi().until(expr.span.shrink_to_hi()),
                                    ")".to_owned(),
                                ),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                    }

                    let reported = err.emit();
                    Ty::new_error(self.tcx, reported)
                }
            }
        }
    }

    /// Try to match an implementation of `Index` against a self type, and report
    /// the unsatisfied predicates that result from confirming this impl.
    ///
    /// Given an index expression, sometimes the `Self` type shallowly but does not
    /// deeply satisfy an impl predicate. Instead of simply saying that the type
    /// does not support being indexed, we want to point out exactly what nested
    /// predicates cause this to be, so that the user can add them to fix their code.
    fn find_and_report_unsatisfied_index_impl(
        &self,
        base_expr: &hir::Expr<'_>,
        base_ty: Ty<'tcx>,
    ) -> Option<(ErrorGuaranteed, Ty<'tcx>, Ty<'tcx>)> {
        let index_trait_def_id = self.tcx.lang_items().index_trait()?;
        let index_trait_output_def_id = self.tcx.get_diagnostic_item(sym::IndexOutput)?;

        let mut relevant_impls = vec![];
        self.tcx.for_each_relevant_impl(index_trait_def_id, base_ty, |impl_def_id| {
            relevant_impls.push(impl_def_id);
        });
        let [impl_def_id] = relevant_impls[..] else {
            // Only report unsatisfied impl predicates if there's one impl
            return None;
        };

        self.commit_if_ok(|snapshot| {
            let outer_universe = self.universe();

            let ocx = ObligationCtxt::new_with_diagnostics(self);
            let impl_args = self.fresh_args_for_item(base_expr.span, impl_def_id);
            let impl_trait_ref =
                self.tcx.impl_trait_ref(impl_def_id).unwrap().instantiate(self.tcx, impl_args);
            let cause = self.misc(base_expr.span);

            // Match the impl self type against the base ty. If this fails,
            // we just skip this impl, since it's not particularly useful.
            let impl_trait_ref = ocx.normalize(&cause, self.param_env, impl_trait_ref);
            ocx.eq(&cause, self.param_env, base_ty, impl_trait_ref.self_ty())?;

            // Register the impl's predicates. One of these predicates
            // must be unsatisfied, or else we wouldn't have gotten here
            // in the first place.
            ocx.register_obligations(traits::predicates_for_generics(
                |idx, span| {
                    cause.clone().derived_cause(
                        ty::Binder::dummy(ty::TraitPredicate {
                            trait_ref: impl_trait_ref,
                            polarity: ty::PredicatePolarity::Positive,
                        }),
                        |derived| {
                            ObligationCauseCode::ImplDerived(Box::new(traits::ImplDerivedCause {
                                derived,
                                impl_or_alias_def_id: impl_def_id,
                                impl_def_predicate_index: Some(idx),
                                span,
                            }))
                        },
                    )
                },
                self.param_env,
                self.tcx.predicates_of(impl_def_id).instantiate(self.tcx, impl_args),
            ));

            // Normalize the output type, which we can use later on as the
            // return type of the index expression...
            let element_ty = ocx.normalize(
                &cause,
                self.param_env,
                Ty::new_projection_from_args(
                    self.tcx,
                    index_trait_output_def_id,
                    impl_trait_ref.args,
                ),
            );

            let true_errors = ocx.select_where_possible();

            // Do a leak check -- we can't really report a useful error here,
            // but it at least avoids an ICE when the error has to do with higher-ranked
            // lifetimes.
            self.leak_check(outer_universe, Some(snapshot))?;

            // Bail if we have ambiguity errors, which we can't report in a useful way.
            let ambiguity_errors = ocx.select_all_or_error();
            if true_errors.is_empty() && !ambiguity_errors.is_empty() {
                return Err(NoSolution);
            }

            // There should be at least one error reported. If not, we
            // will still delay a span bug in `report_fulfillment_errors`.
            Ok::<_, NoSolution>((
                self.err_ctxt().report_fulfillment_errors(true_errors),
                impl_trait_ref.args.type_at(1),
                element_ty,
            ))
        })
        .ok()
    }

    fn point_at_index(&self, errors: &mut Vec<traits::FulfillmentError<'tcx>>, span: Span) {
        let mut seen_preds = FxHashSet::default();
        // We re-sort here so that the outer most root obligations comes first, as we have the
        // subsequent weird logic to identify *every* relevant obligation for proper deduplication
        // of diagnostics.
        errors.sort_by_key(|error| error.root_obligation.recursion_depth);
        for error in errors {
            match (
                error.root_obligation.predicate.kind().skip_binder(),
                error.obligation.predicate.kind().skip_binder(),
            ) {
                (ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)), _)
                    if self.tcx.is_lang_item(predicate.trait_ref.def_id, LangItem::Index) =>
                {
                    seen_preds.insert(error.obligation.predicate.kind().skip_binder());
                }
                (_, ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)))
                    if self.tcx.is_diagnostic_item(sym::SliceIndex, predicate.trait_ref.def_id) =>
                {
                    seen_preds.insert(error.obligation.predicate.kind().skip_binder());
                }
                (root, pred) if seen_preds.contains(&pred) || seen_preds.contains(&root) => {}
                _ => continue,
            }
            error.obligation.cause.span = span;
        }
    }

    fn check_expr_yield(
        &self,
        value: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        match self.coroutine_types {
            Some(CoroutineTypes { resume_ty, yield_ty }) => {
                self.check_expr_coercible_to_type(value, yield_ty, None);

                resume_ty
            }
            _ => {
                self.dcx().emit_err(YieldExprOutsideOfCoroutine { span: expr.span });
                // Avoid expressions without types during writeback (#78653).
                self.check_expr(value);
                self.tcx.types.unit
            }
        }
    }

    fn check_expr_asm_operand(&self, expr: &'tcx hir::Expr<'tcx>, is_input: bool) {
        let needs = if is_input { Needs::None } else { Needs::MutPlace };
        let ty = self.check_expr_with_needs(expr, needs);
        self.require_type_is_sized(ty, expr.span, ObligationCauseCode::InlineAsmSized);

        if !is_input && !expr.is_syntactic_place_expr() {
            self.dcx()
                .struct_span_err(expr.span, "invalid asm output")
                .with_span_label(expr.span, "cannot assign to this expression")
                .emit();
        }

        // If this is an input value, we require its type to be fully resolved
        // at this point. This allows us to provide helpful coercions which help
        // pass the type candidate list in a later pass.
        //
        // We don't require output types to be resolved at this point, which
        // allows them to be inferred based on how they are used later in the
        // function.
        if is_input {
            let ty = self.structurally_resolve_type(expr.span, ty);
            match *ty.kind() {
                ty::FnDef(..) => {
                    let fnptr_ty = Ty::new_fn_ptr(self.tcx, ty.fn_sig(self.tcx));
                    self.demand_coerce(expr, ty, fnptr_ty, None, AllowTwoPhase::No);
                }
                ty::Ref(_, base_ty, mutbl) => {
                    let ptr_ty = Ty::new_ptr(self.tcx, base_ty, mutbl);
                    self.demand_coerce(expr, ty, ptr_ty, None, AllowTwoPhase::No);
                }
                _ => {}
            }
        }
    }

    fn check_expr_asm(&self, asm: &'tcx hir::InlineAsm<'tcx>, span: Span) -> Ty<'tcx> {
        if let rustc_ast::AsmMacro::NakedAsm = asm.asm_macro {
            if !self.tcx.has_attr(self.body_id, sym::naked) {
                self.tcx.dcx().emit_err(NakedAsmOutsideNakedFn { span });
            }
        }

        let mut diverge = asm.asm_macro.diverges(asm.options);

        for (op, _op_sp) in asm.operands {
            match *op {
                hir::InlineAsmOperand::In { expr, .. } => {
                    self.check_expr_asm_operand(expr, true);
                }
                hir::InlineAsmOperand::Out { expr: Some(expr), .. }
                | hir::InlineAsmOperand::InOut { expr, .. } => {
                    self.check_expr_asm_operand(expr, false);
                }
                hir::InlineAsmOperand::Out { expr: None, .. } => {}
                hir::InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    self.check_expr_asm_operand(in_expr, true);
                    if let Some(out_expr) = out_expr {
                        self.check_expr_asm_operand(out_expr, false);
                    }
                }
                hir::InlineAsmOperand::Const { ref anon_const } => {
                    self.check_expr_const_block(anon_const, Expectation::NoExpectation);
                }
                hir::InlineAsmOperand::SymFn { expr } => {
                    self.check_expr(expr);
                }
                hir::InlineAsmOperand::SymStatic { .. } => {}
                hir::InlineAsmOperand::Label { block } => {
                    let previous_diverges = self.diverges.get();

                    // The label blocks should have unit return value or diverge.
                    let ty = self.check_expr_block(block, ExpectHasType(self.tcx.types.unit));
                    if !ty.is_never() {
                        self.demand_suptype(block.span, self.tcx.types.unit, ty);
                        diverge = false;
                    }

                    // We need this to avoid false unreachable warning when a label diverges.
                    self.diverges.set(previous_diverges);
                }
            }
        }

        if diverge { self.tcx.types.never } else { self.tcx.types.unit }
    }

    fn check_expr_offset_of(
        &self,
        container: &'tcx hir::Ty<'tcx>,
        fields: &[Ident],
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let container = self.lower_ty(container).normalized;

        let mut field_indices = Vec::with_capacity(fields.len());
        let mut current_container = container;
        let mut fields = fields.into_iter();

        while let Some(&field) = fields.next() {
            let container = self.structurally_resolve_type(expr.span, current_container);

            match container.kind() {
                ty::Adt(container_def, args) if container_def.is_enum() => {
                    let block = self.tcx.local_def_id_to_hir_id(self.body_id);
                    let (ident, _def_scope) =
                        self.tcx.adjust_ident_and_get_scope(field, container_def.did(), block);

                    if !self.tcx.features().offset_of_enum() {
                        rustc_session::parse::feature_err(
                            &self.tcx.sess,
                            sym::offset_of_enum,
                            ident.span,
                            "using enums in offset_of is experimental",
                        )
                        .emit();
                    }

                    let Some((index, variant)) = container_def
                        .variants()
                        .iter_enumerated()
                        .find(|(_, v)| v.ident(self.tcx).normalize_to_macros_2_0() == ident)
                    else {
                        self.dcx()
                            .create_err(NoVariantNamed { span: ident.span, ident, ty: container })
                            .with_span_label(field.span, "variant not found")
                            .emit_unless(container.references_error());
                        break;
                    };
                    let Some(&subfield) = fields.next() else {
                        type_error_struct!(
                            self.dcx(),
                            ident.span,
                            container,
                            E0795,
                            "`{ident}` is an enum variant; expected field at end of `offset_of`",
                        )
                        .with_span_label(field.span, "enum variant")
                        .emit();
                        break;
                    };
                    let (subident, sub_def_scope) =
                        self.tcx.adjust_ident_and_get_scope(subfield, variant.def_id, block);

                    let Some((subindex, field)) = variant
                        .fields
                        .iter_enumerated()
                        .find(|(_, f)| f.ident(self.tcx).normalize_to_macros_2_0() == subident)
                    else {
                        self.dcx()
                            .create_err(NoFieldOnVariant {
                                span: ident.span,
                                container,
                                ident,
                                field: subfield,
                                enum_span: field.span,
                                field_span: subident.span,
                            })
                            .emit_unless(container.references_error());
                        break;
                    };

                    let field_ty = self.field_ty(expr.span, field, args);

                    // Enums are anyway always sized. But just to safeguard against future
                    // language extensions, let's double-check.
                    self.require_type_is_sized(
                        field_ty,
                        expr.span,
                        ObligationCauseCode::FieldSized {
                            adt_kind: AdtKind::Enum,
                            span: self.tcx.def_span(field.did),
                            last: false,
                        },
                    );

                    if field.vis.is_accessible_from(sub_def_scope, self.tcx) {
                        self.tcx.check_stability(field.did, Some(expr.hir_id), expr.span, None);
                    } else {
                        self.private_field_err(ident, container_def.did()).emit();
                    }

                    // Save the index of all fields regardless of their visibility in case
                    // of error recovery.
                    field_indices.push((index, subindex));
                    current_container = field_ty;

                    continue;
                }
                ty::Adt(container_def, args) => {
                    let block = self.tcx.local_def_id_to_hir_id(self.body_id);
                    let (ident, def_scope) =
                        self.tcx.adjust_ident_and_get_scope(field, container_def.did(), block);

                    let fields = &container_def.non_enum_variant().fields;
                    if let Some((index, field)) = fields
                        .iter_enumerated()
                        .find(|(_, f)| f.ident(self.tcx).normalize_to_macros_2_0() == ident)
                    {
                        let field_ty = self.field_ty(expr.span, field, args);

                        if self.tcx.features().offset_of_slice() {
                            self.require_type_has_static_alignment(field_ty, expr.span);
                        } else {
                            self.require_type_is_sized(
                                field_ty,
                                expr.span,
                                ObligationCauseCode::Misc,
                            );
                        }

                        if field.vis.is_accessible_from(def_scope, self.tcx) {
                            self.tcx.check_stability(field.did, Some(expr.hir_id), expr.span, None);
                        } else {
                            self.private_field_err(ident, container_def.did()).emit();
                        }

                        // Save the index of all fields regardless of their visibility in case
                        // of error recovery.
                        field_indices.push((FIRST_VARIANT, index));
                        current_container = field_ty;

                        continue;
                    }
                }
                ty::Tuple(tys) => {
                    if let Ok(index) = field.as_str().parse::<usize>()
                        && field.name == sym::integer(index)
                    {
                        if let Some(&field_ty) = tys.get(index) {
                            if self.tcx.features().offset_of_slice() {
                                self.require_type_has_static_alignment(field_ty, expr.span);
                            } else {
                                self.require_type_is_sized(
                                    field_ty,
                                    expr.span,
                                    ObligationCauseCode::Misc,
                                );
                            }

                            field_indices.push((FIRST_VARIANT, index.into()));
                            current_container = field_ty;

                            continue;
                        }
                    }
                }
                _ => (),
            };

            self.no_such_field_err(field, container, expr).emit();

            break;
        }

        self.typeck_results
            .borrow_mut()
            .offset_of_data_mut()
            .insert(expr.hir_id, (container, field_indices));

        self.tcx.types.usize
    }
}
