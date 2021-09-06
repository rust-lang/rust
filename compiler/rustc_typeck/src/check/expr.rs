//! Type checking expressions.
//!
//! See `mod.rs` for more context on type checking in general.

use crate::astconv::AstConv as _;
use crate::check::cast;
use crate::check::coercion::CoerceMany;
use crate::check::fatally_break_rust;
use crate::check::method::SelfSource;
use crate::check::report_unexpected_variant_res;
use crate::check::BreakableCtxt;
use crate::check::Diverges;
use crate::check::DynamicCoerceMany;
use crate::check::Expectation::{self, ExpectCastableToType, ExpectHasType, NoExpectation};
use crate::check::FnCtxt;
use crate::check::Needs;
use crate::check::TupleArgumentsFlag::DontTupleArguments;
use crate::errors::{
    FieldMultiplySpecifiedInInitializer, FunctionalRecordUpdateOnNonStruct,
    YieldExprOutsideOfGenerator,
};
use crate::type_error_struct;

use crate::errors::{AddressOfTemporaryTaken, ReturnStmtOutsideOfFnBody, StructExprNonExhaustive};
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::ErrorReported;
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{ExprKind, QPath};
use rustc_infer::infer;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::ty;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AllowTwoPhase};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::Ty;
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{AdtKind, Visibility};
use rustc_span::edition::LATEST_STABLE_EDITION;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::source_map::Span;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_trait_selection::traits::{self, ObligationCauseCode};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn check_expr_eq_type(&self, expr: &'tcx hir::Expr<'tcx>, expected: Ty<'tcx>) {
        let ty = self.check_expr_with_hint(expr, expected);
        self.demand_eqtype(expr.span, expected, ty);
    }

    pub fn check_expr_has_type_or_error(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        extend_err: impl Fn(&mut DiagnosticBuilder<'_>),
    ) -> Ty<'tcx> {
        self.check_expr_meets_expectation_or_error(expr, ExpectHasType(expected), extend_err)
    }

    fn check_expr_meets_expectation_or_error(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        extend_err: impl Fn(&mut DiagnosticBuilder<'_>),
    ) -> Ty<'tcx> {
        let expected_ty = expected.to_option(&self).unwrap_or(self.tcx.types.bool);
        let mut ty = self.check_expr_with_expectation(expr, expected);

        // While we don't allow *arbitrary* coercions here, we *do* allow
        // coercions from ! to `expected`.
        if ty.is_never() {
            assert!(
                !self.typeck_results.borrow().adjustments().contains_key(expr.hir_id),
                "expression with never type wound up being adjusted"
            );
            let adj_ty = self.next_diverging_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::AdjustmentType,
                span: expr.span,
            });
            self.apply_adjustments(
                expr,
                vec![Adjustment { kind: Adjust::NeverToAny, target: adj_ty }],
            );
            ty = adj_ty;
        }

        if let Some(mut err) = self.demand_suptype_diag(expr.span, expected_ty, ty) {
            let expr = expr.peel_drop_temps();
            self.suggest_deref_ref_or_into(&mut err, expr, expected_ty, ty, None);
            extend_err(&mut err);
            // Error possibly reported in `check_assign` so avoid emitting error again.
            err.emit_unless(self.is_assign_to_bool(expr, expected_ty));
        }
        ty
    }

    pub(super) fn check_expr_coercable_to_type(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) -> Ty<'tcx> {
        let ty = self.check_expr_with_hint(expr, expected);
        // checks don't need two phase
        self.demand_coerce(expr, ty, expected, expected_ty_expr, AllowTwoPhase::No)
    }

    pub(super) fn check_expr_with_hint(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, ExpectHasType(expected))
    }

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

    pub(super) fn check_expr(&self, expr: &'tcx hir::Expr<'tcx>) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, NoExpectation)
    }

    pub(super) fn check_expr_with_needs(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        needs: Needs,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_needs(expr, NoExpectation, needs)
    }

    /// Invariant:
    /// If an expression has any sub-expressions that result in a type error,
    /// inspecting that expression's type with `ty.references_error()` will return
    /// true. Likewise, if an expression is known to diverge, inspecting its
    /// type with `ty::type_is_bot` will return true (n.b.: since Rust is
    /// strict, _|_ can appear in the type of an expression that does not,
    /// itself, diverge: for example, fn() -> _|_.)
    /// Note that inspecting a type's structure *directly* may expose the fact
    /// that there are actually multiple representations for `Error`, so avoid
    /// that when err needs to be handled differently.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn check_expr_with_expectation(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        if self.tcx().sess.verbose() {
            // make this code only run with -Zverbose because it is probably slow
            if let Ok(lint_str) = self.tcx.sess.source_map().span_to_snippet(expr.span) {
                if !lint_str.contains('\n') {
                    debug!("expr text: {}", lint_str);
                } else {
                    let mut lines = lint_str.lines();
                    if let Some(line0) = lines.next() {
                        let remaining_lines = lines.count();
                        debug!("expr text: {}", line0);
                        debug!("expr text: ...(and {} more lines)", remaining_lines);
                    }
                }
            }
        }

        // True if `expr` is a `Try::from_ok(())` that is a result of desugaring a try block
        // without the final expr (e.g. `try { return; }`). We don't want to generate an
        // unreachable_code lint for it since warnings for autogenerated code are confusing.
        let is_try_block_generated_unit_expr = match expr.kind {
            ExprKind::Call(_, args) if expr.span.is_desugaring(DesugaringKind::TryBlock) => {
                args.len() == 1 && args[0].span.is_desugaring(DesugaringKind::TryBlock)
            }

            _ => false,
        };

        // Warn for expressions after diverging siblings.
        if !is_try_block_generated_unit_expr {
            self.warn_if_unreachable(expr.hir_id, expr.span, "expression");
        }

        // Hide the outer diverging and has_errors flags.
        let old_diverges = self.diverges.replace(Diverges::Maybe);
        let old_has_errors = self.has_errors.replace(false);

        let ty = ensure_sufficient_stack(|| self.check_expr_kind(expr, expected));

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
            ExprKind::Call(callee, _) => self.warn_if_unreachable(expr.hir_id, callee.span, "call"),
            ExprKind::MethodCall(_, ref span, _, _) => {
                self.warn_if_unreachable(expr.hir_id, *span, "call")
            }
            _ => self.warn_if_unreachable(expr.hir_id, expr.span, "expression"),
        }

        // Any expression that produces a value of type `!` must have diverged
        if ty.is_never() {
            self.diverges.set(self.diverges.get() | Diverges::always(expr.span));
        }

        // Record the type, which applies it effects.
        // We need to do this after the warning above, so that
        // we don't warn for the diverging expression itself.
        self.write_ty(expr.hir_id, ty);

        // Combine the diverging and has_error flags.
        self.diverges.set(self.diverges.get() | old_diverges);
        self.has_errors.set(self.has_errors.get() | old_has_errors);

        debug!("type of {} is...", self.tcx.hir().node_to_string(expr.hir_id));
        debug!("... {:?}, expected is {:?}", ty, expected);

        ty
    }

    fn check_expr_kind(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        debug!("check_expr_kind(expected={:?}, expr={:?})", expected, expr);

        let tcx = self.tcx;
        match expr.kind {
            ExprKind::Box(subexpr) => self.check_expr_box(subexpr, expected),
            ExprKind::Lit(ref lit) => self.check_lit(&lit, expected),
            ExprKind::Binary(op, lhs, rhs) => self.check_binop(expr, op, lhs, rhs),
            ExprKind::Assign(lhs, rhs, ref span) => {
                self.check_expr_assign(expr, expected, lhs, rhs, span)
            }
            ExprKind::AssignOp(op, lhs, rhs) => self.check_binop_assign(expr, op, lhs, rhs),
            ExprKind::Unary(unop, oprnd) => self.check_expr_unary(unop, oprnd, expected, expr),
            ExprKind::AddrOf(kind, mutbl, oprnd) => {
                self.check_expr_addr_of(kind, mutbl, oprnd, expected, expr)
            }
            ExprKind::Path(QPath::LangItem(lang_item, _)) => {
                self.check_lang_item_path(lang_item, expr)
            }
            ExprKind::Path(ref qpath) => self.check_expr_path(qpath, expr),
            ExprKind::InlineAsm(asm) => self.check_expr_asm(asm),
            ExprKind::LlvmInlineAsm(asm) => {
                for expr in asm.outputs_exprs.iter().chain(asm.inputs_exprs.iter()) {
                    self.check_expr(expr);
                }
                tcx.mk_unit()
            }
            ExprKind::Break(destination, ref expr_opt) => {
                self.check_expr_break(destination, expr_opt.as_deref(), expr)
            }
            ExprKind::Continue(destination) => {
                if destination.target_id.is_ok() {
                    tcx.types.never
                } else {
                    // There was an error; make type-check fail.
                    tcx.ty_error()
                }
            }
            ExprKind::Ret(ref expr_opt) => self.check_expr_return(expr_opt.as_deref(), expr),
            ExprKind::Let(pat, let_expr, _) => self.check_expr_let(let_expr, pat),
            ExprKind::Loop(body, _, source, _) => {
                self.check_expr_loop(body, source, expected, expr)
            }
            ExprKind::Match(discrim, arms, match_src) => {
                self.check_match(expr, &discrim, arms, expected, match_src)
            }
            ExprKind::Closure(capture, decl, body_id, _, gen) => {
                self.check_expr_closure(expr, capture, &decl, body_id, gen, expected)
            }
            ExprKind::Block(body, _) => self.check_block_with_expected(&body, expected),
            ExprKind::Call(callee, args) => self.check_call(expr, &callee, args, expected),
            ExprKind::MethodCall(segment, span, args, _) => {
                self.check_method_call(expr, segment, span, args, expected)
            }
            ExprKind::Cast(e, t) => self.check_expr_cast(e, t, expr),
            ExprKind::Type(e, t) => {
                let ty = self.to_ty_saving_user_provided_ty(&t);
                self.check_expr_eq_type(&e, ty);
                ty
            }
            ExprKind::If(cond, then_expr, opt_else_expr) => {
                self.check_then_else(cond, then_expr, opt_else_expr, expr.span, expected)
            }
            ExprKind::DropTemps(e) => self.check_expr_with_expectation(e, expected),
            ExprKind::Array(args) => self.check_expr_array(args, expected, expr),
            ExprKind::ConstBlock(ref anon_const) => self.to_const(anon_const).ty,
            ExprKind::Repeat(element, ref count) => {
                self.check_expr_repeat(element, count, expected, expr)
            }
            ExprKind::Tup(elts) => self.check_expr_tuple(elts, expected, expr),
            ExprKind::Struct(qpath, fields, ref base_expr) => {
                self.check_expr_struct(expr, expected, qpath, fields, base_expr)
            }
            ExprKind::Field(base, field) => self.check_field(expr, &base, field),
            ExprKind::Index(base, idx) => self.check_expr_index(base, idx, expr),
            ExprKind::Yield(value, ref src) => self.check_expr_yield(value, expr, src),
            hir::ExprKind::Err => tcx.ty_error(),
        }
    }

    fn check_expr_box(&self, expr: &'tcx hir::Expr<'tcx>, expected: Expectation<'tcx>) -> Ty<'tcx> {
        let expected_inner = expected.to_option(self).map_or(NoExpectation, |ty| match ty.kind() {
            ty::Adt(def, _) if def.is_box() => Expectation::rvalue_hint(self, ty.boxed_ty()),
            _ => NoExpectation,
        });
        let referent_ty = self.check_expr_with_expectation(expr, expected_inner);
        self.require_type_is_sized(referent_ty, expr.span, traits::SizedBoxType);
        self.tcx.mk_box(referent_ty)
    }

    fn check_expr_unary(
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
        let mut oprnd_t = self.check_expr_with_expectation(&oprnd, expected_inner);

        if !oprnd_t.references_error() {
            oprnd_t = self.structurally_resolved_type(expr.span, oprnd_t);
            match unop {
                hir::UnOp::Deref => {
                    if let Some(ty) = self.lookup_derefing(expr, oprnd, oprnd_t) {
                        oprnd_t = ty;
                    } else {
                        let mut err = type_error_struct!(
                            tcx.sess,
                            expr.span,
                            oprnd_t,
                            E0614,
                            "type `{}` cannot be dereferenced",
                            oprnd_t,
                        );
                        let sp = tcx.sess.source_map().start_point(expr.span);
                        if let Some(sp) =
                            tcx.sess.parse_sess.ambiguous_block_expr_parse.borrow().get(&sp)
                        {
                            tcx.sess.parse_sess.expr_parentheses_needed(&mut err, *sp);
                        }
                        err.emit();
                        oprnd_t = tcx.ty_error();
                    }
                }
                hir::UnOp::Not => {
                    let result = self.check_user_unop(expr, oprnd_t, unop);
                    // If it's builtin, we can reuse the type, this helps inference.
                    if !(oprnd_t.is_integral() || *oprnd_t.kind() == ty::Bool) {
                        oprnd_t = result;
                    }
                }
                hir::UnOp::Neg => {
                    let result = self.check_user_unop(expr, oprnd_t, unop);
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
            match ty.kind() {
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    if oprnd.is_syntactic_place_expr() {
                        // Places may legitimately have unsized types.
                        // For example, dereferences of a fat pointer and
                        // the last field of a struct can be unsized.
                        ExpectHasType(ty)
                    } else {
                        Expectation::rvalue_hint(self, ty)
                    }
                }
                _ => NoExpectation,
            }
        });
        let ty =
            self.check_expr_with_expectation_and_needs(&oprnd, hint, Needs::maybe_mut_place(mutbl));

        let tm = ty::TypeAndMut { ty, mutbl };
        match kind {
            _ if tm.ty.references_error() => self.tcx.ty_error(),
            hir::BorrowKind::Raw => {
                self.check_named_place_expr(oprnd);
                self.tcx.mk_ptr(tm)
            }
            hir::BorrowKind::Ref => {
                // Note: at this point, we cannot say what the best lifetime
                // is to use for resulting pointer.  We want to use the
                // shortest lifetime possible so as to avoid spurious borrowck
                // errors.  Moreover, the longest lifetime will depend on the
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
                let region = self.next_region_var(infer::AddrOfRegion(expr.span));
                self.tcx.mk_ref(region, tm)
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
                .map_or(false, |x| x.iter().any(|adj| matches!(adj.kind, Adjust::Deref(_))))
        });
        if !is_named {
            self.tcx.sess.emit_err(AddressOfTemporaryTaken { span: oprnd.span })
        }
    }

    fn check_lang_item_path(
        &self,
        lang_item: hir::LangItem,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        self.resolve_lang_item_path(lang_item, expr.span, expr.hir_id).1
    }

    fn check_expr_path(
        &self,
        qpath: &'tcx hir::QPath<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let (res, opt_ty, segs) =
            self.resolve_ty_and_res_fully_qualified_call(qpath, expr.hir_id, expr.span);
        let ty = match res {
            Res::Err => {
                self.set_tainted_by_errors();
                tcx.ty_error()
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Fictive), _) => {
                report_unexpected_variant_res(tcx, res, expr.span);
                tcx.ty_error()
            }
            _ => self.instantiate_value_path(segs, opt_ty, res, expr.span, expr.hir_id).0,
        };

        if let ty::FnDef(..) = ty.kind() {
            let fn_sig = ty.fn_sig(tcx);
            if !tcx.features().unsized_fn_params {
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
                    let input = self
                        .replace_bound_vars_with_fresh_vars(
                            expr.span,
                            infer::LateBoundRegionConversionTime::FnCall,
                            fn_sig.input(i),
                        )
                        .0;
                    self.require_type_is_sized_deferred(
                        input,
                        expr.span,
                        traits::SizedArgumentType(None),
                    );
                }
            }
            // Here we want to prevent struct constructors from returning unsized types.
            // There were two cases this happened: fn pointer coercion in stable
            // and usual function call in presence of unsized_locals.
            // Also, as we just want to check sizedness, instead of introducing
            // placeholder lifetimes with probing, we just replace higher lifetimes
            // with fresh vars.
            let output = self
                .replace_bound_vars_with_fresh_vars(
                    expr.span,
                    infer::LateBoundRegionConversionTime::FnCall,
                    fn_sig.output(),
                )
                .0;
            self.require_type_is_sized_deferred(output, expr.span, traits::SizedReturnType);
        }

        // We always require that the type provided as the value for
        // a type parameter outlives the moment of instantiation.
        let substs = self.typeck_results.borrow().node_substs(expr.hir_id);
        self.add_wf_bounds(substs, expr);

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
                    // `ctxt` with the second `encloding_breakables` borrow below.
                    let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
                    match enclosing_breakables.opt_find_breakable(target_id) {
                        Some(ctxt) => ctxt.coerce.as_ref().map(|coerce| coerce.expected_ty()),
                        None => {
                            // Avoid ICE when `break` is inside a closure (#65383).
                            return tcx.ty_error_with_message(
                                expr.span,
                                "break was outside loop, but no error was emitted",
                            );
                        }
                    }
                };

                // If the loop context is not a `loop { }`, then break with
                // a value is illegal, and `opt_coerce_to` will be `None`.
                // Just set expectation to error in that case.
                let coerce_to = opt_coerce_to.unwrap_or_else(|| tcx.ty_error());

                // Recurse without `enclosing_breakables` borrowed.
                e_ty = self.check_expr_with_hint(e, coerce_to);
                cause = self.misc(e.span);
            } else {
                // Otherwise, this is a break *without* a value. That's
                // always legal, and is equivalent to `break ()`.
                e_ty = tcx.mk_unit();
                cause = self.misc(expr.span);
            }

            // Now that we have type-checked `expr_opt`, borrow
            // the `enclosing_loops` field and let's coerce the
            // type of `expr_opt` into what is expected.
            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            let ctxt = match enclosing_breakables.opt_find_breakable(target_id) {
                Some(ctxt) => ctxt,
                None => {
                    // Avoid ICE when `break` is inside a closure (#65383).
                    return tcx.ty_error_with_message(
                        expr.span,
                        "break was outside loop, but no error was emitted",
                    );
                }
            };

            if let Some(ref mut coerce) = ctxt.coerce {
                if let Some(ref e) = expr_opt {
                    coerce.coerce(self, &cause, e, e_ty);
                } else {
                    assert!(e_ty.is_unit());
                    let ty = coerce.expected_ty();
                    coerce.coerce_forced_unit(
                        self,
                        &cause,
                        &mut |mut err| {
                            self.suggest_mismatched_types_on_tail(
                                &mut err, expr, ty, e_ty, target_id,
                            );
                            if let Some(val) = ty_kind_suggestion(ty) {
                                let label = destination
                                    .label
                                    .map(|l| format!(" {}", l.ident))
                                    .unwrap_or_else(String::new);
                                err.span_suggestion(
                                    expr.span,
                                    "give it a value of the expected type",
                                    format!("break{} {}", label, val),
                                    Applicability::HasPlaceholders,
                                );
                            }
                        },
                        false,
                    );
                }
            } else {
                // If `ctxt.coerce` is `None`, we can just ignore
                // the type of the expression.  This is because
                // either this was a break *without* a value, in
                // which case it is always a legal type (`()`), or
                // else an error would have been flagged by the
                // `loops` pass for using break with an expression
                // where you are not supposed to.
                assert!(expr_opt.is_none() || self.tcx.sess.has_errors());
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
            let err = self.tcx.ty_error_with_message(
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
                    if path.segments.len() == 1 && path.segments[0].ident.name == sym::rust {
                        fatally_break_rust(self.tcx.sess);
                    }
                }
            }

            // There was an error; make type-check fail.
            err
        }
    }

    fn check_expr_return(
        &self,
        expr_opt: Option<&'tcx hir::Expr<'tcx>>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        if self.ret_coercion.is_none() {
            let mut err = ReturnStmtOutsideOfFnBody {
                span: expr.span,
                encl_body_span: None,
                encl_fn_span: None,
            };

            let encl_item_id = self.tcx.hir().get_parent_item(expr.hir_id);

            if let Some(hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Fn(..),
                span: encl_fn_span,
                ..
            }))
            | Some(hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(_)),
                span: encl_fn_span,
                ..
            }))
            | Some(hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(..),
                span: encl_fn_span,
                ..
            })) = self.tcx.hir().find(encl_item_id)
            {
                // We are inside a function body, so reporting "return statement
                // outside of function body" needs an explanation.

                let encl_body_owner_id = self.tcx.hir().enclosing_body_owner(expr.hir_id);

                // If this didn't hold, we would not have to report an error in
                // the first place.
                assert_ne!(encl_item_id, encl_body_owner_id);

                let encl_body_id = self.tcx.hir().body_owned_by(encl_body_owner_id);
                let encl_body = self.tcx.hir().body(encl_body_id);

                err.encl_body_span = Some(encl_body.value.span);
                err.encl_fn_span = Some(*encl_fn_span);
            }

            self.tcx.sess.emit_err(err);

            if let Some(e) = expr_opt {
                // We still have to type-check `e` (issue #86188), but calling
                // `check_return_expr` only works inside fn bodies.
                self.check_expr(e);
            }
        } else if let Some(e) = expr_opt {
            if self.ret_coercion_span.get().is_none() {
                self.ret_coercion_span.set(Some(e.span));
            }
            self.check_return_expr(e);
        } else {
            let mut coercion = self.ret_coercion.as_ref().unwrap().borrow_mut();
            if self.ret_coercion_span.get().is_none() {
                self.ret_coercion_span.set(Some(expr.span));
            }
            let cause = self.cause(expr.span, ObligationCauseCode::ReturnNoExpression);
            if let Some((fn_decl, _)) = self.get_fn_decl(expr.hir_id) {
                coercion.coerce_forced_unit(
                    self,
                    &cause,
                    &mut |db| {
                        let span = fn_decl.output.span();
                        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                            db.span_label(
                                span,
                                format!("expected `{}` because of this return type", snippet),
                            );
                        }
                    },
                    true,
                );
            } else {
                coercion.coerce_forced_unit(self, &cause, &mut |_| (), true);
            }
        }
        self.tcx.types.never
    }

    pub(super) fn check_return_expr(&self, return_expr: &'tcx hir::Expr<'tcx>) {
        let ret_coercion = self.ret_coercion.as_ref().unwrap_or_else(|| {
            span_bug!(return_expr.span, "check_return_expr called outside fn body")
        });

        let ret_ty = ret_coercion.borrow().expected_ty();
        let return_expr_ty = self.check_expr_with_hint(return_expr, ret_ty);
        ret_coercion.borrow_mut().coerce(
            self,
            &self.cause(return_expr.span, ObligationCauseCode::ReturnValue(return_expr.hir_id)),
            return_expr,
            return_expr_ty,
        );
    }

    pub(crate) fn check_lhs_assignable(
        &self,
        lhs: &'tcx hir::Expr<'tcx>,
        err_code: &'static str,
        expr_span: &Span,
    ) {
        if lhs.is_syntactic_place_expr() {
            return;
        }

        // FIXME: Make this use SessionDiagnostic once error codes can be dynamically set.
        let mut err = self.tcx.sess.struct_span_err_with_code(
            *expr_span,
            "invalid left-hand side of assignment",
            DiagnosticId::Error(err_code.into()),
        );
        err.span_label(lhs.span, "cannot assign to this expression");
        err.emit();
    }

    // A generic function for checking the 'then' and 'else' clauses in an 'if'
    // or 'if-else' expression.
    fn check_then_else(
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

        let expected = orig_expected.adjust_for_branches(self);
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
            let else_ty = if sp.desugaring_kind() == Some(DesugaringKind::LetElse) {
                // todo introduce `check_expr_with_expectation(.., Expectation::LetElse)`
                //   for errors that point to the offending expression rather than the entire block.
                //   We could use `check_expr_eq_type(.., tcx.types.never)`, but then there is no
                //   way to detect that the expected type originated from let-else and provide
                //   a customized error.
                let else_ty = self.check_expr(else_expr);
                let cause = self.cause(else_expr.span, ObligationCauseCode::LetElse);

                if let Some(mut err) =
                    self.demand_eqtype_with_origin(&cause, self.tcx.types.never, else_ty)
                {
                    err.emit();
                    self.tcx.ty_error()
                } else {
                    else_ty
                }
            } else {
                self.check_expr_with_expectation(else_expr, expected)
            };
            let else_diverges = self.diverges.get();

            let opt_suggest_box_span =
                self.opt_suggest_box_span(else_expr.span, else_ty, orig_expected);
            let if_cause =
                self.if_cause(sp, then_expr, else_expr, then_ty, else_ty, opt_suggest_box_span);

            coerce.coerce(self, &if_cause, else_expr, else_ty);

            // We won't diverge unless both branches do (or the condition does).
            self.diverges.set(cond_diverges | then_diverges & else_diverges);
        } else {
            self.if_fallback_coercion(sp, then_expr, &mut coerce);

            // If the condition is false we can't diverge.
            self.diverges.set(cond_diverges);
        }

        let result_ty = coerce.complete(self);
        if cond_ty.references_error() { self.tcx.ty_error() } else { result_ty }
    }

    /// Type check assignment expression `expr` of form `lhs = rhs`.
    /// The expected type is `()` and is passed to the function for the purposes of diagnostics.
    fn check_expr_assign(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        lhs: &'tcx hir::Expr<'tcx>,
        rhs: &'tcx hir::Expr<'tcx>,
        span: &Span,
    ) -> Ty<'tcx> {
        let expected_ty = expected.coercion_target_type(self, expr.span);
        if expected_ty == self.tcx.types.bool {
            // The expected type is `bool` but this will result in `()` so we can reasonably
            // say that the user intended to write `lhs == rhs` instead of `lhs = rhs`.
            // The likely cause of this is `if foo = bar { .. }`.
            let actual_ty = self.tcx.mk_unit();
            let mut err = self.demand_suptype_diag(expr.span, expected_ty, actual_ty).unwrap();
            let lhs_ty = self.check_expr(&lhs);
            let rhs_ty = self.check_expr(&rhs);
            let (applicability, eq) = if self.can_coerce(rhs_ty, lhs_ty) {
                (Applicability::MachineApplicable, true)
            } else {
                (Applicability::MaybeIncorrect, false)
            };
            if !lhs.is_syntactic_place_expr() {
                // Do not suggest `if let x = y` as `==` is way more likely to be the intention.
                let hir = self.tcx.hir();
                if let hir::Node::Expr(hir::Expr { kind: ExprKind::If { .. }, .. }) =
                    hir.get(hir.get_parent_node(hir.get_parent_node(expr.hir_id)))
                {
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_lo(),
                        "you might have meant to use pattern matching",
                        "let ".to_string(),
                        applicability,
                    );
                }
            }
            if eq {
                err.span_suggestion_verbose(
                    *span,
                    "you might have meant to compare for equality",
                    "==".to_string(),
                    applicability,
                );
            }

            // If the assignment expression itself is ill-formed, don't
            // bother emitting another error
            if lhs_ty.references_error() || rhs_ty.references_error() {
                err.delay_as_bug()
            } else {
                err.emit();
            }
            return self.tcx.ty_error();
        }

        self.check_lhs_assignable(lhs, "E0070", span);

        let lhs_ty = self.check_expr_with_needs(&lhs, Needs::MutPlace);
        let rhs_ty = self.check_expr_coercable_to_type(&rhs, lhs_ty, Some(lhs));

        self.require_type_is_sized(lhs_ty, lhs.span, traits::AssignmentLhsSized);

        if lhs_ty.references_error() || rhs_ty.references_error() {
            self.tcx.ty_error()
        } else {
            self.tcx.mk_unit()
        }
    }

    fn check_expr_let(&self, expr: &'tcx hir::Expr<'tcx>, pat: &'tcx hir::Pat<'tcx>) -> Ty<'tcx> {
        self.warn_if_unreachable(expr.hir_id, expr.span, "block in `let` expression");
        let expr_ty = self.demand_scrutinee_type(expr, pat.contains_explicit_ref_binding(), false);
        self.check_pat_top(pat, expr_ty, Some(expr.span), true);
        self.tcx.types.bool
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
            self.check_block_no_value(&body);
        });

        if ctxt.may_break {
            // No way to know whether it's diverging because
            // of a `break` or an outer `break` or `return`.
            self.diverges.set(Diverges::Maybe);
        }

        // If we permit break with a value, then result type is
        // the LUB of the breaks (possibly ! if none); else, it
        // is nil. This makes sense because infinite loops
        // (which would have type !) are only possible iff we
        // permit break with a value [1].
        if ctxt.coerce.is_none() && !ctxt.may_break {
            // [1]
            self.tcx.sess.delay_span_bug(body.span, "no coercion, but loop may not break");
        }
        ctxt.coerce.map(|c| c.complete(self)).unwrap_or_else(|| self.tcx.mk_unit())
    }

    /// Checks a method call.
    fn check_method_call(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        segment: &hir::PathSegment<'_>,
        span: Span,
        args: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let rcvr = &args[0];
        let rcvr_t = self.check_expr(&rcvr);
        // no need to check for bot/err -- callee does that
        let rcvr_t = self.structurally_resolved_type(args[0].span, rcvr_t);

        let method = match self.lookup_method(rcvr_t, segment, span, expr, rcvr, args) {
            Ok(method) => {
                // We could add a "consider `foo::<params>`" suggestion here, but I wasn't able to
                // trigger this codepath causing `structuraly_resolved_type` to emit an error.

                self.write_method_call(expr.hir_id, method);
                Ok(method)
            }
            Err(error) => {
                if segment.ident.name != kw::Empty {
                    if let Some(mut err) = self.report_method_error(
                        span,
                        rcvr_t,
                        segment.ident,
                        SelfSource::MethodCall(&args[0]),
                        error,
                        Some(args),
                    ) {
                        err.emit();
                    }
                }
                Err(())
            }
        };

        // Call the generic checker.
        self.check_method_argument_types(
            span,
            expr,
            method,
            &args[1..],
            DontTupleArguments,
            expected,
        )
    }

    fn check_expr_cast(
        &self,
        e: &'tcx hir::Expr<'tcx>,
        t: &'tcx hir::Ty<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // Find the type of `e`. Supply hints based on the type we are casting to,
        // if appropriate.
        let t_cast = self.to_ty_saving_user_provided_ty(t);
        let t_cast = self.resolve_vars_if_possible(t_cast);
        let t_expr = self.check_expr_with_expectation(e, ExpectCastableToType(t_cast));
        let t_expr = self.resolve_vars_if_possible(t_expr);

        // Eagerly check for some obvious errors.
        if t_expr.references_error() || t_cast.references_error() {
            self.tcx.ty_error()
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
                Err(ErrorReported) => self.tcx.ty_error(),
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
                .and_then(|uty| match *uty.kind() {
                    ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    self.next_ty_var(TypeVariableOrigin {
                        kind: TypeVariableOriginKind::TypeInference,
                        span: expr.span,
                    })
                });
            let mut coerce = CoerceMany::with_coercion_sites(coerce_to, args);
            assert_eq!(self.diverges.get(), Diverges::Maybe);
            for e in args {
                let e_ty = self.check_expr_with_hint(e, coerce_to);
                let cause = self.misc(e.span);
                coerce.coerce(self, &cause, e, e_ty);
            }
            coerce.complete(self)
        } else {
            self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span: expr.span,
            })
        };
        self.tcx.mk_array(element_ty, args.len() as u64)
    }

    fn check_expr_repeat(
        &self,
        element: &'tcx hir::Expr<'tcx>,
        count: &'tcx hir::AnonConst,
        expected: Expectation<'tcx>,
        _expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let count = self.to_const(count);

        let uty = match expected {
            ExpectHasType(uty) => match *uty.kind() {
                ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                _ => None,
            },
            _ => None,
        };

        let (element_ty, t) = match uty {
            Some(uty) => {
                self.check_expr_coercable_to_type(&element, uty, None);
                (uty, uty)
            }
            None => {
                let ty = self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: element.span,
                });
                let element_ty = self.check_expr_has_type_or_error(&element, ty, |_| {});
                (element_ty, ty)
            }
        };

        if element_ty.references_error() {
            return tcx.ty_error();
        }

        tcx.mk_ty(ty::Array(t, count))
    }

    fn check_expr_tuple(
        &self,
        elts: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        let flds = expected.only_has_type(self).and_then(|ty| {
            let ty = self.resolve_vars_with_obligations(ty);
            match ty.kind() {
                ty::Tuple(flds) => Some(&flds[..]),
                _ => None,
            }
        });

        let elt_ts_iter = elts.iter().enumerate().map(|(i, e)| match flds {
            Some(fs) if i < fs.len() => {
                let ety = fs[i].expect_ty();
                self.check_expr_coercable_to_type(&e, ety, None);
                ety
            }
            _ => self.check_expr_with_expectation(&e, NoExpectation),
        });
        let tuple = self.tcx.mk_tup(elt_ts_iter);
        if tuple.references_error() {
            self.tcx.ty_error()
        } else {
            self.require_type_is_sized(tuple, expr.span, traits::TupleInitializerSized);
            tuple
        }
    }

    fn check_expr_struct(
        &self,
        expr: &hir::Expr<'_>,
        expected: Expectation<'tcx>,
        qpath: &QPath<'_>,
        fields: &'tcx [hir::ExprField<'tcx>],
        base_expr: &'tcx Option<&'tcx hir::Expr<'tcx>>,
    ) -> Ty<'tcx> {
        // Find the relevant variant
        let (variant, adt_ty) = if let Some(variant_ty) = self.check_struct_path(qpath, expr.hir_id)
        {
            variant_ty
        } else {
            self.check_struct_fields_on_error(fields, base_expr);
            return self.tcx.ty_error();
        };

        // Prohibit struct expressions when non-exhaustive flag is set.
        let adt = adt_ty.ty_adt_def().expect("`check_struct_path` returned non-ADT type");
        if !adt.did.is_local() && variant.is_field_list_non_exhaustive() {
            self.tcx
                .sess
                .emit_err(StructExprNonExhaustive { span: expr.span, what: adt.variant_descr() });
        }

        let error_happened = self.check_expr_struct_fields(
            adt_ty,
            expected,
            expr.hir_id,
            qpath.span(),
            variant,
            fields,
            base_expr.is_none(),
            expr.span,
        );
        if let Some(base_expr) = base_expr {
            // If check_expr_struct_fields hit an error, do not attempt to populate
            // the fields with the base_expr. This could cause us to hit errors later
            // when certain fields are assumed to exist that in fact do not.
            if !error_happened {
                self.check_expr_has_type_or_error(base_expr, adt_ty, |_| {});
                match adt_ty.kind() {
                    ty::Adt(adt, substs) if adt.is_struct() => {
                        let fru_field_types = adt
                            .non_enum_variant()
                            .fields
                            .iter()
                            .map(|f| {
                                self.normalize_associated_types_in(
                                    expr.span,
                                    f.ty(self.tcx, substs),
                                )
                            })
                            .collect();

                        self.typeck_results
                            .borrow_mut()
                            .fru_field_types_mut()
                            .insert(expr.hir_id, fru_field_types);
                    }
                    _ => {
                        self.tcx
                            .sess
                            .emit_err(FunctionalRecordUpdateOnNonStruct { span: base_expr.span });
                    }
                }
            }
        }
        self.require_type_is_sized(adt_ty, expr.span, traits::StructInitializerSized);
        adt_ty
    }

    fn check_expr_struct_fields(
        &self,
        adt_ty: Ty<'tcx>,
        expected: Expectation<'tcx>,
        expr_id: hir::HirId,
        span: Span,
        variant: &'tcx ty::VariantDef,
        ast_fields: &'tcx [hir::ExprField<'tcx>],
        check_completeness: bool,
        expr_span: Span,
    ) -> bool {
        let tcx = self.tcx;

        let adt_ty_hint = self
            .expected_inputs_for_expected_output(span, expected, adt_ty, &[adt_ty])
            .get(0)
            .cloned()
            .unwrap_or(adt_ty);
        // re-link the regions that EIfEO can erase.
        self.demand_eqtype(span, adt_ty_hint, adt_ty);

        let (substs, adt_kind, kind_name) = match adt_ty.kind() {
            ty::Adt(adt, substs) => (substs, adt.adt_kind(), adt.variant_descr()),
            _ => span_bug!(span, "non-ADT passed to check_expr_struct_fields"),
        };

        let mut remaining_fields = variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| (field.ident.normalize_to_macros_2_0(), (i, field)))
            .collect::<FxHashMap<_, _>>();

        let mut seen_fields = FxHashMap::default();

        let mut error_happened = false;

        // Type-check each field.
        for field in ast_fields {
            let ident = tcx.adjust_ident(field.ident, variant.def_id);
            let field_type = if let Some((i, v_field)) = remaining_fields.remove(&ident) {
                seen_fields.insert(ident, field.span);
                self.write_field_index(field.hir_id, i);

                // We don't look at stability attributes on
                // struct-like enums (yet...), but it's definitely not
                // a bug to have constructed one.
                if adt_kind != AdtKind::Enum {
                    tcx.check_stability(v_field.did, Some(expr_id), field.span, None);
                }

                self.field_ty(field.span, v_field, substs)
            } else {
                error_happened = true;
                if let Some(prev_span) = seen_fields.get(&ident) {
                    tcx.sess.emit_err(FieldMultiplySpecifiedInInitializer {
                        span: field.ident.span,
                        prev_span: *prev_span,
                        ident,
                    });
                } else {
                    self.report_unknown_field(
                        adt_ty, variant, field, ast_fields, kind_name, expr_span,
                    );
                }

                tcx.ty_error()
            };

            // Make sure to give a type to the field even if there's
            // an error, so we can continue type-checking.
            self.check_expr_coercable_to_type(&field.expr, field_type, None);
        }

        // Make sure the programmer specified correct number of fields.
        if kind_name == "union" {
            if ast_fields.len() != 1 {
                struct_span_err!(
                    tcx.sess,
                    span,
                    E0784,
                    "union expressions should have exactly one field",
                )
                .emit();
            }
        } else if check_completeness && !error_happened && !remaining_fields.is_empty() {
            let inaccessible_remaining_fields = remaining_fields.iter().any(|(_, (_, field))| {
                !field.vis.is_accessible_from(tcx.parent_module(expr_id).to_def_id(), tcx)
            });

            if inaccessible_remaining_fields {
                self.report_inaccessible_fields(adt_ty, span);
            } else {
                self.report_missing_fields(adt_ty, span, remaining_fields);
            }
        }

        error_happened
    }

    fn check_struct_fields_on_error(
        &self,
        fields: &'tcx [hir::ExprField<'tcx>],
        base_expr: &'tcx Option<&'tcx hir::Expr<'tcx>>,
    ) {
        for field in fields {
            self.check_expr(&field.expr);
        }
        if let Some(base) = *base_expr {
            self.check_expr(&base);
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
    /// error: aborting due to previous error
    /// ```
    fn report_missing_fields(
        &self,
        adt_ty: Ty<'tcx>,
        span: Span,
        remaining_fields: FxHashMap<Ident, (usize, &ty::FieldDef)>,
    ) {
        let len = remaining_fields.len();

        let mut displayable_field_names =
            remaining_fields.keys().map(|ident| ident.as_str()).collect::<Vec<_>>();

        displayable_field_names.sort();

        let mut truncated_fields_error = String::new();
        let remaining_fields_names = match &displayable_field_names[..] {
            [field1] => format!("`{}`", field1),
            [field1, field2] => format!("`{}` and `{}`", field1, field2),
            [field1, field2, field3] => format!("`{}`, `{}` and `{}`", field1, field2, field3),
            _ => {
                truncated_fields_error =
                    format!(" and {} other field{}", len - 3, pluralize!(len - 3));
                displayable_field_names
                    .iter()
                    .take(3)
                    .map(|n| format!("`{}`", n))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };

        struct_span_err!(
            self.tcx.sess,
            span,
            E0063,
            "missing field{} {}{} in initializer of `{}`",
            pluralize!(len),
            remaining_fields_names,
            truncated_fields_error,
            adt_ty
        )
        .span_label(span, format!("missing {}{}", remaining_fields_names, truncated_fields_error))
        .emit();
    }

    /// Report an error for a struct field expression when there are invisible fields.
    ///
    /// ```text
    /// error: cannot construct `Foo` with struct literal syntax due to inaccessible fields
    ///  --> src/main.rs:8:5
    ///   |
    /// 8 |     foo::Foo {};
    ///   |     ^^^^^^^^
    ///
    /// error: aborting due to previous error
    /// ```
    fn report_inaccessible_fields(&self, adt_ty: Ty<'tcx>, span: Span) {
        self.tcx.sess.span_err(
            span,
            &format!(
                "cannot construct `{}` with struct literal syntax due to inaccessible fields",
                adt_ty,
            ),
        );
    }

    fn report_unknown_field(
        &self,
        ty: Ty<'tcx>,
        variant: &'tcx ty::VariantDef,
        field: &hir::ExprField<'_>,
        skip_fields: &[hir::ExprField<'_>],
        kind_name: &str,
        expr_span: Span,
    ) {
        if variant.is_recovered() {
            self.set_tainted_by_errors();
            return;
        }
        let mut err = self.type_error_struct_with_diag(
            field.ident.span,
            |actual| match ty.kind() {
                ty::Adt(adt, ..) if adt.is_enum() => struct_span_err!(
                    self.tcx.sess,
                    field.ident.span,
                    E0559,
                    "{} `{}::{}` has no field named `{}`",
                    kind_name,
                    actual,
                    variant.ident,
                    field.ident
                ),
                _ => struct_span_err!(
                    self.tcx.sess,
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
        match variant.ctor_kind {
            CtorKind::Fn => match ty.kind() {
                ty::Adt(adt, ..) if adt.is_enum() => {
                    err.span_label(
                        variant.ident.span,
                        format!(
                            "`{adt}::{variant}` defined here",
                            adt = ty,
                            variant = variant.ident,
                        ),
                    );
                    err.span_label(field.ident.span, "field does not exist");
                    err.span_suggestion_verbose(
                        expr_span,
                        &format!(
                            "`{adt}::{variant}` is a tuple {kind_name}, use the appropriate syntax",
                            adt = ty,
                            variant = variant.ident,
                        ),
                        format!(
                            "{adt}::{variant}(/* fields */)",
                            adt = ty,
                            variant = variant.ident,
                        ),
                        Applicability::HasPlaceholders,
                    );
                }
                _ => {
                    err.span_label(variant.ident.span, format!("`{adt}` defined here", adt = ty));
                    err.span_label(field.ident.span, "field does not exist");
                    err.span_suggestion_verbose(
                        expr_span,
                        &format!(
                            "`{adt}` is a tuple {kind_name}, use the appropriate syntax",
                            adt = ty,
                            kind_name = kind_name,
                        ),
                        format!("{adt}(/* fields */)", adt = ty),
                        Applicability::HasPlaceholders,
                    );
                }
            },
            _ => {
                // prevent all specified fields from being suggested
                let skip_fields = skip_fields.iter().map(|x| x.ident.name);
                if let Some(field_name) =
                    Self::suggest_field_name(variant, field.ident.name, skip_fields.collect())
                {
                    err.span_suggestion(
                        field.ident.span,
                        "a field with a similar name exists",
                        field_name.to_string(),
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    match ty.kind() {
                        ty::Adt(adt, ..) => {
                            if adt.is_enum() {
                                err.span_label(
                                    field.ident.span,
                                    format!("`{}::{}` does not have this field", ty, variant.ident),
                                );
                            } else {
                                err.span_label(
                                    field.ident.span,
                                    format!("`{}` does not have this field", ty),
                                );
                            }
                            let available_field_names = self.available_field_names(variant);
                            if !available_field_names.is_empty() {
                                err.note(&format!(
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
        err.emit();
    }

    // Return an hint about the closest match in field names
    fn suggest_field_name(
        variant: &'tcx ty::VariantDef,
        field: Symbol,
        skip: Vec<Symbol>,
    ) -> Option<Symbol> {
        let names = variant
            .fields
            .iter()
            .filter_map(|field| {
                // ignore already set fields and private fields from non-local crates
                if skip.iter().any(|&x| x == field.ident.name)
                    || (!variant.def_id.is_local() && field.vis != Visibility::Public)
                {
                    None
                } else {
                    Some(field.ident.name)
                }
            })
            .collect::<Vec<Symbol>>();

        find_best_match_for_name(&names, field, None)
    }

    fn available_field_names(&self, variant: &'tcx ty::VariantDef) -> Vec<Symbol> {
        variant
            .fields
            .iter()
            .filter(|field| {
                let def_scope = self
                    .tcx
                    .adjust_ident_and_get_scope(field.ident, variant.def_id, self.body_id)
                    .1;
                field.vis.is_accessible_from(def_scope, self.tcx)
            })
            .map(|field| field.ident.name)
            .collect()
    }

    fn name_series_display(&self, names: Vec<Symbol>) -> String {
        // dynamic limit, to never omit just one field
        let limit = if names.len() == 6 { 6 } else { 5 };
        let mut display =
            names.iter().take(limit).map(|n| format!("`{}`", n)).collect::<Vec<_>>().join(", ");
        if names.len() > limit {
            display = format!("{} ... and {} others", display, names.len() - limit);
        }
        display
    }

    // Check field access expressions
    fn check_field(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        base: &'tcx hir::Expr<'tcx>,
        field: Ident,
    ) -> Ty<'tcx> {
        debug!("check_field(expr: {:?}, base: {:?}, field: {:?})", expr, base, field);
        let expr_t = self.check_expr(base);
        let expr_t = self.structurally_resolved_type(base.span, expr_t);
        let mut private_candidate = None;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, _)) = autoderef.next() {
            debug!("base_t: {:?}", base_t);
            match base_t.kind() {
                ty::Adt(base_def, substs) if !base_def.is_enum() => {
                    debug!("struct named {:?}", base_t);
                    let (ident, def_scope) =
                        self.tcx.adjust_ident_and_get_scope(field, base_def.did, self.body_id);
                    let fields = &base_def.non_enum_variant().fields;
                    if let Some(index) =
                        fields.iter().position(|f| f.ident.normalize_to_macros_2_0() == ident)
                    {
                        let field = &fields[index];
                        let field_ty = self.field_ty(expr.span, field, substs);
                        // Save the index of all fields regardless of their visibility in case
                        // of error recovery.
                        self.write_field_index(expr.hir_id, index);
                        if field.vis.is_accessible_from(def_scope, self.tcx) {
                            let adjustments = self.adjust_steps(&autoderef);
                            self.apply_adjustments(base, adjustments);
                            self.register_predicates(autoderef.into_obligations());

                            self.tcx.check_stability(field.did, Some(expr.hir_id), expr.span, None);
                            return field_ty;
                        }
                        private_candidate = Some((base_def.did, field_ty));
                    }
                }
                ty::Tuple(tys) => {
                    let fstr = field.as_str();
                    if let Ok(index) = fstr.parse::<usize>() {
                        if fstr == index.to_string() {
                            if let Some(field_ty) = tys.get(index) {
                                let adjustments = self.adjust_steps(&autoderef);
                                self.apply_adjustments(base, adjustments);
                                self.register_predicates(autoderef.into_obligations());

                                self.write_field_index(expr.hir_id, index);
                                return field_ty.expect_ty();
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        self.structurally_resolved_type(autoderef.span(), autoderef.final_ty(false));

        if let Some((did, field_ty)) = private_candidate {
            self.ban_private_field_access(expr, expr_t, field, did);
            return field_ty;
        }

        if field.name == kw::Empty {
        } else if self.method_exists(field, expr_t, expr.hir_id, true) {
            self.ban_take_value_of_method(expr, expr_t, field);
        } else if !expr_t.is_primitive_ty() {
            self.ban_nonexisting_field(field, base, expr, expr_t);
        } else {
            type_error_struct!(
                self.tcx().sess,
                field.span,
                expr_t,
                E0610,
                "`{}` is a primitive type and therefore doesn't have fields",
                expr_t
            )
            .emit();
        }

        self.tcx().ty_error()
    }

    fn suggest_await_on_field_access(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        field_ident: Ident,
        base: &'tcx hir::Expr<'tcx>,
        ty: Ty<'tcx>,
    ) {
        let output_ty = match self.infcx.get_impl_future_output_ty(ty) {
            Some(output_ty) => self.resolve_vars_if_possible(output_ty),
            _ => return,
        };
        let mut add_label = true;
        if let ty::Adt(def, _) = output_ty.kind() {
            // no field access on enum type
            if !def.is_enum() {
                if def.non_enum_variant().fields.iter().any(|field| field.ident == field_ident) {
                    add_label = false;
                    err.span_label(
                        field_ident.span,
                        "field not available in `impl Future`, but it is available in its `Output`",
                    );
                    err.span_suggestion_verbose(
                        base.span.shrink_to_hi(),
                        "consider `await`ing on the `Future` and access the field of its `Output`",
                        ".await".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
        if add_label {
            err.span_label(field_ident.span, &format!("field not found in `{}`", ty));
        }
    }

    fn ban_nonexisting_field(
        &self,
        field: Ident,
        base: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        expr_t: Ty<'tcx>,
    ) {
        debug!(
            "ban_nonexisting_field: field={:?}, base={:?}, expr={:?}, expr_ty={:?}",
            field, base, expr, expr_t
        );
        let mut err = self.no_such_field_err(field, expr_t);

        match *expr_t.peel_refs().kind() {
            ty::Array(_, len) => {
                self.maybe_suggest_array_indexing(&mut err, expr, base, field, len);
            }
            ty::RawPtr(..) => {
                self.suggest_first_deref_field(&mut err, expr, base, field);
            }
            ty::Adt(def, _) if !def.is_enum() => {
                self.suggest_fields_on_recordish(&mut err, def, field);
            }
            ty::Param(param_ty) => {
                self.point_at_param_definition(&mut err, param_ty);
            }
            ty::Opaque(_, _) => {
                self.suggest_await_on_field_access(&mut err, field, base, expr_t.peel_refs());
            }
            _ => {}
        }

        if field.name == kw::Await {
            // We know by construction that `<expr>.await` is either on Rust 2015
            // or results in `ExprKind::Await`. Suggest switching the edition to 2018.
            err.note("to `.await` a `Future`, switch to Rust 2018 or later");
            err.help(&format!("set `edition = \"{}\"` in `Cargo.toml`", LATEST_STABLE_EDITION));
            err.note("for more on editions, read https://doc.rust-lang.org/edition-guide");
        }

        err.emit();
    }

    fn ban_private_field_access(
        &self,
        expr: &hir::Expr<'_>,
        expr_t: Ty<'tcx>,
        field: Ident,
        base_did: DefId,
    ) {
        let struct_path = self.tcx().def_path_str(base_did);
        let kind_name = self.tcx().def_kind(base_did).descr(base_did);
        let mut err = struct_span_err!(
            self.tcx().sess,
            field.span,
            E0616,
            "field `{}` of {} `{}` is private",
            field,
            kind_name,
            struct_path
        );
        err.span_label(field.span, "private field");
        // Also check if an accessible method exists, which is often what is meant.
        if self.method_exists(field, expr_t, expr.hir_id, false) && !self.expr_in_place(expr.hir_id)
        {
            self.suggest_method_call(
                &mut err,
                &format!("a method `{}` also exists, call it with parentheses", field),
                field,
                expr_t,
                expr,
            );
        }
        err.emit();
    }

    fn ban_take_value_of_method(&self, expr: &hir::Expr<'_>, expr_t: Ty<'tcx>, field: Ident) {
        let mut err = type_error_struct!(
            self.tcx().sess,
            field.span,
            expr_t,
            E0615,
            "attempted to take value of method `{}` on type `{}`",
            field,
            expr_t
        );
        err.span_label(field.span, "method, not a field");
        if !self.expr_in_place(expr.hir_id) {
            self.suggest_method_call(
                &mut err,
                "use parentheses to call the method",
                field,
                expr_t,
                expr,
            );
        } else {
            err.help("methods are immutable and cannot be assigned to");
        }

        err.emit();
    }

    fn point_at_param_definition(&self, err: &mut DiagnosticBuilder<'_>, param: ty::ParamTy) {
        let generics = self.tcx.generics_of(self.body_id.owner.to_def_id());
        let generic_param = generics.type_param(&param, self.tcx);
        if let ty::GenericParamDefKind::Type { synthetic: Some(..), .. } = generic_param.kind {
            return;
        }
        let param_def_id = generic_param.def_id;
        let param_hir_id = match param_def_id.as_local() {
            Some(x) => self.tcx.hir().local_def_id_to_hir_id(x),
            None => return,
        };
        let param_span = self.tcx.hir().span(param_hir_id);
        let param_name = self.tcx.hir().ty_param_name(param_hir_id);

        err.span_label(param_span, &format!("type parameter '{}' declared here", param_name));
    }

    fn suggest_fields_on_recordish(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        def: &'tcx ty::AdtDef,
        field: Ident,
    ) {
        if let Some(suggested_field_name) =
            Self::suggest_field_name(def.non_enum_variant(), field.name, vec![])
        {
            err.span_suggestion(
                field.span,
                "a field with a similar name exists",
                suggested_field_name.to_string(),
                Applicability::MaybeIncorrect,
            );
        } else {
            err.span_label(field.span, "unknown field");
            let struct_variant_def = def.non_enum_variant();
            let field_names = self.available_field_names(struct_variant_def);
            if !field_names.is_empty() {
                err.note(&format!(
                    "available fields are: {}",
                    self.name_series_display(field_names),
                ));
            }
        }
    }

    fn maybe_suggest_array_indexing(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        base: &hir::Expr<'_>,
        field: Ident,
        len: &ty::Const<'tcx>,
    ) {
        if let (Some(len), Ok(user_index)) =
            (len.try_eval_usize(self.tcx, self.param_env), field.as_str().parse::<u64>())
        {
            if let Ok(base) = self.tcx.sess.source_map().span_to_snippet(base.span) {
                let help = "instead of using tuple indexing, use array indexing";
                let suggestion = format!("{}[{}]", base, field);
                let applicability = if len < user_index {
                    Applicability::MachineApplicable
                } else {
                    Applicability::MaybeIncorrect
                };
                err.span_suggestion(expr.span, help, suggestion, applicability);
            }
        }
    }

    fn suggest_first_deref_field(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        base: &hir::Expr<'_>,
        field: Ident,
    ) {
        if let Ok(base) = self.tcx.sess.source_map().span_to_snippet(base.span) {
            let msg = format!("`{}` is a raw pointer; try dereferencing it", base);
            let suggestion = format!("(*{}).{}", base, field);
            err.span_suggestion(expr.span, &msg, suggestion, Applicability::MaybeIncorrect);
        }
    }

    fn no_such_field_err(
        &self,
        field: Ident,
        expr_t: &'tcx ty::TyS<'tcx>,
    ) -> DiagnosticBuilder<'_> {
        let span = field.span;
        debug!("no_such_field_err(span: {:?}, field: {:?}, expr_t: {:?})", span, field, expr_t);

        let mut err = type_error_struct!(
            self.tcx().sess,
            field.span,
            expr_t,
            E0609,
            "no field `{}` on type `{}`",
            field,
            expr_t
        );

        // try to add a suggestion in case the field is a nested field of a field of the Adt
        if let Some((fields, substs)) = self.get_field_candidates(span, &expr_t) {
            for candidate_field in fields.iter() {
                if let Some(field_path) =
                    self.check_for_nested_field(span, field, candidate_field, substs, vec![])
                {
                    let field_path_str = field_path
                        .iter()
                        .map(|id| id.name.to_ident_string())
                        .collect::<Vec<String>>()
                        .join(".");
                    debug!("field_path_str: {:?}", field_path_str);

                    err.span_suggestion_verbose(
                        field.span.shrink_to_lo(),
                        "one of the expressions' fields has a field of the same name",
                        format!("{}.", field_path_str),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
        err
    }

    fn get_field_candidates(
        &self,
        span: Span,
        base_t: Ty<'tcx>,
    ) -> Option<(&Vec<ty::FieldDef>, SubstsRef<'tcx>)> {
        debug!("get_field_candidates(span: {:?}, base_t: {:?}", span, base_t);

        let mut autoderef = self.autoderef(span, base_t);
        while let Some((base_t, _)) = autoderef.next() {
            match base_t.kind() {
                ty::Adt(base_def, substs) if !base_def.is_enum() => {
                    let fields = &base_def.non_enum_variant().fields;
                    // For compile-time reasons put a limit on number of fields we search
                    if fields.len() > 100 {
                        return None;
                    }
                    return Some((fields, substs));
                }
                _ => {}
            }
        }
        None
    }

    /// This method is called after we have encountered a missing field error to recursively
    /// search for the field
    fn check_for_nested_field(
        &self,
        span: Span,
        target_field: Ident,
        candidate_field: &ty::FieldDef,
        subst: SubstsRef<'tcx>,
        mut field_path: Vec<Ident>,
    ) -> Option<Vec<Ident>> {
        debug!(
            "check_for_nested_field(span: {:?}, candidate_field: {:?}, field_path: {:?}",
            span, candidate_field, field_path
        );

        if candidate_field.ident == target_field {
            Some(field_path)
        } else if field_path.len() > 3 {
            // For compile-time reasons and to avoid infinite recursion we only check for fields
            // up to a depth of three
            None
        } else {
            // recursively search fields of `candidate_field` if it's a ty::Adt

            field_path.push(candidate_field.ident.normalize_to_macros_2_0());
            let field_ty = candidate_field.ty(self.tcx, subst);
            if let Some((nested_fields, subst)) = self.get_field_candidates(span, &field_ty) {
                for field in nested_fields.iter() {
                    let ident = field.ident.normalize_to_macros_2_0();
                    if ident == target_field {
                        return Some(field_path);
                    } else {
                        let field_path = field_path.clone();
                        if let Some(path) = self.check_for_nested_field(
                            span,
                            target_field,
                            field,
                            subst,
                            field_path,
                        ) {
                            return Some(path);
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
    ) -> Ty<'tcx> {
        let base_t = self.check_expr(&base);
        let idx_t = self.check_expr(&idx);

        if base_t.references_error() {
            base_t
        } else if idx_t.references_error() {
            idx_t
        } else {
            let base_t = self.structurally_resolved_type(base.span, base_t);
            match self.lookup_indexing(expr, base, base_t, idx_t) {
                Some((index_ty, element_ty)) => {
                    // two-phase not needed because index_ty is never mutable
                    self.demand_coerce(idx, idx_t, index_ty, None, AllowTwoPhase::No);
                    element_ty
                }
                None => {
                    let mut err = type_error_struct!(
                        self.tcx.sess,
                        expr.span,
                        base_t,
                        E0608,
                        "cannot index into a value of type `{}`",
                        base_t
                    );
                    // Try to give some advice about indexing tuples.
                    if let ty::Tuple(..) = base_t.kind() {
                        let mut needs_note = true;
                        // If the index is an integer, we can show the actual
                        // fixed expression:
                        if let ExprKind::Lit(ref lit) = idx.kind {
                            if let ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) = lit.node {
                                let snip = self.tcx.sess.source_map().span_to_snippet(base.span);
                                if let Ok(snip) = snip {
                                    err.span_suggestion(
                                        expr.span,
                                        "to access tuple elements, use",
                                        format!("{}.{}", snip, i),
                                        Applicability::MachineApplicable,
                                    );
                                    needs_note = false;
                                }
                            }
                        }
                        if needs_note {
                            err.help(
                                "to access tuple elements, use tuple indexing \
                                        syntax (e.g., `tuple.0`)",
                            );
                        }
                    }
                    err.emit();
                    self.tcx.ty_error()
                }
            }
        }
    }

    fn check_expr_yield(
        &self,
        value: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        src: &'tcx hir::YieldSource,
    ) -> Ty<'tcx> {
        match self.resume_yield_tys {
            Some((resume_ty, yield_ty)) => {
                self.check_expr_coercable_to_type(&value, yield_ty, None);

                resume_ty
            }
            // Given that this `yield` expression was generated as a result of lowering a `.await`,
            // we know that the yield type must be `()`; however, the context won't contain this
            // information. Hence, we check the source of the yield expression here and check its
            // value's type against `()` (this check should always hold).
            None if src.is_await() => {
                self.check_expr_coercable_to_type(&value, self.tcx.mk_unit(), None);
                self.tcx.mk_unit()
            }
            _ => {
                self.tcx.sess.emit_err(YieldExprOutsideOfGenerator { span: expr.span });
                // Avoid expressions without types during writeback (#78653).
                self.check_expr(value);
                self.tcx.mk_unit()
            }
        }
    }

    fn check_expr_asm_operand(&self, expr: &'tcx hir::Expr<'tcx>, is_input: bool) {
        let needs = if is_input { Needs::None } else { Needs::MutPlace };
        let ty = self.check_expr_with_needs(expr, needs);
        self.require_type_is_sized(ty, expr.span, traits::InlineAsmSized);

        if !is_input && !expr.is_syntactic_place_expr() {
            let mut err = self.tcx.sess.struct_span_err(expr.span, "invalid asm output");
            err.span_label(expr.span, "cannot assign to this expression");
            err.emit();
        }

        // If this is an input value, we require its type to be fully resolved
        // at this point. This allows us to provide helpful coercions which help
        // pass the type candidate list in a later pass.
        //
        // We don't require output types to be resolved at this point, which
        // allows them to be inferred based on how they are used later in the
        // function.
        if is_input {
            let ty = self.structurally_resolved_type(expr.span, &ty);
            match *ty.kind() {
                ty::FnDef(..) => {
                    let fnptr_ty = self.tcx.mk_fn_ptr(ty.fn_sig(self.tcx));
                    self.demand_coerce(expr, ty, fnptr_ty, None, AllowTwoPhase::No);
                }
                ty::Ref(_, base_ty, mutbl) => {
                    let ptr_ty = self.tcx.mk_ptr(ty::TypeAndMut { ty: base_ty, mutbl });
                    self.demand_coerce(expr, ty, ptr_ty, None, AllowTwoPhase::No);
                }
                _ => {}
            }
        }
    }

    fn check_expr_asm(&self, asm: &'tcx hir::InlineAsm<'tcx>) -> Ty<'tcx> {
        for (op, _op_sp) in asm.operands {
            match op {
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
                hir::InlineAsmOperand::Const { anon_const } => {
                    self.to_const(anon_const);
                }
                hir::InlineAsmOperand::Sym { expr } => {
                    self.check_expr(expr);
                }
            }
        }
        if asm.options.contains(ast::InlineAsmOptions::NORETURN) {
            self.tcx.types.never
        } else {
            self.tcx.mk_unit()
        }
    }
}

pub(super) fn ty_kind_suggestion(ty: Ty<'_>) -> Option<&'static str> {
    Some(match ty.kind() {
        ty::Bool => "true",
        ty::Char => "'a'",
        ty::Int(_) | ty::Uint(_) => "42",
        ty::Float(_) => "3.14159",
        ty::Error(_) | ty::Never => return None,
        _ => "value",
    })
}
