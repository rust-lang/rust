//! Type checking expressions.
//!
//! See `mod.rs` for more context on type checking in general.

use crate::check::BreakableCtxt;
use crate::check::cast;
use crate::check::coercion::CoerceMany;
use crate::check::Diverges;
use crate::check::FnCtxt;
use crate::check::Expectation::{self, NoExpectation, ExpectHasType, ExpectCastableToType};
use crate::check::fatally_break_rust;
use crate::check::report_unexpected_variant_res;
use crate::check::Needs;
use crate::check::TupleArgumentsFlag::DontTupleArguments;
use crate::check::method::SelfSource;
use crate::middle::lang_items;
use crate::util::common::ErrorReported;
use crate::util::nodemap::FxHashMap;
use crate::astconv::AstConv as _;

use errors::{Applicability, DiagnosticBuilder};
use syntax::ast;
use syntax::symbol::{Symbol, LocalInternedString, kw, sym};
use syntax::source_map::Span;
use syntax::util::lev_distance::find_best_match_for_name;
use rustc::hir;
use rustc::hir::{ExprKind, QPath};
use rustc::hir::def::{CtorKind, Res, DefKind};
use rustc::hir::ptr::P;
use rustc::infer;
use rustc::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc::mir::interpret::GlobalId;
use rustc::ty;
use rustc::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability,
};
use rustc::ty::{AdtKind, Visibility};
use rustc::ty::Ty;
use rustc::ty::TypeFoldable;
use rustc::ty::subst::InternalSubsts;
use rustc::traits::{self, ObligationCauseCode};

use std::fmt::Display;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn check_expr_eq_type(&self, expr: &'tcx hir::Expr, expected: Ty<'tcx>) {
        let ty = self.check_expr_with_hint(expr, expected);
        self.demand_eqtype(expr.span, expected, ty);
    }

    pub fn check_expr_has_type_or_error(
        &self,
        expr: &'tcx hir::Expr,
        expected: Ty<'tcx>,
    ) -> Ty<'tcx> {
        self.check_expr_meets_expectation_or_error(expr, ExpectHasType(expected))
    }

    fn check_expr_meets_expectation_or_error(
        &self,
        expr: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let expected_ty = expected.to_option(&self).unwrap_or(self.tcx.types.bool);
        let mut ty = self.check_expr_with_expectation(expr, expected);

        // While we don't allow *arbitrary* coercions here, we *do* allow
        // coercions from ! to `expected`.
        if ty.is_never() {
            assert!(!self.tables.borrow().adjustments().contains_key(expr.hir_id),
                    "expression with never type wound up being adjusted");
            let adj_ty = self.next_diverging_ty_var(
                TypeVariableOrigin {
                    kind: TypeVariableOriginKind::AdjustmentType,
                    span: expr.span,
                },
            );
            self.apply_adjustments(expr, vec![Adjustment {
                kind: Adjust::NeverToAny,
                target: adj_ty
            }]);
            ty = adj_ty;
        }

        if let Some(mut err) = self.demand_suptype_diag(expr.span, expected_ty, ty) {
            let expr = match &expr.node {
                ExprKind::DropTemps(expr) => expr,
                _ => expr,
            };
            // Error possibly reported in `check_assign` so avoid emitting error again.
            err.emit_unless(self.is_assign_to_bool(expr, expected_ty));
        }
        ty
    }

    pub(super) fn check_expr_coercable_to_type(
        &self,
        expr: &'tcx hir::Expr,
        expected: Ty<'tcx>
    ) -> Ty<'tcx> {
        let ty = self.check_expr_with_hint(expr, expected);
        // checks don't need two phase
        self.demand_coerce(expr, ty, expected, AllowTwoPhase::No)
    }

    pub(super) fn check_expr_with_hint(
        &self,
        expr: &'tcx hir::Expr,
        expected: Ty<'tcx>
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, ExpectHasType(expected))
    }

    pub(super) fn check_expr_with_expectation(
        &self,
        expr: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_needs(expr, expected, Needs::None)
    }

    pub(super) fn check_expr(&self, expr: &'tcx hir::Expr) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, NoExpectation)
    }

    pub(super) fn check_expr_with_needs(&self, expr: &'tcx hir::Expr, needs: Needs) -> Ty<'tcx> {
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
    fn check_expr_with_expectation_and_needs(
        &self,
        expr: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
        needs: Needs,
    ) -> Ty<'tcx> {
        debug!(">> type-checking: expr={:?} expected={:?}",
               expr, expected);

        // Warn for expressions after diverging siblings.
        self.warn_if_unreachable(expr.hir_id, expr.span, "expression");

        // Hide the outer diverging and has_errors flags.
        let old_diverges = self.diverges.get();
        let old_has_errors = self.has_errors.get();
        self.diverges.set(Diverges::Maybe);
        self.has_errors.set(false);

        let ty = self.check_expr_kind(expr, expected, needs);

        // Warn for non-block expressions with diverging children.
        match expr.node {
            ExprKind::Block(..) |
            ExprKind::Loop(..) | ExprKind::While(..) |
            ExprKind::Match(..) => {}

            _ => self.warn_if_unreachable(expr.hir_id, expr.span, "expression")
        }

        // Any expression that produces a value of type `!` must have diverged
        if ty.is_never() {
            self.diverges.set(self.diverges.get() | Diverges::Always);
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
        expr: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
        needs: Needs,
    ) -> Ty<'tcx> {
        debug!(
            "check_expr_kind(expr={:?}, expected={:?}, needs={:?})",
            expr,
            expected,
            needs,
        );

        let tcx = self.tcx;
        match expr.node {
            ExprKind::Box(ref subexpr) => {
                self.check_expr_box(subexpr, expected)
            }
            ExprKind::Lit(ref lit) => {
                self.check_lit(&lit, expected)
            }
            ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.check_binop(expr, op, lhs, rhs)
            }
            ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                self.check_binop_assign(expr, op, lhs, rhs)
            }
            ExprKind::Unary(unop, ref oprnd) => {
                self.check_expr_unary(unop, oprnd, expected, needs, expr)
            }
            ExprKind::AddrOf(mutbl, ref oprnd) => {
                self.check_expr_addr_of(mutbl, oprnd, expected, expr)
            }
            ExprKind::Path(ref qpath) => {
                self.check_expr_path(qpath, expr)
            }
            ExprKind::InlineAsm(_, ref outputs, ref inputs) => {
                for expr in outputs.iter().chain(inputs.iter()) {
                    self.check_expr(expr);
                }
                tcx.mk_unit()
            }
            ExprKind::Break(destination, ref expr_opt) => {
                self.check_expr_break(destination, expr_opt.deref(), expr)
            }
            ExprKind::Continue(destination) => {
                if destination.target_id.is_ok() {
                    tcx.types.never
                } else {
                    // There was an error; make type-check fail.
                    tcx.types.err
                }
            }
            ExprKind::Ret(ref expr_opt) => {
                self.check_expr_return(expr_opt.deref(), expr)
            }
            ExprKind::Assign(ref lhs, ref rhs) => {
                self.check_expr_assign(expr, expected, lhs, rhs)
            }
            ExprKind::While(ref cond, ref body, _) => {
                self.check_expr_while(cond, body, expr)
            }
            ExprKind::Loop(ref body, _, source) => {
                self.check_expr_loop(body, source, expected, expr)
            }
            ExprKind::Match(ref discrim, ref arms, match_src) => {
                self.check_match(expr, &discrim, arms, expected, match_src)
            }
            ExprKind::Closure(capture, ref decl, body_id, _, gen) => {
                self.check_expr_closure(expr, capture, &decl, body_id, gen, expected)
            }
            ExprKind::Block(ref body, _) => {
                self.check_block_with_expected(&body, expected)
            }
            ExprKind::Call(ref callee, ref args) => {
                self.check_call(expr, &callee, args, expected)
            }
            ExprKind::MethodCall(ref segment, span, ref args) => {
                self.check_method_call(expr, segment, span, args, expected, needs)
            }
            ExprKind::Cast(ref e, ref t) => {
                self.check_expr_cast(e, t, expr)
            }
            ExprKind::Type(ref e, ref t) => {
                let ty = self.to_ty_saving_user_provided_ty(&t);
                self.check_expr_eq_type(&e, ty);
                ty
            }
            ExprKind::DropTemps(ref e) => {
                self.check_expr_with_expectation(e, expected)
            }
            ExprKind::Array(ref args) => {
                self.check_expr_array(args, expected, expr)
            }
            ExprKind::Repeat(ref element, ref count) => {
                self.check_expr_repeat(element, count, expected, expr)
            }
            ExprKind::Tup(ref elts) => {
                self.check_expr_tuple(elts, expected, expr)
            }
            ExprKind::Struct(ref qpath, ref fields, ref base_expr) => {
                self.check_expr_struct(expr, expected, qpath, fields, base_expr)
            }
            ExprKind::Field(ref base, field) => {
                self.check_field(expr, needs, &base, field)
            }
            ExprKind::Index(ref base, ref idx) => {
                self.check_expr_index(base, idx, needs, expr)
            }
            ExprKind::Yield(ref value, ref src) => {
                self.check_expr_yield(value, expr, src)
            }
            hir::ExprKind::Err => {
                tcx.types.err
            }
        }
    }

    fn check_expr_box(&self, expr: &'tcx hir::Expr, expected: Expectation<'tcx>) -> Ty<'tcx> {
        let expected_inner = expected.to_option(self).map_or(NoExpectation, |ty| {
            match ty.sty {
                ty::Adt(def, _) if def.is_box()
                    => Expectation::rvalue_hint(self, ty.boxed_ty()),
                _ => NoExpectation
            }
        });
        let referent_ty = self.check_expr_with_expectation(expr, expected_inner);
        self.tcx.mk_box(referent_ty)
    }

    fn check_expr_unary(
        &self,
        unop: hir::UnOp,
        oprnd: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
        needs: Needs,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let expected_inner = match unop {
            hir::UnNot | hir::UnNeg => expected,
            hir::UnDeref => NoExpectation,
        };
        let needs = match unop {
            hir::UnDeref => needs,
            _ => Needs::None
        };
        let mut oprnd_t = self.check_expr_with_expectation_and_needs(&oprnd, expected_inner, needs);

        if !oprnd_t.references_error() {
            oprnd_t = self.structurally_resolved_type(expr.span, oprnd_t);
            match unop {
                hir::UnDeref => {
                    if let Some(mt) = oprnd_t.builtin_deref(true) {
                        oprnd_t = mt.ty;
                    } else if let Some(ok) = self.try_overloaded_deref(
                            expr.span, oprnd_t, needs) {
                        let method = self.register_infer_ok_obligations(ok);
                        if let ty::Ref(region, _, mutbl) = method.sig.inputs()[0].sty {
                            let mutbl = match mutbl {
                                hir::MutImmutable => AutoBorrowMutability::Immutable,
                                hir::MutMutable => AutoBorrowMutability::Mutable {
                                    // (It shouldn't actually matter for unary ops whether
                                    // we enable two-phase borrows or not, since a unary
                                    // op has no additional operands.)
                                    allow_two_phase_borrow: AllowTwoPhase::No,
                                }
                            };
                            self.apply_adjustments(oprnd, vec![Adjustment {
                                kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                                target: method.sig.inputs()[0]
                            }]);
                        }
                        oprnd_t = self.make_overloaded_place_return_type(method).ty;
                        self.write_method_call(expr.hir_id, method);
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
                        if let Some(sp) = tcx.sess.parse_sess.ambiguous_block_expr_parse
                            .borrow().get(&sp)
                        {
                            tcx.sess.parse_sess.expr_parentheses_needed(
                                &mut err,
                                *sp,
                                None,
                            );
                        }
                        err.emit();
                        oprnd_t = tcx.types.err;
                    }
                }
                hir::UnNot => {
                    let result = self.check_user_unop(expr, oprnd_t, unop);
                    // If it's builtin, we can reuse the type, this helps inference.
                    if !(oprnd_t.is_integral() || oprnd_t.sty == ty::Bool) {
                        oprnd_t = result;
                    }
                }
                hir::UnNeg => {
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
        mutbl: hir::Mutability,
        oprnd: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let hint = expected.only_has_type(self).map_or(NoExpectation, |ty| {
            match ty.sty {
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    if oprnd.is_place_expr() {
                        // Places may legitimately have unsized types.
                        // For example, dereferences of a fat pointer and
                        // the last field of a struct can be unsized.
                        ExpectHasType(ty)
                    } else {
                        Expectation::rvalue_hint(self, ty)
                    }
                }
                _ => NoExpectation
            }
        });
        let needs = Needs::maybe_mut_place(mutbl);
        let ty = self.check_expr_with_expectation_and_needs(&oprnd, hint, needs);

        let tm = ty::TypeAndMut { ty: ty, mutbl: mutbl };
        if tm.ty.references_error() {
            self.tcx.types.err
        } else {
            // Note: at this point, we cannot say what the best lifetime
            // is to use for resulting pointer.  We want to use the
            // shortest lifetime possible so as to avoid spurious borrowck
            // errors.  Moreover, the longest lifetime will depend on the
            // precise details of the value whose address is being taken
            // (and how long it is valid), which we don't know yet until type
            // inference is complete.
            //
            // Therefore, here we simply generate a region variable.  The
            // region inferencer will then select the ultimate value.
            // Finally, borrowck is charged with guaranteeing that the
            // value whose address was taken can actually be made to live
            // as long as it needs to live.
            let region = self.next_region_var(infer::AddrOfRegion(expr.span));
            self.tcx.mk_ref(region, tm)
        }
    }

    fn check_expr_path(&self, qpath: &hir::QPath, expr: &'tcx hir::Expr) -> Ty<'tcx> {
        let tcx = self.tcx;
        let (res, opt_ty, segs) = self.resolve_ty_and_res_ufcs(qpath, expr.hir_id, expr.span);
        let ty = match res {
            Res::Err => {
                self.set_tainted_by_errors();
                tcx.types.err
            }
            Res::Def(DefKind::Ctor(_, CtorKind::Fictive), _) => {
                report_unexpected_variant_res(tcx, res, expr.span, qpath);
                tcx.types.err
            }
            _ => self.instantiate_value_path(segs, opt_ty, res, expr.span, expr.hir_id).0,
        };

        if let ty::FnDef(..) = ty.sty {
            let fn_sig = ty.fn_sig(tcx);
            if !tcx.features().unsized_locals {
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
                    let input = self.replace_bound_vars_with_fresh_vars(
                        expr.span,
                        infer::LateBoundRegionConversionTime::FnCall,
                        &fn_sig.input(i)).0;
                    self.require_type_is_sized_deferred(input, expr.span,
                                                        traits::SizedArgumentType);
                }
            }
            // Here we want to prevent struct constructors from returning unsized types.
            // There were two cases this happened: fn pointer coercion in stable
            // and usual function call in presense of unsized_locals.
            // Also, as we just want to check sizedness, instead of introducing
            // placeholder lifetimes with probing, we just replace higher lifetimes
            // with fresh vars.
            let output = self.replace_bound_vars_with_fresh_vars(
                expr.span,
                infer::LateBoundRegionConversionTime::FnCall,
                &fn_sig.output()).0;
            self.require_type_is_sized_deferred(output, expr.span, traits::SizedReturnType);
        }

        // We always require that the type provided as the value for
        // a type parameter outlives the moment of instantiation.
        let substs = self.tables.borrow().node_substs(expr.hir_id);
        self.add_wf_bounds(substs, expr);

        ty
    }

    fn check_expr_break(
        &self,
        destination: hir::Destination,
        expr_opt: Option<&'tcx hir::Expr>,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        if let Ok(target_id) = destination.target_id {
            let (e_ty, cause);
            if let Some(ref e) = expr_opt {
                // If this is a break with a value, we need to type-check
                // the expression. Get an expected type from the loop context.
                let opt_coerce_to = {
                    let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
                    enclosing_breakables.find_breakable(target_id)
                                        .coerce
                                        .as_ref()
                                        .map(|coerce| coerce.expected_ty())
                };

                // If the loop context is not a `loop { }`, then break with
                // a value is illegal, and `opt_coerce_to` will be `None`.
                // Just set expectation to error in that case.
                let coerce_to = opt_coerce_to.unwrap_or(tcx.types.err);

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
            let ctxt = enclosing_breakables.find_breakable(target_id);
            if let Some(ref mut coerce) = ctxt.coerce {
                if let Some(ref e) = expr_opt {
                    coerce.coerce(self, &cause, e, e_ty);
                } else {
                    assert!(e_ty.is_unit());
                    coerce.coerce_forced_unit(self, &cause, &mut |_| (), true);
                }
            } else {
                // If `ctxt.coerce` is `None`, we can just ignore
                // the type of the expresison.  This is because
                // either this was a break *without* a value, in
                // which case it is always a legal type (`()`), or
                // else an error would have been flagged by the
                // `loops` pass for using break with an expression
                // where you are not supposed to.
                assert!(expr_opt.is_none() || self.tcx.sess.has_errors());
            }

            ctxt.may_break = true;

            // the type of a `break` is always `!`, since it diverges
            tcx.types.never
        } else {
            // Otherwise, we failed to find the enclosing loop;
            // this can only happen if the `break` was not
            // inside a loop at all, which is caught by the
            // loop-checking pass.
            self.tcx.sess.delay_span_bug(expr.span,
                "break was outside loop, but no error was emitted");

            // We still need to assign a type to the inner expression to
            // prevent the ICE in #43162.
            if let Some(ref e) = expr_opt {
                self.check_expr_with_hint(e, tcx.types.err);

                // ... except when we try to 'break rust;'.
                // ICE this expression in particular (see #43162).
                if let ExprKind::Path(QPath::Resolved(_, ref path)) = e.node {
                    if path.segments.len() == 1 &&
                        path.segments[0].ident.name == sym::rust {
                        fatally_break_rust(self.tcx.sess);
                    }
                }
            }
            // There was an error; make type-check fail.
            tcx.types.err
        }
    }

    fn check_expr_return(
        &self,
        expr_opt: Option<&'tcx hir::Expr>,
        expr: &'tcx hir::Expr
    ) -> Ty<'tcx> {
        if self.ret_coercion.is_none() {
            struct_span_err!(self.tcx.sess, expr.span, E0572,
                                "return statement outside of function body").emit();
        } else if let Some(ref e) = expr_opt {
            if self.ret_coercion_span.borrow().is_none() {
                *self.ret_coercion_span.borrow_mut() = Some(e.span);
            }
            self.check_return_expr(e);
        } else {
            let mut coercion = self.ret_coercion.as_ref().unwrap().borrow_mut();
            if self.ret_coercion_span.borrow().is_none() {
                *self.ret_coercion_span.borrow_mut() = Some(expr.span);
            }
            let cause = self.cause(expr.span, ObligationCauseCode::ReturnNoExpression);
            if let Some((fn_decl, _)) = self.get_fn_decl(expr.hir_id) {
                coercion.coerce_forced_unit(
                    self,
                    &cause,
                    &mut |db| {
                        db.span_label(
                            fn_decl.output.span(),
                            format!(
                                "expected `{}` because of this return type",
                                fn_decl.output,
                            ),
                        );
                    },
                    true,
                );
            } else {
                coercion.coerce_forced_unit(self, &cause, &mut |_| (), true);
            }
        }
        self.tcx.types.never
    }

    pub(super) fn check_return_expr(&self, return_expr: &'tcx hir::Expr) {
        let ret_coercion =
            self.ret_coercion
                .as_ref()
                .unwrap_or_else(|| span_bug!(return_expr.span,
                                             "check_return_expr called outside fn body"));

        let ret_ty = ret_coercion.borrow().expected_ty();
        let return_expr_ty = self.check_expr_with_hint(return_expr, ret_ty.clone());
        ret_coercion.borrow_mut()
                    .coerce(self,
                            &self.cause(return_expr.span,
                                        ObligationCauseCode::ReturnType(return_expr.hir_id)),
                            return_expr,
                            return_expr_ty);
    }

    /// Type check assignment expression `expr` of form `lhs = rhs`.
    /// The expected type is `()` and is passsed to the function for the purposes of diagnostics.
    fn check_expr_assign(
        &self,
        expr: &'tcx hir::Expr,
        expected: Expectation<'tcx>,
        lhs: &'tcx hir::Expr,
        rhs: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let lhs_ty = self.check_expr_with_needs(&lhs, Needs::MutPlace);
        let rhs_ty = self.check_expr_coercable_to_type(&rhs, lhs_ty);

        let expected_ty = expected.coercion_target_type(self, expr.span);
        if expected_ty == self.tcx.types.bool {
            // The expected type is `bool` but this will result in `()` so we can reasonably
            // say that the user intended to write `lhs == rhs` instead of `lhs = rhs`.
            // The likely cause of this is `if foo = bar { .. }`.
            let actual_ty = self.tcx.mk_unit();
            let mut err = self.demand_suptype_diag(expr.span, expected_ty, actual_ty).unwrap();
            let msg = "try comparing for equality";
            let left = self.tcx.sess.source_map().span_to_snippet(lhs.span);
            let right = self.tcx.sess.source_map().span_to_snippet(rhs.span);
            if let (Ok(left), Ok(right)) = (left, right) {
                let help = format!("{} == {}", left, right);
                err.span_suggestion(expr.span, msg, help, Applicability::MaybeIncorrect);
            } else {
                err.help(msg);
            }
            err.emit();
        } else if !lhs.is_place_expr() {
            struct_span_err!(self.tcx.sess, expr.span, E0070,
                                "invalid left-hand side expression")
                .span_label(expr.span, "left-hand of expression not valid")
                .emit();
        }

        self.require_type_is_sized(lhs_ty, lhs.span, traits::AssignmentLhsSized);

        if lhs_ty.references_error() || rhs_ty.references_error() {
            self.tcx.types.err
        } else {
            self.tcx.mk_unit()
        }
    }

    fn check_expr_while(
        &self,
        cond: &'tcx hir::Expr,
        body: &'tcx hir::Block,
        expr: &'tcx hir::Expr
    ) -> Ty<'tcx> {
        let ctxt = BreakableCtxt {
            // Cannot use break with a value from a while loop.
            coerce: None,
            may_break: false, // Will get updated if/when we find a `break`.
        };

        let (ctxt, ()) = self.with_breakable_ctxt(expr.hir_id, ctxt, || {
            self.check_expr_has_type_or_error(&cond, self.tcx.types.bool);
            let cond_diverging = self.diverges.get();
            self.check_block_no_value(&body);

            // We may never reach the body so it diverging means nothing.
            self.diverges.set(cond_diverging);
        });

        if ctxt.may_break {
            // No way to know whether it's diverging because
            // of a `break` or an outer `break` or `return`.
            self.diverges.set(Diverges::Maybe);
        }

        self.tcx.mk_unit()
    }

    fn check_expr_loop(
        &self,
        body: &'tcx hir::Block,
        source: hir::LoopSource,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let coerce = match source {
            // you can only use break with a value from a normal `loop { }`
            hir::LoopSource::Loop => {
                let coerce_to = expected.coercion_target_type(self, body.span);
                Some(CoerceMany::new(coerce_to))
            }

            hir::LoopSource::WhileLet |
            hir::LoopSource::ForLoop => {
                None
            }
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
        expr: &'tcx hir::Expr,
        segment: &hir::PathSegment,
        span: Span,
        args: &'tcx [hir::Expr],
        expected: Expectation<'tcx>,
        needs: Needs,
    ) -> Ty<'tcx> {
        let rcvr = &args[0];
        let rcvr_t = self.check_expr_with_needs(&rcvr, needs);
        // no need to check for bot/err -- callee does that
        let rcvr_t = self.structurally_resolved_type(args[0].span, rcvr_t);

        let method = match self.lookup_method(rcvr_t,
                                              segment,
                                              span,
                                              expr,
                                              rcvr) {
            Ok(method) => {
                self.write_method_call(expr.hir_id, method);
                Ok(method)
            }
            Err(error) => {
                if segment.ident.name != kw::Invalid {
                    self.report_method_error(span,
                                             rcvr_t,
                                             segment.ident,
                                             SelfSource::MethodCall(rcvr),
                                             error,
                                             Some(args));
                }
                Err(())
            }
        };

        // Call the generic checker.
        self.check_method_argument_types(span,
                                         expr.span,
                                         method,
                                         &args[1..],
                                         DontTupleArguments,
                                         expected)
    }

    fn check_expr_cast(
        &self,
        e: &'tcx hir::Expr,
        t: &'tcx hir::Ty,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        // Find the type of `e`. Supply hints based on the type we are casting to,
        // if appropriate.
        let t_cast = self.to_ty_saving_user_provided_ty(t);
        let t_cast = self.resolve_vars_if_possible(&t_cast);
        let t_expr = self.check_expr_with_expectation(e, ExpectCastableToType(t_cast));
        let t_cast = self.resolve_vars_if_possible(&t_cast);

        // Eagerly check for some obvious errors.
        if t_expr.references_error() || t_cast.references_error() {
            self.tcx.types.err
        } else {
            // Defer other checks until we're done type checking.
            let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
            match cast::CastCheck::new(self, e, t_expr, t_cast, t.span, expr.span) {
                Ok(cast_check) => {
                    deferred_cast_checks.push(cast_check);
                    t_cast
                }
                Err(ErrorReported) => {
                    self.tcx.types.err
                }
            }
        }
    }

    fn check_expr_array(
        &self,
        args: &'tcx [hir::Expr],
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr
    ) -> Ty<'tcx> {
        let uty = expected.to_option(self).and_then(|uty| {
            match uty.sty {
                ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                _ => None
            }
        });

        let element_ty = if !args.is_empty() {
            let coerce_to = uty.unwrap_or_else(|| {
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
        element: &'tcx hir::Expr,
        count: &'tcx hir::AnonConst,
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;
        let count_def_id = tcx.hir().local_def_id_from_hir_id(count.hir_id);
        let count = if self.const_param_def_id(count).is_some() {
            Ok(self.to_const(count, tcx.type_of(count_def_id)))
        } else {
            let param_env = ty::ParamEnv::empty();
            let substs = InternalSubsts::identity_for_item(tcx.global_tcx(), count_def_id);
            let instance = ty::Instance::resolve(
                tcx.global_tcx(),
                param_env,
                count_def_id,
                substs,
            ).unwrap();
            let global_id = GlobalId {
                instance,
                promoted: None
            };

            tcx.const_eval(param_env.and(global_id))
        };

        let uty = match expected {
            ExpectHasType(uty) => {
                match uty.sty {
                    ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                    _ => None
                }
            }
            _ => None
        };

        let (element_ty, t) = match uty {
            Some(uty) => {
                self.check_expr_coercable_to_type(&element, uty);
                (uty, uty)
            }
            None => {
                let ty = self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: element.span,
                });
                let element_ty = self.check_expr_has_type_or_error(&element, ty);
                (element_ty, ty)
            }
        };

        if let Ok(count) = count {
            let zero_or_one = count.assert_usize(tcx).map_or(false, |count| count <= 1);
            if !zero_or_one {
                // For [foo, ..n] where n > 1, `foo` must have
                // Copy type:
                let lang_item = tcx.require_lang_item(lang_items::CopyTraitLangItem);
                self.require_type_meets(t, expr.span, traits::RepeatVec, lang_item);
            }
        }

        if element_ty.references_error() {
            tcx.types.err
        } else if let Ok(count) = count {
            tcx.mk_ty(ty::Array(t, count))
        } else {
            tcx.types.err
        }
    }

    fn check_expr_tuple(
        &self,
        elts: &'tcx [hir::Expr],
        expected: Expectation<'tcx>,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let flds = expected.only_has_type(self).and_then(|ty| {
            let ty = self.resolve_type_vars_with_obligations(ty);
            match ty.sty {
                ty::Tuple(ref flds) => Some(&flds[..]),
                _ => None
            }
        });

        let elt_ts_iter = elts.iter().enumerate().map(|(i, e)| {
            let t = match flds {
                Some(ref fs) if i < fs.len() => {
                    let ety = fs[i].expect_ty();
                    self.check_expr_coercable_to_type(&e, ety);
                    ety
                }
                _ => {
                    self.check_expr_with_expectation(&e, NoExpectation)
                }
            };
            t
        });
        let tuple = self.tcx.mk_tup(elt_ts_iter);
        if tuple.references_error() {
            self.tcx.types.err
        } else {
            self.require_type_is_sized(tuple, expr.span, traits::TupleInitializerSized);
            tuple
        }
    }

    fn check_expr_struct(
        &self,
        expr: &hir::Expr,
        expected: Expectation<'tcx>,
        qpath: &QPath,
        fields: &'tcx [hir::Field],
        base_expr: &'tcx Option<P<hir::Expr>>,
    ) -> Ty<'tcx> {
        // Find the relevant variant
        let (variant, adt_ty) =
            if let Some(variant_ty) = self.check_struct_path(qpath, expr.hir_id) {
                variant_ty
            } else {
                self.check_struct_fields_on_error(fields, base_expr);
                return self.tcx.types.err;
            };

        let path_span = match *qpath {
            QPath::Resolved(_, ref path) => path.span,
            QPath::TypeRelative(ref qself, _) => qself.span
        };

        // Prohibit struct expressions when non-exhaustive flag is set.
        let adt = adt_ty.ty_adt_def().expect("`check_struct_path` returned non-ADT type");
        if !adt.did.is_local() && variant.is_field_list_non_exhaustive() {
            span_err!(self.tcx.sess, expr.span, E0639,
                      "cannot create non-exhaustive {} using struct expression",
                      adt.variant_descr());
        }

        let error_happened = self.check_expr_struct_fields(adt_ty, expected, expr.hir_id, path_span,
                                                           variant, fields, base_expr.is_none());
        if let &Some(ref base_expr) = base_expr {
            // If check_expr_struct_fields hit an error, do not attempt to populate
            // the fields with the base_expr. This could cause us to hit errors later
            // when certain fields are assumed to exist that in fact do not.
            if !error_happened {
                self.check_expr_has_type_or_error(base_expr, adt_ty);
                match adt_ty.sty {
                    ty::Adt(adt, substs) if adt.is_struct() => {
                        let fru_field_types = adt.non_enum_variant().fields.iter().map(|f| {
                            self.normalize_associated_types_in(expr.span, &f.ty(self.tcx, substs))
                        }).collect();

                        self.tables
                            .borrow_mut()
                            .fru_field_types_mut()
                            .insert(expr.hir_id, fru_field_types);
                    }
                    _ => {
                        span_err!(self.tcx.sess, base_expr.span, E0436,
                                  "functional record update syntax requires a struct");
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
        ast_fields: &'tcx [hir::Field],
        check_completeness: bool,
    ) -> bool {
        let tcx = self.tcx;

        let adt_ty_hint =
            self.expected_inputs_for_expected_output(span, expected, adt_ty, &[adt_ty])
                .get(0).cloned().unwrap_or(adt_ty);
        // re-link the regions that EIfEO can erase.
        self.demand_eqtype(span, adt_ty_hint, adt_ty);

        let (substs, adt_kind, kind_name) = match &adt_ty.sty {
            &ty::Adt(adt, substs) => {
                (substs, adt.adt_kind(), adt.variant_descr())
            }
            _ => span_bug!(span, "non-ADT passed to check_expr_struct_fields")
        };

        let mut remaining_fields = variant.fields.iter().enumerate().map(|(i, field)|
            (field.ident.modern(), (i, field))
        ).collect::<FxHashMap<_, _>>();

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
                    tcx.check_stability(v_field.did, Some(expr_id), field.span);
                }

                self.field_ty(field.span, v_field, substs)
            } else {
                error_happened = true;
                if let Some(prev_span) = seen_fields.get(&ident) {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                   field.ident.span,
                                                   E0062,
                                                   "field `{}` specified more than once",
                                                   ident);

                    err.span_label(field.ident.span, "used more than once");
                    err.span_label(*prev_span, format!("first use of `{}`", ident));

                    err.emit();
                } else {
                    self.report_unknown_field(adt_ty, variant, field, ast_fields, kind_name, span);
                }

                tcx.types.err
            };

            // Make sure to give a type to the field even if there's
            // an error, so we can continue type-checking.
            self.check_expr_coercable_to_type(&field.expr, field_type);
        }

        // Make sure the programmer specified correct number of fields.
        if kind_name == "union" {
            if ast_fields.len() != 1 {
                tcx.sess.span_err(span, "union expressions should have exactly one field");
            }
        } else if check_completeness && !error_happened && !remaining_fields.is_empty() {
            let len = remaining_fields.len();

            let mut displayable_field_names = remaining_fields
                                              .keys()
                                              .map(|ident| ident.as_str())
                                              .collect::<Vec<_>>();

            displayable_field_names.sort();

            let truncated_fields_error = if len <= 3 {
                String::new()
            } else {
                format!(" and {} other field{}", (len - 3), if len - 3 == 1 {""} else {"s"})
            };

            let remaining_fields_names = displayable_field_names.iter().take(3)
                                        .map(|n| format!("`{}`", n))
                                        .collect::<Vec<_>>()
                                        .join(", ");

            struct_span_err!(tcx.sess, span, E0063,
                             "missing field{} {}{} in initializer of `{}`",
                             if remaining_fields.len() == 1 { "" } else { "s" },
                             remaining_fields_names,
                             truncated_fields_error,
                             adt_ty)
                .span_label(span, format!("missing {}{}",
                                          remaining_fields_names,
                                          truncated_fields_error))
                .emit();
        }
        error_happened
    }

    fn check_struct_fields_on_error(
        &self,
        fields: &'tcx [hir::Field],
        base_expr: &'tcx Option<P<hir::Expr>>,
    ) {
        for field in fields {
            self.check_expr(&field.expr);
        }
        if let Some(ref base) = *base_expr {
            self.check_expr(&base);
        }
    }

    fn report_unknown_field(
        &self,
        ty: Ty<'tcx>,
        variant: &'tcx ty::VariantDef,
        field: &hir::Field,
        skip_fields: &[hir::Field],
        kind_name: &str,
        ty_span: Span
    ) {
        if variant.recovered {
            return;
        }
        let mut err = self.type_error_struct_with_diag(
            field.ident.span,
            |actual| match ty.sty {
                ty::Adt(adt, ..) if adt.is_enum() => {
                    struct_span_err!(self.tcx.sess, field.ident.span, E0559,
                                     "{} `{}::{}` has no field named `{}`",
                                     kind_name, actual, variant.ident, field.ident)
                }
                _ => {
                    struct_span_err!(self.tcx.sess, field.ident.span, E0560,
                                     "{} `{}` has no field named `{}`",
                                     kind_name, actual, field.ident)
                }
            },
            ty);
        match variant.ctor_kind {
            CtorKind::Fn => {
                err.span_label(variant.ident.span, format!("`{adt}` defined here", adt=ty));
                err.span_label(field.ident.span, "field does not exist");
                err.span_label(ty_span, format!(
                        "`{adt}` is a tuple {kind_name}, \
                         use the appropriate syntax: `{adt}(/* fields */)`",
                    adt=ty,
                    kind_name=kind_name
                ));
            }
            _ => {
                // prevent all specified fields from being suggested
                let skip_fields = skip_fields.iter().map(|ref x| x.ident.as_str());
                if let Some(field_name) = Self::suggest_field_name(
                    variant,
                    &field.ident.as_str(),
                    skip_fields.collect()
                ) {
                    err.span_suggestion(
                        field.ident.span,
                        "a field with a similar name exists",
                        field_name.to_string(),
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    match ty.sty {
                        ty::Adt(adt, ..) => {
                            if adt.is_enum() {
                                err.span_label(field.ident.span, format!(
                                    "`{}::{}` does not have this field",
                                    ty,
                                    variant.ident
                                ));
                            } else {
                                err.span_label(field.ident.span, format!(
                                    "`{}` does not have this field",
                                    ty
                                ));
                            }
                            let available_field_names = self.available_field_names(variant);
                            if !available_field_names.is_empty() {
                                err.note(&format!("available fields are: {}",
                                                  self.name_series_display(available_field_names)));
                            }
                        }
                        _ => bug!("non-ADT passed to report_unknown_field")
                    }
                };
            }
        }
        err.emit();
    }

    // Return an hint about the closest match in field names
    fn suggest_field_name(variant: &'tcx ty::VariantDef,
                          field: &str,
                          skip: Vec<LocalInternedString>)
                          -> Option<Symbol> {
        let names = variant.fields.iter().filter_map(|field| {
            // ignore already set fields and private fields from non-local crates
            if skip.iter().any(|x| *x == field.ident.as_str()) ||
               (!variant.def_id.is_local() && field.vis != Visibility::Public)
            {
                None
            } else {
                Some(&field.ident.name)
            }
        });

        find_best_match_for_name(names, field, None)
    }

    fn available_field_names(&self, variant: &'tcx ty::VariantDef) -> Vec<ast::Name> {
        variant.fields.iter().filter(|field| {
            let def_scope =
                self.tcx.adjust_ident_and_get_scope(field.ident, variant.def_id, self.body_id).1;
            field.vis.is_accessible_from(def_scope, self.tcx)
        })
        .map(|field| field.ident.name)
        .collect()
    }

    fn name_series_display(&self, names: Vec<ast::Name>) -> String {
        // dynamic limit, to never omit just one field
        let limit = if names.len() == 6 { 6 } else { 5 };
        let mut display = names.iter().take(limit)
            .map(|n| format!("`{}`", n)).collect::<Vec<_>>().join(", ");
        if names.len() > limit {
            display = format!("{} ... and {} others", display, names.len() - limit);
        }
        display
    }

    // Check field access expressions
    fn check_field(
        &self,
        expr: &'tcx hir::Expr,
        needs: Needs,
        base: &'tcx hir::Expr,
        field: ast::Ident,
    ) -> Ty<'tcx> {
        let expr_t = self.check_expr_with_needs(base, needs);
        let expr_t = self.structurally_resolved_type(base.span,
                                                     expr_t);
        let mut private_candidate = None;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, _)) = autoderef.next() {
            match base_t.sty {
                ty::Adt(base_def, substs) if !base_def.is_enum() => {
                    debug!("struct named {:?}",  base_t);
                    let (ident, def_scope) =
                        self.tcx.adjust_ident_and_get_scope(field, base_def.did, self.body_id);
                    let fields = &base_def.non_enum_variant().fields;
                    if let Some(index) = fields.iter().position(|f| f.ident.modern() == ident) {
                        let field = &fields[index];
                        let field_ty = self.field_ty(expr.span, field, substs);
                        // Save the index of all fields regardless of their visibility in case
                        // of error recovery.
                        self.write_field_index(expr.hir_id, index);
                        if field.vis.is_accessible_from(def_scope, self.tcx) {
                            let adjustments = autoderef.adjust_steps(self, needs);
                            self.apply_adjustments(base, adjustments);
                            autoderef.finalize(self);

                            self.tcx.check_stability(field.did, Some(expr.hir_id), expr.span);
                            return field_ty;
                        }
                        private_candidate = Some((base_def.did, field_ty));
                    }
                }
                ty::Tuple(ref tys) => {
                    let fstr = field.as_str();
                    if let Ok(index) = fstr.parse::<usize>() {
                        if fstr == index.to_string() {
                            if let Some(field_ty) = tys.get(index) {
                                let adjustments = autoderef.adjust_steps(self, needs);
                                self.apply_adjustments(base, adjustments);
                                autoderef.finalize(self);

                                self.write_field_index(expr.hir_id, index);
                                return field_ty.expect_ty();
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        autoderef.unambiguous_final_ty(self);

        if let Some((did, field_ty)) = private_candidate {
            let struct_path = self.tcx().def_path_str(did);
            let mut err = struct_span_err!(self.tcx().sess, expr.span, E0616,
                                           "field `{}` of struct `{}` is private",
                                           field, struct_path);
            // Also check if an accessible method exists, which is often what is meant.
            if self.method_exists(field, expr_t, expr.hir_id, false)
                && !self.expr_in_place(expr.hir_id)
            {
                self.suggest_method_call(
                    &mut err,
                    &format!("a method `{}` also exists, call it with parentheses", field),
                    field,
                    expr_t,
                    expr.hir_id,
                );
            }
            err.emit();
            field_ty
        } else if field.name == kw::Invalid {
            self.tcx().types.err
        } else if self.method_exists(field, expr_t, expr.hir_id, true) {
            let mut err = type_error_struct!(self.tcx().sess, field.span, expr_t, E0615,
                               "attempted to take value of method `{}` on type `{}`",
                               field, expr_t);

            if !self.expr_in_place(expr.hir_id) {
                self.suggest_method_call(
                    &mut err,
                    "use parentheses to call the method",
                    field,
                    expr_t,
                    expr.hir_id
                );
            } else {
                err.help("methods are immutable and cannot be assigned to");
            }

            err.emit();
            self.tcx().types.err
        } else {
            if !expr_t.is_primitive_ty() {
                let mut err = self.no_such_field_err(field.span, field, expr_t);

                match expr_t.sty {
                    ty::Adt(def, _) if !def.is_enum() => {
                        if let Some(suggested_field_name) =
                            Self::suggest_field_name(def.non_enum_variant(),
                                                     &field.as_str(), vec![]) {
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
                                    err.note(&format!("available fields are: {}",
                                                      self.name_series_display(field_names)));
                                }
                            };
                    }
                    ty::Array(_, len) => {
                        if let (Some(len), Ok(user_index)) = (
                            len.assert_usize(self.tcx),
                            field.as_str().parse::<u64>()
                        ) {
                            let base = self.tcx.sess.source_map()
                                .span_to_snippet(base.span)
                                .unwrap_or_else(|_|
                                    self.tcx.hir().hir_to_pretty_string(base.hir_id));
                            let help = "instead of using tuple indexing, use array indexing";
                            let suggestion = format!("{}[{}]", base, field);
                            let applicability = if len < user_index {
                                Applicability::MachineApplicable
                            } else {
                                Applicability::MaybeIncorrect
                            };
                            err.span_suggestion(
                                expr.span, help, suggestion, applicability
                            );
                        }
                    }
                    ty::RawPtr(..) => {
                        let base = self.tcx.sess.source_map()
                            .span_to_snippet(base.span)
                            .unwrap_or_else(|_| self.tcx.hir().hir_to_pretty_string(base.hir_id));
                        let msg = format!("`{}` is a raw pointer; try dereferencing it", base);
                        let suggestion = format!("(*{}).{}", base, field);
                        err.span_suggestion(
                            expr.span,
                            &msg,
                            suggestion,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    _ => {}
                }
                err
            } else {
                type_error_struct!(self.tcx().sess, field.span, expr_t, E0610,
                                   "`{}` is a primitive type and therefore doesn't have fields",
                                   expr_t)
            }.emit();
            self.tcx().types.err
        }
    }

    fn no_such_field_err<T: Display>(&self, span: Span, field: T, expr_t: &ty::TyS<'_>)
        -> DiagnosticBuilder<'_> {
        type_error_struct!(self.tcx().sess, span, expr_t, E0609,
                           "no field `{}` on type `{}`",
                           field, expr_t)
    }

    fn check_expr_index(
        &self,
        base: &'tcx hir::Expr,
        idx: &'tcx hir::Expr,
        needs: Needs,
        expr: &'tcx hir::Expr,
    ) -> Ty<'tcx> {
        let base_t = self.check_expr_with_needs(&base, needs);
        let idx_t = self.check_expr(&idx);

        if base_t.references_error() {
            base_t
        } else if idx_t.references_error() {
            idx_t
        } else {
            let base_t = self.structurally_resolved_type(base.span, base_t);
            match self.lookup_indexing(expr, base, base_t, idx_t, needs) {
                Some((index_ty, element_ty)) => {
                    // two-phase not needed because index_ty is never mutable
                    self.demand_coerce(idx, idx_t, index_ty, AllowTwoPhase::No);
                    element_ty
                }
                None => {
                    let mut err =
                        type_error_struct!(self.tcx.sess, expr.span, base_t, E0608,
                                            "cannot index into a value of type `{}`",
                                            base_t);
                    // Try to give some advice about indexing tuples.
                    if let ty::Tuple(..) = base_t.sty {
                        let mut needs_note = true;
                        // If the index is an integer, we can show the actual
                        // fixed expression:
                        if let ExprKind::Lit(ref lit) = idx.node {
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
                            err.help("to access tuple elements, use tuple indexing \
                                        syntax (e.g., `tuple.0`)");
                        }
                    }
                    err.emit();
                    self.tcx.types.err
                }
            }
        }
    }

    fn check_expr_yield(
        &self,
        value: &'tcx hir::Expr,
        expr: &'tcx hir::Expr,
        src: &'tcx hir::YieldSource
    ) -> Ty<'tcx> {
        match self.yield_ty {
            Some(ty) => {
                self.check_expr_coercable_to_type(&value, ty);
            }
            // Given that this `yield` expression was generated as a result of lowering a `.await`,
            // we know that the yield type must be `()`; however, the context won't contain this
            // information. Hence, we check the source of the yield expression here and check its
            // value's type against `()` (this check should always hold).
            None if src == &hir::YieldSource::Await => {
                self.check_expr_coercable_to_type(&value, self.tcx.mk_unit());
            }
            _ => {
                struct_span_err!(self.tcx.sess, expr.span, E0627,
                                    "yield statement outside of generator literal").emit();
            }
        }
        self.tcx.mk_unit()
    }
}
