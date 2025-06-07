use std::{fmt, iter, mem};

use itertools::Itertools;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, ErrorGuaranteed, MultiSpan, a_or_an, listify, pluralize};
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{ExprKind, HirId, LangItem, Node, QPath};
use rustc_hir_analysis::check::potentially_plural_count;
use rustc_hir_analysis::hir_ty_lowering::{HirTyLowerer, PermitVariants};
use rustc_index::IndexVec;
use rustc_infer::infer::{DefineOpaqueTypes, InferOk, TypeTrace};
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{self, IsSuggestable, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_session::Session;
use rustc_span::{DUMMY_SP, Ident, Span, kw, sym};
use rustc_trait_selection::error_reporting::infer::{FailureCode, ObligationCauseExt};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{self, ObligationCauseCode, ObligationCtxt, SelectionContext};
use smallvec::SmallVec;
use tracing::debug;
use {rustc_ast as ast, rustc_hir as hir};

use crate::Expectation::*;
use crate::TupleArgumentsFlag::*;
use crate::coercion::CoerceMany;
use crate::errors::SuggestPtrNullMut;
use crate::fn_ctxt::arg_matrix::{ArgMatrix, Compatibility, Error, ExpectedIdx, ProvidedIdx};
use crate::fn_ctxt::infer::FnCall;
use crate::gather_locals::Declaration;
use crate::inline_asm::InlineAsmCtxt;
use crate::method::probe::IsSuggestion;
use crate::method::probe::Mode::MethodCall;
use crate::method::probe::ProbeScope::TraitsInScope;
use crate::{
    BreakableCtxt, Diverges, Expectation, FnCtxt, GatherLocalsVisitor, LoweredTy, Needs,
    TupleArgumentsFlag, errors, struct_span_code_err,
};

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "GenericIdx({})"]
    pub(crate) struct GenericIdx {}
}

#[derive(Clone, Copy, Default)]
pub(crate) enum DivergingBlockBehavior {
    /// This is the current stable behavior:
    ///
    /// ```rust
    /// {
    ///     return;
    /// } // block has type = !, even though we are supposedly dropping it with `;`
    /// ```
    #[default]
    Never,

    /// Alternative behavior:
    ///
    /// ```ignore (very-unstable-new-attribute)
    /// #![rustc_never_type_options(diverging_block_default = "unit")]
    /// {
    ///     return;
    /// } // block has type = (), since we are dropping `!` from `return` with `;`
    /// ```
    Unit,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(in super::super) fn check_casts(&mut self) {
        // don't hold the borrow to deferred_cast_checks while checking to avoid borrow checker errors
        // when writing to `self.param_env`.
        let mut deferred_cast_checks = mem::take(&mut *self.deferred_cast_checks.borrow_mut());

        debug!("FnCtxt::check_casts: {} deferred checks", deferred_cast_checks.len());
        for cast in deferred_cast_checks.drain(..) {
            cast.check(self);
        }

        *self.deferred_cast_checks.borrow_mut() = deferred_cast_checks;
    }

    pub(in super::super) fn check_transmutes(&self) {
        let mut deferred_transmute_checks = self.deferred_transmute_checks.borrow_mut();
        debug!("FnCtxt::check_transmutes: {} deferred checks", deferred_transmute_checks.len());
        for (from, to, hir_id) in deferred_transmute_checks.drain(..) {
            self.check_transmute(from, to, hir_id);
        }
    }

    pub(in super::super) fn check_asms(&self) {
        let mut deferred_asm_checks = self.deferred_asm_checks.borrow_mut();
        debug!("FnCtxt::check_asm: {} deferred checks", deferred_asm_checks.len());
        for (asm, hir_id) in deferred_asm_checks.drain(..) {
            let enclosing_id = self.tcx.hir_enclosing_body_owner(hir_id);
            InlineAsmCtxt::new(self, enclosing_id).check_asm(asm);
        }
    }

    pub(in super::super) fn check_repeat_exprs(&self) {
        let mut deferred_repeat_expr_checks = self.deferred_repeat_expr_checks.borrow_mut();
        debug!("FnCtxt::check_repeat_exprs: {} deferred checks", deferred_repeat_expr_checks.len());

        let deferred_repeat_expr_checks = deferred_repeat_expr_checks
            .drain(..)
            .flat_map(|(element, element_ty, count)| {
                // Actual constants as the repeat element are inserted repeatedly instead
                // of being copied via `Copy`, so we don't need to attempt to structurally
                // resolve the repeat count which may unnecessarily error.
                match &element.kind {
                    hir::ExprKind::ConstBlock(..) => return None,
                    hir::ExprKind::Path(qpath) => {
                        let res = self.typeck_results.borrow().qpath_res(qpath, element.hir_id);
                        if let Res::Def(DefKind::Const | DefKind::AssocConst, _) = res {
                            return None;
                        }
                    }
                    _ => {}
                }

                // We want to emit an error if the const is not structurally resolveable
                // as otherwise we can wind up conservatively proving `Copy` which may
                // infer the repeat expr count to something that never required `Copy` in
                // the first place.
                let count = self
                    .structurally_resolve_const(element.span, self.normalize(element.span, count));

                // Avoid run on "`NotCopy: Copy` is not implemented" errors when the
                // repeat expr count is erroneous/unknown. The user might wind up
                // specifying a repeat count of 0/1.
                if count.references_error() {
                    return None;
                }

                Some((element, element_ty, count))
            })
            // We collect to force the side effects of structurally resolving the repeat
            // count to happen in one go, to avoid side effects from proving `Copy`
            // affecting whether repeat counts are known or not. If we did not do this we
            // would get results that depend on the order that we evaluate each repeat
            // expr's `Copy` check.
            .collect::<Vec<_>>();

        let enforce_copy_bound = |element: &hir::Expr<'_>, element_ty| {
            // If someone calls a const fn or constructs a const value, they can extract that
            // out into a separate constant (or a const block in the future), so we check that
            // to tell them that in the diagnostic. Does not affect typeck.
            let is_constable = match element.kind {
                hir::ExprKind::Call(func, _args) => match *self.node_ty(func.hir_id).kind() {
                    ty::FnDef(def_id, _) if self.tcx.is_stable_const_fn(def_id) => {
                        traits::IsConstable::Fn
                    }
                    _ => traits::IsConstable::No,
                },
                hir::ExprKind::Path(qpath) => {
                    match self.typeck_results.borrow().qpath_res(&qpath, element.hir_id) {
                        Res::Def(DefKind::Ctor(_, CtorKind::Const), _) => traits::IsConstable::Ctor,
                        _ => traits::IsConstable::No,
                    }
                }
                _ => traits::IsConstable::No,
            };

            let lang_item = self.tcx.require_lang_item(LangItem::Copy, element.span);
            let code = traits::ObligationCauseCode::RepeatElementCopy {
                is_constable,
                elt_span: element.span,
            };
            self.require_type_meets(element_ty, element.span, code, lang_item);
        };

        for (element, element_ty, count) in deferred_repeat_expr_checks {
            match count.kind() {
                ty::ConstKind::Value(val) => {
                    if val.try_to_target_usize(self.tcx).is_none_or(|count| count > 1) {
                        enforce_copy_bound(element, element_ty)
                    } else {
                        // If the length is 0 or 1 we don't actually copy the element, we either don't create it
                        // or we just use the one value.
                    }
                }

                // If the length is a generic parameter or some rigid alias then conservatively
                // require `element_ty: Copy` as it may wind up being `>1` after monomorphization.
                ty::ConstKind::Param(_)
                | ty::ConstKind::Expr(_)
                | ty::ConstKind::Placeholder(_)
                | ty::ConstKind::Unevaluated(_) => enforce_copy_bound(element, element_ty),

                ty::ConstKind::Bound(_, _) | ty::ConstKind::Infer(_) | ty::ConstKind::Error(_) => {
                    unreachable!()
                }
            }
        }
    }

    /// Generic function that factors out common logic from function calls,
    /// method calls and overloaded operators.
    pub(in super::super) fn check_argument_types(
        &self,
        // Span enclosing the call site
        call_span: Span,
        // Expression of the call site
        call_expr: &'tcx hir::Expr<'tcx>,
        // Types (as defined in the *signature* of the target function)
        formal_input_tys: &[Ty<'tcx>],
        formal_output: Ty<'tcx>,
        // Expected output from the parent expression or statement
        expectation: Expectation<'tcx>,
        // The expressions for each provided argument
        provided_args: &'tcx [hir::Expr<'tcx>],
        // Whether the function is variadic, for example when imported from C
        c_variadic: bool,
        // Whether the arguments have been bundled in a tuple (ex: closures)
        tuple_arguments: TupleArgumentsFlag,
        // The DefId for the function being called, for better error messages
        fn_def_id: Option<DefId>,
    ) {
        let tcx = self.tcx;

        // Conceptually, we've got some number of expected inputs, and some number of provided arguments
        // and we can form a grid of whether each argument could satisfy a given input:
        //      in1 | in2 | in3 | ...
        // arg1  ?  |     |     |
        // arg2     |  ?  |     |
        // arg3     |     |  ?  |
        // ...
        // Initially, we just check the diagonal, because in the case of correct code
        // these are the only checks that matter
        // However, in the unhappy path, we'll fill in this whole grid to attempt to provide
        // better error messages about invalid method calls.

        // All the input types from the fn signature must outlive the call
        // so as to validate implied bounds.
        for (&fn_input_ty, arg_expr) in iter::zip(formal_input_tys, provided_args) {
            self.register_wf_obligation(
                fn_input_ty.into(),
                arg_expr.span,
                ObligationCauseCode::WellFormed(None),
            );
        }

        // First, let's unify the formal method signature with the expectation eagerly.
        // We use this to guide coercion inference; it's output is "fudged" which means
        // any remaining type variables are assigned to new, unrelated variables. This
        // is because the inference guidance here is only speculative.
        let formal_output = self.resolve_vars_with_obligations(formal_output);
        let expected_input_tys: Option<Vec<_>> = expectation
            .only_has_type(self)
            .and_then(|expected_output| {
                self.fudge_inference_if_ok(|| {
                    let ocx = ObligationCtxt::new(self);

                    // Attempt to apply a subtyping relationship between the formal
                    // return type (likely containing type variables if the function
                    // is polymorphic) and the expected return type.
                    // No argument expectations are produced if unification fails.
                    let origin = self.misc(call_span);
                    ocx.sup(&origin, self.param_env, expected_output, formal_output)?;
                    if !ocx.select_where_possible().is_empty() {
                        return Err(TypeError::Mismatch);
                    }

                    // Record all the argument types, with the args
                    // produced from the above subtyping unification.
                    Ok(Some(
                        formal_input_tys
                            .iter()
                            .map(|&ty| self.resolve_vars_if_possible(ty))
                            .collect(),
                    ))
                })
                .ok()
            })
            .unwrap_or_default();

        let mut err_code = E0061;

        // If the arguments should be wrapped in a tuple (ex: closures), unwrap them here
        let (formal_input_tys, expected_input_tys) = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolve_type(call_span, formal_input_tys[0]);
            match tuple_type.kind() {
                // We expected a tuple and got a tuple
                ty::Tuple(arg_types) => {
                    // Argument length differs
                    if arg_types.len() != provided_args.len() {
                        err_code = E0057;
                    }
                    let expected_input_tys = match expected_input_tys {
                        Some(expected_input_tys) => match expected_input_tys.get(0) {
                            Some(ty) => match ty.kind() {
                                ty::Tuple(tys) => Some(tys.iter().collect()),
                                _ => None,
                            },
                            None => None,
                        },
                        None => None,
                    };
                    (arg_types.iter().collect(), expected_input_tys)
                }
                _ => {
                    // Otherwise, there's a mismatch, so clear out what we're expecting, and set
                    // our input types to err_args so we don't blow up the error messages
                    let guar = struct_span_code_err!(
                        self.dcx(),
                        call_span,
                        E0059,
                        "cannot use call notation; the first type parameter \
                         for the function trait is neither a tuple nor unit"
                    )
                    .emit();
                    (self.err_args(provided_args.len(), guar), None)
                }
            }
        } else {
            (formal_input_tys.to_vec(), expected_input_tys)
        };

        // If there are no external expectations at the call site, just use the types from the function defn
        let expected_input_tys = if let Some(expected_input_tys) = expected_input_tys {
            assert_eq!(expected_input_tys.len(), formal_input_tys.len());
            expected_input_tys
        } else {
            formal_input_tys.clone()
        };

        let minimum_input_count = expected_input_tys.len();
        let provided_arg_count = provided_args.len();

        // We introduce a helper function to demand that a given argument satisfy a given input
        // This is more complicated than just checking type equality, as arguments could be coerced
        // This version writes those types back so further type checking uses the narrowed types
        let demand_compatible = |idx| {
            let formal_input_ty: Ty<'tcx> = formal_input_tys[idx];
            let expected_input_ty: Ty<'tcx> = expected_input_tys[idx];
            let provided_arg = &provided_args[idx];

            debug!("checking argument {}: {:?} = {:?}", idx, provided_arg, formal_input_ty);

            // We're on the happy path here, so we'll do a more involved check and write back types
            // To check compatibility, we'll do 3 things:
            // 1. Unify the provided argument with the expected type
            let expectation = Expectation::rvalue_hint(self, expected_input_ty);

            let checked_ty = self.check_expr_with_expectation(provided_arg, expectation);

            // 2. Coerce to the most detailed type that could be coerced
            //    to, which is `expected_ty` if `rvalue_hint` returns an
            //    `ExpectHasType(expected_ty)`, or the `formal_ty` otherwise.
            let coerced_ty = expectation.only_has_type(self).unwrap_or(formal_input_ty);

            // Cause selection errors caused by resolving a single argument to point at the
            // argument and not the call. This lets us customize the span pointed to in the
            // fulfillment error to be more accurate.
            let coerced_ty = self.resolve_vars_with_obligations(coerced_ty);

            let coerce_error =
                self.coerce(provided_arg, checked_ty, coerced_ty, AllowTwoPhase::Yes, None).err();
            if coerce_error.is_some() {
                return Compatibility::Incompatible(coerce_error);
            }

            // 3. Check if the formal type is actually equal to the checked one
            //    and register any such obligations for future type checks.
            let formal_ty_error = self.at(&self.misc(provided_arg.span), self.param_env).eq(
                DefineOpaqueTypes::Yes,
                formal_input_ty,
                coerced_ty,
            );

            // If neither check failed, the types are compatible
            match formal_ty_error {
                Ok(InferOk { obligations, value: () }) => {
                    self.register_predicates(obligations);
                    Compatibility::Compatible
                }
                Err(err) => Compatibility::Incompatible(Some(err)),
            }
        };

        // To start, we only care "along the diagonal", where we expect every
        // provided arg to be in the right spot
        let mut compatibility_diagonal =
            vec![Compatibility::Incompatible(None); provided_args.len()];

        // Keep track of whether we *could possibly* be satisfied, i.e. whether we're on the happy path
        // if the wrong number of arguments were supplied, we CAN'T be satisfied,
        // and if we're c_variadic, the supplied arguments must be >= the minimum count from the function
        // otherwise, they need to be identical, because rust doesn't currently support variadic functions
        let mut call_appears_satisfied = if c_variadic {
            provided_arg_count >= minimum_input_count
        } else {
            provided_arg_count == minimum_input_count
        };

        // Check the arguments.
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for check_closures in [false, true] {
            // More awful hacks: before we check argument types, try to do
            // an "opportunistic" trait resolution of any trait bounds on
            // the call. This helps coercions.
            if check_closures {
                self.select_obligations_where_possible(|_| {})
            }

            // Check each argument, to satisfy the input it was provided for
            // Visually, we're traveling down the diagonal of the compatibility matrix
            for (idx, arg) in provided_args.iter().enumerate() {
                // Warn only for the first loop (the "no closures" one).
                // Closure arguments themselves can't be diverging, but
                // a previous argument can, e.g., `foo(panic!(), || {})`.
                if !check_closures {
                    self.warn_if_unreachable(arg.hir_id, arg.span, "expression");
                }

                // For C-variadic functions, we don't have a declared type for all of
                // the arguments hence we only do our usual type checking with
                // the arguments who's types we do know. However, we *can* check
                // for unreachable expressions (see above).
                // FIXME: unreachable warning current isn't emitted
                if idx >= minimum_input_count {
                    continue;
                }

                // For this check, we do *not* want to treat async coroutine closures (async blocks)
                // as proper closures. Doing so would regress type inference when feeding
                // the return value of an argument-position async block to an argument-position
                // closure wrapped in a block.
                // See <https://github.com/rust-lang/rust/issues/112225>.
                let is_closure = if let ExprKind::Closure(closure) = arg.kind {
                    !tcx.coroutine_is_async(closure.def_id.to_def_id())
                } else {
                    false
                };
                if is_closure != check_closures {
                    continue;
                }

                let compatible = demand_compatible(idx);
                let is_compatible = matches!(compatible, Compatibility::Compatible);
                compatibility_diagonal[idx] = compatible;

                if !is_compatible {
                    call_appears_satisfied = false;
                }
            }
        }

        if c_variadic && provided_arg_count < minimum_input_count {
            err_code = E0060;
        }

        for arg in provided_args.iter().skip(minimum_input_count) {
            // Make sure we've checked this expr at least once.
            let arg_ty = self.check_expr(arg);

            // If the function is c-style variadic, we skipped a bunch of arguments
            // so we need to check those, and write out the types
            // Ideally this would be folded into the above, for uniform style
            // but c-variadic is already a corner case
            if c_variadic {
                fn variadic_error<'tcx>(
                    sess: &'tcx Session,
                    span: Span,
                    ty: Ty<'tcx>,
                    cast_ty: &str,
                ) {
                    sess.dcx().emit_err(errors::PassToVariadicFunction {
                        span,
                        ty,
                        cast_ty,
                        sugg_span: span.shrink_to_hi(),
                        teach: sess.teach(E0617),
                    });
                }

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolve_type(arg.span, arg_ty);
                match arg_ty.kind() {
                    ty::Float(ty::FloatTy::F32) => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_double");
                    }
                    ty::Int(ty::IntTy::I8 | ty::IntTy::I16) | ty::Bool => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_int");
                    }
                    ty::Uint(ty::UintTy::U8 | ty::UintTy::U16) => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_uint");
                    }
                    ty::FnDef(..) => {
                        let fn_ptr = Ty::new_fn_ptr(self.tcx, arg_ty.fn_sig(self.tcx));
                        let fn_ptr = self.resolve_vars_if_possible(fn_ptr).to_string();

                        let fn_item_spa = arg.span;
                        tcx.sess.dcx().emit_err(errors::PassFnItemToVariadicFunction {
                            span: fn_item_spa,
                            sugg_span: fn_item_spa.shrink_to_hi(),
                            replace: fn_ptr,
                        });
                    }
                    _ => {}
                }
            }
        }

        if !call_appears_satisfied {
            let compatibility_diagonal = IndexVec::from_raw(compatibility_diagonal);
            let provided_args = IndexVec::from_iter(provided_args.iter().take(if c_variadic {
                minimum_input_count
            } else {
                provided_arg_count
            }));
            debug_assert_eq!(
                formal_input_tys.len(),
                expected_input_tys.len(),
                "expected formal_input_tys to be the same size as expected_input_tys"
            );
            let formal_and_expected_inputs = IndexVec::from_iter(
                formal_input_tys
                    .iter()
                    .copied()
                    .zip_eq(expected_input_tys.iter().copied())
                    .map(|vars| self.resolve_vars_if_possible(vars)),
            );

            self.report_arg_errors(
                compatibility_diagonal,
                formal_and_expected_inputs,
                provided_args,
                c_variadic,
                err_code,
                fn_def_id,
                call_span,
                call_expr,
                tuple_arguments,
            );
        }
    }

    fn report_arg_errors(
        &self,
        compatibility_diagonal: IndexVec<ProvidedIdx, Compatibility<'tcx>>,
        formal_and_expected_inputs: IndexVec<ExpectedIdx, (Ty<'tcx>, Ty<'tcx>)>,
        provided_args: IndexVec<ProvidedIdx, &'tcx hir::Expr<'tcx>>,
        c_variadic: bool,
        err_code: ErrCode,
        fn_def_id: Option<DefId>,
        call_span: Span,
        call_expr: &'tcx hir::Expr<'tcx>,
        tuple_arguments: TupleArgumentsFlag,
    ) -> ErrorGuaranteed {
        // Next, let's construct the error
        let (error_span, call_ident, full_call_span, call_name, is_method) = match &call_expr.kind {
            hir::ExprKind::Call(
                hir::Expr { hir_id, span, kind: hir::ExprKind::Path(qpath), .. },
                _,
            ) => {
                if let Res::Def(DefKind::Ctor(of, _), _) =
                    self.typeck_results.borrow().qpath_res(qpath, *hir_id)
                {
                    let name = match of {
                        CtorOf::Struct => "struct",
                        CtorOf::Variant => "enum variant",
                    };
                    (call_span, None, *span, name, false)
                } else {
                    (call_span, None, *span, "function", false)
                }
            }
            hir::ExprKind::Call(hir::Expr { span, .. }, _) => {
                (call_span, None, *span, "function", false)
            }
            hir::ExprKind::MethodCall(path_segment, _, _, span) => {
                let ident_span = path_segment.ident.span;
                let ident_span = if let Some(args) = path_segment.args {
                    ident_span.with_hi(args.span_ext.hi())
                } else {
                    ident_span
                };
                (*span, Some(path_segment.ident), ident_span, "method", true)
            }
            k => span_bug!(call_span, "checking argument types on a non-call: `{:?}`", k),
        };
        let args_span = error_span.trim_start(full_call_span).unwrap_or(error_span);

        // Don't print if it has error types or is just plain `_`
        fn has_error_or_infer<'tcx>(tys: impl IntoIterator<Item = Ty<'tcx>>) -> bool {
            tys.into_iter().any(|ty| ty.references_error() || ty.is_ty_var())
        }

        let tcx = self.tcx;

        // Get the argument span in the context of the call span so that
        // suggestions and labels are (more) correct when an arg is a
        // macro invocation.
        let normalize_span = |span: Span| -> Span {
            let normalized_span = span.find_ancestor_inside_same_ctxt(error_span).unwrap_or(span);
            // Sometimes macros mess up the spans, so do not normalize the
            // arg span to equal the error span, because that's less useful
            // than pointing out the arg expr in the wrong context.
            if normalized_span.source_equal(error_span) { span } else { normalized_span }
        };

        // Precompute the provided types and spans, since that's all we typically need for below
        let provided_arg_tys: IndexVec<ProvidedIdx, (Ty<'tcx>, Span)> = provided_args
            .iter()
            .map(|expr| {
                let ty = self
                    .typeck_results
                    .borrow()
                    .expr_ty_adjusted_opt(*expr)
                    .unwrap_or_else(|| Ty::new_misc_error(tcx));
                (self.resolve_vars_if_possible(ty), normalize_span(expr.span))
            })
            .collect();
        let callee_expr = match &call_expr.peel_blocks().kind {
            hir::ExprKind::Call(callee, _) => Some(*callee),
            hir::ExprKind::MethodCall(_, receiver, ..) => {
                if let Some((DefKind::AssocFn, def_id)) =
                    self.typeck_results.borrow().type_dependent_def(call_expr.hir_id)
                    && let Some(assoc) = tcx.opt_associated_item(def_id)
                    && assoc.is_method()
                {
                    Some(*receiver)
                } else {
                    None
                }
            }
            _ => None,
        };
        let callee_ty = callee_expr
            .and_then(|callee_expr| self.typeck_results.borrow().expr_ty_adjusted_opt(callee_expr));

        // Obtain another method on `Self` that have similar name.
        let similar_assoc = |call_name: Ident| -> Option<(ty::AssocItem, ty::FnSig<'_>)> {
            if let Some(callee_ty) = callee_ty
                && let Ok(Some(assoc)) = self.probe_op(
                    call_name.span,
                    MethodCall,
                    Some(call_name),
                    None,
                    IsSuggestion(true),
                    callee_ty.peel_refs(),
                    callee_expr.unwrap().hir_id,
                    TraitsInScope,
                    |mut ctxt| ctxt.probe_for_similar_candidate(),
                )
                && assoc.is_method()
            {
                let args = self.infcx.fresh_args_for_item(call_name.span, assoc.def_id);
                let fn_sig = tcx.fn_sig(assoc.def_id).instantiate(tcx, args);

                self.instantiate_binder_with_fresh_vars(call_name.span, FnCall, fn_sig);
            }
            None
        };

        let suggest_confusable = |err: &mut Diag<'_>| {
            let Some(call_name) = call_ident else {
                return;
            };
            let Some(callee_ty) = callee_ty else {
                return;
            };
            let input_types: Vec<Ty<'_>> = provided_arg_tys.iter().map(|(ty, _)| *ty).collect();
            // Check for other methods in the following order
            //  - methods marked as `rustc_confusables` with the provided arguments
            //  - methods with the same argument type/count and short levenshtein distance
            //  - methods marked as `rustc_confusables` (done)
            //  - methods with short levenshtein distance

            // Look for commonly confusable method names considering arguments.
            if let Some(_name) = self.confusable_method_name(
                err,
                callee_ty.peel_refs(),
                call_name,
                Some(input_types.clone()),
            ) {
                return;
            }
            // Look for method names with short levenshtein distance, considering arguments.
            if let Some((assoc, fn_sig)) = similar_assoc(call_name)
                && fn_sig.inputs()[1..]
                    .iter()
                    .zip(input_types.iter())
                    .all(|(expected, found)| self.may_coerce(*expected, *found))
                && fn_sig.inputs()[1..].len() == input_types.len()
            {
                let assoc_name = assoc.name();
                err.span_suggestion_verbose(
                    call_name.span,
                    format!("you might have meant to use `{}`", assoc_name),
                    assoc_name,
                    Applicability::MaybeIncorrect,
                );
                return;
            }
            // Look for commonly confusable method names disregarding arguments.
            if let Some(_name) =
                self.confusable_method_name(err, callee_ty.peel_refs(), call_name, None)
            {
                return;
            }
            // Look for similarly named methods with levenshtein distance with the right
            // number of arguments.
            if let Some((assoc, fn_sig)) = similar_assoc(call_name)
                && fn_sig.inputs()[1..].len() == input_types.len()
            {
                err.span_note(
                    tcx.def_span(assoc.def_id),
                    format!(
                        "there's is a method with similar name `{}`, but the arguments don't match",
                        assoc.name(),
                    ),
                );
                return;
            }
            // Fallthrough: look for similarly named methods with levenshtein distance.
            if let Some((assoc, _)) = similar_assoc(call_name) {
                err.span_note(
                    tcx.def_span(assoc.def_id),
                    format!(
                        "there's is a method with similar name `{}`, but their argument count \
                         doesn't match",
                        assoc.name(),
                    ),
                );
                return;
            }
        };
        // A "softer" version of the `demand_compatible`, which checks types without persisting them,
        // and treats error types differently
        // This will allow us to "probe" for other argument orders that would likely have been correct
        let check_compatible = |provided_idx: ProvidedIdx, expected_idx: ExpectedIdx| {
            if provided_idx.as_usize() == expected_idx.as_usize() {
                return compatibility_diagonal[provided_idx].clone();
            }

            let (formal_input_ty, expected_input_ty) = formal_and_expected_inputs[expected_idx];
            // If either is an error type, we defy the usual convention and consider them to *not* be
            // coercible. This prevents our error message heuristic from trying to pass errors into
            // every argument.
            if (formal_input_ty, expected_input_ty).references_error() {
                return Compatibility::Incompatible(None);
            }

            let (arg_ty, arg_span) = provided_arg_tys[provided_idx];

            let expectation = Expectation::rvalue_hint(self, expected_input_ty);
            let coerced_ty = expectation.only_has_type(self).unwrap_or(formal_input_ty);
            let can_coerce = self.may_coerce(arg_ty, coerced_ty);
            if !can_coerce {
                return Compatibility::Incompatible(Some(ty::error::TypeError::Sorts(
                    ty::error::ExpectedFound::new(coerced_ty, arg_ty),
                )));
            }

            // Using probe here, since we don't want this subtyping to affect inference.
            let subtyping_error = self.probe(|_| {
                self.at(&self.misc(arg_span), self.param_env)
                    .sup(DefineOpaqueTypes::Yes, formal_input_ty, coerced_ty)
                    .err()
            });

            // Same as above: if either the coerce type or the checked type is an error type,
            // consider them *not* compatible.
            let references_error = (coerced_ty, arg_ty).references_error();
            match (references_error, subtyping_error) {
                (false, None) => Compatibility::Compatible,
                (_, subtyping_error) => Compatibility::Incompatible(subtyping_error),
            }
        };

        let mk_trace = |span, (formal_ty, expected_ty), provided_ty| {
            let mismatched_ty = if expected_ty == provided_ty {
                // If expected == provided, then we must have failed to sup
                // the formal type. Avoid printing out "expected Ty, found Ty"
                // in that case.
                formal_ty
            } else {
                expected_ty
            };
            TypeTrace::types(&self.misc(span), mismatched_ty, provided_ty)
        };

        // The algorithm here is inspired by levenshtein distance and longest common subsequence.
        // We'll try to detect 4 different types of mistakes:
        // - An extra parameter has been provided that doesn't satisfy *any* of the other inputs
        // - An input is missing, which isn't satisfied by *any* of the other arguments
        // - Some number of arguments have been provided in the wrong order
        // - A type is straight up invalid

        // First, let's find the errors
        let (mut errors, matched_inputs) =
            ArgMatrix::new(provided_args.len(), formal_and_expected_inputs.len(), check_compatible)
                .find_errors();

        // First, check if we just need to wrap some arguments in a tuple.
        if let Some((mismatch_idx, terr)) =
            compatibility_diagonal.iter_enumerated().find_map(|(i, c)| {
                if let Compatibility::Incompatible(Some(terr)) = c {
                    Some((i, *terr))
                } else {
                    None
                }
            })
        {
            // Is the first bad expected argument a tuple?
            // Do we have as many extra provided arguments as the tuple's length?
            // If so, we might have just forgotten to wrap some args in a tuple.
            if let Some(ty::Tuple(tys)) =
                formal_and_expected_inputs.get(mismatch_idx.to_expected_idx()).map(|tys| tys.1.kind())
                // If the tuple is unit, we're not actually wrapping any arguments.
                && !tys.is_empty()
                && provided_arg_tys.len() == formal_and_expected_inputs.len() - 1 + tys.len()
            {
                // Wrap up the N provided arguments starting at this position in a tuple.
                let provided_args_to_tuple = &provided_arg_tys[mismatch_idx..];
                let (provided_args_to_tuple, provided_args_after_tuple) =
                    provided_args_to_tuple.split_at(tys.len());
                let provided_as_tuple =
                    Ty::new_tup_from_iter(tcx, provided_args_to_tuple.iter().map(|&(ty, _)| ty));

                let mut satisfied = true;
                // Check if the newly wrapped tuple + rest of the arguments are compatible.
                for ((_, expected_ty), provided_ty) in std::iter::zip(
                    formal_and_expected_inputs[mismatch_idx.to_expected_idx()..].iter(),
                    [provided_as_tuple]
                        .into_iter()
                        .chain(provided_args_after_tuple.iter().map(|&(ty, _)| ty)),
                ) {
                    if !self.may_coerce(provided_ty, *expected_ty) {
                        satisfied = false;
                        break;
                    }
                }

                // If they're compatible, suggest wrapping in an arg, and we're done!
                // Take some care with spans, so we don't suggest wrapping a macro's
                // innards in parenthesis, for example.
                if satisfied
                    && let &[(_, hi @ lo)] | &[(_, lo), .., (_, hi)] = provided_args_to_tuple
                {
                    let mut err;
                    if tys.len() == 1 {
                        // A tuple wrap suggestion actually occurs within,
                        // so don't do anything special here.
                        err = self.err_ctxt().report_and_explain_type_error(
                            mk_trace(
                                lo,
                                formal_and_expected_inputs[mismatch_idx.to_expected_idx()],
                                provided_arg_tys[mismatch_idx].0,
                            ),
                            self.param_env,
                            terr,
                        );
                        err.span_label(
                            full_call_span,
                            format!("arguments to this {call_name} are incorrect"),
                        );
                    } else {
                        err = self.dcx().struct_span_err(
                            full_call_span,
                            format!(
                                "{call_name} takes {}{} but {} {} supplied",
                                if c_variadic { "at least " } else { "" },
                                potentially_plural_count(
                                    formal_and_expected_inputs.len(),
                                    "argument"
                                ),
                                potentially_plural_count(provided_args.len(), "argument"),
                                pluralize!("was", provided_args.len())
                            ),
                        );
                        err.code(err_code.to_owned());
                        err.multipart_suggestion_verbose(
                            "wrap these arguments in parentheses to construct a tuple",
                            vec![
                                (lo.shrink_to_lo(), "(".to_string()),
                                (hi.shrink_to_hi(), ")".to_string()),
                            ],
                            Applicability::MachineApplicable,
                        );
                    };
                    self.label_fn_like(
                        &mut err,
                        fn_def_id,
                        callee_ty,
                        call_expr,
                        None,
                        Some(mismatch_idx.as_usize()),
                        &matched_inputs,
                        &formal_and_expected_inputs,
                        is_method,
                        tuple_arguments,
                    );
                    suggest_confusable(&mut err);
                    return err.emit();
                }
            }
        }

        // Okay, so here's where it gets complicated in regards to what errors
        // we emit and how.
        // There are 3 different "types" of errors we might encounter.
        //   1) Missing/extra/swapped arguments
        //   2) Valid but incorrect arguments
        //   3) Invalid arguments
        //      - Currently I think this only comes up with `CyclicTy`
        //
        // We first need to go through, remove those from (3) and emit those
        // as their own error, particularly since they're error code and
        // message is special. From what I can tell, we *must* emit these
        // here (vs somewhere prior to this function) since the arguments
        // become invalid *because* of how they get used in the function.
        // It is what it is.

        if errors.is_empty() {
            if cfg!(debug_assertions) {
                span_bug!(error_span, "expected errors from argument matrix");
            } else {
                let mut err =
                    self.dcx().create_err(errors::ArgMismatchIndeterminate { span: error_span });
                suggest_confusable(&mut err);
                return err.emit();
            }
        }

        let detect_dotdot = |err: &mut Diag<'_>, ty: Ty<'_>, expr: &hir::Expr<'_>| {
            if let ty::Adt(adt, _) = ty.kind()
                && self.tcx().is_lang_item(adt.did(), hir::LangItem::RangeFull)
                && let hir::ExprKind::Struct(
                    hir::QPath::LangItem(hir::LangItem::RangeFull, _),
                    [],
                    _,
                ) = expr.kind
            {
                // We have `Foo(a, .., c)`, where the user might be trying to use the "rest" syntax
                // from default field values, which is not supported on tuples.
                let explanation = if self.tcx.features().default_field_values() {
                    "this is only supported on non-tuple struct literals"
                } else if self.tcx.sess.is_nightly_build() {
                    "this is only supported on non-tuple struct literals when \
                     `#![feature(default_field_values)]` is enabled"
                } else {
                    "this is not supported"
                };
                let msg = format!(
                    "you might have meant to use `..` to skip providing a value for \
                     expected fields, but {explanation}; it is instead interpreted as a \
                     `std::ops::RangeFull` literal",
                );
                err.span_help(expr.span, msg);
            }
        };

        let mut reported = None;
        errors.retain(|error| {
            let Error::Invalid(provided_idx, expected_idx, Compatibility::Incompatible(Some(e))) =
                error
            else {
                return true;
            };
            let (provided_ty, provided_span) = provided_arg_tys[*provided_idx];
            let trace =
                mk_trace(provided_span, formal_and_expected_inputs[*expected_idx], provided_ty);
            if !matches!(trace.cause.as_failure_code(*e), FailureCode::Error0308) {
                let mut err =
                    self.err_ctxt().report_and_explain_type_error(trace, self.param_env, *e);
                suggest_confusable(&mut err);
                reported = Some(err.emit());
                return false;
            }
            true
        });

        // We're done if we found errors, but we already emitted them.
        if let Some(reported) = reported
            && errors.is_empty()
        {
            return reported;
        }
        assert!(!errors.is_empty());

        // Okay, now that we've emitted the special errors separately, we
        // are only left missing/extra/swapped and mismatched arguments, both
        // can be collated pretty easily if needed.

        // Next special case: if there is only one "Incompatible" error, just emit that
        if let &[
            Error::Invalid(provided_idx, expected_idx, Compatibility::Incompatible(Some(err))),
        ] = &errors[..]
        {
            let (formal_ty, expected_ty) = formal_and_expected_inputs[expected_idx];
            let (provided_ty, provided_arg_span) = provided_arg_tys[provided_idx];
            let trace = mk_trace(provided_arg_span, (formal_ty, expected_ty), provided_ty);
            let mut err = self.err_ctxt().report_and_explain_type_error(trace, self.param_env, err);
            self.emit_coerce_suggestions(
                &mut err,
                provided_args[provided_idx],
                provided_ty,
                Expectation::rvalue_hint(self, expected_ty)
                    .only_has_type(self)
                    .unwrap_or(formal_ty),
                None,
                None,
            );
            err.span_label(full_call_span, format!("arguments to this {call_name} are incorrect"));

            self.label_generic_mismatches(
                &mut err,
                fn_def_id,
                &matched_inputs,
                &provided_arg_tys,
                &formal_and_expected_inputs,
                is_method,
            );

            if let hir::ExprKind::MethodCall(_, rcvr, _, _) = call_expr.kind
                && provided_idx.as_usize() == expected_idx.as_usize()
            {
                self.note_source_of_type_mismatch_constraint(
                    &mut err,
                    rcvr,
                    crate::demand::TypeMismatchSource::Arg {
                        call_expr,
                        incompatible_arg: provided_idx.as_usize(),
                    },
                );
            }

            self.suggest_ptr_null_mut(
                expected_ty,
                provided_ty,
                provided_args[provided_idx],
                &mut err,
            );

            self.suggest_deref_unwrap_or(
                &mut err,
                callee_ty,
                call_ident,
                expected_ty,
                provided_ty,
                provided_args[provided_idx],
                is_method,
            );

            // Call out where the function is defined
            self.label_fn_like(
                &mut err,
                fn_def_id,
                callee_ty,
                call_expr,
                Some(expected_ty),
                Some(expected_idx.as_usize()),
                &matched_inputs,
                &formal_and_expected_inputs,
                is_method,
                tuple_arguments,
            );
            suggest_confusable(&mut err);
            detect_dotdot(&mut err, provided_ty, provided_args[provided_idx]);
            return err.emit();
        }

        // Special case, we found an extra argument is provided, which is very common in practice.
        // but there is a obviously better removing suggestion compared to the current one,
        // try to find the argument with Error type, if we removed it all the types will become good,
        // then we will replace the current suggestion.
        if let [Error::Extra(provided_idx)] = &errors[..] {
            let remove_idx_is_perfect = |idx: usize| -> bool {
                let removed_arg_tys = provided_arg_tys
                    .iter()
                    .enumerate()
                    .filter_map(|(j, arg)| if idx == j { None } else { Some(arg) })
                    .collect::<IndexVec<ProvidedIdx, _>>();
                std::iter::zip(formal_and_expected_inputs.iter(), removed_arg_tys.iter()).all(
                    |((expected_ty, _), (provided_ty, _))| {
                        !provided_ty.references_error()
                            && self.may_coerce(*provided_ty, *expected_ty)
                    },
                )
            };

            if !remove_idx_is_perfect(provided_idx.as_usize()) {
                if let Some(i) = (0..provided_args.len()).find(|&i| remove_idx_is_perfect(i)) {
                    errors = vec![Error::Extra(ProvidedIdx::from_usize(i))];
                }
            }
        }

        let mut err = if formal_and_expected_inputs.len() == provided_args.len() {
            struct_span_code_err!(
                self.dcx(),
                full_call_span,
                E0308,
                "arguments to this {} are incorrect",
                call_name,
            )
        } else {
            self.dcx()
                .struct_span_err(
                    full_call_span,
                    format!(
                        "this {} takes {}{} but {} {} supplied",
                        call_name,
                        if c_variadic { "at least " } else { "" },
                        potentially_plural_count(formal_and_expected_inputs.len(), "argument"),
                        potentially_plural_count(provided_args.len(), "argument"),
                        pluralize!("was", provided_args.len())
                    ),
                )
                .with_code(err_code.to_owned())
        };

        suggest_confusable(&mut err);
        // As we encounter issues, keep track of what we want to provide for the suggestion
        let mut labels = vec![];
        // If there is a single error, we give a specific suggestion; otherwise, we change to
        // "did you mean" with the suggested function call
        enum SuggestionText {
            None,
            Provide(bool),
            Remove(bool),
            Swap,
            Reorder,
            DidYouMean,
        }
        let mut suggestion_text = SuggestionText::None;

        let ty_to_snippet = |ty: Ty<'tcx>, expected_idx: ExpectedIdx| {
            if ty.is_unit() {
                "()".to_string()
            } else if ty.is_suggestable(tcx, false) {
                format!("/* {ty} */")
            } else if let Some(fn_def_id) = fn_def_id
                && self.tcx.def_kind(fn_def_id).is_fn_like()
                && let self_implicit =
                    matches!(call_expr.kind, hir::ExprKind::MethodCall(..)) as usize
                && let Some(Some(arg)) =
                    self.tcx.fn_arg_idents(fn_def_id).get(expected_idx.as_usize() + self_implicit)
                && arg.name != kw::SelfLower
            {
                format!("/* {} */", arg.name)
            } else {
                "/* value */".to_string()
            }
        };

        let mut errors = errors.into_iter().peekable();
        let mut only_extras_so_far = errors
            .peek()
            .is_some_and(|first| matches!(first, Error::Extra(arg_idx) if arg_idx.index() == 0));
        let mut prev_extra_idx = None;
        let mut suggestions = vec![];
        while let Some(error) = errors.next() {
            only_extras_so_far &= matches!(error, Error::Extra(_));

            match error {
                Error::Invalid(provided_idx, expected_idx, compatibility) => {
                    let (formal_ty, expected_ty) = formal_and_expected_inputs[expected_idx];
                    let (provided_ty, provided_span) = provided_arg_tys[provided_idx];
                    if let Compatibility::Incompatible(error) = compatibility {
                        let trace = mk_trace(provided_span, (formal_ty, expected_ty), provided_ty);
                        if let Some(e) = error {
                            self.err_ctxt().note_type_err(
                                &mut err,
                                &trace.cause,
                                None,
                                Some(self.param_env.and(trace.values)),
                                e,
                                true,
                                None,
                            );
                        }
                    }

                    self.emit_coerce_suggestions(
                        &mut err,
                        provided_args[provided_idx],
                        provided_ty,
                        Expectation::rvalue_hint(self, expected_ty)
                            .only_has_type(self)
                            .unwrap_or(formal_ty),
                        None,
                        None,
                    );
                    detect_dotdot(&mut err, provided_ty, provided_args[provided_idx]);
                }
                Error::Extra(arg_idx) => {
                    let (provided_ty, provided_span) = provided_arg_tys[arg_idx];
                    let provided_ty_name = if !has_error_or_infer([provided_ty]) {
                        // FIXME: not suggestable, use something else
                        format!(" of type `{provided_ty}`")
                    } else {
                        "".to_string()
                    };
                    let idx = if provided_arg_tys.len() == 1 {
                        "".to_string()
                    } else {
                        format!(" #{}", arg_idx.as_usize() + 1)
                    };
                    labels.push((
                        provided_span,
                        format!("unexpected argument{idx}{provided_ty_name}"),
                    ));
                    let mut span = provided_span;
                    if span.can_be_used_for_suggestions()
                        && error_span.can_be_used_for_suggestions()
                    {
                        if arg_idx.index() > 0
                            && let Some((_, prev)) =
                                provided_arg_tys.get(ProvidedIdx::from_usize(arg_idx.index() - 1))
                        {
                            // Include previous comma
                            span = prev.shrink_to_hi().to(span);
                        }

                        // Is last argument for deletion in a row starting from the 0-th argument?
                        // Then delete the next comma, so we are not left with `f(, ...)`
                        //
                        //     fn f() {}
                        //   - f(0, 1,)
                        //   + f()
                        let trim_next_comma = match errors.peek() {
                            Some(Error::Extra(provided_idx))
                                if only_extras_so_far
                                    && provided_idx.index() > arg_idx.index() + 1 =>
                            // If the next Error::Extra ("next") doesn't next to current ("current"),
                            // fn foo(_: (), _: u32) {}
                            // - foo("current", (), 1u32, "next")
                            // + foo((), 1u32)
                            // If the previous error is not a `Error::Extra`, then do not trim the next comma
                            // - foo((), "current", 42u32, "next")
                            // + foo((), 42u32)
                            {
                                prev_extra_idx.is_none_or(|prev_extra_idx| {
                                    prev_extra_idx + 1 == arg_idx.index()
                                })
                            }
                            // If no error left, we need to delete the next comma
                            None if only_extras_so_far => true,
                            // Not sure if other error type need to be handled as well
                            _ => false,
                        };

                        if trim_next_comma {
                            let next = provided_arg_tys
                                .get(arg_idx + 1)
                                .map(|&(_, sp)| sp)
                                .unwrap_or_else(|| {
                                    // Try to move before `)`. Note that `)` here is not necessarily
                                    // the latin right paren, it could be a Unicode-confusable that
                                    // looks like a `)`, so we must not use `- BytePos(1)`
                                    // manipulations here.
                                    self.tcx().sess.source_map().end_point(call_expr.span)
                                });

                            // Include next comma
                            span = span.until(next);
                        }

                        suggestions.push((span, String::new()));

                        suggestion_text = match suggestion_text {
                            SuggestionText::None => SuggestionText::Remove(false),
                            SuggestionText::Remove(_) => SuggestionText::Remove(true),
                            _ => SuggestionText::DidYouMean,
                        };
                        prev_extra_idx = Some(arg_idx.index())
                    }
                    detect_dotdot(&mut err, provided_ty, provided_args[arg_idx]);
                }
                Error::Missing(expected_idx) => {
                    // If there are multiple missing arguments adjacent to each other,
                    // then we can provide a single error.

                    let mut missing_idxs = vec![expected_idx];
                    while let Some(e) = errors.next_if(|e| {
                        matches!(e, Error::Missing(next_expected_idx)
                            if *next_expected_idx == *missing_idxs.last().unwrap() + 1)
                    }) {
                        match e {
                            Error::Missing(expected_idx) => missing_idxs.push(expected_idx),
                            _ => unreachable!(
                                "control flow ensures that we should always get an `Error::Missing`"
                            ),
                        }
                    }

                    // NOTE: Because we might be re-arranging arguments, might have extra
                    // arguments, etc. it's hard to *really* know where we should provide
                    // this error label, so as a heuristic, we point to the provided arg, or
                    // to the call if the missing inputs pass the provided args.
                    match &missing_idxs[..] {
                        &[expected_idx] => {
                            let (_, input_ty) = formal_and_expected_inputs[expected_idx];
                            let span = if let Some((_, arg_span)) =
                                provided_arg_tys.get(expected_idx.to_provided_idx())
                            {
                                *arg_span
                            } else {
                                args_span
                            };
                            let rendered = if !has_error_or_infer([input_ty]) {
                                format!(" of type `{input_ty}`")
                            } else {
                                "".to_string()
                            };
                            labels.push((
                                span,
                                format!(
                                    "argument #{}{rendered} is missing",
                                    expected_idx.as_usize() + 1
                                ),
                            ));

                            suggestion_text = match suggestion_text {
                                SuggestionText::None => SuggestionText::Provide(false),
                                SuggestionText::Provide(_) => SuggestionText::Provide(true),
                                _ => SuggestionText::DidYouMean,
                            };
                        }
                        &[first_idx, second_idx] => {
                            let (_, first_expected_ty) = formal_and_expected_inputs[first_idx];
                            let (_, second_expected_ty) = formal_and_expected_inputs[second_idx];
                            let span = if let (Some((_, first_span)), Some((_, second_span))) = (
                                provided_arg_tys.get(first_idx.to_provided_idx()),
                                provided_arg_tys.get(second_idx.to_provided_idx()),
                            ) {
                                first_span.to(*second_span)
                            } else {
                                args_span
                            };
                            let rendered =
                                if !has_error_or_infer([first_expected_ty, second_expected_ty]) {
                                    format!(
                                        " of type `{first_expected_ty}` and `{second_expected_ty}`"
                                    )
                                } else {
                                    "".to_string()
                                };
                            labels.push((span, format!("two arguments{rendered} are missing")));
                            suggestion_text = match suggestion_text {
                                SuggestionText::None | SuggestionText::Provide(_) => {
                                    SuggestionText::Provide(true)
                                }
                                _ => SuggestionText::DidYouMean,
                            };
                        }
                        &[first_idx, second_idx, third_idx] => {
                            let (_, first_expected_ty) = formal_and_expected_inputs[first_idx];
                            let (_, second_expected_ty) = formal_and_expected_inputs[second_idx];
                            let (_, third_expected_ty) = formal_and_expected_inputs[third_idx];
                            let span = if let (Some((_, first_span)), Some((_, third_span))) = (
                                provided_arg_tys.get(first_idx.to_provided_idx()),
                                provided_arg_tys.get(third_idx.to_provided_idx()),
                            ) {
                                first_span.to(*third_span)
                            } else {
                                args_span
                            };
                            let rendered = if !has_error_or_infer([
                                first_expected_ty,
                                second_expected_ty,
                                third_expected_ty,
                            ]) {
                                format!(
                                    " of type `{first_expected_ty}`, `{second_expected_ty}`, and `{third_expected_ty}`"
                                )
                            } else {
                                "".to_string()
                            };
                            labels.push((span, format!("three arguments{rendered} are missing")));
                            suggestion_text = match suggestion_text {
                                SuggestionText::None | SuggestionText::Provide(_) => {
                                    SuggestionText::Provide(true)
                                }
                                _ => SuggestionText::DidYouMean,
                            };
                        }
                        missing_idxs => {
                            let first_idx = *missing_idxs.first().unwrap();
                            let last_idx = *missing_idxs.last().unwrap();
                            // NOTE: Because we might be re-arranging arguments, might have extra arguments, etc.
                            // It's hard to *really* know where we should provide this error label, so this is a
                            // decent heuristic
                            let span = if let (Some((_, first_span)), Some((_, last_span))) = (
                                provided_arg_tys.get(first_idx.to_provided_idx()),
                                provided_arg_tys.get(last_idx.to_provided_idx()),
                            ) {
                                first_span.to(*last_span)
                            } else {
                                args_span
                            };
                            labels.push((span, "multiple arguments are missing".to_string()));
                            suggestion_text = match suggestion_text {
                                SuggestionText::None | SuggestionText::Provide(_) => {
                                    SuggestionText::Provide(true)
                                }
                                _ => SuggestionText::DidYouMean,
                            };
                        }
                    }
                }
                Error::Swap(
                    first_provided_idx,
                    second_provided_idx,
                    first_expected_idx,
                    second_expected_idx,
                ) => {
                    let (first_provided_ty, first_span) = provided_arg_tys[first_provided_idx];
                    let (_, first_expected_ty) = formal_and_expected_inputs[first_expected_idx];
                    let first_provided_ty_name = if !has_error_or_infer([first_provided_ty]) {
                        format!(", found `{first_provided_ty}`")
                    } else {
                        String::new()
                    };
                    labels.push((
                        first_span,
                        format!("expected `{first_expected_ty}`{first_provided_ty_name}"),
                    ));

                    let (second_provided_ty, second_span) = provided_arg_tys[second_provided_idx];
                    let (_, second_expected_ty) = formal_and_expected_inputs[second_expected_idx];
                    let second_provided_ty_name = if !has_error_or_infer([second_provided_ty]) {
                        format!(", found `{second_provided_ty}`")
                    } else {
                        String::new()
                    };
                    labels.push((
                        second_span,
                        format!("expected `{second_expected_ty}`{second_provided_ty_name}"),
                    ));

                    suggestion_text = match suggestion_text {
                        SuggestionText::None => SuggestionText::Swap,
                        _ => SuggestionText::DidYouMean,
                    };
                }
                Error::Permutation(args) => {
                    for (dst_arg, dest_input) in args {
                        let (_, expected_ty) = formal_and_expected_inputs[dst_arg];
                        let (provided_ty, provided_span) = provided_arg_tys[dest_input];
                        let provided_ty_name = if !has_error_or_infer([provided_ty]) {
                            format!(", found `{provided_ty}`")
                        } else {
                            String::new()
                        };
                        labels.push((
                            provided_span,
                            format!("expected `{expected_ty}`{provided_ty_name}"),
                        ));
                    }

                    suggestion_text = match suggestion_text {
                        SuggestionText::None => SuggestionText::Reorder,
                        _ => SuggestionText::DidYouMean,
                    };
                }
            }
        }

        self.label_generic_mismatches(
            &mut err,
            fn_def_id,
            &matched_inputs,
            &provided_arg_tys,
            &formal_and_expected_inputs,
            is_method,
        );

        // Incorporate the argument changes in the removal suggestion.
        // When a type is *missing*, and the rest are additional, we want to suggest these with a
        // multipart suggestion, but in order to do so we need to figure out *where* the arg that
        // was provided but had the wrong type should go, because when looking at `expected_idx`
        // that is the position in the argument list in the definition, while `provided_idx` will
        // not be present. So we have to look at what the *last* provided position was, and point
        // one after to suggest the replacement. FIXME(estebank): This is hacky, and there's
        // probably a better more involved change we can make to make this work.
        // For example, if we have
        // ```
        // fn foo(i32, &'static str) {}
        // foo((), (), ());
        // ```
        // what should be suggested is
        // ```
        // foo(/* i32 */, /* &str */);
        // ```
        // which includes the replacement of the first two `()` for the correct type, and the
        // removal of the last `()`.
        let mut prev = -1;
        for (expected_idx, provided_idx) in matched_inputs.iter_enumerated() {
            // We want to point not at the *current* argument expression index, but rather at the
            // index position where it *should have been*, which is *after* the previous one.
            if let Some(provided_idx) = provided_idx {
                prev = provided_idx.index() as i64;
                continue;
            }
            let idx = ProvidedIdx::from_usize((prev + 1) as usize);
            if let Some((_, arg_span)) = provided_arg_tys.get(idx) {
                prev += 1;
                // There is a type that was *not* found anywhere, so it isn't a move, but a
                // replacement and we look at what type it should have been. This will allow us
                // To suggest a multipart suggestion when encountering `foo(1, "")` where the def
                // was `fn foo(())`.
                let (_, expected_ty) = formal_and_expected_inputs[expected_idx];
                suggestions.push((*arg_span, ty_to_snippet(expected_ty, expected_idx)));
            }
        }

        // If we have less than 5 things to say, it would be useful to call out exactly what's wrong
        if labels.len() <= 5 {
            for (span, label) in labels {
                err.span_label(span, label);
            }
        }

        // Call out where the function is defined
        self.label_fn_like(
            &mut err,
            fn_def_id,
            callee_ty,
            call_expr,
            None,
            None,
            &matched_inputs,
            &formal_and_expected_inputs,
            is_method,
            tuple_arguments,
        );

        // And add a suggestion block for all of the parameters
        let suggestion_text = match suggestion_text {
            SuggestionText::None => None,
            SuggestionText::Provide(plural) => {
                Some(format!("provide the argument{}", if plural { "s" } else { "" }))
            }
            SuggestionText::Remove(plural) => {
                err.multipart_suggestion_verbose(
                    format!("remove the extra argument{}", if plural { "s" } else { "" }),
                    suggestions,
                    Applicability::HasPlaceholders,
                );
                None
            }
            SuggestionText::Swap => Some("swap these arguments".to_string()),
            SuggestionText::Reorder => Some("reorder these arguments".to_string()),
            SuggestionText::DidYouMean => Some("did you mean".to_string()),
        };
        if let Some(suggestion_text) = suggestion_text
            && !full_call_span.in_external_macro(self.sess().source_map())
        {
            let source_map = self.sess().source_map();
            let suggestion_span = if let Some(args_span) = error_span.trim_start(full_call_span) {
                // Span of the braces, e.g. `(a, b, c)`.
                args_span
            } else {
                // The arg span of a function call that wasn't even given braces
                // like what might happen with delegation reuse.
                // e.g. `reuse HasSelf::method;` should suggest `reuse HasSelf::method($args);`.
                full_call_span.shrink_to_hi()
            };
            let mut suggestion = "(".to_owned();
            let mut needs_comma = false;
            for (expected_idx, provided_idx) in matched_inputs.iter_enumerated() {
                if needs_comma {
                    suggestion += ", ";
                } else {
                    needs_comma = true;
                }
                let suggestion_text = if let Some(provided_idx) = provided_idx
                    && let (_, provided_span) = provided_arg_tys[*provided_idx]
                    && let Ok(arg_text) = source_map.span_to_snippet(provided_span)
                {
                    arg_text
                } else {
                    // Propose a placeholder of the correct type
                    let (_, expected_ty) = formal_and_expected_inputs[expected_idx];
                    ty_to_snippet(expected_ty, expected_idx)
                };
                suggestion += &suggestion_text;
            }
            suggestion += ")";
            err.span_suggestion_verbose(
                suggestion_span,
                suggestion_text,
                suggestion,
                Applicability::HasPlaceholders,
            );
        }

        err.emit()
    }

    fn suggest_ptr_null_mut(
        &self,
        expected_ty: Ty<'tcx>,
        provided_ty: Ty<'tcx>,
        arg: &hir::Expr<'tcx>,
        err: &mut Diag<'_>,
    ) {
        if let ty::RawPtr(_, hir::Mutability::Mut) = expected_ty.kind()
            && let ty::RawPtr(_, hir::Mutability::Not) = provided_ty.kind()
            && let hir::ExprKind::Call(callee, _) = arg.kind
            && let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = callee.kind
            && let Res::Def(_, def_id) = path.res
            && self.tcx.get_diagnostic_item(sym::ptr_null) == Some(def_id)
        {
            // The user provided `ptr::null()`, but the function expects
            // `ptr::null_mut()`.
            err.subdiagnostic(SuggestPtrNullMut { span: arg.span });
        }
    }

    // AST fragment checking
    pub(in super::super) fn check_expr_lit(
        &self,
        lit: &hir::Lit,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        match lit.node {
            ast::LitKind::Str(..) => Ty::new_static_str(tcx),
            ast::LitKind::ByteStr(ref v, _) => Ty::new_imm_ref(
                tcx,
                tcx.lifetimes.re_static,
                Ty::new_array(tcx, tcx.types.u8, v.len() as u64),
            ),
            ast::LitKind::Byte(_) => tcx.types.u8,
            ast::LitKind::Char(_) => tcx.types.char,
            ast::LitKind::Int(_, ast::LitIntType::Signed(t)) => Ty::new_int(tcx, ty::int_ty(t)),
            ast::LitKind::Int(_, ast::LitIntType::Unsigned(t)) => Ty::new_uint(tcx, ty::uint_ty(t)),
            ast::LitKind::Int(i, ast::LitIntType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| match ty.kind() {
                    ty::Int(_) | ty::Uint(_) => Some(ty),
                    // These exist to direct casts like `0x61 as char` to use
                    // the right integer type to cast from, instead of falling back to
                    // i32 due to no further constraints.
                    ty::Char => Some(tcx.types.u8),
                    ty::RawPtr(..) => Some(tcx.types.usize),
                    ty::FnDef(..) | ty::FnPtr(..) => Some(tcx.types.usize),
                    &ty::Pat(base, _) if base.is_integral() => {
                        let layout = tcx
                            .layout_of(self.typing_env(self.param_env).as_query_input(ty))
                            .ok()?;
                        assert!(!layout.uninhabited);

                        match layout.backend_repr {
                            rustc_abi::BackendRepr::Scalar(scalar) => {
                                scalar.valid_range(&tcx).contains(u128::from(i.get())).then_some(ty)
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => None,
                });
                opt_ty.unwrap_or_else(|| self.next_int_var())
            }
            ast::LitKind::Float(_, ast::LitFloatType::Suffixed(t)) => {
                Ty::new_float(tcx, ty::float_ty(t))
            }
            ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| match ty.kind() {
                    ty::Float(_) => Some(ty),
                    _ => None,
                });
                opt_ty.unwrap_or_else(|| self.next_float_var())
            }
            ast::LitKind::Bool(_) => tcx.types.bool,
            ast::LitKind::CStr(_, _) => Ty::new_imm_ref(
                tcx,
                tcx.lifetimes.re_static,
                tcx.type_of(tcx.require_lang_item(hir::LangItem::CStr, lit.span)).skip_binder(),
            ),
            ast::LitKind::Err(guar) => Ty::new_error(tcx, guar),
        }
    }

    pub(crate) fn check_struct_path(
        &self,
        qpath: &QPath<'tcx>,
        hir_id: HirId,
    ) -> Result<(&'tcx ty::VariantDef, Ty<'tcx>), ErrorGuaranteed> {
        let path_span = qpath.span();
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, hir_id);
        let variant = match def {
            Res::Err => {
                let guar =
                    self.dcx().span_delayed_bug(path_span, "`Res::Err` but no error emitted");
                self.set_tainted_by_errors(guar);
                return Err(guar);
            }
            Res::Def(DefKind::Variant, _) => match ty.normalized.ty_adt_def() {
                Some(adt) => {
                    Some((adt.variant_of_res(def), adt.did(), Self::user_args_for_adt(ty)))
                }
                _ => bug!("unexpected type: {:?}", ty.normalized),
            },
            Res::Def(
                DefKind::Struct | DefKind::Union | DefKind::TyAlias { .. } | DefKind::AssocTy,
                _,
            )
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. } => match ty.normalized.ty_adt_def() {
                Some(adt) if !adt.is_enum() => {
                    Some((adt.non_enum_variant(), adt.did(), Self::user_args_for_adt(ty)))
                }
                _ => None,
            },
            _ => bug!("unexpected definition: {:?}", def),
        };

        if let Some((variant, did, ty::UserArgs { args, user_self_ty })) = variant {
            debug!("check_struct_path: did={:?} args={:?}", did, args);

            // Register type annotation.
            self.write_user_type_annotation_from_args(hir_id, did, args, user_self_ty);

            // Check bounds on type arguments used in the path.
            self.add_required_obligations_for_hir(path_span, did, args, hir_id);

            Ok((variant, ty.normalized))
        } else {
            Err(match *ty.normalized.kind() {
                ty::Error(guar) => {
                    // E0071 might be caused by a spelling error, which will have
                    // already caused an error message and probably a suggestion
                    // elsewhere. Refrain from emitting more unhelpful errors here
                    // (issue #88844).
                    guar
                }
                _ => struct_span_code_err!(
                    self.dcx(),
                    path_span,
                    E0071,
                    "expected struct, variant or union type, found {}",
                    ty.normalized.sort_string(self.tcx)
                )
                .with_span_label(path_span, "not a struct")
                .emit(),
            })
        }
    }

    fn check_decl_initializer(
        &self,
        hir_id: HirId,
        pat: &'tcx hir::Pat<'tcx>,
        init: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // FIXME(tschottdorf): `contains_explicit_ref_binding()` must be removed
        // for #42640 (default match binding modes).
        //
        // See #44848.
        let ref_bindings = pat.contains_explicit_ref_binding();

        let local_ty = self.local_ty(init.span, hir_id);
        if let Some(m) = ref_bindings {
            // Somewhat subtle: if we have a `ref` binding in the pattern,
            // we want to avoid introducing coercions for the RHS. This is
            // both because it helps preserve sanity and, in the case of
            // ref mut, for soundness (issue #23116). In particular, in
            // the latter case, we need to be clear that the type of the
            // referent for the reference that results is *equal to* the
            // type of the place it is referencing, and not some
            // supertype thereof.
            let init_ty = self.check_expr_with_needs(init, Needs::maybe_mut_place(m));
            if let Err(mut diag) = self.demand_eqtype_diag(init.span, local_ty, init_ty) {
                self.emit_type_mismatch_suggestions(
                    &mut diag,
                    init.peel_drop_temps(),
                    init_ty,
                    local_ty,
                    None,
                    None,
                );
                diag.emit();
            }
            init_ty
        } else {
            self.check_expr_coercible_to_type(init, local_ty, None)
        }
    }

    pub(in super::super) fn check_decl(&self, decl: Declaration<'tcx>) -> Ty<'tcx> {
        // Determine and write the type which we'll check the pattern against.
        let decl_ty = self.local_ty(decl.span, decl.hir_id);

        // Type check the initializer.
        if let Some(ref init) = decl.init {
            let init_ty = self.check_decl_initializer(decl.hir_id, decl.pat, init);
            self.overwrite_local_ty_if_err(decl.hir_id, decl.pat, init_ty);
        }

        // Does the expected pattern type originate from an expression and what is the span?
        let (origin_expr, ty_span) = match (decl.ty, decl.init) {
            (Some(ty), _) => (None, Some(ty.span)), // Bias towards the explicit user type.
            (_, Some(init)) => {
                (Some(init), Some(init.span.find_ancestor_inside(decl.span).unwrap_or(init.span)))
            } // No explicit type; so use the scrutinee.
            _ => (None, None), // We have `let $pat;`, so the expected type is unconstrained.
        };

        // Type check the pattern. Override if necessary to avoid knock-on errors.
        self.check_pat_top(decl.pat, decl_ty, ty_span, origin_expr, Some(decl.origin));
        let pat_ty = self.node_ty(decl.pat.hir_id);
        self.overwrite_local_ty_if_err(decl.hir_id, decl.pat, pat_ty);

        if let Some(blk) = decl.origin.try_get_else() {
            let previous_diverges = self.diverges.get();
            let else_ty = self.check_expr_block(blk, NoExpectation);
            let cause = self.cause(blk.span, ObligationCauseCode::LetElse);
            if let Err(err) = self.demand_eqtype_with_origin(&cause, self.tcx.types.never, else_ty)
            {
                err.emit();
            }
            self.diverges.set(previous_diverges);
        }
        decl_ty
    }

    /// Type check a `let` statement.
    fn check_decl_local(&self, local: &'tcx hir::LetStmt<'tcx>) {
        GatherLocalsVisitor::gather_from_local(self, local);

        let ty = self.check_decl(local.into());
        self.write_ty(local.hir_id, ty);
        if local.pat.is_never_pattern() {
            self.diverges.set(Diverges::Always {
                span: local.pat.span,
                custom_note: Some("any code following a never pattern is unreachable"),
            });
        }
    }

    fn check_stmt(&self, stmt: &'tcx hir::Stmt<'tcx>) {
        // Don't do all the complex logic below for `DeclItem`.
        match stmt.kind {
            hir::StmtKind::Item(..) => return,
            hir::StmtKind::Let(..) | hir::StmtKind::Expr(..) | hir::StmtKind::Semi(..) => {}
        }

        self.warn_if_unreachable(stmt.hir_id, stmt.span, "statement");

        // Hide the outer diverging flags.
        let old_diverges = self.diverges.replace(Diverges::Maybe);

        match stmt.kind {
            hir::StmtKind::Let(l) => {
                self.check_decl_local(l);
            }
            // Ignore for now.
            hir::StmtKind::Item(_) => {}
            hir::StmtKind::Expr(ref expr) => {
                // Check with expected type of `()`.
                self.check_expr_has_type_or_error(expr, self.tcx.types.unit, |err| {
                    if expr.can_have_side_effects() {
                        self.suggest_semicolon_at_end(expr.span, err);
                    }
                });
            }
            hir::StmtKind::Semi(expr) => {
                self.check_expr(expr);
            }
        }

        // Combine the diverging and `has_error` flags.
        self.diverges.set(self.diverges.get() | old_diverges);
    }

    pub(crate) fn check_block_no_value(&self, blk: &'tcx hir::Block<'tcx>) {
        let unit = self.tcx.types.unit;
        let ty = self.check_expr_block(blk, ExpectHasType(unit));

        // if the block produces a `!` value, that can always be
        // (effectively) coerced to unit.
        if !ty.is_never() {
            self.demand_suptype(blk.span, unit, ty);
        }
    }

    pub(in super::super) fn check_expr_block(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        // In some cases, blocks have just one exit, but other blocks
        // can be targeted by multiple breaks. This can happen both
        // with labeled blocks as well as when we desugar
        // a `try { ... }` expression.
        //
        // Example 1:
        //
        //    'a: { if true { break 'a Err(()); } Ok(()) }
        //
        // Here we would wind up with two coercions, one from
        // `Err(())` and the other from the tail expression
        // `Ok(())`. If the tail expression is omitted, that's a
        // "forced unit" -- unless the block diverges, in which
        // case we can ignore the tail expression (e.g., `'a: {
        // break 'a 22; }` would not force the type of the block
        // to be `()`).
        let coerce_to_ty = expected.coercion_target_type(self, blk.span);
        let coerce = if blk.targeted_by_break {
            CoerceMany::new(coerce_to_ty)
        } else {
            CoerceMany::with_coercion_sites(coerce_to_ty, blk.expr.as_slice())
        };

        let prev_diverges = self.diverges.get();
        let ctxt = BreakableCtxt { coerce: Some(coerce), may_break: false };

        let (ctxt, ()) = self.with_breakable_ctxt(blk.hir_id, ctxt, || {
            for s in blk.stmts {
                self.check_stmt(s);
            }

            // check the tail expression **without** holding the
            // `enclosing_breakables` lock below.
            let tail_expr_ty =
                blk.expr.map(|expr| (expr, self.check_expr_with_expectation(expr, expected)));

            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            let ctxt = enclosing_breakables.find_breakable(blk.hir_id);
            let coerce = ctxt.coerce.as_mut().unwrap();
            if let Some((tail_expr, tail_expr_ty)) = tail_expr_ty {
                let span = self.get_expr_coercion_span(tail_expr);
                let cause = self.cause(
                    span,
                    ObligationCauseCode::BlockTailExpression(blk.hir_id, hir::MatchSource::Normal),
                );
                let ty_for_diagnostic = coerce.merged_ty();
                // We use coerce_inner here because we want to augment the error
                // suggesting to wrap the block in square brackets if it might've
                // been mistaken array syntax
                coerce.coerce_inner(
                    self,
                    &cause,
                    Some(tail_expr),
                    tail_expr_ty,
                    |diag| {
                        self.suggest_block_to_brackets(diag, blk, tail_expr_ty, ty_for_diagnostic);
                    },
                    false,
                );
            } else {
                // Subtle: if there is no explicit tail expression,
                // that is typically equivalent to a tail expression
                // of `()` -- except if the block diverges. In that
                // case, there is no value supplied from the tail
                // expression (assuming there are no other breaks,
                // this implies that the type of the block will be
                // `!`).
                //
                // #41425 -- label the implicit `()` as being the
                // "found type" here, rather than the "expected type".
                if !self.diverges.get().is_always()
                    || matches!(self.diverging_block_behavior, DivergingBlockBehavior::Unit)
                {
                    // #50009 -- Do not point at the entire fn block span, point at the return type
                    // span, as it is the cause of the requirement, and
                    // `consider_hint_about_removing_semicolon` will point at the last expression
                    // if it were a relevant part of the error. This improves usability in editors
                    // that highlight errors inline.
                    let mut sp = blk.span;
                    let mut fn_span = None;
                    if let Some((fn_def_id, decl)) = self.get_fn_decl(blk.hir_id) {
                        let ret_sp = decl.output.span();
                        if let Some(block_sp) = self.parent_item_span(blk.hir_id) {
                            // HACK: on some cases (`ui/liveness/liveness-issue-2163.rs`) the
                            // output would otherwise be incorrect and even misleading. Make sure
                            // the span we're aiming at correspond to a `fn` body.
                            if block_sp == blk.span {
                                sp = ret_sp;
                                fn_span = self.tcx.def_ident_span(fn_def_id);
                            }
                        }
                    }
                    coerce.coerce_forced_unit(
                        self,
                        &self.misc(sp),
                        |err| {
                            if let Some(expected_ty) = expected.only_has_type(self) {
                                if blk.stmts.is_empty() && blk.expr.is_none() {
                                    self.suggest_boxing_when_appropriate(
                                        err,
                                        blk.span,
                                        blk.hir_id,
                                        expected_ty,
                                        self.tcx.types.unit,
                                    );
                                }
                                if !self.err_ctxt().consider_removing_semicolon(
                                    blk,
                                    expected_ty,
                                    err,
                                ) {
                                    self.err_ctxt().consider_returning_binding(
                                        blk,
                                        expected_ty,
                                        err,
                                    );
                                }
                                if expected_ty == self.tcx.types.bool {
                                    // If this is caused by a missing `let` in a `while let`,
                                    // silence this redundant error, as we already emit E0070.

                                    // Our block must be a `assign desugar local; assignment`
                                    if let hir::Block {
                                        stmts:
                                            [
                                                hir::Stmt {
                                                    kind:
                                                        hir::StmtKind::Let(hir::LetStmt {
                                                            source:
                                                                hir::LocalSource::AssignDesugar(_),
                                                            ..
                                                        }),
                                                    ..
                                                },
                                                hir::Stmt {
                                                    kind:
                                                        hir::StmtKind::Expr(hir::Expr {
                                                            kind: hir::ExprKind::Assign(lhs, ..),
                                                            ..
                                                        }),
                                                    ..
                                                },
                                            ],
                                        ..
                                    } = blk
                                    {
                                        self.comes_from_while_condition(blk.hir_id, |_| {
                                            // We cannot suppress the error if the LHS of assignment
                                            // is a syntactic place expression because E0070 would
                                            // not be emitted by `check_lhs_assignable`.
                                            let res = self.typeck_results.borrow().expr_ty_opt(lhs);

                                            if !lhs.is_syntactic_place_expr()
                                                || res.references_error()
                                            {
                                                err.downgrade_to_delayed_bug();
                                            }
                                        })
                                    }
                                }
                            }
                            if let Some(fn_span) = fn_span {
                                err.span_label(
                                    fn_span,
                                    "implicitly returns `()` as its body has no tail or `return` \
                                     expression",
                                );
                            }
                        },
                        false,
                    );
                }
            }
        });

        if ctxt.may_break {
            // If we can break from the block, then the block's exit is always reachable
            // (... as long as the entry is reachable) - regardless of the tail of the block.
            self.diverges.set(prev_diverges);
        }

        let ty = ctxt.coerce.unwrap().complete(self);

        self.write_ty(blk.hir_id, ty);

        ty
    }

    fn parent_item_span(&self, id: HirId) -> Option<Span> {
        let node = self.tcx.hir_node_by_def_id(self.tcx.hir_get_parent_item(id).def_id);
        match node {
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn { body: body_id, .. }, .. })
            | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(_, body_id), .. }) => {
                let body = self.tcx.hir_body(body_id);
                if let ExprKind::Block(block, _) = &body.value.kind {
                    return Some(block.span);
                }
            }
            _ => {}
        }
        None
    }

    /// If `expr` is a `match` expression that has only one non-`!` arm, use that arm's tail
    /// expression's `Span`, otherwise return `expr.span`. This is done to give better errors
    /// when given code like the following:
    /// ```text
    /// if false { return 0i32; } else { 1u32 }
    /// //                               ^^^^ point at this instead of the whole `if` expression
    /// ```
    fn get_expr_coercion_span(&self, expr: &hir::Expr<'_>) -> rustc_span::Span {
        let check_in_progress = |elem: &hir::Expr<'_>| {
            self.typeck_results.borrow().node_type_opt(elem.hir_id).filter(|ty| !ty.is_never()).map(
                |_| match elem.kind {
                    // Point at the tail expression when possible.
                    hir::ExprKind::Block(block, _) => block.expr.map_or(block.span, |e| e.span),
                    _ => elem.span,
                },
            )
        };

        if let hir::ExprKind::If(_, _, Some(el)) = expr.kind {
            if let Some(rslt) = check_in_progress(el) {
                return rslt;
            }
        }

        if let hir::ExprKind::Match(_, arms, _) = expr.kind {
            let mut iter = arms.iter().filter_map(|arm| check_in_progress(arm.body));
            if let Some(span) = iter.next() {
                if iter.next().is_none() {
                    return span;
                }
            }
        }

        expr.span
    }

    fn overwrite_local_ty_if_err(&self, hir_id: HirId, pat: &'tcx hir::Pat<'tcx>, ty: Ty<'tcx>) {
        if let Err(guar) = ty.error_reported() {
            struct OverwritePatternsWithError {
                pat_hir_ids: Vec<hir::HirId>,
            }
            impl<'tcx> Visitor<'tcx> for OverwritePatternsWithError {
                fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
                    self.pat_hir_ids.push(p.hir_id);
                    hir::intravisit::walk_pat(self, p);
                }
            }
            // Override the types everywhere with `err()` to avoid knock on errors.
            let err = Ty::new_error(self.tcx, guar);
            self.write_ty(hir_id, err);
            self.write_ty(pat.hir_id, err);
            let mut visitor = OverwritePatternsWithError { pat_hir_ids: vec![] };
            hir::intravisit::walk_pat(&mut visitor, pat);
            // Mark all the subpatterns as `{type error}` as well. This allows errors for specific
            // subpatterns to be silenced.
            for hir_id in visitor.pat_hir_ids {
                self.write_ty(hir_id, err);
            }
            self.locals.borrow_mut().insert(hir_id, err);
            self.locals.borrow_mut().insert(pat.hir_id, err);
        }
    }

    // Finish resolving a path in a struct expression or pattern `S::A { .. }` if necessary.
    // The newly resolved definition is written into `type_dependent_defs`.
    fn finish_resolving_struct_path(
        &self,
        qpath: &QPath<'tcx>,
        path_span: Span,
        hir_id: HirId,
    ) -> (Res, LoweredTy<'tcx>) {
        match *qpath {
            QPath::Resolved(ref maybe_qself, path) => {
                let self_ty = maybe_qself.as_ref().map(|qself| self.lower_ty(qself).raw);
                let ty = self.lowerer().lower_resolved_ty_path(
                    self_ty,
                    path,
                    hir_id,
                    PermitVariants::Yes,
                );
                (path.res, LoweredTy::from_raw(self, path_span, ty))
            }
            QPath::TypeRelative(hir_self_ty, segment) => {
                let self_ty = self.lower_ty(hir_self_ty);

                let result = self.lowerer().lower_type_relative_ty_path(
                    self_ty.raw,
                    hir_self_ty,
                    segment,
                    hir_id,
                    path_span,
                    PermitVariants::Yes,
                );
                let ty = result
                    .map(|(ty, _, _)| ty)
                    .unwrap_or_else(|guar| Ty::new_error(self.tcx(), guar));
                let ty = LoweredTy::from_raw(self, path_span, ty);
                let result = result.map(|(_, kind, def_id)| (kind, def_id));

                // Write back the new resolution.
                self.write_resolution(hir_id, result);

                (result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)), ty)
            }
            QPath::LangItem(lang_item, span) => {
                let (res, ty) = self.resolve_lang_item_path(lang_item, span, hir_id);
                (res, LoweredTy::from_raw(self, path_span, ty))
            }
        }
    }

    /// Given a vector of fulfillment errors, try to adjust the spans of the
    /// errors to more accurately point at the cause of the failure.
    ///
    /// This applies to calls, methods, and struct expressions. This will also
    /// try to deduplicate errors that are due to the same cause but might
    /// have been created with different [`ObligationCause`][traits::ObligationCause]s.
    pub(super) fn adjust_fulfillment_errors_for_expr_obligation(
        &self,
        errors: &mut Vec<traits::FulfillmentError<'tcx>>,
    ) {
        // Store a mapping from `(Span, Predicate) -> ObligationCause`, so that
        // other errors that have the same span and predicate can also get fixed,
        // even if their `ObligationCauseCode` isn't an `Expr*Obligation` kind.
        // This is important since if we adjust one span but not the other, then
        // we will have "duplicated" the error on the UI side.
        let mut remap_cause = FxIndexSet::default();
        let mut not_adjusted = vec![];

        for error in errors {
            let before_span = error.obligation.cause.span;
            if self.adjust_fulfillment_error_for_expr_obligation(error)
                || before_span != error.obligation.cause.span
            {
                remap_cause.insert((
                    before_span,
                    error.obligation.predicate,
                    error.obligation.cause.clone(),
                ));
            } else {
                // If it failed to be adjusted once around, it may be adjusted
                // via the "remap cause" mapping the second time...
                not_adjusted.push(error);
            }
        }

        // Adjust any other errors that come from other cause codes, when these
        // errors are of the same predicate as one we successfully adjusted, and
        // when their spans overlap (suggesting they're due to the same root cause).
        //
        // This is because due to normalization, we often register duplicate
        // obligations with misc obligations that are basically impossible to
        // line back up with a useful WhereClauseInExpr.
        for error in not_adjusted {
            for (span, predicate, cause) in &remap_cause {
                if *predicate == error.obligation.predicate
                    && span.contains(error.obligation.cause.span)
                {
                    error.obligation.cause = cause.clone();
                    continue;
                }
            }
        }
    }

    fn label_fn_like(
        &self,
        err: &mut Diag<'_>,
        callable_def_id: Option<DefId>,
        callee_ty: Option<Ty<'tcx>>,
        call_expr: &'tcx hir::Expr<'tcx>,
        expected_ty: Option<Ty<'tcx>>,
        // A specific argument should be labeled, instead of all of them
        expected_idx: Option<usize>,
        matched_inputs: &IndexVec<ExpectedIdx, Option<ProvidedIdx>>,
        formal_and_expected_inputs: &IndexVec<ExpectedIdx, (Ty<'tcx>, Ty<'tcx>)>,
        is_method: bool,
        tuple_arguments: TupleArgumentsFlag,
    ) {
        let Some(mut def_id) = callable_def_id else {
            return;
        };

        // If we're calling a method of a Fn/FnMut/FnOnce trait object implicitly
        // (eg invoking a closure) we want to point at the underlying callable,
        // not the method implicitly invoked (eg call_once).
        // TupleArguments is set only when this is an implicit call (my_closure(...)) rather than explicit (my_closure.call(...))
        if tuple_arguments == TupleArguments
            && let Some(assoc_item) = self.tcx.opt_associated_item(def_id)
            // Since this is an associated item, it might point at either an impl or a trait item.
            // We want it to always point to the trait item.
            // If we're pointing at an inherent function, we don't need to do anything,
            // so we fetch the parent and verify if it's a trait item.
            && let maybe_trait_item_def_id = assoc_item.trait_item_def_id.unwrap_or(def_id)
            && let maybe_trait_def_id = self.tcx.parent(maybe_trait_item_def_id)
            // Just an easy way to check "trait_def_id == Fn/FnMut/FnOnce"
            && let Some(call_kind) = self.tcx.fn_trait_kind_from_def_id(maybe_trait_def_id)
            && let Some(callee_ty) = callee_ty
        {
            let callee_ty = callee_ty.peel_refs();
            match *callee_ty.kind() {
                ty::Param(param) => {
                    let param = self.tcx.generics_of(self.body_id).type_param(param, self.tcx);
                    if param.kind.is_synthetic() {
                        // if it's `impl Fn() -> ..` then just fall down to the def-id based logic
                        def_id = param.def_id;
                    } else {
                        // Otherwise, find the predicate that makes this generic callable,
                        // and point at that.
                        let instantiated = self
                            .tcx
                            .explicit_predicates_of(self.body_id)
                            .instantiate_identity(self.tcx);
                        // FIXME(compiler-errors): This could be problematic if something has two
                        // fn-like predicates with different args, but callable types really never
                        // do that, so it's OK.
                        for (predicate, span) in instantiated {
                            if let ty::ClauseKind::Trait(pred) = predicate.kind().skip_binder()
                                && pred.self_ty().peel_refs() == callee_ty
                                && self.tcx.is_fn_trait(pred.def_id())
                            {
                                err.span_note(span, "callable defined here");
                                return;
                            }
                        }
                    }
                }
                ty::Alias(ty::Opaque, ty::AliasTy { def_id: new_def_id, .. })
                | ty::Closure(new_def_id, _)
                | ty::FnDef(new_def_id, _) => {
                    def_id = new_def_id;
                }
                _ => {
                    // Look for a user-provided impl of a `Fn` trait, and point to it.
                    let new_def_id = self.probe(|_| {
                        let trait_ref = ty::TraitRef::new(
                            self.tcx,
                            self.tcx.fn_trait_kind_to_def_id(call_kind)?,
                            [callee_ty, self.next_ty_var(DUMMY_SP)],
                        );
                        let obligation = traits::Obligation::new(
                            self.tcx,
                            traits::ObligationCause::dummy(),
                            self.param_env,
                            trait_ref,
                        );
                        match SelectionContext::new(self).select(&obligation) {
                            Ok(Some(traits::ImplSource::UserDefined(impl_source))) => {
                                Some(impl_source.impl_def_id)
                            }
                            _ => None,
                        }
                    });
                    if let Some(new_def_id) = new_def_id {
                        def_id = new_def_id;
                    } else {
                        return;
                    }
                }
            }
        }

        if let Some(def_span) = self.tcx.def_ident_span(def_id)
            && !def_span.is_dummy()
        {
            let mut spans: MultiSpan = def_span.into();
            if let Some((params_with_generics, hir_generics)) =
                self.get_hir_param_info(def_id, is_method)
            {
                struct MismatchedParam<'a> {
                    idx: ExpectedIdx,
                    generic: GenericIdx,
                    param: &'a FnParam<'a>,
                    deps: SmallVec<[ExpectedIdx; 4]>,
                }

                debug_assert_eq!(params_with_generics.len(), matched_inputs.len());
                // Gather all mismatched parameters with generics.
                let mut mismatched_params = Vec::<MismatchedParam<'_>>::new();
                if let Some(expected_idx) = expected_idx {
                    let expected_idx = ExpectedIdx::from_usize(expected_idx);
                    let &(expected_generic, ref expected_param) =
                        &params_with_generics[expected_idx];
                    if let Some(expected_generic) = expected_generic {
                        mismatched_params.push(MismatchedParam {
                            idx: expected_idx,
                            generic: expected_generic,
                            param: expected_param,
                            deps: SmallVec::new(),
                        });
                    } else {
                        // Still mark the mismatched parameter
                        spans.push_span_label(expected_param.span(), "");
                    }
                } else {
                    mismatched_params.extend(
                        params_with_generics.iter_enumerated().zip(matched_inputs).filter_map(
                            |((idx, &(generic, ref param)), matched_idx)| {
                                if matched_idx.is_some() {
                                    None
                                } else if let Some(generic) = generic {
                                    Some(MismatchedParam {
                                        idx,
                                        generic,
                                        param,
                                        deps: SmallVec::new(),
                                    })
                                } else {
                                    // Still mark mismatched parameters
                                    spans.push_span_label(param.span(), "");
                                    None
                                }
                            },
                        ),
                    );
                }

                if !mismatched_params.is_empty() {
                    // For each mismatched parameter, create a two-way link to each matched parameter
                    // of the same type.
                    let mut dependants = IndexVec::<ExpectedIdx, _>::from_fn_n(
                        |_| SmallVec::<[u32; 4]>::new(),
                        params_with_generics.len(),
                    );
                    let mut generic_uses = IndexVec::<GenericIdx, _>::from_fn_n(
                        |_| SmallVec::<[ExpectedIdx; 4]>::new(),
                        hir_generics.params.len(),
                    );
                    for (idx, param) in mismatched_params.iter_mut().enumerate() {
                        for ((other_idx, &(other_generic, _)), &other_matched_idx) in
                            params_with_generics.iter_enumerated().zip(matched_inputs)
                        {
                            if other_generic == Some(param.generic) && other_matched_idx.is_some() {
                                generic_uses[param.generic].extend([param.idx, other_idx]);
                                dependants[other_idx].push(idx as u32);
                                param.deps.push(other_idx);
                            }
                        }
                    }

                    // Highlight each mismatched type along with a note about which other parameters
                    // the type depends on (if any).
                    for param in &mismatched_params {
                        if let Some(deps_list) = listify(&param.deps, |&dep| {
                            params_with_generics[dep].1.display(dep.as_usize()).to_string()
                        }) {
                            spans.push_span_label(
                                param.param.span(),
                                format!(
                                    "this parameter needs to match the {} type of {deps_list}",
                                    self.resolve_vars_if_possible(
                                        formal_and_expected_inputs[param.deps[0]].1
                                    )
                                    .sort_string(self.tcx),
                                ),
                            );
                        } else {
                            // Still mark mismatched parameters
                            spans.push_span_label(param.param.span(), "");
                        }
                    }
                    // Highligh each parameter being depended on for a generic type.
                    for ((&(_, param), deps), &(_, expected_ty)) in
                        params_with_generics.iter().zip(&dependants).zip(formal_and_expected_inputs)
                    {
                        if let Some(deps_list) = listify(deps, |&dep| {
                            let param = &mismatched_params[dep as usize];
                            param.param.display(param.idx.as_usize()).to_string()
                        }) {
                            spans.push_span_label(
                                param.span(),
                                format!(
                                    "{deps_list} need{} to match the {} type of this parameter",
                                    pluralize!((deps.len() != 1) as u32),
                                    self.resolve_vars_if_possible(expected_ty)
                                        .sort_string(self.tcx),
                                ),
                            );
                        }
                    }
                    // Highlight each generic parameter in use.
                    for (param, uses) in hir_generics.params.iter().zip(&mut generic_uses) {
                        uses.sort();
                        uses.dedup();
                        if let Some(param_list) = listify(uses, |&idx| {
                            params_with_generics[idx].1.display(idx.as_usize()).to_string()
                        }) {
                            spans.push_span_label(
                                param.span,
                                format!(
                                    "{param_list} {} reference this parameter `{}`",
                                    if uses.len() == 2 { "both" } else { "all" },
                                    param.name.ident().name,
                                ),
                            );
                        }
                    }
                }
            }
            err.span_note(spans, format!("{} defined here", self.tcx.def_descr(def_id)));
        } else if let Some(hir::Node::Expr(e)) = self.tcx.hir_get_if_local(def_id)
            && let hir::ExprKind::Closure(hir::Closure { body, .. }) = &e.kind
        {
            let param = expected_idx
                .and_then(|expected_idx| self.tcx.hir_body(*body).params.get(expected_idx));
            let (kind, span) = if let Some(param) = param {
                // Try to find earlier invocations of this closure to find if the type mismatch
                // is because of inference. If we find one, point at them.
                let mut call_finder = FindClosureArg { tcx: self.tcx, calls: vec![] };
                let parent_def_id = self.tcx.hir_get_parent_item(call_expr.hir_id).def_id;
                match self.tcx.hir_node_by_def_id(parent_def_id) {
                    hir::Node::Item(item) => call_finder.visit_item(item),
                    hir::Node::TraitItem(item) => call_finder.visit_trait_item(item),
                    hir::Node::ImplItem(item) => call_finder.visit_impl_item(item),
                    _ => {}
                }
                let typeck = self.typeck_results.borrow();
                for (rcvr, args) in call_finder.calls {
                    if rcvr.hir_id.owner == typeck.hir_owner
                        && let Some(rcvr_ty) = typeck.node_type_opt(rcvr.hir_id)
                        && let ty::Closure(call_def_id, _) = rcvr_ty.kind()
                        && def_id == *call_def_id
                        && let Some(idx) = expected_idx
                        && let Some(arg) = args.get(idx)
                        && let Some(arg_ty) = typeck.node_type_opt(arg.hir_id)
                        && let Some(expected_ty) = expected_ty
                        && self.can_eq(self.param_env, arg_ty, expected_ty)
                    {
                        let mut sp: MultiSpan = vec![arg.span].into();
                        sp.push_span_label(
                            arg.span,
                            format!("expected because this argument is of type `{arg_ty}`"),
                        );
                        sp.push_span_label(rcvr.span, "in this closure call");
                        err.span_note(
                            sp,
                            format!(
                                "expected because the closure was earlier called with an \
                                argument of type `{arg_ty}`",
                            ),
                        );
                        break;
                    }
                }

                ("closure parameter", param.span)
            } else {
                ("closure", self.tcx.def_span(def_id))
            };
            err.span_note(span, format!("{kind} defined here"));
        } else {
            err.span_note(
                self.tcx.def_span(def_id),
                format!("{} defined here", self.tcx.def_descr(def_id)),
            );
        }
    }

    fn label_generic_mismatches(
        &self,
        err: &mut Diag<'_>,
        callable_def_id: Option<DefId>,
        matched_inputs: &IndexVec<ExpectedIdx, Option<ProvidedIdx>>,
        provided_arg_tys: &IndexVec<ProvidedIdx, (Ty<'tcx>, Span)>,
        formal_and_expected_inputs: &IndexVec<ExpectedIdx, (Ty<'tcx>, Ty<'tcx>)>,
        is_method: bool,
    ) {
        let Some(def_id) = callable_def_id else {
            return;
        };

        if let Some((params_with_generics, _)) = self.get_hir_param_info(def_id, is_method) {
            debug_assert_eq!(params_with_generics.len(), matched_inputs.len());
            for (idx, (generic_param, _)) in params_with_generics.iter_enumerated() {
                if matched_inputs[idx].is_none() {
                    continue;
                }

                let Some((_, matched_arg_span)) = provided_arg_tys.get(idx.to_provided_idx())
                else {
                    continue;
                };

                let Some(generic_param) = generic_param else {
                    continue;
                };

                let idxs_matched = params_with_generics
                    .iter_enumerated()
                    .filter(|&(other_idx, (other_generic_param, _))| {
                        if other_idx == idx {
                            return false;
                        }
                        let Some(other_generic_param) = other_generic_param else {
                            return false;
                        };
                        if matched_inputs[other_idx].is_some() {
                            return false;
                        }
                        other_generic_param == generic_param
                    })
                    .count();

                if idxs_matched == 0 {
                    continue;
                }

                let expected_display_type = self
                    .resolve_vars_if_possible(formal_and_expected_inputs[idx].1)
                    .sort_string(self.tcx);
                let label = if idxs_matched == params_with_generics.len() - 1 {
                    format!(
                        "expected all arguments to be this {} type because they need to match the type of this parameter",
                        expected_display_type
                    )
                } else {
                    format!(
                        "expected some other arguments to be {} {} type to match the type of this parameter",
                        a_or_an(&expected_display_type),
                        expected_display_type,
                    )
                };

                err.span_label(*matched_arg_span, label);
            }
        }
    }

    /// Returns the parameters of a function, with their generic parameters if those are the full
    /// type of that parameter.
    ///
    /// Returns `None` if the body is not a named function (e.g. a closure).
    fn get_hir_param_info(
        &self,
        def_id: DefId,
        is_method: bool,
    ) -> Option<(IndexVec<ExpectedIdx, (Option<GenericIdx>, FnParam<'_>)>, &hir::Generics<'_>)>
    {
        let (sig, generics, body_id, params) = match self.tcx.hir_get_if_local(def_id)? {
            hir::Node::TraitItem(&hir::TraitItem {
                generics,
                kind: hir::TraitItemKind::Fn(sig, trait_fn),
                ..
            }) => match trait_fn {
                hir::TraitFn::Required(params) => (sig, generics, None, Some(params)),
                hir::TraitFn::Provided(body) => (sig, generics, Some(body), None),
            },
            hir::Node::ImplItem(&hir::ImplItem {
                generics,
                kind: hir::ImplItemKind::Fn(sig, body),
                ..
            })
            | hir::Node::Item(&hir::Item {
                kind: hir::ItemKind::Fn { sig, generics, body, .. },
                ..
            }) => (sig, generics, Some(body), None),
            hir::Node::ForeignItem(&hir::ForeignItem {
                kind: hir::ForeignItemKind::Fn(sig, params, generics),
                ..
            }) => (sig, generics, None, Some(params)),
            _ => return None,
        };

        // Make sure to remove both the receiver and variadic argument. Both are removed
        // when matching parameter types.
        let fn_inputs = sig.decl.inputs.get(is_method as usize..)?.iter().map(|param| {
            if let hir::TyKind::Path(QPath::Resolved(
                _,
                &hir::Path { res: Res::Def(_, res_def_id), .. },
            )) = param.kind
            {
                generics
                    .params
                    .iter()
                    .position(|param| param.def_id.to_def_id() == res_def_id)
                    .map(GenericIdx::from_usize)
            } else {
                None
            }
        });
        match (body_id, params) {
            (Some(_), Some(_)) | (None, None) => unreachable!(),
            (Some(body), None) => {
                let params = self.tcx.hir_body(body).params;
                let params =
                    params.get(is_method as usize..params.len() - sig.decl.c_variadic as usize)?;
                debug_assert_eq!(params.len(), fn_inputs.len());
                Some((
                    fn_inputs.zip(params.iter().map(|param| FnParam::Param(param))).collect(),
                    generics,
                ))
            }
            (None, Some(params)) => {
                let params =
                    params.get(is_method as usize..params.len() - sig.decl.c_variadic as usize)?;
                debug_assert_eq!(params.len(), fn_inputs.len());
                Some((
                    fn_inputs.zip(params.iter().map(|&ident| FnParam::Ident(ident))).collect(),
                    generics,
                ))
            }
        }
    }
}

struct FindClosureArg<'tcx> {
    tcx: TyCtxt<'tcx>,
    calls: Vec<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
}

impl<'tcx> Visitor<'tcx> for FindClosureArg<'tcx> {
    type NestedFilter = rustc_middle::hir::nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Call(rcvr, args) = ex.kind {
            self.calls.push((rcvr, args));
        }
        hir::intravisit::walk_expr(self, ex);
    }
}

#[derive(Clone, Copy)]
enum FnParam<'hir> {
    Param(&'hir hir::Param<'hir>),
    Ident(Option<Ident>),
}

impl FnParam<'_> {
    fn span(&self) -> Span {
        match self {
            Self::Param(param) => param.span,
            Self::Ident(ident) => {
                if let Some(ident) = ident {
                    ident.span
                } else {
                    DUMMY_SP
                }
            }
        }
    }

    fn display(&self, idx: usize) -> impl '_ + fmt::Display {
        struct D<'a>(FnParam<'a>, usize);
        impl fmt::Display for D<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // A "unique" param name is one that (a) exists, and (b) is guaranteed to be unique
                // among the parameters, i.e. `_` does not count.
                let unique_name = match self.0 {
                    FnParam::Param(param)
                        if let hir::PatKind::Binding(_, _, ident, _) = param.pat.kind =>
                    {
                        Some(ident.name)
                    }
                    FnParam::Ident(ident)
                        if let Some(ident) = ident
                            && ident.name != kw::Underscore =>
                    {
                        Some(ident.name)
                    }
                    _ => None,
                };
                if let Some(unique_name) = unique_name {
                    write!(f, "`{unique_name}`")
                } else {
                    write!(f, "parameter #{}", self.1 + 1)
                }
            }
        }
        D(*self, idx)
    }
}
