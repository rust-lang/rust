use crate::coercion::CoerceMany;
use crate::fn_ctxt::arg_matrix::{ArgMatrix, Compatibility, Error, ExpectedIdx, ProvidedIdx};
use crate::gather_locals::Declaration;
use crate::method::MethodCallee;
use crate::TupleArgumentsFlag::*;
use crate::{errors, Expectation::*};
use crate::{
    struct_span_err, BreakableCtxt, Diverges, Expectation, FnCtxt, LocalTy, Needs, RawTy,
    TupleArgumentsFlag,
};
use rustc_ast as ast;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{
    pluralize, Applicability, Diagnostic, DiagnosticId, ErrorGuaranteed, MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_hir_analysis::astconv::AstConv;
use rustc_hir_analysis::check::intrinsicck::InlineAsmCtxt;
use rustc_hir_analysis::check::potentially_plural_count;
use rustc_hir_analysis::structured_errors::StructuredDiagnostic;
use rustc_index::IndexVec;
use rustc_infer::infer::error_reporting::{FailureCode, ObligationCauseExt};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::TypeTrace;
use rustc_infer::infer::{DefineOpaqueTypes, InferOk};
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, IsSuggestable, Ty};
use rustc_session::Session;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{self, sym, BytePos, Span};
use rustc_trait_selection::traits::{self, ObligationCauseCode, SelectionContext};

use std::iter;
use std::mem;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(in super::super) fn check_casts(&mut self) {
        // don't hold the borrow to deferred_cast_checks while checking to avoid borrow checker errors
        // when writing to `self.param_env`.
        let mut deferred_cast_checks = mem::take(&mut *self.deferred_cast_checks.borrow_mut());

        debug!("FnCtxt::check_casts: {} deferred checks", deferred_cast_checks.len());
        for cast in deferred_cast_checks.drain(..) {
            let prev_env = self.param_env;
            self.param_env = self.param_env.with_constness(cast.constness);

            cast.check(self);

            self.param_env = prev_env;
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
            let enclosing_id = self.tcx.hir().enclosing_body_owner(hir_id);
            let get_operand_ty = |expr| {
                let ty = self.typeck_results.borrow().expr_ty_adjusted(expr);
                let ty = self.resolve_vars_if_possible(ty);
                if ty.has_non_region_infer() {
                    self.tcx.ty_error_misc()
                } else {
                    self.tcx.erase_regions(ty)
                }
            };
            InlineAsmCtxt::new_in_fn(self.tcx, self.param_env, get_operand_ty)
                .check_asm(asm, enclosing_id);
        }
    }

    pub(in super::super) fn check_method_argument_types(
        &self,
        sp: Span,
        expr: &'tcx hir::Expr<'tcx>,
        method: Result<MethodCallee<'tcx>, ()>,
        args_no_rcvr: &'tcx [hir::Expr<'tcx>],
        tuple_arguments: TupleArgumentsFlag,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let has_error = match method {
            Ok(method) => method.substs.references_error() || method.sig.references_error(),
            Err(_) => true,
        };
        if has_error {
            let err_inputs = self.err_args(args_no_rcvr.len());

            let err_inputs = match tuple_arguments {
                DontTupleArguments => err_inputs,
                TupleArguments => vec![self.tcx.mk_tup(&err_inputs)],
            };

            self.check_argument_types(
                sp,
                expr,
                &err_inputs,
                None,
                args_no_rcvr,
                false,
                tuple_arguments,
                method.ok().map(|method| method.def_id),
            );
            return self.tcx.ty_error_misc();
        }

        let method = method.unwrap();
        // HACK(eddyb) ignore self in the definition (see above).
        let expected_input_tys = self.expected_inputs_for_expected_output(
            sp,
            expected,
            method.sig.output(),
            &method.sig.inputs()[1..],
        );
        self.check_argument_types(
            sp,
            expr,
            &method.sig.inputs()[1..],
            expected_input_tys,
            args_no_rcvr,
            method.sig.c_variadic,
            tuple_arguments,
            Some(method.def_id),
        );

        method.sig.output()
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
        // More specific expected types, after unifying with caller output types
        expected_input_tys: Option<Vec<Ty<'tcx>>>,
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
            self.register_wf_obligation(fn_input_ty.into(), arg_expr.span, traits::MiscObligation);
        }

        let mut err_code = "E0061";

        // If the arguments should be wrapped in a tuple (ex: closures), unwrap them here
        let (formal_input_tys, expected_input_tys) = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolved_type(call_span, formal_input_tys[0]);
            match tuple_type.kind() {
                // We expected a tuple and got a tuple
                ty::Tuple(arg_types) => {
                    // Argument length differs
                    if arg_types.len() != provided_args.len() {
                        err_code = "E0057";
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
                    struct_span_err!(
                        tcx.sess,
                        call_span,
                        E0059,
                        "cannot use call notation; the first type parameter \
                         for the function trait is neither a tuple nor unit"
                    )
                    .emit();
                    (self.err_args(provided_args.len()), None)
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

        let is_const_eval_select = matches!(fn_def_id, Some(def_id) if
            self.tcx.def_kind(def_id) == hir::def::DefKind::Fn
            && self.tcx.is_intrinsic(def_id)
            && self.tcx.item_name(def_id) == sym::const_eval_select);

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

            let coerce_error = self
                .try_coerce(provided_arg, checked_ty, coerced_ty, AllowTwoPhase::Yes, None)
                .err();

            if coerce_error.is_some() {
                return Compatibility::Incompatible(coerce_error);
            }

            // Check that second and third argument of `const_eval_select` must be `FnDef`, and additionally that
            // the second argument must be `const fn`. The first argument must be a tuple, but this is already expressed
            // in the function signature (`F: FnOnce<ARG>`), so I did not bother to add another check here.
            //
            // This check is here because there is currently no way to express a trait bound for `FnDef` types only.
            if is_const_eval_select && (1..=2).contains(&idx) {
                if let ty::FnDef(def_id, _) = checked_ty.kind() {
                    if idx == 1 && !self.tcx.is_const_fn_raw(*def_id) {
                        self.tcx
                            .sess
                            .emit_err(errors::ConstSelectMustBeConst { span: provided_arg.span });
                    }
                } else {
                    self.tcx.sess.emit_err(errors::ConstSelectMustBeFn {
                        span: provided_arg.span,
                        ty: checked_ty,
                    });
                }
            }

            // 3. Check if the formal type is a supertype of the checked one
            //    and register any such obligations for future type checks
            let supertype_error = self.at(&self.misc(provided_arg.span), self.param_env).sup(
                DefineOpaqueTypes::No,
                formal_input_ty,
                coerced_ty,
            );
            let subtyping_error = match supertype_error {
                Ok(InferOk { obligations, value: () }) => {
                    self.register_predicates(obligations);
                    None
                }
                Err(err) => Some(err),
            };

            // If neither check failed, the types are compatible
            match subtyping_error {
                None => Compatibility::Compatible,
                Some(_) => Compatibility::Incompatible(subtyping_error),
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

                // For this check, we do *not* want to treat async generator closures (async blocks)
                // as proper closures. Doing so would regress type inference when feeding
                // the return value of an argument-position async block to an argument-position
                // closure wrapped in a block.
                // See <https://github.com/rust-lang/rust/issues/112225>.
                let is_closure = if let ExprKind::Closure(closure) = arg.kind {
                    !tcx.generator_is_async(closure.def_id.to_def_id())
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
            err_code = "E0060";
        }

        for arg in provided_args.iter().skip(minimum_input_count) {
            // Make sure we've checked this expr at least once.
            let arg_ty = self.check_expr(&arg);

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
                    use rustc_hir_analysis::structured_errors::MissingCastForVariadicArg;

                    MissingCastForVariadicArg { sess, span, ty, cast_ty }.diagnostic().emit();
                }

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolved_type(arg.span, arg_ty);
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
                        let ptr_ty = self.tcx.mk_fn_ptr(arg_ty.fn_sig(self.tcx));
                        let ptr_ty = self.resolve_vars_if_possible(ptr_ty);
                        variadic_error(tcx.sess, arg.span, arg_ty, &ptr_ty.to_string());
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
                    .zip(expected_input_tys.iter().copied())
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
            );
        }
    }

    fn report_arg_errors(
        &self,
        compatibility_diagonal: IndexVec<ProvidedIdx, Compatibility<'tcx>>,
        formal_and_expected_inputs: IndexVec<ExpectedIdx, (Ty<'tcx>, Ty<'tcx>)>,
        provided_args: IndexVec<ProvidedIdx, &'tcx hir::Expr<'tcx>>,
        c_variadic: bool,
        err_code: &str,
        fn_def_id: Option<DefId>,
        call_span: Span,
        call_expr: &'tcx hir::Expr<'tcx>,
    ) {
        // Next, let's construct the error
        let (error_span, full_call_span, call_name, is_method) = match &call_expr.kind {
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
                    (call_span, *span, name, false)
                } else {
                    (call_span, *span, "function", false)
                }
            }
            hir::ExprKind::Call(hir::Expr { span, .. }, _) => (call_span, *span, "function", false),
            hir::ExprKind::MethodCall(path_segment, _, _, span) => {
                let ident_span = path_segment.ident.span;
                let ident_span = if let Some(args) = path_segment.args {
                    ident_span.with_hi(args.span_ext.hi())
                } else {
                    ident_span
                };
                (*span, ident_span, "method", true)
            }
            k => span_bug!(call_span, "checking argument types on a non-call: `{:?}`", k),
        };
        let args_span = error_span.trim_start(full_call_span).unwrap_or(error_span);

        // Don't print if it has error types or is just plain `_`
        fn has_error_or_infer<'tcx>(tys: impl IntoIterator<Item = Ty<'tcx>>) -> bool {
            tys.into_iter().any(|ty| ty.references_error() || ty.is_ty_var())
        }

        let tcx = self.tcx;
        // FIXME: taint after emitting errors and pass through an `ErrorGuaranteed`
        self.set_tainted_by_errors(
            tcx.sess.delay_span_bug(call_span, "no errors reported for args"),
        );

        // Get the argument span in the context of the call span so that
        // suggestions and labels are (more) correct when an arg is a
        // macro invocation.
        let normalize_span = |span: Span| -> Span {
            let normalized_span = span.find_ancestor_inside(error_span).unwrap_or(span);
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
                    .unwrap_or_else(|| tcx.ty_error_misc());
                (self.resolve_vars_if_possible(ty), normalize_span(expr.span))
            })
            .collect();
        let callee_expr = match &call_expr.peel_blocks().kind {
            hir::ExprKind::Call(callee, _) => Some(*callee),
            hir::ExprKind::MethodCall(_, receiver, ..) => {
                if let Some((DefKind::AssocFn, def_id)) =
                    self.typeck_results.borrow().type_dependent_def(call_expr.hir_id)
                    && let Some(assoc) = tcx.opt_associated_item(def_id)
                    && assoc.fn_has_self_parameter
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
            let can_coerce = self.can_coerce(arg_ty, coerced_ty);
            if !can_coerce {
                return Compatibility::Incompatible(Some(ty::error::TypeError::Sorts(
                    ty::error::ExpectedFound::new(true, coerced_ty, arg_ty),
                )));
            }

            // Using probe here, since we don't want this subtyping to affect inference.
            let subtyping_error = self.probe(|_| {
                self.at(&self.misc(arg_span), self.param_env)
                    .sup(DefineOpaqueTypes::No, formal_input_ty, coerced_ty)
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
            TypeTrace::types(&self.misc(span), true, mismatched_ty, provided_ty)
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
            compatibility_diagonal.iter().enumerate().find_map(|(i, c)| {
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
                formal_and_expected_inputs.get(mismatch_idx.into()).map(|tys| tys.1.kind())
                // If the tuple is unit, we're not actually wrapping any arguments.
                && !tys.is_empty()
                && provided_arg_tys.len() == formal_and_expected_inputs.len() - 1 + tys.len()
            {
                // Wrap up the N provided arguments starting at this position in a tuple.
                let provided_as_tuple = tcx.mk_tup_from_iter(
                    provided_arg_tys.iter().map(|(ty, _)| *ty).skip(mismatch_idx).take(tys.len()),
                );

                let mut satisfied = true;
                // Check if the newly wrapped tuple + rest of the arguments are compatible.
                for ((_, expected_ty), provided_ty) in std::iter::zip(
                    formal_and_expected_inputs.iter().skip(mismatch_idx),
                    [provided_as_tuple].into_iter().chain(
                        provided_arg_tys.iter().map(|(ty, _)| *ty).skip(mismatch_idx + tys.len()),
                    ),
                ) {
                    if !self.can_coerce(provided_ty, *expected_ty) {
                        satisfied = false;
                        break;
                    }
                }

                // If they're compatible, suggest wrapping in an arg, and we're done!
                // Take some care with spans, so we don't suggest wrapping a macro's
                // innards in parenthesis, for example.
                if satisfied
                    && let Some((_, lo)) =
                        provided_arg_tys.get(ProvidedIdx::from_usize(mismatch_idx))
                    && let Some((_, hi)) =
                        provided_arg_tys.get(ProvidedIdx::from_usize(mismatch_idx + tys.len() - 1))
                {
                    let mut err;
                    if tys.len() == 1 {
                        // A tuple wrap suggestion actually occurs within,
                        // so don't do anything special here.
                        err = self.err_ctxt().report_and_explain_type_error(
                            mk_trace(
                                *lo,
                                formal_and_expected_inputs[mismatch_idx.into()],
                                provided_arg_tys[mismatch_idx.into()].0,
                            ),
                            terr,
                        );
                        err.span_label(
                            full_call_span,
                            format!("arguments to this {} are incorrect", call_name),
                        );
                    } else {
                        err = tcx.sess.struct_span_err_with_code(
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
                            DiagnosticId::Error(err_code.to_owned()),
                        );
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
                        Some(mismatch_idx),
                        is_method,
                    );
                    err.emit();
                    return;
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
                tcx.sess.emit_err(errors::ArgMismatchIndeterminate { span: error_span });
            }
            return;
        }

        errors.drain_filter(|error| {
            let Error::Invalid(
                provided_idx,
                expected_idx,
                Compatibility::Incompatible(Some(e)),
            ) = error else { return false };
            let (provided_ty, provided_span) = provided_arg_tys[*provided_idx];
            let trace =
                mk_trace(provided_span, formal_and_expected_inputs[*expected_idx], provided_ty);
            if !matches!(trace.cause.as_failure_code(*e), FailureCode::Error0308) {
                self.err_ctxt().report_and_explain_type_error(trace, *e).emit();
                return true;
            }
            false
        });

        // We're done if we found errors, but we already emitted them.
        if errors.is_empty() {
            return;
        }

        // Okay, now that we've emitted the special errors separately, we
        // are only left missing/extra/swapped and mismatched arguments, both
        // can be collated pretty easily if needed.

        // Next special case: if there is only one "Incompatible" error, just emit that
        if let [
            Error::Invalid(provided_idx, expected_idx, Compatibility::Incompatible(Some(err))),
        ] = &errors[..]
        {
            let (formal_ty, expected_ty) = formal_and_expected_inputs[*expected_idx];
            let (provided_ty, provided_arg_span) = provided_arg_tys[*provided_idx];
            let trace = mk_trace(provided_arg_span, (formal_ty, expected_ty), provided_ty);
            let mut err = self.err_ctxt().report_and_explain_type_error(trace, *err);
            self.emit_coerce_suggestions(
                &mut err,
                &provided_args[*provided_idx],
                provided_ty,
                Expectation::rvalue_hint(self, expected_ty)
                    .only_has_type(self)
                    .unwrap_or(formal_ty),
                None,
                None,
            );
            err.span_label(
                full_call_span,
                format!("arguments to this {} are incorrect", call_name),
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

            // Call out where the function is defined
            self.label_fn_like(
                &mut err,
                fn_def_id,
                callee_ty,
                Some(expected_idx.as_usize()),
                is_method,
            );
            err.emit();
            return;
        }

        let mut err = if formal_and_expected_inputs.len() == provided_args.len() {
            struct_span_err!(
                tcx.sess,
                full_call_span,
                E0308,
                "arguments to this {} are incorrect",
                call_name,
            )
        } else {
            tcx.sess.struct_span_err_with_code(
                full_call_span,
                format!(
                    "this {} takes {}{} but {} {} supplied",
                    call_name,
                    if c_variadic { "at least " } else { "" },
                    potentially_plural_count(formal_and_expected_inputs.len(), "argument"),
                    potentially_plural_count(provided_args.len(), "argument"),
                    pluralize!("was", provided_args.len())
                ),
                DiagnosticId::Error(err_code.to_owned()),
            )
        };

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
                format!("/* {} */", ty)
            } else if let Some(fn_def_id) = fn_def_id
                && self.tcx.def_kind(fn_def_id).is_fn_like()
                && let self_implicit =
                    matches!(call_expr.kind, hir::ExprKind::MethodCall(..)) as usize
                && let Some(arg) = self.tcx.fn_arg_names(fn_def_id)
                    .get(expected_idx.as_usize() + self_implicit)
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
                                Some(trace.values),
                                e,
                                false,
                                true,
                            );
                        }
                    }

                    self.emit_coerce_suggestions(
                        &mut err,
                        &provided_args[provided_idx],
                        provided_ty,
                        Expectation::rvalue_hint(self, expected_ty)
                            .only_has_type(self)
                            .unwrap_or(formal_ty),
                        None,
                        None,
                    );
                }
                Error::Extra(arg_idx) => {
                    let (provided_ty, provided_span) = provided_arg_tys[arg_idx];
                    let provided_ty_name = if !has_error_or_infer([provided_ty]) {
                        // FIXME: not suggestable, use something else
                        format!(" of type `{}`", provided_ty)
                    } else {
                        "".to_string()
                    };
                    labels
                        .push((provided_span, format!("unexpected argument{}", provided_ty_name)));
                    let mut span = provided_span;
                    if span.can_be_used_for_suggestions() {
                        if arg_idx.index() > 0
                        && let Some((_, prev)) = provided_arg_tys
                            .get(ProvidedIdx::from_usize(arg_idx.index() - 1)
                        ) {
                            // Include previous comma
                            span = prev.shrink_to_hi().to(span);
                        }

                        // Is last argument for deletion in a row starting from the 0-th argument?
                        // Then delete the next comma, so we are not left with `f(, ...)`
                        //
                        //     fn f() {}
                        //   - f(0, 1,)
                        //   + f()
                        if only_extras_so_far
                            && errors
                                .peek()
                                .map_or(true, |next_error| !matches!(next_error, Error::Extra(_)))
                        {
                            let next = provided_arg_tys
                                .get(arg_idx + 1)
                                .map(|&(_, sp)| sp)
                                .unwrap_or_else(|| {
                                    // Subtract one to move before `)`
                                    call_expr.span.with_lo(call_expr.span.hi() - BytePos(1))
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
                    }
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
                            _ => unreachable!(),
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
                                format!(" of type `{}`", input_ty)
                            } else {
                                "".to_string()
                            };
                            labels.push((span, format!("an argument{} is missing", rendered)));
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
                                        " of type `{}` and `{}`",
                                        first_expected_ty, second_expected_ty
                                    )
                                } else {
                                    "".to_string()
                                };
                            labels.push((span, format!("two arguments{} are missing", rendered)));
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
                                    " of type `{}`, `{}`, and `{}`",
                                    first_expected_ty, second_expected_ty, third_expected_ty
                                )
                            } else {
                                "".to_string()
                            };
                            labels.push((span, format!("three arguments{} are missing", rendered)));
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
                        format!(", found `{}`", first_provided_ty)
                    } else {
                        String::new()
                    };
                    labels.push((
                        first_span,
                        format!("expected `{}`{}", first_expected_ty, first_provided_ty_name),
                    ));

                    let (second_provided_ty, second_span) = provided_arg_tys[second_provided_idx];
                    let (_, second_expected_ty) = formal_and_expected_inputs[second_expected_idx];
                    let second_provided_ty_name = if !has_error_or_infer([second_provided_ty]) {
                        format!(", found `{}`", second_provided_ty)
                    } else {
                        String::new()
                    };
                    labels.push((
                        second_span,
                        format!("expected `{}`{}", second_expected_ty, second_provided_ty_name),
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
                            format!(", found `{}`", provided_ty)
                        } else {
                            String::new()
                        };
                        labels.push((
                            provided_span,
                            format!("expected `{}`{}", expected_ty, provided_ty_name),
                        ));
                    }

                    suggestion_text = match suggestion_text {
                        SuggestionText::None => SuggestionText::Reorder,
                        _ => SuggestionText::DidYouMean,
                    };
                }
            }
        }

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
        self.label_fn_like(&mut err, fn_def_id, callee_ty, None, is_method);

        // And add a suggestion block for all of the parameters
        let suggestion_text = match suggestion_text {
            SuggestionText::None => None,
            SuggestionText::Provide(plural) => {
                Some(format!("provide the argument{}", if plural { "s" } else { "" }))
            }
            SuggestionText::Remove(plural) => {
                err.multipart_suggestion(
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
        if let Some(suggestion_text) = suggestion_text {
            let source_map = self.sess().source_map();
            let (mut suggestion, suggestion_span) =
                if let Some(call_span) = full_call_span.find_ancestor_inside(error_span) {
                    ("(".to_string(), call_span.shrink_to_hi().to(error_span.shrink_to_hi()))
                } else {
                    (
                        format!(
                            "{}(",
                            source_map.span_to_snippet(full_call_span).unwrap_or_else(|_| {
                                fn_def_id.map_or("".to_string(), |fn_def_id| {
                                    tcx.item_name(fn_def_id).to_string()
                                })
                            })
                        ),
                        error_span,
                    )
                };
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

        err.emit();
    }

    // AST fragment checking
    pub(in super::super) fn check_lit(
        &self,
        lit: &hir::Lit,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx;

        match lit.node {
            ast::LitKind::Str(..) => tcx.mk_static_str(),
            ast::LitKind::ByteStr(ref v, _) => {
                tcx.mk_imm_ref(tcx.lifetimes.re_static, tcx.mk_array(tcx.types.u8, v.len() as u64))
            }
            ast::LitKind::Byte(_) => tcx.types.u8,
            ast::LitKind::Char(_) => tcx.types.char,
            ast::LitKind::Int(_, ast::LitIntType::Signed(t)) => tcx.mk_mach_int(ty::int_ty(t)),
            ast::LitKind::Int(_, ast::LitIntType::Unsigned(t)) => tcx.mk_mach_uint(ty::uint_ty(t)),
            ast::LitKind::Int(_, ast::LitIntType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| match ty.kind() {
                    ty::Int(_) | ty::Uint(_) => Some(ty),
                    ty::Char => Some(tcx.types.u8),
                    ty::RawPtr(..) => Some(tcx.types.usize),
                    ty::FnDef(..) | ty::FnPtr(_) => Some(tcx.types.usize),
                    _ => None,
                });
                opt_ty.unwrap_or_else(|| self.next_int_var())
            }
            ast::LitKind::Float(_, ast::LitFloatType::Suffixed(t)) => {
                tcx.mk_mach_float(ty::float_ty(t))
            }
            ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| match ty.kind() {
                    ty::Float(_) => Some(ty),
                    _ => None,
                });
                opt_ty.unwrap_or_else(|| self.next_float_var())
            }
            ast::LitKind::Bool(_) => tcx.types.bool,
            ast::LitKind::CStr(_, _) => tcx.mk_imm_ref(
                tcx.lifetimes.re_static,
                tcx.type_of(tcx.require_lang_item(hir::LangItem::CStr, Some(lit.span)))
                    .skip_binder(),
            ),
            ast::LitKind::Err => tcx.ty_error_misc(),
        }
    }

    pub fn check_struct_path(
        &self,
        qpath: &QPath<'_>,
        hir_id: hir::HirId,
    ) -> Result<(&'tcx ty::VariantDef, Ty<'tcx>), ErrorGuaranteed> {
        let path_span = qpath.span();
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, hir_id);
        let variant = match def {
            Res::Err => {
                let guar =
                    self.tcx.sess.delay_span_bug(path_span, "`Res::Err` but no error emitted");
                self.set_tainted_by_errors(guar);
                return Err(guar);
            }
            Res::Def(DefKind::Variant, _) => match ty.normalized.ty_adt_def() {
                Some(adt) => {
                    Some((adt.variant_of_res(def), adt.did(), Self::user_substs_for_adt(ty)))
                }
                _ => bug!("unexpected type: {:?}", ty.normalized),
            },
            Res::Def(DefKind::Struct | DefKind::Union | DefKind::TyAlias | DefKind::AssocTy, _)
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. } => match ty.normalized.ty_adt_def() {
                Some(adt) if !adt.is_enum() => {
                    Some((adt.non_enum_variant(), adt.did(), Self::user_substs_for_adt(ty)))
                }
                _ => None,
            },
            _ => bug!("unexpected definition: {:?}", def),
        };

        if let Some((variant, did, ty::UserSubsts { substs, user_self_ty })) = variant {
            debug!("check_struct_path: did={:?} substs={:?}", did, substs);

            // Register type annotation.
            self.write_user_type_annotation_from_substs(hir_id, did, substs, user_self_ty);

            // Check bounds on type arguments used in the path.
            self.add_required_obligations_for_hir(path_span, did, substs, hir_id);

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
                _ => struct_span_err!(
                    self.tcx.sess,
                    path_span,
                    E0071,
                    "expected struct, variant or union type, found {}",
                    ty.normalized.sort_string(self.tcx)
                )
                .span_label(path_span, "not a struct")
                .emit(),
            })
        }
    }

    pub fn check_decl_initializer(
        &self,
        hir_id: hir::HirId,
        pat: &'tcx hir::Pat<'tcx>,
        init: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // FIXME(tschottdorf): `contains_explicit_ref_binding()` must be removed
        // for #42640 (default match binding modes).
        //
        // See #44848.
        let ref_bindings = pat.contains_explicit_ref_binding();

        let local_ty = self.local_ty(init.span, hir_id).revealed_ty;
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
            self.demand_eqtype(init.span, local_ty, init_ty);
            init_ty
        } else {
            self.check_expr_coercible_to_type(init, local_ty, None)
        }
    }

    pub(in super::super) fn check_decl(&self, decl: Declaration<'tcx>) {
        // Determine and write the type which we'll check the pattern against.
        let decl_ty = self.local_ty(decl.span, decl.hir_id).decl_ty;
        self.write_ty(decl.hir_id, decl_ty);

        // Type check the initializer.
        if let Some(ref init) = decl.init {
            let init_ty = self.check_decl_initializer(decl.hir_id, decl.pat, &init);
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
        self.check_pat_top(&decl.pat, decl_ty, ty_span, origin_expr);
        let pat_ty = self.node_ty(decl.pat.hir_id);
        self.overwrite_local_ty_if_err(decl.hir_id, decl.pat, pat_ty);

        if let Some(blk) = decl.els {
            let previous_diverges = self.diverges.get();
            let else_ty = self.check_block_with_expected(blk, NoExpectation);
            let cause = self.cause(blk.span, ObligationCauseCode::LetElse);
            if let Some(mut err) =
                self.demand_eqtype_with_origin(&cause, self.tcx.types.never, else_ty)
            {
                err.emit();
            }
            self.diverges.set(previous_diverges);
        }
    }

    /// Type check a `let` statement.
    pub fn check_decl_local(&self, local: &'tcx hir::Local<'tcx>) {
        self.check_decl(local.into());
    }

    pub fn check_stmt(&self, stmt: &'tcx hir::Stmt<'tcx>, is_last: bool) {
        // Don't do all the complex logic below for `DeclItem`.
        match stmt.kind {
            hir::StmtKind::Item(..) => return,
            hir::StmtKind::Local(..) | hir::StmtKind::Expr(..) | hir::StmtKind::Semi(..) => {}
        }

        self.warn_if_unreachable(stmt.hir_id, stmt.span, "statement");

        // Hide the outer diverging and `has_errors` flags.
        let old_diverges = self.diverges.replace(Diverges::Maybe);

        match stmt.kind {
            hir::StmtKind::Local(l) => {
                self.check_decl_local(l);
            }
            // Ignore for now.
            hir::StmtKind::Item(_) => {}
            hir::StmtKind::Expr(ref expr) => {
                // Check with expected type of `()`.
                self.check_expr_has_type_or_error(&expr, self.tcx.mk_unit(), |err| {
                    if expr.can_have_side_effects() {
                        self.suggest_semicolon_at_end(expr.span, err);
                    }
                });
            }
            hir::StmtKind::Semi(ref expr) => {
                // All of this is equivalent to calling `check_expr`, but it is inlined out here
                // in order to capture the fact that this `match` is the last statement in its
                // function. This is done for better suggestions to remove the `;`.
                let expectation = match expr.kind {
                    hir::ExprKind::Match(..) if is_last => IsLast(stmt.span),
                    _ => NoExpectation,
                };
                self.check_expr_with_expectation(expr, expectation);
            }
        }

        // Combine the diverging and `has_error` flags.
        self.diverges.set(self.diverges.get() | old_diverges);
    }

    pub fn check_block_no_value(&self, blk: &'tcx hir::Block<'tcx>) {
        let unit = self.tcx.mk_unit();
        let ty = self.check_block_with_expected(blk, ExpectHasType(unit));

        // if the block produces a `!` value, that can always be
        // (effectively) coerced to unit.
        if !ty.is_never() {
            self.demand_suptype(blk.span, unit, ty);
        }
    }

    pub(in super::super) fn check_block_with_expected(
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
            for (pos, s) in blk.stmts.iter().enumerate() {
                self.check_stmt(s, blk.stmts.len() - 1 == pos);
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
                let cause = self.cause(span, ObligationCauseCode::BlockTailExpression(blk.hir_id));
                let ty_for_diagnostic = coerce.merged_ty();
                // We use coerce_inner here because we want to augment the error
                // suggesting to wrap the block in square brackets if it might've
                // been mistaken array syntax
                coerce.coerce_inner(
                    self,
                    &cause,
                    Some(tail_expr),
                    tail_expr_ty,
                    Some(&mut |diag: &mut Diagnostic| {
                        self.suggest_block_to_brackets(diag, blk, tail_expr_ty, ty_for_diagnostic);
                    }),
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
                if !self.diverges.get().is_always() {
                    // #50009 -- Do not point at the entire fn block span, point at the return type
                    // span, as it is the cause of the requirement, and
                    // `consider_hint_about_removing_semicolon` will point at the last expression
                    // if it were a relevant part of the error. This improves usability in editors
                    // that highlight errors inline.
                    let mut sp = blk.span;
                    let mut fn_span = None;
                    if let Some((decl, ident)) = self.get_parent_fn_decl(blk.hir_id) {
                        let ret_sp = decl.output.span();
                        if let Some(block_sp) = self.parent_item_span(blk.hir_id) {
                            // HACK: on some cases (`ui/liveness/liveness-issue-2163.rs`) the
                            // output would otherwise be incorrect and even misleading. Make sure
                            // the span we're aiming at correspond to a `fn` body.
                            if block_sp == blk.span {
                                sp = ret_sp;
                                fn_span = Some(ident.span);
                            }
                        }
                    }
                    coerce.coerce_forced_unit(
                        self,
                        &self.misc(sp),
                        &mut |err| {
                            if let Some(expected_ty) = expected.only_has_type(self) {
                                if blk.stmts.is_empty() && blk.expr.is_none() {
                                    self.suggest_boxing_when_appropriate(
                                        err,
                                        blk.span,
                                        blk.hir_id,
                                        expected_ty,
                                        self.tcx.mk_unit(),
                                    );
                                }
                                if !self.consider_removing_semicolon(blk, expected_ty, err) {
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
                                                        hir::StmtKind::Local(hir::Local {
                                                            source:
                                                                hir::LocalSource::AssignDesugar(_),
                                                            ..
                                                        }),
                                                    ..
                                                },
                                                hir::Stmt {
                                                    kind:
                                                        hir::StmtKind::Expr(hir::Expr {
                                                            kind: hir::ExprKind::Assign(..),
                                                            ..
                                                        }),
                                                    ..
                                                },
                                            ],
                                        ..
                                    } = blk
                                    {
                                        self.comes_from_while_condition(blk.hir_id, |_| {
                                            err.downgrade_to_delayed_bug();
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

    fn parent_item_span(&self, id: hir::HirId) -> Option<Span> {
        let node = self.tcx.hir().get_by_def_id(self.tcx.hir().get_parent_item(id).def_id);
        match node {
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn(_, _, body_id), .. })
            | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(_, body_id), .. }) => {
                let body = self.tcx.hir().body(body_id);
                if let ExprKind::Block(block, _) = &body.value.kind {
                    return Some(block.span);
                }
            }
            _ => {}
        }
        None
    }

    /// Given a function block's `HirId`, returns its `FnDecl` if it exists, or `None` otherwise.
    fn get_parent_fn_decl(&self, blk_id: hir::HirId) -> Option<(&'tcx hir::FnDecl<'tcx>, Ident)> {
        let parent = self.tcx.hir().get_by_def_id(self.tcx.hir().get_parent_item(blk_id).def_id);
        self.get_node_fn_decl(parent).map(|(_, fn_decl, ident, _)| (fn_decl, ident))
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

    fn overwrite_local_ty_if_err(
        &self,
        hir_id: hir::HirId,
        pat: &'tcx hir::Pat<'tcx>,
        ty: Ty<'tcx>,
    ) {
        if let Err(guar) = ty.error_reported() {
            // Override the types everywhere with `err()` to avoid knock on errors.
            let err = self.tcx.ty_error(guar);
            self.write_ty(hir_id, err);
            self.write_ty(pat.hir_id, err);
            let local_ty = LocalTy { decl_ty: err, revealed_ty: err };
            self.locals.borrow_mut().insert(hir_id, local_ty);
            self.locals.borrow_mut().insert(pat.hir_id, local_ty);
        }
    }

    // Finish resolving a path in a struct expression or pattern `S::A { .. }` if necessary.
    // The newly resolved definition is written into `type_dependent_defs`.
    fn finish_resolving_struct_path(
        &self,
        qpath: &QPath<'_>,
        path_span: Span,
        hir_id: hir::HirId,
    ) -> (Res, RawTy<'tcx>) {
        match *qpath {
            QPath::Resolved(ref maybe_qself, ref path) => {
                let self_ty = maybe_qself.as_ref().map(|qself| self.to_ty(qself).raw);
                let ty = self.astconv().res_to_ty(self_ty, path, hir_id, true);
                (path.res, self.handle_raw_ty(path_span, ty))
            }
            QPath::TypeRelative(ref qself, ref segment) => {
                let ty = self.to_ty(qself);

                let result = self
                    .astconv()
                    .associated_path_to_ty(hir_id, path_span, ty.raw, qself, segment, true);
                let ty =
                    result.map(|(ty, _, _)| ty).unwrap_or_else(|guar| self.tcx().ty_error(guar));
                let ty = self.handle_raw_ty(path_span, ty);
                let result = result.map(|(_, kind, def_id)| (kind, def_id));

                // Write back the new resolution.
                self.write_resolution(hir_id, result);

                (result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)), ty)
            }
            QPath::LangItem(lang_item, span, id) => {
                let (res, ty) = self.resolve_lang_item_path(lang_item, span, hir_id, id);
                (res, self.handle_raw_ty(path_span, ty))
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
                // Store both the predicate and the predicate *without constness*
                // since sometimes we instantiate and check both of these in a
                // method call, for example.
                remap_cause.insert((
                    before_span,
                    error.obligation.predicate,
                    error.obligation.cause.clone(),
                ));
                remap_cause.insert((
                    before_span,
                    error.obligation.predicate.without_const(self.tcx),
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
        // line back up with a useful ExprBindingObligation.
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
        err: &mut Diagnostic,
        callable_def_id: Option<DefId>,
        callee_ty: Option<Ty<'tcx>>,
        // A specific argument should be labeled, instead of all of them
        expected_idx: Option<usize>,
        is_method: bool,
    ) {
        let Some(mut def_id) = callable_def_id else {
            return;
        };

        if let Some(assoc_item) = self.tcx.opt_associated_item(def_id)
            // Possibly points at either impl or trait item, so try to get it
            // to point to trait item, then get the parent.
            // This parent might be an impl in the case of an inherent function,
            // but the next check will fail.
            && let maybe_trait_item_def_id = assoc_item.trait_item_def_id.unwrap_or(def_id)
            && let maybe_trait_def_id = self.tcx.parent(maybe_trait_item_def_id)
            // Just an easy way to check "trait_def_id == Fn/FnMut/FnOnce"
            && let Some(call_kind) = self.tcx.fn_trait_kind_from_def_id(maybe_trait_def_id)
            && let Some(callee_ty) = callee_ty
        {
            let callee_ty = callee_ty.peel_refs();
            match *callee_ty.kind() {
                ty::Param(param) => {
                    let param =
                        self.tcx.generics_of(self.body_id).type_param(&param, self.tcx);
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
                        for (predicate, span) in instantiated
                        {
                            if let ty::PredicateKind::Clause(ty::Clause::Trait(pred)) = predicate.kind().skip_binder()
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
                        let trait_ref = ty::TraitRef::new(self.tcx,
                            call_kind.to_def_id(self.tcx),
                            [
                                callee_ty,
                                self.next_ty_var(TypeVariableOrigin {
                                    kind: TypeVariableOriginKind::MiscVariable,
                                    span: rustc_span::DUMMY_SP,
                                }),
                            ],
                        );
                        let obligation = traits::Obligation::new(
                            self.tcx,
                            traits::ObligationCause::dummy(),
                            self.param_env,
                            ty::Binder::dummy(trait_ref),
                        );
                        match SelectionContext::new(&self).select(&obligation) {
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

        if let Some(def_span) = self.tcx.def_ident_span(def_id) && !def_span.is_dummy() {
            let mut spans: MultiSpan = def_span.into();

            let params = self
                .tcx
                .hir()
                .get_if_local(def_id)
                .and_then(|node| node.body_id())
                .into_iter()
                .flat_map(|id| self.tcx.hir().body(id).params)
                .skip(if is_method { 1 } else { 0 });

            for (_, param) in params
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| expected_idx.map_or(true, |expected_idx| expected_idx == *idx))
            {
                spans.push_span_label(param.span, "");
            }

            err.span_note(spans, format!("{} defined here", self.tcx.def_descr(def_id)));
        } else if let Some(hir::Node::Expr(e)) = self.tcx.hir().get_if_local(def_id)
            && let hir::ExprKind::Closure(hir::Closure { body, .. }) = &e.kind
        {
            let param = expected_idx
                .and_then(|expected_idx| self.tcx.hir().body(*body).params.get(expected_idx));
            let (kind, span) = if let Some(param) = param {
                ("closure parameter", param.span)
            } else {
                ("closure", self.tcx.def_span(def_id))
            };
            err.span_note(span, format!("{} defined here", kind));
        } else {
            err.span_note(
                self.tcx.def_span(def_id),
                format!("{} defined here", self.tcx.def_descr(def_id)),
            );
        }
    }
}
