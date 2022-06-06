use crate::astconv::AstConv;
use crate::check::coercion::CoerceMany;
use crate::check::fn_ctxt::arg_matrix::{ArgMatrix, Compatibility, Error};
use crate::check::gather_locals::Declaration;
use crate::check::method::MethodCallee;
use crate::check::Expectation::*;
use crate::check::TupleArgumentsFlag::*;
use crate::check::{
    potentially_plural_count, struct_span_err, BreakableCtxt, Diverges, Expectation, FnCtxt,
    LocalTy, Needs, TupleArgumentsFlag,
};
use crate::structured_errors::StructuredDiagnostic;

use rustc_ast as ast;
use rustc_errors::{Applicability, Diagnostic, DiagnosticId, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_infer::infer::error_reporting::{FailureCode, ObligationCauseExt};
use rustc_infer::infer::InferOk;
use rustc_infer::infer::TypeTrace;
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::{self, Span};
use rustc_trait_selection::traits::{self, ObligationCauseCode, StatementAsExpression};

use std::iter;
use std::slice;

enum TupleMatchFound {
    None,
    Single,
    /// Beginning and end Span
    Multiple(Span, Span),
}
impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(in super::super) fn check_casts(&self) {
        let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
        debug!("FnCtxt::check_casts: {} deferred checks", deferred_cast_checks.len());
        for cast in deferred_cast_checks.drain(..) {
            cast.check(self);
        }
    }

    pub(in super::super) fn check_transmutes(&self) {
        let mut deferred_transmute_checks = self.deferred_transmute_checks.borrow_mut();
        debug!("FnCtxt::check_transmutes: {} deferred checks", deferred_transmute_checks.len());
        for (from, to, span) in deferred_transmute_checks.drain(..) {
            self.check_transmute(span, from, to);
        }
    }

    pub(in super::super) fn check_asms(&self) {
        let mut deferred_asm_checks = self.deferred_asm_checks.borrow_mut();
        debug!("FnCtxt::check_asm: {} deferred checks", deferred_asm_checks.len());
        for (asm, hir_id) in deferred_asm_checks.drain(..) {
            let enclosing_id = self.tcx.hir().enclosing_body_owner(hir_id);
            self.check_asm(asm, enclosing_id);
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
                TupleArguments => vec![self.tcx.intern_tup(&err_inputs)],
            };

            self.check_argument_types(
                sp,
                expr,
                &err_inputs,
                None,
                args_no_rcvr,
                false,
                tuple_arguments,
                None,
            );
            return self.tcx.ty_error();
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

        // Conceptually, we've got some number of expected inputs, and some number of provided aguments
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

        // We'll also want to keep track of the fully coerced argument types, for an awkward hack near the end
        let mut final_arg_types: Vec<Option<(Ty<'_>, Ty<'_>)>> = vec![None; provided_arg_count];

        // We introduce a helper function to demand that a given argument satisfy a given input
        // This is more complicated than just checking type equality, as arguments could be coerced
        // This version writes those types back so further type checking uses the narrowed types
        let demand_compatible = |idx, final_arg_types: &mut Vec<Option<(Ty<'tcx>, Ty<'tcx>)>>| {
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

            // Keep track of these for below
            final_arg_types[idx] = Some((checked_ty, coerced_ty));

            // Cause selection errors caused by resolving a single argument to point at the
            // argument and not the call. This lets us customize the span pointed to in the
            // fulfillment error to be more accurate.
            let coerced_ty =
                self.resolve_vars_with_obligations_and_mutate_fulfillment(coerced_ty, |errors| {
                    self.point_at_type_arg_instead_of_call_if_possible(errors, call_expr);
                    self.point_at_arg_instead_of_call_if_possible(
                        errors,
                        &final_arg_types,
                        call_expr,
                        call_span,
                        provided_args,
                    );
                });

            // Make sure we store the resolved type
            final_arg_types[idx] = Some((checked_ty, coerced_ty));

            let coerce_error = self
                .try_coerce(provided_arg, checked_ty, coerced_ty, AllowTwoPhase::Yes, None)
                .err();

            if coerce_error.is_some() {
                return Compatibility::Incompatible(coerce_error);
            }

            // 3. Check if the formal type is a supertype of the checked one
            //    and register any such obligations for future type checks
            let supertype_error = self
                .at(&self.misc(provided_arg.span), self.param_env)
                .sup(formal_input_ty, coerced_ty);
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

        // A "softer" version of the helper above, which checks types without persisting them,
        // and treats error types differently
        // This will allow us to "probe" for other argument orders that would likely have been correct
        let check_compatible = |input_idx, arg_idx| {
            let formal_input_ty: Ty<'tcx> = formal_input_tys[arg_idx];
            let expected_input_ty: Ty<'tcx> = expected_input_tys[arg_idx];

            // If either is an error type, we defy the usual convention and consider them to *not* be
            // coercible.  This prevents our error message heuristic from trying to pass errors into
            // every argument.
            if formal_input_ty.references_error() || expected_input_ty.references_error() {
                return Compatibility::Incompatible(None);
            }

            let provided_arg: &hir::Expr<'tcx> = &provided_args[input_idx];
            let expectation = Expectation::rvalue_hint(self, expected_input_ty);
            // FIXME: check that this is safe; I don't believe this commits any of the obligations, but I can't be sure.
            //
            //   I had another method of "soft" type checking before,
            //   but it was failing to find the type of some expressions (like "")
            //   so I prodded this method and made it pub(super) so I could call it, and it seems to work well.
            let checked_ty = self.check_expr_kind(provided_arg, expectation);

            let coerced_ty = expectation.only_has_type(self).unwrap_or(formal_input_ty);
            let can_coerce = self.can_coerce(checked_ty, coerced_ty);

            if !can_coerce {
                return Compatibility::Incompatible(None);
            }

            let subtyping_result = self
                .at(&self.misc(provided_arg.span), self.param_env)
                .sup(formal_input_ty, coerced_ty);

            // Same as above: if either the coerce type or the checked type is an error type,
            // consider them *not* compatible.
            let coercible =
                !coerced_ty.references_error() && !checked_ty.references_error() && can_coerce;

            match (coercible, &subtyping_result) {
                (true, Ok(_)) => Compatibility::Compatible,
                _ => Compatibility::Incompatible(subtyping_result.err()),
            }
        };

        // To start, we only care "along the diagonal", where we expect every
        // provided arg to be in the right spot
        let mut compatibility = vec![Compatibility::Incompatible(None); provided_args.len()];

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
                self.select_obligations_where_possible(false, |errors| {
                    self.point_at_type_arg_instead_of_call_if_possible(errors, call_expr);
                    self.point_at_arg_instead_of_call_if_possible(
                        errors,
                        &final_arg_types,
                        call_expr,
                        call_span,
                        &provided_args,
                    );
                })
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

                let is_closure = matches!(arg.kind, ExprKind::Closure(..));
                if is_closure != check_closures {
                    continue;
                }

                let compatible = demand_compatible(idx, &mut final_arg_types);
                let is_compatible = matches!(compatible, Compatibility::Compatible);
                compatibility[idx] = compatible;

                if !is_compatible {
                    call_appears_satisfied = false;
                }
            }
        }

        // Logic here is a bit hairy
        'errors: {
            // If something above didn't typecheck, we've fallen off the happy path
            // and we should make some effort to provide better error messages
            if call_appears_satisfied {
                break 'errors;
            }

            self.set_tainted_by_errors();

            // The algorithm here is inspired by levenshtein distance and longest common subsequence.
            // We'll try to detect 4 different types of mistakes:
            // - An extra parameter has been provided that doesn't satisfy *any* of the other inputs
            // - An input is missing, which isn't satisfied by *any* of the other arguments
            // - Some number of arguments have been provided in the wrong order
            // - A type is straight up invalid

            // First, let's find the errors
            let mut compatibility: Vec<_> = compatibility.into_iter().map(Some).collect();
            let (mut errors, matched_inputs) =
                ArgMatrix::new(minimum_input_count, provided_arg_count, |i, j| {
                    if i == j { compatibility[i].take().unwrap() } else { check_compatible(i, j) }
                })
                .find_errors();

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

            let found_errors = !errors.is_empty();

            errors.drain_filter(|error| {
                let Error::Invalid(input_idx, arg_idx, Compatibility::Incompatible(error)) = error else { return false };
                let expected_ty = expected_input_tys[*arg_idx];
                let provided_ty = final_arg_types[*input_idx].map(|ty| ty.0).unwrap();
                let cause = &self.misc(provided_args[*input_idx].span);
                let trace = TypeTrace::types(cause, true, expected_ty, provided_ty);
                if let Some(e) = error {
                    if !matches!(trace.cause.as_failure_code(e), FailureCode::Error0308(_)) {
                        self.report_and_explain_type_error(trace, e).emit();
                        return true;
                    }
                }
                false
            });

            // We're done if we found errors, but we already emitted them.
            // I don't think we *should* be able to enter this bit of code
            // (`!call_appears_satisfied`) without *also* finding errors, but we
            // don't want to accidentally not emit an error if there is some
            // logic bug in the `ArgMatrix` code.
            if found_errors && errors.is_empty() {
                break 'errors;
            }

            // Next, let's construct the error
            let (error_span, full_call_span, ctor_of) = match &call_expr.kind {
                hir::ExprKind::Call(
                    hir::Expr {
                        span,
                        kind:
                            hir::ExprKind::Path(hir::QPath::Resolved(
                                _,
                                hir::Path { res: Res::Def(DefKind::Ctor(of, _), _), .. },
                            )),
                        ..
                    },
                    _,
                ) => (call_span, *span, Some(of)),
                hir::ExprKind::Call(hir::Expr { span, .. }, _) => (call_span, *span, None),
                hir::ExprKind::MethodCall(path_segment, _, span) => {
                    let ident_span = path_segment.ident.span;
                    let ident_span = if let Some(args) = path_segment.args {
                        ident_span.with_hi(args.span_ext.hi())
                    } else {
                        ident_span
                    };
                    (
                        *span, ident_span, None, // methods are never ctors
                    )
                }
                k => span_bug!(call_span, "checking argument types on a non-call: `{:?}`", k),
            };
            let args_span = error_span.trim_start(full_call_span).unwrap_or(error_span);
            let call_name = match ctor_of {
                Some(CtorOf::Struct) => "struct",
                Some(CtorOf::Variant) => "enum variant",
                None => "function",
            };
            if c_variadic && provided_arg_count < minimum_input_count {
                err_code = "E0060";
            }

            // Next special case: The case where we expect a single tuple and
            // wrapping all the args in parentheses (or adding a comma to
            // already existing parentheses) will result in a tuple that
            // satisfies the call.
            // This isn't super ideal code, because we copy code from elsewhere
            // and somewhat duplicate this. We also delegate to the general type
            // mismatch suggestions for the single arg case.
            let sugg_tuple_wrap_args =
                self.suggested_tuple_wrap(&expected_input_tys, provided_args);
            match sugg_tuple_wrap_args {
                TupleMatchFound::None => {}
                TupleMatchFound::Single => {
                    let expected_ty = expected_input_tys[0];
                    let provided_ty = final_arg_types[0].map(|ty| ty.0).unwrap();
                    let expected_ty = self.resolve_vars_if_possible(expected_ty);
                    let provided_ty = self.resolve_vars_if_possible(provided_ty);
                    let cause = &self.misc(provided_args[0].span);
                    let compatibility = demand_compatible(0, &mut final_arg_types);
                    let type_error = match compatibility {
                        Compatibility::Incompatible(Some(error)) => error,
                        _ => TypeError::Mismatch,
                    };
                    let trace = TypeTrace::types(cause, true, expected_ty, provided_ty);
                    let mut err = self.report_and_explain_type_error(trace, &type_error);
                    self.emit_coerce_suggestions(
                        &mut err,
                        &provided_args[0],
                        final_arg_types[0].map(|ty| ty.0).unwrap(),
                        final_arg_types[0].map(|ty| ty.1).unwrap(),
                        None,
                        None,
                    );
                    err.span_label(
                        full_call_span,
                        format!("arguments to this {} are incorrect", call_name),
                    );
                    // Call out where the function is defined
                    label_fn_like(tcx, &mut err, fn_def_id);
                    err.emit();
                    break 'errors;
                }
                TupleMatchFound::Multiple(start, end) => {
                    let mut err = tcx.sess.struct_span_err_with_code(
                        full_call_span,
                        &format!(
                            "this {} takes {}{} but {} {} supplied",
                            call_name,
                            if c_variadic { "at least " } else { "" },
                            potentially_plural_count(minimum_input_count, "argument"),
                            potentially_plural_count(provided_arg_count, "argument"),
                            if provided_arg_count == 1 { "was" } else { "were" }
                        ),
                        DiagnosticId::Error(err_code.to_owned()),
                    );
                    // Call out where the function is defined
                    label_fn_like(tcx, &mut err, fn_def_id);
                    err.multipart_suggestion(
                        "use parentheses to construct a tuple",
                        vec![(start, '('.to_string()), (end, ')'.to_string())],
                        Applicability::MachineApplicable,
                    );
                    err.emit();
                    break 'errors;
                }
            }

            // Okay, now that we've emitted the special errors separately, we
            // are only left missing/extra/swapped and mismatched arguments, both
            // can be collated pretty easily if needed.

            // Next special case: if there is only one "Incompatible" error, just emit that
            if errors.len() == 1 {
                if let Some(Error::Invalid(
                    input_idx,
                    arg_idx,
                    Compatibility::Incompatible(Some(error)),
                )) = errors.iter().next()
                {
                    let expected_ty = expected_input_tys[*arg_idx];
                    let provided_ty = final_arg_types[*arg_idx].map(|ty| ty.0).unwrap();
                    let expected_ty = self.resolve_vars_if_possible(expected_ty);
                    let provided_ty = self.resolve_vars_if_possible(provided_ty);
                    let cause = &self.misc(provided_args[*input_idx].span);
                    let trace = TypeTrace::types(cause, true, expected_ty, provided_ty);
                    let mut err = self.report_and_explain_type_error(trace, error);
                    self.emit_coerce_suggestions(
                        &mut err,
                        &provided_args[*input_idx],
                        provided_ty,
                        final_arg_types[*input_idx].map(|ty| ty.1).unwrap(),
                        None,
                        None,
                    );
                    err.span_label(
                        full_call_span,
                        format!("arguments to this {} are incorrect", call_name),
                    );
                    // Call out where the function is defined
                    label_fn_like(tcx, &mut err, fn_def_id);
                    err.emit();
                    break 'errors;
                }
            }

            let mut err = if minimum_input_count == provided_arg_count {
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
                    &format!(
                        "this {} takes {}{} but {} {} supplied",
                        call_name,
                        if c_variadic { "at least " } else { "" },
                        potentially_plural_count(minimum_input_count, "argument"),
                        potentially_plural_count(provided_arg_count, "argument"),
                        if provided_arg_count == 1 { "was" } else { "were" }
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

            let mut errors = errors.into_iter().peekable();
            while let Some(error) = errors.next() {
                match error {
                    Error::Invalid(input_idx, arg_idx, compatibility) => {
                        let expected_ty = expected_input_tys[arg_idx];
                        let provided_ty = final_arg_types[input_idx].map(|ty| ty.0).unwrap();
                        let expected_ty = self.resolve_vars_if_possible(expected_ty);
                        let provided_ty = self.resolve_vars_if_possible(provided_ty);
                        if let Compatibility::Incompatible(error) = &compatibility {
                            let cause = &self.misc(provided_args[input_idx].span);
                            let trace = TypeTrace::types(cause, true, expected_ty, provided_ty);
                            if let Some(e) = error {
                                self.note_type_err(
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
                            &provided_args[input_idx],
                            final_arg_types[input_idx].map(|ty| ty.0).unwrap(),
                            final_arg_types[input_idx].map(|ty| ty.1).unwrap(),
                            None,
                            None,
                        );
                    }
                    Error::Extra(arg_idx) => {
                        let arg_type = if let Some((_, ty)) = final_arg_types[arg_idx] {
                            if ty.references_error() || ty.has_infer_types() {
                                "".into()
                            } else {
                                format!(" of type `{}`", ty)
                            }
                        } else {
                            "".into()
                        };
                        labels.push((
                            provided_args[arg_idx].span,
                            format!("argument{} unexpected", arg_type),
                        ));
                        suggestion_text = match suggestion_text {
                            SuggestionText::None => SuggestionText::Remove(false),
                            SuggestionText::Remove(_) => SuggestionText::Remove(true),
                            _ => SuggestionText::DidYouMean,
                        };
                    }
                    Error::Missing(input_idx) => {
                        // If there are multiple missing arguments adjacent to each other,
                        // then we can provide a single error.

                        let mut missing_idxs = vec![input_idx];
                        while let Some(e) = errors.next_if(|e| matches!(e, Error::Missing(input_idx) if *input_idx == (missing_idxs.last().unwrap() + 1))) {
                            match e {
                                Error::Missing(input_idx) => missing_idxs.push(input_idx),
                                _ => unreachable!(),
                            }
                        }

                        // NOTE: Because we might be re-arranging arguments, might have extra
                        // arguments, etc. it's hard to *really* know where we should provide
                        // this error label, so as a heuristic, we point to the provided arg, or
                        // to the call if the missing inputs pass the provided args.
                        match &missing_idxs[..] {
                            &[input_idx] => {
                                let expected_ty = expected_input_tys[input_idx];
                                let input_ty = self.resolve_vars_if_possible(expected_ty);
                                let span = if input_idx < provided_arg_count {
                                    let arg_span = provided_args[input_idx].span;
                                    Span::new(arg_span.lo(), arg_span.hi(), arg_span.ctxt(), None)
                                } else {
                                    args_span
                                };
                                let arg_type =
                                    if input_ty.references_error() || input_ty.has_infer_types() {
                                        "".into()
                                    } else {
                                        format!(" of type `{}`", input_ty)
                                    };
                                labels.push((span, format!("an argument{} is missing", arg_type)));
                                suggestion_text = match suggestion_text {
                                    SuggestionText::None => SuggestionText::Provide(false),
                                    SuggestionText::Provide(_) => SuggestionText::Provide(true),
                                    _ => SuggestionText::DidYouMean,
                                };
                            }
                            &[first_idx, second_idx] => {
                                let first_input_ty =
                                    self.resolve_vars_if_possible(expected_input_tys[first_idx]);
                                let second_input_ty =
                                    self.resolve_vars_if_possible(expected_input_tys[second_idx]);

                                let span = if second_idx < provided_arg_count {
                                    let first_arg_span = provided_args[first_idx].span;
                                    let second_arg_span = provided_args[second_idx].span;
                                    Span::new(
                                        first_arg_span.lo(),
                                        second_arg_span.hi(),
                                        first_arg_span.ctxt(),
                                        None,
                                    )
                                } else {
                                    args_span
                                };
                                let any_unnameable = false
                                    || first_input_ty.references_error()
                                    || first_input_ty.has_infer_types()
                                    || second_input_ty.references_error()
                                    || second_input_ty.has_infer_types();
                                let arg_type = if any_unnameable {
                                    "".into()
                                } else {
                                    format!(
                                        " of type `{}` and `{}`",
                                        first_input_ty, second_input_ty
                                    )
                                };
                                labels
                                    .push((span, format!("two arguments{} are missing", arg_type)));
                                suggestion_text = match suggestion_text {
                                    SuggestionText::None | SuggestionText::Provide(_) => {
                                        SuggestionText::Provide(true)
                                    }
                                    _ => SuggestionText::DidYouMean,
                                };
                            }
                            &[first_idx, second_idx, third_idx] => {
                                let first_input_ty =
                                    self.resolve_vars_if_possible(expected_input_tys[first_idx]);
                                let second_input_ty =
                                    self.resolve_vars_if_possible(expected_input_tys[second_idx]);
                                let third_input_ty =
                                    self.resolve_vars_if_possible(expected_input_tys[third_idx]);
                                let span = if third_idx < provided_arg_count {
                                    let first_arg_span = provided_args[first_idx].span;
                                    let third_arg_span = provided_args[third_idx].span;
                                    Span::new(
                                        first_arg_span.lo(),
                                        third_arg_span.hi(),
                                        first_arg_span.ctxt(),
                                        None,
                                    )
                                } else {
                                    args_span
                                };
                                let any_unnameable = false
                                    || first_input_ty.references_error()
                                    || first_input_ty.has_infer_types()
                                    || second_input_ty.references_error()
                                    || second_input_ty.has_infer_types()
                                    || third_input_ty.references_error()
                                    || third_input_ty.has_infer_types();
                                let arg_type = if any_unnameable {
                                    "".into()
                                } else {
                                    format!(
                                        " of type `{}`, `{}`, and `{}`",
                                        first_input_ty, second_input_ty, third_input_ty
                                    )
                                };
                                labels.push((
                                    span,
                                    format!("three arguments{} are missing", arg_type),
                                ));
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
                                let span = if last_idx < provided_arg_count {
                                    let first_arg_span = provided_args[first_idx].span;
                                    let last_arg_span = provided_args[last_idx].span;
                                    Span::new(
                                        first_arg_span.lo(),
                                        last_arg_span.hi(),
                                        first_arg_span.ctxt(),
                                        None,
                                    )
                                } else {
                                    // Otherwise just label the whole function
                                    args_span
                                };
                                labels.push((span, format!("multiple arguments are missing")));
                                suggestion_text = match suggestion_text {
                                    SuggestionText::None | SuggestionText::Provide(_) => {
                                        SuggestionText::Provide(true)
                                    }
                                    _ => SuggestionText::DidYouMean,
                                };
                            }
                        }
                    }
                    Error::Swap(input_idx, other_input_idx, arg_idx, other_arg_idx) => {
                        let first_span = provided_args[input_idx].span;
                        let second_span = provided_args[other_input_idx].span;

                        let first_expected_ty =
                            self.resolve_vars_if_possible(expected_input_tys[arg_idx]);
                        let first_provided_ty = if let Some((ty, _)) = final_arg_types[input_idx] {
                            format!(",found `{}`", ty)
                        } else {
                            String::new()
                        };
                        labels.push((
                            first_span,
                            format!("expected `{}`{}", first_expected_ty, first_provided_ty),
                        ));
                        let other_expected_ty =
                            self.resolve_vars_if_possible(expected_input_tys[other_arg_idx]);
                        let other_provided_ty =
                            if let Some((ty, _)) = final_arg_types[other_input_idx] {
                                format!(",found `{}`", ty)
                            } else {
                                String::new()
                            };
                        labels.push((
                            second_span,
                            format!("expected `{}`{}", other_expected_ty, other_provided_ty),
                        ));
                        suggestion_text = match suggestion_text {
                            SuggestionText::None => SuggestionText::Swap,
                            _ => SuggestionText::DidYouMean,
                        };
                    }
                    Error::Permutation(args) => {
                        for (dst_arg, dest_input) in args {
                            let expected_ty =
                                self.resolve_vars_if_possible(expected_input_tys[dest_input]);
                            let provided_ty = if let Some((ty, _)) = final_arg_types[dst_arg] {
                                format!(",found `{}`", ty)
                            } else {
                                String::new()
                            };
                            labels.push((
                                provided_args[dst_arg].span,
                                format!("expected `{}`{}", expected_ty, provided_ty),
                            ));
                        }

                        suggestion_text = match suggestion_text {
                            SuggestionText::None => SuggestionText::Reorder,
                            _ => SuggestionText::DidYouMean,
                        };
                    }
                }
            }

            // If we have less than 5 things to say, it would be useful to call out exactly what's wrong
            if labels.len() <= 5 {
                for (span, label) in labels {
                    err.span_label(span, label);
                }
            }

            // Call out where the function is defined
            label_fn_like(tcx, &mut err, fn_def_id);

            // And add a suggestion block for all of the parameters
            let suggestion_text = match suggestion_text {
                SuggestionText::None => None,
                SuggestionText::Provide(plural) => {
                    Some(format!("provide the argument{}", if plural { "s" } else { "" }))
                }
                SuggestionText::Remove(plural) => {
                    Some(format!("remove the extra argument{}", if plural { "s" } else { "" }))
                }
                SuggestionText::Swap => Some("swap these arguments".to_string()),
                SuggestionText::Reorder => Some("reorder these arguments".to_string()),
                SuggestionText::DidYouMean => Some("did you mean".to_string()),
            };
            if let Some(suggestion_text) = suggestion_text {
                let source_map = self.sess().source_map();
                let mut suggestion = format!(
                    "{}(",
                    source_map.span_to_snippet(full_call_span).unwrap_or_else(|_| String::new())
                );
                for (arg_index, input_idx) in matched_inputs.iter().enumerate() {
                    let suggestion_text = if let Some(input_idx) = input_idx {
                        let arg_span = provided_args[*input_idx].span.source_callsite();
                        let arg_text = source_map.span_to_snippet(arg_span).unwrap();
                        arg_text
                    } else {
                        // Propose a placeholder of the correct type
                        let expected_ty = expected_input_tys[arg_index];
                        let input_ty = self.resolve_vars_if_possible(expected_ty);
                        if input_ty.is_unit() {
                            "()".to_string()
                        } else {
                            format!("{{{}}}", input_ty)
                        }
                    };
                    suggestion += &suggestion_text;
                    if arg_index < minimum_input_count - 1 {
                        suggestion += ", ";
                    }
                }
                suggestion += ")";
                err.span_suggestion_verbose(
                    error_span,
                    &suggestion_text,
                    suggestion,
                    Applicability::HasPlaceholders,
                );
            }
            err.emit();
        }

        for arg in provided_args.iter().skip(minimum_input_count) {
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
                    use crate::structured_errors::MissingCastForVariadicArg;

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
    }

    fn suggested_tuple_wrap(
        &self,
        expected_input_tys: &[Ty<'tcx>],
        provided_args: &'tcx [hir::Expr<'tcx>],
    ) -> TupleMatchFound {
        // Only handle the case where we expect only one tuple arg
        let [expected_arg_type] = expected_input_tys[..] else { return TupleMatchFound::None };
        let &ty::Tuple(expected_types) = self.resolve_vars_if_possible(expected_arg_type).kind()
            else { return TupleMatchFound::None };

        // First check that there are the same number of types.
        if expected_types.len() != provided_args.len() {
            return TupleMatchFound::None;
        }

        let supplied_types: Vec<_> = provided_args.iter().map(|arg| self.check_expr(arg)).collect();

        let all_match = iter::zip(expected_types, supplied_types)
            .all(|(expected, supplied)| self.can_eq(self.param_env, expected, supplied).is_ok());

        if !all_match {
            return TupleMatchFound::None;
        }
        match provided_args {
            [] => TupleMatchFound::None,
            [_] => TupleMatchFound::Single,
            [first, .., last] => {
                TupleMatchFound::Multiple(first.span.shrink_to_lo(), last.span.shrink_to_hi())
            }
        }
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
            ast::LitKind::ByteStr(ref v) => {
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
            ast::LitKind::Err(_) => tcx.ty_error(),
        }
    }

    pub fn check_struct_path(
        &self,
        qpath: &QPath<'_>,
        hir_id: hir::HirId,
    ) -> Option<(&'tcx ty::VariantDef, Ty<'tcx>)> {
        let path_span = qpath.span();
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, hir_id);
        let variant = match def {
            Res::Err => {
                self.set_tainted_by_errors();
                return None;
            }
            Res::Def(DefKind::Variant, _) => match ty.kind() {
                ty::Adt(adt, substs) => Some((adt.variant_of_res(def), adt.did(), substs)),
                _ => bug!("unexpected type: {:?}", ty),
            },
            Res::Def(DefKind::Struct | DefKind::Union | DefKind::TyAlias | DefKind::AssocTy, _)
            | Res::SelfTy { .. } => match ty.kind() {
                ty::Adt(adt, substs) if !adt.is_enum() => {
                    Some((adt.non_enum_variant(), adt.did(), substs))
                }
                _ => None,
            },
            _ => bug!("unexpected definition: {:?}", def),
        };

        if let Some((variant, did, substs)) = variant {
            debug!("check_struct_path: did={:?} substs={:?}", did, substs);
            self.write_user_type_annotation_from_substs(hir_id, did, substs, None);

            // Check bounds on type arguments used in the path.
            self.add_required_obligations(path_span, did, substs);

            Some((variant, ty))
        } else {
            match ty.kind() {
                ty::Error(_) => {
                    // E0071 might be caused by a spelling error, which will have
                    // already caused an error message and probably a suggestion
                    // elsewhere. Refrain from emitting more unhelpful errors here
                    // (issue #88844).
                }
                _ => {
                    struct_span_err!(
                        self.tcx.sess,
                        path_span,
                        E0071,
                        "expected struct, variant or union type, found {}",
                        ty.sort_string(self.tcx)
                    )
                    .span_label(path_span, "not a struct")
                    .emit();
                }
            }
            None
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
            self.check_expr_coercable_to_type(init, local_ty, None)
        }
    }

    pub(in super::super) fn check_decl(&self, decl: Declaration<'tcx>) {
        // Determine and write the type which we'll check the pattern against.
        let decl_ty = self.local_ty(decl.span, decl.hir_id).decl_ty;
        self.write_ty(decl.hir_id, decl_ty);

        // Type check the initializer.
        if let Some(ref init) = decl.init {
            let init_ty = self.check_decl_initializer(decl.hir_id, decl.pat, &init);
            self.overwrite_local_ty_if_err(decl.hir_id, decl.pat, decl_ty, init_ty);
        }

        // Does the expected pattern type originate from an expression and what is the span?
        let (origin_expr, ty_span) = match (decl.ty, decl.init) {
            (Some(ty), _) => (false, Some(ty.span)), // Bias towards the explicit user type.
            (_, Some(init)) => (true, Some(init.span)), // No explicit type; so use the scrutinee.
            _ => (false, None), // We have `let $pat;`, so the expected type is unconstrained.
        };

        // Type check the pattern. Override if necessary to avoid knock-on errors.
        self.check_pat_top(&decl.pat, decl_ty, ty_span, origin_expr);
        let pat_ty = self.node_ty(decl.pat.hir_id);
        self.overwrite_local_ty_if_err(decl.hir_id, decl.pat, decl_ty, pat_ty);
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
        let old_has_errors = self.has_errors.replace(false);

        match stmt.kind {
            hir::StmtKind::Local(ref l) => {
                self.check_decl_local(&l);
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
        self.has_errors.set(self.has_errors.get() | old_has_errors);
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
        let prev = self.ps.replace(self.ps.get().recurse(blk));

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
        let tail_expr = blk.expr.as_ref();
        let coerce_to_ty = expected.coercion_target_type(self, blk.span);
        let coerce = if blk.targeted_by_break {
            CoerceMany::new(coerce_to_ty)
        } else {
            let tail_expr: &[&hir::Expr<'_>] = match tail_expr {
                Some(e) => slice::from_ref(e),
                None => &[],
            };
            CoerceMany::with_coercion_sites(coerce_to_ty, tail_expr)
        };

        let prev_diverges = self.diverges.get();
        let ctxt = BreakableCtxt { coerce: Some(coerce), may_break: false };

        let (ctxt, ()) = self.with_breakable_ctxt(blk.hir_id, ctxt, || {
            for (pos, s) in blk.stmts.iter().enumerate() {
                self.check_stmt(s, blk.stmts.len() - 1 == pos);
            }

            // check the tail expression **without** holding the
            // `enclosing_breakables` lock below.
            let tail_expr_ty = tail_expr.map(|t| self.check_expr_with_expectation(t, expected));

            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            let ctxt = enclosing_breakables.find_breakable(blk.hir_id);
            let coerce = ctxt.coerce.as_mut().unwrap();
            if let Some(tail_expr_ty) = tail_expr_ty {
                let tail_expr = tail_expr.unwrap();
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
                                self.consider_hint_about_removing_semicolon(blk, expected_ty, err);
                                if expected_ty == self.tcx.types.bool {
                                    // If this is caused by a missing `let` in a `while let`,
                                    // silence this redundant error, as we already emit E0070.

                                    // Our block must be a `assign desugar local; assignment`
                                    if let Some(hir::Node::Block(hir::Block {
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
                                    })) = self.tcx.hir().find(blk.hir_id)
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

        let mut ty = ctxt.coerce.unwrap().complete(self);

        if self.has_errors.get() || ty.references_error() {
            ty = self.tcx.ty_error()
        }

        self.write_ty(blk.hir_id, ty);

        self.ps.set(prev);
        ty
    }

    /// A common error is to add an extra semicolon:
    ///
    /// ```compile_fail,E0308
    /// fn foo() -> usize {
    ///     22;
    /// }
    /// ```
    ///
    /// This routine checks if the final statement in a block is an
    /// expression with an explicit semicolon whose type is compatible
    /// with `expected_ty`. If so, it suggests removing the semicolon.
    fn consider_hint_about_removing_semicolon(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
        err: &mut Diagnostic,
    ) {
        if let Some((span_semi, boxed)) = self.could_remove_semicolon(blk, expected_ty) {
            if let StatementAsExpression::NeedsBoxing = boxed {
                err.span_suggestion_verbose(
                    span_semi,
                    "consider removing this semicolon and boxing the expression",
                    String::new(),
                    Applicability::HasPlaceholders,
                );
            } else {
                err.span_suggestion_short(
                    span_semi,
                    "remove this semicolon",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    fn parent_item_span(&self, id: hir::HirId) -> Option<Span> {
        let node = self.tcx.hir().get_by_def_id(self.tcx.hir().get_parent_item(id));
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
        let parent = self.tcx.hir().get_by_def_id(self.tcx.hir().get_parent_item(blk_id));
        self.get_node_fn_decl(parent).map(|(fn_decl, ident, _)| (fn_decl, ident))
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
            self.in_progress_typeck_results
                .and_then(|typeck_results| typeck_results.borrow().node_type_opt(elem.hir_id))
                .and_then(|ty| {
                    if ty.is_never() {
                        None
                    } else {
                        Some(match elem.kind {
                            // Point at the tail expression when possible.
                            hir::ExprKind::Block(block, _) => {
                                block.expr.map_or(block.span, |e| e.span)
                            }
                            _ => elem.span,
                        })
                    }
                })
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
        decl_ty: Ty<'tcx>,
        ty: Ty<'tcx>,
    ) {
        if ty.references_error() {
            // Override the types everywhere with `err()` to avoid knock on errors.
            self.write_ty(hir_id, ty);
            self.write_ty(pat.hir_id, ty);
            let local_ty = LocalTy { decl_ty, revealed_ty: ty };
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
    ) -> (Res, Ty<'tcx>) {
        match *qpath {
            QPath::Resolved(ref maybe_qself, ref path) => {
                let self_ty = maybe_qself.as_ref().map(|qself| self.to_ty(qself));
                let ty = <dyn AstConv<'_>>::res_to_ty(self, self_ty, path, true);
                (path.res, ty)
            }
            QPath::TypeRelative(ref qself, ref segment) => {
                let ty = self.to_ty(qself);

                let result = <dyn AstConv<'_>>::associated_path_to_ty(
                    self, hir_id, path_span, ty, qself, segment, true,
                );
                let ty = result.map(|(ty, _, _)| ty).unwrap_or_else(|_| self.tcx().ty_error());
                let result = result.map(|(_, kind, def_id)| (kind, def_id));

                // Write back the new resolution.
                self.write_resolution(hir_id, result);

                (result.map_or(Res::Err, |(kind, def_id)| Res::Def(kind, def_id)), ty)
            }
            QPath::LangItem(lang_item, span, id) => {
                self.resolve_lang_item_path(lang_item, span, hir_id, id)
            }
        }
    }

    /// Given a vec of evaluated `FulfillmentError`s and an `fn` call argument expressions, we walk
    /// the checked and coerced types for each argument to see if any of the `FulfillmentError`s
    /// reference a type argument. The reason to walk also the checked type is that the coerced type
    /// can be not easily comparable with predicate type (because of coercion). If the types match
    /// for either checked or coerced type, and there's only *one* argument that does, we point at
    /// the corresponding argument's expression span instead of the `fn` call path span.
    fn point_at_arg_instead_of_call_if_possible(
        &self,
        errors: &mut Vec<traits::FulfillmentError<'tcx>>,
        final_arg_types: &[Option<(Ty<'tcx>, Ty<'tcx>)>],
        expr: &'tcx hir::Expr<'tcx>,
        call_sp: Span,
        args: &'tcx [hir::Expr<'tcx>],
    ) {
        // We *do not* do this for desugared call spans to keep good diagnostics when involving
        // the `?` operator.
        if call_sp.desugaring_kind().is_some() {
            return;
        }

        for error in errors {
            // Only if the cause is somewhere inside the expression we want try to point at arg.
            // Otherwise, it means that the cause is somewhere else and we should not change
            // anything because we can break the correct span.
            if !call_sp.contains(error.obligation.cause.span) {
                continue;
            }

            // Peel derived obligation, because it's the type that originally
            // started this inference chain that matters, not the one we wound
            // up with at the end.
            fn unpeel_to_top<'a, 'tcx>(
                mut code: &'a ObligationCauseCode<'tcx>,
            ) -> &'a ObligationCauseCode<'tcx> {
                let mut result_code = code;
                loop {
                    let parent = match code {
                        ObligationCauseCode::ImplDerivedObligation(c) => &c.derived.parent_code,
                        ObligationCauseCode::BuiltinDerivedObligation(c)
                        | ObligationCauseCode::DerivedObligation(c) => &c.parent_code,
                        _ => break result_code,
                    };
                    (result_code, code) = (code, parent);
                }
            }
            let self_: ty::subst::GenericArg<'_> = match unpeel_to_top(error.obligation.cause.code()) {
                ObligationCauseCode::BuiltinDerivedObligation(code) |
                ObligationCauseCode::DerivedObligation(code) => {
                    code.parent_trait_pred.self_ty().skip_binder().into()
                }
                ObligationCauseCode::ImplDerivedObligation(code) => {
                    code.derived.parent_trait_pred.self_ty().skip_binder().into()
                }
                _ if let ty::PredicateKind::Trait(predicate) =
                    error.obligation.predicate.kind().skip_binder() => {
                        predicate.self_ty().into()
                    }
                _ =>  continue,
            };
            let self_ = self.resolve_vars_if_possible(self_);

            // Collect the argument position for all arguments that could have caused this
            // `FulfillmentError`.
            let mut referenced_in = final_arg_types
                .iter()
                .enumerate()
                .filter_map(|(i, arg)| match arg {
                    Some((checked_ty, coerce_ty)) => Some([(i, *checked_ty), (i, *coerce_ty)]),
                    _ => None,
                })
                .flatten()
                .flat_map(|(i, ty)| {
                    let ty = self.resolve_vars_if_possible(ty);
                    // We walk the argument type because the argument's type could have
                    // been `Option<T>`, but the `FulfillmentError` references `T`.
                    if ty.walk().any(|arg| arg == self_) { Some(i) } else { None }
                })
                .collect::<Vec<usize>>();

            // Both checked and coerced types could have matched, thus we need to remove
            // duplicates.

            // We sort primitive type usize here and can use unstable sort
            referenced_in.sort_unstable();
            referenced_in.dedup();

            if let (Some(ref_in), None) = (referenced_in.pop(), referenced_in.pop()) {
                // Do not point at the inside of a macro.
                // That would often result in poor error messages.
                if args[ref_in].span.from_expansion() {
                    return;
                }
                // We make sure that only *one* argument matches the obligation failure
                // and we assign the obligation's span to its expression's.
                error.obligation.cause.span = args[ref_in].span;
                error.obligation.cause.map_code(|parent_code| {
                    ObligationCauseCode::FunctionArgumentObligation {
                        arg_hir_id: args[ref_in].hir_id,
                        call_hir_id: expr.hir_id,
                        parent_code,
                    }
                });
            } else if error.obligation.cause.span == call_sp {
                // Make function calls point at the callee, not the whole thing.
                if let hir::ExprKind::Call(callee, _) = expr.kind {
                    error.obligation.cause.span = callee.span;
                }
            }
        }
    }

    /// Given a vec of evaluated `FulfillmentError`s and an `fn` call expression, we walk the
    /// `PathSegment`s and resolve their type parameters to see if any of the `FulfillmentError`s
    /// were caused by them. If they were, we point at the corresponding type argument's span
    /// instead of the `fn` call path span.
    fn point_at_type_arg_instead_of_call_if_possible(
        &self,
        errors: &mut Vec<traits::FulfillmentError<'tcx>>,
        call_expr: &'tcx hir::Expr<'tcx>,
    ) {
        if let hir::ExprKind::Call(path, _) = &call_expr.kind {
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = &path.kind {
                for error in errors {
                    if let ty::PredicateKind::Trait(predicate) =
                        error.obligation.predicate.kind().skip_binder()
                    {
                        // If any of the type arguments in this path segment caused the
                        // `FulfillmentError`, point at its span (#61860).
                        for arg in path
                            .segments
                            .iter()
                            .filter_map(|seg| seg.args.as_ref())
                            .flat_map(|a| a.args.iter())
                        {
                            if let hir::GenericArg::Type(hir_ty) = &arg {
                                if let hir::TyKind::Path(hir::QPath::TypeRelative(..)) =
                                    &hir_ty.kind
                                {
                                    // Avoid ICE with associated types. As this is best
                                    // effort only, it's ok to ignore the case. It
                                    // would trigger in `is_send::<T::AssocType>();`
                                    // from `typeck-default-trait-impl-assoc-type.rs`.
                                } else {
                                    let ty = <dyn AstConv<'_>>::ast_ty_to_ty(self, hir_ty);
                                    let ty = self.resolve_vars_if_possible(ty);
                                    if ty == predicate.self_ty() {
                                        error.obligation.cause.span = hir_ty.span;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn label_fn_like<'tcx>(
    tcx: TyCtxt<'tcx>,
    err: &mut rustc_errors::DiagnosticBuilder<'tcx, rustc_errors::ErrorGuaranteed>,
    def_id: Option<DefId>,
) {
    let Some(def_id) = def_id else {
        return;
    };

    if let Some(def_span) = tcx.def_ident_span(def_id) {
        let mut spans: MultiSpan = def_span.into();

        let params = tcx
            .hir()
            .get_if_local(def_id)
            .and_then(|node| node.body_id())
            .into_iter()
            .flat_map(|id| tcx.hir().body(id).params);

        for param in params {
            spans.push_span_label(param.span, String::new());
        }

        let def_kind = tcx.def_kind(def_id);
        err.span_note(spans, &format!("{} defined here", def_kind.descr(def_id)));
    } else {
        match tcx.hir().get_if_local(def_id) {
            Some(hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(_, _, _, span, ..),
                ..
            })) => {
                let spans: MultiSpan = (*span).into();

                // Note: We don't point to param spans here because they overlap
                // with the closure span itself

                err.span_note(spans, "closure defined here");
            }
            _ => {}
        }
    }
}
