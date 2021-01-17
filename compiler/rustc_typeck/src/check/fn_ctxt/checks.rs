use crate::astconv::AstConv;
use crate::check::coercion::CoerceMany;
use crate::check::method::MethodCallee;
use crate::check::Expectation::*;
use crate::check::TupleArgumentsFlag::*;
use crate::check::{
    struct_span_err, BreakableCtxt, Diverges, Expectation, FnCtxt, LocalTy, Needs,
    TupleArgumentsFlag,
};

use rustc_ast as ast;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_infer::infer::InferOk;
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty};
use rustc_session::Session;
use rustc_span::{self, MultiSpan, Span};
use rustc_span::{
    symbol::{sym, Ident},
    BytePos,
};
use rustc_trait_selection::traits::{self, ObligationCauseCode, StatementAsExpression};

use std::cmp;
use std::mem::replace;
use std::slice;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(in super::super) fn check_casts(&self) {
        let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
        for cast in deferred_cast_checks.drain(..) {
            cast.check(self);
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
                TupleArguments => vec![self.tcx.intern_tup(&err_inputs[..])],
            };

            self.check_argument_types(
                sp,
                expr,
                &err_inputs[..],
                &[],
                args_no_rcvr,
                false,
                tuple_arguments,
                None,
            );
            return self.tcx.ty_error();
        }

        let method = method.unwrap();
        // HACK(eddyb) ignore self in the definition (see above).
        let expected_arg_tys = self.expected_inputs_for_expected_output(
            sp,
            expected,
            method.sig.output(),
            &method.sig.inputs()[1..],
        );
        self.check_argument_types(
            sp,
            expr,
            &method.sig.inputs()[1..],
            &expected_arg_tys[..],
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
        expected_input_tys: &[Ty<'tcx>],
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
        for (&fn_input_ty, arg_expr) in formal_input_tys.iter().zip(provided_args.iter()) {
            self.register_wf_obligation(fn_input_ty.into(), arg_expr.span, traits::MiscObligation);
        }

        let mut expected_input_tys = expected_input_tys.to_vec();
        // If the arguments should be wrapped in a tuple (ex: closures), unwrap them here
        let formal_input_tys = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolved_type(call_span, formal_input_tys[0]);
            match tuple_type.kind() {
                ty::Tuple(arg_types) => {
                    expected_input_tys = match expected_input_tys.get(0) {
                        Some(&ty) => match ty.kind() {
                            ty::Tuple(ref tys) => tys.iter().map(|k| k.expect_ty()).collect(),
                            _ => vec![],
                        },
                        None => vec![],
                    };
                    arg_types.iter().map(|k| k.expect_ty()).collect()
                }
                _ => {
                    // Otherwise, there's a mismatch, so clear out what we're expecting, and set
                    // our input typs to err_args so we don't blow up the error messages
                    expected_input_tys = vec![];
                    self.err_args(provided_args.len())
                }
            }
        } else {
            formal_input_tys.to_vec()
        };

        // If there are no external expectations at the call site, just use the types from the function defn
        let expected_input_tys = if !expected_input_tys.is_empty() {
            expected_input_tys
        } else {
            formal_input_tys.clone()
        };

        let minimum_input_count = expected_input_tys.len();
        let provided_arg_count: usize = provided_args.len();

        // Allocate a small grid;
        // compatibility_matrix[i][j] will represent whether provided argument i could satisfy input j
        let mut compatibility_matrix = vec![vec![false; minimum_input_count]; provided_arg_count];

        // Keep track of whether we *could possibly* be satisfied, i.e. whether we're on the happy path
        // if the wrong number of arguments were supplied, we CAN'T be satisfied,
        // and if we're c_variadic, the supplied arguments must be >= the minimum count from the function
        // otherwise, they need to be identical, because rust doesn't currently support variadic functions
        let mut call_appears_satisfied = if c_variadic {
            provided_arg_count >= minimum_input_count
        } else {
            provided_arg_count == minimum_input_count
        };

        // The type of any closure arguments we encounter may be subject to obligations imposed by later arguments,
        // so we defer checking any closures until the end, when all of those obligations have been registered
        let mut deferred_arguments: Vec<usize> = vec![];

        // We'll also want to keep track of the fully coerced argument types, for an awkward hack near the end
        let mut final_arg_types: Vec<Option<(Ty<'_>, Ty<'_>)>> = vec![None; provided_arg_count];

        // We introduce a helper function to demand that a given argument satisfy a given input
        // This is more complicated than just checking type equality, as arguments could be coerced
        // This version writes those types back so further type checking uses the narrowed types
        let demand_compatible = |arg_idx, input_idx| {
            let formal_input_ty: Ty<'tcx> = formal_input_tys[input_idx];
            let expected_input_ty: Ty<'tcx> = expected_input_tys[input_idx];
            let provided_arg = &provided_args[arg_idx];

            // We're on the happy path here, so we'll do a more involved check and write back types
            // To check compatibility, we'll do 3 things:
            // 1. Unify the provided argument with the expected type
            let expectation = Expectation::rvalue_hint(self, expected_input_ty);
            let checked_ty = self.check_expr_with_expectation(provided_arg, expectation);

            // 2. Find and check the most detailed coercible type
            let coerced_ty = expectation.only_has_type(self).unwrap_or(formal_input_ty);
            let coerced_ty = self.resolve_vars_with_obligations(coerced_ty);

            let coerce_error =
                self.try_coerce(provided_arg, checked_ty, coerced_ty, AllowTwoPhase::Yes);

            // 3. Check if the formal type is a supertype of the checked one
            //    and register any such obligations for future type checks
            let supertype_error = self
                .at(&self.misc(provided_arg.span), self.param_env)
                .sup(formal_input_ty, coerced_ty);
            let is_supertype = match supertype_error {
                Ok(InferOk { obligations, value: () }) => {
                    self.register_predicates(obligations);
                    true
                }
                _ => false,
            };

            // If neither check failed, the types are compatible
            (coerce_error.is_ok() && is_supertype, checked_ty, coerced_ty)
        };

        // A "softer" version of the helper above, which checks types without persisting them,
        // and treats error types differently
        // This will allow us to "probe" for other argument orders that would likely have been correct
        let check_compatible = |arg_idx, input_idx| {
            let formal_input_ty: Ty<'tcx> = formal_input_tys[input_idx];
            let expected_input_ty: Ty<'tcx> = expected_input_tys[input_idx];
            

            // If either is an error type, we defy the usual convention and consider them to *not* be
            // coercible.  This prevents our error message heuristic from trying to pass errors into
            // every argument.
            if formal_input_ty.references_error() || expected_input_ty.references_error() {
                return false;
            }

            let provided_arg: &hir::Expr<'tcx> = &provided_args[arg_idx];
            let expectation = Expectation::rvalue_hint(self, expected_input_ty);
            // TODO: check that this is safe; I don't believe this commits any of the obligations, but I can't be sure.
            //
            //   I had another method of "soft" type checking before,
            //   but it was failing to find the type of some expressions (like "")
            //   so I prodded this method and made it pub(super) so I could call it, and it seems to work well. 
            let checked_ty = self.check_expr_kind(provided_arg, expectation);

            let coerced_ty = expectation.only_has_type(self).unwrap_or(formal_input_ty);
            let can_coerce = self.can_coerce(checked_ty, coerced_ty);

            let is_super = self
                .at(&self.misc(provided_arg.span), self.param_env)
                .sup(formal_input_ty, coerced_ty)
                .is_ok();
            // Same as above: if either the coerce type or the checked type is an error type,
            // consider them *not* compatible.
            return !coerced_ty.references_error()
                && !checked_ty.references_error()
                && can_coerce
                && is_super;
        };

        // Check each argument, to satisfy the input it was provided for
        // Visually, we're traveling down the diagonal of the compatibility matrix
        for idx in 0..provided_arg_count {
            // First, warn if this expression is unreachable
            // ex: myFn(panic!(), 2 + 2)
            //                    ^^^^^
            self.warn_if_unreachable(
                provided_args[idx].hir_id,
                provided_args[idx].span,
                "expression",
            );

            // If we're past the end of the expected inputs, we won't have anything to check against
            if idx >= minimum_input_count {
                break;
            }

            // If this argument is a closure, we defer this to a second pass, so we have more type information
            if matches!(provided_args[idx].kind, ExprKind::Closure(..)) {
                deferred_arguments.push(idx);
                continue;
            }

            // Demand that this argument satisfies the input in the slot it's in
            let (compatible, checked_ty, coerced_ty) = demand_compatible(idx, idx);
            // Keep track of these for below
            final_arg_types[idx] = Some((checked_ty, coerced_ty));

            // If we fail at some point, we'll want to provide better error messages, so hold onto this info
            if compatible {
                compatibility_matrix[idx][idx] = true;
            } else {
                call_appears_satisfied = false;
            }
        }

        // Next, check any closures, since we have more type info at this point
        // To help with type resolution, we can do an "opportunistic" vtable resolution
        // on any trait bounds.  This is considered by some to be a pretty awful hack.
        self.select_obligations_where_possible(false, |errors| {
            // Clean up the error messages a bit
            self.point_at_type_arg_instead_of_call_if_possible(errors, call_expr);
            self.point_at_arg_instead_of_call_if_possible(
                errors,
                &final_arg_types,
                call_span,
                &provided_args,
            );
        });

        for idx in deferred_arguments {
            let (compatible, _, _) = demand_compatible(idx, idx);
            // Note that, unlike the first pass, we ignore the checked/coerced types,
            // since we don't plan on running select_obligations_where_possible again
            if compatible {
                compatibility_matrix[idx][idx] = true;
            } else {
                call_appears_satisfied = false;
            }
        }

        // If something above didn't typecheck, we've fallen off the happy path
        // and we should make some effort to provide better error messages
        if !call_appears_satisfied {
            // The algorithm here is inspired by levenshtein distance and longest common subsequence.
            // We'll try to detect 4 different types of mistakes:
            // - An extra parameter has been provided that doesn't satisfy *any* of the other inputs
            // - An input is missing, which isn't satisfied by *any* of the other arguments
            // - Some number of arguments have been provided in the wrong order
            // - A type is straight up invalid

            // First, fill in the rest of our compatibility matrix
            for i in 0..provided_arg_count {
                for j in 0..minimum_input_count {
                    if i == j {
                        continue;
                    }
                    compatibility_matrix[i][j] = check_compatible(i, j);
                }
            }

            // Obviously, detecting exact user intention is impossible, so the goal here is to
            // come up with as likely of a story as we can to be helpful.
            //
            // We'll iteratively removed "satisfied" input/argument paris,
            // then check for the cases above, until we've eliminated the entire grid
            //
            // We'll want to know which arguments and inputs these rows and columns correspond to
            // even after we delete them, so these lookups will keep track of that
            let mut input_indexes: Vec<usize> = (0..minimum_input_count).collect();
            let mut arg_indexes: Vec<usize> = (0..provided_arg_count).collect();

            // First, set up some utility functions for the algorithm below
            // Remove a given input or argument from consideration
            let eliminate_input = |mat: &mut Vec<Vec<bool>>, ii: &mut Vec<usize>, idx| {
                ii.remove(idx);
                for row in mat {
                    row.remove(idx);
                }
            };
            let eliminate_arg = |mat: &mut Vec<Vec<bool>>, ai: &mut Vec<usize>, idx| {
                ai.remove(idx);
                mat.remove(idx);
            };
            // "satisfy" an input with a given arg, removing both from consideration
            let satisfy_input = |mat: &mut Vec<Vec<bool>>,
                                 ii: &mut Vec<usize>,
                                 ai: &mut Vec<usize>,
                                 input_idx,
                                 arg_idx| {
                eliminate_input(mat, ii, input_idx);
                eliminate_arg(mat, ai, arg_idx);
            };

            let eliminate_satisfied = |mat: &mut Vec<Vec<bool>>, ii: &mut Vec<usize>, ai: &mut Vec<usize>| {
                let mut i = cmp::min(ii.len(), ai.len());
                while i > 0 {
                    let idx = i - 1;
                    if mat[idx][idx] {
                        satisfy_input(
                            mat,
                            ii,
                            ai,
                            idx,
                            idx,
                        );
                    }
                    i -= 1;
                }
            };

            // A list of the issues we might find
            enum Issue {
                Invalid(usize),
                Missing(usize),
                Extra(usize),
                Swap(usize, usize),
                Permutation(Vec<Option<usize>>),
            }
            // Check for the above mismatch cases
            let find_issue = |mat: &Vec<Vec<bool>>, ii: &Vec<usize>, ai: &Vec<usize>| {
                for i in 0..cmp::max(ai.len(), ii.len()) {
                    // If we eliminate the last row, any left-over inputs are considered missing
                    if i >= mat.len() {
                        return Some(Issue::Missing(i));
                    }
                    // If we eliminate the last column, any left-over arguments are extra
                    if mat[i].len() == 0 {
                        return Some(Issue::Extra(i));
                    }

                    // Make sure we don't pass the bounds of our matrix
                    let is_arg = i < ai.len();
                    let is_input = i < ii.len();
                    if is_arg && is_input && mat[i][i] {
                        // This is a satisfied input, so move along
                        continue;
                    }

                    let mut useless = true;
                    let mut unsatisfiable = true;
                    if is_arg {
                        for j in 0..ii.len() {
                            // If we find at least one input this argument could satisfy
                            // this argument isn't completely useless
                            if mat[i][j] {
                                useless = false;
                                break;
                            }
                        }
                    }
                    if is_input {
                        for j in 0..ai.len() {
                            // If we find at least one argument that could satisfy this input
                            // this argument isn't unsatisfiable
                            if mat[j][i] {
                                unsatisfiable = false;
                                break;
                            }
                        }
                    }

                    match (is_arg, is_input, useless, unsatisfiable) {
                        // If an input is unsatisfied, and the argument in its position is useless
                        // then the most likely explanation is that we just got the types wrong
                        (true, true, true, true) => return Some(Issue::Invalid(i)),
                        // Otherwise, if an input is useless, then indicate that this is an extra argument
                        (true, _, true, _) => return Some(Issue::Extra(i)),
                        // Otherwise, if an argument is unsatisfiable, indicate that it's missing
                        (_, true, _, true) => return Some(Issue::Missing(i)),
                        (true, true, _, _) => {
                            // The argument isn't useless, and the input isn't unsatisfied,
                            // so look for a parameter we might swap it with
                            // We look for swaps explicitly, instead of just falling back on permutations
                            // so that cases like (A,B,C,D) given (B,A,D,C) show up as two swaps,
                            // instead of a large permutation of 4 elements.
                            for j in 0..cmp::min(ai.len(), ii.len()) {
                                if i == j || mat[j][j] {
                                    continue;
                                }
                                if mat[i][j] && mat[j][i] {
                                    return Some(Issue::Swap(i, j));
                                }
                            }
                        }
                        _ => {
                            continue;
                        }
                    };
                }

                // We didn't find any of the individual issues above, but
                // there might be a larger permutation of parameters, so we now check for that
                // by checking for cycles
                // We use a double option at position i in this vec to represent:
                // - None: We haven't computed anything about this argument yet
                // - Some(None): This argument definitely doesn't participate in a cycle
                // - Some(Some(x)): the i-th argument could permute to the x-th position
                let mut permutation: Vec<Option<Option<usize>>> = vec![None; mat.len()];
                let mut permutation_found = false;
                for i in 0..mat.len() {
                    if permutation[i].is_some() {
                        // We've already decided whether this argument is or is not in a loop
                        continue;
                    }

                    let mut stack = vec![];
                    let mut j = i;
                    let mut last = i;
                    let mut is_cycle = true;
                    loop {
                        stack.push(j);
                        // Look for params this one could slot into
                        let compat: Vec<_> =
                            mat[j].iter().enumerate().filter_map(|(i, &c)| if c { Some(i) } else { None }).collect();
                        if compat.len() != 1 {
                            // this could go into multiple slots, don't bother exploring both
                            is_cycle = false;
                            break;
                        }
                        j = compat[0];
                        if stack.contains(&j) {
                            last = j;
                            break;
                        }
                    }
                    if stack.len() <= 2 {
                        // If we encounter a cycle of 1 or 2 elements, we'll let the
                        // "satisfy" and "swap" code above handle those
                        is_cycle = false;
                    }
                    // We've built up some chain, some of which might be a cycle
                    // ex: [1,2,3,4]; last = 2; j = 2;
                    // So, we want to mark 4, 3, and 2 as part of a permutation
                    permutation_found = is_cycle;
                    while let Some(x) = stack.pop() {
                        if is_cycle {
                            permutation[x] = Some(Some(j));
                            j = x;
                            if j == last {
                                // From here on out, we're a tail leading into a cycle,
                                // not the cycle itself
                                is_cycle = false;
                            }
                        } else {
                            // Some(None) ensures we save time by skipping this argument again
                            permutation[x] = Some(None);
                        }
                    }
                }

                if permutation_found {
                    // Map unwrap to remove the first layer of Some
                    let final_permutation: Vec<Option<usize>> =
                        permutation.into_iter().map(|x| x.unwrap()).collect();
                    return Some(Issue::Permutation(final_permutation));
                }
                return None;
            };

            // Before we start looking for issues, eliminate any arguments that are already satisfied,
            // so that an argument which is already spoken for by the input it's in doesn't
            // spill over into another similarly typed input
            // ex:
            //   fn some_func(_a: i32, _a: i32) {}
            //   some_func(1, "");
            // Without this elimination, the first argument causes the second argument
            // to show up as both a missing input and extra argument, rather than
            // just an invalid type.
            eliminate_satisfied(&mut compatibility_matrix, &mut input_indexes, &mut arg_indexes);

            // As we encounter issues, we'll transcribe them to their actual indices
            let mut issues: Vec<Issue> = vec![];
            // Until we've elimineated / satisfied all arguments/inputs
            while input_indexes.len() > 0 || arg_indexes.len() > 0 {
                // Check for the first relevant issue
                match find_issue(&compatibility_matrix, &input_indexes, &arg_indexes) {
                    Some(Issue::Invalid(idx)) => {
                        // Eliminate the input and the arg, while transposing to the original index
                        issues.push(Issue::Invalid(arg_indexes[idx]));
                        eliminate_input(&mut compatibility_matrix, &mut input_indexes, idx);
                        eliminate_arg(&mut compatibility_matrix, &mut arg_indexes, idx);
                    }
                    Some(Issue::Extra(idx)) => {
                        issues.push(Issue::Extra(arg_indexes[idx]));
                        eliminate_arg(&mut compatibility_matrix, &mut arg_indexes, idx);
                    }
                    Some(Issue::Missing(idx)) => {
                        // FIXME: improve these with help from code reviewers
                        let input_ty =
                            self.resolve_vars_if_possible(expected_input_tys[input_indexes[idx]]);
                        if input_ty.is_unit() {
                            info!("~~~ Issue: Maybe use ()?"); // FIXME
                        }
                        issues.push(Issue::Missing(input_indexes[idx]));
                        eliminate_input(&mut compatibility_matrix, &mut input_indexes, idx);
                    }
                    Some(Issue::Swap(idx, other)) => {
                        issues.push(Issue::Swap(arg_indexes[idx], arg_indexes[other]));
                        let (min, max) = (cmp::min(idx, other), cmp::max(idx, other));
                        satisfy_input(
                            &mut compatibility_matrix,
                            &mut input_indexes,
                            &mut arg_indexes,
                            min,
                            max,
                        );
                        satisfy_input(
                            &mut compatibility_matrix,
                            &mut input_indexes,
                            &mut arg_indexes,
                            max - 1, // Subtract 1 because we already removed the "min" row
                            min,
                        );
                    }
                    Some(Issue::Permutation(args)) => {
                        // FIXME: If satisfy_input ever did anything non-trivial (emit obligations to help type checking, for example)
                        // we'd want to call this function with the correct arg/input pairs, but for now, we just throw them in a bucket.
                        // This works because they force a cycle, so each row is guaranteed to also be a column
                        let mut idxs: Vec<usize> =
                            args.iter()
                                .filter_map(|&a| a)
                                .collect();
                        // FIXME: Is there a cleaner way to do this?
                        let mut real_idxs = vec![None; provided_args.len()];
                        for (src, dst) in args.iter().enumerate() {
                            real_idxs[arg_indexes[src]] = dst.map(|dst| arg_indexes[dst]);
                        }
                        issues.push(Issue::Permutation(real_idxs));
                        idxs.sort();
                        idxs.reverse();
                        for i in idxs {
                            satisfy_input(
                                &mut compatibility_matrix,
                                &mut input_indexes,
                                &mut arg_indexes,
                                i,
                                i,
                            );
                        }
                    }
                    None => {
                        // We didn't find any issues, so we need to push the algorithm forward
                        // First, eliminate any arguments that currently satisfy their inputs
                        eliminate_satisfied(&mut compatibility_matrix, &mut input_indexes, &mut arg_indexes);
                    }
                };
            }

            let issue_count = issues.len();
            if issue_count > 0 {
                // This represents the kind of wording we want to use on the suggestion
                enum SuggestionType {
                    NoSuggestion,
                    Remove,  // Suggest removing an argument
                    Provide, // Suggest providing an argument of a given type
                    Swap,    // Suggest swapping a few arguments
                    Reorder, // Suggest an arbitrary permutation (more complicated than swaps)
                    Changes  // Suggest a mixed bag of generic changes, which may include multiple of the above
                }
                use SuggestionType::*;

                // We found issues, so let's construct a diagnostic that summarizes the issues we found
                // FIXME: This might need some refining in code review
                let mut labels = vec![];
                let mut suggestions = vec![];
                let mut suggestion_type = NoSuggestion;
                let mut eliminated_args = vec![false; provided_arg_count];
                let source_map = self.sess().source_map();
                for issue in issues {
                    match issue {
                        Issue::Invalid(arg) => {
                            let span = provided_args[arg].span;
                            let expected_type = expected_input_tys[arg];
                            let found_type = final_arg_types[arg].unwrap().0;
                            labels.push((span, format!("expected {}, found {}", expected_type, found_type)));
                            suggestion_type = match suggestion_type {
                                NoSuggestion | Provide => Provide,
                                _ => Changes,
                            };
                            suggestions.push((span, format!("{{{}}}", expected_type)));
                        }
                        Issue::Extra(arg) => {
                            let span = provided_args[arg].span;

                            // We want to suggest deleting this arg,
                            // but dealing with formatting before and after is tricky
                            // so we mark it as deleted, so we can do a scan afterwards
                            eliminated_args[arg] = true;
                            let found_type = self.check_expr(&provided_args[arg]);
                            // TODO: if someone knows the method-chain-foo to achieve this, let me know
                            // but I didn't have the patience for it lol
                            let fn_name = if let Some(def_id) = fn_def_id {
                                let node = tcx.hir().get_if_local(def_id);
                                let def_kind = tcx.def_kind(def_id);
                                let descr = def_kind.descr(def_id);
                                if let Some(node) = node {
                                    if let Some(ident) = node.ident() {
                                        format!("for {}", ident)
                                    } else {
                                        format!("for this {}", descr)
                                    }
                                } else {
                                    format!("for this {}", descr)
                                }
                            } else {
                                "here".to_string()
                            };

                            labels.push((span, format!("this parameter of type {} isn't needed {}", found_type, fn_name)));
                            suggestion_type = match suggestion_type {
                                NoSuggestion | Remove => Remove,
                                _ => Changes,
                            };
                        }
                        Issue::Missing(arg) => {
                            // FIXME: The spans here are all kinds of wrong for multiple missing arguments etc.
                            let prev_span = if arg < provided_args.len() {
                                provided_args[arg].span
                            } else {
                                call_span
                            };
                            // If we're missing something in the middle, shift the span slightly to eat the comma
                            let missing_span = if arg < provided_args.len() {
                                Span::new(
                                    prev_span.hi() + BytePos(1),
                                    prev_span.hi() + BytePos(1),
                                    prev_span.ctxt(),
                                )
                            } else {
                                Span::new(
                                    prev_span.hi() - BytePos(1),
                                    prev_span.hi() - BytePos(1),
                                    prev_span.ctxt(),
                                )
                            };
                            let expected_type = expected_input_tys[arg];
                            labels.push((missing_span, format!("missing argument of type {}", expected_type)));
                            suggestion_type = match suggestion_type {
                                NoSuggestion | Provide => Provide,
                                _ => Changes,
                            };
                            suggestions.push((missing_span, format!(" {{{}}},", expected_type)));
                        }
                        Issue::Swap(arg, other) => {
                            let first_span = provided_args[arg].span;
                            let second_span = provided_args[other].span;
                            let first_snippet = source_map.span_to_snippet(first_span).unwrap();
                            let second_snippet = source_map.span_to_snippet(second_span).unwrap();
                            let expected_types = (expected_input_tys[arg], expected_input_tys[other]);
                            let found_types = (
                                final_arg_types[arg].unwrap().1,
                                final_arg_types[other].unwrap().1,
                            );
                            labels.push((first_span, format!("expected {}, found {}", expected_types.0, found_types.0)));
                            suggestions.push((first_span, second_snippet));
                            labels.push((second_span, format!("expected {}, found {}", expected_types.1, found_types.1)));
                            suggestions.push((second_span, first_snippet));
                            suggestion_type = match suggestion_type {
                                NoSuggestion | Swap => Swap,
                                _ => Changes,
                            };
                        }
                        Issue::Permutation(args) => {
                            for (src, &arg) in args.iter().enumerate() {
                                if let Some(dst) = arg {
                                    let src_span = provided_args[src].span;
                                    let dst_span = provided_args[dst].span;
                                    let snippet = source_map.span_to_snippet(src_span).unwrap();
                                    let expected_type = expected_input_tys[dst];
                                    let found_type = final_arg_types[dst].unwrap().1;
                                    labels.push((dst_span, format!("expected {}, found {}", expected_type, found_type)));
                                    suggestions.push((dst_span, snippet));
                                    suggestion_type = match suggestion_type {
                                        NoSuggestion | Reorder => Reorder,
                                        _ => Changes,
                                    };
                                }
                            }
                        }
                    }
                }

                // Now construct our error from the various things we've labeled
                let highlight_sp = MultiSpan::from_spans(labels.iter().map(|s| s.0).collect());
                let mut err = struct_span_err!(
                    tcx.sess,
                    highlight_sp,
                    E0059, // FIXME: Choose a different code?
                    "{}arguments to this function are incorrect",
                    if issue_count > 1 { "multiple " } else { "" }
                );

                // Call out where the function is defined
                if let Some(def_id) = fn_def_id {
                    if let Some(node) = tcx.hir().get_if_local(def_id) {
                        let mut spans: MultiSpan = node
                            .ident()
                            .map(|ident| ident.span)
                            .unwrap_or_else(|| tcx.hir().span(node.hir_id().unwrap()))
                            .into();

                        if let Some(id) = node.body_id() {
                            let body = tcx.hir().body(id);
                            for param in body.params {
                                spans.push_span_label(param.span, String::new());
                            }
                        }
                        let def_kind = tcx.def_kind(def_id);
                        err.span_note(spans, &format!("{} defined here", def_kind.descr(def_id)));
                    }
                }

                // annotate each of the labels
                for (span, label) in labels {
                    err.span_label(span, label);
                }
                
                // Before constructing our multipart suggestions, we need to add in all the suggestions
                // to eliminate extra parameters
                // We can't overlap spans in a suggestion or weird things happen
                let mut lo = None;
                let mut hi = None;
                let mut in_block = false;
                // Scan backwards over the args (scanning backwards lets us favor the commas after, rather than before)
                for (idx, eliminated) in eliminated_args.iter().enumerate().rev() {
                    match (in_block, eliminated, idx == 0) {
                        (false, true, false) => {
                            // We just encountered the start of a block of eliminated parameters
                            hi = Some(idx);
                            in_block = true;
                        },
                        (false, true, true) => {
                            // We encountered a single eliminated arg at the end of the arg list
                            lo = Some(idx);
                            hi = Some(idx);
                        }
                        (true, false, _) => {
                            // We encountered the end of a block, set the hi so the logic below kicks in
                            lo = Some(idx + 1);
                            in_block = false;
                        },
                        (true, true, true) => {
                            lo = Some(idx);
                            in_block = false;
                        }
                        _ => {}
                    }
                    if lo.is_some() && hi.is_some() {
                        let (lo_idx, hi_idx) = (lo.unwrap(), hi.unwrap());
                        // We've found a contiguous block, so emit the elimination
                        // be careful abound the boundaries
                        let ctxt = provided_args[0].span.ctxt();
                        let (lo_bp, hi_bp) = match (lo_idx == 0, hi_idx == provided_arg_count - 1) {
                            // If we have an arg to our right, we need to eat the comma of the last eliminated param
                            // xx, xx, xx, a
                            // [-----------)
                            (_, false) => {
                                (provided_args[lo_idx].span.lo(), provided_args[hi_idx + 1].span.lo())
                            },
                            // If this block extends to the last argument, and theres an arg to the left, eat its comma
                            // a, xx, xx, xx
                            //  [-----------)
                            (false, true) => {
                                (provided_args[lo_idx - 1].span.hi(), provided_args[hi_idx].span.hi())
                            },
                            // If every argument was eliminated, don't need to worry about commas before or after
                            // xx, xx, xx, xx
                            // [-------------)
                            (true, true) => {
                                (provided_args[lo_idx].span.lo(), provided_args[hi_idx].span.hi())
                            },
                        };
                        let eliminated_span = Span::new(
                            lo_bp,
                            hi_bp,
                            ctxt,
                        );
                        suggestions.push((eliminated_span, "".to_string()));
                        lo = None;
                        hi = None;
                    }
                }

                // And add a series of suggestions
                // FIXME: for simpler cases, this might be overkill
                if suggestions.len() > 0 {
                    let suggestion_text = match (&suggestion_type, issue_count) {
                        (Remove, 1) => Some("remove this argument"),
                        (Remove, _) => Some("remove these arguments"),
                        (Provide, 1) => Some("provide a parameter of the correct type here"),
                        (Provide, _) => Some("provide parameters of the correct types"),
                        (Swap, 1) => Some("swap these two arguments"),
                        (Swap, _) => Some("swap these arguments"),
                        (Reorder, _) => Some("reorder these parameters"),
                        _ => Some("make these changes"),
                    };
                    if let Some(suggestion_text) = suggestion_text {
                        err.multipart_suggestion_verbose(
                            suggestion_text,
                            suggestions,
                            if matches!(suggestion_type, Provide) { Applicability::HasPlaceholders } else { Applicability::MaybeIncorrect },
                        );
                    }
                }

                err.emit();
            }
        }

        // If the function is c-style variadic, we skipped a bunch of arguments
        // so we need to check those, and write out the types
        // Ideally this would be folded into the above, for uniform style
        // but c-variadic is already a corner case
        if c_variadic {
            fn variadic_error<'tcx>(s: &Session, span: Span, t: Ty<'tcx>, cast_ty: &str) {
                use crate::structured_errors::{StructuredDiagnostic, VariadicError};
                VariadicError::new(s, span, t, cast_ty).diagnostic().emit();
            }

            for arg in provided_args.iter().skip(minimum_input_count) {
                let arg_ty = self.check_expr(&arg);

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolved_type(arg.span, arg_ty);
                match arg_ty.kind() {
                    ty::Float(ast::FloatTy::F32) => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_double");
                    }
                    ty::Int(ast::IntTy::I8 | ast::IntTy::I16) | ty::Bool => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_int");
                    }
                    ty::Uint(ast::UintTy::U8 | ast::UintTy::U16) => {
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
            ast::LitKind::Int(_, ast::LitIntType::Signed(t)) => tcx.mk_mach_int(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsigned(t)) => tcx.mk_mach_uint(t),
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
            ast::LitKind::Float(_, ast::LitFloatType::Suffixed(t)) => tcx.mk_mach_float(t),
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
        let path_span = qpath.qself_span();
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, hir_id);
        let variant = match def {
            Res::Err => {
                self.set_tainted_by_errors();
                return None;
            }
            Res::Def(DefKind::Variant, _) => match ty.kind() {
                ty::Adt(adt, substs) => Some((adt.variant_of_res(def), adt.did, substs)),
                _ => bug!("unexpected type: {:?}", ty),
            },
            Res::Def(DefKind::Struct | DefKind::Union | DefKind::TyAlias | DefKind::AssocTy, _)
            | Res::SelfTy(..) => match ty.kind() {
                ty::Adt(adt, substs) if !adt.is_enum() => {
                    Some((adt.non_enum_variant(), adt.did, substs))
                }
                _ => None,
            },
            _ => bug!("unexpected definition: {:?}", def),
        };

        if let Some((variant, did, substs)) = variant {
            debug!("check_struct_path: did={:?} substs={:?}", did, substs);
            self.write_user_type_annotation_from_substs(hir_id, did, substs, None);

            // Check bounds on type arguments used in the path.
            let (bounds, _) = self.instantiate_bounds(path_span, did, substs);
            let cause =
                traits::ObligationCause::new(path_span, self.body_id, traits::ItemObligation(did));
            self.add_obligations_for_parameters(cause, bounds);

            Some((variant, ty))
        } else {
            struct_span_err!(
                self.tcx.sess,
                path_span,
                E0071,
                "expected struct, variant or union type, found {}",
                ty.sort_string(self.tcx)
            )
            .span_label(path_span, "not a struct")
            .emit();
            None
        }
    }

    pub fn check_decl_initializer(
        &self,
        local: &'tcx hir::Local<'tcx>,
        init: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // FIXME(tschottdorf): `contains_explicit_ref_binding()` must be removed
        // for #42640 (default match binding modes).
        //
        // See #44848.
        let ref_bindings = local.pat.contains_explicit_ref_binding();

        let local_ty = self.local_ty(init.span, local.hir_id).revealed_ty;
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

    /// Type check a `let` statement.
    pub fn check_decl_local(&self, local: &'tcx hir::Local<'tcx>) {
        // Determine and write the type which we'll check the pattern against.
        let ty = self.local_ty(local.span, local.hir_id).decl_ty;
        self.write_ty(local.hir_id, ty);

        // Type check the initializer.
        if let Some(ref init) = local.init {
            let init_ty = self.check_decl_initializer(local, &init);
            self.overwrite_local_ty_if_err(local, ty, init_ty);
        }

        // Does the expected pattern type originate from an expression and what is the span?
        let (origin_expr, ty_span) = match (local.ty, local.init) {
            (Some(ty), _) => (false, Some(ty.span)), // Bias towards the explicit user type.
            (_, Some(init)) => (true, Some(init.span)), // No explicit type; so use the scrutinee.
            _ => (false, None), // We have `let $pat;`, so the expected type is unconstrained.
        };

        // Type check the pattern. Override if necessary to avoid knock-on errors.
        self.check_pat_top(&local.pat, ty, ty_span, origin_expr);
        let pat_ty = self.node_ty(local.pat.hir_id);
        self.overwrite_local_ty_if_err(local, ty, pat_ty);
    }

    pub fn check_stmt(&self, stmt: &'tcx hir::Stmt<'tcx>) {
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
                    self.suggest_semicolon_at_end(expr.span, err);
                });
            }
            hir::StmtKind::Semi(ref expr) => {
                self.check_expr(&expr);
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
        let prev = {
            let mut fcx_ps = self.ps.borrow_mut();
            let unsafety_state = fcx_ps.recurse(blk);
            replace(&mut *fcx_ps, unsafety_state)
        };

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
            for s in blk.stmts {
                self.check_stmt(s);
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
                coerce.coerce(self, &cause, tail_expr, tail_expr_ty);
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

        *self.ps.borrow_mut() = prev;
        ty
    }

    pub(in super::super) fn check_rustc_args_require_const(
        &self,
        def_id: DefId,
        hir_id: hir::HirId,
        span: Span,
    ) {
        // We're only interested in functions tagged with
        // #[rustc_args_required_const], so ignore anything that's not.
        if !self.tcx.has_attr(def_id, sym::rustc_args_required_const) {
            return;
        }

        // If our calling expression is indeed the function itself, we're good!
        // If not, generate an error that this can only be called directly.
        if let Node::Expr(expr) = self.tcx.hir().get(self.tcx.hir().get_parent_node(hir_id)) {
            if let ExprKind::Call(ref callee, ..) = expr.kind {
                if callee.hir_id == hir_id {
                    return;
                }
            }
        }

        self.tcx.sess.span_err(
            span,
            "this function can only be invoked directly, not through a function pointer",
        );
    }

    /// A common error is to add an extra semicolon:
    ///
    /// ```
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
        err: &mut DiagnosticBuilder<'_>,
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
                    "consider removing this semicolon",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    fn parent_item_span(&self, id: hir::HirId) -> Option<Span> {
        let node = self.tcx.hir().get(self.tcx.hir().get_parent_item(id));
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
        let parent = self.tcx.hir().get(self.tcx.hir().get_parent_item(blk_id));
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
        if let hir::ExprKind::Match(_, arms, _) = &expr.kind {
            let arm_spans: Vec<Span> = arms
                .iter()
                .filter_map(|arm| {
                    self.in_progress_typeck_results
                        .and_then(|typeck_results| {
                            typeck_results.borrow().node_type_opt(arm.body.hir_id)
                        })
                        .and_then(|arm_ty| {
                            if arm_ty.is_never() {
                                None
                            } else {
                                Some(match &arm.body.kind {
                                    // Point at the tail expression when possible.
                                    hir::ExprKind::Block(block, _) => {
                                        block.expr.as_ref().map(|e| e.span).unwrap_or(block.span)
                                    }
                                    _ => arm.body.span,
                                })
                            }
                        })
                })
                .collect();
            if arm_spans.len() == 1 {
                return arm_spans[0];
            }
        }
        expr.span
    }

    fn overwrite_local_ty_if_err(
        &self,
        local: &'tcx hir::Local<'tcx>,
        decl_ty: Ty<'tcx>,
        ty: Ty<'tcx>,
    ) {
        if ty.references_error() {
            // Override the types everywhere with `err()` to avoid knock on errors.
            self.write_ty(local.hir_id, ty);
            self.write_ty(local.pat.hir_id, ty);
            let local_ty = LocalTy { decl_ty, revealed_ty: ty };
            self.locals.borrow_mut().insert(local.hir_id, local_ty);
            self.locals.borrow_mut().insert(local.pat.hir_id, local_ty);
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
                let ty = AstConv::res_to_ty(self, self_ty, path, true);
                (path.res, ty)
            }
            QPath::TypeRelative(ref qself, ref segment) => {
                let ty = self.to_ty(qself);

                let res = if let hir::TyKind::Path(QPath::Resolved(_, ref path)) = qself.kind {
                    path.res
                } else {
                    Res::Err
                };
                let result =
                    AstConv::associated_path_to_ty(self, hir_id, path_span, ty, res, segment, true);
                let ty = result.map(|(ty, _, _)| ty).unwrap_or_else(|_| self.tcx().ty_error());
                let result = result.map(|(_, kind, def_id)| (kind, def_id));

                // Write back the new resolution.
                self.write_resolution(hir_id, result);

                (result.map(|(kind, def_id)| Res::Def(kind, def_id)).unwrap_or(Res::Err), ty)
            }
            QPath::LangItem(lang_item, span) => {
                self.resolve_lang_item_path(lang_item, span, hir_id)
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

            if let ty::PredicateAtom::Trait(predicate, _) =
                error.obligation.predicate.skip_binders()
            {
                // Collect the argument position for all arguments that could have caused this
                // `FulfillmentError`.
                let mut checked_tys = vec![];
                let mut coerced_tys = vec![];
                for (i, &type_pair) in final_arg_types.iter().enumerate() {
                    if let Some((checked_ty, coerced_ty)) = type_pair {
                        checked_tys.push((i, checked_ty));
                        coerced_tys.push((i, coerced_ty));
                    }
                }
                let mut referenced_in = vec![];
                for &(i, ty) in checked_tys.iter().chain(coerced_tys.iter()) {
                    let ty = self.resolve_vars_if_possible(ty);
                    // We walk the argument type because the argument's type could have
                    // been `Option<T>`, but the `FulfillmentError` references `T`.
                    if ty.walk().any(|arg| arg == predicate.self_ty().into()) {
                        referenced_in.push(i);
                    }
                }

                // Both checked and coerced types could have matched, thus we need to remove
                // duplicates.

                // We sort primitive type usize here and can use unstable sort
                referenced_in.sort_unstable();
                referenced_in.dedup();

                if let (Some(ref_in), None) = (referenced_in.pop(), referenced_in.pop()) {
                    // We make sure that only *one* argument matches the obligation failure
                    // and we assign the obligation's span to its expression's.
                    error.obligation.cause.make_mut().span = args[ref_in].span;
                    error.points_at_arg_span = true;
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
            if let hir::ExprKind::Path(qpath) = &path.kind {
                if let hir::QPath::Resolved(_, path) = &qpath {
                    for error in errors {
                        if let ty::PredicateAtom::Trait(predicate, _) =
                            error.obligation.predicate.skip_binders()
                        {
                            // If any of the type arguments in this path segment caused the
                            // `FullfillmentError`, point at its span (#61860).
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
                                        let ty = AstConv::ast_ty_to_ty(self, hir_ty);
                                        let ty = self.resolve_vars_if_possible(ty);
                                        if ty == predicate.self_ty() {
                                            error.obligation.cause.make_mut().span = hir_ty.span;
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
}
