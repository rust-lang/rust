use super::IsMethodCall;
use crate::astconv::{
    errors::prohibit_assoc_ty_binding, CreateSubstsForGenericArgsCtxt, ExplicitLateBound,
    GenericArgCountMismatch, GenericArgCountResult, GenericArgPosition,
};
use crate::structured_errors::{GenericArgsInfo, StructuredDiagnostic, WrongNumberOfGenericArgs};
use rustc_ast::ast::ParamKindOrd;
use rustc_errors::{struct_span_err, Applicability, Diagnostic, ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::GenericArg;
use rustc_middle::ty::{
    self, subst, subst::SubstsRef, GenericParamDef, GenericParamDefKind, IsSuggestable, Ty, TyCtxt,
};
use rustc_session::lint::builtin::LATE_BOUND_LIFETIME_ARGUMENTS;
use rustc_span::{symbol::kw, Span};
use smallvec::SmallVec;

/// Report an error that a generic argument did not match the generic parameter that was
/// expected.
fn generic_arg_mismatch_err(
    tcx: TyCtxt<'_>,
    arg: &GenericArg<'_>,
    param: &GenericParamDef,
    possible_ordering_error: bool,
    help: Option<&str>,
) -> ErrorGuaranteed {
    let sess = tcx.sess;
    let mut err = struct_span_err!(
        sess,
        arg.span(),
        E0747,
        "{} provided when a {} was expected",
        arg.descr(),
        param.kind.descr(),
    );

    if let GenericParamDefKind::Const { .. } = param.kind {
        if matches!(arg, GenericArg::Type(hir::Ty { kind: hir::TyKind::Infer, .. })) {
            err.help("const arguments cannot yet be inferred with `_`");
            if sess.is_nightly_build() {
                err.help("add `#![feature(generic_arg_infer)]` to the crate attributes to enable");
            }
        }
    }

    let add_braces_suggestion = |arg: &GenericArg<'_>, err: &mut Diagnostic| {
        let suggestions = vec![
            (arg.span().shrink_to_lo(), String::from("{ ")),
            (arg.span().shrink_to_hi(), String::from(" }")),
        ];
        err.multipart_suggestion(
            "if this generic argument was intended as a const parameter, \
                 surround it with braces",
            suggestions,
            Applicability::MaybeIncorrect,
        );
    };

    // Specific suggestion set for diagnostics
    match (arg, &param.kind) {
        (
            GenericArg::Type(hir::Ty {
                kind: hir::TyKind::Path(rustc_hir::QPath::Resolved(_, path)),
                ..
            }),
            GenericParamDefKind::Const { .. },
        ) => match path.res {
            Res::Err => {
                add_braces_suggestion(arg, &mut err);
                return err
                    .set_primary_message("unresolved item provided when a constant was expected")
                    .emit();
            }
            Res::Def(DefKind::TyParam, src_def_id) => {
                if let Some(param_local_id) = param.def_id.as_local() {
                    let param_name = tcx.hir().ty_param_name(param_local_id);
                    let param_type = tcx.type_of(param.def_id).subst_identity();
                    if param_type.is_suggestable(tcx, false) {
                        err.span_suggestion(
                            tcx.def_span(src_def_id),
                            "consider changing this type parameter to a const parameter",
                            format!("const {}: {}", param_name, param_type),
                            Applicability::MaybeIncorrect,
                        );
                    };
                }
            }
            _ => add_braces_suggestion(arg, &mut err),
        },
        (
            GenericArg::Type(hir::Ty { kind: hir::TyKind::Path(_), .. }),
            GenericParamDefKind::Const { .. },
        ) => add_braces_suggestion(arg, &mut err),
        (
            GenericArg::Type(hir::Ty { kind: hir::TyKind::Array(_, len), .. }),
            GenericParamDefKind::Const { .. },
        ) if tcx.type_of(param.def_id).skip_binder() == tcx.types.usize => {
            let snippet = sess.source_map().span_to_snippet(tcx.hir().span(len.hir_id()));
            if let Ok(snippet) = snippet {
                err.span_suggestion(
                    arg.span(),
                    "array type provided where a `usize` was expected, try",
                    format!("{{ {} }}", snippet),
                    Applicability::MaybeIncorrect,
                );
            }
        }
        (GenericArg::Const(cnst), GenericParamDefKind::Type { .. }) => {
            let body = tcx.hir().body(cnst.value.body);
            if let rustc_hir::ExprKind::Path(rustc_hir::QPath::Resolved(_, path)) = body.value.kind
            {
                if let Res::Def(DefKind::Fn { .. }, id) = path.res {
                    err.help(&format!("`{}` is a function item, not a type", tcx.item_name(id)));
                    err.help("function item types cannot be named directly");
                }
            }
        }
        _ => {}
    }

    let kind_ord = param.kind.to_ord();
    let arg_ord = arg.to_ord();

    // This note is only true when generic parameters are strictly ordered by their kind.
    if possible_ordering_error && kind_ord.cmp(&arg_ord) != core::cmp::Ordering::Equal {
        let (first, last) = if kind_ord < arg_ord {
            (param.kind.descr(), arg.descr())
        } else {
            (arg.descr(), param.kind.descr())
        };
        err.note(&format!("{} arguments must be provided before {} arguments", first, last));
        if let Some(help) = help {
            err.help(help);
        }
    }

    err.emit()
}

/// Creates the relevant generic argument substitutions
/// corresponding to a set of generic parameters. This is a
/// rather complex function. Let us try to explain the role
/// of each of its parameters:
///
/// To start, we are given the `def_id` of the thing we are
/// creating the substitutions for, and a partial set of
/// substitutions `parent_substs`. In general, the substitutions
/// for an item begin with substitutions for all the "parents" of
/// that item -- e.g., for a method it might include the
/// parameters from the impl.
///
/// Therefore, the method begins by walking down these parents,
/// starting with the outermost parent and proceed inwards until
/// it reaches `def_id`. For each parent `P`, it will check `parent_substs`
/// first to see if the parent's substitutions are listed in there. If so,
/// we can append those and move on. Otherwise, it invokes the
/// three callback functions:
///
/// - `args_for_def_id`: given the `DefId` `P`, supplies back the
///   generic arguments that were given to that parent from within
///   the path; so e.g., if you have `<T as Foo>::Bar`, the `DefId`
///   might refer to the trait `Foo`, and the arguments might be
///   `[T]`. The boolean value indicates whether to infer values
///   for arguments whose values were not explicitly provided.
/// - `provided_kind`: given the generic parameter and the value from `args_for_def_id`,
///   instantiate a `GenericArg`.
/// - `inferred_kind`: if no parameter was provided, and inference is enabled, then
///   creates a suitable inference variable.
pub fn create_substs_for_generic_args<'tcx, 'a>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    parent_substs: &[subst::GenericArg<'tcx>],
    has_self: bool,
    self_ty: Option<Ty<'tcx>>,
    arg_count: &GenericArgCountResult,
    ctx: &mut impl CreateSubstsForGenericArgsCtxt<'a, 'tcx>,
) -> SubstsRef<'tcx> {
    // Collect the segments of the path; we need to substitute arguments
    // for parameters throughout the entire path (wherever there are
    // generic parameters).
    let mut parent_defs = tcx.generics_of(def_id);
    let count = parent_defs.count();
    let mut stack = vec![(def_id, parent_defs)];
    while let Some(def_id) = parent_defs.parent {
        parent_defs = tcx.generics_of(def_id);
        stack.push((def_id, parent_defs));
    }

    // We manually build up the substitution, rather than using convenience
    // methods in `subst.rs`, so that we can iterate over the arguments and
    // parameters in lock-step linearly, instead of trying to match each pair.
    let mut substs: SmallVec<[subst::GenericArg<'tcx>; 8]> = SmallVec::with_capacity(count);
    // Iterate over each segment of the path.
    while let Some((def_id, defs)) = stack.pop() {
        let mut params = defs.params.iter().peekable();

        // If we have already computed substitutions for parents, we can use those directly.
        while let Some(&param) = params.peek() {
            if let Some(&kind) = parent_substs.get(param.index as usize) {
                substs.push(kind);
                params.next();
            } else {
                break;
            }
        }

        // `Self` is handled first, unless it's been handled in `parent_substs`.
        if has_self {
            if let Some(&param) = params.peek() {
                if param.index == 0 {
                    if let GenericParamDefKind::Type { .. } = param.kind {
                        substs.push(
                            self_ty
                                .map(|ty| ty.into())
                                .unwrap_or_else(|| ctx.inferred_kind(None, param, true)),
                        );
                        params.next();
                    }
                }
            }
        }

        // Check whether this segment takes generic arguments and the user has provided any.
        let (generic_args, infer_args) = ctx.args_for_def_id(def_id);

        let args_iter = generic_args.iter().flat_map(|generic_args| generic_args.args.iter());
        let mut args = args_iter.clone().peekable();

        // If we encounter a type or const when we expect a lifetime, we infer the lifetimes.
        // If we later encounter a lifetime, we know that the arguments were provided in the
        // wrong order. `force_infer_lt` records the type or const that forced lifetimes to be
        // inferred, so we can use it for diagnostics later.
        let mut force_infer_lt = None;

        loop {
            // We're going to iterate through the generic arguments that the user
            // provided, matching them with the generic parameters we expect.
            // Mismatches can occur as a result of elided lifetimes, or for malformed
            // input. We try to handle both sensibly.
            match (args.peek(), params.peek()) {
                (Some(&arg), Some(&param)) => {
                    match (arg, &param.kind, arg_count.explicit_late_bound) {
                        (GenericArg::Lifetime(_), GenericParamDefKind::Lifetime, _)
                        | (
                            GenericArg::Type(_) | GenericArg::Infer(_),
                            GenericParamDefKind::Type { .. },
                            _,
                        )
                        | (
                            GenericArg::Const(_) | GenericArg::Infer(_),
                            GenericParamDefKind::Const { .. },
                            _,
                        ) => {
                            substs.push(ctx.provided_kind(param, arg));
                            args.next();
                            params.next();
                        }
                        (
                            GenericArg::Infer(_) | GenericArg::Type(_) | GenericArg::Const(_),
                            GenericParamDefKind::Lifetime,
                            _,
                        ) => {
                            // We expected a lifetime argument, but got a type or const
                            // argument. That means we're inferring the lifetimes.
                            substs.push(ctx.inferred_kind(None, param, infer_args));
                            force_infer_lt = Some((arg, param));
                            params.next();
                        }
                        (GenericArg::Lifetime(_), _, ExplicitLateBound::Yes) => {
                            // We've come across a lifetime when we expected something else in
                            // the presence of explicit late bounds. This is most likely
                            // due to the presence of the explicit bound so we're just going to
                            // ignore it.
                            args.next();
                        }
                        (_, _, _) => {
                            // We expected one kind of parameter, but the user provided
                            // another. This is an error. However, if we already know that
                            // the arguments don't match up with the parameters, we won't issue
                            // an additional error, as the user already knows what's wrong.
                            if arg_count.correct.is_ok() {
                                // We're going to iterate over the parameters to sort them out, and
                                // show that order to the user as a possible order for the parameters
                                let mut param_types_present = defs
                                    .params
                                    .iter()
                                    .map(|param| (param.kind.to_ord(), param.clone()))
                                    .collect::<Vec<(ParamKindOrd, GenericParamDef)>>();
                                param_types_present.sort_by_key(|(ord, _)| *ord);
                                let (mut param_types_present, ordered_params): (
                                    Vec<ParamKindOrd>,
                                    Vec<GenericParamDef>,
                                ) = param_types_present.into_iter().unzip();
                                param_types_present.dedup();

                                generic_arg_mismatch_err(
                                    tcx,
                                    arg,
                                    param,
                                    !args_iter.clone().is_sorted_by_key(|arg| arg.to_ord()),
                                    Some(&format!(
                                        "reorder the arguments: {}: `<{}>`",
                                        param_types_present
                                            .into_iter()
                                            .map(|ord| format!("{}s", ord))
                                            .collect::<Vec<String>>()
                                            .join(", then "),
                                        ordered_params
                                            .into_iter()
                                            .filter_map(|param| {
                                                if param.name == kw::SelfUpper {
                                                    None
                                                } else {
                                                    Some(param.name.to_string())
                                                }
                                            })
                                            .collect::<Vec<String>>()
                                            .join(", ")
                                    )),
                                );
                            }

                            // We've reported the error, but we want to make sure that this
                            // problem doesn't bubble down and create additional, irrelevant
                            // errors. In this case, we're simply going to ignore the argument
                            // and any following arguments. The rest of the parameters will be
                            // inferred.
                            while args.next().is_some() {}
                        }
                    }
                }

                (Some(&arg), None) => {
                    // We should never be able to reach this point with well-formed input.
                    // There are three situations in which we can encounter this issue.
                    //
                    //  1. The number of arguments is incorrect. In this case, an error
                    //     will already have been emitted, and we can ignore it.
                    //  2. There are late-bound lifetime parameters present, yet the
                    //     lifetime arguments have also been explicitly specified by the
                    //     user.
                    //  3. We've inferred some lifetimes, which have been provided later (i.e.
                    //     after a type or const). We want to throw an error in this case.

                    if arg_count.correct.is_ok()
                        && arg_count.explicit_late_bound == ExplicitLateBound::No
                    {
                        let kind = arg.descr();
                        assert_eq!(kind, "lifetime");
                        let (provided_arg, param) =
                            force_infer_lt.expect("lifetimes ought to have been inferred");
                        generic_arg_mismatch_err(tcx, provided_arg, param, false, None);
                    }

                    break;
                }

                (None, Some(&param)) => {
                    // If there are fewer arguments than parameters, it means
                    // we're inferring the remaining arguments.
                    substs.push(ctx.inferred_kind(Some(&substs), param, infer_args));
                    params.next();
                }

                (None, None) => break,
            }
        }
    }

    tcx.mk_substs(&substs)
}

/// Checks that the correct number of generic arguments have been provided.
/// Used specifically for function calls.
pub fn check_generic_arg_count_for_call(
    tcx: TyCtxt<'_>,
    span: Span,
    def_id: DefId,
    generics: &ty::Generics,
    seg: &hir::PathSegment<'_>,
    is_method_call: IsMethodCall,
) -> GenericArgCountResult {
    let empty_args = hir::GenericArgs::none();
    let gen_args = seg.args.unwrap_or(&empty_args);
    let gen_pos = match is_method_call {
        IsMethodCall::Yes => GenericArgPosition::MethodCall,
        IsMethodCall::No => GenericArgPosition::Value,
    };
    let has_self = generics.parent.is_none() && generics.has_self;

    check_generic_arg_count(
        tcx,
        span,
        def_id,
        seg,
        generics,
        gen_args,
        gen_pos,
        has_self,
        seg.infer_args,
    )
}

/// Checks that the correct number of generic arguments have been provided.
/// This is used both for datatypes and function calls.
#[instrument(skip(tcx, gen_pos), level = "debug")]
pub(crate) fn check_generic_arg_count(
    tcx: TyCtxt<'_>,
    span: Span,
    def_id: DefId,
    seg: &hir::PathSegment<'_>,
    gen_params: &ty::Generics,
    gen_args: &hir::GenericArgs<'_>,
    gen_pos: GenericArgPosition,
    has_self: bool,
    infer_args: bool,
) -> GenericArgCountResult {
    let default_counts = gen_params.own_defaults();
    let param_counts = gen_params.own_counts();

    // Subtracting from param count to ensure type params synthesized from `impl Trait`
    // cannot be explicitly specified.
    let synth_type_param_count = gen_params
        .params
        .iter()
        .filter(|param| matches!(param.kind, ty::GenericParamDefKind::Type { synthetic: true, .. }))
        .count();
    let named_type_param_count = param_counts.types - has_self as usize - synth_type_param_count;
    let infer_lifetimes =
        (gen_pos != GenericArgPosition::Type || infer_args) && !gen_args.has_lifetime_params();

    if gen_pos != GenericArgPosition::Type && let Some(b) = gen_args.bindings.first() {
             prohibit_assoc_ty_binding(tcx, b.span, None);
        }

    let explicit_late_bound =
        prohibit_explicit_late_bound_lifetimes(tcx, gen_params, gen_args, gen_pos);

    let mut invalid_args = vec![];

    let mut check_lifetime_args = |min_expected_args: usize,
                                   max_expected_args: usize,
                                   provided_args: usize,
                                   late_bounds_ignore: bool| {
        if (min_expected_args..=max_expected_args).contains(&provided_args) {
            return Ok(());
        }

        if late_bounds_ignore {
            return Ok(());
        }

        if provided_args > max_expected_args {
            invalid_args.extend(
                gen_args.args[max_expected_args..provided_args].iter().map(|arg| arg.span()),
            );
        };

        let gen_args_info = if provided_args > min_expected_args {
            invalid_args.extend(
                gen_args.args[min_expected_args..provided_args].iter().map(|arg| arg.span()),
            );
            let num_redundant_args = provided_args - min_expected_args;
            GenericArgsInfo::ExcessLifetimes { num_redundant_args }
        } else {
            let num_missing_args = min_expected_args - provided_args;
            GenericArgsInfo::MissingLifetimes { num_missing_args }
        };

        let reported = WrongNumberOfGenericArgs::new(
            tcx,
            gen_args_info,
            seg,
            gen_params,
            has_self as usize,
            gen_args,
            def_id,
        )
        .diagnostic()
        .emit();

        Err(reported)
    };

    let min_expected_lifetime_args = if infer_lifetimes { 0 } else { param_counts.lifetimes };
    let max_expected_lifetime_args = param_counts.lifetimes;
    let num_provided_lifetime_args = gen_args.num_lifetime_params();

    let lifetimes_correct = check_lifetime_args(
        min_expected_lifetime_args,
        max_expected_lifetime_args,
        num_provided_lifetime_args,
        explicit_late_bound == ExplicitLateBound::Yes,
    );

    let mut check_types_and_consts = |expected_min,
                                      expected_max,
                                      expected_max_with_synth,
                                      provided,
                                      params_offset,
                                      args_offset| {
        debug!(
            ?expected_min,
            ?expected_max,
            ?provided,
            ?params_offset,
            ?args_offset,
            "check_types_and_consts"
        );
        if (expected_min..=expected_max).contains(&provided) {
            return Ok(());
        }

        let num_default_params = expected_max - expected_min;

        let gen_args_info = if provided > expected_max {
            invalid_args.extend(
                gen_args.args[args_offset + expected_max..args_offset + provided]
                    .iter()
                    .map(|arg| arg.span()),
            );
            let num_redundant_args = provided - expected_max;

            // Provide extra note if synthetic arguments like `impl Trait` are specified.
            let synth_provided = provided <= expected_max_with_synth;

            GenericArgsInfo::ExcessTypesOrConsts {
                num_redundant_args,
                num_default_params,
                args_offset,
                synth_provided,
            }
        } else {
            let num_missing_args = expected_max - provided;

            GenericArgsInfo::MissingTypesOrConsts {
                num_missing_args,
                num_default_params,
                args_offset,
            }
        };

        debug!(?gen_args_info);

        let reported = WrongNumberOfGenericArgs::new(
            tcx,
            gen_args_info,
            seg,
            gen_params,
            params_offset,
            gen_args,
            def_id,
        )
        .diagnostic()
        .emit_unless(gen_args.has_err());

        Err(reported)
    };

    let args_correct = {
        let expected_min = if infer_args {
            0
        } else {
            param_counts.consts + named_type_param_count
                - default_counts.types
                - default_counts.consts
        };
        debug!(?expected_min);
        debug!(arg_counts.lifetimes=?gen_args.num_lifetime_params());

        check_types_and_consts(
            expected_min,
            param_counts.consts + named_type_param_count,
            param_counts.consts + named_type_param_count + synth_type_param_count,
            gen_args.num_generic_params(),
            param_counts.lifetimes + has_self as usize,
            gen_args.num_lifetime_params(),
        )
    };

    GenericArgCountResult {
        explicit_late_bound,
        correct: lifetimes_correct
            .and(args_correct)
            .map_err(|reported| GenericArgCountMismatch { reported: Some(reported), invalid_args }),
    }
}

/// Prohibits explicit lifetime arguments if late-bound lifetime parameters
/// are present. This is used both for datatypes and function calls.
pub(crate) fn prohibit_explicit_late_bound_lifetimes(
    tcx: TyCtxt<'_>,
    def: &ty::Generics,
    args: &hir::GenericArgs<'_>,
    position: GenericArgPosition,
) -> ExplicitLateBound {
    let param_counts = def.own_counts();
    let infer_lifetimes = position != GenericArgPosition::Type && !args.has_lifetime_params();

    if infer_lifetimes {
        return ExplicitLateBound::No;
    }

    if let Some(span_late) = def.has_late_bound_regions {
        let msg = "cannot specify lifetime arguments explicitly \
                       if late bound lifetime parameters are present";
        let note = "the late bound lifetime parameter is introduced here";
        let span = args.args[0].span();

        if position == GenericArgPosition::Value
            && args.num_lifetime_params() != param_counts.lifetimes
        {
            let mut err = tcx.sess.struct_span_err(span, msg);
            err.span_note(span_late, note);
            err.emit();
        } else {
            let mut multispan = MultiSpan::from_span(span);
            multispan.push_span_label(span_late, note);
            tcx.struct_span_lint_hir(
                LATE_BOUND_LIFETIME_ARGUMENTS,
                args.args[0].hir_id(),
                multispan,
                msg,
                |lint| lint,
            );
        }

        ExplicitLateBound::Yes
    } else {
        ExplicitLateBound::No
    }
}
