use crate::astconv::{
    AstConv, CreateSubstsForGenericArgsCtxt, ExplicitLateBound, GenericArgCountMismatch,
    GenericArgCountResult, GenericArgPosition,
};
use crate::errors::AssocTypeBindingNotAllowed;
use rustc_ast::ast::ParamKindOrd;
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticId, ErrorReported};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::GenericArg;
use rustc_middle::ty::{
    self, subst, subst::SubstsRef, GenericParamDef, GenericParamDefKind, Ty, TyCtxt,
};
use rustc_session::{lint::builtin::LATE_BOUND_LIFETIME_ARGUMENTS, Session};
use rustc_span::{symbol::kw, MultiSpan, Span};

use smallvec::SmallVec;

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    /// Report an error that a generic argument did not match the generic parameter that was
    /// expected.
    fn generic_arg_mismatch_err(
        sess: &Session,
        arg: &GenericArg<'_>,
        kind: &'static str,
        possible_ordering_error: bool,
        help: Option<&str>,
    ) {
        let mut err = struct_span_err!(
            sess,
            arg.span(),
            E0747,
            "{} provided when a {} was expected",
            arg.descr(),
            kind,
        );

        let unordered = sess.features_untracked().const_generics;
        let kind_ord = match kind {
            "lifetime" => ParamKindOrd::Lifetime,
            "type" => ParamKindOrd::Type,
            "constant" => ParamKindOrd::Const { unordered },
            // It's more concise to match on the string representation, though it means
            // the match is non-exhaustive.
            _ => bug!("invalid generic parameter kind {}", kind),
        };

        if let ParamKindOrd::Const { .. } = kind_ord {
            if let GenericArg::Type(hir::Ty { kind: hir::TyKind::Infer, .. }) = arg {
                err.help("const arguments cannot yet be inferred with `_`");
            }
        }

        let arg_ord = match arg {
            GenericArg::Lifetime(_) => ParamKindOrd::Lifetime,
            GenericArg::Type(_) => ParamKindOrd::Type,
            GenericArg::Const(_) => ParamKindOrd::Const { unordered },
        };

        if matches!(arg, GenericArg::Type(hir::Ty { kind: hir::TyKind::Path { .. }, .. }))
            && matches!(kind_ord, ParamKindOrd::Const { .. })
        {
            let suggestions = vec![
                (arg.span().shrink_to_lo(), String::from("{ ")),
                (arg.span().shrink_to_hi(), String::from(" }")),
            ];
            err.multipart_suggestion(
                "if this generic argument was intended as a const parameter, \
                try surrounding it with braces:",
                suggestions,
                Applicability::MaybeIncorrect,
            );
        }

        // This note is only true when generic parameters are strictly ordered by their kind.
        if possible_ordering_error && kind_ord.cmp(&arg_ord) != core::cmp::Ordering::Equal {
            let (first, last) =
                if kind_ord < arg_ord { (kind, arg.descr()) } else { (arg.descr(), kind) };
            err.note(&format!("{} arguments must be provided before {} arguments", first, last));
            if let Some(help) = help {
                err.help(help);
            }
        }

        err.emit();
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
    pub fn create_substs_for_generic_args<'a>(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        parent_substs: &[subst::GenericArg<'tcx>],
        has_self: bool,
        self_ty: Option<Ty<'tcx>>,
        arg_count: GenericArgCountResult,
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
                            | (GenericArg::Type(_), GenericParamDefKind::Type { .. }, _)
                            | (GenericArg::Const(_), GenericParamDefKind::Const, _) => {
                                substs.push(ctx.provided_kind(param, arg));
                                args.next();
                                params.next();
                            }
                            (
                                GenericArg::Type(_) | GenericArg::Const(_),
                                GenericParamDefKind::Lifetime,
                                _,
                            ) => {
                                // We expected a lifetime argument, but got a type or const
                                // argument. That means we're inferring the lifetimes.
                                substs.push(ctx.inferred_kind(None, param, infer_args));
                                force_infer_lt = Some(arg);
                                params.next();
                            }
                            (GenericArg::Lifetime(_), _, ExplicitLateBound::Yes) => {
                                // We've come across a lifetime when we expected something else in
                                // the presence of explicit late bounds. This is most likely
                                // due to the presence of the explicit bound so we're just going to
                                // ignore it.
                                args.next();
                            }
                            (_, kind, _) => {
                                // We expected one kind of parameter, but the user provided
                                // another. This is an error. However, if we already know that
                                // the arguments don't match up with the parameters, we won't issue
                                // an additional error, as the user already knows what's wrong.
                                if arg_count.correct.is_ok()
                                    && arg_count.explicit_late_bound == ExplicitLateBound::No
                                {
                                    // We're going to iterate over the parameters to sort them out, and
                                    // show that order to the user as a possible order for the parameters
                                    let mut param_types_present = defs
                                        .params
                                        .clone()
                                        .into_iter()
                                        .map(|param| {
                                            (
                                                match param.kind {
                                                    GenericParamDefKind::Lifetime => {
                                                        ParamKindOrd::Lifetime
                                                    }
                                                    GenericParamDefKind::Type { .. } => {
                                                        ParamKindOrd::Type
                                                    }
                                                    GenericParamDefKind::Const => {
                                                        ParamKindOrd::Const {
                                                            unordered: tcx
                                                                .features()
                                                                .const_generics,
                                                        }
                                                    }
                                                },
                                                param,
                                            )
                                        })
                                        .collect::<Vec<(ParamKindOrd, GenericParamDef)>>();
                                    param_types_present.sort_by_key(|(ord, _)| *ord);
                                    let (mut param_types_present, ordered_params): (
                                        Vec<ParamKindOrd>,
                                        Vec<GenericParamDef>,
                                    ) = param_types_present.into_iter().unzip();
                                    param_types_present.dedup();

                                    Self::generic_arg_mismatch_err(
                                        tcx.sess,
                                        arg,
                                        kind.descr(),
                                        !args_iter.clone().is_sorted_by_key(|arg| match arg {
                                            GenericArg::Lifetime(_) => ParamKindOrd::Lifetime,
                                            GenericArg::Type(_) => ParamKindOrd::Type,
                                            GenericArg::Const(_) => ParamKindOrd::Const {
                                                unordered: tcx.features().const_generics,
                                            },
                                        }),
                                        Some(&format!(
                                            "reorder the arguments: {}: `<{}>`",
                                            param_types_present
                                                .into_iter()
                                                .map(|ord| format!("{}s", ord.to_string()))
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
                        //  1.  The number of arguments is incorrect. In this case, an error
                        //      will already have been emitted, and we can ignore it.
                        //  2.  There are late-bound lifetime parameters present, yet the
                        //      lifetime arguments have also been explicitly specified by the
                        //      user.
                        //  3.  We've inferred some lifetimes, which have been provided later (i.e.
                        //      after a type or const). We want to throw an error in this case.

                        if arg_count.correct.is_ok()
                            && arg_count.explicit_late_bound == ExplicitLateBound::No
                        {
                            let kind = arg.descr();
                            assert_eq!(kind, "lifetime");
                            let provided =
                                force_infer_lt.expect("lifetimes ought to have been inferred");
                            Self::generic_arg_mismatch_err(tcx.sess, provided, kind, false, None);
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

        tcx.intern_substs(&substs)
    }

    /// Checks that the correct number of generic arguments have been provided.
    /// Used specifically for function calls.
    pub fn check_generic_arg_count_for_call(
        tcx: TyCtxt<'_>,
        span: Span,
        def: &ty::Generics,
        seg: &hir::PathSegment<'_>,
        is_method_call: bool,
    ) -> GenericArgCountResult {
        let empty_args = hir::GenericArgs::none();
        let suppress_mismatch = Self::check_impl_trait(tcx, seg, &def);
        Self::check_generic_arg_count(
            tcx,
            span,
            def,
            if let Some(ref args) = seg.args { args } else { &empty_args },
            if is_method_call { GenericArgPosition::MethodCall } else { GenericArgPosition::Value },
            def.parent.is_none() && def.has_self, // `has_self`
            seg.infer_args || suppress_mismatch,  // `infer_args`
        )
    }

    /// Checks that the correct number of generic arguments have been provided.
    /// This is used both for datatypes and function calls.
    pub(crate) fn check_generic_arg_count(
        tcx: TyCtxt<'_>,
        span: Span,
        def: &ty::Generics,
        args: &hir::GenericArgs<'_>,
        position: GenericArgPosition,
        has_self: bool,
        infer_args: bool,
    ) -> GenericArgCountResult {
        // At this stage we are guaranteed that the generic arguments are in the correct order, e.g.
        // that lifetimes will proceed types. So it suffices to check the number of each generic
        // arguments in order to validate them with respect to the generic parameters.
        let param_counts = def.own_counts();
        let named_type_param_count = param_counts.types - has_self as usize;
        let arg_counts = args.own_counts();
        let infer_lifetimes = position != GenericArgPosition::Type && arg_counts.lifetimes == 0;

        let mut defaults: ty::GenericParamCount = Default::default();
        for param in &def.params {
            match param.kind {
                GenericParamDefKind::Lifetime => {}
                GenericParamDefKind::Type { has_default, .. } => {
                    defaults.types += has_default as usize
                }
                GenericParamDefKind::Const => {
                    // FIXME(const_generics_defaults)
                }
            };
        }

        if position != GenericArgPosition::Type && !args.bindings.is_empty() {
            AstConv::prohibit_assoc_ty_binding(tcx, args.bindings[0].span);
        }

        let explicit_late_bound =
            Self::prohibit_explicit_late_bound_lifetimes(tcx, def, args, position);

        let check_kind_count = |kind,
                                required,
                                permitted,
                                provided,
                                offset,
                                unexpected_spans: &mut Vec<Span>,
                                silent| {
            debug!(
                "check_kind_count: kind: {} required: {} permitted: {} provided: {} offset: {}",
                kind, required, permitted, provided, offset
            );
            // We enforce the following: `required` <= `provided` <= `permitted`.
            // For kinds without defaults (e.g.., lifetimes), `required == permitted`.
            // For other kinds (i.e., types), `permitted` may be greater than `required`.
            if required <= provided && provided <= permitted {
                return true;
            }

            if silent {
                return false;
            }

            // Unfortunately lifetime and type parameter mismatches are typically styled
            // differently in diagnostics, which means we have a few cases to consider here.
            let (bound, quantifier) = if required != permitted {
                if provided < required {
                    (required, "at least ")
                } else {
                    // provided > permitted
                    (permitted, "at most ")
                }
            } else {
                (required, "")
            };

            let (spans, labels) = if provided > permitted {
                // In the case when the user has provided too many arguments,
                // we want to point to the unexpected arguments.
                let (spans, labels): (Vec<Span>, Vec<String>) = args.args
                    [offset + permitted..offset + provided]
                    .iter()
                    .map(|arg| (arg.span(), format!("unexpected {} argument", arg.short_descr())))
                    .unzip();
                unexpected_spans.extend(spans.clone());
                (spans, labels)
            } else {
                (
                    vec![span],
                    vec![format!(
                        "expected {}{} {} argument{}",
                        quantifier,
                        bound,
                        kind,
                        pluralize!(bound),
                    )],
                )
            };

            let mut err = tcx.sess.struct_span_err_with_code(
                spans.clone(),
                &format!(
                    "wrong number of {} arguments: expected {}{}, found {}",
                    kind, quantifier, bound, provided,
                ),
                DiagnosticId::Error("E0107".into()),
            );
            for (span, label) in spans.into_iter().zip(labels) {
                err.span_label(span, label.as_str());
            }
            err.emit();
            false
        };

        let mut unexpected_spans = vec![];

        let lifetime_count_correct = check_kind_count(
            "lifetime",
            if infer_lifetimes { 0 } else { param_counts.lifetimes },
            param_counts.lifetimes,
            arg_counts.lifetimes,
            0,
            &mut unexpected_spans,
            explicit_late_bound == ExplicitLateBound::Yes,
        );

        let kind_str = if param_counts.consts + arg_counts.consts == 0 {
            "type"
        } else if named_type_param_count + arg_counts.types == 0 {
            "const"
        } else {
            "generic"
        };

        let arg_count_correct = check_kind_count(
            kind_str,
            if infer_args {
                0
            } else {
                param_counts.consts + named_type_param_count - defaults.types
            },
            param_counts.consts + named_type_param_count,
            arg_counts.consts + arg_counts.types,
            arg_counts.lifetimes,
            &mut unexpected_spans,
            false,
        );

        GenericArgCountResult {
            explicit_late_bound,
            correct: if lifetime_count_correct && arg_count_correct {
                Ok(())
            } else {
                Err(GenericArgCountMismatch {
                    reported: Some(ErrorReported),
                    invalid_args: unexpected_spans,
                })
            },
        }
    }

    /// Report error if there is an explicit type parameter when using `impl Trait`.
    pub(crate) fn check_impl_trait(
        tcx: TyCtxt<'_>,
        seg: &hir::PathSegment<'_>,
        generics: &ty::Generics,
    ) -> bool {
        let explicit = !seg.infer_args;
        let impl_trait = generics.params.iter().any(|param| {
            matches!(param.kind, ty::GenericParamDefKind::Type {
                synthetic:
                    Some(
                        hir::SyntheticTyParamKind::ImplTrait
                        | hir::SyntheticTyParamKind::FromAttr,
                    ),
                ..
            })
        });

        if explicit && impl_trait {
            let spans = seg
                .generic_args()
                .args
                .iter()
                .filter_map(|arg| match arg {
                    GenericArg::Type(_) | GenericArg::Const(_) => Some(arg.span()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let mut err = struct_span_err! {
                tcx.sess,
                spans.clone(),
                E0632,
                "cannot provide explicit generic arguments when `impl Trait` is \
                used in argument position"
            };

            for span in spans {
                err.span_label(span, "explicit generic argument not allowed");
            }

            err.emit();
        }

        impl_trait
    }

    /// Emits an error regarding forbidden type binding associations
    pub fn prohibit_assoc_ty_binding(tcx: TyCtxt<'_>, span: Span) {
        tcx.sess.emit_err(AssocTypeBindingNotAllowed { span });
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
        let arg_counts = args.own_counts();
        let infer_lifetimes = position != GenericArgPosition::Type && arg_counts.lifetimes == 0;

        if infer_lifetimes {
            ExplicitLateBound::No
        } else if let Some(span_late) = def.has_late_bound_regions {
            let msg = "cannot specify lifetime arguments explicitly \
                       if late bound lifetime parameters are present";
            let note = "the late bound lifetime parameter is introduced here";
            let span = args.args[0].span();
            if position == GenericArgPosition::Value
                && arg_counts.lifetimes != param_counts.lifetimes
            {
                let mut err = tcx.sess.struct_span_err(span, msg);
                err.span_note(span_late, note);
                err.emit();
            } else {
                let mut multispan = MultiSpan::from_span(span);
                multispan.push_span_label(span_late, note.to_string());
                tcx.struct_span_lint_hir(
                    LATE_BOUND_LIFETIME_ARGUMENTS,
                    args.args[0].id(),
                    multispan,
                    |lint| lint.build(msg).emit(),
                );
            }
            ExplicitLateBound::Yes
        } else {
            ExplicitLateBound::No
        }
    }
}
