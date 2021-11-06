use crate::structured_errors::StructuredDiagnostic;
use rustc_errors::{pluralize, Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_hir as hir;
use rustc_middle::hir::map::fn_sig;
use rustc_middle::middle::resolve_lifetime::LifetimeScopeForPath;
use rustc_middle::ty::{self as ty, TyCtxt};
use rustc_session::Session;
use rustc_span::{def_id::DefId, MultiSpan};

use GenericArgsInfo::*;

/// Handles the `wrong number of type / lifetime / ... arguments` family of error messages.
pub struct WrongNumberOfGenericArgs<'a, 'tcx> {
    crate tcx: TyCtxt<'tcx>,

    crate angle_brackets: AngleBrackets,

    crate gen_args_info: GenericArgsInfo,

    /// Offending path segment
    crate path_segment: &'a hir::PathSegment<'a>,

    /// Generic parameters as expected by type or trait
    crate gen_params: &'a ty::Generics,

    /// Index offset into parameters. Depends on whether `Self` is included and on
    /// number of lifetime parameters in case we're processing missing or redundant
    /// type or constant arguments.
    crate params_offset: usize,

    /// Generic arguments as provided by user
    crate gen_args: &'a hir::GenericArgs<'a>,

    /// DefId of the generic type
    crate def_id: DefId,
}

// Provides information about the kind of arguments that were provided for
// the PathSegment, for which missing generic arguments were detected
#[derive(Debug)]
pub(crate) enum AngleBrackets {
    // No angle brackets were provided, but generic arguments exist in elided form
    Implied,

    // No angle brackets were provided
    Missing,

    // Angle brackets are available, but missing some generic arguments
    Available,
}

// Information about the kind of arguments that are either missing or are unexpected
#[derive(Debug)]
pub enum GenericArgsInfo {
    MissingLifetimes {
        num_missing_args: usize,
    },
    ExcessLifetimes {
        num_redundant_args: usize,
    },
    MissingTypesOrConsts {
        num_missing_args: usize,

        // type or const generic arguments can have default values
        num_default_params: usize,

        // lifetime arguments precede type and const parameters, this
        // field gives the number of generic lifetime arguments to let
        // us infer the position of type and const generic arguments
        // in the angle brackets
        args_offset: usize,
    },

    ExcessTypesOrConsts {
        num_redundant_args: usize,

        // type or const generic arguments can have default values
        num_default_params: usize,

        // lifetime arguments precede type and const parameters, this
        // field gives the number of generic lifetime arguments to let
        // us infer the position of type and const generic arguments
        // in the angle brackets
        args_offset: usize,
    },
}

impl<'a, 'tcx> WrongNumberOfGenericArgs<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        gen_args_info: GenericArgsInfo,
        path_segment: &'a hir::PathSegment<'_>,
        gen_params: &'a ty::Generics,
        params_offset: usize,
        gen_args: &'a hir::GenericArgs<'a>,
        def_id: DefId,
    ) -> Self {
        let angle_brackets = if gen_args.span_ext().is_none() {
            if gen_args.is_empty() { AngleBrackets::Missing } else { AngleBrackets::Implied }
        } else {
            AngleBrackets::Available
        };

        Self {
            tcx,
            angle_brackets,
            gen_args_info,
            path_segment,
            gen_params,
            params_offset,
            gen_args,
            def_id,
        }
    }

    fn missing_lifetimes(&self) -> bool {
        match self.gen_args_info {
            MissingLifetimes { .. } | ExcessLifetimes { .. } => true,
            MissingTypesOrConsts { .. } | ExcessTypesOrConsts { .. } => false,
        }
    }

    fn kind(&self) -> String {
        if self.missing_lifetimes() { "lifetime".to_string() } else { "generic".to_string() }
    }

    fn num_provided_args(&self) -> usize {
        if self.missing_lifetimes() {
            self.num_provided_lifetime_args()
        } else {
            self.num_provided_type_or_const_args()
        }
    }

    fn num_provided_lifetime_args(&self) -> usize {
        match self.angle_brackets {
            AngleBrackets::Missing => 0,
            // Only lifetime arguments can be implied
            AngleBrackets::Implied => self.gen_args.args.len(),
            AngleBrackets::Available => self.gen_args.num_lifetime_params(),
        }
    }

    fn num_provided_type_or_const_args(&self) -> usize {
        match self.angle_brackets {
            AngleBrackets::Missing => 0,
            // Only lifetime arguments can be implied
            AngleBrackets::Implied => 0,
            AngleBrackets::Available => self.gen_args.num_generic_params(),
        }
    }

    fn num_expected_lifetime_args(&self) -> usize {
        let num_provided_args = self.num_provided_lifetime_args();
        match self.gen_args_info {
            MissingLifetimes { num_missing_args } => num_provided_args + num_missing_args,
            ExcessLifetimes { num_redundant_args } => num_provided_args - num_redundant_args,
            _ => 0,
        }
    }

    fn num_expected_type_or_const_args(&self) -> usize {
        let num_provided_args = self.num_provided_type_or_const_args();
        match self.gen_args_info {
            MissingTypesOrConsts { num_missing_args, .. } => num_provided_args + num_missing_args,
            ExcessTypesOrConsts { num_redundant_args, .. } => {
                num_provided_args - num_redundant_args
            }
            _ => 0,
        }
    }

    // Gives the number of expected arguments taking into account default arguments
    fn num_expected_type_or_const_args_including_defaults(&self) -> usize {
        let provided_args = self.num_provided_type_or_const_args();
        match self.gen_args_info {
            MissingTypesOrConsts { num_missing_args, num_default_params, .. } => {
                provided_args + num_missing_args - num_default_params
            }
            ExcessTypesOrConsts { num_redundant_args, num_default_params, .. } => {
                provided_args - num_redundant_args - num_default_params
            }
            _ => 0,
        }
    }

    fn num_missing_lifetime_args(&self) -> usize {
        let missing_args = self.num_expected_lifetime_args() - self.num_provided_lifetime_args();
        assert!(missing_args > 0);
        missing_args
    }

    fn num_missing_type_or_const_args(&self) -> usize {
        let missing_args = self.num_expected_type_or_const_args_including_defaults()
            - self.num_provided_type_or_const_args();
        assert!(missing_args > 0);
        missing_args
    }

    fn num_excess_lifetime_args(&self) -> usize {
        match self.gen_args_info {
            ExcessLifetimes { num_redundant_args } => num_redundant_args,
            _ => 0,
        }
    }

    fn num_excess_type_or_const_args(&self) -> usize {
        match self.gen_args_info {
            ExcessTypesOrConsts { num_redundant_args, .. } => num_redundant_args,
            _ => 0,
        }
    }

    fn too_many_args_provided(&self) -> bool {
        match self.gen_args_info {
            MissingLifetimes { .. } | MissingTypesOrConsts { .. } => false,
            ExcessLifetimes { num_redundant_args }
            | ExcessTypesOrConsts { num_redundant_args, .. } => {
                assert!(num_redundant_args > 0);
                true
            }
        }
    }

    fn not_enough_args_provided(&self) -> bool {
        match self.gen_args_info {
            MissingLifetimes { num_missing_args }
            | MissingTypesOrConsts { num_missing_args, .. } => {
                assert!(num_missing_args > 0);
                true
            }
            ExcessLifetimes { .. } | ExcessTypesOrConsts { .. } => false,
        }
    }

    // Helper method to get the index offset in angle brackets, at which type or const arguments
    // start appearing
    fn get_lifetime_args_offset(&self) -> usize {
        match self.gen_args_info {
            MissingLifetimes { .. } | ExcessLifetimes { .. } => 0,
            MissingTypesOrConsts { args_offset, .. } | ExcessTypesOrConsts { args_offset, .. } => {
                args_offset
            }
        }
    }

    fn get_num_default_params(&self) -> usize {
        match self.gen_args_info {
            MissingTypesOrConsts { num_default_params, .. }
            | ExcessTypesOrConsts { num_default_params, .. } => num_default_params,
            _ => 0,
        }
    }

    // Helper function to choose a quantifier word for the number of expected arguments
    // and to give a bound for the number of expected arguments
    fn get_quantifier_and_bound(&self) -> (&'static str, usize) {
        if self.get_num_default_params() == 0 {
            match self.gen_args_info {
                MissingLifetimes { .. } | ExcessLifetimes { .. } => {
                    ("", self.num_expected_lifetime_args())
                }
                MissingTypesOrConsts { .. } | ExcessTypesOrConsts { .. } => {
                    ("", self.num_expected_type_or_const_args())
                }
            }
        } else {
            match self.gen_args_info {
                MissingLifetimes { .. } => ("at least ", self.num_expected_lifetime_args()),
                MissingTypesOrConsts { .. } => {
                    ("at least ", self.num_expected_type_or_const_args_including_defaults())
                }
                ExcessLifetimes { .. } => ("at most ", self.num_expected_lifetime_args()),
                ExcessTypesOrConsts { .. } => ("at most ", self.num_expected_type_or_const_args()),
            }
        }
    }

    // Creates lifetime name suggestions from the lifetime parameter names
    fn get_lifetime_args_suggestions_from_param_names(&self, num_params_to_take: usize) -> String {
        self.gen_params
            .params
            .iter()
            .skip(self.params_offset + self.num_provided_lifetime_args())
            .take(num_params_to_take)
            .map(|param| param.name.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    // Creates type or constant name suggestions from the provided parameter names
    fn get_type_or_const_args_suggestions_from_param_names(
        &self,
        num_params_to_take: usize,
    ) -> String {
        let fn_sig = self.tcx.hir().get_if_local(self.def_id).and_then(fn_sig);
        let is_used_in_input = |def_id| {
            fn_sig.map_or(false, |fn_sig| {
                fn_sig.decl.inputs.iter().any(|ty| match ty.kind {
                    hir::TyKind::Path(hir::QPath::Resolved(
                        None,
                        hir::Path { res: hir::def::Res::Def(_, id), .. },
                    )) => *id == def_id,
                    _ => false,
                })
            })
        };
        self.gen_params
            .params
            .iter()
            .skip(self.params_offset + self.num_provided_type_or_const_args())
            .take(num_params_to_take)
            .map(|param| match param.kind {
                // This is being infered from the item's inputs, no need to set it.
                ty::GenericParamDefKind::Type { .. } if is_used_in_input(param.def_id) => {
                    "_".to_string()
                }
                _ => param.name.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn create_error_message(&self) -> String {
        let def_path = self.tcx.def_path_str(self.def_id);
        let def_kind = self.tcx.def_kind(self.def_id).descr(self.def_id);
        let (quantifier, bound) = self.get_quantifier_and_bound();
        let kind = self.kind();
        let provided_lt_args = self.num_provided_lifetime_args();
        let provided_type_or_const_args = self.num_provided_type_or_const_args();

        let get_verb = |num_args| if num_args == 1 { "was" } else { "were" };

        let (provided_args_str, verb) = match self.gen_args_info {
            MissingLifetimes { .. } | ExcessLifetimes { .. } => (
                format!("{} lifetime argument{}", provided_lt_args, pluralize!(provided_lt_args)),
                get_verb(provided_lt_args),
            ),
            MissingTypesOrConsts { .. } | ExcessTypesOrConsts { .. } => (
                format!(
                    "{} generic argument{}",
                    provided_type_or_const_args,
                    pluralize!(provided_type_or_const_args)
                ),
                get_verb(provided_type_or_const_args),
            ),
        };

        if self.gen_args.span_ext().is_some() {
            format!(
                "this {} takes {}{} {} argument{} but {} {} supplied",
                def_kind,
                quantifier,
                bound,
                kind,
                pluralize!(bound),
                provided_args_str.as_str(),
                verb
            )
        } else {
            format!("missing generics for {} `{}`", def_kind, def_path)
        }
    }

    fn start_diagnostics(&self) -> DiagnosticBuilder<'tcx> {
        let span = self.path_segment.ident.span;
        let msg = self.create_error_message();

        self.tcx.sess.struct_span_err_with_code(span, &msg, self.code())
    }

    /// Builds the `expected 1 type argument / supplied 2 type arguments` message.
    fn notify(&self, err: &mut DiagnosticBuilder<'_>) {
        let (quantifier, bound) = self.get_quantifier_and_bound();
        let provided_args = self.num_provided_args();

        err.span_label(
            self.path_segment.ident.span,
            format!(
                "expected {}{} {} argument{}",
                quantifier,
                bound,
                self.kind(),
                pluralize!(bound),
            ),
        );

        // When too many arguments were provided, we don't highlight each of them, because it
        // would overlap with the suggestion to remove them:
        //
        // ```
        // type Foo = Bar<usize, usize>;
        //                -----  ----- supplied 2 type arguments
        //                     ^^^^^^^ remove this type argument
        // ```
        if self.too_many_args_provided() {
            return;
        }

        let args = self
            .gen_args
            .args
            .iter()
            .skip(self.get_lifetime_args_offset())
            .take(provided_args)
            .enumerate();

        for (i, arg) in args {
            err.span_label(
                arg.span(),
                if i + 1 == provided_args {
                    format!(
                        "supplied {} {} argument{}",
                        provided_args,
                        self.kind(),
                        pluralize!(provided_args)
                    )
                } else {
                    String::new()
                },
            );
        }
    }

    fn suggest(&self, err: &mut DiagnosticBuilder<'_>) {
        debug!(
            "suggest(self.provided {:?}, self.gen_args.span(): {:?})",
            self.num_provided_args(),
            self.gen_args.span(),
        );

        match self.angle_brackets {
            AngleBrackets::Missing | AngleBrackets::Implied => self.suggest_adding_args(err),
            AngleBrackets::Available => {
                if self.not_enough_args_provided() {
                    self.suggest_adding_args(err);
                } else if self.too_many_args_provided() {
                    self.suggest_removing_args_or_generics(err);
                } else {
                    unreachable!();
                }
            }
        }
    }

    /// Suggests to add missing argument(s) when current invocation site already contains some
    /// generics:
    ///
    /// ```text
    /// type Map = HashMap<String>;
    /// ```
    fn suggest_adding_args(&self, err: &mut DiagnosticBuilder<'_>) {
        if self.gen_args.parenthesized {
            return;
        }

        match self.gen_args_info {
            MissingLifetimes { .. } => {
                self.suggest_adding_lifetime_args(err);
            }
            MissingTypesOrConsts { .. } => {
                self.suggest_adding_type_and_const_args(err);
            }
            _ => unreachable!(),
        }
    }

    fn suggest_adding_lifetime_args(&self, err: &mut DiagnosticBuilder<'_>) {
        debug!("suggest_adding_lifetime_args(path_segment: {:?})", self.path_segment);
        let num_missing_args = self.num_missing_lifetime_args();
        let num_params_to_take = num_missing_args;
        let msg = format!("add missing {} argument{}", self.kind(), pluralize!(num_missing_args));

        // we first try to get lifetime name suggestions from scope or elision information. If none is
        // available we use the parameter defintions
        let suggested_args = if let Some(hir_id) = self.path_segment.hir_id {
            if let Some(lifetimes_in_scope) = self.tcx.lifetime_scope(hir_id) {
                match lifetimes_in_scope {
                    LifetimeScopeForPath::NonElided(param_names) => {
                        debug!("NonElided(param_names: {:?})", param_names);

                        if param_names.len() >= num_params_to_take {
                            // use lifetime parameters in scope for suggestions
                            param_names
                                .iter()
                                .take(num_params_to_take)
                                .map(|p| (*p).clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        } else {
                            // Not enough lifetime arguments in scope -> create suggestions from
                            // lifetime parameter names in definition. An error for the incorrect
                            // lifetime scope will be output later.
                            self.get_lifetime_args_suggestions_from_param_names(num_params_to_take)
                        }
                    }
                    LifetimeScopeForPath::Elided => {
                        debug!("Elided");
                        // use suggestions of the form `<'_, '_>` in case lifetime can be elided
                        ["'_"].repeat(num_params_to_take).join(",")
                    }
                }
            } else {
                self.get_lifetime_args_suggestions_from_param_names(num_params_to_take)
            }
        } else {
            self.get_lifetime_args_suggestions_from_param_names(num_params_to_take)
        };

        debug!("suggested_args: {:?}", &suggested_args);

        match self.angle_brackets {
            AngleBrackets::Missing => {
                let span = self.path_segment.ident.span;

                // insert a suggestion of the form "Y<'a, 'b>"
                let ident = self.path_segment.ident.name.to_ident_string();
                let sugg = format!("{}<{}>", ident, suggested_args);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(span, &msg, sugg, Applicability::HasPlaceholders);
            }

            AngleBrackets::Available => {
                let (sugg_span, is_first) = if self.num_provided_lifetime_args() == 0 {
                    (self.gen_args.span().unwrap().shrink_to_lo(), true)
                } else {
                    let last_lt = &self.gen_args.args[self.num_provided_lifetime_args() - 1];
                    (last_lt.span().shrink_to_hi(), false)
                };
                let has_non_lt_args = self.num_provided_type_or_const_args() != 0;
                let has_bindings = !self.gen_args.bindings.is_empty();

                let sugg_prefix = if is_first { "" } else { ", " };
                let sugg_suffix =
                    if is_first && (has_non_lt_args || has_bindings) { ", " } else { "" };

                let sugg = format!("{}{}{}", sugg_prefix, suggested_args, sugg_suffix);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(sugg_span, &msg, sugg, Applicability::HasPlaceholders);
            }
            AngleBrackets::Implied => {
                // We never encounter missing lifetimes in situations in which lifetimes are elided
                unreachable!();
            }
        }
    }

    fn suggest_adding_type_and_const_args(&self, err: &mut DiagnosticBuilder<'_>) {
        let num_missing_args = self.num_missing_type_or_const_args();
        let msg = format!("add missing {} argument{}", self.kind(), pluralize!(num_missing_args));

        let suggested_args =
            self.get_type_or_const_args_suggestions_from_param_names(num_missing_args);
        debug!("suggested_args: {:?}", suggested_args);

        match self.angle_brackets {
            AngleBrackets::Missing | AngleBrackets::Implied => {
                let span = self.path_segment.ident.span;

                // insert a suggestion of the form "Y<T, U>"
                let ident = self.path_segment.ident.name.to_ident_string();
                let sugg = format!("{}<{}>", ident, suggested_args);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(span, &msg, sugg, Applicability::HasPlaceholders);
            }
            AngleBrackets::Available => {
                let gen_args_span = self.gen_args.span().unwrap();
                let sugg_offset =
                    self.get_lifetime_args_offset() + self.num_provided_type_or_const_args();

                let (sugg_span, is_first) = if sugg_offset == 0 {
                    (gen_args_span.shrink_to_lo(), true)
                } else {
                    let arg_span = self.gen_args.args[sugg_offset - 1].span();
                    // If we came here then inferred lifetimes's spans can only point
                    // to either the opening bracket or to the space right after.
                    // Both of these spans have an `hi` lower than or equal to the span
                    // of the generics excluding the brackets.
                    // This allows us to check if `arg_span` is the artificial span of
                    // an inferred lifetime, in which case the generic we're suggesting to
                    // add will be the first visible, even if it isn't the actual first generic.
                    (arg_span.shrink_to_hi(), arg_span.hi() <= gen_args_span.lo())
                };

                let sugg_prefix = if is_first { "" } else { ", " };
                let sugg_suffix =
                    if is_first && !self.gen_args.bindings.is_empty() { ", " } else { "" };

                let sugg = format!("{}{}{}", sugg_prefix, suggested_args, sugg_suffix);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(sugg_span, &msg, sugg, Applicability::HasPlaceholders);
            }
        }
    }

    /// Suggests to remove redundant argument(s):
    ///
    /// ```text
    /// type Map = HashMap<String, String, String, String>;
    /// ```
    fn suggest_removing_args_or_generics(&self, err: &mut DiagnosticBuilder<'_>) {
        let num_provided_lt_args = self.num_provided_lifetime_args();
        let num_provided_type_const_args = self.num_provided_type_or_const_args();
        let num_provided_args = num_provided_lt_args + num_provided_type_const_args;
        assert!(num_provided_args > 0);

        let num_redundant_lt_args = self.num_excess_lifetime_args();
        let num_redundant_type_or_const_args = self.num_excess_type_or_const_args();
        let num_redundant_args = num_redundant_lt_args + num_redundant_type_or_const_args;

        let redundant_lifetime_args = num_redundant_lt_args > 0;
        let redundant_type_or_const_args = num_redundant_type_or_const_args > 0;

        let remove_entire_generics = num_redundant_args >= self.gen_args.args.len();

        let remove_lifetime_args = |err: &mut DiagnosticBuilder<'_>| {
            let mut lt_arg_spans = Vec::new();
            let mut found_redundant = false;
            for arg in self.gen_args.args {
                if let hir::GenericArg::Lifetime(_) = arg {
                    lt_arg_spans.push(arg.span());
                    if lt_arg_spans.len() > self.num_expected_lifetime_args() {
                        found_redundant = true;
                    }
                } else if found_redundant {
                    // Argument which is redundant and separated like this `'c`
                    // is not included to avoid including `Bar` in span.
                    // ```
                    // type Foo<'a, T> = &'a T;
                    // let _: Foo<'a, 'b, Bar, 'c>;
                    // ```
                    break;
                }
            }

            let span_lo_redundant_lt_args = lt_arg_spans[self.num_expected_lifetime_args()];
            let span_hi_redundant_lt_args = lt_arg_spans[lt_arg_spans.len() - 1];

            let span_redundant_lt_args = span_lo_redundant_lt_args.to(span_hi_redundant_lt_args);
            debug!("span_redundant_lt_args: {:?}", span_redundant_lt_args);

            let num_redundant_lt_args = lt_arg_spans.len() - self.num_expected_lifetime_args();
            let msg_lifetimes = format!(
                "remove {} {} argument{}",
                if num_redundant_lt_args == 1 { "this" } else { "these" },
                "lifetime",
                pluralize!(num_redundant_lt_args),
            );

            err.span_suggestion(
                span_redundant_lt_args,
                &msg_lifetimes,
                String::new(),
                Applicability::MaybeIncorrect,
            );
        };

        let remove_type_or_const_args = |err: &mut DiagnosticBuilder<'_>| {
            let mut gen_arg_spans = Vec::new();
            let mut found_redundant = false;
            for arg in self.gen_args.args {
                match arg {
                    hir::GenericArg::Type(_)
                    | hir::GenericArg::Const(_)
                    | hir::GenericArg::Infer(_) => {
                        gen_arg_spans.push(arg.span());
                        if gen_arg_spans.len() > self.num_expected_type_or_const_args() {
                            found_redundant = true;
                        }
                    }
                    _ if found_redundant => break,
                    _ => {}
                }
            }

            let span_lo_redundant_type_or_const_args =
                gen_arg_spans[self.num_expected_type_or_const_args()];
            let span_hi_redundant_type_or_const_args = gen_arg_spans[gen_arg_spans.len() - 1];

            let span_redundant_type_or_const_args =
                span_lo_redundant_type_or_const_args.to(span_hi_redundant_type_or_const_args);
            debug!("span_redundant_type_or_const_args: {:?}", span_redundant_type_or_const_args);

            let num_redundant_gen_args =
                gen_arg_spans.len() - self.num_expected_type_or_const_args();
            let msg_types_or_consts = format!(
                "remove {} {} argument{}",
                if num_redundant_gen_args == 1 { "this" } else { "these" },
                "generic",
                pluralize!(num_redundant_type_or_const_args),
            );

            err.span_suggestion(
                span_redundant_type_or_const_args,
                &msg_types_or_consts,
                String::new(),
                Applicability::MaybeIncorrect,
            );
        };

        if remove_entire_generics {
            let span = self
                .path_segment
                .args
                .unwrap()
                .span_ext()
                .unwrap()
                .with_lo(self.path_segment.ident.span.hi());

            let msg = format!(
                "remove these {}generics",
                if self.gen_args.parenthesized { "parenthetical " } else { "" },
            );

            err.span_suggestion(span, &msg, String::new(), Applicability::MaybeIncorrect);
        } else if redundant_lifetime_args && redundant_type_or_const_args {
            remove_lifetime_args(err);
            remove_type_or_const_args(err);
        } else if redundant_lifetime_args {
            remove_lifetime_args(err);
        } else {
            assert!(redundant_type_or_const_args);
            remove_type_or_const_args(err);
        }
    }

    /// Builds the `type defined here` message.
    fn show_definition(&self, err: &mut DiagnosticBuilder<'_>) {
        let mut spans: MultiSpan = if let Some(def_span) = self.tcx.def_ident_span(self.def_id) {
            if self.tcx.sess.source_map().span_to_snippet(def_span).is_ok() {
                def_span.into()
            } else {
                return;
            }
        } else {
            return;
        };

        let msg = {
            let def_kind = self.tcx.def_kind(self.def_id).descr(self.def_id);
            let (quantifier, bound) = self.get_quantifier_and_bound();

            let params = if bound == 0 {
                String::new()
            } else {
                let params = self
                    .gen_params
                    .params
                    .iter()
                    .skip(self.params_offset)
                    .take(bound)
                    .map(|param| {
                        let span = self.tcx.def_span(param.def_id);
                        spans.push_span_label(span, String::new());
                        param
                    })
                    .map(|param| format!("`{}`", param.name))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(": {}", params)
            };

            format!(
                "{} defined here, with {}{} {} parameter{}{}",
                def_kind,
                quantifier,
                bound,
                self.kind(),
                pluralize!(bound),
                params,
            )
        };

        err.span_note(spans, &msg);
    }
}

impl<'tcx> StructuredDiagnostic<'tcx> for WrongNumberOfGenericArgs<'_, 'tcx> {
    fn session(&self) -> &Session {
        self.tcx.sess
    }

    fn code(&self) -> DiagnosticId {
        rustc_errors::error_code!(E0107)
    }

    fn diagnostic_common(&self) -> DiagnosticBuilder<'tcx> {
        let mut err = self.start_diagnostics();

        self.notify(&mut err);
        self.suggest(&mut err);
        self.show_definition(&mut err);

        err
    }
}
