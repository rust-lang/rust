use crate::structured_errors::StructuredDiagnostic;
use hir::def::DefKind;
use rustc_errors::{pluralize, Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_hir as hir;
use rustc_middle::ty::{self as ty, TyCtxt};
use rustc_session::Session;
use rustc_span::Span;
use rustc_span::{def_id::DefId, MultiSpan};

/// Handles the `wrong number of type / lifetime / ... arguments` family of error messages.
pub struct WrongNumberOfGenericArgs<'a, 'tcx> {
    crate tcx: TyCtxt<'tcx>,

    /// "type", "lifetime" etc., put verbatim into the message
    crate kind: &'static str,

    /// Minimum number of expected generic arguments (e.g. `2` for `HashMap`)
    crate expected_min: usize,

    /// Maximum number of expected generic arguments (e.g. `3` for `HashMap`)
    crate expected_max: usize,

    /// Number of generic arguments provided by the user
    crate provided: usize,

    /// Offset into `gen_params` - depends on the `kind`; might be different than `args_offset` when
    /// user passed e.g. more arguments than was actually expected
    crate params_offset: usize,

    /// Offset into `gen_args` - depends on the `kind`
    crate args_offset: usize,

    /// Offending path segment
    crate path_segment: &'a hir::PathSegment<'a>,

    /// Generic parameters as expected by type or trait
    crate gen_params: &'a ty::Generics,

    /// Generic arguments as provided by user
    crate gen_args: &'a hir::GenericArgs<'a>,

    /// DefId of the generic type
    crate def_id: DefId,

    /// Offending place where the generic type has been misused
    crate span: Span,
}

impl<'tcx> WrongNumberOfGenericArgs<'_, 'tcx> {
    fn quantifier_and_bound(&self) -> (&'static str, usize) {
        if self.expected_min == self.expected_max {
            ("", self.expected_min)
        } else if self.provided < self.expected_min {
            ("at least ", self.expected_min)
        } else {
            ("at most ", self.expected_max)
        }
    }

    fn start_diagnostics(&self) -> DiagnosticBuilder<'tcx> {
        let span = self.path_segment.ident.span;

        let msg = {
            let def_path = self.tcx.def_path_str(self.def_id);
            let def_kind = self.tcx.def_kind(self.def_id).descr(self.def_id);
            let (quantifier, bound) = self.quantifier_and_bound();

            if self.gen_args.span().is_some() {
                format!(
                    "this {} takes {}{} {} argument{} but {}{} {} argument{} {} supplied",
                    def_kind,
                    quantifier,
                    bound,
                    self.kind,
                    pluralize!(bound),
                    if self.provided > 0 && self.provided < self.expected_min {
                        "only "
                    } else {
                        ""
                    },
                    self.provided,
                    self.kind,
                    pluralize!(self.provided),
                    if self.provided == 1 { "was" } else { "were" },
                )
            } else {
                format!("missing generics for {} `{}`", def_kind, def_path)
            }
        };

        self.tcx.sess.struct_span_err_with_code(span, &msg, self.code())
    }

    /// Builds the `expected 1 type argument / supplied 2 type arguments` message.
    fn notify(&self, err: &mut DiagnosticBuilder<'_>) {
        let (quantifier, bound) = self.quantifier_and_bound();

        err.span_label(
            self.path_segment.ident.span,
            format!(
                "expected {}{} {} argument{}",
                quantifier,
                bound,
                self.kind,
                pluralize!(bound),
            ),
        );

        // When user's provided too many arguments, we don't highlight each of them, because it
        // would overlap with the suggestion to remove them:
        //
        // ```
        // type Foo = Bar<usize, usize>;
        //                -----  ----- supplied 2 type arguments
        //                     ^^^^^^^ remove this type argument
        // ```
        if self.provided > self.expected_max {
            return;
        }

        let args = self.gen_args.args.iter().skip(self.args_offset).take(self.provided).enumerate();

        for (i, arg) in args {
            err.span_label(
                arg.span(),
                if i + 1 == self.provided {
                    format!(
                        "supplied {} {} argument{}",
                        self.provided,
                        self.kind,
                        pluralize!(self.provided)
                    )
                } else {
                    String::new()
                },
            );
        }
    }

    fn suggest(&self, err: &mut DiagnosticBuilder<'_>) {
        if self.provided == 0 {
            if self.gen_args.span().is_some() {
                self.suggest_adding_args(err);
            } else {
                self.suggest_creating_generics(err);
            }
        } else if self.provided < self.expected_min {
            self.suggest_adding_args(err);
        } else {
            self.suggest_removing_args_or_generics(err);
        }
    }

    /// Suggests to create generics (`<...>`) when current invocation site contains no generics at
    /// all:
    ///
    /// ```text
    /// type Map = HashMap;
    /// ```
    fn suggest_creating_generics(&self, err: &mut DiagnosticBuilder<'_>) {
        let params = self
            .gen_params
            .params
            .iter()
            .skip(self.params_offset)
            .take(self.expected_min)
            .map(|param| param.name.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let def_kind = self.tcx.def_kind(self.def_id);

        let sugg = if matches!(def_kind, DefKind::Fn | DefKind::AssocFn) {
            format!("::<{}>", params)
        } else {
            format!("<{}>", params)
        };

        let msg = format!(
            "use angle brackets to add missing {} argument{}",
            self.kind,
            pluralize!(self.expected_min),
        );

        err.span_suggestion_verbose(
            self.path_segment.ident.span.shrink_to_hi(),
            &msg,
            sugg,
            Applicability::HasPlaceholders,
        );
    }

    /// Suggests to add missing argument(s) when current invocation site already contains some
    /// generics:
    ///
    /// ```text
    /// type Map = HashMap<String>;
    /// ```
    fn suggest_adding_args(&self, err: &mut DiagnosticBuilder<'_>) {
        assert!(!self.gen_args.is_empty());

        if self.gen_args.parenthesized {
            return;
        }

        let missing_arg_count = self.expected_min - self.provided;

        let (span, sugg_prefix) = if self.args_offset + self.provided == 0 {
            let span = self.gen_args.args[0].span().shrink_to_lo();
            (span, "")
        } else {
            let span =
                self.gen_args.args[self.args_offset + self.provided - 1].span().shrink_to_hi();
            (span, ", ")
        };

        let msg = format!("add missing {} argument{}", self.kind, pluralize!(missing_arg_count));

        let sugg = self
            .gen_params
            .params
            .iter()
            .skip(self.params_offset + self.provided)
            .take(missing_arg_count)
            .map(|param| param.name.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let sugg = format!("{}{}", sugg_prefix, sugg);

        err.span_suggestion_verbose(span, &msg, sugg, Applicability::HasPlaceholders);
    }

    /// Suggests to remove redundant argument(s):
    ///
    /// ```text
    /// type Map = HashMap<String, String, String, String>;
    /// ```
    fn suggest_removing_args_or_generics(&self, err: &mut DiagnosticBuilder<'_>) {
        assert!(self.provided > 0);

        let redundant_args_count = self.provided - self.expected_max;
        let remove_entire_generics = redundant_args_count >= self.gen_args.args.len();

        let (span, msg) = if remove_entire_generics {
            let sm = self.tcx.sess.source_map();

            let span = self
                .path_segment
                .args
                .unwrap()
                .span_ext(sm)
                .unwrap()
                .with_lo(self.path_segment.ident.span.hi());

            let msg = format!(
                "remove these {}generics",
                if self.gen_args.parenthesized { "parenthetical " } else { "" },
            );

            (span, msg)
        } else {
            // When it comes to removing particular argument(s) from the generics, there are two
            // edge cases we have to consider:
            //
            // When the first redundant argument is at the beginning or in the middle of the
            // generics, like so:
            //
            // ```
            // type Map = HashMap<String, String, String, String>;
            //                    ^^^^^^^^^^^^^^^^
            //                    | span must start with the argument
            // ```
            //
            // When the last redundant argument is at the ending of the generics, like so:
            //
            // ```
            // type Map = HashMap<String, String, String, String>;
            //                                  ^^^^^^^^^^^^^^^^
            //                                  | span must start with the comma
            // ```

            // Index of the first redundant argument
            let from_idx = self.args_offset + self.expected_max;

            // Index of the last redundant argument
            let to_idx = self.args_offset + self.provided - 1;

            assert!(from_idx <= to_idx);

            let (from, comma_eaten) = {
                let first_argument_starts_generics = from_idx == 0;
                let last_argument_ends_generics = to_idx + 1 == self.gen_args.args.len();

                if !first_argument_starts_generics && last_argument_ends_generics {
                    (self.gen_args.args[from_idx - 1].span().hi(), true)
                } else {
                    (self.gen_args.args[from_idx].span().lo(), false)
                }
            };

            let to = {
                let hi = self.gen_args.args[to_idx].span().hi();

                if comma_eaten {
                    hi
                } else {
                    self.gen_args.args.get(to_idx + 1).map(|arg| arg.span().lo()).unwrap_or(hi)
                }
            };

            let span = Span::new(from, to, self.span.ctxt());

            let msg = format!(
                "remove {} {} argument{}",
                if redundant_args_count == 1 { "this" } else { "these" },
                self.kind,
                pluralize!(redundant_args_count),
            );

            (span, msg)
        };

        err.span_suggestion(span, &msg, String::new(), Applicability::MaybeIncorrect);
    }

    /// Builds the `type defined here` message.
    fn show_definition(&self, err: &mut DiagnosticBuilder<'_>) {
        let mut spans: MultiSpan = if let Some(def_span) = self.tcx.def_ident_span(self.def_id) {
            def_span.into()
        } else {
            return;
        };

        let msg = {
            let def_kind = self.tcx.def_kind(self.def_id).descr(self.def_id);
            let (quantifier, bound) = self.quantifier_and_bound();

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
                self.kind,
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
