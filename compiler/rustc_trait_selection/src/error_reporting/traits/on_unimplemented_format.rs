pub mod errors {
    use rustc_macros::LintDiagnostic;
    use rustc_middle::ty::TyCtxt;
    use rustc_session::lint::builtin::UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES;
    use rustc_span::Ident;

    use super::*;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_unknown_format_parameter_for_on_unimplemented_attr)]
    #[help]
    pub struct UnknownFormatParameterForOnUnimplementedAttr {
        pub argument_name: Symbol,
        pub trait_name: Ident,
    }

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_disallowed_positional_argument)]
    #[help]
    pub struct DisallowedPositionalArgument;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_invalid_format_specifier)]
    #[help]
    pub struct InvalidFormatSpecifier;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_wrapped_parser_error)]
    pub struct WrappedParserError {
        pub description: String,
        pub label: String,
    }
    #[derive(LintDiagnostic)]
    #[diag(trait_selection_malformed_on_unimplemented_attr)]
    #[help]
    pub struct MalformedOnUnimplementedAttrLint {
        #[label]
        pub span: Span,
    }

    impl MalformedOnUnimplementedAttrLint {
        pub fn new(span: Span) -> Self {
            Self { span }
        }
    }

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_missing_options_for_on_unimplemented_attr)]
    #[help]
    pub struct MissingOptionsForOnUnimplementedAttr;

    #[derive(LintDiagnostic)]
    #[diag(trait_selection_ignored_diagnostic_option)]
    pub struct IgnoredDiagnosticOption {
        pub option_name: &'static str,
        #[label]
        pub span: Span,
        #[label(trait_selection_other_label)]
        pub prev_span: Span,
    }

    impl IgnoredDiagnosticOption {
        pub fn maybe_emit_warning<'tcx>(
            tcx: TyCtxt<'tcx>,
            item_def_id: DefId,
            new: Option<Span>,
            old: Option<Span>,
            option_name: &'static str,
        ) {
            if let (Some(new_item), Some(old_item)) = (new, old) {
                if let Some(item_def_id) = item_def_id.as_local() {
                    tcx.emit_node_span_lint(
                        UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        tcx.local_def_id_to_hir_id(item_def_id),
                        new_item,
                        IgnoredDiagnosticOption {
                            span: new_item,
                            prev_span: old_item,
                            option_name,
                        },
                    );
                }
            }
        }
    }
}
