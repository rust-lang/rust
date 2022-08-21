use rustc_errors::{fluent, AddSubdiagnostic};
use rustc_hir::FnRetTy;
use rustc_macros::SessionDiagnostic;
use rustc_span::{BytePos, Span};

#[derive(SessionDiagnostic)]
#[diag(infer::opaque_hidden_type)]
pub struct OpaqueHiddenTypeDiag {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(infer::opaque_type)]
    pub opaque_type: Span,
    #[note(infer::hidden_type)]
    pub hidden_type: Span,
}

#[derive(SessionDiagnostic)]
#[diag(infer::type_annotations_needed, code = "E0282")]
pub struct AnnotationRequired<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
}

// Copy of `AnnotationRequired` for E0283
#[derive(SessionDiagnostic)]
#[diag(infer::type_annotations_needed, code = "E0283")]
pub struct AmbigousImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
}

// Copy of `AnnotationRequired` for E0284
#[derive(SessionDiagnostic)]
#[diag(infer::type_annotations_needed, code = "E0284")]
pub struct AmbigousReturn<'a> {
    #[primary_span]
    pub span: Span,
    pub source_kind: &'static str,
    pub source_name: &'a str,
    #[label]
    pub failure_span: Option<Span>,
    #[subdiagnostic]
    pub bad_label: Option<InferenceBadError<'a>>,
    #[subdiagnostic]
    pub infer_subdiags: Vec<SourceKindSubdiag<'a>>,
    #[subdiagnostic]
    pub multi_suggestions: Vec<SourceKindMultiSuggestion<'a>>,
}

#[derive(SessionDiagnostic)]
#[diag(infer::need_type_info_in_generator, code = "E0698")]
pub struct NeedTypeInfoInGenerator<'a> {
    #[primary_span]
    pub span: Span,
    pub generator_kind: String,
    #[subdiagnostic]
    pub bad_label: InferenceBadError<'a>,
}

// Used when a better one isn't available
#[derive(SessionSubdiagnostic)]
#[label(infer::label_bad)]
pub struct InferenceBadError<'a> {
    #[primary_span]
    pub span: Span,
    pub bad_kind: &'static str,
    pub prefix_kind: &'static str,
    pub has_parent: bool,
    pub prefix: &'a str,
    pub parent_prefix: &'a str,
    pub parent_name: String,
    pub name: String,
}

#[derive(SessionSubdiagnostic)]
pub enum SourceKindSubdiag<'a> {
    #[suggestion_verbose(
        infer::source_kind_subdiag_let,
        code = ": {type_name}",
        applicability = "has-placeholders"
    )]
    LetLike {
        #[primary_span]
        span: Span,
        name: String,
        type_name: String,
        kind: &'static str,
        x_kind: &'static str,
        prefix_kind: &'static str,
        prefix: &'a str,
        arg_name: String,
    },
    #[label(infer::source_kind_subdiag_generic_label)]
    GenericLabel {
        #[primary_span]
        span: Span,
        is_type: bool,
        param_name: String,
        parent_exists: bool,
        parent_prefix: String,
        parent_name: String,
    },
    #[suggestion_verbose(
        infer::source_kind_subdiag_generic_suggestion,
        code = "::<{args}>",
        applicability = "has-placeholders"
    )]
    GenericSuggestion {
        #[primary_span]
        span: Span,
        arg_count: usize,
        args: String,
    },
}

// Has to be implemented manually because multipart suggestions are not supported by the derive macro.
// Would be a part of `SourceKindSubdiag` otherwise.
pub enum SourceKindMultiSuggestion<'a> {
    FullyQualified {
        span: Span,
        def_path: String,
        adjustment: &'a str,
        successor: (&'a str, BytePos),
    },
    ClosureReturn {
        ty_info: String,
        data: &'a FnRetTy<'a>,
        should_wrap_expr: Option<Span>,
    },
}

impl AddSubdiagnostic for SourceKindMultiSuggestion<'_> {
    fn add_to_diagnostic(self, diag: &mut rustc_errors::Diagnostic) {
        match self {
            Self::FullyQualified { span, def_path, adjustment, successor } => {
                let suggestion = vec![
                    (span.shrink_to_lo(), format!("{def_path}({adjustment}")),
                    (span.shrink_to_hi().with_hi(successor.1), successor.0.to_string()),
                ];
                diag.multipart_suggestion_verbose(
                    fluent::infer::source_kind_fully_qualified,
                    suggestion,
                    rustc_errors::Applicability::HasPlaceholders,
                );
            }
            Self::ClosureReturn { ty_info, data, should_wrap_expr } => {
                let (arrow, post) = match data {
                    FnRetTy::DefaultReturn(_) => ("-> ", " "),
                    _ => ("", ""),
                };
                let suggestion = match should_wrap_expr {
                    Some(end_span) => vec![
                        (data.span(), format!("{}{}{}{{ ", arrow, ty_info, post)),
                        (end_span, " }".to_string()),
                    ],
                    None => vec![(data.span(), format!("{}{}{}", arrow, ty_info, post))],
                };
                diag.multipart_suggestion_verbose(
                    fluent::infer::source_kind_closure_return,
                    suggestion,
                    rustc_errors::Applicability::HasPlaceholders,
                );
            }
        }
    }
}
