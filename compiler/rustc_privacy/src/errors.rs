use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[error(code = "E0451", slug = "privacy-field-is-private")]
pub struct FieldIsPrivate {
    #[primary_span]
    pub span: Span,
    pub field_name: Symbol,
    pub variant_descr: &'static str,
    pub def_path_str: String,
    #[subdiagnostic]
    pub label: FieldIsPrivateLabel,
}

#[derive(SessionSubdiagnostic)]
pub enum FieldIsPrivateLabel {
    #[label(slug = "privacy-field-is-private-is-update-syntax-label")]
    IsUpdateSyntax {
        #[primary_span]
        span: Span,
        field_name: Symbol,
    },
    #[label(slug = "privacy-field-is-private-label")]
    Other {
        #[primary_span]
        span: Span,
    },
}
