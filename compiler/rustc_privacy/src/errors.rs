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

#[derive(SessionDiagnostic)]
#[error(slug = "privacy-item-is-private")]
pub struct ItemIsPrivate<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: &'a str,
    pub descr: String,
}

#[derive(SessionDiagnostic)]
#[error(slug = "privacy-unnamed-item-is-private")]
pub struct UnnamedItemIsPrivate {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

// Duplicate of `InPublicInterface` but with a different error code, shares the same slug.
#[derive(SessionDiagnostic)]
#[error(code = "E0445", slug = "privacy-in-public-interface")]
pub struct InPublicInterfaceTraits<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub vis_descr: &'static str,
    pub kind: &'a str,
    pub descr: String,
    #[label = "visibility-label"]
    pub vis_span: Span,
}

// Duplicate of `InPublicInterfaceTraits` but with a different error code, shares the same slug.
#[derive(SessionDiagnostic)]
#[error(code = "E0446", slug = "privacy-in-public-interface")]
pub struct InPublicInterface<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub vis_descr: &'static str,
    pub kind: &'a str,
    pub descr: String,
    #[label = "visibility-label"]
    pub vis_span: Span,
}
