use std::num::IntErrorKind;

use rustc_errors::{error_code, fluent, Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_macros::SessionDiagnostic;
use rustc_session::{parse::ParseSess, SessionDiagnostic};
use rustc_span::Span;

use crate::UnsupportedLiteralReason;

#[derive(SessionDiagnostic)]
#[error(attr::multiple_item, code = "E0538")]
pub(crate) struct MultipleItem {
    #[primary_span]
    pub span: Span,

    pub item: String,
}

#[derive(SessionDiagnostic)]
#[error(attr::missing_since, code = "E0542")]
pub(crate) struct MissingSince {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::non_ident_feature, code = "E0546")]
pub(crate) struct NonIdentFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::missing_feature, code = "E0546")]
pub(crate) struct MissingFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::multiple_stability_levels, code = "E0544")]
pub(crate) struct MultipleStabilityLevels {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::invalid_meta_item, code = "E0539")]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::missing_issue, code = "E0547")]
pub(crate) struct MissingIssue {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::rustc_promotable_pairing, code = "E0717")]
pub(crate) struct RustcPromotablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::rustc_allowed_unstable_pairing, code = "E0789")]
pub(crate) struct RustcAllowedUnstablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::soft_no_args)]
pub(crate) struct SoftNoArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(attr::invalid_issue_string, code = "E0545")]
pub(crate) struct InvalidIssueString {
    #[primary_span]
    pub span: Span,

    #[subdiagnostic]
    pub cause: Option<InvalidIssueStringCause>,
}

// The error kinds of `IntErrorKind` are duplicated here in order to allow the messages to be
// translatable.
#[derive(SessionSubdiagnostic)]
pub(crate) enum InvalidIssueStringCause {
    #[label(attr::must_not_be_zero)]
    MustNotBeZero {
        #[primary_span]
        span: Span,
    },

    #[label(attr::empty)]
    Empty {
        #[primary_span]
        span: Span,
    },

    #[label(attr::invalid_digit)]
    InvalidDigit {
        #[primary_span]
        span: Span,
    },

    #[label(attr::pos_overflow)]
    PosOverflow {
        #[primary_span]
        span: Span,
    },

    #[label(attr::neg_overflow)]
    NegOverflow {
        #[primary_span]
        span: Span,
    },
}

impl InvalidIssueStringCause {
    pub fn from_int_error_kind(span: Span, kind: &IntErrorKind) -> Option<Self> {
        match kind {
            IntErrorKind::Empty => Some(Self::Empty { span }),
            IntErrorKind::InvalidDigit => Some(Self::InvalidDigit { span }),
            IntErrorKind::PosOverflow => Some(Self::PosOverflow { span }),
            IntErrorKind::NegOverflow => Some(Self::NegOverflow { span }),
            IntErrorKind::Zero => Some(Self::MustNotBeZero { span }),
            _ => None,
        }
    }
}

pub(crate) struct UnknownMetaItem<'a> {
    pub span: Span,
    pub item: String,
    pub expected: &'a [&'a str],
}

// Manual implementation to be able to format `expected` items correctly.
impl<'a> SessionDiagnostic<'a> for UnknownMetaItem<'_> {
    fn into_diagnostic(self, sess: &'a ParseSess) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let expected = self.expected.iter().map(|name| format!("`{}`", name)).collect::<Vec<_>>();
        let mut diag = sess.span_diagnostic.struct_span_err_with_code(
            self.span,
            fluent::attr::unknown_meta_item,
            error_code!(E0541),
        );
        diag.set_arg("item", self.item);
        diag.set_arg("expected", expected.join(", "));
        diag.span_label(self.span, fluent::attr::label);
        diag
    }
}

pub(crate) struct UnsupportedLiteral {
    pub span: Span,
    pub reason: UnsupportedLiteralReason,
    pub is_bytestr: bool,
}

impl<'a> SessionDiagnostic<'a> for UnsupportedLiteral {
    fn into_diagnostic(self, sess: &'a ParseSess) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut diag = sess.span_diagnostic.struct_span_err_with_code(
            self.span,
            match self.reason {
                UnsupportedLiteralReason::Generic => fluent::attr::unsupported_literal_generic,
                UnsupportedLiteralReason::CfgString => fluent::attr::unsupported_literal_cfg_string,
                UnsupportedLiteralReason::DeprecatedString => {
                    fluent::attr::unsupported_literal_deprecated_string
                }
                UnsupportedLiteralReason::DeprecatedKvPair => {
                    fluent::attr::unsupported_literal_deprecated_kv_pair
                }
            },
            error_code!(E0565),
        );
        if self.is_bytestr {
            if let Ok(lint_str) = sess.source_map().span_to_snippet(self.span) {
                diag.span_suggestion(
                    self.span,
                    fluent::attr::unsupported_literal_suggestion,
                    &lint_str[1..],
                    Applicability::MaybeIncorrect,
                );
            }
        }
        diag
    }
}
