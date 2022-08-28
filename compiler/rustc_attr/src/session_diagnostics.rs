use std::num::IntErrorKind;

use rustc_ast as ast;
use rustc_errors::{error_code, fluent, Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_macros::SessionDiagnostic;
use rustc_session::{parse::ParseSess, SessionDiagnostic};
use rustc_span::{Span, Symbol};

use crate::UnsupportedLiteralReason;

#[derive(SessionDiagnostic)]
#[diag(attr::expected_one_cfg_pattern, code = "E0536")]
pub(crate) struct ExpectedOneCfgPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::invalid_predicate, code = "E0537")]
pub(crate) struct InvalidPredicate {
    #[primary_span]
    pub span: Span,

    pub predicate: String,
}

#[derive(SessionDiagnostic)]
#[diag(attr::multiple_item, code = "E0538")]
pub(crate) struct MultipleItem {
    #[primary_span]
    pub span: Span,

    pub item: String,
}

#[derive(SessionDiagnostic)]
#[diag(attr::incorrect_meta_item, code = "E0539")]
pub(crate) struct IncorrectMetaItem {
    #[primary_span]
    pub span: Span,
}

// Error code: E0541
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

#[derive(SessionDiagnostic)]
#[diag(attr::missing_since, code = "E0542")]
pub(crate) struct MissingSince {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::missing_note, code = "E0543")]
pub(crate) struct MissingNote {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::multiple_stability_levels, code = "E0544")]
pub(crate) struct MultipleStabilityLevels {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::invalid_issue_string, code = "E0545")]
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

#[derive(SessionDiagnostic)]
#[diag(attr::missing_feature, code = "E0546")]
pub(crate) struct MissingFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::non_ident_feature, code = "E0546")]
pub(crate) struct NonIdentFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::missing_issue, code = "E0547")]
pub(crate) struct MissingIssue {
    #[primary_span]
    pub span: Span,
}

// FIXME: This diagnostic is identical to `IncorrectMetaItem`, barring the error code. Consider
// changing this to `IncorrectMetaItem`. See #51489.
#[derive(SessionDiagnostic)]
#[diag(attr::incorrect_meta_item, code = "E0551")]
pub(crate) struct IncorrectMetaItem2 {
    #[primary_span]
    pub span: Span,
}

// FIXME: Why is this the same error code as `InvalidReprHintNoParen` and `InvalidReprHintNoValue`?
// It is more similar to `IncorrectReprFormatGeneric`.
#[derive(SessionDiagnostic)]
#[diag(attr::incorrect_repr_format_packed_one_or_zero_arg, code = "E0552")]
pub(crate) struct IncorrectReprFormatPackedOneOrZeroArg {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::invalid_repr_hint_no_paren, code = "E0552")]
pub(crate) struct InvalidReprHintNoParen {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(SessionDiagnostic)]
#[diag(attr::invalid_repr_hint_no_value, code = "E0552")]
pub(crate) struct InvalidReprHintNoValue {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

// Error code: E0565
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
            diag.span_suggestion(
                sess.source_map().start_point(self.span),
                fluent::attr::unsupported_literal_suggestion,
                "",
                Applicability::MaybeIncorrect,
            );
        }
        diag
    }
}

#[derive(SessionDiagnostic)]
#[diag(attr::invalid_repr_align_need_arg, code = "E0589")]
pub(crate) struct InvalidReprAlignNeedArg {
    #[primary_span]
    #[suggestion(code = "align(...)", applicability = "has-placeholders")]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::invalid_repr_generic, code = "E0589")]
pub(crate) struct InvalidReprGeneric<'a> {
    #[primary_span]
    pub span: Span,

    pub repr_arg: String,
    pub error_part: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(attr::incorrect_repr_format_align_one_arg, code = "E0693")]
pub(crate) struct IncorrectReprFormatAlignOneArg {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::incorrect_repr_format_generic, code = "E0693")]
pub(crate) struct IncorrectReprFormatGeneric<'a> {
    #[primary_span]
    pub span: Span,

    pub repr_arg: &'a str,

    #[subdiagnostic]
    pub cause: Option<IncorrectReprFormatGenericCause<'a>>,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum IncorrectReprFormatGenericCause<'a> {
    #[suggestion(attr::suggestion, code = "{name}({int})", applicability = "machine-applicable")]
    Int {
        #[primary_span]
        span: Span,

        #[skip_arg]
        name: &'a str,

        #[skip_arg]
        int: u128,
    },

    #[suggestion(
        attr::suggestion,
        code = "{name}({symbol})",
        applicability = "machine-applicable"
    )]
    Symbol {
        #[primary_span]
        span: Span,

        #[skip_arg]
        name: &'a str,

        #[skip_arg]
        symbol: Symbol,
    },
}

impl<'a> IncorrectReprFormatGenericCause<'a> {
    pub fn from_lit_kind(span: Span, kind: &ast::LitKind, name: &'a str) -> Option<Self> {
        match kind {
            ast::LitKind::Int(int, ast::LitIntType::Unsuffixed) => {
                Some(Self::Int { span, name, int: *int })
            }
            ast::LitKind::Str(symbol, _) => Some(Self::Symbol { span, name, symbol: *symbol }),
            _ => None,
        }
    }
}

#[derive(SessionDiagnostic)]
#[diag(attr::rustc_promotable_pairing, code = "E0717")]
pub(crate) struct RustcPromotablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::rustc_allowed_unstable_pairing, code = "E0789")]
pub(crate) struct RustcAllowedUnstablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::cfg_predicate_identifier)]
pub(crate) struct CfgPredicateIdentifier {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::deprecated_item_suggestion)]
pub(crate) struct DeprecatedItemSuggestion {
    #[primary_span]
    pub span: Span,

    #[help]
    pub is_nightly: Option<()>,

    #[note]
    pub details: (),
}

#[derive(SessionDiagnostic)]
#[diag(attr::expected_single_version_literal)]
pub(crate) struct ExpectedSingleVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::expected_version_literal)]
pub(crate) struct ExpectedVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::expects_feature_list)]
pub(crate) struct ExpectsFeatureList {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(SessionDiagnostic)]
#[diag(attr::expects_features)]
pub(crate) struct ExpectsFeatures {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(SessionDiagnostic)]
#[diag(attr::soft_no_args)]
pub(crate) struct SoftNoArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(attr::unknown_version_literal)]
pub(crate) struct UnknownVersionLiteral {
    #[primary_span]
    pub span: Span,
}
