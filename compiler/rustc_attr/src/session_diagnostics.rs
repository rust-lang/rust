use std::num::IntErrorKind;

use rustc_ast as ast;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::{UnsupportedLiteralReason, fluent_generated as fluent};

#[derive(Diagnostic)]
#[diag(attr_expected_one_cfg_pattern, code = E0536)]
pub(crate) struct ExpectedOneCfgPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_invalid_predicate, code = E0537)]
pub(crate) struct InvalidPredicate {
    #[primary_span]
    pub span: Span,

    pub predicate: String,
}

#[derive(Diagnostic)]
#[diag(attr_multiple_item, code = E0538)]
pub(crate) struct MultipleItem {
    #[primary_span]
    pub span: Span,

    pub item: String,
}

#[derive(Diagnostic)]
#[diag(attr_incorrect_meta_item, code = E0539)]
pub(crate) struct IncorrectMetaItem {
    #[primary_span]
    pub span: Span,
}

/// Error code: E0541
pub(crate) struct UnknownMetaItem<'a> {
    pub span: Span,
    pub item: String,
    pub expected: &'a [&'a str],
}

// Manual implementation to be able to format `expected` items correctly.
impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for UnknownMetaItem<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let expected = self.expected.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>();
        Diag::new(dcx, level, fluent::attr_unknown_meta_item)
            .with_span(self.span)
            .with_code(E0541)
            .with_arg("item", self.item)
            .with_arg("expected", expected.join(", "))
            .with_span_label(self.span, fluent::attr_label)
    }
}

#[derive(Diagnostic)]
#[diag(attr_missing_since, code = E0542)]
pub(crate) struct MissingSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_missing_note, code = E0543)]
pub(crate) struct MissingNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_multiple_stability_levels, code = E0544)]
pub(crate) struct MultipleStabilityLevels {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_invalid_issue_string, code = E0545)]
pub(crate) struct InvalidIssueString {
    #[primary_span]
    pub span: Span,

    #[subdiagnostic]
    pub cause: Option<InvalidIssueStringCause>,
}

// The error kinds of `IntErrorKind` are duplicated here in order to allow the messages to be
// translatable.
#[derive(Subdiagnostic)]
pub(crate) enum InvalidIssueStringCause {
    #[label(attr_must_not_be_zero)]
    MustNotBeZero {
        #[primary_span]
        span: Span,
    },

    #[label(attr_empty)]
    Empty {
        #[primary_span]
        span: Span,
    },

    #[label(attr_invalid_digit)]
    InvalidDigit {
        #[primary_span]
        span: Span,
    },

    #[label(attr_pos_overflow)]
    PosOverflow {
        #[primary_span]
        span: Span,
    },

    #[label(attr_neg_overflow)]
    NegOverflow {
        #[primary_span]
        span: Span,
    },
}

impl InvalidIssueStringCause {
    pub(crate) fn from_int_error_kind(span: Span, kind: &IntErrorKind) -> Option<Self> {
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

#[derive(Diagnostic)]
#[diag(attr_missing_feature, code = E0546)]
pub(crate) struct MissingFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_non_ident_feature, code = E0546)]
pub(crate) struct NonIdentFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_missing_issue, code = E0547)]
pub(crate) struct MissingIssue {
    #[primary_span]
    pub span: Span,
}

// FIXME: Why is this the same error code as `InvalidReprHintNoParen` and `InvalidReprHintNoValue`?
// It is more similar to `IncorrectReprFormatGeneric`.
#[derive(Diagnostic)]
#[diag(attr_incorrect_repr_format_packed_one_or_zero_arg, code = E0552)]
pub(crate) struct IncorrectReprFormatPackedOneOrZeroArg {
    #[primary_span]
    pub span: Span,
}
#[derive(Diagnostic)]
#[diag(attr_incorrect_repr_format_packed_expect_integer, code = E0552)]
pub(crate) struct IncorrectReprFormatPackedExpectInteger {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_invalid_repr_hint_no_paren, code = E0552)]
pub(crate) struct InvalidReprHintNoParen {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag(attr_invalid_repr_hint_no_value, code = E0552)]
pub(crate) struct InvalidReprHintNoValue {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

/// Error code: E0565
pub(crate) struct UnsupportedLiteral {
    pub span: Span,
    pub reason: UnsupportedLiteralReason,
    pub is_bytestr: bool,
    pub start_point_span: Span,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for UnsupportedLiteral {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(dcx, level, match self.reason {
            UnsupportedLiteralReason::Generic => fluent::attr_unsupported_literal_generic,
            UnsupportedLiteralReason::CfgString => fluent::attr_unsupported_literal_cfg_string,
            UnsupportedLiteralReason::CfgBoolean => fluent::attr_unsupported_literal_cfg_boolean,
            UnsupportedLiteralReason::DeprecatedString => {
                fluent::attr_unsupported_literal_deprecated_string
            }
            UnsupportedLiteralReason::DeprecatedKvPair => {
                fluent::attr_unsupported_literal_deprecated_kv_pair
            }
        });
        diag.span(self.span);
        diag.code(E0565);
        if self.is_bytestr {
            diag.span_suggestion(
                self.start_point_span,
                fluent::attr_unsupported_literal_suggestion,
                "",
                Applicability::MaybeIncorrect,
            );
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(attr_invalid_repr_align_need_arg, code = E0589)]
pub(crate) struct InvalidReprAlignNeedArg {
    #[primary_span]
    #[suggestion(code = "align(...)", applicability = "has-placeholders")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_invalid_repr_generic, code = E0589)]
pub(crate) struct InvalidReprGeneric<'a> {
    #[primary_span]
    pub span: Span,

    pub repr_arg: String,
    pub error_part: &'a str,
}

#[derive(Diagnostic)]
#[diag(attr_incorrect_repr_format_align_one_arg, code = E0693)]
pub(crate) struct IncorrectReprFormatAlignOneArg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_incorrect_repr_format_expect_literal_integer, code = E0693)]
pub(crate) struct IncorrectReprFormatExpectInteger {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_incorrect_repr_format_generic, code = E0693)]
pub(crate) struct IncorrectReprFormatGeneric<'a> {
    #[primary_span]
    pub span: Span,

    pub repr_arg: &'a str,

    #[subdiagnostic]
    pub cause: Option<IncorrectReprFormatGenericCause<'a>>,
}

#[derive(Subdiagnostic)]
pub(crate) enum IncorrectReprFormatGenericCause<'a> {
    #[suggestion(attr_suggestion, code = "{name}({int})", applicability = "machine-applicable")]
    Int {
        #[primary_span]
        span: Span,

        #[skip_arg]
        name: &'a str,

        #[skip_arg]
        int: u128,
    },

    #[suggestion(attr_suggestion, code = "{name}({symbol})", applicability = "machine-applicable")]
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
    pub(crate) fn from_lit_kind(span: Span, kind: &ast::LitKind, name: &'a str) -> Option<Self> {
        match kind {
            ast::LitKind::Int(int, ast::LitIntType::Unsuffixed) => {
                Some(Self::Int { span, name, int: int.get() })
            }
            ast::LitKind::Str(symbol, _) => Some(Self::Symbol { span, name, symbol: *symbol }),
            _ => None,
        }
    }
}

#[derive(Diagnostic)]
#[diag(attr_rustc_promotable_pairing, code = E0717)]
pub(crate) struct RustcPromotablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_rustc_const_stable_indirect_pairing)]
pub(crate) struct RustcConstStableIndirectPairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_rustc_allowed_unstable_pairing, code = E0789)]
pub(crate) struct RustcAllowedUnstablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_cfg_predicate_identifier)]
pub(crate) struct CfgPredicateIdentifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_deprecated_item_suggestion)]
pub(crate) struct DeprecatedItemSuggestion {
    #[primary_span]
    pub span: Span,

    #[help]
    pub is_nightly: bool,

    #[note]
    pub details: (),
}

#[derive(Diagnostic)]
#[diag(attr_expected_single_version_literal)]
pub(crate) struct ExpectedSingleVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_expected_version_literal)]
pub(crate) struct ExpectedVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_expects_feature_list)]
pub(crate) struct ExpectsFeatureList {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag(attr_expects_features)]
pub(crate) struct ExpectsFeatures {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag(attr_invalid_since)]
pub(crate) struct InvalidSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_soft_no_args)]
pub(crate) struct SoftNoArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_unknown_version_literal)]
pub(crate) struct UnknownVersionLiteral {
    #[primary_span]
    pub span: Span,
}
