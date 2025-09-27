use std::num::IntErrorKind;

use rustc_ast::{self as ast, AttrStyle, Path};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level,
};
use rustc_feature::AttributeTemplate;
use rustc_hir::{AttrPath, Target};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::fluent_generated as fluent;

pub(crate) enum UnsupportedLiteralReason {
    Generic,
    CfgString,
    CfgBoolean,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_expected_one_cfg_pattern, code = E0536)]
pub(crate) struct ExpectedOneCfgPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_predicate, code = E0537)]
pub(crate) struct InvalidPredicate {
    #[primary_span]
    pub span: Span,

    pub predicate: String,
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
        Diag::new(dcx, level, fluent::attr_parsing_unknown_meta_item)
            .with_span(self.span)
            .with_code(E0541)
            .with_arg("item", self.item)
            .with_arg("expected", expected.join(", "))
            .with_span_label(self.span, fluent::attr_parsing_label)
    }
}

#[derive(Diagnostic)]
#[diag(attr_parsing_missing_since, code = E0542)]
pub(crate) struct MissingSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_missing_note, code = E0543)]
pub(crate) struct MissingNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_multiple_stability_levels, code = E0544)]
pub(crate) struct MultipleStabilityLevels {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_issue_string, code = E0545)]
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
    #[label(attr_parsing_must_not_be_zero)]
    MustNotBeZero {
        #[primary_span]
        span: Span,
    },

    #[label(attr_parsing_empty)]
    Empty {
        #[primary_span]
        span: Span,
    },

    #[label(attr_parsing_invalid_digit)]
    InvalidDigit {
        #[primary_span]
        span: Span,
    },

    #[label(attr_parsing_pos_overflow)]
    PosOverflow {
        #[primary_span]
        span: Span,
    },

    #[label(attr_parsing_neg_overflow)]
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
#[diag(attr_parsing_missing_feature, code = E0546)]
pub(crate) struct MissingFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_non_ident_feature, code = E0546)]
pub(crate) struct NonIdentFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_missing_issue, code = E0547)]
pub(crate) struct MissingIssue {
    #[primary_span]
    pub span: Span,
}

// FIXME: Why is this the same error code as `InvalidReprHintNoParen` and `InvalidReprHintNoValue`?
// It is more similar to `IncorrectReprFormatGeneric`.
#[derive(Diagnostic)]
#[diag(attr_parsing_incorrect_repr_format_packed_one_or_zero_arg, code = E0552)]
pub(crate) struct IncorrectReprFormatPackedOneOrZeroArg {
    #[primary_span]
    pub span: Span,
}
#[derive(Diagnostic)]
#[diag(attr_parsing_incorrect_repr_format_packed_expect_integer, code = E0552)]
pub(crate) struct IncorrectReprFormatPackedExpectInteger {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_repr_hint_no_paren, code = E0552)]
pub(crate) struct InvalidReprHintNoParen {
    #[primary_span]
    pub span: Span,

    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_repr_hint_no_value, code = E0552)]
pub(crate) struct InvalidReprHintNoValue {
    #[primary_span]
    pub span: Span,

    pub name: Symbol,
}

/// Error code: E0565
// FIXME(jdonszelmann): slowly phased out
pub(crate) struct UnsupportedLiteral {
    pub span: Span,
    pub reason: UnsupportedLiteralReason,
    pub is_bytestr: bool,
    pub start_point_span: Span,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for UnsupportedLiteral {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            match self.reason {
                UnsupportedLiteralReason::Generic => {
                    fluent::attr_parsing_unsupported_literal_generic
                }
                UnsupportedLiteralReason::CfgString => {
                    fluent::attr_parsing_unsupported_literal_cfg_string
                }
                UnsupportedLiteralReason::CfgBoolean => {
                    fluent::attr_parsing_unsupported_literal_cfg_boolean
                }
            },
        );
        diag.span(self.span);
        diag.code(E0565);
        if self.is_bytestr {
            diag.span_suggestion(
                self.start_point_span,
                fluent::attr_parsing_unsupported_literal_suggestion,
                "",
                Applicability::MaybeIncorrect,
            );
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_repr_align_need_arg, code = E0589)]
pub(crate) struct InvalidReprAlignNeedArg {
    #[primary_span]
    #[suggestion(code = "align(...)", applicability = "has-placeholders")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_repr_generic, code = E0589)]
pub(crate) struct InvalidReprGeneric<'a> {
    #[primary_span]
    pub span: Span,

    pub repr_arg: String,
    pub error_part: &'a str,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_incorrect_repr_format_align_one_arg, code = E0693)]
pub(crate) struct IncorrectReprFormatAlignOneArg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_incorrect_repr_format_expect_literal_integer, code = E0693)]
pub(crate) struct IncorrectReprFormatExpectInteger {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_incorrect_repr_format_generic, code = E0693)]
pub(crate) struct IncorrectReprFormatGeneric {
    #[primary_span]
    pub span: Span,

    pub repr_arg: Symbol,

    #[subdiagnostic]
    pub cause: Option<IncorrectReprFormatGenericCause>,
}

#[derive(Subdiagnostic)]
pub(crate) enum IncorrectReprFormatGenericCause {
    #[suggestion(
        attr_parsing_suggestion,
        code = "{name}({value})",
        applicability = "machine-applicable"
    )]
    Int {
        #[primary_span]
        span: Span,

        #[skip_arg]
        name: Symbol,

        #[skip_arg]
        value: u128,
    },

    #[suggestion(
        attr_parsing_suggestion,
        code = "{name}({value})",
        applicability = "machine-applicable"
    )]
    Symbol {
        #[primary_span]
        span: Span,

        #[skip_arg]
        name: Symbol,

        #[skip_arg]
        value: Symbol,
    },
}

impl IncorrectReprFormatGenericCause {
    pub(crate) fn from_lit_kind(span: Span, kind: &ast::LitKind, name: Symbol) -> Option<Self> {
        match *kind {
            ast::LitKind::Int(value, ast::LitIntType::Unsuffixed) => {
                Some(Self::Int { span, name, value: value.get() })
            }
            ast::LitKind::Str(value, _) => Some(Self::Symbol { span, name, value }),
            _ => None,
        }
    }
}

#[derive(Diagnostic)]
#[diag(attr_parsing_rustc_promotable_pairing, code = E0717)]
pub(crate) struct RustcPromotablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_rustc_allowed_unstable_pairing, code = E0789)]
pub(crate) struct RustcAllowedUnstablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_cfg_predicate_identifier)]
pub(crate) struct CfgPredicateIdentifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_deprecated_item_suggestion)]
pub(crate) struct DeprecatedItemSuggestion {
    #[primary_span]
    pub span: Span,

    #[help]
    pub is_nightly: bool,

    #[note]
    pub details: (),
}

#[derive(Diagnostic)]
#[diag(attr_parsing_expected_single_version_literal)]
pub(crate) struct ExpectedSingleVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_expected_version_literal)]
pub(crate) struct ExpectedVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_expects_feature_list)]
pub(crate) struct ExpectsFeatureList {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_expects_features)]
pub(crate) struct ExpectsFeatures {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_since)]
pub(crate) struct InvalidSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_soft_no_args)]
pub(crate) struct SoftNoArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_unknown_version_literal)]
pub(crate) struct UnknownVersionLiteral {
    #[primary_span]
    pub span: Span,
}

// FIXME(jdonszelmann) duplicated from `rustc_passes`, remove once `check_attr` is integrated.
#[derive(Diagnostic)]
#[diag(attr_parsing_unused_multiple)]
pub(crate) struct UnusedMultiple {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(attr_parsing_unused_duplicate)]
pub(crate) struct UnusedDuplicate {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note]
    pub other: Span,
    #[warning]
    pub warning: bool,
}

// FIXME(jdonszelmann): duplicated in rustc_lints, should be moved here completely.
#[derive(LintDiagnostic)]
#[diag(attr_parsing_ill_formed_attribute_input)]
pub(crate) struct IllFormedAttributeInput {
    pub num_suggestions: usize,
    pub suggestions: DiagArgValue,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_ill_formed_attribute_input)]
pub(crate) struct IllFormedAttributeInputLint {
    #[primary_span]
    pub span: Span,
    pub num_suggestions: usize,
    pub suggestions: DiagArgValue,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_null_on_export, code = E0648)]
pub(crate) struct NullOnExport {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_null_on_link_section, code = E0648)]
pub(crate) struct NullOnLinkSection {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_null_on_objc_class)]
pub(crate) struct NullOnObjcClass {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_null_on_objc_selector)]
pub(crate) struct NullOnObjcSelector {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_objc_class_expected_string_literal)]
pub(crate) struct ObjcClassExpectedStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_objc_selector_expected_string_literal)]
pub(crate) struct ObjcSelectorExpectedStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_stability_outside_std, code = E0734)]
pub(crate) struct StabilityOutsideStd {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_empty_confusables)]
pub(crate) struct EmptyConfusables {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(attr_parsing_empty_attribute)]
#[note]
pub(crate) struct EmptyAttributeList {
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    pub attr_path: AttrPath,
    pub valid_without_list: bool,
}

#[derive(LintDiagnostic)]
#[diag(attr_parsing_invalid_target_lint)]
#[warning]
#[help]
pub(crate) struct InvalidTargetLint {
    pub name: AttrPath,
    pub target: &'static str,
    pub applied: DiagArgValue,
    pub only: &'static str,
    #[suggestion(code = "", applicability = "machine-applicable", style = "tool-only")]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[help]
#[diag(attr_parsing_invalid_target)]
pub(crate) struct InvalidTarget {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "tool-only")]
    pub span: Span,
    pub name: AttrPath,
    pub target: &'static str,
    pub applied: DiagArgValue,
    pub only: &'static str,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_alignment_value, code = E0589)]
pub(crate) struct InvalidAlignmentValue {
    #[primary_span]
    pub span: Span,
    pub error_part: &'static str,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_repr_ident, code = E0565)]
pub(crate) struct ReprIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_unrecognized_repr_hint, code = E0552)]
#[help]
#[note]
pub(crate) struct UnrecognizedReprHint {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_unstable_feature_bound_incompatible_stability)]
#[help]
pub(crate) struct UnstableFeatureBoundIncompatibleStability {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_naked_functions_incompatible_attribute, code = E0736)]
pub(crate) struct NakedFunctionIncompatibleAttribute {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(attr_parsing_naked_attribute)]
    pub naked_span: Span,
    pub attr: String,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_link_ordinal_out_of_range)]
#[note]
pub(crate) struct LinkOrdinalOutOfRange {
    #[primary_span]
    pub span: Span,
    pub ordinal: u128,
}

pub(crate) enum AttributeParseErrorReason<'a> {
    ExpectedNoArgs,
    ExpectedStringLiteral {
        byte_string: Option<Span>,
    },
    ExpectedIntegerLiteral,
    ExpectedAtLeastOneArgument,
    ExpectedSingleArgument,
    ExpectedList,
    UnexpectedLiteral,
    ExpectedNameValue(Option<Symbol>),
    DuplicateKey(Symbol),
    ExpectedSpecificArgument {
        possibilities: &'a [Symbol],
        strings: bool,
        /// Should we tell the user to write a list when they didn't?
        list: bool,
    },
    ExpectedIdentifier,
}

pub(crate) struct AttributeParseError<'a> {
    pub(crate) span: Span,
    pub(crate) attr_span: Span,
    pub(crate) attr_style: AttrStyle,
    pub(crate) template: AttributeTemplate,
    pub(crate) attribute: AttrPath,
    pub(crate) reason: AttributeParseErrorReason<'a>,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for AttributeParseError<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let name = self.attribute.to_string();

        let mut diag = Diag::new(dcx, level, format!("malformed `{name}` attribute input"));
        diag.span(self.attr_span);
        diag.code(E0539);
        match self.reason {
            AttributeParseErrorReason::ExpectedStringLiteral { byte_string } => {
                if let Some(start_point_span) = byte_string {
                    diag.span_suggestion(
                        start_point_span,
                        fluent::attr_parsing_unsupported_literal_suggestion,
                        "",
                        Applicability::MaybeIncorrect,
                    );
                    diag.note("expected a normal string literal, not a byte string literal");

                    return diag;
                } else {
                    diag.span_label(self.span, "expected a string literal here");
                }
            }
            AttributeParseErrorReason::ExpectedIntegerLiteral => {
                diag.span_label(self.span, "expected an integer literal here");
            }
            AttributeParseErrorReason::ExpectedSingleArgument => {
                diag.span_label(self.span, "expected a single argument here");
                diag.code(E0805);
            }
            AttributeParseErrorReason::ExpectedAtLeastOneArgument => {
                diag.span_label(self.span, "expected at least 1 argument here");
            }
            AttributeParseErrorReason::ExpectedList => {
                diag.span_label(self.span, "expected this to be a list");
            }
            AttributeParseErrorReason::DuplicateKey(key) => {
                diag.span_label(self.span, format!("found `{key}` used as a key more than once"));
                diag.code(E0538);
            }
            AttributeParseErrorReason::UnexpectedLiteral => {
                diag.span_label(self.span, "didn't expect a literal here");
                diag.code(E0565);
            }
            AttributeParseErrorReason::ExpectedNoArgs => {
                diag.span_label(self.span, "didn't expect any arguments here");
                diag.code(E0565);
            }
            AttributeParseErrorReason::ExpectedNameValue(None) => {
                // If the span is the entire attribute, the suggestion we add below this match already contains enough information
                if self.span != self.attr_span {
                    diag.span_label(
                        self.span,
                        format!("expected this to be of the form `... = \"...\"`"),
                    );
                }
            }
            AttributeParseErrorReason::ExpectedNameValue(Some(name)) => {
                diag.span_label(
                    self.span,
                    format!("expected this to be of the form `{name} = \"...\"`"),
                );
            }
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings,
                list: false,
            } => {
                let quote = if strings { '"' } else { '`' };
                match possibilities {
                    &[] => {}
                    &[x] => {
                        diag.span_label(
                            self.span,
                            format!("the only valid argument here is {quote}{x}{quote}"),
                        );
                    }
                    [first, second] => {
                        diag.span_label(self.span, format!("valid arguments are {quote}{first}{quote} or {quote}{second}{quote}"));
                    }
                    [first @ .., second_to_last, last] => {
                        let mut res = String::new();
                        for i in first {
                            res.push_str(&format!("{quote}{i}{quote}, "));
                        }
                        res.push_str(&format!(
                            "{quote}{second_to_last}{quote} or {quote}{last}{quote}"
                        ));

                        diag.span_label(self.span, format!("valid arguments are {res}"));
                    }
                }
            }
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings,
                list: true,
            } => {
                let quote = if strings { '"' } else { '`' };
                match possibilities {
                    &[] => {}
                    &[x] => {
                        diag.span_label(
                            self.span,
                            format!(
                                "this attribute is only valid with {quote}{x}{quote} as an argument"
                            ),
                        );
                    }
                    [first, second] => {
                        diag.span_label(self.span, format!("this attribute is only valid with either {quote}{first}{quote} or {quote}{second}{quote} as an argument"));
                    }
                    [first @ .., second_to_last, last] => {
                        let mut res = String::new();
                        for i in first {
                            res.push_str(&format!("{quote}{i}{quote}, "));
                        }
                        res.push_str(&format!(
                            "{quote}{second_to_last}{quote} or {quote}{last}{quote}"
                        ));

                        diag.span_label(self.span, format!("this attribute is only valid with one of the following arguments: {res}"));
                    }
                }
            }
            AttributeParseErrorReason::ExpectedIdentifier => {
                diag.span_label(self.span, "expected a valid identifier here");
            }
        }

        if let Some(link) = self.template.docs {
            diag.note(format!("for more information, visit <{link}>"));
        }
        let suggestions = self.template.suggestions(self.attr_style, &name);

        diag.span_suggestions(
            self.attr_span,
            if suggestions.len() == 1 {
                "must be of the form"
            } else {
                "try changing it to one of the following valid forms of the attribute"
            },
            suggestions,
            Applicability::HasPlaceholders,
        );

        diag
    }
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_attr_unsafe)]
#[note]
pub(crate) struct InvalidAttrUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Path,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_unsafe_attr_outside_unsafe)]
pub(crate) struct UnsafeAttrOutsideUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: UnsafeAttrOutsideUnsafeSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    attr_parsing_unsafe_attr_outside_unsafe_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct UnsafeAttrOutsideUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_meta_bad_delim)]
pub(crate) struct MetaBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    attr_parsing_meta_bad_delim_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct MetaBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_meta_item)]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    #[subdiagnostic]
    pub quote_ident_sugg: Option<InvalidMetaItemQuoteIdentSugg>,
    #[subdiagnostic]
    pub remove_neg_sugg: Option<InvalidMetaItemRemoveNegSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(attr_parsing_quote_ident_sugg, applicability = "machine-applicable")]
pub(crate) struct InvalidMetaItemQuoteIdentSugg {
    #[suggestion_part(code = "\"")]
    pub before: Span,
    #[suggestion_part(code = "\"")]
    pub after: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(attr_parsing_remove_neg_sugg, applicability = "machine-applicable")]
pub(crate) struct InvalidMetaItemRemoveNegSugg {
    #[suggestion_part(code = "")]
    pub negative_sign: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_suffixed_literal_in_attribute)]
#[help]
pub(crate) struct SuffixedLiteralInAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(attr_parsing_invalid_style)]
pub(crate) struct InvalidAttrStyle {
    pub name: AttrPath,
    pub is_used_as_inner: bool,
    #[note]
    pub target_span: Option<Span>,
    pub target: Target,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_empty_link_name, code = E0454)]
pub(crate) struct EmptyLinkName {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_link_framework_apple, code = E0455)]
pub(crate) struct LinkFrameworkApple {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_incompatible_wasm_link)]
pub(crate) struct IncompatibleWasmLink {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_link_requires_name, code = E0459)]
pub(crate) struct LinkRequiresName {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_raw_dylib_no_nul)]
pub(crate) struct RawDylibNoNul {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_raw_dylib_only_windows, code = E0455)]
pub(crate) struct RawDylibOnlyWindows {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_invalid_link_modifier)]
pub(crate) struct InvalidLinkModifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_multiple_modifiers)]
pub(crate) struct MultipleModifiers {
    #[primary_span]
    pub span: Span,
    pub modifier: Symbol,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_import_name_type_x86)]
pub(crate) struct ImportNameTypeX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_bundle_needs_static)]
pub(crate) struct BundleNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_whole_archive_needs_static)]
pub(crate) struct WholeArchiveNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_as_needed_compatibility)]
pub(crate) struct AsNeededCompatibility {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_import_name_type_raw)]
pub(crate) struct ImportNameTypeRaw {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(attr_parsing_limit_invalid)]
pub(crate) struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub value_span: Span,
    pub error_str: &'a str,
}
