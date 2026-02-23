use std::num::IntErrorKind;

use rustc_ast::{self as ast};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level,
};
use rustc_feature::AttributeTemplate;
use rustc_hir::AttrPath;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};
use rustc_target::spec::TargetTuple;

#[derive(Diagnostic)]
#[diag("invalid predicate `{$predicate}`", code = E0537)]
pub(crate) struct InvalidPredicate {
    #[primary_span]
    pub span: Span,

    pub predicate: String,
}

#[derive(Diagnostic)]
#[diag("{$attr_str} attribute cannot have empty value")]
pub(crate) struct DocAliasEmpty<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("{$char_} character isn't allowed in {$attr_str}")]
pub(crate) struct DocAliasBadChar<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
    pub char_: char,
}

#[derive(Diagnostic)]
#[diag("{$attr_str} cannot start or end with ' '")]
pub(crate) struct DocAliasStartEnd<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("`#[{$name})]` is missing a `{$field}` argument")]
pub(crate) struct CguFieldsMissing<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a AttrPath,
    pub field: Symbol,
}

#[derive(Diagnostic)]
#[diag("`#![doc({$attr_name} = \"...\")]` isn't allowed as a crate-level attribute")]
pub(crate) struct DocAttrNotCrateLevel {
    #[primary_span]
    pub span: Span,
    pub attr_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("nonexistent keyword `{$keyword}` used in `#[doc(keyword = \"...\")]`")]
#[help("only existing keywords are allowed in core/std")]
pub(crate) struct DocKeywordNotKeyword {
    #[primary_span]
    pub span: Span,
    pub keyword: Symbol,
}

#[derive(Diagnostic)]
#[diag("nonexistent builtin attribute `{$attribute}` used in `#[doc(attribute = \"...\")]`")]
#[help("only existing builtin attributes are allowed in core/std")]
pub(crate) struct DocAttributeNotAttribute {
    #[primary_span]
    pub span: Span,
    pub attribute: Symbol,
}

#[derive(Diagnostic)]
#[diag("missing 'since'", code = E0542)]
pub(crate) struct MissingSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("missing 'note'", code = E0543)]
pub(crate) struct MissingNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("multiple stability levels", code = E0544)]
pub(crate) struct MultipleStabilityLevels {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`issue` must be a non-zero numeric string or \"none\"", code = E0545)]
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
    #[label("`issue` must not be \"0\", use \"none\" instead")]
    MustNotBeZero {
        #[primary_span]
        span: Span,
    },

    #[label("cannot parse integer from empty string")]
    Empty {
        #[primary_span]
        span: Span,
    },

    #[label("invalid digit found in string")]
    InvalidDigit {
        #[primary_span]
        span: Span,
    },

    #[label("number too large to fit in target type")]
    PosOverflow {
        #[primary_span]
        span: Span,
    },

    #[label("number too small to fit in target type")]
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
#[diag("missing 'feature'", code = E0546)]
pub(crate) struct MissingFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("'feature' is not an identifier", code = E0546)]
pub(crate) struct NonIdentFeature {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("missing 'issue'", code = E0547)]
pub(crate) struct MissingIssue {
    #[primary_span]
    pub span: Span,
}

// FIXME: Why is this the same error code as `InvalidReprHintNoParen` and `InvalidReprHintNoValue`?
// It is more similar to `IncorrectReprFormatGeneric`.
#[derive(Diagnostic)]
#[diag("incorrect `repr(packed)` attribute format: `packed` takes exactly one parenthesized argument, or no parentheses at all", code = E0552)]
pub(crate) struct IncorrectReprFormatPackedOneOrZeroArg {
    #[primary_span]
    pub span: Span,
}
#[derive(Diagnostic)]
#[diag("incorrect `repr(packed)` attribute format: `packed` expects a literal integer as argument", code = E0552)]
pub(crate) struct IncorrectReprFormatPackedExpectInteger {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid representation hint: `{$name}` does not take a parenthesized argument list", code = E0552)]
pub(crate) struct InvalidReprHintNoParen {
    #[primary_span]
    pub span: Span,

    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("invalid representation hint: `{$name}` does not take a value", code = E0552)]
pub(crate) struct InvalidReprHintNoValue {
    #[primary_span]
    pub span: Span,

    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("invalid `repr(align)` attribute: `align` needs an argument", code = E0589)]
pub(crate) struct InvalidReprAlignNeedArg {
    #[primary_span]
    #[suggestion(
        "supply an argument here",
        code = "align(...)",
        applicability = "has-placeholders"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid `repr({$repr_arg})` attribute: {$error_part}", code = E0589)]
pub(crate) struct InvalidReprGeneric<'a> {
    #[primary_span]
    pub span: Span,

    pub repr_arg: String,
    pub error_part: &'a str,
}

#[derive(Diagnostic)]
#[diag("incorrect `repr(align)` attribute format: `align` takes exactly one argument in parentheses", code = E0693)]
pub(crate) struct IncorrectReprFormatAlignOneArg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("incorrect `repr(align)` attribute format: `align` expects a literal integer as argument", code = E0693)]
pub(crate) struct IncorrectReprFormatExpectInteger {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("incorrect `repr({$repr_arg})` attribute format", code = E0693)]
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
        "use parentheses instead",
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
        "use parentheses instead",
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
#[diag("`rustc_promotable` attribute must be paired with either a `rustc_const_unstable` or a `rustc_const_stable` attribute", code = E0717)]
pub(crate) struct RustcPromotablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute", code = E0789)]
pub(crate) struct RustcAllowedUnstablePairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("suggestions on deprecated items are unstable")]
pub(crate) struct DeprecatedItemSuggestion {
    #[primary_span]
    pub span: Span,

    #[help("add `#![feature(deprecated_suggestion)]` to the crate root")]
    pub is_nightly: bool,

    #[note("see #94785 for more details")]
    pub details: (),
}

#[derive(Diagnostic)]
#[diag("expected single version literal")]
pub(crate) struct ExpectedSingleVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a version literal")]
pub(crate) struct ExpectedVersionLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$name}` expects a list of feature names")]
pub(crate) struct ExpectsFeatureList {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag("`{$name}` expects feature names")]
pub(crate) struct ExpectsFeatures {
    #[primary_span]
    pub span: Span,

    pub name: String,
}

#[derive(Diagnostic)]
#[diag("'since' must be a Rust version number, such as \"1.31.0\"")]
pub(crate) struct InvalidSince {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`soft` should not have any arguments")]
pub(crate) struct SoftNoArgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unknown version literal format, assuming it refers to a future version")]
pub(crate) struct UnknownVersionLiteral {
    #[primary_span]
    pub span: Span,
}

// FIXME(jdonszelmann) duplicated from `rustc_passes`, remove once `check_attr` is integrated.
#[derive(Diagnostic)]
#[diag("multiple `{$name}` attributes")]
pub(crate) struct UnusedMultiple {
    #[primary_span]
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("`export_name` may not contain null characters", code = E0648)]
pub(crate) struct NullOnExport {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`link_section` may not contain null characters", code = E0648)]
pub(crate) struct NullOnLinkSection {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::class!` may not contain null characters")]
pub(crate) struct NullOnObjcClass {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::selector!` may not contain null characters")]
pub(crate) struct NullOnObjcSelector {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::class!` expected a string literal")]
pub(crate) struct ObjcClassExpectedStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`objc::selector!` expected a string literal")]
pub(crate) struct ObjcSelectorExpectedStringLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("stability attributes may not be used outside of the standard library", code = E0734)]
pub(crate) struct StabilityOutsideStd {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected at least one confusable name")]
pub(crate) struct EmptyConfusables {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[help("`#[{$name}]` can {$only}be applied to {$applied}")]
#[diag("`#[{$name}]` attribute cannot be used on {$target}")]
pub(crate) struct InvalidTarget {
    #[primary_span]
    #[suggestion(
        "remove the attribute",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub span: Span,
    pub name: AttrPath,
    pub target: &'static str,
    pub applied: DiagArgValue,
    pub only: &'static str,
}

#[derive(Diagnostic)]
#[diag("invalid alignment value: {$error_part}", code = E0589)]
pub(crate) struct InvalidAlignmentValue {
    #[primary_span]
    pub span: Span,
    pub error_part: &'static str,
}

#[derive(Diagnostic)]
#[diag("meta item in `repr` must be an identifier", code = E0565)]
pub(crate) struct ReprIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unrecognized representation hint", code = E0552)]
#[help(
    "valid reprs are `Rust` (default), `C`, `align`, `packed`, `transparent`, `simd`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `i128`, `u128`, `isize`, `usize`"
)]
#[note(
    "for more information, visit <https://doc.rust-lang.org/reference/type-layout.html?highlight=repr#representations>"
)]
pub(crate) struct UnrecognizedReprHint {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("item annotated with `#[unstable_feature_bound]` should not be stable")]
#[help(
    "if this item is meant to be stable, do not use any functions annotated with `#[unstable_feature_bound]`. Otherwise, mark this item as unstable with `#[unstable]`"
)]
pub(crate) struct UnstableFeatureBoundIncompatibleStability {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute incompatible with `#[unsafe(naked)]`", code = E0736)]
pub(crate) struct NakedFunctionIncompatibleAttribute {
    #[primary_span]
    #[label("the `{$attr}` attribute is incompatible with `#[unsafe(naked)]`")]
    pub span: Span,
    #[label("function marked with `#[unsafe(naked)]` here")]
    pub naked_span: Span,
    pub attr: String,
}

#[derive(Diagnostic)]
#[diag("ordinal value in `link_ordinal` is too large: `{$ordinal}`")]
#[note("the value may not exceed `u16::MAX`")]
pub(crate) struct LinkOrdinalOutOfRange {
    #[primary_span]
    pub span: Span,
    pub ordinal: u128,
}

#[derive(Diagnostic)]
#[diag("element count in `rustc_scalable_vector` is too large: `{$n}`")]
#[note("the value may not exceed `u16::MAX`")]
pub(crate) struct RustcScalableVectorCountOutOfRange {
    #[primary_span]
    pub span: Span,
    pub n: u128,
}

#[derive(Diagnostic)]
#[diag("attribute requires {$opt} to be enabled")]
pub(crate) struct AttributeRequiresOpt {
    #[primary_span]
    pub span: Span,
    pub opt: &'static str,
}

pub(crate) enum AttributeParseErrorReason<'a> {
    ExpectedNoArgs,
    ExpectedStringLiteral {
        byte_string: Option<Span>,
    },
    ExpectedFilenameLiteral,
    ExpectedIntegerLiteral,
    ExpectedIntegerLiteralInRange {
        lower_bound: isize,
        upper_bound: isize,
    },
    ExpectedAtLeastOneArgument,
    ExpectedSingleArgument,
    ExpectedList,
    ExpectedListOrNoArgs,
    ExpectedListWithNumArgsOrMore {
        args: usize,
    },
    ExpectedNameValueOrNoArgs,
    ExpectedNonEmptyStringLiteral,
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

/// A description of a thing that can be parsed using an attribute parser.
#[derive(Copy, Clone)]
pub enum ParsedDescription {
    /// Used when parsing attributes.
    Attribute,
    /// Used when parsing some macros, such as the `cfg!()` macro.
    Macro,
}

pub(crate) struct AttributeParseError<'a> {
    pub(crate) span: Span,
    pub(crate) attr_span: Span,
    pub(crate) template: AttributeTemplate,
    pub(crate) path: AttrPath,
    pub(crate) description: ParsedDescription,
    pub(crate) reason: AttributeParseErrorReason<'a>,
    pub(crate) suggestions: Vec<String>,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for AttributeParseError<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let name = self.path.to_string();

        let description = match self.description {
            ParsedDescription::Attribute => "attribute",
            ParsedDescription::Macro => "macro",
        };

        let mut diag = Diag::new(dcx, level, format!("malformed `{name}` {description} input"));
        diag.span(self.attr_span);
        diag.code(E0539);
        match self.reason {
            AttributeParseErrorReason::ExpectedStringLiteral { byte_string } => {
                if let Some(start_point_span) = byte_string {
                    diag.span_suggestion(
                        start_point_span,
                        "consider removing the prefix",
                        "",
                        Applicability::MaybeIncorrect,
                    );
                    diag.note("expected a normal string literal, not a byte string literal");

                    return diag;
                } else {
                    diag.span_label(self.span, "expected a string literal here");
                }
            }
            AttributeParseErrorReason::ExpectedFilenameLiteral => {
                diag.span_label(self.span, "expected a filename string literal here");
            }
            AttributeParseErrorReason::ExpectedIntegerLiteral => {
                diag.span_label(self.span, "expected an integer literal here");
            }
            AttributeParseErrorReason::ExpectedIntegerLiteralInRange {
                lower_bound,
                upper_bound,
            } => {
                diag.span_label(
                    self.span,
                    format!(
                        "expected an integer literal in the range of {lower_bound}..={upper_bound}"
                    ),
                );
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
            AttributeParseErrorReason::ExpectedListOrNoArgs => {
                diag.span_label(self.span, "expected a list or no arguments here");
            }
            AttributeParseErrorReason::ExpectedListWithNumArgsOrMore { args } => {
                diag.span_label(self.span, format!("expected {args} or more items"));
            }
            AttributeParseErrorReason::ExpectedNameValueOrNoArgs => {
                diag.span_label(self.span, "didn't expect a list here");
            }
            AttributeParseErrorReason::ExpectedNonEmptyStringLiteral => {
                diag.span_label(self.span, "string is not allowed to be empty");
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
                                "this {description} is only valid with {quote}{x}{quote} as an argument"
                            ),
                        );
                    }
                    [first, second] => {
                        diag.span_label(self.span, format!("this {description} is only valid with either {quote}{first}{quote} or {quote}{second}{quote} as an argument"));
                    }
                    [first @ .., second_to_last, last] => {
                        let mut res = String::new();
                        for i in first {
                            res.push_str(&format!("{quote}{i}{quote}, "));
                        }
                        res.push_str(&format!(
                            "{quote}{second_to_last}{quote} or {quote}{last}{quote}"
                        ));

                        diag.span_label(self.span, format!("this {description} is only valid with one of the following arguments: {res}"));
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

        if self.suggestions.len() < 4 {
            diag.span_suggestions(
                self.attr_span,
                if self.suggestions.len() == 1 {
                    "must be of the form".to_string()
                } else {
                    format!(
                        "try changing it to one of the following valid forms of the {description}"
                    )
                },
                self.suggestions,
                Applicability::HasPlaceholders,
            );
        }

        diag
    }
}

#[derive(Diagnostic)]
#[diag("`{$name}` is not an unsafe attribute")]
#[note("extraneous unsafe is not allowed in attributes")]
pub(crate) struct InvalidAttrUnsafe {
    #[primary_span]
    #[label("this is not an unsafe attribute")]
    pub span: Span,
    pub name: AttrPath,
}

#[derive(Diagnostic)]
#[diag("unsafe attribute used without unsafe")]
pub(crate) struct UnsafeAttrOutsideUnsafe {
    #[primary_span]
    #[label("usage of unsafe attribute")]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: Option<UnsafeAttrOutsideUnsafeSuggestion>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("wrap the attribute in `unsafe(...)`", applicability = "machine-applicable")]
pub(crate) struct UnsafeAttrOutsideUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("wrong meta list delimiters")]
pub(crate) struct MetaBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "the delimiters should be `(` and `)`",
    applicability = "machine-applicable"
)]
pub(crate) struct MetaBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(Diagnostic)]
#[diag("expected a literal (`1u8`, `1.0f32`, `\"string\"`, etc.) here, found {$descr}")]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    #[subdiagnostic]
    pub quote_ident_sugg: Option<InvalidMetaItemQuoteIdentSugg>,
    #[subdiagnostic]
    pub remove_neg_sugg: Option<InvalidMetaItemRemoveNegSugg>,
    #[label("{$descr}s are not allowed here")]
    pub label: Option<Span>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "surround the identifier with quotation marks to make it into a string literal",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidMetaItemQuoteIdentSugg {
    #[suggestion_part(code = "\"")]
    pub before: Span,
    #[suggestion_part(code = "\"")]
    pub after: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "negative numbers are not literals, try removing the `-` sign",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidMetaItemRemoveNegSugg {
    #[suggestion_part(code = "")]
    pub negative_sign: Span,
}

#[derive(Diagnostic)]
#[diag("suffixed literals are not allowed in attributes")]
#[help(
    "instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), use an unsuffixed version (`1`, `1.0`, etc.)"
)]
pub(crate) struct SuffixedLiteralInAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link name must not be empty", code = E0454)]
pub(crate) struct EmptyLinkName {
    #[primary_span]
    #[label("empty link name")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link kind `framework` is only supported on Apple targets", code = E0455)]
pub(crate) struct LinkFrameworkApple {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`wasm_import_module` is incompatible with other arguments in `#[link]` attributes")]
pub(crate) struct IncompatibleWasmLink {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[link]` attribute requires a `name = \"string\"` argument", code = E0459)]
pub(crate) struct LinkRequiresName {
    #[primary_span]
    #[label("missing `name` argument")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link name must not contain NUL characters if link kind is `raw-dylib`")]
pub(crate) struct RawDylibNoNul {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("link kind `raw-dylib` is only supported on Windows targets", code = E0455)]
pub(crate) struct RawDylibOnlyWindows {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "invalid linking modifier syntax, expected '+' or '-' prefix before one of: bundle, verbatim, whole-archive, as-needed, export-symbols"
)]
pub(crate) struct InvalidLinkModifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("multiple `{$modifier}` modifiers in a single `modifiers` argument")]
pub(crate) struct MultipleModifiers {
    #[primary_span]
    pub span: Span,
    pub modifier: Symbol,
}

#[derive(Diagnostic)]
#[diag("import name type is only supported on x86")]
pub(crate) struct ImportNameTypeX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("linking modifier `bundle` is only compatible with `static` linking kind")]
pub(crate) struct BundleNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("linking modifier `export-symbols` is only compatible with `static` linking kind")]
pub(crate) struct ExportSymbolsNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("linking modifier `whole-archive` is only compatible with `static` linking kind")]
pub(crate) struct WholeArchiveNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "linking modifier `as-needed` is only compatible with `dylib`, `framework` and `raw-dylib` linking kinds"
)]
pub(crate) struct AsNeededCompatibility {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("import name type can only be used with link kind `raw-dylib`")]
pub(crate) struct ImportNameTypeRaw {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`limit` must be a non-negative integer")]
pub(crate) struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label("{$error_str}")]
    pub value_span: Span,
    pub error_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("wrong `cfg_attr` delimiters")]
pub(crate) struct CfgAttrBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Diagnostic)]
#[diag(
    "doc alias attribute expects a string `#[doc(alias = \"a\")]` or a list of strings `#[doc(alias(\"a\", \"b\"))]`"
)]
pub(crate) struct DocAliasMalformed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("definition of an unknown lang item: `{$name}`", code = E0522)]
pub(crate) struct UnknownLangItem {
    #[primary_span]
    #[label("definition of unknown lang item `{$name}`")]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("target `{$current_target}` does not support `#[instruction_set({$instruction_set}::*)]`")]
pub(crate) struct UnsupportedInstructionSet<'a> {
    #[primary_span]
    pub span: Span,
    pub instruction_set: Symbol,
    pub current_target: &'a TargetTuple,
}
