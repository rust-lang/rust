use rustc_errors::{fluent, AddSubdiagnostic, Applicability, Diagnostic};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_args_after_clobber_abi)]
pub(crate) struct AsmArgsAfterClobberAbi {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros::abi_label)]
    pub(crate) abi_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_args_after_options)]
pub(crate) struct AsmArgsAfterOptions {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros::previous_options_label)]
    pub(crate) options_spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_args_named_after_explicit_register)]
pub(crate) struct AsmArgsNamedAfterExplicitRegister {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros::register_label)]
    pub(crate) register_spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_args_positional_after_named_or_explicit_register)]
pub(crate) struct AsmArgsPositionalAfterNamedOrExplicitRegister {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros::named_label)]
    pub(crate) named_spans: Vec<Span>,
    #[label(builtin_macros::register_label)]
    pub(crate) register_spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_cannot_be_used_with)]
pub(crate) struct AsmCannotBeUsedWith {
    pub(crate) left: &'static str,
    pub(crate) right: &'static str,
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_clobber_abi_after_options)]
pub(crate) struct AsmClobberAbiAfterOptions {
    #[primary_span]
    pub(crate) span: Span,
    #[label(builtin_macros::options_label)]
    pub(crate) options_spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_clobber_abi_needs_an_abi)]
pub(crate) struct AsmClobberAbiNeedsAnAbi {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_clobber_abi_needs_explicit_registers)]
pub(crate) struct AsmClobberAbiNeedsExplicitRegisters {
    #[primary_span]
    #[label]
    pub(crate) spans: Vec<Span>,
    #[label(builtin_macros::abi_label)]
    pub(crate) abi_spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_duplicate_argument)]
pub(crate) struct AsmDuplicateArgument {
    pub(crate) name: Symbol,
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(builtin_macros::previously_label)]
    pub(crate) prev_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_duplicate_option)]
pub(crate) struct AsmDuplicateOption {
    pub(crate) symbol: Symbol,
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    // FIXME: This should be a `tool_only_span_suggestion`
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub(crate) full_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_expected_operand_options_or_template_string)]
pub(crate) struct AsmExpectedOperandOptionsOrTemplateString {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_expected_operand_clobber_abi_options_or_template_string)]
pub(crate) struct AsmExpectedOperandClobberAbiOptionsOrTemplateString {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_expected_path_arg_to_sym)]
pub(crate) struct AsmExpectedPathArgToSym {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_expected_register_class_or_explicit_register)]
pub(crate) struct AsmExpectedRegisterClassOrExplicitRegister {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_expected_string_literal)]
pub(crate) struct AsmExpectedStringLiteral {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_expected_token_comma)]
pub(crate) struct AsmExpectedTokenComma {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_explicit_register_arg_with_name)]
pub(crate) struct AsmExplicitRegisterArgWithName {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_no_argument_named)]
pub(crate) struct AsmNoArgumentNamed {
    pub(crate) name: String,
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_option_must_be_combined_with_either)]
pub(crate) struct AsmOptionMustBeCombinedWithEither {
    pub(crate) option: &'static str,
    pub(crate) left: &'static str,
    pub(crate) right: &'static str,
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_option_noreturn_with_outputs)]
pub(crate) struct AsmOptionNoreturnWithOutputs {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_option_pure_needs_one_output)]
pub(crate) struct AsmOptionPureNeedsOneOutput {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_options_mutually_exclusive)]
pub(crate) struct AsmOptionsMutuallyExclusive {
    pub(crate) left: &'static str,
    pub(crate) right: &'static str,
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_requires_template_string_arg)]
pub(crate) struct AsmRequiresTemplateStringArg {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_template_modifier_single_char)]
pub(crate) struct AsmTemplateModifierSingleChar {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::asm_underscore_for_input_operands)]
pub(crate) struct AsmUnderscoreForInputOperands {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::boolean_expression_required)]
pub struct BooleanExpressionRequired {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::unexpected_string_literal)]
pub struct UnexpectedStringLiteral {
    #[primary_span]
    #[suggestion_short(code = ", ", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::argument_expression_required)]
pub struct ArgumentExpressionRequired {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::not_specified)]
pub struct NotSpecified {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::multiple_paths_specified)]
pub struct MultiplePathsSpecified {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::unallowed_literal_path)]
pub struct UnallowedLiteralPath {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::unaccepted_arguments)]
pub struct UnacceptedArguments {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::nondeterministic_access)]
pub struct NondeterministicAccess {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::compile_error)]
pub struct CompileError {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::byte_string_literal_concatenate)]
pub struct ByteStringLiteralConcatenate {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::missing_literal)]
#[note]
pub struct MissingLiteral {
    #[primary_span]
    pub span: Span,
}

pub enum Snippet {
    ByteCharacter { span: Span, snippet: String },
    ByteString { span: Span, snippet: String },
    WrappingNumberInArray { span: Span, snippet: String },
}

impl AddSubdiagnostic for Snippet {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        match self {
            Self::ByteCharacter { span, snippet } => {
                diag.span_suggestion(
                    span,
                    fluent::builtin_macros::use_byte_character,
                    snippet,
                    Applicability::MachineApplicable,
                );
            }
            Self::ByteString { span, snippet } => {
                diag.span_suggestion(
                    span,
                    fluent::builtin_macros::use_byte_string,
                    snippet,
                    Applicability::MachineApplicable,
                );
            }
            Self::WrappingNumberInArray { span, snippet } => {
                diag.span_suggestion(
                    span,
                    fluent::builtin_macros::wrap_number_in_array,
                    snippet,
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::character_literals_concatenate)]
pub struct CharacterLiteralsConcatenate {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<Snippet>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::string_literals_concatenate)]
pub struct StringLiteralsConcatenate {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<Snippet>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::float_literals_concatenate)]
pub struct FloatLiteralsConcatenate {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::boolean_literals_concatenate)]
pub struct BooleanLiteralsConcatenate {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::numeric_literals_concatenate)]
pub struct NumericLiteralsConcatenate {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<Snippet>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::out_of_bound_numeric_literal)]
pub struct OutOfBoundNumericLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::invalid_numeric_literal)]
pub struct InvalidNumericLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::doubly_nested_array_concatenate)]
pub struct DoublyNestedArrayConcatenate {
    #[primary_span]
    pub span: Span,
    #[note]
    pub note: Option<()>,
    #[help]
    pub help: Option<()>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::invalid_repeat_count)]
pub struct InvalidRepeatCount {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[note]
#[diag(builtin_macros::byte_literal_expected)]
pub struct ByteLiteralExpected {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::missing_arguments)]
pub struct MissingArguments {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::comma_expected)]
pub struct CommaExpected {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::ident_args_required)]
pub struct IdentArgsRequired {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::not_applicable_derive, code = "E0774")]
pub struct NotApplicableDerive {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(builtin_macros::item_label)]
    pub item_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::trait_path_expected, code = "E0777")]
pub struct TraitPathExpected<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub help_msg: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::path_rejected)]
pub struct PathRejected<'a> {
    #[primary_span]
    #[label]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    pub title: &'a str,
    pub action: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::no_default)]
pub struct NoDefault {
    #[primary_span]
    pub span: Span,
    // FIXME: This should be a `tool_only_span_suggestion`
    #[subdiagnostic]
    pub suggestions: Vec<NoDefaultSuggestion>,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(
    builtin_macros::variant_suggestion,
    code = "#[default] {ident}",
    applicability = "maybe-incorrect"
)]
pub struct NoDefaultSuggestion {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::multiple_declared_defaults)]
pub struct MultipleDeclaredDefaults {
    #[primary_span]
    pub span: Span,
    #[label(builtin_macros::multiple_declared_defaults_first)]
    pub first: Span,
    #[label(builtin_macros::multiple_declared_defaults_additional)]
    pub variants: Vec<Span>,
    // FIXME: This should be a `tool_only_multipart_suggestion`
    #[subdiagnostic]
    pub suggestions: Vec<MultipleDeclaredDefaultsSuggestion>,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(builtin_macros::variant_suggestion, applicability = "maybe-incorrect")]
pub struct MultipleDeclaredDefaultsSuggestion {
    #[suggestion_part(code = "")]
    pub span: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::default_not_allowed)]
pub struct DefaultNotAllowed {
    #[primary_span]
    pub span: Span,
    #[help]
    pub help: Option<()>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::default_non_exhaustive)]
pub struct DefaultNonExhaustive {
    #[primary_span]
    pub span: Span,
    #[label(builtin_macros::default_non_exhaustive_instruction)]
    pub non_exhustive_attr: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::multiple_default_attributes)]
pub struct MultipleDefaultAttributes<'a> {
    #[primary_span]
    pub span: Span,
    #[label(builtin_macros::multiple_default_attributes_first)]
    pub first: Span,
    #[label(builtin_macros::multiple_default_attributes_rest)]
    pub rest: Span,
    #[help(builtin_macros::multiple_default_attributes_suggestion_text)]
    pub help: Vec<Span>,
    pub suggestion_text: &'a str,
    // FIXME: This should be a `tool_only_multipart_suggestion`
    #[subdiagnostic]
    pub suggestions: MultipleDefaultAttributesSuggestions,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(
    builtin_macros::multiple_default_attributes_suggestion_text,
    applicability = "machine-applicable"
)]
pub struct MultipleDefaultAttributesSuggestions {
    #[suggestion_part(code = "")]
    pub span: Vec<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::default_not_accept_value)]
pub struct DefaultNotAcceptValue {
    #[primary_span]
    #[suggestion_hidden(code = "#[default]", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::unallowed_derive)]
pub struct UnallowedDerive {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::cannot_be_derived_unions)]
pub struct CannotBeDerivedUnions {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::empty_argument)]
pub struct EmptyArgument {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::env_expand_error)]
pub struct EnvExpandError<'a> {
    #[primary_span]
    pub span: Span,
    pub msg: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::format_string_argument_required)]
pub struct FormatStringArgumentRequired {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::duplicated_argument)]
pub struct DuplicatedArgument {
    #[primary_span]
    pub span: Span,
    #[label(builtin_macros::multiple_default_attributes_first)]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[diag(builtin_macros::invalid_positional_arguments)]
pub struct InvalidPositionalArguments {
    #[primary_span]
    pub span: Span,
    #[label(builtin_macros::invalid_positional_arguments_names)]
    pub names: Vec<Span>,
}
