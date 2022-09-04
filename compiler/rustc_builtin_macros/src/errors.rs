use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

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
    // TODO: This should be a `tool_only_span_suggestion`
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
