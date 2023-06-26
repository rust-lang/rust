use std::{io, path::Path};

use rustc_macros::{Diagnostic, Subdiagnostic};

#[derive(Diagnostic)]
#[diag(rustdoc_couldnt_generate_documentation)]
pub(crate) struct CouldntGenerateDocumentation {
    pub(crate) error: String,
    #[subdiagnostic]
    pub(crate) file: Option<FailedToCreateOrModifyFile>,
}

#[derive(Subdiagnostic)]
#[note(rustdoc_failed_to_create_or_modify_file)]
pub(crate) struct FailedToCreateOrModifyFile {
    pub(crate) file: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_compilation_failed)]
pub(crate) struct CompilationFailed;

#[derive(Diagnostic)]
#[diag(rustdoc_load_string_error_read_fail)]
pub(crate) struct LoadStringErrorReadFail<'a> {
    pub(crate) file_path: &'a Path,
    pub(crate) err: io::Error,
}

#[derive(Diagnostic)]
#[diag(rustdoc_load_string_error_bad_utf8)]
pub(crate) struct LoadStringErrorBadUtf8<'a> {
    pub(crate) file_path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(rustdoc_scrape_examples_must_use_output_path_and_target_crate_together)]
pub(crate) struct ScrapeExamplesMustUseOutputPathAndTargetCrateTogether;

#[derive(Diagnostic)]
#[diag(rustdoc_scrape_examples_must_use_output_path_and_target_crate_with_scrape_tests)]
pub(crate) struct ScrapeExamplesMustUseOutputPathAndTargetCrateWithScrapeTests;

#[derive(Diagnostic)]
#[diag(rustdoc_load_examples_failed)]
pub(crate) struct LoadExamplesFailed {
    pub(crate) err: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_arguments_to_theme_must_be_files)]
#[help]
pub(crate) struct ArgumentsToThemeMustBeFiles {
    pub(crate) theme: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_arguments_to_theme_must_have_a_css_extension)]
#[help]
pub(crate) struct ArgumentsToThemeMustHaveACssExtension {
    pub(crate) theme: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_error_loading_theme_file)]
pub(crate) struct ErrorLoadingThemeFile {
    pub(crate) theme: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_theme_file_missing_css_rules_from_default_theme)]
#[warning]
#[help]
pub(crate) struct ThemeFileMissingCssRulesFromDefaultTheme {
    pub(crate) theme: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_unrecognized_emission_type)]
pub(crate) struct UnrecognizedEmissionType<'a> {
    pub(crate) kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_missing_file_operand)]
pub(crate) struct MissingFileOperand;

#[derive(Diagnostic)]
#[diag(rustdoc_too_many_file_operands)]
pub(crate) struct TooManyFileOperands;

#[derive(Diagnostic)]
#[diag(rustdoc_test_flag_must_be_passed_to_enable_no_run)]
pub(crate) struct TestFlagMustBePassedToEnableNoRun;

#[derive(Diagnostic)]
#[diag(rustdoc_extend_css_arg_must_be_a_file)]
pub(crate) struct ExtendCssArgMustBeAFile;

#[derive(Diagnostic)]
#[diag(rustdoc_cannot_use_both_out_dir_and_output_at_once)]
pub(crate) struct CannotUseBothOutDirAndOutputAtOnce;

#[derive(Diagnostic)]
#[diag(rustdoc_unknown_input_format)]
pub(crate) struct UnknownInputFormat<'a> {
    pub(crate) format: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_index_page_arg_must_be_a_file)]
pub(crate) struct IndexPageArgMustBeAFile;

#[derive(Diagnostic)]
#[diag(rustdoc_unknown_crate_type)]
pub(crate) struct UnknownCrateType {
    pub(crate) err: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_html_output_format_unsupported_for_show_coverage_option)]
pub(crate) struct HtmlOutputFormatUnsupportedForShowCoverageOption;

#[derive(Diagnostic)]
#[diag(rustdoc_generate_link_to_definition_option_can_only_be_used_with_html_output_format)]
pub(crate) struct GenerateLinkToDefinitionOptionCanOnlyBeUsedWithHtmlOutputFormat;

#[derive(Diagnostic)]
#[diag(rustdoc_flag_is_deprecated)]
#[note]
pub(crate) struct FlagIsDeprecated<'a> {
    pub(crate) flag: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_flag_no_longer_functions)]
#[note]
pub(crate) struct FlagNoLongerFunctions<'a> {
    flag: &'a str,
    #[subdiagnostic]
    help: Option<MayWantToUseDocumentPrivateItems>,
    #[subdiagnostic]
    warning: Option<SeeCve20181000622>,
}

#[derive(Subdiagnostic)]
#[help(rustdoc_may_want_to_use_document_private_items)]
pub(crate) struct MayWantToUseDocumentPrivateItems;

#[derive(Subdiagnostic)]
#[warning(rustdoc_see_cve_2018_1000622)]
pub(crate) struct SeeCve20181000622;

impl<'a> FlagNoLongerFunctions<'a> {
    pub(crate) fn new(flag: &'a str) -> Self {
        let (help, warning) = match flag {
            "no-defaults" | "passes" => (Some(MayWantToUseDocumentPrivateItems), None),
            "plugins" | "plugin-path" => (None, Some(SeeCve20181000622)),
            _ => (None, None),
        };
        Self { flag, help, warning }
    }
}
