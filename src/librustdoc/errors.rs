use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::Span;

use std::io;
use std::path::{Path, PathBuf};

#[derive(Diagnostic)]
#[diag(rustdoc_compilation_failed)]
pub struct CompilationFailed;

#[derive(LintDiagnostic)]
#[help]
#[diag(rustdoc_missing_crate_level_docs)]
pub struct MissingCrateLevelDocs {
    pub doc_rust_lang_org_channel: &'static str,
}

#[derive(Diagnostic)]
#[note]
#[diag(rustdoc_deprecated_attr)]
pub struct DeprecatedAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'a str,
    #[subdiagnostic]
    pub kind: Option<DeprecatedAttrKind>,
}

#[derive(Subdiagnostic)]
pub enum DeprecatedAttrKind {
    #[help(rustdoc_deprecated_attr_no_default_passes)]
    NoDefaultPasses,
    #[help(rustdoc_deprecated_attr_passes)]
    Passes,
    #[warning(rustdoc_deprecated_attr_plugins)]
    Plugins,
}

#[derive(Diagnostic)]
#[note]
#[help]
#[diag(rustdoc_could_not_resolve_path, code = "E0433")]
pub struct CouldNotResolvePath {
    #[primary_span]
    #[label]
    pub span: Span,
    pub path: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_unrecognized_emission_type)]
pub struct UnrecognizedEmissionType<'a> {
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_invalid_extern_html_root_url)]
pub struct InvalidExternHtmlRootUrl;

#[derive(Diagnostic)]
#[diag(rustdoc_missing_file_operand)]
pub struct MissingFileOperand;

#[derive(Diagnostic)]
#[diag(rustdoc_too_many_file_operands)]
pub struct TooManyFileOperands;

#[derive(Diagnostic)]
#[diag(rustdoc_no_run_flag_without_test_flag)]
pub struct NoRunFlagWithoutTestFlag;

#[derive(Diagnostic)]
#[diag(rustdoc_cannot_use_out_dir_and_output_flags)]
pub struct CannotUseOutDirAndOutputFlags;

#[derive(Diagnostic)]
#[diag(rustdoc_option_extend_css_arg_not_file)]
pub struct ExtendCssArgNotFile;

#[derive(Diagnostic)]
#[help]
#[diag(rustdoc_theme_arg_not_file)]
pub struct ThemeArgNotFile<'a> {
    pub theme_arg: &'a str,
}

#[derive(Diagnostic)]
#[help]
#[diag(rustdoc_theme_arg_not_css_file)]
pub struct ThemeArgNotCssFile<'a> {
    pub theme_arg: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_error_loading_theme_file)]
pub struct ErrorLoadingThemeFile<'a> {
    pub theme_arg: &'a str,
}

#[derive(Diagnostic)]
#[help]
#[diag(rustdoc_theme_file_missing_default_theme_css_rules)]
pub struct ThemeFileMissingDefaultThemeCssRules<'a> {
    pub theme_arg: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_unknown_input_format)]
pub struct UnknownInputFormat<'a> {
    pub input_format_arg: &'a str,
}

#[derive(Diagnostic)]
#[diag(rustdoc_index_page_arg_not_file)]
pub struct IndexPageArgNotFile;

#[derive(Diagnostic)]
#[diag(rustdoc_unknown_crate_type)]
pub struct UnknownCrateType {
    pub error: String,
}

#[derive(Diagnostic)]
#[diag(rustdoc_html_output_not_supported_with_show_coverage_flag)]
pub struct HtmlOutputNotSupportedWithShowCoverageFlag;

#[derive(Diagnostic)]
#[diag(rustdoc_generate_link_to_definition_flag_not_with_html_output_format)]
pub struct GenerateLinkToDefinitionFlagNotWithHtmlOutputFormat;

#[derive(Diagnostic)]
#[diag(rustdoc_scrape_examples_output_path_and_target_crate_not_used_together)]
pub struct ScrapeExamplesOutputPathAndTargetCrateNotTogether;

#[derive(Diagnostic)]
#[diag(rustdoc_scrape_tests_not_with_scrape_examples_output_path_and_target_crate)]
pub struct ScrapeTestsNotWithScrapeExamplesOutputPathAndTargetCrate;

#[derive(Diagnostic)]
#[note]
#[diag(rustdoc_flag_deprecated)]
pub struct FlagDeprecated<'a> {
    pub flag: &'a str,
}

#[derive(Diagnostic)]
#[note]
#[diag(rustdoc_flag_removed)]
pub struct FlagRemoved<'a> {
    pub flag: &'a str,
    #[subdiagnostic]
    pub suggestion: Option<FlagRemovedSuggestion>,
}

#[derive(Subdiagnostic)]
pub enum FlagRemovedSuggestion {
    #[help(rustdoc_use_document_private_items_flag)]
    DocumentPrivateItems,
    #[warning(rustdoc_see_rustdoc_plugins_cve)]
    SeeRustdocPluginsCve,
}

#[derive(Diagnostic)]
#[diag(rustdoc_error_reading_file)]
pub struct ErrorReadingFile<'a> {
    pub file_path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(rustdoc_error_reading_file_not_utf8)]
pub struct ErrorReadingFileNotUtf8<'a> {
    pub file_path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(rustdoc_error_loading_examples)]
pub struct ErrorLoadingExamples {
    pub error: io::Error,
    pub path: String,
}
