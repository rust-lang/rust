// use rustc_errors::ErrorGuaranteed;
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(metadata::rlib_required)]
pub struct RlibRequired {
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::lib_required)]
pub struct LibRequired {
    pub crate_name: String,
    pub kind: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::crate_dep_multiple)]
#[help]
pub struct CrateDepMultiple {
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::two_panic_runtimes)]
pub struct TwoPanicRuntimes {
    pub prev_name: String,
    pub cur_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::bad_panic_strategy)]
pub struct BadPanicStrategy {
    pub runtime: String,
    pub strategy: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::required_panic_strategy)]
pub struct RequiredPanicStrategy {
    pub crate_name: String,
    pub found_strategy: String,
    pub desired_strategy: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::incompatible_panic_in_drop_strategy)]
pub struct IncompatiblePanicInDropStrategy {
    pub crate_name: String,
    pub found_strategy: String,
    pub desired_strategy: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_names_in_link)]
pub struct MultipleNamesInLink {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_kinds_in_link)]
pub struct MultipleKindsInLink {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_name_form)]
pub struct LinkNameForm {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_kind_form)]
pub struct LinkKindForm {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_modifiers_form)]
pub struct LinkModifiersForm {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_cfg_form)]
pub struct LinkCfgForm {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::wasm_import_form)]
pub struct WasmImportForm {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::empty_link_name, code = "E0454")]
pub struct EmptyLinkName {
    #[label]
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_framework_apple, code = "E0455")]
pub struct LinkFrameworkApple {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::framework_only_windows, code = "E0455")]
pub struct FrameworkOnlyWindows {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::unknown_link_kind, code = "E0458")]
pub struct UnknownLinkKind {
    #[label]
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_link_modifiers)]
pub struct MultipleLinkModifiers {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_cfgs)]
pub struct MultipleCfgs {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_cfg_single_predicate)]
pub struct LinkCfgSinglePredicate {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_wasm_import)]
pub struct MultipleWasmImport {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::unexpected_link_arg)]
pub struct UnexpectedLinkArg {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::invalid_link_modifier)]
pub struct InvalidLinkModifier {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_modifiers)]
pub struct MultipleModifiers {
    #[primary_span]
    pub span: Span,
    pub modifier: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::bundle_needs_static)]
pub struct BundleNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::whole_archive_needs_static)]
pub struct WholeArchiveNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::as_needed_compatibility)]
pub struct AsNeededCompatibility {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::unknown_link_modifier)]
pub struct UnknownLinkModifier {
    #[primary_span]
    pub span: Span,
    pub modifier: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::incompatible_wasm_link)]
pub struct IncompatibleWasmLink {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_requires_name, code = "E0459")]
pub struct LinkRequiresName {
    #[label]
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::raw_dylib_no_nul)]
pub struct RawDylibNoNul {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::link_ordinal_raw_dylib)]
pub struct LinkOrdinalRawDylib {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::lib_framework_apple)]
pub struct LibFrameworkApple;

#[derive(SessionDiagnostic)]
#[diag(metadata::empty_renaming_target)]
pub struct EmptyRenamingTarget {
    pub lib_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::renaming_no_link)]
pub struct RenamingNoLink {
    pub lib_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_renamings)]
pub struct MultipleRenamings {
    pub lib_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::no_link_mod_override)]
pub struct NoLinkModOverride {
    #[primary_span]
    pub span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::unsupported_abi_i686)]
pub struct UnsupportedAbiI686 {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::unsupported_abi)]
pub struct UnsupportedAbi {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::fail_create_file_encoder)]
pub struct FailCreateFileEncoder {
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::fail_seek_file)]
pub struct FailSeekFile {
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::fail_write_file)]
pub struct FailWriteFile {
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::crate_not_panic_runtime)]
pub struct CrateNotPanicRuntime {
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::no_panic_strategy)]
pub struct NoPanicStrategy {
    pub crate_name: String,
    pub strategy: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::profiler_builtins_needs_core)]
pub struct ProfilerBuiltinsNeedsCore;

#[derive(SessionDiagnostic)]
#[diag(metadata::not_profiler_runtime)]
pub struct NotProfilerRuntime {
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::no_multiple_global_alloc)]
pub struct NoMultipleGlobalAlloc {
    #[primary_span]
    #[label]
    pub span2: Span,
    #[label(metadata::prev_global_alloc)]
    pub span1: Span,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::conflicting_global_alloc)]
pub struct ConflictingGlobalAlloc {
    pub crate_name: String,
    pub other_crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::global_alloc_required)]
pub struct GlobalAllocRequired;

#[derive(SessionDiagnostic)]
#[diag(metadata::no_transitive_needs_dep)]
pub struct NoTransitiveNeedsDep {
    pub crate_name: String,
    pub needs_crate_name: String,
    pub deps_crate_name: String,
}
