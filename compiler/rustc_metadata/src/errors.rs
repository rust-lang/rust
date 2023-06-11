use std::{
    io::Error,
    path::{Path, PathBuf},
};

use rustc_errors::{error_code, ErrorGuaranteed, IntoDiagnostic};
use rustc_macros::Diagnostic;
use rustc_session::config;
use rustc_span::{sym, Span, Symbol};
use rustc_target::spec::{PanicStrategy, TargetTriple};

use crate::fluent_generated as fluent;
use crate::locator::CrateFlavor;

#[derive(Diagnostic)]
#[diag(metadata_rlib_required)]
pub struct RlibRequired {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_lib_required)]
pub struct LibRequired<'a> {
    pub crate_name: Symbol,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_rustc_lib_required)]
#[help]
pub struct RustcLibRequired<'a> {
    pub crate_name: Symbol,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_crate_dep_multiple)]
#[help]
pub struct CrateDepMultiple {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_two_panic_runtimes)]
pub struct TwoPanicRuntimes {
    pub prev_name: Symbol,
    pub cur_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_bad_panic_strategy)]
pub struct BadPanicStrategy {
    pub runtime: Symbol,
    pub strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag(metadata_required_panic_strategy)]
pub struct RequiredPanicStrategy {
    pub crate_name: Symbol,
    pub found_strategy: PanicStrategy,
    pub desired_strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag(metadata_incompatible_panic_in_drop_strategy)]
pub struct IncompatiblePanicInDropStrategy {
    pub crate_name: Symbol,
    pub found_strategy: PanicStrategy,
    pub desired_strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_names_in_link)]
pub struct MultipleNamesInLink {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_kinds_in_link)]
pub struct MultipleKindsInLink {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_name_form)]
pub struct LinkNameForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_kind_form)]
pub struct LinkKindForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_modifiers_form)]
pub struct LinkModifiersForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_cfg_form)]
pub struct LinkCfgForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_wasm_import_form)]
pub struct WasmImportForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_empty_link_name, code = "E0454")]
pub struct EmptyLinkName {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_framework_apple, code = "E0455")]
pub struct LinkFrameworkApple {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_framework_only_windows, code = "E0455")]
pub struct FrameworkOnlyWindows {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_unknown_link_kind, code = "E0458")]
pub struct UnknownLinkKind<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_link_modifiers)]
pub struct MultipleLinkModifiers {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_cfgs)]
pub struct MultipleCfgs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_cfg_single_predicate)]
pub struct LinkCfgSinglePredicate {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_wasm_import)]
pub struct MultipleWasmImport {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_unexpected_link_arg)]
pub struct UnexpectedLinkArg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_invalid_link_modifier)]
pub struct InvalidLinkModifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_modifiers)]
pub struct MultipleModifiers<'a> {
    #[primary_span]
    pub span: Span,
    pub modifier: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_bundle_needs_static)]
pub struct BundleNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_whole_archive_needs_static)]
pub struct WholeArchiveNeedsStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_as_needed_compatibility)]
pub struct AsNeededCompatibility {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_unknown_link_modifier)]
pub struct UnknownLinkModifier<'a> {
    #[primary_span]
    pub span: Span,
    pub modifier: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_incompatible_wasm_link)]
pub struct IncompatibleWasmLink {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_requires_name, code = "E0459")]
pub struct LinkRequiresName {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_raw_dylib_no_nul)]
pub struct RawDylibNoNul {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_link_ordinal_raw_dylib)]
pub struct LinkOrdinalRawDylib {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_lib_framework_apple)]
pub struct LibFrameworkApple;

#[derive(Diagnostic)]
#[diag(metadata_empty_renaming_target)]
pub struct EmptyRenamingTarget<'a> {
    pub lib_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_renaming_no_link)]
pub struct RenamingNoLink<'a> {
    pub lib_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_renamings)]
pub struct MultipleRenamings<'a> {
    pub lib_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_no_link_mod_override)]
pub struct NoLinkModOverride {
    #[primary_span]
    pub span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(metadata_unsupported_abi_i686)]
pub struct UnsupportedAbiI686 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_unsupported_abi)]
pub struct UnsupportedAbi {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_fail_create_file_encoder)]
pub struct FailCreateFileEncoder {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_fail_seek_file)]
pub struct FailSeekFile {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_fail_write_file)]
pub struct FailWriteFile {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_crate_not_panic_runtime)]
pub struct CrateNotPanicRuntime {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_no_panic_strategy)]
pub struct NoPanicStrategy {
    pub crate_name: Symbol,
    pub strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag(metadata_profiler_builtins_needs_core)]
pub struct ProfilerBuiltinsNeedsCore;

#[derive(Diagnostic)]
#[diag(metadata_not_profiler_runtime)]
pub struct NotProfilerRuntime {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_no_multiple_global_alloc)]
pub struct NoMultipleGlobalAlloc {
    #[primary_span]
    #[label]
    pub span2: Span,
    #[label(metadata_prev_global_alloc)]
    pub span1: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_no_multiple_alloc_error_handler)]
pub struct NoMultipleAllocErrorHandler {
    #[primary_span]
    #[label]
    pub span2: Span,
    #[label(metadata_prev_alloc_error_handler)]
    pub span1: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_conflicting_global_alloc)]
pub struct ConflictingGlobalAlloc {
    pub crate_name: Symbol,
    pub other_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_conflicting_alloc_error_handler)]
pub struct ConflictingAllocErrorHandler {
    pub crate_name: Symbol,
    pub other_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_global_alloc_required)]
pub struct GlobalAllocRequired;

#[derive(Diagnostic)]
#[diag(metadata_no_transitive_needs_dep)]
pub struct NoTransitiveNeedsDep<'a> {
    pub crate_name: Symbol,
    pub needs_crate_name: &'a str,
    pub deps_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_failed_write_error)]
pub struct FailedWriteError {
    pub filename: PathBuf,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_failed_copy_to_stdout)]
pub struct FailedCopyToStdout {
    pub filename: PathBuf,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_binary_output_to_tty)]
pub struct BinaryOutputToTty;

#[derive(Diagnostic)]
#[diag(metadata_missing_native_library)]
pub struct MissingNativeLibrary<'a> {
    libname: &'a str,
    #[subdiagnostic]
    suggest_name: Option<SuggestLibraryName<'a>>,
}

impl<'a> MissingNativeLibrary<'a> {
    pub fn new(libname: &'a str, verbatim: bool) -> Self {
        // if it looks like the user has provided a complete filename rather just the bare lib name,
        // then provide a note that they might want to try trimming the name
        let suggested_name = if !verbatim {
            if let Some(libname) = libname.strip_prefix("lib") && let Some(libname) = libname.strip_suffix(".a") {
                // this is a unix style filename so trim prefix & suffix
                Some(libname)
            } else if let Some(libname) = libname.strip_suffix(".lib") {
                // this is a Windows style filename so just trim the suffix
                Some(libname)
            } else {
                None
            }
        } else {
            None
        };

        Self {
            libname,
            suggest_name: suggested_name
                .map(|suggested_name| SuggestLibraryName { suggested_name }),
        }
    }
}

#[derive(Subdiagnostic)]
#[help(metadata_only_provide_library_name)]
pub struct SuggestLibraryName<'a> {
    suggested_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_failed_create_tempdir)]
pub struct FailedCreateTempdir {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_failed_create_file)]
pub struct FailedCreateFile<'a> {
    pub filename: &'a Path,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_failed_create_encoded_metadata)]
pub struct FailedCreateEncodedMetadata {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(metadata_non_ascii_name)]
pub struct NonAsciiName {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_extern_location_not_exist)]
pub struct ExternLocationNotExist<'a> {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub location: &'a Path,
}

#[derive(Diagnostic)]
#[diag(metadata_extern_location_not_file)]
pub struct ExternLocationNotFile<'a> {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub location: &'a Path,
}

pub(crate) struct MultipleCandidates {
    pub span: Span,
    pub flavor: CrateFlavor,
    pub crate_name: Symbol,
    pub candidates: Vec<PathBuf>,
}

impl IntoDiagnostic<'_> for MultipleCandidates {
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::metadata_multiple_candidates);
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("flavor", self.flavor);
        diag.code(error_code!(E0464));
        diag.set_span(self.span);
        for (i, candidate) in self.candidates.iter().enumerate() {
            diag.note(format!("candidate #{}: {}", i + 1, candidate.display()));
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(metadata_symbol_conflicts_current, code = "E0519")]
pub struct SymbolConflictsCurrent {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_stable_crate_id_collision)]
pub struct StableCrateIdCollision {
    #[primary_span]
    pub span: Span,
    pub crate_name0: Symbol,
    pub crate_name1: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_dl_error)]
pub struct DlError {
    #[primary_span]
    pub span: Span,
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(metadata_newer_crate_version, code = "E0460")]
#[note]
#[note(metadata_found_crate_versions)]
pub struct NewerCrateVersion {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(Diagnostic)]
#[diag(metadata_no_crate_with_triple, code = "E0461")]
#[note(metadata_found_crate_versions)]
pub struct NoCrateWithTriple<'a> {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub locator_triple: &'a str,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(Diagnostic)]
#[diag(metadata_found_staticlib, code = "E0462")]
#[note(metadata_found_crate_versions)]
#[help]
pub struct FoundStaticlib {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(Diagnostic)]
#[diag(metadata_incompatible_rustc, code = "E0514")]
#[note(metadata_found_crate_versions)]
#[help]
pub struct IncompatibleRustc {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub found_crates: String,
    pub rustc_version: String,
}

pub struct InvalidMetadataFiles {
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub crate_rejections: Vec<String>,
}

impl IntoDiagnostic<'_> for InvalidMetadataFiles {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::metadata_invalid_meta_files);
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("add_info", self.add_info);
        diag.code(error_code!(E0786));
        diag.set_span(self.span);
        for crate_rejection in self.crate_rejections {
            diag.note(crate_rejection);
        }
        diag
    }
}

pub struct CannotFindCrate {
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub missing_core: bool,
    pub current_crate: String,
    pub is_nightly_build: bool,
    pub profiler_runtime: Symbol,
    pub locator_triple: TargetTriple,
}

impl IntoDiagnostic<'_> for CannotFindCrate {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::metadata_cannot_find_crate);
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("current_crate", self.current_crate);
        diag.set_arg("add_info", self.add_info);
        diag.set_arg("locator_triple", self.locator_triple.triple());
        diag.code(error_code!(E0463));
        diag.set_span(self.span);
        if (self.crate_name == sym::std || self.crate_name == sym::core)
            && self.locator_triple != TargetTriple::from_triple(config::host_triple())
        {
            if self.missing_core {
                diag.note(fluent::metadata_target_not_installed);
            } else {
                diag.note(fluent::metadata_target_no_std_support);
            }
            // NOTE: this suggests using rustup, even though the user may not have it installed.
            // That's because they could choose to install it; or this may give them a hint which
            // target they need to install from their distro.
            if self.missing_core {
                diag.help(fluent::metadata_consider_downloading_target);
            }
            // Suggest using #![no_std]. #[no_core] is unstable and not really supported anyway.
            // NOTE: this is a dummy span if `extern crate std` was injected by the compiler.
            // If it's not a dummy, that means someone added `extern crate std` explicitly and
            // `#![no_std]` won't help.
            if !self.missing_core && self.span.is_dummy() {
                diag.note(fluent::metadata_std_required);
            }
            if self.is_nightly_build {
                diag.help(fluent::metadata_consider_building_std);
            }
        } else if self.crate_name == self.profiler_runtime {
            diag.note(fluent::metadata_compiler_missing_profiler);
        } else if self.crate_name.as_str().starts_with("rustc_") {
            diag.help(fluent::metadata_install_missing_components);
        }
        diag.span_label(self.span, fluent::metadata_cant_find_crate);
        diag
    }
}

#[derive(Diagnostic)]
#[diag(metadata_no_dylib_plugin, code = "E0457")]
pub struct NoDylibPlugin {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_crate_location_unknown_type)]
pub struct CrateLocationUnknownType<'a> {
    #[primary_span]
    pub span: Span,
    pub path: &'a Path,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(metadata_lib_filename_form)]
pub struct LibFilenameForm<'a> {
    #[primary_span]
    pub span: Span,
    pub dll_prefix: &'a str,
    pub dll_suffix: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_multiple_import_name_type)]
pub struct MultipleImportNameType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_import_name_type_form)]
pub struct ImportNameTypeForm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_import_name_type_x86)]
pub struct ImportNameTypeX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(metadata_unknown_import_name_type)]
pub struct UnknownImportNameType<'a> {
    #[primary_span]
    pub span: Span,
    pub import_name_type: &'a str,
}

#[derive(Diagnostic)]
#[diag(metadata_import_name_type_raw)]
pub struct ImportNameTypeRaw {
    #[primary_span]
    pub span: Span,
}
