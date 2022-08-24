use std::path::PathBuf;

use rustc_errors::{DiagnosticId, ErrorGuaranteed};
use rustc_macros::SessionDiagnostic;
use rustc_session::{config, SessionDiagnostic};
use rustc_span::{sym, Span, Symbol};
use rustc_target::spec::TargetTriple;

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
    #[primary_span]
    #[label]
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
    #[primary_span]
    #[label]
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
    #[primary_span]
    #[label]
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

#[derive(SessionDiagnostic)]
#[diag(metadata::failed_write_error)]
pub struct FailedWriteError {
    pub filename: String,
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::failed_create_tempdir)]
pub struct FailedCreateTempdir {
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::failed_create_file)]
pub struct FailedCreateFile {
    pub filename: String,
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::failed_create_encoded_metadata)]
pub struct FailedCreateEncodedMetadata {
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::non_ascii_name)]
pub struct NonAsciiName {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::extern_location_not_exist)]
pub struct ExternLocationNotExist {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub location: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::extern_location_not_file)]
pub struct ExternLocationNotFile {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub location: String,
}

pub struct MultipleCandidates {
    pub span: Span,
    pub flavor: String,
    pub crate_name: String,
    pub candidates: Vec<PathBuf>,
}

impl SessionDiagnostic<'_> for MultipleCandidates {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(rustc_errors::fluent::metadata::multiple_candidates);
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("flavor", self.flavor);
        diag.code(DiagnosticId::Error("E0465".into()));
        diag.set_span(self.span);
        for (i, candidate) in self.candidates.iter().enumerate() {
            diag.span_note(self.span, &format!("candidate #{}: {}", i + 1, candidate.display()));
        }
        diag
    }
}

#[derive(SessionDiagnostic)]
#[diag(metadata::multiple_matching_crates, code = "E0464")]
#[note]
pub struct MultipleMatchingCrates {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub candidates: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::symbol_conflicts_current, code = "E0519")]
pub struct SymbolConflictsCurrent {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::symbol_conflicts_others, code = "E0523")]
pub struct SymbolConflictsOthers {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::stable_crate_id_collision)]
pub struct StableCrateIdCollision {
    #[primary_span]
    pub span: Span,
    pub crate_name0: String,
    pub crate_name1: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::dl_error)]
pub struct DlError {
    #[primary_span]
    pub span: Span,
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::newer_crate_version, code = "E0460")]
#[note]
#[note(metadata::found_crate_versions)]
pub struct NewerCrateVersion {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::no_crate_with_triple, code = "E0461")]
#[note(metadata::found_crate_versions)]
pub struct NoCrateWithTriple {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub locator_triple: String,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::found_staticlib, code = "E0462")]
#[note(metadata::found_crate_versions)]
#[help]
pub struct FoundStaticlib {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::incompatible_rustc, code = "E0514")]
#[note(metadata::found_crate_versions)]
#[help]
pub struct IncompatibleRustc {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
    pub add_info: String,
    pub found_crates: String,
    pub rustc_version: String,
}

pub struct InvalidMetadataFiles {
    pub span: Span,
    pub crate_name: String,
    pub add_info: String,
    pub crate_rejections: Vec<String>,
}

impl SessionDiagnostic<'_> for InvalidMetadataFiles {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(rustc_errors::fluent::metadata::invalid_meta_files);
        diag.set_arg("crate_name", self.crate_name);
        diag.set_arg("add_info", self.add_info);
        diag.code(DiagnosticId::Error("E0786".into()));
        diag.set_span(self.span);
        for crate_rejection in self.crate_rejections {
            diag.note(crate_rejection);
        }
        diag
    }
}

pub struct CannotFindCrate {
    pub span: Span,
    pub crate_name: String,
    pub crate_name_symbol: Symbol,
    pub add_info: String,
    pub missing_core: bool,
    pub current_crate: String,
    pub is_nightly_build: bool,
    pub profiler_runtime: Symbol,
    pub locator_triple: TargetTriple,
}

impl SessionDiagnostic<'_> for CannotFindCrate {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(rustc_errors::fluent::metadata::cannot_find_crate);
        diag.set_arg("crate_name", self.crate_name.clone());
        diag.set_arg("add_info", self.add_info);
        diag.code(DiagnosticId::Error("E0463".into()));
        diag.set_span(self.span);
        // FIXME: Find a way to distill this logic down into the derived SessionDiagnostic form
        if (self.crate_name_symbol == sym::std || self.crate_name_symbol == sym::core)
            && self.locator_triple != TargetTriple::from_triple(config::host_triple())
        {
            if self.missing_core {
                diag.note(&format!("the `{}` target may not be installed", self.locator_triple));
            } else {
                diag.note(&format!(
                    "the `{}` target may not support the standard library",
                    self.locator_triple
                ));
            }
            // NOTE: this suggests using rustup, even though the user may not have it installed.
            // That's because they could choose to install it; or this may give them a hint which
            // target they need to install from their distro.
            if self.missing_core {
                diag.help(&format!(
                    "consider downloading the target with `rustup target add {}`",
                    self.locator_triple
                ));
            }
            // Suggest using #![no_std]. #[no_core] is unstable and not really supported anyway.
            // NOTE: this is a dummy span if `extern crate std` was injected by the compiler.
            // If it's not a dummy, that means someone added `extern crate std` explicitly and
            // `#![no_std]` won't help.
            if !self.missing_core && self.span.is_dummy() {
                diag.note(&format!(
                    "`std` is required by `{}` because it does not declare `#![no_std]`",
                    self.current_crate
                ));
            }
            if self.is_nightly_build {
                diag.help("consider building the standard library from source with `cargo build -Zbuild-std`");
            }
        } else if self.crate_name_symbol == self.profiler_runtime {
            diag.note("the compiler may have been built without the profiler runtime");
        } else if self.crate_name.starts_with("rustc_") {
            diag.help(
                "maybe you need to install the missing components with: \
                             `rustup component add rust-src rustc-dev llvm-tools-preview`",
            );
        }
        diag.span_label(self.span, "can't find crate");
        diag
    }
}

#[derive(SessionDiagnostic)]
#[diag(metadata::no_dylib_plugin, code = "E0457")]
pub struct NoDylibPlugin {
    #[primary_span]
    pub span: Span,
    pub crate_name: String,
}
