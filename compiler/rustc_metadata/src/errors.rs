use std::io::Error;
use std::path::{Path, PathBuf};

use rustc_errors::codes::*;
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level, inline_fluent};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol, sym};
use rustc_target::spec::{PanicStrategy, TargetTuple};

use crate::locator::CrateFlavor;

#[derive(Diagnostic)]
#[diag(
    "crate `{$crate_name}` required to be available in rlib format, but was not found in this form"
)]
pub struct RlibRequired {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form"
)]
pub struct LibRequired<'a> {
    pub crate_name: Symbol,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(
    "crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form"
)]
#[help("try adding `extern crate rustc_driver;` at the top level of this crate")]
pub struct RustcLibRequired<'a> {
    pub crate_name: Symbol,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag("cannot satisfy dependencies so `{$crate_name}` only shows up once")]
#[help("having upstream crates all available in one format will likely make this go away")]
pub struct CrateDepMultiple {
    pub crate_name: Symbol,
    #[subdiagnostic]
    pub non_static_deps: Vec<NonStaticCrateDep>,
    #[help("`feature(rustc_private)` is needed to link to the compiler's `rustc_driver` library")]
    pub rustc_driver_help: bool,
}

#[derive(Subdiagnostic)]
#[note("`{$crate_name}` was unavailable as a static crate, preventing fully static linking")]
pub struct NonStaticCrateDep {
    /// It's different from `crate_name` in main Diagnostic.
    pub crate_name_: Symbol,
}

#[derive(Diagnostic)]
#[diag("cannot link together two panic runtimes: {$prev_name} and {$cur_name}")]
pub struct TwoPanicRuntimes {
    pub prev_name: Symbol,
    pub cur_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "the linked panic runtime `{$runtime}` is not compiled with this crate's panic strategy `{$strategy}`"
)]
pub struct BadPanicStrategy {
    pub runtime: Symbol,
    pub strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag(
    "the crate `{$crate_name}` requires panic strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`"
)]
pub struct RequiredPanicStrategy {
    pub crate_name: Symbol,
    pub found_strategy: PanicStrategy,
    pub desired_strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag(
    "the crate `{$crate_name}` was compiled with a panic strategy which is incompatible with `immediate-abort`"
)]
pub struct IncompatibleWithImmediateAbort {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "the crate `core` was compiled with a panic strategy which is incompatible with `immediate-abort`"
)]
pub struct IncompatibleWithImmediateAbortCore;

#[derive(Diagnostic)]
#[diag(
    "the crate `{$crate_name}` is compiled with the panic-in-drop strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`"
)]
pub struct IncompatiblePanicInDropStrategy {
    pub crate_name: Symbol,
    pub found_strategy: PanicStrategy,
    pub desired_strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag("`#[link_ordinal]` is only supported if link kind is `raw-dylib`")]
pub struct LinkOrdinalRawDylib {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("library kind `framework` is only supported on Apple targets")]
pub struct LibFrameworkApple;

#[derive(Diagnostic)]
#[diag("an empty renaming target was specified for library `{$lib_name}`")]
pub struct EmptyRenamingTarget<'a> {
    pub lib_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(
    "renaming of the library `{$lib_name}` was specified, however this crate contains no `#[link(...)]` attributes referencing this library"
)]
pub struct RenamingNoLink<'a> {
    pub lib_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("multiple renamings were specified for library `{$lib_name}`")]
pub struct MultipleRenamings<'a> {
    pub lib_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("overriding linking modifiers from command line is not supported")]
pub struct NoLinkModOverride {
    #[primary_span]
    pub span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("ABI not supported by `#[link(kind = \"raw-dylib\")]` on this architecture")]
pub struct RawDylibUnsupportedAbi {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("failed to create file encoder: {$err}")]
pub struct FailCreateFileEncoder {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag("failed to write to `{$path}`: {$err}")]
pub struct FailWriteFile<'a> {
    pub path: &'a Path,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag("the crate `{$crate_name}` is not a panic runtime")]
pub struct CrateNotPanicRuntime {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "the crate `{$crate_name}` resolved as `compiler_builtins` but is not `#![compiler_builtins]`"
)]
pub struct CrateNotCompilerBuiltins {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("the crate `{$crate_name}` does not have the panic strategy `{$strategy}`")]
pub struct NoPanicStrategy {
    pub crate_name: Symbol,
    pub strategy: PanicStrategy,
}

#[derive(Diagnostic)]
#[diag("the crate `{$crate_name}` is not a profiler runtime")]
pub struct NotProfilerRuntime {
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("cannot define multiple global allocators")]
pub struct NoMultipleGlobalAlloc {
    #[primary_span]
    #[label("cannot define a new global allocator")]
    pub span2: Span,
    #[label("previous global allocator defined here")]
    pub span1: Span,
}

#[derive(Diagnostic)]
#[diag("cannot define multiple allocation error handlers")]
pub struct NoMultipleAllocErrorHandler {
    #[primary_span]
    #[label("cannot define a new allocation error handler")]
    pub span2: Span,
    #[label("previous allocation error handler defined here")]
    pub span1: Span,
}

#[derive(Diagnostic)]
#[diag(
    "the `#[global_allocator]` in {$other_crate_name} conflicts with global allocator in: {$crate_name}"
)]
pub struct ConflictingGlobalAlloc {
    pub crate_name: Symbol,
    pub other_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "the `#[alloc_error_handler]` in {$other_crate_name} conflicts with allocation error handler in: {$crate_name}"
)]
pub struct ConflictingAllocErrorHandler {
    pub crate_name: Symbol,
    pub other_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "no global memory allocator found but one is required; link to std or add `#[global_allocator]` to a static item that implements the GlobalAlloc trait"
)]
pub struct GlobalAllocRequired;

#[derive(Diagnostic)]
#[diag(
    "the crate `{$crate_name}` cannot depend on a crate that needs {$needs_crate_name}, but it depends on `{$deps_crate_name}`"
)]
pub struct NoTransitiveNeedsDep<'a> {
    pub crate_name: Symbol,
    pub needs_crate_name: &'a str,
    pub deps_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("failed to write {$filename}: {$err}")]
pub struct FailedWriteError {
    pub filename: PathBuf,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag("failed to copy {$filename} to stdout: {$err}")]
pub struct FailedCopyToStdout {
    pub filename: PathBuf,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag(
    "option `-o` or `--emit` is used to write binary output type `metadata` to stdout, but stdout is a tty"
)]
pub struct BinaryOutputToTty;

#[derive(Diagnostic)]
#[diag("could not find native static library `{$libname}`, perhaps an -L flag is missing?")]
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
            if let Some(libname) = libname.strip_circumfix("lib", ".a") {
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
#[help("only provide the library name `{$suggested_name}`, not the full filename")]
pub struct SuggestLibraryName<'a> {
    suggested_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("couldn't create a temp dir: {$err}")]
pub struct FailedCreateTempdir {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag("failed to create the file {$filename}: {$err}")]
pub struct FailedCreateFile<'a> {
    pub filename: &'a Path,
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag("failed to create encoded metadata from file: {$err}")]
pub struct FailedCreateEncodedMetadata {
    pub err: Error,
}

#[derive(Diagnostic)]
#[diag("cannot load a crate with a non-ascii name `{$crate_name}`")]
pub struct NonAsciiName {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("extern location for {$crate_name} does not exist: {$location}")]
pub struct ExternLocationNotExist<'a> {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub location: &'a Path,
}

#[derive(Diagnostic)]
#[diag("extern location for {$crate_name} is not a file: {$location}")]
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

impl<G: EmissionGuarantee> Diagnostic<'_, G> for MultipleCandidates {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            inline_fluent!("multiple candidates for `{$flavor}` dependency `{$crate_name}` found"),
        );
        diag.arg("crate_name", self.crate_name);
        diag.arg("flavor", self.flavor);
        diag.code(E0464);
        diag.span(self.span);
        for (i, candidate) in self.candidates.iter().enumerate() {
            diag.note(format!("candidate #{}: {}", i + 1, candidate.display()));
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag(
    "only metadata stub found for `{$flavor}` dependency `{$crate_name}` please provide path to the corresponding .rmeta file with full metadata"
)]
pub(crate) struct FullMetadataNotFound {
    #[primary_span]
    pub span: Span,
    pub flavor: CrateFlavor,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("the current crate is indistinguishable from one of its dependencies: it has the same crate-name `{$crate_name}` and was compiled with the same `-C metadata` arguments, so this will result in symbol conflicts between the two", code = E0519)]
pub struct SymbolConflictsCurrent {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("found crates (`{$crate_name0}` and `{$crate_name1}`) with colliding StableCrateId values")]
pub struct StableCrateIdCollision {
    #[primary_span]
    pub span: Span,
    pub crate_name0: Symbol,
    pub crate_name1: Symbol,
}

#[derive(Diagnostic)]
#[diag("{$path}{$err}")]
pub struct DlError {
    #[primary_span]
    pub span: Span,
    pub path: String,
    pub err: String,
}

#[derive(Diagnostic)]
#[diag("found possibly newer version of crate `{$crate_name}`{$add_info}", code = E0460)]
#[note("perhaps that crate needs to be recompiled?")]
#[note("the following crate versions were found:{$found_crates}")]
pub struct NewerCrateVersion {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(Diagnostic)]
#[diag("couldn't find crate `{$crate_name}` with expected target triple {$locator_triple}{$add_info}", code = E0461)]
#[note("the following crate versions were found:{$found_crates}")]
pub struct NoCrateWithTriple<'a> {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub locator_triple: &'a str,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(Diagnostic)]
#[diag("found staticlib `{$crate_name}` instead of rlib or dylib{$add_info}", code = E0462)]
#[note("the following crate versions were found:{$found_crates}")]
#[help("please recompile that crate using --crate-type lib")]
pub struct FoundStaticlib {
    #[primary_span]
    pub span: Span,
    pub crate_name: Symbol,
    pub add_info: String,
    pub found_crates: String,
}

#[derive(Diagnostic)]
#[diag("found crate `{$crate_name}` compiled by an incompatible version of rustc{$add_info}", code = E0514)]
#[note("the following crate versions were found:{$found_crates}")]
#[help(
    "please recompile that crate using this compiler ({$rustc_version}) (consider running `cargo clean` first)"
)]
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

impl<G: EmissionGuarantee> Diagnostic<'_, G> for InvalidMetadataFiles {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            inline_fluent!("found invalid metadata files for crate `{$crate_name}`{$add_info}"),
        );
        diag.arg("crate_name", self.crate_name);
        diag.arg("add_info", self.add_info);
        diag.code(E0786);
        diag.span(self.span);
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
    pub locator_triple: TargetTuple,
    pub is_ui_testing: bool,
    pub is_tier_3: bool,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for CannotFindCrate {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            inline_fluent!("can't find crate for `{$crate_name}`{$add_info}"),
        );
        diag.arg("crate_name", self.crate_name);
        diag.arg("current_crate", self.current_crate);
        diag.arg("add_info", self.add_info);
        diag.arg("locator_triple", self.locator_triple.tuple());
        diag.code(E0463);
        diag.span(self.span);
        if self.crate_name == sym::std || self.crate_name == sym::core {
            if self.missing_core {
                diag.note(inline_fluent!("the `{$locator_triple}` target may not be installed"));
            } else {
                diag.note(inline_fluent!(
                    "the `{$locator_triple}` target may not support the standard library"
                ));
            }

            let has_precompiled_std = !self.is_tier_3;

            if self.missing_core {
                if env!("CFG_RELEASE_CHANNEL") == "dev" && !self.is_ui_testing {
                    // Note: Emits the nicer suggestion only for the dev channel.
                    diag.help(inline_fluent!("consider adding the standard library to the sysroot with `x build library --target {$locator_triple}`"));
                } else if has_precompiled_std {
                    // NOTE: this suggests using rustup, even though the user may not have it installed.
                    // That's because they could choose to install it; or this may give them a hint which
                    // target they need to install from their distro.
                    diag.help(inline_fluent!(
                        "consider downloading the target with `rustup target add {$locator_triple}`"
                    ));
                }
            }

            // Suggest using #![no_std]. #[no_core] is unstable and not really supported anyway.
            // NOTE: this is a dummy span if `extern crate std` was injected by the compiler.
            // If it's not a dummy, that means someone added `extern crate std` explicitly and
            // `#![no_std]` won't help.
            if !self.missing_core && self.span.is_dummy() {
                diag.note(inline_fluent!("`std` is required by `{$current_crate}` because it does not declare `#![no_std]`"));
            }
            // Recommend -Zbuild-std even on stable builds for Tier 3 targets because
            // it's the recommended way to use the target, the user should switch to nightly.
            if self.is_nightly_build || !has_precompiled_std {
                diag.help(inline_fluent!("consider building the standard library from source with `cargo build -Zbuild-std`"));
            }
        } else if self.crate_name == self.profiler_runtime {
            diag.note(inline_fluent!(
                "the compiler may have been built without the profiler runtime"
            ));
        } else if self.crate_name.as_str().starts_with("rustc_") {
            diag.help(inline_fluent!("maybe you need to install the missing components with: `rustup component add rust-src rustc-dev llvm-tools-preview`"));
        }
        diag.span_label(self.span, inline_fluent!("can't find crate"));
        diag
    }
}

#[derive(Diagnostic)]
#[diag("extern location for {$crate_name} is of an unknown type: {$path}")]
pub struct CrateLocationUnknownType<'a> {
    #[primary_span]
    pub span: Span,
    pub path: &'a Path,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("file name should be lib*.rlib or {$dll_prefix}*{$dll_suffix}")]
pub struct LibFilenameForm<'a> {
    #[primary_span]
    pub span: Span,
    pub dll_prefix: &'a str,
    pub dll_suffix: &'a str,
}

#[derive(Diagnostic)]
#[diag(
    "older versions of the `wasm-bindgen` crate are incompatible with current versions of Rust; please update to `wasm-bindgen` v0.2.88"
)]
pub(crate) struct WasmCAbi {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`")]
#[help(
    "the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely"
)]
#[note(
    "`{$flag_name_prefixed}={$local_value}` in this crate is incompatible with `{$flag_name_prefixed}={$extern_value}` in dependency `{$extern_crate}`"
)]
#[help(
    "set `{$flag_name_prefixed}={$extern_value}` in this crate or `{$flag_name_prefixed}={$local_value}` in `{$extern_crate}`"
)]
#[help(
    "if you are sure this will not cause problems, you may use `-Cunsafe-allow-abi-mismatch={$flag_name}` to silence this error"
)]
pub struct IncompatibleTargetModifiers {
    #[primary_span]
    pub span: Span,
    pub extern_crate: Symbol,
    pub local_crate: Symbol,
    pub flag_name: String,
    pub flag_name_prefixed: String,
    pub local_value: String,
    pub extern_value: String,
}

#[derive(Diagnostic)]
#[diag("mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`")]
#[help(
    "the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely"
)]
#[note(
    "unset `{$flag_name_prefixed}` in this crate is incompatible with `{$flag_name_prefixed}={$extern_value}` in dependency `{$extern_crate}`"
)]
#[help(
    "set `{$flag_name_prefixed}={$extern_value}` in this crate or unset `{$flag_name_prefixed}` in `{$extern_crate}`"
)]
#[help(
    "if you are sure this will not cause problems, you may use `-Cunsafe-allow-abi-mismatch={$flag_name}` to silence this error"
)]
pub struct IncompatibleTargetModifiersLMissed {
    #[primary_span]
    pub span: Span,
    pub extern_crate: Symbol,
    pub local_crate: Symbol,
    pub flag_name: String,
    pub flag_name_prefixed: String,
    pub extern_value: String,
}

#[derive(Diagnostic)]
#[diag("mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`")]
#[help(
    "the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely"
)]
#[note(
    "`{$flag_name_prefixed}={$local_value}` in this crate is incompatible with unset `{$flag_name_prefixed}` in dependency `{$extern_crate}`"
)]
#[help(
    "unset `{$flag_name_prefixed}` in this crate or set `{$flag_name_prefixed}={$local_value}` in `{$extern_crate}`"
)]
#[help(
    "if you are sure this will not cause problems, you may use `-Cunsafe-allow-abi-mismatch={$flag_name}` to silence this error"
)]
pub struct IncompatibleTargetModifiersRMissed {
    #[primary_span]
    pub span: Span,
    pub extern_crate: Symbol,
    pub local_crate: Symbol,
    pub flag_name: String,
    pub flag_name_prefixed: String,
    pub local_value: String,
}

#[derive(Diagnostic)]
#[diag(
    "unknown target modifier `{$flag_name}`, requested by `-Cunsafe-allow-abi-mismatch={$flag_name}`"
)]
pub struct UnknownTargetModifierUnsafeAllowed {
    #[primary_span]
    pub span: Span,
    pub flag_name: String,
}

#[derive(Diagnostic)]
#[diag(
    "found async drop types in dependency `{$extern_crate}`, but async_drop feature is disabled for `{$local_crate}`"
)]
#[help(
    "if async drop type will be dropped in a crate without `feature(async_drop)`, sync Drop will be used"
)]
pub struct AsyncDropTypesInDependency {
    #[primary_span]
    pub span: Span,
    pub extern_crate: Symbol,
    pub local_crate: Symbol,
}

#[derive(Diagnostic)]
#[diag("link name must be well-formed if link kind is `raw-dylib`")]
pub struct RawDylibMalformed {
    #[primary_span]
    pub span: Span,
}
