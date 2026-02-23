use std::io;
use std::path::Path;

use rustc_hir::attrs::CrateType;
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};
use rustc_target::spec::TargetTuple;

#[derive(Diagnostic)]
#[diag(
    "`--crate-name` and `#[crate_name]` are required to match, but `{$crate_name}` != `{$attr_crate_name}`"
)]
pub(crate) struct CrateNameDoesNotMatch {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) crate_name: Symbol,
    pub(crate) attr_crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("crate names cannot start with a `-`, but `{$crate_name}` has a leading hyphen")]
pub(crate) struct CrateNameInvalid<'a> {
    pub(crate) crate_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("Ferris cannot be used as an identifier")]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(
        "try using their name instead",
        code = "{ferris_fix}",
        applicability = "maybe-incorrect"
    )]
    pub first_span: Span,
    pub ferris_fix: &'static str,
}

#[derive(Diagnostic)]
#[diag("identifiers cannot contain emoji: `{$ident}`")]
pub struct EmojiIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag("cannot mix `bin` crate type with others")]
pub struct MixedBinCrate;

#[derive(Diagnostic)]
#[diag("cannot mix `proc-macro` crate type with others")]
pub struct MixedProcMacroCrate;

#[derive(Diagnostic)]
#[diag("error writing dependencies to `{$path}`: {$error}")]
pub struct ErrorWritingDependencies<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag("the input file \"{$path}\" would be overwritten by the generated executable")]
pub struct InputFileWouldBeOverWritten<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(
    "the generated executable for the input file \"{$input_path}\" conflicts with the existing directory \"{$dir_path}\""
)]
pub struct GeneratedFileConflictsWithDirectory<'a> {
    pub input_path: &'a Path,
    pub dir_path: &'a Path,
}

#[derive(Diagnostic)]
#[diag("failed to find or create the directory specified by `--temps-dir`")]
pub struct TempsDirError;

#[derive(Diagnostic)]
#[diag("failed to find or create the directory specified by `--out-dir`")]
pub struct OutDirError;

#[derive(Diagnostic)]
#[diag("failed to write file {$path}: {$error}\"")]
pub struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "building proc macro crate with `panic=abort` or `panic=immediate-abort` may crash the compiler should the proc-macro panic"
)]
pub struct ProcMacroCratePanicAbort;

#[derive(Diagnostic)]
#[diag(
    "due to multiple output types requested, the explicitly specified output file name will be adapted for each output type"
)]
pub struct MultipleOutputTypesAdaption;

#[derive(Diagnostic)]
#[diag("ignoring -C extra-filename flag due to -o flag")]
pub struct IgnoringExtraFilename;

#[derive(Diagnostic)]
#[diag("ignoring --out-dir flag due to -o flag")]
pub struct IgnoringOutDir;

#[derive(Diagnostic)]
#[diag("can't use option `-o` or `--emit` to write multiple output types to stdout")]
pub struct MultipleOutputTypesToStdout;

#[derive(Diagnostic)]
#[diag(
    "target feature `{$feature}` must be {$enabled} to ensure that the ABI of the current target can be implemented correctly"
)]
#[note(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
#[note("for more information, see issue #116344 <https://github.com/rust-lang/rust/issues/116344>")]
pub(crate) struct AbiRequiredTargetFeature<'a> {
    pub feature: &'a str,
    pub enabled: &'a str,
}

#[derive(Diagnostic)]
#[diag("dropping unsupported crate type `{$crate_type}` for codegen backend `{$codegen_backend}`")]
pub(crate) struct UnsupportedCrateTypeForCodegenBackend {
    pub(crate) crate_type: CrateType,
    pub(crate) codegen_backend: &'static str,
}

#[derive(Diagnostic)]
#[diag("dropping unsupported crate type `{$crate_type}` for target `{$target_triple}`")]
pub(crate) struct UnsupportedCrateTypeForTarget<'a> {
    pub(crate) crate_type: CrateType,
    pub(crate) target_triple: &'a TargetTuple,
}
