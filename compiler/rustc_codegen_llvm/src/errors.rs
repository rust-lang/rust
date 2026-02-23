use std::ffi::CString;
use std::path::Path;

use rustc_data_structures::small_c_str::SmallCStr;
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level, msg};
use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag("symbol `{$symbol_name}` is already defined")]
pub(crate) struct SymbolAlreadyDefined<'a> {
    #[primary_span]
    pub span: Span,
    pub symbol_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("`-Zsanitizer=memtag` requires `-Ctarget-feature=+mte`")]
pub(crate) struct SanitizerMemtagRequiresMte;

pub(crate) struct ParseTargetMachineConfig<'a>(pub LlvmError<'a>);

impl<G: EmissionGuarantee> Diagnostic<'_, G> for ParseTargetMachineConfig<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let diag: Diag<'_, G> = self.0.into_diag(dcx, level);
        let (message, _) = diag.messages.first().expect("`LlvmError` with no message");
        let message = dcx.eagerly_translate_to_string(message.clone(), diag.args.iter());
        Diag::new(
            dcx,
            level,
            msg!("failed to parse target machine config to target machine: {$error}"),
        )
        .with_arg("error", message)
    }
}

#[derive(Diagnostic)]
#[diag("failed to load our autodiff backend: {$err}")]
pub(crate) struct AutoDiffComponentUnavailable {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag("autodiff backend not found in the sysroot: {$err}")]
#[note("it will be distributed via rustup in the future")]
pub(crate) struct AutoDiffComponentMissing {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag("using the autodiff feature requires setting `lto=\"fat\"` in your Cargo.toml")]
pub(crate) struct AutoDiffWithoutLto;

#[derive(Diagnostic)]
#[diag("using the autodiff feature requires -Z autodiff=Enable")]
pub(crate) struct AutoDiffWithoutEnable;

#[derive(Diagnostic)]
#[diag("using the offload feature requires -Z offload=<Device or Host=/absolute/path/to/host.out>")]
pub(crate) struct OffloadWithoutEnable;

#[derive(Diagnostic)]
#[diag("using the offload feature requires -C lto=fat")]
pub(crate) struct OffloadWithoutFatLTO;

#[derive(Diagnostic)]
#[diag("using the `-Z offload=Host=/absolute/path/to/host.out` flag requires an absolute path")]
pub(crate) struct OffloadWithoutAbsPath;

#[derive(Diagnostic)]
#[diag(
    "using the `-Z offload=Host=/absolute/path/to/host.out` flag must point to a `host.out` file"
)]
pub(crate) struct OffloadWrongFileName;

#[derive(Diagnostic)]
#[diag(
    "the given path/file to `host.out` does not exist. Did you forget to run the device compilation first?"
)]
pub(crate) struct OffloadNonexistingPath;

#[derive(Diagnostic)]
#[diag("call to BundleImages failed, `host.out` was not created")]
pub(crate) struct OffloadBundleImagesFailed;

#[derive(Diagnostic)]
#[diag("call to EmbedBufferInModule failed, `host.o` was not created")]
pub(crate) struct OffloadEmbedFailed;

#[derive(Diagnostic)]
#[diag("failed to get bitcode from object file for LTO ({$err})")]
pub(crate) struct LtoBitcodeFromRlib {
    pub err: String,
}

#[derive(Diagnostic)]
pub(crate) enum LlvmError<'a> {
    #[diag("could not write output to {$path}")]
    WriteOutput { path: &'a Path },
    #[diag("could not create LLVM TargetMachine for triple: {$triple}")]
    CreateTargetMachine { triple: SmallCStr },
    #[diag("failed to run LLVM passes")]
    RunLlvmPasses,
    #[diag("failed to write LLVM IR to {$path}")]
    WriteIr { path: &'a Path },
    #[diag("failed to prepare thin LTO context")]
    PrepareThinLtoContext,
    #[diag("failed to load bitcode of module \"{$name}\"")]
    LoadBitcode { name: CString },
    #[diag("error while writing ThinLTO key data: {$err}")]
    WriteThinLtoKey { err: std::io::Error },
    #[diag("failed to prepare thin LTO module")]
    PrepareThinLtoModule,
    #[diag("failed to parse bitcode for LTO module")]
    ParseBitcode,
}

pub(crate) struct WithLlvmError<'a>(pub LlvmError<'a>, pub String);

impl<G: EmissionGuarantee> Diagnostic<'_, G> for WithLlvmError<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        use LlvmError::*;
        let msg_with_llvm_err = match &self.0 {
            WriteOutput { .. } => msg!("could not write output to {$path}: {$llvm_err}"),
            CreateTargetMachine { .. } => {
                msg!("could not create LLVM TargetMachine for triple: {$triple}: {$llvm_err}")
            }
            RunLlvmPasses => msg!("failed to run LLVM passes: {$llvm_err}"),
            WriteIr { .. } => msg!("failed to write LLVM IR to {$path}: {$llvm_err}"),
            PrepareThinLtoContext => {
                msg!("failed to prepare thin LTO context: {$llvm_err}")
            }
            LoadBitcode { .. } => {
                msg!("failed to load bitcode of module \"{$name}\": {$llvm_err}")
            }
            WriteThinLtoKey { .. } => {
                msg!("error while writing ThinLTO key data: {$err}: {$llvm_err}")
            }
            PrepareThinLtoModule => {
                msg!("failed to prepare thin LTO module: {$llvm_err}")
            }
            ParseBitcode => msg!("failed to parse bitcode for LTO module: {$llvm_err}"),
        };
        self.0
            .into_diag(dcx, level)
            .with_primary_message(msg_with_llvm_err)
            .with_arg("llvm_err", self.1)
    }
}

#[derive(Diagnostic)]
#[diag("{$filename}:{$line}:{$column} {$pass_name} ({$kind}): {$message}")]
pub(crate) struct FromLlvmOptimizationDiag<'a> {
    pub filename: &'a str,
    pub line: std::ffi::c_uint,
    pub column: std::ffi::c_uint,
    pub pass_name: &'a str,
    pub kind: &'a str,
    pub message: &'a str,
}

#[derive(Diagnostic)]
#[diag("{$message}")]
pub(crate) struct FromLlvmDiag {
    pub message: String,
}

#[derive(Diagnostic)]
#[diag("failed to write bytecode to {$path}: {$err}")]
pub(crate) struct WriteBytecode<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to copy bitcode to object file: {$err}")]
pub(crate) struct CopyBitcode {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "unknown debuginfo compression algorithm {$algorithm} - will fall back to uncompressed debuginfo"
)]
pub(crate) struct UnknownCompression {
    pub algorithm: &'static str,
}

#[derive(Diagnostic)]
#[diag(
    "data-layout for target `{$rustc_target}`, `{$rustc_layout}`, differs from LLVM target's `{$llvm_target}` default layout, `{$llvm_layout}`"
)]
pub(crate) struct MismatchedDataLayout<'a> {
    pub rustc_target: &'a str,
    pub rustc_layout: &'a str,
    pub llvm_target: &'a str,
    pub llvm_layout: &'a str,
}

#[derive(Diagnostic)]
#[diag("the `-Zfixed-x18` flag is not supported on the `{$arch}` architecture")]
pub(crate) struct FixedX18InvalidArch<'a> {
    pub arch: &'a str,
}

#[derive(Diagnostic)]
#[diag("`-Zsanitizer-kcfi-arity` requires LLVM 21.0.0 or later")]
pub(crate) struct SanitizerKcfiArityRequiresLLVM2100;
