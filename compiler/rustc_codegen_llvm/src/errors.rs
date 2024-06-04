use std::borrow::Cow;
use std::ffi::CString;
use std::path::Path;

use crate::fluent_generated as fluent;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_errors::{Diag, DiagCtxt, Diagnostic, EmissionGuarantee, Level};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(codegen_llvm_unknown_ctarget_feature_prefix)]
#[note]
pub(crate) struct UnknownCTargetFeaturePrefix<'a> {
    pub(crate) feature: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_unknown_ctarget_feature)]
#[note]
pub(crate) struct UnknownCTargetFeature<'a> {
    pub(crate) feature: &'a str,
    #[subdiagnostic]
    pub(crate) rust_feature: PossibleFeature<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_unstable_ctarget_feature)]
#[note]
pub(crate) struct UnstableCTargetFeature<'a> {
    pub(crate) feature: &'a str,
}

#[derive(Subdiagnostic)]
pub(crate) enum PossibleFeature<'a> {
    #[help(codegen_llvm_possible_feature)]
    Some { rust_feature: &'a str },
    #[help(codegen_llvm_consider_filing_feature_request)]
    None,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_error_creating_import_library)]
pub(crate) struct ErrorCreatingImportLibrary<'a> {
    pub(crate) lib_name: &'a str,
    pub(crate) error: String,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_symbol_already_defined)]
pub(crate) struct SymbolAlreadyDefined<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) symbol_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_invalid_minimum_alignment_not_power_of_two)]
pub(crate) struct InvalidMinimumAlignmentNotPowerOfTwo {
    pub(crate) align: u64,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_invalid_minimum_alignment_too_large)]
pub(crate) struct InvalidMinimumAlignmentTooLarge {
    pub(crate) align: u64,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_sanitizer_memtag_requires_mte)]
pub(crate) struct SanitizerMemtagRequiresMte;

#[derive(Diagnostic)]
#[diag(codegen_llvm_error_writing_def_file)]
pub(crate) struct ErrorWritingDEFFile {
    pub(crate) error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_error_calling_dlltool)]
pub(crate) struct ErrorCallingDllTool<'a> {
    pub(crate) dlltool_path: Cow<'a, str>,
    pub(crate) error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_dlltool_fail_import_library)]
pub(crate) struct DlltoolFailImportLibrary<'a> {
    pub(crate) dlltool_path: Cow<'a, str>,
    pub(crate) dlltool_args: String,
    pub(crate) stdout: Cow<'a, str>,
    pub(crate) stderr: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_dynamic_linking_with_lto)]
#[note]
pub(crate) struct DynamicLinkingWithLTO;

pub(crate) struct ParseTargetMachineConfig<'a>(pub(crate) LlvmError<'a>);

impl<G: EmissionGuarantee> Diagnostic<'_, G> for ParseTargetMachineConfig<'_> {
    fn into_diag(self, dcx: &'_ DiagCtxt, level: Level) -> Diag<'_, G> {
        let diag: Diag<'_, G> = self.0.into_diag(dcx, level);
        let (message, _) = diag.messages.first().expect("`LlvmError` with no message");
        let message = dcx.eagerly_translate_to_string(message.clone(), diag.args.iter());
        Diag::new(dcx, level, fluent::codegen_llvm_parse_target_machine_config)
            .with_arg("error", message)
    }
}

pub(crate) struct TargetFeatureDisableOrEnable<'a> {
    pub(crate) features: &'a [&'a str],
    pub(crate) span: Option<Span>,
    pub(crate) missing_features: Option<MissingFeatures>,
}

#[derive(Subdiagnostic)]
#[help(codegen_llvm_missing_features)]
pub(crate) struct MissingFeatures;

impl<G: EmissionGuarantee> Diagnostic<'_, G> for TargetFeatureDisableOrEnable<'_> {
    fn into_diag(self, dcx: &'_ DiagCtxt, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(dcx, level, fluent::codegen_llvm_target_feature_disable_or_enable);
        if let Some(span) = self.span {
            diag.span(span);
        };
        if let Some(missing_features) = self.missing_features {
            diag.subdiagnostic(dcx, missing_features);
        }
        diag.arg("features", self.features.join(", "));
        diag
    }
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_lto_disallowed)]
pub(crate) struct LtoDisallowed;

#[derive(Diagnostic)]
#[diag(codegen_llvm_lto_dylib)]
pub(crate) struct LtoDylib;

#[derive(Diagnostic)]
#[diag(codegen_llvm_lto_proc_macro)]
pub(crate) struct LtoProcMacro;

#[derive(Diagnostic)]
#[diag(codegen_llvm_lto_bitcode_from_rlib)]
pub(crate) struct LtoBitcodeFromRlib {
    pub(crate) llvm_err: String,
}

#[derive(Diagnostic)]
pub enum LlvmError<'a> {
    #[diag(codegen_llvm_write_output)]
    WriteOutput { path: &'a Path },
    #[diag(codegen_llvm_target_machine)]
    CreateTargetMachine { triple: SmallCStr },
    #[diag(codegen_llvm_run_passes)]
    RunLlvmPasses,
    #[diag(codegen_llvm_serialize_module)]
    SerializeModule { name: &'a str },
    #[diag(codegen_llvm_write_ir)]
    WriteIr { path: &'a Path },
    #[diag(codegen_llvm_prepare_thin_lto_context)]
    PrepareThinLtoContext,
    #[diag(codegen_llvm_load_bitcode)]
    LoadBitcode { name: CString },
    #[diag(codegen_llvm_write_thinlto_key)]
    WriteThinLtoKey { err: std::io::Error },
    #[diag(codegen_llvm_multiple_source_dicompileunit)]
    MultipleSourceDiCompileUnit,
    #[diag(codegen_llvm_prepare_thin_lto_module)]
    PrepareThinLtoModule,
    #[diag(codegen_llvm_parse_bitcode)]
    ParseBitcode,
}

pub(crate) struct WithLlvmError<'a>(pub(crate) LlvmError<'a>, pub(crate) String);

impl<G: EmissionGuarantee> Diagnostic<'_, G> for WithLlvmError<'_> {
    fn into_diag(self, dcx: &'_ DiagCtxt, level: Level) -> Diag<'_, G> {
        use LlvmError::*;
        let msg_with_llvm_err = match &self.0 {
            WriteOutput { .. } => fluent::codegen_llvm_write_output_with_llvm_err,
            CreateTargetMachine { .. } => fluent::codegen_llvm_target_machine_with_llvm_err,
            RunLlvmPasses => fluent::codegen_llvm_run_passes_with_llvm_err,
            SerializeModule { .. } => fluent::codegen_llvm_serialize_module_with_llvm_err,
            WriteIr { .. } => fluent::codegen_llvm_write_ir_with_llvm_err,
            PrepareThinLtoContext => fluent::codegen_llvm_prepare_thin_lto_context_with_llvm_err,
            LoadBitcode { .. } => fluent::codegen_llvm_load_bitcode_with_llvm_err,
            WriteThinLtoKey { .. } => fluent::codegen_llvm_write_thinlto_key_with_llvm_err,
            MultipleSourceDiCompileUnit => {
                fluent::codegen_llvm_multiple_source_dicompileunit_with_llvm_err
            }
            PrepareThinLtoModule => fluent::codegen_llvm_prepare_thin_lto_module_with_llvm_err,
            ParseBitcode => fluent::codegen_llvm_parse_bitcode_with_llvm_err,
        };
        self.0
            .into_diag(dcx, level)
            .with_primary_message(msg_with_llvm_err)
            .with_arg("llvm_err", self.1)
    }
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_from_llvm_optimization_diag)]
pub(crate) struct FromLlvmOptimizationDiag<'a> {
    pub(crate) filename: &'a str,
    pub(crate) line: std::ffi::c_uint,
    pub(crate) column: std::ffi::c_uint,
    pub(crate) pass_name: &'a str,
    pub(crate) kind: &'a str,
    pub(crate) message: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_from_llvm_diag)]
pub(crate) struct FromLlvmDiag {
    pub(crate) message: String,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_write_bytecode)]
pub(crate) struct WriteBytecode<'a> {
    pub(crate) path: &'a Path,
    pub(crate) err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_copy_bitcode)]
pub(crate) struct CopyBitcode {
    pub(crate) err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_unknown_debuginfo_compression)]
pub(crate) struct UnknownCompression {
    pub(crate) algorithm: &'static str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_mismatch_data_layout)]
pub(crate) struct MismatchedDataLayout<'a> {
    pub(crate) rustc_target: &'a str,
    pub(crate) rustc_layout: &'a str,
    pub(crate) llvm_target: &'a str,
    pub(crate) llvm_layout: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_invalid_target_feature_prefix)]
pub(crate) struct InvalidTargetFeaturePrefix<'a> {
    pub(crate) feature: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_fixed_x18_invalid_arch)]
pub(crate) struct FixedX18InvalidArch<'a> {
    pub(crate) arch: &'a str,
}
