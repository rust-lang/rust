use std::path::PathBuf;

use rustc_errors::ErrorGuaranteed;
use rustc_errors::IntoDiagnostic;
use rustc_macros::{Diagnostic, LintDiagnostic};
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(monomorphize_recursion_limit)]
pub struct RecursionLimit {
    #[primary_span]
    pub span: Span,
    pub shrunk: String,
    #[note]
    pub def_span: Span,
    pub def_path_str: String,
    #[note(monomorphize_written_to_path)]
    pub was_written: Option<()>,
    pub path: PathBuf,
}

#[derive(Diagnostic)]
#[diag(monomorphize_type_length_limit)]
#[help(monomorphize_consider_type_length_limit)]
pub struct TypeLengthLimit {
    #[primary_span]
    pub span: Span,
    pub shrunk: String,
    #[note(monomorphize_written_to_path)]
    pub was_written: Option<()>,
    pub path: PathBuf,
    pub type_length: usize,
}

pub struct UnusedGenericParamsHint {
    pub span: Span,
    pub param_spans: Vec<Span>,
    pub param_names: Vec<String>,
}

impl IntoDiagnostic<'_> for UnusedGenericParamsHint {
    #[track_caller]
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(rustc_errors::fluent::monomorphize_unused_generic_params);
        diag.set_span(self.span);
        for (span, name) in self.param_spans.into_iter().zip(self.param_names) {
            // FIXME: I can figure out how to do a label with a fluent string with a fixed message,
            // or a label with a dynamic value in a hard-coded string, but I haven't figured out
            // how to combine the two. 😢
            diag.span_label(span, format!("generic parameter `{name}` is unused"));
        }
        diag
    }
}

#[derive(LintDiagnostic)]
#[diag(monomorphize_large_assignments)]
#[note]
pub struct LargeAssignmentsLint {
    #[label]
    pub span: Span,
    pub size: u64,
    pub limit: u64,
}

#[derive(Diagnostic)]
#[diag(monomorphize_unknown_partition_strategy)]
pub struct UnknownPartitionStrategy;

#[derive(Diagnostic)]
#[diag(monomorphize_symbol_already_defined)]
pub struct SymbolAlreadyDefined {
    #[primary_span]
    pub span: Option<Span>,
    pub symbol: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_couldnt_dump_mono_stats)]
pub struct CouldntDumpMonoStats {
    pub error: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_encountered_error_while_instantiating)]
pub struct EncounteredErrorWhileInstantiating {
    #[primary_span]
    pub span: Span,
    pub formatted_item: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_unknown_cgu_collection_mode)]
pub struct UnknownCguCollectionMode<'a> {
    pub mode: &'a str,
}
