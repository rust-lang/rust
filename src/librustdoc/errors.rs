use crate::fluent_generated as fluent;
use rustc_errors::IntoDiagnostic;
use rustc_macros::Diagnostic;
use rustc_span::ErrorGuaranteed;

#[derive(Diagnostic)]
#[diag(rustdoc_main_error)]
pub(crate) struct MainError {
    pub(crate) error: String,
}

pub(crate) struct CouldntGenerateDocumentation {
    pub(crate) error: String,
    pub(crate) file: String,
}

impl IntoDiagnostic<'_> for CouldntGenerateDocumentation {
    fn into_diagnostic(
        self,
        handler: &'_ rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::rustdoc_couldnt_generate_documentation);
        diag.set_arg("error", self.error);
        if !self.file.is_empty() {
            diag.note(fluent::_subdiag::note);
        }

        diag
    }
}

#[derive(Diagnostic)]
#[diag(rustdoc_compilation_failed)]
pub(crate) struct CompilationFailed;
