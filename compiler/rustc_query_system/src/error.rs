use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed};
use rustc_session::SessionDiagnostic;
use rustc_span::Span;

pub struct Cycle {
    pub span: Span,
    pub stack_bottom: String,
    pub upper_stack_info: Vec<(Span, String)>,
    pub recursive_ty_alias: bool,
    pub recursive_trait_alias: bool,
    pub cycle_usage: Option<(Span, String)>,
}

impl SessionDiagnostic<'_> for Cycle {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(rustc_errors::fluent::query_system::cycle);
        diag.set_span(self.span);
        diag.code(rustc_errors::DiagnosticId::Error("E0391".to_string()));
        let upper_stack_len = self.upper_stack_info.len();
        for (span, desc) in self.upper_stack_info.into_iter() {
            // FIXME(#100717): use fluent translation
            diag.span_note(span, &format!("...which requires {}...", desc));
        }
        diag.set_arg("stack_bottom", self.stack_bottom);
        if upper_stack_len == 0 {
            diag.note(rustc_errors::fluent::query_system::cycle_stack_single);
        } else {
            diag.note(rustc_errors::fluent::query_system::cycle_stack_multiple);
        }
        if self.recursive_trait_alias {
            diag.note(rustc_errors::fluent::query_system::cycle_recursive_trait_alias);
        } else if self.recursive_ty_alias {
            diag.note(rustc_errors::fluent::query_system::cycle_recursive_ty_alias);
            diag.help(rustc_errors::fluent::query_system::cycle_recursive_ty_alias_help1);
            diag.help(rustc_errors::fluent::query_system::cycle_recursive_ty_alias_help2);
        }
        if let Some((span, desc)) = self.cycle_usage {
            diag.set_arg("usage", desc);
            diag.span_note(span, rustc_errors::fluent::query_system::cycle_usage);
        }
        diag
    }
}

#[derive(SessionDiagnostic)]
#[diag(query_system::reentrant)]
pub struct Reentrant;

#[derive(SessionDiagnostic)]
#[diag(query_system::increment_compilation)]
#[help]
#[note(query_system::increment_compilation_note1)]
#[note(query_system::increment_compilation_note2)]
pub struct IncrementCompilation {
    pub run_cmd: String,
    pub dep_node: String,
}
