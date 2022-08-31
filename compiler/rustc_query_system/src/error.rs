use rustc_errors::AddSubdiagnostic;
use rustc_span::Span;

pub struct CycleStack {
    pub span: Span,
    pub desc: String,
}

impl AddSubdiagnostic for CycleStack {
    fn add_to_diagnostic(self, diag: &mut rustc_errors::Diagnostic) {
        diag.span_note(self.span, &format!("...which requires {}...", self.desc));
    }
}

#[derive(SessionSubdiagnostic)]
pub enum StackCount {
    #[note(query_system::cycle_stack_single)]
    Single,
    #[note(query_system::cycle_stack_multiple)]
    Multiple,
}

#[derive(SessionSubdiagnostic)]
pub enum Alias {
    #[note(query_system::cycle_recursive_ty_alias)]
    #[help(query_system::cycle_recursive_ty_alias_help1)]
    #[help(query_system::cycle_recursive_ty_alias_help2)]
    Ty,
    #[note(query_system::cycle_recursive_trait_alias)]
    Trait,
}

#[derive(SessionSubdiagnostic)]
#[note(query_system::cycle_usage)]
pub struct CycleUsage {
    #[primary_span]
    pub span: Span,
    pub usage: String,
}

#[derive(SessionDiagnostic)]
#[diag(query_system::cycle, code = "E0391")]
pub struct Cycle {
    #[primary_span]
    pub span: Span,
    pub stack_bottom: String,
    #[subdiagnostic]
    pub cycle_stack: Vec<CycleStack>,
    #[subdiagnostic]
    pub stack_count: StackCount,
    #[subdiagnostic]
    pub alias: Option<Alias>,
    #[subdiagnostic]
    pub cycle_usage: Option<CycleUsage>,
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

#[derive(SessionDiagnostic)]
#[diag(query_system::query_overflow)]
pub struct QueryOverflow;
