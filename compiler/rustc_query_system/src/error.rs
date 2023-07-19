use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::Limit;
use rustc_span::{Span, Symbol};

#[derive(Subdiagnostic)]
#[note(query_system_cycle_stack_middle)]
pub struct CycleStack {
    #[primary_span]
    pub span: Span,
    pub desc: String,
}

#[derive(Copy, Clone)]
pub enum HandleCycleError {
    Error,
    Fatal,
    DelayBug,
}

#[derive(Subdiagnostic)]
pub enum StackCount {
    #[note(query_system_cycle_stack_single)]
    Single,
    #[note(query_system_cycle_stack_multiple)]
    Multiple,
}

#[derive(Subdiagnostic)]
pub enum Alias {
    #[note(query_system_cycle_recursive_ty_alias)]
    #[help(query_system_cycle_recursive_ty_alias_help1)]
    #[help(query_system_cycle_recursive_ty_alias_help2)]
    Ty,
    #[note(query_system_cycle_recursive_trait_alias)]
    Trait,
}

#[derive(Subdiagnostic)]
#[note(query_system_cycle_usage)]
pub struct CycleUsage {
    #[primary_span]
    pub span: Span,
    pub usage: String,
}

#[derive(Diagnostic)]
#[diag(query_system_cycle, code = "E0391")]
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
    #[note]
    pub note_span: (),
}

#[derive(Diagnostic)]
#[diag(query_system_reentrant)]
pub struct Reentrant;

#[derive(Diagnostic)]
#[diag(query_system_increment_compilation)]
#[help]
#[note(query_system_increment_compilation_note1)]
#[note(query_system_increment_compilation_note2)]
pub struct IncrementCompilation {
    pub run_cmd: String,
    pub dep_node: String,
}

#[derive(Diagnostic)]
#[help]
#[diag(query_system_query_overflow)]
pub struct QueryOverflow {
    #[primary_span]
    pub span: Option<Span>,
    #[subdiagnostic]
    pub layout_of_depth: Option<LayoutOfDepth>,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(Subdiagnostic)]
#[note(query_system_layout_of_depth)]
pub struct LayoutOfDepth {
    pub desc: String,
    pub depth: usize,
}
