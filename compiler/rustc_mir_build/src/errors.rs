use rustc_macros::LintDiagnostic;
use rustc_span::Span;

#[derive(LintDiagnostic)]
#[lint(mir_build::unconditional_recursion)]
#[help]
pub struct UnconditionalRecursion {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(mir_build::unconditional_recursion_call_site_label)]
    pub call_sites: Vec<Span>,
}
