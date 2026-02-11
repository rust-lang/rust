use rustc_macros::Diagnostic;

#[derive(Diagnostic)]
#[diag("internal compiler error: reentrant incremental verify failure, suppressing message")]
pub(crate) struct Reentrant;

#[derive(Diagnostic)]
#[diag("internal compiler error: encountered incremental compilation error with {$dep_node}")]
#[note("please follow the instructions below to create a bug report with the provided information")]
#[note("for incremental compilation bugs, having a reproduction is vital")]
#[note(
    "an ideal reproduction consists of the code before and some patch that then triggers the bug when applied and compiled again"
)]
#[note("as a workaround, you can run {$run_cmd} to allow your project to compile")]
pub(crate) struct IncrementCompilation {
    pub run_cmd: String,
    pub dep_node: String,
}
