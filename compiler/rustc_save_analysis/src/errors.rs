use rustc_macros::DiagnosticHandler;

use std::path::Path;

#[derive(DiagnosticHandler)]
#[diag(save_analysis::could_not_open)]
pub(crate) struct CouldNotOpen<'a> {
    pub file_name: &'a Path,
    pub err: std::io::Error,
}
