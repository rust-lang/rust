use rustc_macros::SessionDiagnostic;

use std::path::Path;

#[derive(SessionDiagnostic)]
#[diag(save_analysis::could_not_open)]
pub(crate) struct CouldNotOpen<'a> {
    pub file_name: &'a Path,
    pub err: std::io::Error,
}
