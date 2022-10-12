use rustc_macros::Diagnostic;

use std::path::Path;

#[derive(Diagnostic)]
#[diag(save_analysis::could_not_open)]
pub(crate) struct CouldNotOpen<'a> {
    pub file_name: &'a Path,
    pub err: std::io::Error,
}
