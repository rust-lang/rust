pub(crate) use self::checkstyle::*;
pub(crate) use self::diff::*;
pub(crate) use self::files::*;
pub(crate) use self::files_with_backup::*;
pub(crate) use self::json::*;
pub(crate) use self::modified_lines::*;
pub(crate) use self::stdout::*;
use crate::FileName;
use std::io::{self, Write};
use std::path::Path;

mod checkstyle;
mod diff;
mod files;
mod files_with_backup;
mod json;
mod modified_lines;
mod stdout;

pub(crate) struct FormattedFile<'a> {
    pub(crate) filename: &'a FileName,
    pub(crate) original_text: &'a str,
    pub(crate) formatted_text: &'a str,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct EmitterResult {
    pub(crate) has_diff: bool,
}

pub(crate) trait Emitter {
    fn emit_formatted_file(
        &mut self,
        output: &mut dyn Write,
        formatted_file: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error>;

    fn emit_header(&self, _output: &mut dyn Write) -> Result<(), io::Error> {
        Ok(())
    }

    fn emit_footer(&self, _output: &mut dyn Write) -> Result<(), io::Error> {
        Ok(())
    }
}

fn ensure_real_path(filename: &FileName) -> &Path {
    match *filename {
        FileName::Real(ref path) => path,
        _ => panic!("cannot format `{filename}` and emit to files"),
    }
}
