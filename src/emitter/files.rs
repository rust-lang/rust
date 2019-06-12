use super::*;
use std::fs;

#[derive(Debug, Default)]
pub(crate) struct FilesEmitter;

impl Emitter for FilesEmitter {
    fn emit_formatted_file(
        &self,
        _output: &mut dyn Write,
        FormattedFile {
            filename,
            original_text,
            formatted_text,
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        // Write text directly over original file if there is a diff.
        let filename = ensure_real_path(filename);
        if original_text != formatted_text {
            fs::write(filename, formatted_text)?;
        }
        Ok(EmitterResult::default())
    }
}
