use super::*;
use std::fs;

#[derive(Debug, Default)]
pub(crate) struct FilesWithBackupEmitter;

impl Emitter for FilesWithBackupEmitter {
    fn emit_formatted_file(
        &mut self,
        _output: &mut dyn Write,
        FormattedFile {
            filename,
            original_text,
            formatted_text,
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        let filename = ensure_real_path(filename);
        if original_text != formatted_text {
            // Do a little dance to make writing safer - write to a temp file
            // rename the original to a .bk, then rename the temp file to the
            // original.
            let tmp_name = filename.with_extension("tmp");
            let bk_name = filename.with_extension("bk");

            fs::write(&tmp_name, formatted_text)?;
            fs::rename(filename, bk_name)?;
            fs::rename(tmp_name, filename)?;
        }
        Ok(EmitterResult::default())
    }
}
