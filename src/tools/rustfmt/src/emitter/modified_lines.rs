use super::*;
use crate::rustfmt_diff::{ModifiedLines, make_diff};

#[derive(Debug, Default)]
pub(crate) struct ModifiedLinesEmitter;

impl Emitter for ModifiedLinesEmitter {
    fn emit_formatted_file(
        &mut self,
        output: &mut dyn Write,
        FormattedFile {
            original_text,
            formatted_text,
            ..
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        const CONTEXT_SIZE: usize = 0;
        let mismatch = make_diff(original_text, formatted_text, CONTEXT_SIZE);
        let has_diff = !mismatch.is_empty();
        write!(output, "{}", ModifiedLines::from(mismatch))?;
        Ok(EmitterResult { has_diff })
    }
}
