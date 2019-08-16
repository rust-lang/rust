use super::*;
use crate::config::Config;
use crate::rustfmt_diff::{make_diff, print_diff};

pub(crate) struct DiffEmitter {
    config: Config,
}

impl DiffEmitter {
    pub(crate) fn new(config: Config) -> Self {
        Self { config }
    }
}

impl Emitter for DiffEmitter {
    fn emit_formatted_file(
        &mut self,
        _output: &mut dyn Write,
        FormattedFile {
            filename,
            original_text,
            formatted_text,
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        const CONTEXT_SIZE: usize = 3;
        let mismatch = make_diff(&original_text, formatted_text, CONTEXT_SIZE);
        let has_diff = !mismatch.is_empty();
        print_diff(
            mismatch,
            |line_num| format!("Diff in {} at line {}:", filename, line_num),
            &self.config,
        );
        return Ok(EmitterResult { has_diff });
    }
}
