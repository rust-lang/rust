use self::xml::XmlEscaped;
use super::*;
use crate::rustfmt_diff::{make_diff, DiffLine, Mismatch};
use std::io::{self, Write};
use std::path::Path;

mod xml;

#[derive(Debug, Default)]
pub(crate) struct CheckstyleEmitter;

impl Emitter for CheckstyleEmitter {
    fn emit_header(&self, output: &mut dyn Write) -> Result<(), io::Error> {
        writeln!(output, r#"<?xml version="1.0" encoding="utf-8"?>"#)?;
        write!(output, r#"<checkstyle version="4.3">"#)?;
        Ok(())
    }

    fn emit_footer(&self, output: &mut dyn Write) -> Result<(), io::Error> {
        writeln!(output, "</checkstyle>")
    }

    fn emit_formatted_file(
        &self,
        output: &mut dyn Write,
        FormattedFile {
            filename,
            original_text,
            formatted_text,
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        const CONTEXT_SIZE: usize = 3;
        let filename = ensure_real_path(filename);
        let diff = make_diff(original_text, formatted_text, CONTEXT_SIZE);
        output_checkstyle_file(output, filename, diff)?;
        Ok(EmitterResult::default())
    }
}

pub(crate) fn output_checkstyle_file<T>(
    mut writer: T,
    filename: &Path,
    diff: Vec<Mismatch>,
) -> Result<(), io::Error>
where
    T: Write,
{
    write!(writer, r#"<file name="{}">"#, filename.display())?;
    for mismatch in diff {
        for line in mismatch.lines {
            // Do nothing with `DiffLine::Context` and `DiffLine::Resulting`.
            if let DiffLine::Expected(message) = line {
                write!(
                    writer,
                    r#"<error line="{}" severity="warning" message="Should be `{}`" />"#,
                    mismatch.line_number,
                    XmlEscaped(&message)
                )?;
            }
        }
    }
    write!(writer, "</file>")?;
    Ok(())
}
