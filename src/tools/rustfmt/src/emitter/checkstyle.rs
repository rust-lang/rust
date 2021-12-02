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
        &mut self,
        output: &mut dyn Write,
        FormattedFile {
            filename,
            original_text,
            formatted_text,
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        const CONTEXT_SIZE: usize = 0;
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
        let begin_line = mismatch.line_number;
        let mut current_line;
        let mut line_counter = 0;
        for line in mismatch.lines {
            // Do nothing with `DiffLine::Context` and `DiffLine::Resulting`.
            if let DiffLine::Expected(message) = line {
                current_line = begin_line + line_counter;
                line_counter += 1;
                write!(
                    writer,
                    r#"<error line="{}" severity="warning" message="Should be `{}`" />"#,
                    current_line,
                    XmlEscaped(&message)
                )?;
            }
        }
    }
    write!(writer, "</file>")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn emits_empty_record_on_file_with_no_mismatches() {
        let file_name = "src/well_formatted.rs";
        let mut writer = Vec::new();
        let _ = output_checkstyle_file(&mut writer, &PathBuf::from(file_name), vec![]);
        assert_eq!(
            &writer[..],
            format!(r#"<file name="{}"></file>"#, file_name).as_bytes()
        );
    }

    // https://github.com/rust-lang/rustfmt/issues/1636
    #[test]
    fn emits_single_xml_tree_containing_all_files() {
        let bin_file = "src/bin.rs";
        let bin_original = vec!["fn main() {", "println!(\"Hello, world!\");", "}"];
        let bin_formatted = vec!["fn main() {", "    println!(\"Hello, world!\");", "}"];
        let lib_file = "src/lib.rs";
        let lib_original = vec!["fn greet() {", "println!(\"Greetings!\");", "}"];
        let lib_formatted = vec!["fn greet() {", "    println!(\"Greetings!\");", "}"];
        let mut writer = Vec::new();
        let mut emitter = CheckstyleEmitter::default();
        let _ = emitter.emit_header(&mut writer);
        let _ = emitter
            .emit_formatted_file(
                &mut writer,
                FormattedFile {
                    filename: &FileName::Real(PathBuf::from(bin_file)),
                    original_text: &bin_original.join("\n"),
                    formatted_text: &bin_formatted.join("\n"),
                },
            )
            .unwrap();
        let _ = emitter
            .emit_formatted_file(
                &mut writer,
                FormattedFile {
                    filename: &FileName::Real(PathBuf::from(lib_file)),
                    original_text: &lib_original.join("\n"),
                    formatted_text: &lib_formatted.join("\n"),
                },
            )
            .unwrap();
        let _ = emitter.emit_footer(&mut writer);
        let exp_bin_xml = vec![
            format!(r#"<file name="{}">"#, bin_file),
            format!(
                r#"<error line="2" severity="warning" message="Should be `{}`" />"#,
                XmlEscaped(r#"    println!("Hello, world!");"#),
            ),
            String::from("</file>"),
        ];
        let exp_lib_xml = vec![
            format!(r#"<file name="{}">"#, lib_file),
            format!(
                r#"<error line="2" severity="warning" message="Should be `{}`" />"#,
                XmlEscaped(r#"    println!("Greetings!");"#),
            ),
            String::from("</file>"),
        ];
        assert_eq!(
            String::from_utf8(writer).unwrap(),
            vec![
                r#"<?xml version="1.0" encoding="utf-8"?>"#,
                "\n",
                r#"<checkstyle version="4.3">"#,
                &format!("{}{}", exp_bin_xml.join(""), exp_lib_xml.join("")),
                "</checkstyle>\n",
            ]
            .join(""),
        );
    }
}
