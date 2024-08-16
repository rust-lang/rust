use super::*;
use crate::rustfmt_diff::{DiffLine, Mismatch, make_diff};
use serde::Serialize;
use serde_json::to_string as to_json_string;

#[derive(Debug, Default)]
pub(crate) struct JsonEmitter {
    mismatched_files: Vec<MismatchedFile>,
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct MismatchedBlock {
    original_begin_line: u32,
    original_end_line: u32,
    expected_begin_line: u32,
    expected_end_line: u32,
    original: String,
    expected: String,
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct MismatchedFile {
    name: String,
    mismatches: Vec<MismatchedBlock>,
}

impl Emitter for JsonEmitter {
    fn emit_footer(&self, output: &mut dyn Write) -> Result<(), io::Error> {
        writeln!(output, "{}", &to_json_string(&self.mismatched_files)?)
    }

    fn emit_formatted_file(
        &mut self,
        _output: &mut dyn Write,
        FormattedFile {
            filename,
            original_text,
            formatted_text,
        }: FormattedFile<'_>,
    ) -> Result<EmitterResult, io::Error> {
        const CONTEXT_SIZE: usize = 0;
        let diff = make_diff(original_text, formatted_text, CONTEXT_SIZE);
        let has_diff = !diff.is_empty();

        if has_diff {
            self.add_misformatted_file(filename, diff)?;
        }

        Ok(EmitterResult { has_diff })
    }
}

impl JsonEmitter {
    fn add_misformatted_file(
        &mut self,
        filename: &FileName,
        diff: Vec<Mismatch>,
    ) -> Result<(), io::Error> {
        let mut mismatches = vec![];
        for mismatch in diff {
            let original_begin_line = mismatch.line_number_orig;
            let expected_begin_line = mismatch.line_number;
            let mut original_end_line = original_begin_line;
            let mut expected_end_line = expected_begin_line;
            let mut original_line_counter = 0;
            let mut expected_line_counter = 0;
            let mut original = String::new();
            let mut expected = String::new();

            for line in mismatch.lines {
                match line {
                    DiffLine::Expected(msg) => {
                        expected_end_line = expected_begin_line + expected_line_counter;
                        expected_line_counter += 1;
                        expected.push_str(&msg);
                        expected.push('\n');
                    }
                    DiffLine::Resulting(msg) => {
                        original_end_line = original_begin_line + original_line_counter;
                        original_line_counter += 1;
                        original.push_str(&msg);
                        original.push('\n');
                    }
                    DiffLine::Context(_) => continue,
                }
            }

            mismatches.push(MismatchedBlock {
                original_begin_line,
                original_end_line,
                expected_begin_line,
                expected_end_line,
                original,
                expected,
            });
        }
        self.mismatched_files.push(MismatchedFile {
            name: format!("{filename}"),
            mismatches,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn expected_line_range_correct_when_single_line_split() {
        let mut emitter = JsonEmitter {
            mismatched_files: vec![],
        };
        let file = "foo/bar.rs";
        let mismatched_file = MismatchedFile {
            name: String::from(file),
            mismatches: vec![MismatchedBlock {
                original_begin_line: 79,
                original_end_line: 79,
                expected_begin_line: 79,
                expected_end_line: 82,
                original: String::from("fn Foo<T>() where T: Bar {\n"),
                expected: String::from("fn Foo<T>()\nwhere\n    T: Bar,\n{\n"),
            }],
        };
        let mismatch = Mismatch {
            line_number: 79,
            line_number_orig: 79,
            lines: vec![
                DiffLine::Resulting(String::from("fn Foo<T>() where T: Bar {")),
                DiffLine::Expected(String::from("fn Foo<T>()")),
                DiffLine::Expected(String::from("where")),
                DiffLine::Expected(String::from("    T: Bar,")),
                DiffLine::Expected(String::from("{")),
            ],
        };

        let _ = emitter
            .add_misformatted_file(&FileName::Real(PathBuf::from(file)), vec![mismatch])
            .unwrap();

        assert_eq!(emitter.mismatched_files.len(), 1);
        assert_eq!(emitter.mismatched_files[0], mismatched_file);
    }

    #[test]
    fn context_lines_ignored() {
        let mut emitter = JsonEmitter {
            mismatched_files: vec![],
        };
        let file = "src/lib.rs";
        let mismatched_file = MismatchedFile {
            name: String::from(file),
            mismatches: vec![MismatchedBlock {
                original_begin_line: 5,
                original_end_line: 5,
                expected_begin_line: 5,
                expected_end_line: 5,
                original: String::from(
                    "fn foo(_x: &u64) -> Option<&(dyn::std::error::Error + 'static)> {\n",
                ),
                expected: String::from(
                    "fn foo(_x: &u64) -> Option<&(dyn ::std::error::Error + 'static)> {\n",
                ),
            }],
        };
        let mismatch = Mismatch {
            line_number: 5,
            line_number_orig: 5,
            lines: vec![
                DiffLine::Context(String::new()),
                DiffLine::Resulting(String::from(
                    "fn foo(_x: &u64) -> Option<&(dyn::std::error::Error + 'static)> {",
                )),
                DiffLine::Context(String::new()),
                DiffLine::Expected(String::from(
                    "fn foo(_x: &u64) -> Option<&(dyn ::std::error::Error + 'static)> {",
                )),
                DiffLine::Context(String::new()),
            ],
        };

        let _ = emitter
            .add_misformatted_file(&FileName::Real(PathBuf::from(file)), vec![mismatch])
            .unwrap();

        assert_eq!(emitter.mismatched_files.len(), 1);
        assert_eq!(emitter.mismatched_files[0], mismatched_file);
    }

    #[test]
    fn emits_empty_array_on_no_diffs() {
        let mut writer = Vec::new();
        let mut emitter = JsonEmitter::default();
        let _ = emitter.emit_header(&mut writer);
        let result = emitter
            .emit_formatted_file(
                &mut writer,
                FormattedFile {
                    filename: &FileName::Real(PathBuf::from("src/lib.rs")),
                    original_text: "fn empty() {}\n",
                    formatted_text: "fn empty() {}\n",
                },
            )
            .unwrap();
        let _ = emitter.emit_footer(&mut writer);
        assert_eq!(result.has_diff, false);
        assert_eq!(&writer[..], "[]\n".as_bytes());
    }

    #[test]
    fn emits_array_with_files_with_diffs() {
        let file_name = "src/bin.rs";
        let original = [
            "fn main() {",
            "println!(\"Hello, world!\");",
            "}",
            "",
            "#[cfg(test)]",
            "mod tests {",
            "#[test]",
            "fn it_works() {",
            "    assert_eq!(2 + 2, 4);",
            "}",
            "}",
        ];
        let formatted = [
            "fn main() {",
            "    println!(\"Hello, world!\");",
            "}",
            "",
            "#[cfg(test)]",
            "mod tests {",
            "    #[test]",
            "    fn it_works() {",
            "        assert_eq!(2 + 2, 4);",
            "    }",
            "}",
        ];
        let mut writer = Vec::new();
        let mut emitter = JsonEmitter::default();
        let _ = emitter.emit_header(&mut writer);
        let result = emitter
            .emit_formatted_file(
                &mut writer,
                FormattedFile {
                    filename: &FileName::Real(PathBuf::from(file_name)),
                    original_text: &original.join("\n"),
                    formatted_text: &formatted.join("\n"),
                },
            )
            .unwrap();
        let _ = emitter.emit_footer(&mut writer);
        let exp_json = to_json_string(&vec![MismatchedFile {
            name: String::from(file_name),
            mismatches: vec![
                MismatchedBlock {
                    original_begin_line: 2,
                    original_end_line: 2,
                    expected_begin_line: 2,
                    expected_end_line: 2,
                    original: String::from("println!(\"Hello, world!\");\n"),
                    expected: String::from("    println!(\"Hello, world!\");\n"),
                },
                MismatchedBlock {
                    original_begin_line: 7,
                    original_end_line: 10,
                    expected_begin_line: 7,
                    expected_end_line: 10,
                    original: String::from(
                        "#[test]\nfn it_works() {\n    assert_eq!(2 + 2, 4);\n}\n",
                    ),
                    expected: String::from(
                        "    #[test]\n    fn it_works() {\n        assert_eq!(2 + 2, 4);\n    }\n",
                    ),
                },
            ],
        }])
        .unwrap();
        assert_eq!(result.has_diff, true);
        assert_eq!(&writer[..], format!("{exp_json}\n").as_bytes());
    }

    #[test]
    fn emits_valid_json_with_multiple_files() {
        let bin_file = "src/bin.rs";
        let bin_original = ["fn main() {", "println!(\"Hello, world!\");", "}"];
        let bin_formatted = ["fn main() {", "    println!(\"Hello, world!\");", "}"];
        let lib_file = "src/lib.rs";
        let lib_original = ["fn greet() {", "println!(\"Greetings!\");", "}"];
        let lib_formatted = ["fn greet() {", "    println!(\"Greetings!\");", "}"];
        let mut writer = Vec::new();
        let mut emitter = JsonEmitter::default();
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
        let exp_bin = MismatchedFile {
            name: String::from(bin_file),
            mismatches: vec![MismatchedBlock {
                original_begin_line: 2,
                original_end_line: 2,
                expected_begin_line: 2,
                expected_end_line: 2,
                original: String::from("println!(\"Hello, world!\");\n"),
                expected: String::from("    println!(\"Hello, world!\");\n"),
            }],
        };

        let exp_lib = MismatchedFile {
            name: String::from(lib_file),
            mismatches: vec![MismatchedBlock {
                original_begin_line: 2,
                original_end_line: 2,
                expected_begin_line: 2,
                expected_end_line: 2,
                original: String::from("println!(\"Greetings!\");\n"),
                expected: String::from("    println!(\"Greetings!\");\n"),
            }],
        };

        let exp_json = to_json_string(&vec![exp_bin, exp_lib]).unwrap();
        assert_eq!(&writer[..], format!("{exp_json}\n").as_bytes());
    }
}
