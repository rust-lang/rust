use std::collections::VecDeque;
use std::fs::{File, FileType};
use std::path::Path;

#[derive(Debug, PartialEq)]
pub enum DiffLine {
    Context(String),
    Expected(String),
    Resulting(String),
}

#[derive(Debug, PartialEq)]
pub struct Mismatch {
    pub line_number: u32,
    pub lines: Vec<DiffLine>,
}

impl Mismatch {
    fn new(line_number: u32) -> Mismatch {
        Mismatch { line_number, lines: Vec::new() }
    }
}

// Produces a diff between the expected output and actual output.
pub fn make_diff(expected: &str, actual: &str, context_size: usize) -> Vec<Mismatch> {
    let mut line_number = 1;
    let mut context_queue: VecDeque<&str> = VecDeque::with_capacity(context_size);
    let mut lines_since_mismatch = context_size + 1;
    let mut results = Vec::new();
    let mut mismatch = Mismatch::new(0);

    for result in diff::lines(expected, actual) {
        match result {
            diff::Result::Left(str) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Expected(str.to_owned()));
                line_number += 1;
                lines_since_mismatch = 0;
            }
            diff::Result::Right(str) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Resulting(str.to_owned()));
                lines_since_mismatch = 0;
            }
            diff::Result::Both(str, _) => {
                if context_queue.len() >= context_size {
                    let _ = context_queue.pop_front();
                }

                if lines_since_mismatch < context_size {
                    mismatch.lines.push(DiffLine::Context(str.to_owned()));
                } else if context_size > 0 {
                    context_queue.push_back(str);
                }

                line_number += 1;
                lines_since_mismatch += 1;
            }
        }
    }

    results.push(mismatch);
    results.remove(0);

    results
}

pub(crate) fn write_diff(expected: &str, actual: &str, context_size: usize) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    let diff_results = make_diff(expected, actual, context_size);
    for result in diff_results {
        let mut line_number = result.line_number;
        for line in result.lines {
            match line {
                DiffLine::Expected(e) => {
                    writeln!(output, "-\t{}", e).unwrap();
                    line_number += 1;
                }
                DiffLine::Context(c) => {
                    writeln!(output, "{}\t{}", line_number, c).unwrap();
                    line_number += 1;
                }
                DiffLine::Resulting(r) => {
                    writeln!(output, "+\t{}", r).unwrap();
                }
            }
        }
        writeln!(output).unwrap();
    }
    output
}

/// Filters based on filetype and extension whether to diff a file.
///
/// Returns whether any data was actually written.
pub(crate) fn write_filtered_diff<Filter>(
    diff_filename: &str,
    out_dir: &Path,
    compare_dir: &Path,
    verbose: bool,
    filter: Filter,
) -> bool
where
    Filter: Fn(FileType, Option<&str>) -> bool,
{
    use std::io::{Read, Write};
    let mut diff_output = File::create(diff_filename).unwrap();
    let mut wrote_data = false;
    for entry in walkdir::WalkDir::new(out_dir) {
        let entry = entry.expect("failed to read file");
        let extension = entry.path().extension().and_then(|p| p.to_str());
        if filter(entry.file_type(), extension) {
            let expected_path = compare_dir.join(entry.path().strip_prefix(&out_dir).unwrap());
            let expected = if let Ok(s) = std::fs::read(&expected_path) { s } else { continue };
            let actual_path = entry.path();
            let actual = std::fs::read(&actual_path).unwrap();
            let diff = unified_diff::diff(
                &expected,
                &expected_path.to_string_lossy(),
                &actual,
                &actual_path.to_string_lossy(),
                3,
            );
            wrote_data |= !diff.is_empty();
            diff_output.write_all(&diff).unwrap();
        }
    }

    if !wrote_data {
        println!("note: diff is identical to nightly rustdoc");
        assert!(diff_output.metadata().unwrap().len() == 0);
        return false;
    } else if verbose {
        eprintln!("printing diff:");
        let mut buf = Vec::new();
        diff_output.read_to_end(&mut buf).unwrap();
        std::io::stderr().lock().write_all(&mut buf).unwrap();
    }
    true
}
