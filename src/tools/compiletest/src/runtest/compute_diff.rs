use std::collections::VecDeque;

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
            diff::Result::Left(s) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Expected(s.to_owned()));
                line_number += 1;
                lines_since_mismatch = 0;
            }
            diff::Result::Right(s) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Resulting(s.to_owned()));
                lines_since_mismatch = 0;
            }
            diff::Result::Both(s, _) => {
                if context_queue.len() >= context_size {
                    let _ = context_queue.pop_front();
                }

                if lines_since_mismatch < context_size {
                    mismatch.lines.push(DiffLine::Context(s.to_owned()));
                } else if context_size > 0 {
                    context_queue.push_back(s);
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
