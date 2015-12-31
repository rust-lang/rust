use std::collections::VecDeque;
use diff;
use term;

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
        Mismatch {
            line_number: line_number,
            lines: Vec::new(),
        }
    }
}

// Produces a diff between the expected output and actual output of rustfmt.
pub fn make_diff(expected: &str, actual: &str, context_size: usize) -> Vec<Mismatch> {
    let mut line_number = 1;
    let mut context_queue: VecDeque<&str> = VecDeque::with_capacity(context_size);
    let mut lines_since_mismatch = context_size + 1;
    let mut results = Vec::new();
    let mut mismatch = Mismatch::new(0);

    for result in diff::lines(expected, actual) {
        match result {
            diff::Result::Left(str) => {
                if lines_since_mismatch >= context_size {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Resulting(str.to_owned()));
                lines_since_mismatch = 0;
            }
            diff::Result::Right(str) => {
                if lines_since_mismatch >= context_size {
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
            diff::Result::Both(str, _) => {
                if context_queue.len() >= context_size {
                    let _ = context_queue.pop_front();
                }

                if lines_since_mismatch < context_size {
                    mismatch.lines.push(DiffLine::Context(str.to_owned()));
                } else {
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

pub fn print_diff<F>(diff: Vec<Mismatch>, get_section_title: F)
    where F: Fn(u32) -> String
{
    let mut t = term::stdout().unwrap();

    for mismatch in diff {
        let title = get_section_title(mismatch.line_number);
        writeln!(t, "{}", title).unwrap();

        for line in mismatch.lines {
            match line {
                DiffLine::Context(ref str) => {
                    t.reset().unwrap();
                    writeln!(t, " {}⏎", str).unwrap();
                }
                DiffLine::Expected(ref str) => {
                    t.fg(term::color::GREEN).unwrap();
                    writeln!(t, "+{}⏎", str).unwrap();
                }
                DiffLine::Resulting(ref str) => {
                    t.fg(term::color::RED).unwrap();
                    writeln!(t, "-{}⏎", str).unwrap();
                }
            }
        }
        t.reset().unwrap();
    }
}
