// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use config::{Color, Config, Verbosity};
use diff;
use std::collections::VecDeque;
use std::io;
use std::io::Write;

#[derive(Debug, PartialEq)]
pub enum DiffLine {
    Context(String),
    Expected(String),
    Resulting(String),
}

#[derive(Debug, PartialEq)]
pub struct Mismatch {
    /// The line number in the formatted version.
    pub line_number: u32,
    /// The line number in the original version.
    pub line_number_orig: u32,
    /// The set of lines (context and old/new) in the mismatch.
    pub lines: Vec<DiffLine>,
}

impl Mismatch {
    fn new(line_number: u32, line_number_orig: u32) -> Mismatch {
        Mismatch {
            line_number,
            line_number_orig,
            lines: Vec::new(),
        }
    }
}

// This struct handles writing output to stdout and abstracts away the logic
// of printing in color, if it's possible in the executing environment.
pub struct OutputWriter {
    terminal: Option<Box<term::Terminal<Output = io::Stdout>>>,
}

impl OutputWriter {
    // Create a new OutputWriter instance based on the caller's preference
    // for colorized output and the capabilities of the terminal.
    pub fn new(color: Color) -> Self {
        if let Some(t) = term::stdout() {
            if color.use_colored_tty() && t.supports_color() {
                return OutputWriter { terminal: Some(t) };
            }
        }
        OutputWriter { terminal: None }
    }

    // Write output in the optionally specified color. The output is written
    // in the specified color if this OutputWriter instance contains a
    // Terminal in its `terminal` field.
    pub fn writeln(&mut self, msg: &str, color: Option<term::color::Color>) {
        match &mut self.terminal {
            Some(ref mut t) => {
                if let Some(color) = color {
                    t.fg(color).unwrap();
                }
                writeln!(t, "{}", msg).unwrap();
                if color.is_some() {
                    t.reset().unwrap();
                }
            }
            None => println!("{}", msg),
        }
    }
}

// Produces a diff between the expected output and actual output of rustfmt.
pub fn make_diff(expected: &str, actual: &str, context_size: usize) -> Vec<Mismatch> {
    let mut line_number = 1;
    let mut line_number_orig = 1;
    let mut context_queue: VecDeque<&str> = VecDeque::with_capacity(context_size);
    let mut lines_since_mismatch = context_size + 1;
    let mut results = Vec::new();
    let mut mismatch = Mismatch::new(0, 0);

    for result in diff::lines(expected, actual) {
        match result {
            diff::Result::Left(str) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(
                        line_number - context_queue.len() as u32,
                        line_number_orig - context_queue.len() as u32,
                    );
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Resulting(str.to_owned()));
                line_number_orig += 1;
                lines_since_mismatch = 0;
            }
            diff::Result::Right(str) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(
                        line_number - context_queue.len() as u32,
                        line_number_orig - context_queue.len() as u32,
                    );
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
                } else if context_size > 0 {
                    context_queue.push_back(str);
                }

                line_number += 1;
                line_number_orig += 1;
                lines_since_mismatch += 1;
            }
        }
    }

    results.push(mismatch);
    results.remove(0);

    results
}

pub fn print_diff<F>(diff: Vec<Mismatch>, get_section_title: F, config: &Config)
where
    F: Fn(u32) -> String,
{
    let color = config.color();
    let line_terminator = if config.verbose() == Verbosity::Verbose {
        "⏎"
    } else {
        ""
    };

    let mut writer = OutputWriter::new(color);

    for mismatch in diff {
        let title = get_section_title(mismatch.line_number);
        writer.writeln(&title, None);

        for line in mismatch.lines {
            match line {
                DiffLine::Context(ref str) => {
                    writer.writeln(&format!(" {}{}", str, line_terminator), None)
                }
                DiffLine::Expected(ref str) => writer.writeln(
                    &format!("+{}{}", str, line_terminator),
                    Some(term::color::GREEN),
                ),
                DiffLine::Resulting(ref str) => writer.writeln(
                    &format!("-{}{}", str, line_terminator),
                    Some(term::color::RED),
                ),
            }
        }
    }
}

/// Convert a Mismatch into a serialised form which just includes
/// enough information to modify the original file.
/// Each section starts with a line with three integers, space separated:
///     lineno num_removed num_added
/// followed by (num_added) lines of added text.  The line numbers are
/// relative to the original file.
pub fn output_modified<W>(mut out: W, diff: Vec<Mismatch>)
where
    W: Write,
{
    for mismatch in diff {
        let (num_removed, num_added) =
            mismatch
                .lines
                .iter()
                .fold((0, 0), |(rem, add), line| match *line {
                    DiffLine::Context(_) => panic!("No Context expected"),
                    DiffLine::Expected(_) => (rem, add + 1),
                    DiffLine::Resulting(_) => (rem + 1, add),
                });
        // Write a header with enough information to separate the modified lines.
        writeln!(
            out,
            "{} {} {}",
            mismatch.line_number_orig, num_removed, num_added
        ).unwrap();

        for line in mismatch.lines {
            match line {
                DiffLine::Context(_) | DiffLine::Resulting(_) => (),
                DiffLine::Expected(ref str) => {
                    writeln!(out, "{}", str).unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::DiffLine::*;
    use super::{make_diff, Mismatch};

    #[test]
    fn diff_simple() {
        let src = "one\ntwo\nthree\nfour\nfive\n";
        let dest = "one\ntwo\ntrois\nfour\nfive\n";
        let diff = make_diff(src, dest, 1);
        assert_eq!(
            diff,
            vec![Mismatch {
                line_number: 2,
                line_number_orig: 2,
                lines: vec![
                    Context("two".to_owned()),
                    Resulting("three".to_owned()),
                    Expected("trois".to_owned()),
                    Context("four".to_owned()),
                ],
            }]
        );
    }

    #[test]
    fn diff_simple2() {
        let src = "one\ntwo\nthree\nfour\nfive\nsix\nseven\n";
        let dest = "one\ntwo\ntrois\nfour\ncinq\nsix\nseven\n";
        let diff = make_diff(src, dest, 1);
        assert_eq!(
            diff,
            vec![
                Mismatch {
                    line_number: 2,
                    line_number_orig: 2,
                    lines: vec![
                        Context("two".to_owned()),
                        Resulting("three".to_owned()),
                        Expected("trois".to_owned()),
                        Context("four".to_owned()),
                    ],
                },
                Mismatch {
                    line_number: 5,
                    line_number_orig: 5,
                    lines: vec![
                        Resulting("five".to_owned()),
                        Expected("cinq".to_owned()),
                        Context("six".to_owned()),
                    ],
                },
            ]
        );
    }

    #[test]
    fn diff_zerocontext() {
        let src = "one\ntwo\nthree\nfour\nfive\n";
        let dest = "one\ntwo\ntrois\nfour\nfive\n";
        let diff = make_diff(src, dest, 0);
        assert_eq!(
            diff,
            vec![Mismatch {
                line_number: 3,
                line_number_orig: 3,
                lines: vec![Resulting("three".to_owned()), Expected("trois".to_owned())],
            }]
        );
    }

    #[test]
    fn diff_trailing_newline() {
        let src = "one\ntwo\nthree\nfour\nfive";
        let dest = "one\ntwo\nthree\nfour\nfive\n";
        let diff = make_diff(src, dest, 1);
        assert_eq!(
            diff,
            vec![Mismatch {
                line_number: 5,
                line_number_orig: 5,
                lines: vec![Context("five".to_owned()), Expected("".to_owned())],
            }]
        );
    }
}
