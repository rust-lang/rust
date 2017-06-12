use std::collections::VecDeque;
use diff;
use term;
use std::io;

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
where
    F: Fn(u32) -> String,
{
    match term::stdout() {
        Some(ref t) if isatty() && t.supports_color() => {
            print_diff_fancy(diff, get_section_title, term::stdout().unwrap())
        }
        _ => print_diff_basic(diff, get_section_title),
    }

    // isatty shamelessly adapted from cargo.
    #[cfg(unix)]
    fn isatty() -> bool {
        extern crate libc;

        unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 }
    }
    #[cfg(windows)]
    fn isatty() -> bool {
        extern crate kernel32;
        extern crate winapi;

        unsafe {
            let handle = kernel32::GetStdHandle(winapi::winbase::STD_OUTPUT_HANDLE);
            let mut out = 0;
            kernel32::GetConsoleMode(handle, &mut out) != 0
        }
    }
}

fn print_diff_fancy<F>(
    diff: Vec<Mismatch>,
    get_section_title: F,
    mut t: Box<term::Terminal<Output = io::Stdout>>,
) where
    F: Fn(u32) -> String,
{
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

pub fn print_diff_basic<F>(diff: Vec<Mismatch>, get_section_title: F)
where
    F: Fn(u32) -> String,
{
    for mismatch in diff {
        let title = get_section_title(mismatch.line_number);
        println!("{}", title);

        for line in mismatch.lines {
            match line {
                DiffLine::Context(ref str) => {
                    println!(" {}⏎", str);
                }
                DiffLine::Expected(ref str) => {
                    println!("+{}⏎", str);
                }
                DiffLine::Resulting(ref str) => {
                    println!("-{}⏎", str);
                }
            }
        }
    }
}
