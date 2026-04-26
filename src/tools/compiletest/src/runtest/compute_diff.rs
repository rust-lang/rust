use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub(crate) enum DiffLine {
    Context(String),
    Expected(String),
    Resulting(String),
}

#[derive(Debug, PartialEq)]
pub(crate) struct Mismatch {
    pub(crate) line_number: u32,
    pub(crate) lines: Vec<DiffLine>,
}

impl Mismatch {
    fn new(line_number: u32) -> Mismatch {
        Mismatch { line_number, lines: Vec::new() }
    }
}

// Produces a diff between the expected output and actual output.
pub(crate) fn make_diff(expected: &str, actual: &str, context_size: usize) -> Vec<Mismatch> {
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

pub(crate) fn diff_by_lines(expected: &str, actual: &str) -> String {
    use std::collections::HashMap;
    use std::fmt::Write;
    let mut output = String::new();
    let mut expected_counts: HashMap<&str, usize> = HashMap::new();
    let mut actual_counts: HashMap<&str, usize> = HashMap::new();

    for line in expected.lines() {
        *expected_counts.entry(line).or_insert(0) += 1;
    }
    for line in actual.lines() {
        *actual_counts.entry(line).or_insert(0) += 1;
    }

    fn write_expected_only_lines(
        output: &mut String,
        expected_lines: &HashMap<&str, usize>,
        actual_lines: &HashMap<&str, usize>,
    ) {
        let mut expected_only: Vec<(&str, usize)> = expected_lines
            .iter()
            .filter_map(|(&line, &expected_count)| {
                let actual_count = actual_lines.get(line).copied().unwrap_or(0);
                if expected_count > actual_count {
                    Some((line, expected_count - actual_count))
                } else {
                    None
                }
            })
            .collect();
        expected_only.sort_by(|(a, _), (b, _)| a.cmp(b));

        if expected_only.is_empty() {
            writeln!(output, "(no lines found)").unwrap();
        } else {
            for (line, diff) in expected_only {
                for _ in 0..diff {
                    writeln!(output, "{line}").unwrap();
                }
            }
        }
    }

    writeln!(output, "Compare output by lines enabled, diff by lines:").unwrap();
    writeln!(output, "Expected contains these lines that are not in actual:").unwrap();
    write_expected_only_lines(&mut output, &expected_counts, &actual_counts);
    writeln!(output, "Actual contains these lines that are not in expected:").unwrap();
    write_expected_only_lines(&mut output, &actual_counts, &expected_counts);
    output
}
