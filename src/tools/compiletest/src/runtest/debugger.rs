use crate::common::Config;
use crate::runtest::ProcRes;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub(super) struct DebuggerCommands {
    pub commands: Vec<String>,
    pub check_lines: Vec<String>,
    pub breakpoint_lines: Vec<usize>,
}

impl DebuggerCommands {
    pub(super) fn parse_from(
        file: &Path,
        config: &Config,
        debugger_prefixes: &[&str],
    ) -> Result<Self, String> {
        let directives = debugger_prefixes
            .iter()
            .map(|prefix| (format!("{}-command", prefix), format!("{}-check", prefix)))
            .collect::<Vec<_>>();

        let mut breakpoint_lines = vec![];
        let mut commands = vec![];
        let mut check_lines = vec![];
        let mut counter = 1;
        let reader = BufReader::new(File::open(file).unwrap());
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    let line =
                        if line.starts_with("//") { line[2..].trim_start() } else { line.as_str() };

                    if line.contains("#break") {
                        breakpoint_lines.push(counter);
                    }

                    for &(ref command_directive, ref check_directive) in &directives {
                        config
                            .parse_name_value_directive(&line, command_directive)
                            .map(|cmd| commands.push(cmd));

                        config
                            .parse_name_value_directive(&line, check_directive)
                            .map(|cmd| check_lines.push(cmd));
                    }
                }
                Err(e) => return Err(format!("Error while parsing debugger commands: {}", e)),
            }
            counter += 1;
        }

        Ok(Self { commands, check_lines, breakpoint_lines })
    }
}

pub(super) fn check_debugger_output(
    debugger_run_result: &ProcRes,
    check_lines: &[String],
) -> Result<(), String> {
    let num_check_lines = check_lines.len();

    let mut check_line_index = 0;
    for line in debugger_run_result.stdout.lines() {
        if check_line_index >= num_check_lines {
            break;
        }

        if check_single_line(line, &(check_lines[check_line_index])[..]) {
            check_line_index += 1;
        }
    }
    if check_line_index != num_check_lines && num_check_lines > 0 {
        Err(format!("line not found in debugger output: {}", check_lines[check_line_index]))
    } else {
        Ok(())
    }
}

fn check_single_line(line: &str, check_line: &str) -> bool {
    // Allow check lines to leave parts unspecified (e.g., uninitialized
    // bits in the  wrong case of an enum) with the notation "[...]".
    let line = line.trim();
    let check_line = check_line.trim();
    let can_start_anywhere = check_line.starts_with("[...]");
    let can_end_anywhere = check_line.ends_with("[...]");

    let check_fragments: Vec<&str> =
        check_line.split("[...]").filter(|frag| !frag.is_empty()).collect();
    if check_fragments.is_empty() {
        return true;
    }

    let (mut rest, first_fragment) = if can_start_anywhere {
        match line.find(check_fragments[0]) {
            Some(pos) => (&line[pos + check_fragments[0].len()..], 1),
            None => return false,
        }
    } else {
        (line, 0)
    };

    for current_fragment in &check_fragments[first_fragment..] {
        match rest.find(current_fragment) {
            Some(pos) => {
                rest = &rest[pos + current_fragment.len()..];
            }
            None => return false,
        }
    }

    if !can_end_anywhere && !rest.is_empty() { false } else { true }
}
