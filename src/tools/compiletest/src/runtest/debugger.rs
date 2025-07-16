use std::fmt::Write;
use std::fs::File;
use std::io::{BufRead, BufReader};

use camino::{Utf8Path, Utf8PathBuf};

use crate::common::Config;
use crate::runtest::ProcRes;

/// Representation of information to invoke a debugger and check its output
pub(super) struct DebuggerCommands {
    /// Commands for the debuuger
    pub commands: Vec<String>,
    /// Lines to insert breakpoints at
    pub breakpoint_lines: Vec<usize>,
    /// Contains the source line number to check and the line itself
    check_lines: Vec<(usize, String)>,
    /// Source file name
    file: Utf8PathBuf,
}

impl DebuggerCommands {
    pub fn parse_from(
        file: &Utf8Path,
        config: &Config,
        debugger_prefix: &str,
    ) -> Result<Self, String> {
        let command_directive = format!("{debugger_prefix}-command");
        let check_directive = format!("{debugger_prefix}-check");

        let mut breakpoint_lines = vec![];
        let mut commands = vec![];
        let mut check_lines = vec![];
        let mut counter = 0;
        let reader = BufReader::new(File::open(file.as_std_path()).unwrap());
        for (line_no, line) in reader.lines().enumerate() {
            counter += 1;
            let line = line.map_err(|e| format!("Error while parsing debugger commands: {}", e))?;

            // Breakpoints appear on lines with actual code, typically at the end of the line.
            if line.contains("#break") {
                breakpoint_lines.push(counter);
                continue;
            }

            let Some(line) = line.trim_start().strip_prefix("//").map(str::trim_start) else {
                continue;
            };

            if let Some(command) =
                config.parse_name_value_directive(&line, &command_directive, file, line_no)
            {
                commands.push(command);
            }
            if let Some(pattern) =
                config.parse_name_value_directive(&line, &check_directive, file, line_no)
            {
                check_lines.push((line_no, pattern));
            }
        }

        Ok(Self { commands, breakpoint_lines, check_lines, file: file.to_path_buf() })
    }

    /// Given debugger output and lines to check, ensure that every line is
    /// contained in the debugger output. The check lines need to be found in
    /// order, but there can be extra lines between.
    pub fn check_output(&self, debugger_run_result: &ProcRes) -> Result<(), String> {
        // (src_lineno, ck_line)  that we did find
        let mut found = vec![];
        // (src_lineno, ck_line) that we couldn't find
        let mut missing = vec![];
        //  We can find our any current match anywhere after our last match
        let mut last_idx = 0;
        let dbg_lines: Vec<&str> = debugger_run_result.stdout.lines().collect();

        for (src_lineno, ck_line) in &self.check_lines {
            if let Some(offset) = dbg_lines
                .iter()
                .skip(last_idx)
                .position(|out_line| check_single_line(out_line, &ck_line))
            {
                last_idx += offset;
                found.push((src_lineno, dbg_lines[last_idx]));
            } else {
                missing.push((src_lineno, ck_line));
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            let fname = self.file.file_name().unwrap();
            let mut msg = format!(
                "check directive(s) from `{}` not found in debugger output. errors:",
                self.file
            );

            for (src_lineno, err_line) in missing {
                write!(msg, "\n    ({fname}:{num}) `{err_line}`", num = src_lineno + 1).unwrap();
            }

            if !found.is_empty() {
                let init = "\nthe following subset of check directive(s) was found successfully:";
                msg.push_str(init);
                for (src_lineno, found_line) in found {
                    write!(msg, "\n    ({fname}:{num}) `{found_line}`", num = src_lineno + 1)
                        .unwrap();
                }
            }

            Err(msg)
        }
    }
}

/// Check that the pattern in `check_line` applies to `line`. Returns `true` if they do match.
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
        let Some(pos) = line.find(check_fragments[0]) else {
            return false;
        };
        (&line[pos + check_fragments[0].len()..], 1)
    } else {
        (line, 0)
    };

    for current_fragment in &check_fragments[first_fragment..] {
        let Some(pos) = rest.find(current_fragment) else {
            return false;
        };
        rest = &rest[pos + current_fragment.len()..];
    }

    can_end_anywhere || rest.is_empty()
}
