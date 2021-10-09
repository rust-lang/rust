use crate::common::Config;

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
