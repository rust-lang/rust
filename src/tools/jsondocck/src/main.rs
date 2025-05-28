use std::process::ExitCode;
use std::sync::LazyLock;
use std::{env, fs};

use regex::{Regex, RegexBuilder};

mod cache;
mod config;
mod directive;
mod error;

use cache::Cache;
use config::parse_config;
use directive::{Command, CommandKind};
use error::CkError;

fn main() -> ExitCode {
    let config = parse_config(env::args().collect());

    let mut failed = Vec::new();
    let mut cache = Cache::new(&config);
    let Ok(commands) = get_commands(&config.template) else {
        eprintln!("Jsondocck failed for {}", &config.template);
        return ExitCode::FAILURE;
    };

    for command in commands {
        if let Err(message) = command.check(            &mut cache) {
            failed.push(CkError { command, message });
        }
    }

    if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        for i in failed {
            eprintln!("{}:{}, command failed", config.template, i.command.lineno);
            eprintln!("{}", i.message)
        }
        ExitCode::FAILURE
    }
}

static LINE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(
        r#"
        ^\s*
        //@\s+
        (?P<negated>!?)
        (?P<cmd>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)
        (?P<args>.*)$
    "#,
    )
    .ignore_whitespace(true)
    .unicode(true)
    .build()
    .unwrap()
});

static DEPRECATED_LINE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(
        r#"
        //\s+@
    "#,
    )
    .ignore_whitespace(true)
    .unicode(true)
    .build()
    .unwrap()
});

fn print_err(msg: &str, lineno: usize) {
    eprintln!("Invalid command: {} on line {}", msg, lineno)
}

/// Get a list of commands from a file.
fn get_commands(template: &str) -> Result<Vec<Command>, ()> {
    let mut commands = Vec::new();
    let mut errors = false;
    let file = fs::read_to_string(template).unwrap();

    for (lineno, line) in file.split('\n').enumerate() {
        let lineno = lineno + 1;

        if DEPRECATED_LINE_PATTERN.is_match(line) {
            print_err("Deprecated command syntax, replace `// @` with `//@ `", lineno);
            errors = true;
            continue;
        }

        let Some(cap) = LINE_PATTERN.captures(line) else {
            continue;
        };

        let negated = &cap["negated"] == "!";

        let args_str = &cap["args"];
        let Some(args) = shlex::split(args_str) else {
            print_err(&format!("Invalid arguments to shlex::split: `{args_str}`",), lineno);
            errors = true;
            continue;
        };

        if let Some((kind, path)) = CommandKind::parse(&cap["cmd"], negated, &args) {
            commands.push(Command { kind, lineno, path: path.to_owned() })
        }
    }

    if !errors { Ok(commands) } else { Err(()) }
}
