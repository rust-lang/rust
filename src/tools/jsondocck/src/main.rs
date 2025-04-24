use std::borrow::Cow;
use std::process::ExitCode;
use std::sync::LazyLock;
use std::{env, fs};

use regex::{Regex, RegexBuilder};
use serde_json::Value;

mod cache;
mod config;
mod error;

use cache::Cache;
use config::parse_config;
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
        if let Err(message) = check_command(&command, &mut cache) {
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

#[derive(Debug)]
pub struct Command {
    kind: CommandKind,
    path: String,
    lineno: usize,
}

#[derive(Debug)]
enum CommandKind {
    /// `//@ has <path>`
    ///
    /// Checks the path exists.
    HasPath,

    /// `//@ has <path> <value>`
    ///
    /// Check one thing at the path  is equal to the value.
    HasValue { value: String },

    /// `//@ !has <path>`
    ///
    /// Checks the path doesn't exist.
    HasNotPath,

    /// `//@ !has <path> <value>`
    ///
    /// Checks the path exists, but doesn't have the given value.
    HasNotValue { value: String },

    /// `//@ is <path> <value>`
    ///
    /// Check the path is the given value.
    Is { value: String },

    /// `//@ is <path> <value> <value>...`
    ///
    /// Check that the path matches to exactly every given value.
    IsMany { values: Vec<String> },

    /// `//@ !is <path> <value>`
    ///
    /// Check the path isn't the given value.
    IsNot { value: String },

    /// `//@ count <path> <value>`
    ///
    /// Check the path has the expected number of matches.
    CountIs { expected: usize },

    /// `//@ set <name> = <path>`
    Set { variable: String },
}

impl CommandKind {
    /// Returns both the kind and the path.
    ///
    /// Returns `None` if the command isn't from jsondocck (e.g. from compiletest).
    fn parse<'a>(command_name: &str, negated: bool, args: &'a [String]) -> Option<(Self, &'a str)> {
        let kind = match (command_name, negated) {
            ("count", false) => {
                assert_eq!(args.len(), 2);
                let expected = args[1].parse().expect("invalid number for `count`");
                Self::CountIs { expected }
            }

            ("ismany", false) => {
                // FIXME: Make this >= 3, and migrate len(values)==1 cases to @is
                assert!(args.len() >= 2, "Not enough args to `ismany`");
                let values = args[1..].to_owned();
                Self::IsMany { values }
            }

            ("is", false) => {
                assert_eq!(args.len(), 2);
                Self::Is { value: args[1].clone() }
            }
            ("is", true) => {
                assert_eq!(args.len(), 2);
                Self::IsNot { value: args[1].clone() }
            }

            ("set", false) => {
                assert_eq!(args.len(), 3);
                assert_eq!(args[1], "=");
                return Some((Self::Set { variable: args[0].clone() }, &args[2]));
            }

            ("has", false) => match args {
                [_path] => Self::HasPath,
                [_path, value] => Self::HasValue { value: value.clone() },
                _ => panic!("`//@ has` must have 2 or 3 arguments, but got {args:?}"),
            },
            ("has", true) => match args {
                [_path] => Self::HasNotPath,
                [_path, value] => Self::HasNotValue { value: value.clone() },
                _ => panic!("`//@ !has` must have 2 or 3 arguments, but got {args:?}"),
            },

            (_, false) if KNOWN_DIRECTIVE_NAMES.contains(&command_name) => {
                return None;
            }
            _ => {
                panic!("Invalid command `//@ {}{command_name}`", if negated { "!" } else { "" })
            }
        };

        Some((kind, &args[0]))
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

// FIXME: This setup is temporary until we figure out how to improve this situation.
//        See <https://github.com/rust-lang/rust/issues/125813#issuecomment-2141953780>.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../compiletest/src/directive-list.rs"));

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

/// Performs the actual work of ensuring a command passes.
fn check_command(command: &Command, cache: &mut Cache) -> Result<(), String> {
    let matches = cache.select(&command.path);
    match &command.kind {
        CommandKind::HasPath => {
            if matches.is_empty() {
                return Err("matched to no values".to_owned());
            }
        }
        CommandKind::HasNotPath => {
            if !matches.is_empty() {
                return Err(format!("matched to {matches:?}, but wanted no matches"));
            }
        }
        CommandKind::HasValue { value } => {
            let want_value = string_to_value(value, cache);
            if !matches.contains(&want_value.as_ref()) {
                return Err(format!("matched to {matches:?}, which didn't contain {want_value:?}"));
            }
        }
        CommandKind::HasNotValue { value } => {
            let wantnt_value = string_to_value(value, cache);
            if matches.contains(&wantnt_value.as_ref()) {
                return Err(format!(
                    "matched to {matches:?}, which contains unwanted {wantnt_value:?}"
                ));
            } else if matches.is_empty() {
                return Err(format!(
                    "got no matches, but expected some matched (not containing {wantnt_value:?}"
                ));
            }
        }

        CommandKind::Is { value } => {
            let want_value = string_to_value(value, cache);
            let matched = get_one(&matches)?;
            if matched != want_value.as_ref() {
                return Err(format!("matched to {matched:?} but want {want_value:?}"));
            }
        }
        CommandKind::IsNot { value } => {
            let wantnt_value = string_to_value(value, cache);
            let matched = get_one(&matches)?;
            if matched == wantnt_value.as_ref() {
                return Err(format!("got value {wantnt_value:?}, but want anything else"));
            }
        }

        CommandKind::IsMany { values } => {
            // Serde json doesn't implement Ord or Hash for Value, so we must
            // use a Vec here. While in theory that makes setwize equality
            // O(n^2), in practice n will never be large enough to matter.
            let expected_values =
                values.iter().map(|v| string_to_value(v, cache)).collect::<Vec<_>>();
            if expected_values.len() != matches.len() {
                return Err(format!(
                    "Expected {} values, but matched to {} values ({:?})",
                    expected_values.len(),
                    matches.len(),
                    matches
                ));
            };
            for got_value in matches {
                if !expected_values.iter().any(|exp| &**exp == got_value) {
                    return Err(format!("has match {got_value:?}, which was not expected",));
                }
            }
        }
        CommandKind::CountIs { expected } => {
            if *expected != matches.len() {
                return Err(format!(
                    "matched to `{matches:?}` with length {}, but expected length {expected}",
                    matches.len(),
                ));
            }
        }
        CommandKind::Set { variable } => {
            let value = get_one(&matches)?;
            let r = cache.variables.insert(variable.to_owned(), value.clone());
            assert!(r.is_none(), "name collision: {variable:?} is duplicated");
        }
    }

    Ok(())
}

fn get_one<'a>(matches: &[&'a Value]) -> Result<&'a Value, String> {
    match matches {
        [] => Err("matched to no values".to_owned()),
        [matched] => Ok(matched),
        _ => Err(format!("matched to multiple values {matches:?}, but want exactly 1")),
    }
}

fn string_to_value<'a>(s: &str, cache: &'a Cache) -> Cow<'a, Value> {
    if s.starts_with("$") {
        Cow::Borrowed(&cache.variables.get(&s[1..]).unwrap_or_else(|| {
            // FIXME(adotinthevoid): Show line number
            panic!("No variable: `{}`. Current state: `{:?}`", &s[1..], cache.variables)
        }))
    } else {
        Cow::Owned(serde_json::from_str(s).expect(&format!("Cannot convert `{}` to json", s)))
    }
}
