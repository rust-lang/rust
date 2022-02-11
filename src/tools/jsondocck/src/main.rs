use jsonpath_lib::select;
use once_cell::sync::Lazy;
use regex::{Regex, RegexBuilder};
use serde_json::Value;
use std::borrow::Cow;
use std::{env, fmt, fs};

mod cache;
mod config;
mod error;

use cache::Cache;
use config::parse_config;
use error::CkError;

fn main() -> Result<(), String> {
    let config = parse_config(env::args().collect());

    let mut failed = Vec::new();
    let mut cache = Cache::new(&config.doc_dir);
    let commands = get_commands(&config.template)
        .map_err(|_| format!("Jsondocck failed for {}", &config.template))?;

    for command in commands {
        if let Err(e) = check_command(command, &mut cache) {
            failed.push(e);
        }
    }

    if failed.is_empty() {
        Ok(())
    } else {
        for i in failed {
            eprintln!("{}", i);
        }
        Err(format!("Jsondocck failed for {}", &config.template))
    }
}

#[derive(Debug)]
pub struct Command {
    negated: bool,
    kind: CommandKind,
    args: Vec<String>,
    lineno: usize,
}

#[derive(Debug)]
pub enum CommandKind {
    Has,
    Count,
    Is,
    Set,
}

impl CommandKind {
    fn validate(&self, args: &[String], command_num: usize, lineno: usize) -> bool {
        let count = match self {
            CommandKind::Has => (1..=3).contains(&args.len()),
            CommandKind::Count | CommandKind::Is => 3 == args.len(),
            CommandKind::Set => 4 == args.len(),
        };

        if !count {
            print_err(&format!("Incorrect number of arguments to `@{}`", self), lineno);
            return false;
        }

        if args[0] == "-" && command_num == 0 {
            print_err(&format!("Tried to use the previous path in the first command"), lineno);
            return false;
        }

        if let CommandKind::Count = self {
            if args[2].parse::<usize>().is_err() {
                print_err(&format!("Third argument to @count must be a valid usize"), lineno);
                return false;
            }
        }

        true
    }
}

impl fmt::Display for CommandKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            CommandKind::Has => "has",
            CommandKind::Count => "count",
            CommandKind::Is => "is",
            CommandKind::Set => "set",
        };
        write!(f, "{}", text)
    }
}

static LINE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    RegexBuilder::new(
        r#"
        \s(?P<invalid>!?)@(?P<negated>!?)
        (?P<cmd>[A-Za-z]+(?:-[A-Za-z]+)*)
        (?P<args>.*)$
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

/// Get a list of commands from a file. Does the work of ensuring the commands
/// are syntactically valid.
fn get_commands(template: &str) -> Result<Vec<Command>, ()> {
    let mut commands = Vec::new();
    let mut errors = false;
    let file = fs::read_to_string(template).unwrap();

    for (lineno, line) in file.split('\n').enumerate() {
        let lineno = lineno + 1;

        let cap = match LINE_PATTERN.captures(line) {
            Some(c) => c,
            None => continue,
        };

        let negated = cap.name("negated").unwrap().as_str() == "!";
        let cmd = cap.name("cmd").unwrap().as_str();

        let cmd = match cmd {
            "has" => CommandKind::Has,
            "count" => CommandKind::Count,
            "is" => CommandKind::Is,
            "set" => CommandKind::Set,
            _ => {
                print_err(&format!("Unrecognized command name `@{}`", cmd), lineno);
                errors = true;
                continue;
            }
        };

        if let Some(m) = cap.name("invalid") {
            if m.as_str() == "!" {
                print_err(
                    &format!(
                        "`!@{0}{1}`, (help: try with `@!{1}`)",
                        if negated { "!" } else { "" },
                        cmd,
                    ),
                    lineno,
                );
                errors = true;
                continue;
            }
        }

        let args = cap.name("args").map_or(Some(vec![]), |m| shlex::split(m.as_str()));

        let args = match args {
            Some(args) => args,
            None => {
                print_err(
                    &format!(
                        "Invalid arguments to shlex::split: `{}`",
                        cap.name("args").unwrap().as_str()
                    ),
                    lineno,
                );
                errors = true;
                continue;
            }
        };

        if !cmd.validate(&args, commands.len(), lineno) {
            errors = true;
            continue;
        }

        commands.push(Command { negated, kind: cmd, args, lineno })
    }

    if !errors { Ok(commands) } else { Err(()) }
}

/// Performs the actual work of ensuring a command passes. Generally assumes the command
/// is syntactically valid.
fn check_command(command: Command, cache: &mut Cache) -> Result<(), CkError> {
    // FIXME: Be more granular about why, (e.g. syntax error, count not equal)
    let result = match command.kind {
        CommandKind::Has => {
            match command.args.len() {
                // @has <path> = file existence
                1 => cache.get_file(&command.args[0]).is_ok(),
                // @has <path> <jsonpath> = check path exists
                2 => {
                    let val = cache.get_value(&command.args[0])?;
                    let results = select(&val, &command.args[1]).unwrap();
                    !results.is_empty()
                }
                // @has <path> <jsonpath> <value> = check *any* item matched by path equals value
                3 => {
                    let val = cache.get_value(&command.args[0])?;
                    let results = select(&val, &command.args[1]).unwrap();
                    let pat = string_to_value(&command.args[2], cache);
                    let has = results.contains(&pat.as_ref());
                    // Give better error for when @has check fails
                    if !command.negated && !has {
                        return Err(CkError::FailedCheck(
                            format!(
                                "{} matched to {:?} but didn't have {:?}",
                                &command.args[1],
                                results,
                                pat.as_ref()
                            ),
                            command,
                        ));
                    } else {
                        has
                    }
                }
                _ => unreachable!(),
            }
        }
        CommandKind::Count => {
            // @count <path> <jsonpath> <count> = Check that the jsonpath matches exactly [count] times
            assert_eq!(command.args.len(), 3);
            let expected: usize = command.args[2].parse().unwrap();

            let val = cache.get_value(&command.args[0])?;
            let results = select(&val, &command.args[1]).unwrap();
            let eq = results.len() == expected;
            if !command.negated && !eq {
                return Err(CkError::FailedCheck(
                    format!(
                        "`{}` matched to `{:?}` with length {}, but expected length {}",
                        &command.args[1],
                        results,
                        results.len(),
                        expected
                    ),
                    command,
                ));
            } else {
                eq
            }
        }
        CommandKind::Is => {
            // @has <path> <jsonpath> <value> = check *exactly one* item matched by path, and it equals value
            assert_eq!(command.args.len(), 3);
            let val = cache.get_value(&command.args[0])?;
            let results = select(&val, &command.args[1]).unwrap();
            let pat = string_to_value(&command.args[2], cache);
            let is = results.len() == 1 && results[0] == pat.as_ref();
            if !command.negated && !is {
                return Err(CkError::FailedCheck(
                    format!(
                        "{} matched to {:?}, but expected {:?}",
                        &command.args[1],
                        results,
                        pat.as_ref()
                    ),
                    command,
                ));
            } else {
                is
            }
        }
        CommandKind::Set => {
            // @set <name> = <path> <jsonpath>
            assert_eq!(command.args.len(), 4);
            assert_eq!(command.args[1], "=", "Expected an `=`");
            let val = cache.get_value(&command.args[2])?;
            let results = select(&val, &command.args[3]).unwrap();
            assert_eq!(
                results.len(),
                1,
                "Didn't get 1 result for `{}`: got {:?}",
                command.args[3],
                results
            );
            match results.len() {
                0 => false,
                1 => {
                    let r = cache.variables.insert(command.args[0].clone(), results[0].clone());
                    assert!(r.is_none(), "Name collision: {} is duplicated", command.args[0]);
                    true
                }
                _ => {
                    panic!(
                        "Got multiple results in `@set` for `{}`: {:?}",
                        &command.args[3], results
                    );
                }
            }
        }
    };

    if result == command.negated {
        if command.negated {
            Err(CkError::FailedCheck(
                format!(
                    "`@!{} {}` matched when it shouldn't",
                    command.kind,
                    command.args.join(" ")
                ),
                command,
            ))
        } else {
            // FIXME: In the future, try 'peeling back' each step, and see at what level the match failed
            Err(CkError::FailedCheck(
                format!(
                    "`@{} {}` didn't match when it should",
                    command.kind,
                    command.args.join(" ")
                ),
                command,
            ))
        }
    } else {
        Ok(())
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
