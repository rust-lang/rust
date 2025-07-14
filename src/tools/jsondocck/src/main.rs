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
use directive::{Directive, DirectiveKind};
use error::CkError;

fn main() -> ExitCode {
    let config = parse_config(env::args().collect());

    let mut failed = Vec::new();
    let mut cache = Cache::new(&config);
    let Ok(directives) = get_directives(&config.template) else {
        eprintln!("Jsondocck failed for {}", &config.template);
        return ExitCode::FAILURE;
    };

    for directive in directives {
        if let Err(message) = directive.check(&mut cache) {
            failed.push(CkError { directive, message });
        }
    }

    if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        for i in failed {
            eprintln!("{}:{}, directive failed", config.template, i.directive.lineno);
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
        (?P<directive>.+?)
        (?:[\s:](?P<args>.*))?$
    "#,
    )
    .ignore_whitespace(true)
    .unicode(true)
    .build()
    .unwrap()
});

static DEPRECATED_LINE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(r"//\s+@").ignore_whitespace(true).unicode(true).build().unwrap()
});

fn print_err(msg: &str, lineno: usize) {
    eprintln!("Invalid directive: {} on line {}", msg, lineno)
}

/// Get a list of directives from a file.
fn get_directives(template: &str) -> Result<Vec<Directive>, ()> {
    let mut directives = Vec::new();
    let mut errors = false;
    let file = fs::read_to_string(template).unwrap();

    for (lineno, line) in file.split('\n').enumerate() {
        let lineno = lineno + 1;

        if DEPRECATED_LINE_PATTERN.is_match(line) {
            print_err("Deprecated directive syntax, replace `// @` with `//@ `", lineno);
            errors = true;
            continue;
        }

        let Some(cap) = LINE_PATTERN.captures(line) else {
            continue;
        };

        let negated = &cap["negated"] == "!";

        let args_str = cap.name("args").map(|m| m.as_str()).unwrap_or_default();
        let Some(args) = shlex::split(args_str) else {
            print_err(&format!("Invalid arguments to shlex::split: `{args_str}`",), lineno);
            errors = true;
            continue;
        };

        if let Some((kind, path)) = DirectiveKind::parse(&cap["directive"], negated, &args) {
            directives.push(Directive { kind, lineno, path: path.to_owned() })
        }
    }

    if !errors { Ok(directives) } else { Err(()) }
}
