use std::fmt::Display;
use std::process::ExitCode;
use std::sync::LazyLock;
use std::{env, fs};

use regex::{Regex, RegexBuilder};

mod cache;
mod config;
mod directive;

use cache::Cache;
use config::Config;
use directive::{Directive, DirectiveKind};

static LINE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(
        r#"
        ^\s*
        //@\s+
        (?P<negated>!?)
        (?P<directive>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)
        (?P<args>.*)$
    "#,
    )
    .ignore_whitespace(true)
    .build()
    .unwrap()
});

static DEPRECATED_LINE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| RegexBuilder::new(r#"//\s+@"#).build().unwrap());

struct ErrorReporter<'a> {
    /// See [`Config::template`].
    template: &'a str,
    errors: bool,
}

impl ErrorReporter<'_> {
    fn print(&mut self, msg: impl Display, lineno: usize) {
        self.errors = true;

        eprintln!("{}:{lineno}: {msg}", self.template);
    }
}

fn main() -> ExitCode {
    let Some(config @ Config { template, .. }) = &Config::parse(env::args()) else {
        return ExitCode::FAILURE;
    };

    let mut cache = Cache::new(config);
    let mut error_reporter = ErrorReporter { errors: false, template };
    let file = fs::read_to_string(template).unwrap();

    for (mut lineno, line) in file.split('\n').enumerate() {
        lineno += 1;

        if DEPRECATED_LINE_PATTERN.is_match(line) {
            error_reporter.print("Deprecated directive syntax, replace `// @` with `//@ `", lineno);

            continue;
        }

        let Some(cap) = LINE_PATTERN.captures(line) else {
            continue;
        };

        let negated = &cap["negated"] == "!";

        let args_str = &cap["args"];
        let Some(args) = shlex::split(args_str) else {
            error_reporter
                .print(&format!("Invalid arguments to shlex::split: `{args_str}`",), lineno);

            continue;
        };

        if let Some((kind, path)) = DirectiveKind::parse(&cap["directive"], negated, &args) {
            let directive = Directive { kind, lineno, path: path.to_owned() };

            if let Err(message) = directive.check(&mut cache) {
                error_reporter.print(format_args!("directive failed: {message}"), directive.lineno);
            }
        }
    }

    if error_reporter.errors { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
