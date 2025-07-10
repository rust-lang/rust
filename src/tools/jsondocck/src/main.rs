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

use crate::directive::Directive;

static LINE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(
        r"
            # Any number of whitespaces.
            \s*
            # The directive prefix (`//@`) and 1 or more whitespaces after.
            //@\s+
            # The directive itself (1 or more word or `-` characters).
            (?P<directive>[\w-]+)
            # The optional remainder (1 non-word character and 0 or more of any characters after).
            (?P<args>\W.*)?
        ",
    )
    .ignore_whitespace(true)
    .build()
    .unwrap()
});

static DEPRECATED_LINE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| RegexBuilder::new(r"//\s+@").build().unwrap());

/// ```
/// // Directive on its own line
/// //@ correct-directive
///
/// // Directive on a line after code
/// struct S; //@ ignored-directive
/// ```
static MIXED_LINE: LazyLock<Regex> =
    LazyLock::new(|| RegexBuilder::new(r".*\S.*//@").build().unwrap());

struct ErrorReporter<'a> {
    /// See [`Config::template`].
    template: &'a str,
    errors: bool,
}

impl ErrorReporter<'_> {
    fn print(&mut self, lineno: usize, msg: impl Display) {
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
            error_reporter.print(lineno, "deprecated directive syntax, replace `// @` with `//@ `");

            continue;
        }

        if MIXED_LINE.is_match(line) {
            error_reporter.print(
                lineno,
                "directives must be on their own line, directives after code are ignored",
            );

            continue;
        }

        let Some(cap) = LINE_PATTERN.captures(line) else {
            continue;
        };

        let mut args = cap.name("args").map(|m| m.as_str()).unwrap_or_default();

        let directive = match Directive::parse(&cap["directive"], &mut args) {
            Ok(Some(directive)) => directive,
            Ok(None) => continue,
            Err(message) => {
                error_reporter
                    .print(lineno, format_args!("failed to parse a directive: {message}"));

                continue;
            }
        };

        if let Err(message) = directive.process(&mut cache, args) {
            error_reporter.print(lineno, format_args!("directive failed: {message}"));
        }
    }

    if error_reporter.errors { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
