//! HtmlDocCk is a test framework for rustdoc's HTML backend.
use std::process::ExitCode;

use error::Source;
use regex::Regex;

mod cache;
mod channel;
mod check;
mod config;
mod error;
mod parse;

fn main() -> ExitCode {
    let result = error::DiagCtxt::scope(|dcx| {
        let args: Vec<_> = std::env::args().collect();
        let config = config::Config::parse(&args, dcx)?;

        // FIXME: better error message
        let template = std::fs::read_to_string(&config.template)
            .map_err(|error| dcx.emit(&format!("failed to read file: {error}"), None, None))?;

        let commands = parse::commands(&template, dcx);

        let mut cache = cache::Cache::new(&config.doc_dir);
        commands.into_iter().try_for_each(|command| command.check(&mut cache, dcx))
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}

/// A check command.
struct Command<'src> {
    kind: CommandKind,
    negated: bool,
    source: Source<'src>,
}

/// The kind of check command.
enum CommandKind {
    /// `@has <PATH>`.
    HasFile { path: String },
    /// `@has-dir <PATH>`.
    HasDir { path: String },
    /// `@has <PATH> <XPATH> <TEXT>`.
    Has { path: String, xpath: String, text: String },
    /// `@hasraw <PATH> <TEXT>`.
    HasRaw { path: String, text: String },
    /// `@matches <PATH> <XPATH> <PATTERN>`.
    Matches { path: String, xpath: String, pattern: Regex },
    /// `@matchesraw <PATH> <PATTERN>`.
    MatchesRaw { path: String, pattern: Regex },
    /// `@count <PATH> <XPATH> [<TEXT>] <COUNT>`.
    Count { path: String, xpath: String, text: Option<String>, count: u32 },
    /// `@files <PATH> <ARRAY>`.
    Files { path: String, files: String },
    /// `@snapshot <NAME> <PATH> <XPATH>`.
    Snapshot { name: String, path: String, xpath: String },
}

impl CommandKind {
    /// Whether this kind of command may be negated with `!`.
    fn may_be_negated(&self) -> bool {
        // We match exhaustively to get a compile error if we add a new kind of command.
        match self {
            Self::Has { .. }
            | Self::HasFile { .. }
            | Self::HasDir { .. }
            | Self::HasRaw { .. }
            | Self::Matches { .. }
            | Self::MatchesRaw { .. }
            | Self::Count { .. }
            | Self::Snapshot { .. } => true,
            Self::Files { .. } => false,
        }
    }
}
