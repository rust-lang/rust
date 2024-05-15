use std::sync::OnceLock;

use regex::Regex;

use crate::error::{DiagCtxt, Source};
use crate::{channel, Command, CommandKind};

/// Parse all commands inside of the given template.
// FIXME: Add comment that this doesn't conflict with the ui_test-style compiletest directives
pub(crate) fn commands<'src>(template: &'src str, dcx: &mut DiagCtxt) -> Vec<Command<'src>> {
    // FIXME: Add comment that we do not respect Rust syntax for simplicity of implementation.

    // FIXME: port behavior of "concat_multi_lines(template)"
    // FIXME: or `.split('\n')`?
    template
        .lines()
        .enumerate()
        .filter_map(|(index, line)| Command::parse(line, index + 1, dcx).ok())
        .collect()
}

impl<'src> Command<'src> {
    fn parse(line: &'src str, lineno: usize, dcx: &mut DiagCtxt) -> Result<Self, ()> {
        let captures = command_regex().captures(line).ok_or(())?;

        // FIXME: more accurate range
        let source = Source { line, lineno, range: 0..line.len() };

        let args = captures.name(group::ARGUMENTS).unwrap();
        let args = shlex::split(args.as_str()).ok_or_else(|| {
            // Unfortunately, `shlex` doesn't provide us with the precise cause of failure.
            // Nor does it provide the location of the erroneous string it encountered.
            // Therefore we can't easily reconstruct this piece of information ourselves and
            // we have no option but to emit a vague error for an imprecise location.
            dcx.emit(
                "command arguments are not properly terminated or escaped",
                Source { line, lineno, range: args.range() },
                None,
            );
        })?;

        let name = captures.name(group::NAME).unwrap();
        let kind = CommandKind::parse(name, &args, source.clone(), dcx)?;

        let negated = if let Some(negation) = captures.name(group::NEGATION) {
            if !kind.may_be_negated() {
                dcx.emit(
                    &format!("command `{}` may not be negated", name.as_str()),
                    Source { line, lineno, range: negation.range() },
                    "remove the `!`",
                );
                return Err(());
            }
            true
        } else {
            false
        };

        if let Some(misplaced_negation) = captures.name(group::NEGATION_MISPLACED) {
            // FIXME: better message
            dcx.emit(
                "misplaced negation `!`",
                Source { line, lineno, range: misplaced_negation.range() },
                if negated && kind.may_be_negated() {
                    "move the `!` after the `@`"
                } else {
                    // FIXME: more context
                    "remove the `!`"
                },
            );
            return Err(());
        }

        Ok(Self { kind, negated, source })
    }
}

impl CommandKind {
    // FIXME: improve signature
    fn parse(
        name: regex::Match<'_>,
        args: &[String],
        source: Source<'_>,
        dcx: &mut DiagCtxt,
    ) -> Result<Self, ()> {
        // FIXME: heavily improve this diagnostic
        let mut wrong_arity = |expected: &str| {
            dcx.emit(
                "incorrect number of arguments provided",
                source.clone(),
                format!("got {} but expected {expected}", args.len()).as_str(),
            );
        };

        // FIXME: avoid cloning by try_into'ing the args into arrays and moving the Strings
        // or by draining the Vec & using Iterator::next
        // FIXME: Add comment "unfortunately, `shlex` doesn't yield slices, only owned stuff"
        // FIXME: parse `XPath`s here and provide beautiful errs with location info
        match name.as_str() {
            "has" => match args {
                [path] => Ok(Self::HasFile { path: path.clone() }),
                [path, xpath, text] => {
                    Ok(Self::Has { path: path.clone(), xpath: xpath.clone(), text: text.clone() })
                }
                _ => Err(wrong_arity("1 or 3")),
            },
            "hasraw" => match args {
                [path, text] => Ok(Self::HasRaw { path: path.clone(), text: text.clone() }),
                _ => Err(wrong_arity("2")),
            },
            "matches" => match args {
                [path, xpath, pattern] => Ok(Self::Matches {
                    path: path.clone(),
                    xpath: xpath.clone(),
                    pattern: parse_regex(pattern, source.clone(), dcx)?,
                }),
                _ => Err(wrong_arity("3")),
            },
            "matchesraw" => match args {
                [path, pattern] => Ok(Self::MatchesRaw {
                    path: path.clone(),
                    pattern: parse_regex(pattern, source.clone(), dcx)?,
                }),
                _ => Err(wrong_arity("2")),
            },
            "files" => match args {
                [path, files] => Ok(Self::Files { path: path.clone(), files: files.clone() }),
                _ => Err(wrong_arity("2")),
            },
            "count" => match args {
                [path, xpath, count] => Ok(Self::Count {
                    path: path.clone(),
                    xpath: xpath.clone(),
                    text: None,
                    count: parse_count(count, source.clone(), dcx)?,
                }),
                [path, xpath, text, count] => Ok(Self::Count {
                    path: path.clone(),
                    xpath: xpath.clone(),
                    text: Some(text.clone()),
                    count: parse_count(count, source.clone(), dcx)?,
                }),
                _ => Err(wrong_arity("3 or 4")),
            },
            "snapshot" => match args {
                [name, path, xpath] => Ok(Self::Snapshot {
                    name: name.clone(),
                    path: path.clone(),
                    xpath: xpath.clone(),
                }),
                _ => Err(wrong_arity("3")),
            },
            "has-dir" => match args {
                [path] => Ok(Self::HasDir { path: path.clone() }),
                _ => Err(wrong_arity("1")),
            },
            _ => {
                // FIXME: Suggest potential typo candidates.
                // FIXME: Suggest "escaping" via non-whitespace char like backslash
                // FIXME: Note that it's parsed as a HtmlDocCk command, not as a ui_test-style compiletest directive
                dcx.emit(
                    &format!("unrecognized command `{}`", name.as_str()),
                    Source { range: name.range(), ..source },
                    None,
                );
                Err(())
            }
        }
    }
}

fn parse_regex(pattern: &str, source: Source<'_>, dcx: &mut DiagCtxt) -> Result<Regex, ()> {
    let pattern = channel::instantiate(&pattern, dcx)?;
    regex::RegexBuilder::new(&pattern).unicode(true).build().map_err(|_error| {
        // FIXME: better error message and location
        // FIXME: Use `regex_syntax` directly. Its error type exposes the
        // underlying span which we can then translate/offset.
        dcx.emit(&format!("malformed regex"), source, None)
    })
}

fn parse_count(count: &str, source: Source<'_>, dcx: &mut DiagCtxt) -> Result<u32, ()> {
    count.parse().map_err(|_error| {
        // FIXME: better error message & location
        dcx.emit("malformed count", source, None);
    })
}

fn command_regex() -> &'static Regex {
    // FIXME: Use `LazyLock` here instead once it's stable on beta.
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        use group::*;

        regex::RegexBuilder::new(&format!(
            r#"
            \s(?P<{NEGATION_MISPLACED}>!)?@(?P<{NEGATION}>!)?
            (?P<{NAME}>[A-Za-z]+(?:-[A-Za-z]+)*)
            (?P<{ARGUMENTS}>.*)$
            "#
        ))
        .ignore_whitespace(true)
        .unicode(true)
        .build()
        .unwrap()
    })
}

/// Regular expression capture groups.
mod group {
    pub(super) const ARGUMENTS: &str = "args";
    pub(super) const NAME: &str = "name";
    pub(super) const NEGATION_MISPLACED: &str = "prebang";
    pub(super) const NEGATION: &str = "postbang";
}
