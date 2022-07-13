use std::path::Path;

use regex::Regex;

use crate::rustc_stderr::Level;

use color_eyre::eyre::{bail, ensure, eyre, Result};

#[cfg(test)]
mod tests;

/// This crate supports various magic comments that get parsed as file-specific
/// configuration values. This struct parses them all in one go and then they
/// get processed by their respective use sites.
#[derive(Default, Debug)]
pub(crate) struct Comments {
    /// List of revision names to execute. Can only be speicified once
    pub revisions: Option<Vec<String>>,
    /// Don't run this test if any of these filters apply
    pub ignore: Vec<Condition>,
    /// Only run this test if all of these filters apply
    pub only: Vec<Condition>,
    /// Generate one .stderr file per bit width, by prepending with `.64bit` and similar
    pub stderr_per_bitwidth: bool,
    /// Additional flags to pass to the executable
    pub compile_flags: Vec<String>,
    /// Additional env vars to set for the executable
    pub env_vars: Vec<(String, String)>,
    /// Normalizations to apply to the stderr output before emitting it to disk
    pub normalize_stderr: Vec<(Regex, String)>,
    /// An arbitrary pattern to look for in the stderr.
    pub error_pattern: Option<(Pattern, usize)>,
    pub error_matches: Vec<ErrorMatch>,
    /// Ignore diagnostics below this level.
    /// `None` means pick the lowest level from the `error_pattern`s.
    pub require_annotations_for_level: Option<Level>,
}

/// The conditions used for "ignore" and "only" filters.
#[derive(Debug)]
pub(crate) enum Condition {
    /// The given string must appear in the target.
    Target(String),
    /// Tests that the bitwidth is the given one.
    Bitwidth(u8),
}

#[derive(Debug, Clone)]
pub(crate) enum Pattern {
    SubString(String),
    Regex(Regex),
}

#[derive(Debug)]
pub(crate) struct ErrorMatch {
    pub pattern: Pattern,
    pub revision: Option<String>,
    pub level: Level,
    /// The line where the message was defined, for reporting issues with it (e.g. in case it wasn't found).
    pub definition_line: usize,
    /// The line this pattern is expecting to find a message in.
    pub line: usize,
}

impl Condition {
    fn parse(c: &str) -> Self {
        if let Some(bits) = c.strip_suffix("bit") {
            let bits: u8 = bits.parse().expect(
                "ignore/only filter ending in 'bit' must be of the form 'Nbit' for some integer N",
            );
            Condition::Bitwidth(bits)
        } else {
            Condition::Target(c.to_owned())
        }
    }
}

impl Comments {
    pub(crate) fn parse_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(path, &content)
    }

    /// Parse comments in `content`.
    /// `path` is only used to emit diagnostics if parsing fails.
    pub(crate) fn parse(path: &Path, content: &str) -> Result<Self> {
        let mut this = Self::default();

        let mut fallthrough_to = None; // The line that a `|` will refer to.
        for (l, line) in content.lines().enumerate() {
            let l = l + 1; // enumerate starts at 0, but line numbers start at 1
            this.parse_checked_line(l, &mut fallthrough_to, line).map_err(|err| {
                err.wrap_err(format!("{}:{l}: failed to parse annotation", path.display()))
            })?;
        }
        Ok(this)
    }

    fn parse_checked_line(
        &mut self,
        l: usize,
        fallthrough_to: &mut Option<usize>,
        line: &str,
    ) -> Result<()> {
        if let Some((_, command)) = line.split_once("//@") {
            self.parse_command(command.trim(), l)
        } else if let Some((_, pattern)) = line.split_once("//~") {
            self.parse_pattern(pattern, fallthrough_to, l)
        } else if let Some((_, pattern)) = line.split_once("//[") {
            self.parse_revisioned_pattern(pattern, fallthrough_to, l)
        } else {
            *fallthrough_to = None;
            Ok(())
        }
    }

    fn parse_command(&mut self, command: &str, l: usize) -> Result<()> {
        // Commands are letters or dashes, grab everything until the first character that is neither of those.
        let (command, args) =
            match command.chars().position(|c: char| !c.is_alphanumeric() && c != '-') {
                None => (command, ""),
                Some(i) => {
                    let (command, args) = command.split_at(i);
                    let mut args = args.chars();
                    // Commands are separated from their arguments by ':' or ' '
                    let next = args
                        .next()
                        .expect("the `position` above guarantees that there is at least one char");
                    ensure!(next == ':', "test command must be followed by : (or end the line)");
                    (command, args.as_str().trim())
                }
            };

        match command {
            "revisions" => {
                ensure!(self.revisions.is_none(), "cannot specifiy revisions twice");
                self.revisions = Some(args.split_whitespace().map(|s| s.to_string()).collect());
            }
            "compile-flags" => {
                self.compile_flags.extend(args.split_whitespace().map(|s| s.to_string()));
            }
            "rustc-env" =>
                for env in args.split_whitespace() {
                    let (k, v) = env.split_once('=').ok_or_else(|| {
                        eyre!("environment variables must be key/value pairs separated by a `=`")
                    })?;
                    self.env_vars.push((k.to_string(), v.to_string()));
                },
            "normalize-stderr-test" => {
                /// Parses a string literal. `s` has to start with `"`; everything until the next `"` is
                /// returned in the first component. `\` can be used to escape arbitrary character.
                /// Second return component is the rest of the string with leading whitespace removed.
                fn parse_str(s: &str) -> Result<(&str, &str)> {
                    let mut chars = s.char_indices();
                    match chars.next().ok_or_else(|| eyre!("missing arguments"))?.1 {
                        '"' => {
                            let s = chars.as_str();
                            let mut escaped = false;
                            for (i, c) in chars {
                                if escaped {
                                    // Accept any character as literal after a `\`.
                                    escaped = false;
                                } else if c == '"' {
                                    return Ok((&s[..(i - 1)], s[i..].trim_start()));
                                } else {
                                    escaped = c == '\\';
                                }
                            }
                            bail!("no closing quotes found for {s}")
                        }
                        c => bail!("expected '\"', got {c}"),
                    }
                }

                let (from, rest) = parse_str(args)?;

                let to = rest.strip_prefix("->").ok_or_else(|| {
                    eyre!("normalize-stderr-test needs a pattern and replacement separated by `->`")
                })?.trim_start();
                let (to, rest) = parse_str(to)?;

                ensure!(rest.is_empty(), "trailing text after pattern replacement: {rest}");

                let from = Regex::new(from)?;
                self.normalize_stderr.push((from, to.to_string()));
            }
            "error-pattern" => {
                ensure!(
                    self.error_pattern.is_none(),
                    "cannot specifiy error_pattern twice, previous: {:?}",
                    self.error_pattern
                );
                self.error_pattern = Some((Pattern::parse(args.trim())?, l));
            }
            "stderr-per-bitwidth" => {
                // args are ignored (can be used as comment)
                ensure!(!self.stderr_per_bitwidth, "cannot specifiy stderr-per-bitwidth twice");
                self.stderr_per_bitwidth = true;
            }
            "require-annotations-for-level" => {
                ensure!(
                    self.require_annotations_for_level.is_none(),
                    "cannot specify `require-annotations-for-level` twice"
                );
                self.require_annotations_for_level = Some(args.trim().parse()?);
            }
            command => {
                if let Some(s) = command.strip_prefix("ignore-") {
                    // args are ignored (can be sue as comment)
                    self.ignore.push(Condition::parse(s));
                    return Ok(());
                }

                if let Some(s) = command.strip_prefix("only-") {
                    // args are ignored (can be sue as comment)
                    self.only.push(Condition::parse(s));
                    return Ok(());
                }
                bail!("unknown command {command}");
            }
        }

        Ok(())
    }

    fn parse_pattern(
        &mut self,
        pattern: &str,
        fallthrough_to: &mut Option<usize>,
        l: usize,
    ) -> Result<()> {
        self.parse_pattern_inner(pattern, fallthrough_to, None, l)
    }

    fn parse_revisioned_pattern(
        &mut self,
        pattern: &str,
        fallthrough_to: &mut Option<usize>,
        l: usize,
    ) -> Result<()> {
        let (revision, pattern) =
            pattern.split_once(']').ok_or_else(|| eyre!("`//[` without corresponding `]`"))?;
        if let Some(pattern) = pattern.strip_prefix('~') {
            self.parse_pattern_inner(pattern, fallthrough_to, Some(revision.to_owned()), l)
        } else {
            bail!("revisioned pattern must have `~` following the `]`");
        }
    }

    // parse something like (?P<offset>\||[\^]+)? *(?P<level>ERROR|HELP|WARN|NOTE): (?P<text>.*)
    fn parse_pattern_inner(
        &mut self,
        pattern: &str,
        fallthrough_to: &mut Option<usize>,
        revision: Option<String>,
        l: usize,
    ) -> Result<()> {
        let (match_line, pattern) =
            match pattern.chars().next().ok_or_else(|| eyre!("no pattern specified"))? {
                '|' =>
                    (
                        *fallthrough_to
                            .as_mut()
                            .ok_or_else(|| eyre!("`//~|` pattern without preceding line"))?,
                        &pattern[1..],
                    ),
                '^' => {
                    let offset = pattern.chars().take_while(|&c| c == '^').count();
                    (l - offset, &pattern[offset..])
                }
                _ => (l, pattern),
            };

        let pattern = pattern.trim_start();
        let offset = pattern
            .chars()
            .position(|c| !matches!(c, 'A'..='Z' | 'a'..='z'))
            .ok_or_else(|| eyre!("pattern without level"))?;

        let level = pattern[..offset].parse()?;
        let pattern = &pattern[offset..];
        let pattern = pattern.strip_prefix(':').ok_or_else(|| eyre!("no `:` after level found"))?;

        let pattern = pattern.trim();

        ensure!(!pattern.is_empty(), "no pattern specified");

        let pattern = Pattern::parse(pattern)?;

        *fallthrough_to = Some(match_line);

        self.error_matches.push(ErrorMatch {
            pattern,
            revision,
            level,
            definition_line: l,
            line: match_line,
        });

        Ok(())
    }
}

impl Pattern {
    pub(crate) fn matches(&self, message: &str) -> bool {
        match self {
            Pattern::SubString(s) => message.contains(s),
            Pattern::Regex(r) => r.is_match(message),
        }
    }

    pub(crate) fn parse(pattern: &str) -> Result<Self> {
        if let Some(pattern) = pattern.strip_prefix('/') {
            let regex =
                pattern.strip_suffix('/').ok_or_else(|| eyre!("regex must end with `/`"))?;
            Ok(Pattern::Regex(Regex::new(regex)?))
        } else {
            Ok(Pattern::SubString(pattern.to_string()))
        }
    }
}
