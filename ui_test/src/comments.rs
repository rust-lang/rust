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
    pub error_pattern: Option<(String, usize)>,
    pub error_matches: Vec<ErrorMatch>,
}

/// The conditions used for "ignore" and "only" filters.
#[derive(Debug)]
pub(crate) enum Condition {
    /// The given string must appear in the target.
    Target(String),
    /// Tests that the bitwidth is the given one.
    Bitwidth(u8),
}

#[derive(Debug)]
pub(crate) struct ErrorMatch {
    pub matched: String,
    pub revision: Option<String>,
    pub level: Option<Level>,
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
    ///
    /// This function will only parse `//@` and `//~` style comments (and the `//[xxx]~` variant)
    /// and ignore all others
    fn parse_checked(path: &Path, content: &str) -> Result<Self> {
        let mut this = Self::default();

        // The line that a `|` will refer to
        let mut fallthrough_to = None;
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
            let command = command.trim();
            if let Some((command, args)) = command.split_once(':') {
                self.parse_command_with_args(command, args, l)
            } else if let Some((command, _comments)) = command.split_once(' ') {
                self.parse_command(command)
            } else {
                self.parse_command(command)
            }
        } else if let Some((_, pattern)) = line.split_once("//~") {
            self.parse_pattern(pattern, fallthrough_to, l)
        } else if let Some((_, pattern)) = line.split_once("//[") {
            self.parse_revisioned_pattern(pattern, fallthrough_to, l)
        } else {
            *fallthrough_to = None;
            Ok(())
        }
    }

    /// Parse comments in `content`.
    /// `path` is only used to emit diagnostics if parsing fails.
    pub(crate) fn parse(path: &Path, content: &str) -> Result<Self> {
        let mut this = Self::parse_checked(path, content)?;
        if content.contains("//@") {
            // Migration mode: if new syntax is used, ignore all old syntax
            return Ok(this);
        }

        for (l, line) in content.lines().enumerate() {
            let l = l + 1; // enumerate starts at 0, but line numbers start at 1
            if let Some(revisions) = line.strip_prefix("// revisions:") {
                assert_eq!(
                    this.revisions,
                    None,
                    "{}:{l}, cannot specifiy revisions twice",
                    path.display()
                );
                this.revisions =
                    Some(revisions.split_whitespace().map(|s| s.to_string()).collect());
            }
            if let Some(s) = line.strip_prefix("// ignore-") {
                let s = s
                    .split_once(|c: char| c == ':' || c.is_whitespace())
                    .map(|(s, _)| s)
                    .unwrap_or(s);
                this.ignore.push(Condition::parse(s));
            }
            if let Some(s) = line.strip_prefix("// only-") {
                let s = s
                    .split_once(|c: char| c == ':' || c.is_whitespace())
                    .map(|(s, _)| s)
                    .unwrap_or(s);
                this.only.push(Condition::parse(s));
            }
            if line.starts_with("// stderr-per-bitwidth") {
                assert!(
                    !this.stderr_per_bitwidth,
                    "{}:{l}, cannot specifiy stderr-per-bitwidth twice",
                    path.display()
                );
                this.stderr_per_bitwidth = true;
            }
            if let Some(s) = line.strip_prefix("// compile-flags:") {
                this.compile_flags.extend(s.split_whitespace().map(|s| s.to_string()));
            }
            if let Some(s) = line.strip_prefix("// rustc-env:") {
                for env in s.split_whitespace() {
                    if let Some((k, v)) = env.split_once('=') {
                        this.env_vars.push((k.to_string(), v.to_string()));
                    }
                }
            }
            if let Some(s) = line.strip_prefix("// normalize-stderr-test:") {
                let (from, to) = s.split_once("->").expect("normalize-stderr-test needs a `->`");
                let from = from.trim().trim_matches('"');
                let to = to.trim().trim_matches('"');
                let from = Regex::new(from).unwrap();
                this.normalize_stderr.push((from, to.to_string()));
            }
            if let Some(s) = line.strip_prefix("// error-pattern:") {
                assert_eq!(
                    this.error_pattern,
                    None,
                    "{}:{l}, cannot specifiy error_pattern twice",
                    path.display()
                );
                this.error_pattern = Some((s.trim().to_string(), l));
            }
        }
        Ok(this)
    }

    fn parse_command_with_args(&mut self, command: &str, args: &str, l: usize) -> Result<()> {
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
                let (from, to) = args
                    .split_once("->")
                    .ok_or_else(|| eyre!("normalize-stderr-test needs a `->`"))?;
                let from = from.trim().trim_matches('"');
                let to = to.trim().trim_matches('"');
                let from = Regex::new(from).ok().ok_or_else(|| eyre!("invalid regex"))?;
                self.normalize_stderr.push((from, to.to_string()));
            }
            "error-pattern" => {
                ensure!(
                    self.error_pattern.is_none(),
                    "cannot specifiy error_pattern twice, previous: {:?}",
                    self.error_pattern
                );
                self.error_pattern = Some((args.trim().to_string(), l));
            }
            // Maybe the user just left a comment explaining a command without arguments
            _ => self.parse_command(command)?,
        }
        Ok(())
    }

    fn parse_command(&mut self, command: &str) -> Result<()> {
        if let Some(s) = command.strip_prefix("ignore-") {
            self.ignore.push(Condition::parse(s));
            return Ok(());
        }

        if let Some(s) = command.strip_prefix("only-") {
            self.only.push(Condition::parse(s));
            return Ok(());
        }

        if command.starts_with("stderr-per-bitwidth") {
            ensure!(!self.stderr_per_bitwidth, "cannot specifiy stderr-per-bitwidth twice");
            self.stderr_per_bitwidth = true;
            return Ok(());
        }

        bail!("unknown command {command}");
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

    // parse something like (?P<offset>\||[\^]+)? *(?P<level>ERROR|HELP|WARN|NOTE)?:?(?P<text>.*)
    fn parse_pattern_inner(
        &mut self,
        pattern: &str,
        fallthrough_to: &mut Option<usize>,
        revision: Option<String>,
        l: usize,
    ) -> Result<()> {
        // FIXME: check that the error happens on the marked line

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

        let (level, pattern) = match pattern.trim_start().split_once(|c| matches!(c, ':' | ' ')) {
            None => (None, pattern),
            Some((level, pattern_without_level)) =>
                match level.parse().ok() {
                    Some(level) => (Some(level), pattern_without_level),
                    None => (None, pattern),
                },
        };

        let matched = pattern.trim().to_string();

        ensure!(!matched.is_empty(), "no pattern specified");

        *fallthrough_to = Some(match_line);

        self.error_matches.push(ErrorMatch {
            matched,
            revision,
            level,
            definition_line: l,
            line: match_line,
        });

        Ok(())
    }
}
