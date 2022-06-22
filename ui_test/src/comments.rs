use std::path::Path;

use regex::Regex;

use crate::rustc_stderr::Level;

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
    pub(crate) fn parse_file(path: &Path) -> Self {
        let content = std::fs::read_to_string(path).unwrap();
        Self::parse(path, &content)
    }

    /// Parse comments in `content`.
    /// `path` is only used to emit diagnostics if parsing fails.
    pub(crate) fn parse(path: &Path, content: &str) -> Self {
        let mut this = Self::default();
        let error_pattern_regex =
            Regex::new(r"//(\[(?P<revision>[^\]]+)\])?~(?P<offset>\||[\^]+)?\s*(?P<level>ERROR|HELP|WARN|NOTE)?:?(?P<text>.*)")
                .unwrap();

        // The line that a `|` will refer to
        let mut fallthrough_to = None;
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
                    Some(revisions.trim().split_whitespace().map(|s| s.to_string()).collect());
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
            if let Some(captures) = error_pattern_regex.captures(line) {
                // FIXME: check that the error happens on the marked line
                let matched = captures["text"].trim().to_string();

                let revision = captures.name("revision").map(|rev| rev.as_str().to_string());

                let level = captures.name("level").map(|rev| rev.as_str().parse().unwrap());

                let match_line = match captures.name("offset").map(|rev| rev.as_str()) {
                    Some("|") => fallthrough_to.expect("`//~|` pattern without preceding line"),
                    Some(pat) => {
                        debug_assert!(pat.chars().all(|c| c == '^'));
                        l - pat.len()
                    }
                    None => l,
                };

                fallthrough_to = Some(match_line);

                this.error_matches.push(ErrorMatch {
                    matched,
                    revision,
                    level,
                    definition_line: l,
                    line: match_line,
                });
            } else {
                fallthrough_to = None;
            }
        }
        this
    }
}
