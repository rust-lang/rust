/// Lint data parsed from the Clippy source code.
#[derive(Clone, PartialEq, Debug)]
pub struct Lint {
    pub name: &'static str,
    pub group: &'static str,
    pub desc: &'static str,
    pub deprecation: Option<&'static str>,
    pub module: &'static str,
}

#[derive(PartialOrd, PartialEq, Ord, Eq)]
pub enum LintLevel {
    Allow,
    Warn,
    Deny,
}

pub const LINT_LEVELS: [(&str, LintLevel); 8] = [
    ("correctness", LintLevel::Deny),
    ("style", LintLevel::Warn),
    ("complexity", LintLevel::Warn),
    ("perf", LintLevel::Warn),
    ("restriction", LintLevel::Allow),
    ("pedantic", LintLevel::Allow),
    ("nursery", LintLevel::Allow),
    ("cargo", LintLevel::Allow),
];
