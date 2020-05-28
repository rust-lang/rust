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
pub enum Level {
    Allow,
    Warn,
    Deny,
}

pub const LINT_LEVELS: [(&str, Level); 8] = [
    ("correctness", Level::Deny),
    ("style", Level::Warn),
    ("complexity", Level::Warn),
    ("perf", Level::Warn),
    ("restriction", Level::Allow),
    ("pedantic", Level::Allow),
    ("nursery", Level::Allow),
    ("cargo", Level::Allow),
];
