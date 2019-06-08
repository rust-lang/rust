/// Lint data parsed from the Clippy source code.
#[derive(Clone, PartialEq, Debug)]
pub struct Lint {
    pub name: &'static str,
    pub group: &'static str,
    pub desc: &'static str,
    pub deprecation: Option<&'static str>,
    pub module: &'static str,
}
