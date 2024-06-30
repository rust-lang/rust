use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
#[value(rename_all = "PascalCase")]
pub enum Profile {
    Check,
    Debug,
    Doc,
    Opt,
    Clippy,
}

impl Display for Profile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Profile::Check => "Check",
            Profile::Debug => "Debug",
            Profile::Doc => "Doc",
            Profile::Opt => "Opt",
            Profile::Clippy => "Clippy",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
#[value(rename_all = "PascalCase")]
pub enum Scenario {
    Full,
    IncrFull,
    IncrUnchanged,
    IncrPatched,
}

impl Display for Scenario {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Scenario::Full => "Full",
            Scenario::IncrFull => "IncrFull",
            Scenario::IncrUnchanged => "IncrUnchanged",
            Scenario::IncrPatched => "IncrPatched",
        };
        f.write_str(name)
    }
}
