use std::str::FromStr;

use crate::fatal;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Edition {
    // Note that the ordering here is load-bearing, as we want the future edition to be greater than
    // any year-based edition.
    Year(u32),
    Future,
}

impl std::fmt::Display for Edition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Edition::Year(year) => write!(f, "{year}"),
            Edition::Future => f.write_str("future"),
        }
    }
}

impl From<u32> for Edition {
    fn from(value: u32) -> Self {
        Edition::Year(value)
    }
}

impl FromStr for Edition {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let input = input.trim();
        if input == "future" {
            Ok(Edition::Future)
        } else {
            let year: u32 =
                input.parse().map_err(|v| format!("{input} is not a valid edition year: {v}"))?;
            Ok(Edition::Year(year))
        }
    }
}

pub(crate) fn parse_edition(input: &str) -> Edition {
    input.parse().unwrap_or_else(|_| fatal!("`{input}` doesn't look like an edition"))
}
