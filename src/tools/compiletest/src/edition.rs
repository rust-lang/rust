use crate::fatal;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Edition {
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

pub fn parse_edition(mut input: &str) -> Edition {
    input = input.trim();
    if input == "future" {
        Edition::Future
    } else {
        Edition::Year(input.parse().unwrap_or_else(|_| {
            fatal!("`{input}` doesn't look like an edition");
        }))
    }
}
