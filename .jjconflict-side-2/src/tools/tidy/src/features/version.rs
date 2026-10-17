use std::fmt;
use std::num::ParseIntError;
use std::str::FromStr;

#[cfg(test)]
mod tests;

pub const VERSION_PLACEHOLDER: &str = "CURRENT_RUSTC_VERSION";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "build-metrics", derive(serde::Serialize))]
pub enum Version {
    Explicit { parts: [u32; 3] },
    CurrentPlaceholder,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Version::Explicit { parts } => {
                f.pad(&format!("{}.{}.{}", parts[0], parts[1], parts[2]))
            }
            Version::CurrentPlaceholder => f.pad("CURRENT"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseVersionError {
    ParseIntError(ParseIntError),
    WrongNumberOfParts,
}

impl From<ParseIntError> for ParseVersionError {
    fn from(err: ParseIntError) -> Self {
        ParseVersionError::ParseIntError(err)
    }
}

impl FromStr for Version {
    type Err = ParseVersionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == VERSION_PLACEHOLDER {
            return Ok(Version::CurrentPlaceholder);
        }
        let mut iter = s.split('.').map(|part| Ok(part.parse()?));

        let mut part = || iter.next().unwrap_or(Err(ParseVersionError::WrongNumberOfParts));

        let parts = [part()?, part()?, part()?];

        if iter.next().is_some() {
            // Ensure we don't have more than 3 parts.
            return Err(ParseVersionError::WrongNumberOfParts);
        }

        Ok(Version::Explicit { parts })
    }
}
