use std::fmt;
use std::num::ParseIntError;
use std::str::FromStr;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    parts: [u32; 3],
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&format!("{}.{}.{}", self.parts[0], self.parts[1], self.parts[2]))
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
        let mut iter = s.split('.').map(|part| Ok(part.parse()?));

        let mut part = || iter.next().unwrap_or(Err(ParseVersionError::WrongNumberOfParts));

        let parts = [part()?, part()?, part()?];

        if iter.next().is_some() {
            // Ensure we don't have more than 3 parts.
            return Err(ParseVersionError::WrongNumberOfParts);
        }

        Ok(Self { parts })
    }
}
