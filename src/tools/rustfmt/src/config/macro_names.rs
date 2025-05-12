//! This module contains types and functions to support formatting specific macros.

use itertools::Itertools;
use std::{fmt, str};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json as json;
use thiserror::Error;

/// Defines the name of a macro.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct MacroName(String);

impl MacroName {
    pub fn new(other: String) -> Self {
        Self(other)
    }
}

impl fmt::Display for MacroName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<MacroName> for String {
    fn from(other: MacroName) -> Self {
        other.0
    }
}

/// Defines a selector to match against a macro.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize)]
pub enum MacroSelector {
    Name(MacroName),
    All,
}

impl<'de> Deserialize<'de> for MacroSelector {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(de)?;
        std::str::FromStr::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for MacroSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => name.fmt(f),
            Self::All => write!(f, "*"),
        }
    }
}

impl str::FromStr for MacroSelector {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "*" => MacroSelector::All,
            name => MacroSelector::Name(MacroName(name.to_owned())),
        })
    }
}

/// A set of macro selectors.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct MacroSelectors(pub Vec<MacroSelector>);

impl fmt::Display for MacroSelectors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.iter().format(", "))
    }
}

#[derive(Error, Debug)]
pub enum MacroSelectorsError {
    #[error("{0}")]
    Json(json::Error),
}

// This impl is needed for `Config::override_value` to work for use in tests.
impl str::FromStr for MacroSelectors {
    type Err = MacroSelectorsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let raw: Vec<&str> = json::from_str(s).map_err(MacroSelectorsError::Json)?;
        Ok(Self(
            raw.into_iter()
                .map(|raw| {
                    MacroSelector::from_str(raw).expect("MacroSelector from_str is infallible")
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn macro_names_from_str() {
        let macro_names = MacroSelectors::from_str(r#"["foo", "*", "bar"]"#).unwrap();
        assert_eq!(
            macro_names,
            MacroSelectors(
                [
                    MacroSelector::Name(MacroName("foo".to_owned())),
                    MacroSelector::All,
                    MacroSelector::Name(MacroName("bar".to_owned()))
                ]
                .into_iter()
                .collect()
            )
        );
    }

    #[test]
    fn macro_names_display() {
        let macro_names = MacroSelectors::from_str(r#"["foo", "*", "bar"]"#).unwrap();
        assert_eq!(format!("{macro_names}"), "foo, *, bar");
    }
}
