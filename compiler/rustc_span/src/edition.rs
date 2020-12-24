use crate::symbol::{sym, Symbol};
use std::fmt;
use std::str::FromStr;

use rustc_macros::HashStable_Generic;

/// The edition of the compiler. (See [RFC 2052](https://github.com/rust-lang/rfcs/blob/master/text/2052-epochs.md).)
#[derive(Clone, Copy, Hash, PartialEq, PartialOrd, Debug, Encodable, Decodable, Eq)]
#[derive(HashStable_Generic)]
pub enum Edition {
    // When adding new editions, be sure to do the following:
    //
    // - update the `ALL_EDITIONS` const
    // - update the `EDITION_NAME_LIST` const
    // - add a `rust_####()` function to the session
    // - update the enum in Cargo's sources as well
    //
    // Editions *must* be kept in order, oldest to newest.
    /// The 2015 edition
    Edition2015,
    /// The 2018 edition
    Edition2018,
}

// Must be in order from oldest to newest.
pub const ALL_EDITIONS: &[Edition] = &[Edition::Edition2015, Edition::Edition2018];

pub const EDITION_NAME_LIST: &str = "2015|2018";

pub const DEFAULT_EDITION: Edition = Edition::Edition2015;

impl fmt::Display for Edition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match *self {
            Edition::Edition2015 => "2015",
            Edition::Edition2018 => "2018",
        };
        write!(f, "{}", s)
    }
}

impl Edition {
    pub fn lint_name(&self) -> &'static str {
        match *self {
            Edition::Edition2015 => "rust_2015_compatibility",
            Edition::Edition2018 => "rust_2018_compatibility",
        }
    }

    pub fn feature_name(&self) -> Symbol {
        match *self {
            Edition::Edition2015 => sym::rust_2015_preview,
            Edition::Edition2018 => sym::rust_2018_preview,
        }
    }

    pub fn is_stable(&self) -> bool {
        match *self {
            Edition::Edition2015 => true,
            Edition::Edition2018 => true,
        }
    }
}

impl FromStr for Edition {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "2015" => Ok(Edition::Edition2015),
            "2018" => Ok(Edition::Edition2018),
            _ => Err(()),
        }
    }
}
