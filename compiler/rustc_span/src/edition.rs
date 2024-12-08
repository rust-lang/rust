use std::fmt;
use std::str::FromStr;

use rustc_macros::{Decodable, Encodable, HashStable_Generic};

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
    /// The 2021 edition
    Edition2021,
    /// The 2024 edition
    Edition2024,
}

// Must be in order from oldest to newest.
pub const ALL_EDITIONS: &[Edition] =
    &[Edition::Edition2015, Edition::Edition2018, Edition::Edition2021, Edition::Edition2024];

pub const EDITION_NAME_LIST: &str = "2015|2018|2021|2024";

pub const DEFAULT_EDITION: Edition = Edition::Edition2015;

pub const LATEST_STABLE_EDITION: Edition = Edition::Edition2024;

impl fmt::Display for Edition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match *self {
            Edition::Edition2015 => "2015",
            Edition::Edition2018 => "2018",
            Edition::Edition2021 => "2021",
            Edition::Edition2024 => "2024",
        };
        write!(f, "{s}")
    }
}

impl Edition {
    pub fn lint_name(self) -> &'static str {
        match self {
            Edition::Edition2015 => "rust_2015_compatibility",
            Edition::Edition2018 => "rust_2018_compatibility",
            Edition::Edition2021 => "rust_2021_compatibility",
            Edition::Edition2024 => "rust_2024_compatibility",
        }
    }

    pub fn is_stable(self) -> bool {
        match self {
            Edition::Edition2015 => true,
            Edition::Edition2018 => true,
            Edition::Edition2021 => true,
            Edition::Edition2024 => true,
        }
    }

    /// Is this edition 2015?
    pub fn is_rust_2015(self) -> bool {
        self == Edition::Edition2015
    }

    /// Are we allowed to use features from the Rust 2018 edition?
    pub fn at_least_rust_2018(self) -> bool {
        self >= Edition::Edition2018
    }

    /// Are we allowed to use features from the Rust 2021 edition?
    pub fn at_least_rust_2021(self) -> bool {
        self >= Edition::Edition2021
    }

    /// Are we allowed to use features from the Rust 2024 edition?
    pub fn at_least_rust_2024(self) -> bool {
        self >= Edition::Edition2024
    }
}

impl FromStr for Edition {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "2015" => Ok(Edition::Edition2015),
            "2018" => Ok(Edition::Edition2018),
            "2021" => Ok(Edition::Edition2021),
            "2024" => Ok(Edition::Edition2024),
            _ => Err(()),
        }
    }
}
