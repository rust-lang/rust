// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::str::FromStr;

/// The edition of the compiler (RFC 2052)
#[derive(Clone, Copy, Hash, PartialOrd, Ord, Eq, PartialEq, Debug)]
#[non_exhaustive]
pub enum Edition {
    // editions must be kept in order, newest to oldest

    /// The 2015 edition
    Edition2015,
    /// The 2018 edition
    Edition2018,

    // when adding new editions, be sure to update:
    //
    // - the list in the `parse_edition` static in librustc::session::config
    // - add a `rust_####()` function to the session
    // - update the enum in Cargo's sources as well
    //
    // When -Zedition becomes --edition, there will
    // also be a check for the edition being nightly-only
    // somewhere. That will need to be updated
    // whenever we're stabilizing/introducing a new edition
    // as well as changing the default Cargo template.
}

// must be in order from oldest to newest
pub const ALL_EPOCHS: &[Edition] = &[Edition::Edition2015, Edition::Edition2018];

impl fmt::Display for Edition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
            Edition::Edition2015 => "edition_2015",
            Edition::Edition2018 => "edition_2018",
        }
    }
}

impl FromStr for Edition {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "2015" => Ok(Edition::Edition2015),
            "2018" => Ok(Edition::Edition2018),
            _ => Err(())
        }
    }
}
