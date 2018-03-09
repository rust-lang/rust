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

/// The epoch of the compiler (RFC 2052)
#[derive(Clone, Copy, Hash, PartialOrd, Ord, Eq, PartialEq, Debug)]
#[non_exhaustive]
pub enum Epoch {
    // epochs must be kept in order, newest to oldest

    /// The 2015 epoch
    Epoch2015,
    /// The 2018 epoch
    Epoch2018,

    // when adding new epochs, be sure to update:
    //
    // - the list in the `parse_epoch` static in librustc::session::config
    // - add a `rust_####()` function to the session
    // - update the enum in Cargo's sources as well
    //
    // When -Zepoch becomes --epoch, there will
    // also be a check for the epoch being nightly-only
    // somewhere. That will need to be updated
    // whenever we're stabilizing/introducing a new epoch
    // as well as changing the default Cargo template.
}

// must be in order from oldest to newest
pub const ALL_EPOCHS: &[Epoch] = &[Epoch::Epoch2015, Epoch::Epoch2018];

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match *self {
            Epoch::Epoch2015 => "2015",
            Epoch::Epoch2018 => "2018",
        };
        write!(f, "{}", s)
    }
}

impl Epoch {
    pub fn lint_name(&self) -> &'static str {
        match *self {
            Epoch::Epoch2015 => "epoch_2015",
            Epoch::Epoch2018 => "epoch_2018",
        }
    }
}

impl FromStr for Epoch {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "2015" => Ok(Epoch::Epoch2015),
            "2018" => Ok(Epoch::Epoch2018),
            _ => Err(())
        }
    }
}
