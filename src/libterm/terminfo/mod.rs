// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use std::hashmap::HashMap;

/// A parsed terminfo entry.
pub struct TermInfo {
    /// Names for the terminal
    priv names: ~[~str],
    /// Map of capability name to boolean value
    priv bools: HashMap<~str, bool>,
    /// Map of capability name to numeric value
    numbers: HashMap<~str, u16>,
    /// Map of capability name to raw (unexpanded) string
    strings: HashMap<~str, ~[u8]>
}

pub mod searcher;
pub mod parser {
    pub mod compiled;
}
pub mod parm;
