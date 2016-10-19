// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;

#[derive(Clone)]
pub struct Registry {
    descriptions: HashMap<&'static str, &'static str>,
}

impl Registry {
    pub fn new(descriptions: &[(&'static str, &'static str)]) -> Registry {
        Registry { descriptions: descriptions.iter().cloned().collect() }
    }

    pub fn find_description(&self, code: &str) -> Option<&'static str> {
        self.descriptions.get(code).cloned()
    }
}
