// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -g

extern crate issue_31702_1;

use std::collections::HashMap;
use issue_31702_1::U256;

pub struct Ethash {
    engine_params: for<'a> fn() -> Option<&'a Vec<u8>>,
    u256_params: HashMap<String, U256>,
}

impl Ethash {
    pub fn u256_param(&mut self, name: &str) -> U256 {
        let engine = self.engine_params;
        *self.u256_params.entry(name.to_owned()).or_insert_with(|| {
            engine().map_or(U256::new(0u64), |_a| loop {})
        })
    }
}
