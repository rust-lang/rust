// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(not(stage1))]
#[phase(syntax)]
extern crate regex_macros;

// Dirty hack: During stage1, test dynamic regexs. For stage2, we test
// native regexs.
#[cfg(stage1)]
macro_rules! regex(
    ($re:expr) => (
        match ::regex::Regex::new($re) {
            Ok(re) => re,
            Err(err) => fail!("{}", err),
        }
    );
)

mod bench;
mod tests;

