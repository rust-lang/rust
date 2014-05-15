// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use regex::Regex;
static RE: Regex = regex!(r"\d+");

#[test]
fn static_splitn() {
    let text = "cauchy123plato456tyler789binx";
    let subs: Vec<&str> = RE.splitn(text, 2).collect();
    assert_eq!(subs, vec!("cauchy", "plato456tyler789binx"));
}

#[test]
fn static_split() {
    let text = "cauchy123plato456tyler789binx";
    let subs: Vec<&str> = RE.split(text).collect();
    assert_eq!(subs, vec!("cauchy", "plato", "tyler", "binx"));
}
