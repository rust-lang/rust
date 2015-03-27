// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we don't ICE when translating a generic impl method from
// an extern crate that contains a match expression on a local
// variable lvalue where one of the match case bodies contains an
// expression that autoderefs through an overloaded generic deref
// impl.

// aux-build:issue-18514.rs
// pretty-expanded FIXME #23616

extern crate issue_18514 as ice;
use ice::{Tr, St};

fn main() {
    let st: St<()> = St(vec![]);
    st.tr();
}
