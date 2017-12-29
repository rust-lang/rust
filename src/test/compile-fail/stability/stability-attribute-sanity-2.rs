// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// More checks that stability attributes are used correctly

#![feature(staged_api)]

#![stable(feature = "test_feature", since = "1.0.0")]

#[stable(feature = "a", feature = "b", since = "1.0.0")] //~ ERROR multiple 'feature' items
fn f1() { }

#[stable(feature = "a", sinse = "1.0.0")] //~ ERROR unknown meta item 'sinse'
fn f2() { }

#[unstable(feature = "a", issue = "no")] //~ ERROR incorrect 'issue'
fn f3() { }

fn main() { }
