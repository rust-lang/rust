// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(staged_api)]
#![staged_api]
#![doc(issue_tracker_base_url = "http://issue_url/")]

#![unstable(feature="test", issue="27759")]

// @has issue_27759/unstable/index.html
// @has - '<code>test</code>'
// @has - '<a href="http://issue_url/27759">#27759</a>'
#[unstable(feature="test", issue="27759")]
pub mod unstable {
    // @has issue_27759/unstable/fn.issue.html
    // @has - '<code>test_function</code>'
    // @has - '<a href="http://issue_url/1234567890">#1234567890</a>'
    #[unstable(feature="test_function", issue="1234567890")]
    pub fn issue() {}
}
