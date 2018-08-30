// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The number of `#`s used to wrap the documentation comment should differ regarding the content.
//
// Related issue: #27489

macro_rules! homura {
    ($x:expr, #[$y:meta]) => (assert_eq!($x, stringify!($y)))
}

fn main() {
    homura! {
        r#"doc = r" Madoka""#,
        /// Madoka
    };

    homura! {
        r##"doc = r#" One quote mark: ["]"#"##,
        /// One quote mark: ["]
    };

    homura! {
        r##"doc = r#" Two quote marks: [""]"#"##,
        /// Two quote marks: [""]
    };

    homura! {
        r#####"doc = r####" Raw string ending sequences: ["###]"####"#####,
        /// Raw string ending sequences: ["###]
    };
}
