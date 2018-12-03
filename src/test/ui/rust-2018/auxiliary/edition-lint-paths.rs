// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn foo() {}

#[macro_export]
macro_rules! macro_2015 {
    () => {
        use edition_lint_paths as other_name;
        use edition_lint_paths::foo as other_foo;
        fn check_macro_2015() {
            ::edition_lint_paths::foo();
        }
    }
}
