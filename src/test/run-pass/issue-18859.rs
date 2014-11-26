// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub mod bar {
        pub mod baz {
            pub fn name() -> &'static str {
                module_path!()
            }
        }
    }
}

fn main() {
    assert_eq!(module_path!(), "issue-18859");
    assert_eq!(foo::bar::baz::name(), "issue-18859::foo::bar::baz");
}
