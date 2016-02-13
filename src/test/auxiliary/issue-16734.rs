// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod inner_private_module {
    pub struct Unnameable;

    impl Unnameable {
        pub fn method_of_unnameable_type(&self) -> &'static str {
            "Hello!"
        }
    }
}

pub fn public_function_returning_unnameable_type() -> inner_private_module::Unnameable {
    inner_private_module::Unnameable
}
