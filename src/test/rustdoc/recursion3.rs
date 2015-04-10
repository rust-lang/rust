// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]

pub mod longhands {
    pub use super::*;

    pub use super::common_types::computed::compute_CSSColor as to_computed_value;

    pub fn computed_as_specified() {}
}

pub mod common_types {
    pub mod computed {
        pub use super::super::longhands::computed_as_specified as compute_CSSColor;
    }
}
