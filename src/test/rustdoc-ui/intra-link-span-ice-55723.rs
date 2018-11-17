// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-end-whitespace

#![deny(intra_doc_link_resolution_failure)]

// An error in calculating spans while reporting intra-doc link resolution errors caused rustdoc to
// attempt to slice in the middle of a multibyte character. See
// https://github.com/rust-lang/rust/issues/55723

/// ## For example:
///  
/// （arr[i]）
pub fn test_ice() {
    unimplemented!();
}
