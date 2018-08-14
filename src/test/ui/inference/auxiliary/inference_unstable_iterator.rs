// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(staged_api)]

#![stable(feature = "ipu_iterator", since = "1.0.0")]

#[stable(feature = "ipu_iterator", since = "1.0.0")]
pub trait IpuIterator {
    #[unstable(feature = "ipu_flatten", issue = "99999")]
    fn ipu_flatten(&self) -> u32 {
        0
    }
}

#[stable(feature = "ipu_iterator", since = "1.0.0")]
impl IpuIterator for char {}
