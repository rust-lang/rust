// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait T {
    fn a_method(&self) -> usize;
}

// @has manual_impl/struct.S.html '//*[@class="trait"]' 'T'
// @has - '//*[@class="docblock"]' 'Docs associated with the trait implementation.'
// @has - '//*[@class="docblock"]' 'Docs associated with the trait method implementation.'
pub struct S(usize);

/// Docs associated with the trait implementation.
impl T for S {
    /// Docs associated with the trait method implementation.
    fn a_method(&self) -> usize {
        self.0
    }
}
