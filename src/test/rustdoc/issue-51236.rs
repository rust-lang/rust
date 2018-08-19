// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;

pub mod traits {
    pub trait Owned<'a> {
        type Reader;
    }
}

// @has issue_51236/struct.Owned.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<T> Send for \
// Owned<T> where <T as Owned<'static>>::Reader: Send"
pub struct Owned<T> where T: for<'a> ::traits::Owned<'a> {
    marker: PhantomData<<T as ::traits::Owned<'static>>::Reader>,
}
