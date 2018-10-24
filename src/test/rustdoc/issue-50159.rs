// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub trait Signal {
    type Item;
}

pub trait Signal2 {
    type Item2;
}

impl<B, C> Signal2 for B where B: Signal<Item = C> {
    type Item2 = C;
}

// @has issue_50159/struct.Switch.html
// @has - '//code' 'impl<B> Send for Switch<B> where <B as Signal>::Item: Send'
// @has - '//code' 'impl<B> Sync for Switch<B> where <B as Signal>::Item: Sync'
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 0
// @count - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]' 2
pub struct Switch<B: Signal> {
    pub inner: <B as Signal2>::Item2,
}
