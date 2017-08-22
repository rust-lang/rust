// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(doc_masked)]

#![doc(masked)]
extern crate std as realstd;

// @has doc_masked/struct.LocalStruct.html
// @has - '//*[@class="impl"]//code' 'impl LocalTrait for LocalStruct'
// @!has - '//*[@class="impl"]//code' 'impl Copy for LocalStruct'
#[derive(Copy, Clone)]
pub struct LocalStruct;

// @has doc_masked/trait.LocalTrait.html
// @has - '//*[@id="implementors-list"]//code' 'impl LocalTrait for LocalStruct'
// @!has - '//*[@id="implementors-list"]//code' 'impl LocalTrait for usize'
pub trait LocalTrait { }

impl LocalTrait for LocalStruct { }

impl LocalTrait for usize { }
