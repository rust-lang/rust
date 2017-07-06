// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait MyTrait {
    fn method_on_mytrait() {}
}

pub struct MyStruct;

impl MyStruct {
    pub fn method_on_mystruct() {}
}

// @has typedef/type.MyAlias.html
// @has - '//*[@class="impl"]//code' 'impl MyAlias'
// @has - '//*[@class="impl"]//code' 'impl MyTrait for MyAlias'
// @has - 'Alias docstring'
// @has - '//*[@class="sidebar"]//p[@class="location"]' 'Type Definition MyAlias'
// @has - '//*[@class="sidebar"]//a[@href="#methods"]' 'Methods'
// @has - '//*[@class="sidebar"]//a[@href="#implementations"]' 'Trait Implementations'
/// Alias docstring
pub type MyAlias = MyStruct;

impl MyAlias {
    pub fn method_on_myalias() {}
}

impl MyTrait for MyAlias {}
