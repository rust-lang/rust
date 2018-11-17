// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod inner {
    pub struct SomeStruct;

    impl SomeStruct {
        pub fn new() -> SomeStruct { SomeStruct }
    }
}

// @has rename/index.html
// @has - '//a/@href' 'struct.MyStruct.html'
// @has rename/struct.MyStruct.html
// @has - '//code' 'impl MyStruct'
// @!has - '//code' 'impl SomeStruct'
pub use inner::SomeStruct as MyStruct;
