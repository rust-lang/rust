// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check-search-index

#![feature(associated_consts)]

pub trait Foo {
    // @has assoc_consts/trait.Foo.html '//*[@class="rust trait"]' \
    //      'const FOO: usize;'
    // @has - '//*[@id="associatedconstant.FOO"]' 'const FOO'
    const FOO: usize;
}

pub struct Bar;

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.BAR"]' \
    //      'const BAR: usize = 3'
    pub const BAR: usize = 3;
}

/* !search-index
{
    "assoc_consts": {
        "assoc_consts::Bar": [
            "Struct"
        ],
        "assoc_consts::Bar<Struct>::BAR": [
            "AssociatedConst"
        ],
        "assoc_consts::Foo": [
            "Trait"
        ],
        "assoc_consts::Foo<Trait>::FOO": [
            "AssociatedConst"
        ]
    }
}
*/
