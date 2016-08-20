// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(item_like_imports)]
#![allow(unused)]

// Like other items, private imports can be imported and used non-lexically in paths.
mod a {
    use a as foo;
    use self::foo::foo as bar;

    mod b {
        use super::bar;
    }
}

fn main() {}
