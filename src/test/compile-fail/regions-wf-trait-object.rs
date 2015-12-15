// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.t

// Check that the explicit lifetime bound (`'b`, in this example) must
// outlive all the superbound from the trait (`'a`, in this example).

trait TheTrait<'t>: 't { }

struct Foo<'a,'b> {
    x: Box<TheTrait<'a>+'b> //~ ERROR E0478
}

fn main() { }
