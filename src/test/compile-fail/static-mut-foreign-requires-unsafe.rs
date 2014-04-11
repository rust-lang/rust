// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



mod test {
    use std::kinds::marker;

    pub struct NonSharable {
        pub field: uint,
        noshare: marker::NoShare
    }
}

extern {
    static mut a: test::NonSharable;
}

fn main() {
    a.field += 3;     //~ ERROR: this use of mutable static requires unsafe function or block
    a.field = 4;      //~ ERROR: this use of mutable static requires unsafe function or block
    let _b = a;       //~ ERROR: this use of mutable static requires unsafe function or block
}
