 // Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Destructuring struct variants would ICE where regular structs wouldn't

enum Foo {
    VBar { num: isize }
}

struct SBar { num: isize }

pub fn main() {
    let vbar = Foo::VBar { num: 1 };
    let Foo::VBar { num } = vbar;
    assert_eq!(num, 1);

    let sbar = SBar { num: 2 };
    let SBar { num } = sbar;
    assert_eq!(num, 2);
}
