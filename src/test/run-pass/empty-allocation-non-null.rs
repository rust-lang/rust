// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME (#22405): Replace `Box::new` with `box` here when/if possible.

// pretty-expanded FIXME #23616

pub fn main() {
    assert!(Some(Box::new(())).is_some());

    let xs: Box<[()]> = Box::<[(); 0]>::new([]);
    assert!(Some(xs).is_some());

    struct Foo;
    assert!(Some(Box::new(Foo)).is_some());

    let ys: Box<[Foo]> = Box::<[Foo; 0]>::new([]);
    assert!(Some(ys).is_some());
}
