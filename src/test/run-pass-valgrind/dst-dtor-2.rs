// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut DROP_RAN: isize = 0;

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { DROP_RAN += 1; }
    }
}

struct Fat<T: ?Sized> {
    f: T
}

pub fn main() {
    {
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        let _x: Box<Fat<[Foo]>> = Box::<Fat<[Foo; 3]>>::new(Fat { f: [Foo, Foo, Foo] });
    }
    unsafe {
        assert!(DROP_RAN == 3);
    }
}
