// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _ = (~"foo").as_bytes_with_null();
    let _ = (@"foo").as_bytes_with_null();

    // a plain static slice is null terminated, but such a slice can
    // be sliced shorter (i.e. become non-null terminated) and still
    // have the static lifetime
    let foo: &'static str = "foo";
    let _ = foo.as_bytes_with_null();
     //~^ ERROR does not implement any method in scope named `as_bytes_with_null`
}
