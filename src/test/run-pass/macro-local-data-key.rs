// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

local_data_key!(foo: int)

mod bar {
    local_data_key!(pub baz: f64)
}

pub fn main() {
    assert!(foo.get().is_none());
    assert!(bar::baz.get().is_none());

    foo.replace(Some(3));
    bar::baz.replace(Some(-10.0));

    assert_eq!(*foo.get().unwrap(), 3);
    assert_eq!(*bar::baz.get().unwrap(), -10.0);
}
