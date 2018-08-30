// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not error out because of a (False) ambiguity
// between the builtin rules for Sized and the where clause. Issue
// #20959.

// pretty-expanded FIXME #23616

fn foo<K>(x: Option<K>)
    where Option<K> : Sized
{
    let _y = x;
}

fn main() {
    foo(Some(22));
}
