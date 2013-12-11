// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:linkage-visibility.rs
// xfail-fast check-fast doesn't like 'extern mod'
// xfail-android: FIXME(#10379)
// xfail-win32: std::unstable::dynamic_lib does not work on win32 well

extern mod foo = "linkage-visibility";

fn main() {
    foo::test();
    foo::foo2::<int>();
    foo::foo();
}
