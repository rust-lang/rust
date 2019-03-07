// Copyright 2019 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustc will remove one of the two redundant references to foo below.  Depending on which one gets
// removed, we'll get a linker error on SOME platforms (like Linux).  On these platforms, when a
// library is referenced, the linker will only pull in the symbols needed _at that point in time_.
// If a later library depends on additional symbols from the library, they will not have been
// pulled in, and you'll get undefined symbols errors.
//
// So in this example, we need to ensure that rustc keeps the _later_ reference to foo, and not the
// former one.

extern "C" {
    fn bar();
    fn baz();
}

fn main() {
    unsafe {
        bar();
        baz();
    }
}
