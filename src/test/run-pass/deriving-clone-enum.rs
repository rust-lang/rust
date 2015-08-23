// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-pretty-expanded FIXME #15189

#[derive(Clone)]
enum E {
    A,
    B(()),
    C
}

pub fn main() {
    let mut foo = E::A.clone();

    // Test both code-paths of clone_from (same variant/different variant)
    foo.clone_from(&E::A);
    foo.clone_from(&E::B(()));
    foo.clone_from(&E::B(()));
}
