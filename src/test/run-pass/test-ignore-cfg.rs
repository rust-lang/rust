// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test --cfg ignorecfg
// ignore-pretty: does not work well with `--test`

#[test]
#[ignore(cfg(ignorecfg))]
fn shouldignore() {
}

#[test]
#[ignore(cfg(noignorecfg))]
fn shouldnotignore() {
}

#[test]
fn checktests() {
    // Pull the tests out of the secreturn test module
    let tests = __test::TESTS;

    assert!(
        tests.iter().any(|t| t.desc.name.to_str().as_slice() == "shouldignore" &&
                         t.desc.ignore));

    assert!(
        tests.iter().any(|t| t.desc.name.to_str().as_slice() == "shouldnotignore" &&
                         !t.desc.ignore));
}
