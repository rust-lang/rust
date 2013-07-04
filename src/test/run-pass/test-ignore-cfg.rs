// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test --cfg ignorecfg
// xfail-fast

extern mod extra;

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
        tests.iter().any_(|t| t.desc.name.to_str() == ~"shouldignore" && t.desc.ignore));

    assert!(
        tests.iter().any_(|t| t.desc.name.to_str() == ~"shouldnotignore" && !t.desc.ignore));
}
