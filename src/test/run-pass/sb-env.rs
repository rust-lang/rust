// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to see how environment sandboxing is working
// rustc-env:BLOBBIE=FOO
// rustc-env:FOOBIE=BAR
// rustc-env:THINGY=OLD
// compile-flags:--env-allow BLOBBIE --env-define ZIPPIE=BLARG --env-define THINGY=NEW

fn main() {
    assert_eq!(option_env!("BLOBBIE"), Some("FOO")); // actual environment, allowed to be seen
    assert_eq!(option_env!("ZIPPIE"), Some("BLARG")); // defined on command-line
    assert_eq!(option_env!("THINGY"), Some("NEW")); // overridden on command-line
    assert_eq!(option_env!("FOOBIE"), None); // actual environment, but not allowed to be seen
}
