// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test transitive analysis for associated types. Collected types
// should be normalized and new obligations generated.

// pretty-expanded FIXME #23616

use std::borrow::{ToOwned, Cow};

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(Cow::Borrowed("foo"));
}
