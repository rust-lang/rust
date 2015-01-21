// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #7580

// ignore-pretty
//
// Expanded pretty printing causes resolve conflicts.

// error-pattern:panic works

use std::*;

fn main() {
    panic!("panic works")
}
