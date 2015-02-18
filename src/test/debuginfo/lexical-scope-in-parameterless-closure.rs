// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-lldb-version: 310

// compile-flags:-C debuginfo=1

// gdb-command:run
// lldb-command:run

// Nothing to do here really, just make sure it compiles. See issue #8513.
fn main() {
    let _ = ||();
    let _ = (1_usize..3).map(|_| 5);
}

