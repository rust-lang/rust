// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast feature doesn't work
#[feature(trace_macros, log_syntax)];

// make sure these macros can be used as in the various places that
// macros can occur.

// items
trace_macros!(false)
log_syntax!()

fn main() {

    // statements
    trace_macros!(false);
    log_syntax!();

    // expressions
    (trace_macros!(false),
     log_syntax!());
}
