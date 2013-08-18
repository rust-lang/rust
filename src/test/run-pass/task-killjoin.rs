// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test linked failure
// xfail-win32

// Create a task that is supervised by another task, join the supervised task
// from the supervising task, then fail the supervised task. The supervised
// task will kill the supervising task, waking it up. The supervising task no
// longer needs to be wakened when the supervised task exits.

use std::task;

fn supervised() {
    // Deschedule to make sure the supervisor joins before we fail. This is
    // currently not needed because the supervisor runs first, but I can
    // imagine that changing.
    task::deschedule();
    fail!();
}

fn supervisor() {
    // Unsupervise this task so the process doesn't return a failure status as
    // a result of the main task being killed.
    let f = supervised;
    task::try(supervised);
}

pub fn main() {
    task::spawn_unlinked(supervisor)
}
