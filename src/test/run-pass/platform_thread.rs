// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Jump back and forth between the OS main thread and a new scheduler.
// The OS main scheduler should continue to be available and not terminate
// while it is not in use.

pub fn main() {
    run(100);
}

fn run(i: int) {

    debug!(i);

    if i == 0 {
        return;
    }

    let mut builder = task::task();
    builder.sched_mode(task::PlatformThread);
    builder.unlinked();
    do builder.spawn {
        task::yield();
        let mut builder = task::task();
        builder.sched_mode(task::SingleThreaded);
        builder.unlinked();
        do builder.spawn {
            task::yield();
            run(i - 1);
            task::yield();
        }
        task::yield();
    }
}
