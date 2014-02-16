// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty
// ignore-test linked failure

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**
 * Test performance of killing many tasks in a taskgroup.
 * Along the way, tests various edge cases of ancestor group management.
 * In particular, this tries to get each grandchild task to hit the
 * "nobe_is_dead" case in each_ancestor only during task exit, but not during
 * task spawn. This makes sure that defunct ancestor groups are handled correctly
 * w.r.t. possibly leaving stale *rust_tasks lying around.
 */

// Creates in the background 'num_tasks' tasks, all blocked forever.
// Doesn't return until all such tasks are ready, but doesn't block forever itself.

use std::comm::{stream, Chan};
use std::os;
use std::result;
use std::task;
use std::uint;

fn grandchild_group(num_tasks: uint) {
    let (po, ch) = stream();
    let ch = Chan::new(ch);

    for _ in range(0, num_tasks) {
        let ch = ch.clone();
        let mut t = task::task();
        t.spawn(proc() { // linked
            ch.send(());
            let (p, _c) = stream::<()>();
            p.recv(); // block forever
        });
    }
    error!("Grandchild group getting started");
    for _ in range(0, num_tasks) {
        // Make sure all above children are fully spawned; i.e., enlisted in
        // their ancestor groups.
        po.recv();
    }
    error!("Grandchild group ready to go.");
    // Master grandchild task exits early.
}

fn spawn_supervised_blocking(myname: &str, f: proc()) {
    let mut builder = task::task();
    let res = builder.future_result();
    builder.supervised();
    builder.spawn(f);
    error!("{} group waiting", myname);
    let x = res.recv();
    assert!(x.is_ok());
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~[~"", ~"100000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100"]
    } else {
        args.clone()
    };

    let num_tasks = from_str::<uint>(args[1]).unwrap();

    // Main group #0 waits for unsupervised group #1.
    // Grandparent group #1 waits for middle group #2, then fails, killing #3.
    // Middle group #2 creates grandchild_group #3, waits for it to be ready, exits.
    let x: result::Result<(), ~Any> = task::try(proc() { // unlinked
        spawn_supervised_blocking("grandparent", proc() {
            spawn_supervised_blocking("middle", proc() {
                grandchild_group(num_tasks);
            });
            // When grandchild group is ready to go, make the middle group exit.
            error!("Middle group wakes up and exits");
        });
        // Grandparent group waits for middle group to be gone, then fails
        error!("Grandparent group wakes up and fails");
        fail!();
    });
    assert!(x.is_err());
}
