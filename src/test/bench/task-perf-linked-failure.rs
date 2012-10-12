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
fn grandchild_group(num_tasks: uint) {
    let po = comm::Port();
    let ch = comm::Chan(&po);

    for num_tasks.times {
        do task::spawn { // linked
            comm::send(ch, ());
            comm::recv(comm::Port::<()>()); // block forever
        }
    }
    error!("Grandchild group getting started");
    for num_tasks.times {
        // Make sure all above children are fully spawned; i.e., enlisted in
        // their ancestor groups.
        comm::recv(po);
    }
    error!("Grandchild group ready to go.");
    // Master grandchild task exits early.
}

fn spawn_supervised_blocking(myname: &str, +f: fn~()) {
    let mut res = None;
    task::task().future_result(|+r| res = Some(r)).supervised().spawn(f);
    error!("%s group waiting", myname);
    let x = future::get(&option::unwrap(res));
    assert x == task::Success;
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100000"]
    } else if args.len() <= 1u {
        ~[~"", ~"100"]
    } else {
        copy args
    };

    let num_tasks = uint::from_str(args[1]).get();

    // Main group #0 waits for unsupervised group #1.
    // Grandparent group #1 waits for middle group #2, then fails, killing #3.
    // Middle group #2 creates grandchild_group #3, waits for it to be ready, exits.
    let x: result::Result<(),()> = do task::try { // unlinked
        do spawn_supervised_blocking("grandparent") {
            do spawn_supervised_blocking("middle") {
                grandchild_group(num_tasks);
            }
            // When grandchild group is ready to go, make the middle group exit.
            error!("Middle group wakes up and exits");
        }
        // Grandparent group waits for middle group to be gone, then fails
        error!("Grandparent group wakes up and fails");
        fail;
    };
    assert x.is_err();
}
