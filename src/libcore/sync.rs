/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */

export semaphore, new_semaphore;

// FIXME (#3119) This shouldn't be a thing exported from core.
import arc::exclusive;

// Each waiting task receives on one of these. FIXME #3125 make these oneshot.
type wait_end = pipes::port<()>;
type signal_end = pipes::chan<()>;
// A doubly-ended queue of waiting tasks.
type waitqueue = { head: pipes::port<signal_end>,
                   tail: pipes::chan<signal_end> };

fn new_waiter() -> (signal_end, wait_end) { pipes::stream() }

/// A counting semaphore.
enum semaphore = exclusive<{
    mut count: int,
    waiters:   waitqueue,
}>;

/// Create a new semaphore with the specified count.
fn new_semaphore(count: int) -> semaphore {
    let (tail, head) = pipes::stream();
    semaphore(exclusive({ mut count: count,
                          waiters: { head: head, tail: tail } }))
}

impl semaphore for &semaphore {
    /// Creates a new handle to the semaphore.
    fn clone() -> semaphore {
        semaphore((**self).clone())
    }

    /**
     * Acquires a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    fn wait() {
        let mut waiter_nobe = none;
        unsafe {
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    let (signal_end,wait_end) = new_waiter();
                    waiter_nobe = some(wait_end);
                    // Enqueue ourself.
                    state.waiters.tail.send(signal_end);
                }
            }
        }
        for 1000.times { task::yield(); }
        // Need to wait outside the exclusive.
        if waiter_nobe.is_some() {
            let _ = option::unwrap(waiter_nobe).recv();
        }
    }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist.
     */
    fn signal() {
        unsafe {
            do (**self).with |state| {
                state.count += 1;
                // The peek is mandatory to make sure recv doesn't block.
                if state.count >= 0 && state.waiters.head.peek() {
                    // Pop off the waitqueue and send a wakeup signal. If the
                    // waiter was killed, its port will have closed, and send
                    // will fail. Keep trying until we get a live task.
                    state.waiters.head.recv().send(());
                    // to-do: use this version when it's ready, kill-friendly.
                    // while !state.waiters.head.recv().try_send(()) { }
                }
            }
        }
    }

    /// Runs a function with ownership of one of the semaphore's resources.
    fn access<U>(blk: fn() -> U) -> U {
        self.wait();
        let _x = sem_release(self);
        blk()
    }
}

// FIXME(#3136) should go inside of access()
struct sem_release {
    sem: &semaphore;
    new(sem: &semaphore) { self.sem = sem; }
    drop { self.sem.signal(); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sem_as_mutex() {
        let s = ~new_semaphore(1);
        let s2 = ~s.clone();
        do task::spawn {
            do s2.access {
                for 10.times { task::yield(); }
            }
        }
        do s.access {
            for 10.times { task::yield(); }
        }
    }
    #[test]
    fn test_sem_as_cvar() {
        /* Child waits and parent signals */
        let (c,p) = pipes::stream();
        let s = ~new_semaphore(0);
        let s2 = ~s.clone();
        do task::spawn {
            s2.wait();
            c.send(());
        }
        for 10.times { task::yield(); }
        s.signal();
        let _ = p.recv();

        /* Parent waits and child signals */
        let (c,p) = pipes::stream();
        let s = ~new_semaphore(0);
        let s2 = ~s.clone();
        do task::spawn {
            for 10.times { task::yield(); }
            s2.signal();
            let _ = p.recv();
        }
        s.wait();
        c.send(());
    }
    #[test]
    fn test_sem_mutual_exclusion() {
        let (c,p) = pipes::stream();
        let s = ~new_semaphore(1);
        let s2 = ~s.clone();
        let sharedstate = ~0;
        let ptr = ptr::addr_of(*sharedstate);
        do task::spawn {
            let sharedstate = unsafe { unsafe::reinterpret_cast(ptr) };
            access_shared(sharedstate, s2, 10);
            c.send(());
        }
        access_shared(sharedstate, s, 10);
        let _ = p.recv();

        assert *sharedstate == 20;

        fn access_shared(sharedstate: &mut int, sem: &semaphore, n: uint) {
            for n.times {
                do sem.access {
                    let oldval = *sharedstate;
                    task::yield();
                    *sharedstate = oldval + 1;
                }
            }
        }
    }
    #[test]
    fn test_sem_runtime_friendly_blocking() {
        do task::spawn_sched(task::manual_threads(1)) {
            let s = ~new_semaphore(1);
            let s2 = ~s.clone();
            let (c,p) = pipes::stream();
            let child_data = ~mut some((s2,c));
            do s.access {
                let (s2,c) = option::swap_unwrap(child_data);
                do task::spawn {
                    c.send(());
                    do s2.access { }
                    c.send(());
                }
                let _ = p.recv(); // wait for child to come alive
                for 5.times { task::yield(); } // let the child contend
            }
            let _ = p.recv(); // wait for child to be done
        }
    }
}
