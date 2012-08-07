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

fn waitqueue() -> waitqueue {
    let (tail, head) = pipes::stream();
    { head: head, tail: tail }
}

/// A counting, blocking, bounded-waiting semaphore.
enum semaphore = exclusive<semaphore_inner>;
type semaphore_inner = {
    mut count: int,
    waiters:   waitqueue,
    //blocked:   waitqueue,
};

/// Create a new semaphore with the specified count.
fn new_semaphore(count: int) -> semaphore {
    semaphore(exclusive({ mut count: count,
                          waiters: waitqueue(), /* blocked: waitqueue() */ }))
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
    fn acquire() {
        let mut waiter_nobe = none;
        unsafe {
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    let (signal_end,wait_end) = pipes::stream();
                    waiter_nobe = some(wait_end);
                    // Enqueue ourself.
                    state.waiters.tail.send(signal_end);
                }
            }
        }
        // Uncomment if you wish to test for sem races. Not valgrind-friendly.
        /* for 1000.times { task::yield(); } */
        // Need to wait outside the exclusive.
        if waiter_nobe.is_some() {
            let _ = option::unwrap(waiter_nobe).recv();
        }
    }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist.
     */
    fn release() {
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
        self.acquire();
        let _x = sem_release(self);
        blk()
    }
}

// FIXME(#3136) should go inside of access()
struct sem_release {
    sem: &semaphore;
    new(sem: &semaphore) { self.sem = sem; }
    drop { self.sem.release(); }
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
            s2.acquire();
            c.send(());
        }
        for 10.times { task::yield(); }
        s.release();
        let _ = p.recv();

        /* Parent waits and child signals */
        let (c,p) = pipes::stream();
        let s = ~new_semaphore(0);
        let s2 = ~s.clone();
        do task::spawn {
            for 10.times { task::yield(); }
            s2.release();
            let _ = p.recv();
        }
        s.acquire();
        c.send(());
    }
    #[test]
    fn test_sem_mutual_exclusion() {
        // Unsafely achieve shared state, and do the textbook
        // "load tmp <- ptr; inc tmp; store ptr <- tmp" dance.
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
    fn test_sem_multi_resource() {
        // Parent and child both get in the critical section at the same
        // time, and shake hands.
        let s = ~new_semaphore(2);
        let s2 = ~s.clone();
        let (c1,p1) = pipes::stream();
        let (c2,p2) = pipes::stream();
        do task::spawn {
            do s2.access {
                let _ = p2.recv();
                c1.send(());
            }
        }
        do s.access {
            c2.send(());
            let _ = p1.recv();
        }
    }
    #[test]
    fn test_sem_runtime_friendly_blocking() {
        // Force the runtime to schedule two threads on the same sched_loop.
        // When one blocks, it should schedule the other one.
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
