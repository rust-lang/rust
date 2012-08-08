/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */

export condvar;
export semaphore, new_semaphore;
export mutex, new_mutex;

// FIXME (#3119) This shouldn't be a thing exported from core.
import arc::exclusive;

/****************************************************************************
 * Internals
 ****************************************************************************/

// Each waiting task receives on one of these. FIXME #3125 make these oneshot.
type wait_end = pipes::port<()>;
type signal_end = pipes::chan<()>;
// A doubly-ended queue of waiting tasks.
type waitqueue = { head: pipes::port<signal_end>,
                   tail: pipes::chan<signal_end> };

// The building-block used to make semaphores, lock-and-signals, and rwlocks.
enum sem<Q: send> = exclusive<{
    mut count: int,
    waiters:   waitqueue,
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked:   Q,
}>;

impl sem<Q: send> for &sem<Q> {
    fn acquire() {
        let mut waiter_nobe = none;
        unsafe {
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    let (signal_end, wait_end) = pipes::stream();
                    // Tell outer scope we need to block.
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
    fn release() {
        unsafe {
            do (**self).with |state| {
                state.count += 1;
                // The peek is mandatory to make sure recv doesn't block.
                if state.count <= 0 && state.waiters.head.peek() {
                    // Pop off the waitqueue and send a wakeup signal. If the
                    // waiter was killed, its port will have closed, and send
                    // will fail. Keep trying until we get a live task.
                    state.waiters.head.recv().send(());
                    // FIXME(#3145) use kill-friendly version when ready
                    // while !state.waiters.head.recv().try_send(()) { }
                }
            }
        }
    }
}
// FIXME(#3154) move both copies of this into sem<Q>, and unify the 2 structs
impl sem_access for &sem<()> {
    fn access<U>(blk: fn() -> U) -> U {
        self.acquire();
        let _x = sem_release(self);
        blk()
    }
}
impl sem_access for &sem<waitqueue> {
    fn access<U>(blk: fn() -> U) -> U {
        self.acquire();
        let _x = sem_and_signal_release(self);
        blk()
    }
}

// FIXME(#3136) should go inside of access()
struct sem_release {
    sem: &sem<()>;
    new(sem: &sem<()>) { self.sem = sem; }
    drop { self.sem.release(); }
}
struct sem_and_signal_release {
    sem: &sem<waitqueue>;
    new(sem: &sem<waitqueue>) { self.sem = sem; }
    drop { self.sem.release(); }
}

/// A mechanism for atomic-unlock-and-deschedule blocking and signalling.
enum condvar = &sem<waitqueue>;

impl condvar for condvar {
    /// Atomically drop the associated lock, and block until a signal is sent.
    fn wait() {
        let (signal_end, wait_end) = pipes::stream();
        let mut signal_end = some(signal_end);
        unsafe {
            do (***self).with |state| {
                // Drop the lock.
                // FIXME(#3145) investigate why factoring doesn't compile.
                state.count += 1;
                if state.count <= 0 && state.waiters.head.peek() {
                    state.waiters.head.recv().send(());
                    // FIXME(#3145) use kill-friendly version when ready
                }
                // Enqueue ourself to be woken up by a signaller.
                state.blocked.tail.send(option::swap_unwrap(&mut signal_end));
            }
        }
        // Unconditionally "block". (Might not actually block if a signaller
        // did send -- I mean 'unconditionally' in contrast with acquire().)
        let _ = wait_end.recv();
        // Pick up the lock again. FIXME(#3145): unkillable? destructor?
        (*self).acquire();
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    fn signal() -> bool {
        unsafe {
            do (***self).with |state| {
                if state.blocked.head.peek() {
                    state.blocked.head.recv().send(());
                    // FIXME(#3145) use kill-friendly version when ready
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    fn broadcast() -> uint {
        unsafe {
            do (***self).with |state| {
                let mut count = 0;
                while state.blocked.head.peek() {
                    // This is already kill-friendly.
                    state.blocked.head.recv().send(());
                    count += 1;
                }
                count
            }
        }
    }
}

impl sem_and_signal for &sem<waitqueue> {
    fn access_cond<U>(blk: fn(condvar) -> U) -> U {
        do self.access { blk(condvar(self)) }
    }
}

/****************************************************************************
 * Semaphores
 ****************************************************************************/

/// A counting, blocking, bounded-waiting semaphore.
enum semaphore = sem<()>;

/// Create a new semaphore with the specified count.
fn new_semaphore(count: int) -> semaphore {
    let (wait_tail, wait_head)  = pipes::stream();
    semaphore(sem(exclusive({ mut count: count,
                              waiters: { head: wait_head, tail: wait_tail },
                              blocked: () })))
}

impl semaphore for &semaphore {
    /// Create a new handle to the semaphore.
    fn clone() -> semaphore { semaphore(sem((***self).clone())) }

    /**
     * Acquire a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    fn acquire() { (&**self).acquire() }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist.
     */
    fn release() { (&**self).release() }

    /// Run a function with ownership of one of the semaphore's resources.
    fn access<U>(blk: fn() -> U) -> U { (&**self).access(blk) }
}

/****************************************************************************
 * Mutexes
 ****************************************************************************/

/**
 * A blocking, bounded-waiting, mutual exclusion lock with an associated
 * FIFO condition variable.
 */
enum mutex = sem<waitqueue>;

/// Create a new mutex.
fn new_mutex() -> mutex {
    let (wait_tail,  wait_head)  = pipes::stream();
    let (block_tail, block_head) = pipes::stream();
    mutex(sem(exclusive({ mut count: 1,
                          waiters: { head: wait_head,  tail: wait_tail  },
                          blocked: { head: block_head, tail: block_tail } })))
}

impl mutex for &mutex {
    /// Create a new handle to the mutex.
    fn clone() -> mutex { mutex(sem((***self).clone())) }

    /// Run a function with ownership of the mutex.
    fn lock<U>(blk: fn() -> U) -> U { (&**self).access(blk) }

    /// Run a function with ownership of the mutex and a handle to a condvar.
    fn lock_cond<U>(blk: fn(condvar) -> U) -> U {
        (&**self).access_cond(blk)
    }
}

/****************************************************************************
 * Reader-writer locks
 ****************************************************************************/

// FIXME(#3145) implement

/****************************************************************************
 * Tests
 ****************************************************************************/

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
    #[test]
    fn test_mutex() {
        // Unsafely achieve shared state, and do the textbook
        // "load tmp <- ptr; inc tmp; store ptr <- tmp" dance.
        let (c,p) = pipes::stream();
        let m = ~new_mutex();
        let m2 = ~m.clone();
        let sharedstate = ~0;
        let ptr = ptr::addr_of(*sharedstate);
        do task::spawn {
            let sharedstate = unsafe { unsafe::reinterpret_cast(ptr) };
            access_shared(sharedstate, m2, 10);
            c.send(());
        }
        access_shared(sharedstate, m, 10);
        let _ = p.recv();

        assert *sharedstate == 20;

        fn access_shared(sharedstate: &mut int, sem: &mutex, n: uint) {
            for n.times {
                do sem.lock {
                    let oldval = *sharedstate;
                    task::yield();
                    *sharedstate = oldval + 1;
                }
            }
        }
    }
    #[test]
    fn test_mutex_cond_wait() {
        let m = ~new_mutex();
        let mut m2 = some(~m.clone());

        // Child wakes up parent
        do m.lock_cond |cond| {
            let m2 = option::swap_unwrap(&mut m2);
            do task::spawn {
                do m2.lock_cond |cond| { cond.signal(); }
            }
            cond.wait();
        }
        // Parent wakes up child
        let (chan,port) = pipes::stream();
        let m3 = ~m.clone();
        do task::spawn {
            do m3.lock_cond |cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            }
        }
        let _ = port.recv(); // Wait until child gets in the mutex
        do m.lock_cond |cond| {
            cond.signal();
        }
        let _ = port.recv(); // Wait until child wakes up
    }
    #[test]
    fn test_mutex_cond_broadcast() {
        let num_waiters: uint = 12;
        let m = ~new_mutex();
        let mut ports = ~[];

        for num_waiters.times {
            let mi = ~m.clone();
            let (chan, port) = pipes::stream();
            vec::push(ports, port);
            do task::spawn {
                do mi.lock_cond |cond| {
                    chan.send(());
                    cond.wait();
                    chan.send(());
                }
            }
        }

        // wait until all children get in the mutex
        for ports.each |port| { let _ = port.recv(); }
        do m.lock_cond |cond| {
            let num_woken = cond.broadcast();
            assert num_woken == num_waiters;
        }
        // wait until all children wake up
        for ports.each |port| { let _ = port.recv(); }
    }
}
