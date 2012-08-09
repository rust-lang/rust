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

// Signals one live task from the queue.
fn signal_waitqueue(q: &waitqueue) -> bool {
    // The peek is mandatory to make sure recv doesn't block.
    if q.head.peek() {
        // Pop and send a wakeup signal. If the waiter was killed, its port
        // will have closed. Keep trying until we get a live task.
        if q.head.recv().try_send(()) {
            true
        } else {
            signal_waitqueue(q)
        }
    } else {
        false
    }
}

fn broadcast_waitqueue(q: &waitqueue) -> uint {
    let mut count = 0;
    while q.head.peek() {
        if q.head.recv().try_send(()) {
            count += 1;
        }
    }
    count
}

// The building-block used to make semaphores, mutexes, and rwlocks.
enum sem<Q: send> = exclusive<{
    mut count: int,
    waiters:   waitqueue,
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked:   Q,
}>;

impl<Q: send> &sem<Q> {
    fn acquire() {
        let mut waiter_nobe = none;
        unsafe {
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    // Create waiter nobe.
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
                if state.count <= 0 {
                    signal_waitqueue(&state.waiters);
                }
            }
        }
    }
}
// FIXME(#3154) move both copies of this into sem<Q>, and unify the 2 structs
impl &sem<()> {
    fn access<U>(blk: fn() -> U) -> U {
        let mut release = none;
        unsafe {
            do task::unkillable {
                self.acquire();
                release = some(sem_release(self));
            }
        }
        blk()
    }
}
impl &sem<waitqueue> {
    fn access<U>(blk: fn() -> U) -> U {
        let mut release = none;
        unsafe {
            do task::unkillable {
                self.acquire();
                release = some(sem_and_signal_release(self));
            }
        }
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

impl condvar {
    /// Atomically drop the associated lock, and block until a signal is sent.
    fn wait() {
        // This is needed for a failing condition variable to reacquire the
        // mutex during unwinding. As long as the wrapper (mutex, etc) is
        // bounded in when it gets released, this shouldn't hang forever.
        struct sem_and_signal_reacquire {
            sem: &sem<waitqueue>;
            new(sem: &sem<waitqueue>) { self.sem = sem; }
            drop unsafe {
                do task::unkillable {
                    self.sem.acquire();
                }
            }
        }

        // Create waiter nobe.
        let (signal_end, wait_end) = pipes::stream();
        let mut signal_end = some(signal_end);
        let mut reacquire = none;
        unsafe {
            do task::unkillable {
                // If yield checks start getting inserted anywhere, we can be
                // killed before or after enqueueing. Deciding whether to
                // unkillably reacquire the lock needs to happen atomically
                // wrt enqueuing.
                reacquire = some(sem_and_signal_reacquire(*self));

                // Release lock, 'atomically' enqueuing ourselves in so doing.
                do (***self).with |state| {
                    // Drop the lock.
                    // FIXME(#3145) investigate why factoring doesn't compile.
                    state.count += 1;
                    if state.count <= 0 {
                        signal_waitqueue(&state.waiters);
                    }
                    // Enqueue ourself to be woken up by a signaller.
                    let signal_end = option::swap_unwrap(&mut signal_end);
                    state.blocked.tail.send(signal_end);
                }
            }
        }
        // Unconditionally "block". (Might not actually block if a signaller
        // did send -- I mean 'unconditionally' in contrast with acquire().)
        let _ = wait_end.recv();
        // 'reacquire' will pick up the lock again in its destructor - it must
        // happen whether or not we are killed, and it needs to succeed at
        // reacquiring instead of itself dying.
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    fn signal() -> bool {
        unsafe {
            do (***self).with |state| {
                signal_waitqueue(&state.blocked)
            }
        }
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    fn broadcast() -> uint {
        unsafe {
            do (***self).with |state| {
                // FIXME(#3145) fix :broadcast_heavy
                broadcast_waitqueue(&state.blocked)
            }
        }
    }
}

impl &sem<waitqueue> {
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

impl &semaphore {
    /// Create a new handle to the semaphore.
    fn clone() -> semaphore { semaphore(sem((***self).clone())) }

    /**
     * Acquire a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    fn acquire() { (&**self).acquire() }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist. Won't block the caller.
     */
    fn release() { (&**self).release() }

    /// Run a function with ownership of one of the semaphore's resources.
    // FIXME(#3145): figure out whether or not this should get exported.
    fn access<U>(blk: fn() -> U) -> U { (&**self).access(blk) }
}

/****************************************************************************
 * Mutexes
 ****************************************************************************/

/**
 * A blocking, bounded-waiting, mutual exclusion lock with an associated
 * FIFO condition variable.
 * FIXME(#3145): document killability
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

impl &mutex {
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
    fn test_sem_acquire_release() {
        let s = ~new_semaphore(1);
        s.acquire();
        s.release();
        s.acquire();
    }
    #[test]
    fn test_sem_basic() {
        let s = ~new_semaphore(1);
        do s.access { }
    }
    #[test]
    fn test_sem_as_mutex() {
        let s = ~new_semaphore(1);
        let s2 = ~s.clone();
        do task::spawn {
            do s2.access {
                for 5.times { task::yield(); }
            }
        }
        do s.access {
            for 5.times { task::yield(); }
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
        for 5.times { task::yield(); }
        s.release();
        let _ = p.recv();

        /* Parent waits and child signals */
        let (c,p) = pipes::stream();
        let s = ~new_semaphore(0);
        let s2 = ~s.clone();
        do task::spawn {
            for 5.times { task::yield(); }
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
    fn test_mutex_lock() {
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

        fn access_shared(sharedstate: &mut int, m: &mutex, n: uint) {
            for n.times {
                do m.lock {
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

        // Child wakes up parent
        do m.lock_cond |cond| {
            let m2 = ~m.clone();
            do task::spawn {
                do m2.lock_cond |cond| {
                    let woken = cond.signal();
                    assert woken;
                }
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
            let woken = cond.signal();
            assert woken;
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
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_killed_simple() {
        // Mutex must get automatically unlocked if failed/killed within.
        let m = ~new_mutex();
        let m2 = ~m.clone();

        let result: result::result<(),()> = do task::try {
            do m2.lock {
                fail;
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do m.lock { }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_killed_cond() {
        // Getting killed during cond wait must not corrupt the mutex while
        // unwinding (e.g. double unlock).
        let m = ~new_mutex();
        let m2 = ~m.clone();

        let result: result::result<(),()> = do task::try {
            let (c,p) = pipes::stream();
            do task::spawn { // linked
                let _ = p.recv(); // wait for sibling to get in the mutex
                task::yield();
                fail;
            }
            do m2.lock_cond |cond| {
                c.send(()); // tell sibling go ahead
                cond.wait(); // block forever
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do m.lock_cond |cond| {
            let woken = cond.signal();
            // FIXME(#3145) - The semantics of pipes are not quite what I want
            // here - the pipe doesn't get 'terminated' if the child was
            // punted awake during failure.
            // assert !woken;
        }
    }
}
