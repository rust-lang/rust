/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */

export condvar;
export semaphore, new_semaphore;
export mutex, new_mutex;
export rwlock;

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

fn new_sem<Q: send>(count: int, +q: Q) -> sem<Q> {
    let (wait_tail, wait_head)  = pipes::stream();
    sem(exclusive({ mut count: count,
                    waiters: { head: wait_head, tail: wait_tail },
                    blocked: q }))
}
fn new_sem_and_signal(count: int) -> sem<waitqueue> {
    let (block_tail, block_head) = pipes::stream();
    new_sem(count, { head: block_head, tail: block_tail })
}

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

        // This is needed for a failing condition variable to reacquire the
        // mutex during unwinding. As long as the wrapper (mutex, etc) is
        // bounded in when it gets released, this shouldn't hang forever.
        struct sem_and_signal_reacquire {
            sem: &sem<waitqueue>;
            new(sem: &sem<waitqueue>) { self.sem = sem; }
            drop unsafe {
                // Needs to succeed, instead of itself dying.
                do task::unkillable {
                    self.sem.acquire();
                }
            }
        }
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
fn new_semaphore(count: int) -> semaphore { semaphore(new_sem(count, ())) }

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
fn new_mutex() -> mutex { mutex(new_sem_and_signal(1)) }

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

// NB: Wikipedia - Readers-writers_problem#The_third_readers-writers_problem

struct rwlock_inner {
    read_mode:  bool;
    read_count: uint;
}

/// A blocking, no-starvation, reader-writer lock with an associated condvar.
struct rwlock {
    order_lock:  semaphore;
    access_lock: sem<waitqueue>;
    state:       arc::exclusive<rwlock_inner>;
}

/// Create a new rwlock.
fn rwlock() -> rwlock {
    rwlock { order_lock: new_semaphore(1), access_lock: new_sem_and_signal(1),
             state: arc::exclusive(rwlock_inner { read_mode:  false,
                                                  read_count: 0 }) }
}

impl &rwlock {
    /// Create a new handle to the rwlock.
    fn clone() -> rwlock {
        rwlock { order_lock:  (&(self.order_lock)).clone(),
                 access_lock: sem((*self.access_lock).clone()),
                 state:       self.state.clone() }
    }

    /**
     * Run a function with the rwlock in read mode. Calls to 'read' from other
     * tasks may run concurrently with this one.
     */
    fn read<U>(blk: fn() -> U) -> U {
        unsafe {
            do task::unkillable {
                do (&self.order_lock).access {
                    let mut first_reader = false;
                    do self.state.with |state| {
                        state.read_mode = true;
                        first_reader = (state.read_count == 0);
                        state.read_count += 1;
                    }
                    if first_reader {
                        (&self.access_lock).acquire();
                    }
                }
            }
        }
        let _z = rwlock_release_read(self);
        blk()
    }

    /**
     * Run a function with the rwlock in write mode. No calls to 'read' or
     * 'write' from other tasks will run concurrently with this one.
     */
    fn write<U>(blk: fn() -> U) -> U {
        unsafe {
            do task::unkillable {
                (&self.order_lock).acquire();
                (&self.access_lock).acquire();
                (&self.order_lock).release();
            }
        }
        let _z = rwlock_release_write(self);
        blk()
    }

    /**
     * As write(), but also with a handle to a condvar. Waiting on this
     * condvar will allow readers and writers alike to take the rwlock before
     * the waiting task is signalled.
     */
    fn write_cond<U>(_blk: fn(condvar) -> U) -> U {
        fail ~"Need implement lock order lock before access lock";
    }

    // to-do implement downgrade
}

// FIXME(#3136) should go inside of write() and read() respectively
struct rwlock_release_write {
    lock: &rwlock;
    new(lock: &rwlock) { self.lock = lock; }
    drop unsafe {
        do task::unkillable { (&self.lock.access_lock).release(); }
    }
}
struct rwlock_release_read {
    lock: &rwlock;
    new(lock: &rwlock) { self.lock = lock; }
    drop unsafe {
        do task::unkillable {
            let mut last_reader = false;
            do self.lock.state.with |state| {
                assert state.read_mode;
                state.read_count -= 1;
                last_reader = (state.read_count == 0);
            }
            if last_reader {
                (&self.lock.access_lock).release();
            }
        }
    }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    /************************************************************************
     * Semaphore tests
     ************************************************************************/
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
    /************************************************************************
     * Mutex tests
     ************************************************************************/
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
    #[cfg(test)]
    fn test_mutex_cond_broadcast_helper(num_waiters: uint) {
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
    #[test]
    fn test_mutex_cond_broadcast() {
        test_mutex_cond_broadcast_helper(12);
    }
    #[test]
    fn test_mutex_cond_broadcast_none() {
        test_mutex_cond_broadcast_helper(0);
    }
    #[test]
    fn test_mutex_cond_no_waiter() {
        let m = ~new_mutex();
        let m2 = ~m.clone();
        do task::try {
            do m.lock_cond |_x| { }
        };
        do m2.lock_cond |cond| {
            assert !cond.signal();
        }
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
            let _woken = cond.signal();
            // FIXME(#3145) - The semantics of pipes are not quite what I want
            // here - the pipe doesn't get 'terminated' if the child was
            // punted awake during failure.
            // assert !woken;
        }
    }
    /************************************************************************
     * Reader/writer lock tests
     ************************************************************************/
    #[cfg(test)]
    fn lock_rwlock_in_mode(x: &rwlock, reader: bool, blk: fn()) {
        if reader { x.read(blk); } else { x.write(blk); }
    }
    #[cfg(test)]
    fn test_rwlock_exclusion(reader1: bool, reader2: bool) {
        // Test mutual exclusion between readers and writers. Just like the
        // mutex mutual exclusion test, a ways above.
        let (c,p) = pipes::stream();
        let x = ~rwlock();
        let x2 = ~x.clone();
        let sharedstate = ~0;
        let ptr = ptr::addr_of(*sharedstate);
        do task::spawn {
            let sharedstate = unsafe { unsafe::reinterpret_cast(ptr) };
            access_shared(sharedstate, x2, reader1, 10);
            c.send(());
        }
        access_shared(sharedstate, x, reader2, 10);
        let _ = p.recv();

        assert *sharedstate == 20;

        fn access_shared(sharedstate: &mut int, x: &rwlock, reader: bool,
                         n: uint) {
            for n.times {
                do lock_rwlock_in_mode(x, reader) {
                    let oldval = *sharedstate;
                    task::yield();
                    *sharedstate = oldval + 1;
                }
            }
        }
    }
    #[test]
    fn test_rwlock_readers_wont_modify_the_data() {
        test_rwlock_exclusion(true, false);
        test_rwlock_exclusion(false, true);
    }
    #[test]
    fn test_rwlock_writers_and_writers() {
        test_rwlock_exclusion(false, false);
    }
    #[test]
    fn test_rwlock_readers_and_readers() {
        // Much like sem_multi_resource.
        let x = ~rwlock();
        let x2 = ~x.clone();
        let (c1,p1) = pipes::stream();
        let (c2,p2) = pipes::stream();
        do task::spawn {
            do x2.read {
                let _ = p2.recv();
                c1.send(());
            }
        }
        do x.read {
            c2.send(());
            let _ = p1.recv();
        }
    }
    #[cfg(test)] #[ignore(cfg(windows))]
    fn rwlock_kill_helper(reader1: bool, reader2: bool) {
        // Mutex must get automatically unlocked if failed/killed within.
        let x = ~rwlock();
        let x2 = ~x.clone();

        let result: result::result<(),()> = do task::try {
            do lock_rwlock_in_mode(x2, reader1) {
                fail;
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do lock_rwlock_in_mode(x, reader2) { }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_reader_killed_writer() { rwlock_kill_helper(true, false); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_writer_killed_reader() { rwlock_kill_helper(false,true ); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_reader_killed_reader() { rwlock_kill_helper(true, true ); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_writer_killed_writer() { rwlock_kill_helper(false,false); }
}
