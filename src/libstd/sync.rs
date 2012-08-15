// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
/**
 * The concurrency primitives you know and love.
 *
 * Maybe once we have a "core exports x only to std" mechanism, these can be
 * in std.
 */

export condvar, semaphore, mutex, mutex_with_condvars;
export rwlock, rwlock_with_condvars, rwlock_write_mode, rwlock_read_mode;

// FIXME (#3119) This shouldn't be a thing exported from core.
import unsafe::{Exclusive, exclusive};

/****************************************************************************
 * Internals
 ****************************************************************************/

// Each waiting task receives on one of these.
#[doc(hidden)]
type wait_end = pipes::port_one<()>;
#[doc(hidden)]
type signal_end = pipes::chan_one<()>;
// A doubly-ended queue of waiting tasks.
#[doc(hidden)]
struct waitqueue { head: pipes::port<signal_end>;
                   tail: pipes::chan<signal_end>; }

// Signals one live task from the queue.
#[doc(hidden)]
fn signal_waitqueue(q: &waitqueue) -> bool {
    // The peek is mandatory to make sure recv doesn't block.
    if q.head.peek() {
        // Pop and send a wakeup signal. If the waiter was killed, its port
        // will have closed. Keep trying until we get a live task.
        if pipes::try_send_one(q.head.recv(), ()) {
            true
        } else {
            signal_waitqueue(q)
        }
    } else {
        false
    }
}

#[doc(hidden)]
fn broadcast_waitqueue(q: &waitqueue) -> uint {
    let mut count = 0;
    while q.head.peek() {
        if pipes::try_send_one(q.head.recv(), ()) {
            count += 1;
        }
    }
    count
}

// The building-block used to make semaphores, mutexes, and rwlocks.
#[doc(hidden)]
struct sem_inner<Q> {
    mut count: int;
    waiters:   waitqueue;
    // Can be either unit or another waitqueue. Some sems shouldn't come with
    // a condition variable attached, others should.
    blocked:   Q;
}
#[doc(hidden)]
enum sem<Q: send> = Exclusive<sem_inner<Q>>;

#[doc(hidden)]
fn new_sem<Q: send>(count: int, +q: Q) -> sem<Q> {
    let (wait_tail, wait_head)  = pipes::stream();
    sem(exclusive(sem_inner {
        mut count: count,
        waiters: waitqueue { head: wait_head, tail: wait_tail },
        blocked: q }))
}
#[doc(hidden)]
fn new_sem_and_signal(count: int, num_condvars: uint) -> sem<~[waitqueue]> {
    let mut queues = ~[];
    for num_condvars.times {
        let (block_tail, block_head) = pipes::stream();
        vec::push(queues, waitqueue { head: block_head, tail: block_tail });
    }
    new_sem(count, queues)
}

#[doc(hidden)]
impl<Q: send> &sem<Q> {
    fn acquire() {
        let mut waiter_nobe = none;
        unsafe {
            do (**self).with |state| {
                state.count -= 1;
                if state.count < 0 {
                    // Create waiter nobe.
                    let (signal_end, wait_end) = pipes::oneshot();
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
            let _ = pipes::recv_one(option::unwrap(waiter_nobe));
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
#[doc(hidden)]
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
#[doc(hidden)]
impl &sem<~[waitqueue]> {
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
#[doc(hidden)]
struct sem_release {
    sem: &sem<()>;
    new(sem: &sem<()>) { self.sem = sem; }
    drop { self.sem.release(); }
}
#[doc(hidden)]
struct sem_and_signal_release {
    sem: &sem<~[waitqueue]>;
    new(sem: &sem<~[waitqueue]>) { self.sem = sem; }
    drop { self.sem.release(); }
}

/// A mechanism for atomic-unlock-and-deschedule blocking and signalling.
struct condvar { priv sem: &sem<~[waitqueue]>; drop { } }

impl &condvar {
    /**
     * Atomically drop the associated lock, and block until a signal is sent.
     *
     * # Failure
     * A task which is killed (i.e., by linked failure with another task)
     * while waiting on a condition variable will wake up, fail, and unlock
     * the associated lock as it unwinds.
     */
    fn wait() { self.wait_on(0) }
    /**
     * As wait(), but can specify which of multiple condition variables to
     * wait on. Only a signal_on() or broadcast_on() with the same condvar_id
     * will wake this thread.
     *
     * The associated lock must have been initialised with an appropriate
     * number of condvars. The condvar_id must be between 0 and num_condvars-1
     * or else this call will fail.
     *
     * wait() is equivalent to wait_on(0).
     */
    fn wait_on(condvar_id: uint) {
        // Create waiter nobe.
        let (signal_end, wait_end) = pipes::oneshot();
        let mut wait_end   = some(wait_end);
        let mut signal_end = some(signal_end);
        let mut reacquire = none;
        let mut out_of_bounds = none;
        unsafe {
            do task::unkillable {
                // Release lock, 'atomically' enqueuing ourselves in so doing.
                do (**self.sem).with |state| {
                    if condvar_id < vec::len(state.blocked) {
                        // Drop the lock.
                        state.count += 1;
                        if state.count <= 0 {
                            signal_waitqueue(&state.waiters);
                        }
                        // Enqueue ourself to be woken up by a signaller.
                        let signal_end = option::swap_unwrap(&mut signal_end);
                        state.blocked[condvar_id].tail.send(signal_end);
                    } else {
                        out_of_bounds = some(vec::len(state.blocked));
                    }
                }

                // If yield checks start getting inserted anywhere, we can be
                // killed before or after enqueueing. Deciding whether to
                // unkillably reacquire the lock needs to happen atomically
                // wrt enqueuing.
                if out_of_bounds.is_none() {
                    reacquire = some(sem_and_signal_reacquire(self.sem));
                }
            }
        }
        do check_cvar_bounds(out_of_bounds, condvar_id, "cond.wait_on()") {
            // Unconditionally "block". (Might not actually block if a
            // signaller already sent -- I mean 'unconditionally' in contrast
            // with acquire().)
            let _ = pipes::recv_one(option::swap_unwrap(&mut wait_end));
        }

        // This is needed for a failing condition variable to reacquire the
        // mutex during unwinding. As long as the wrapper (mutex, etc) is
        // bounded in when it gets released, this shouldn't hang forever.
        struct sem_and_signal_reacquire {
            sem: &sem<~[waitqueue]>;
            new(sem: &sem<~[waitqueue]>) { self.sem = sem; }
            drop unsafe {
                // Needs to succeed, instead of itself dying.
                do task::unkillable {
                    self.sem.acquire();
                }
            }
        }
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    fn signal() -> bool { self.signal_on(0) }
    /// As signal, but with a specified condvar_id. See wait_on.
    fn signal_on(condvar_id: uint) -> bool {
        let mut out_of_bounds = none;
        let mut result = false;
        unsafe {
            do (**self.sem).with |state| {
                if condvar_id < vec::len(state.blocked) {
                    result = signal_waitqueue(&state.blocked[condvar_id]);
                } else {
                    out_of_bounds = some(vec::len(state.blocked));
                }
            }
        }
        do check_cvar_bounds(out_of_bounds, condvar_id, "cond.signal_on()") {
            result
        }
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    fn broadcast() -> uint { self.broadcast_on(0) }
    /// As broadcast, but with a specified condvar_id. See wait_on.
    fn broadcast_on(condvar_id: uint) -> uint {
        let mut out_of_bounds = none;
        let mut result = 0;
        unsafe {
            do (**self.sem).with |state| {
                if condvar_id < vec::len(state.blocked) {
                    // FIXME(#3145) fix :broadcast_heavy
                    result = broadcast_waitqueue(&state.blocked[condvar_id])
                } else {
                    out_of_bounds = some(vec::len(state.blocked));
                }
            }
        }
        do check_cvar_bounds(out_of_bounds, condvar_id, "cond.signal_on()") {
            result
        }
    }
}

// Checks whether a condvar ID was out of bounds, and fails if so, or does
// something else next on success.
#[inline(always)]
#[doc(hidden)]
fn check_cvar_bounds<U>(out_of_bounds: option<uint>, id: uint, act: &str,
                        blk: fn() -> U) -> U {
    match out_of_bounds {
        some(0) =>
            fail fmt!("%s with illegal ID %u - this lock has no condvars!",
                      act, id),
        some(length) =>
            fail fmt!("%s with illegal ID %u - ID must be less than %u",
                      act, id, length),
        none => blk()
    }
}

#[doc(hidden)]
impl &sem<~[waitqueue]> {
    // The only other place that condvars get built is rwlock_write_mode.
    fn access_cond<U>(blk: fn(c: &condvar) -> U) -> U {
        do self.access { blk(&condvar { sem: self }) }
    }
}

/****************************************************************************
 * Semaphores
 ****************************************************************************/

/// A counting, blocking, bounded-waiting semaphore.
struct semaphore { priv sem: sem<()>; }

/// Create a new semaphore with the specified count.
fn semaphore(count: int) -> semaphore {
    semaphore { sem: new_sem(count, ()) }
}

impl &semaphore {
    /// Create a new handle to the semaphore.
    fn clone() -> semaphore { semaphore { sem: sem((*self.sem).clone()) } }

    /**
     * Acquire a resource represented by the semaphore. Blocks if necessary
     * until resource(s) become available.
     */
    fn acquire() { (&self.sem).acquire() }

    /**
     * Release a held resource represented by the semaphore. Wakes a blocked
     * contending task, if any exist. Won't block the caller.
     */
    fn release() { (&self.sem).release() }

    /// Run a function with ownership of one of the semaphore's resources.
    fn access<U>(blk: fn() -> U) -> U { (&self.sem).access(blk) }
}

/****************************************************************************
 * Mutexes
 ****************************************************************************/

/**
 * A blocking, bounded-waiting, mutual exclusion lock with an associated
 * FIFO condition variable.
 *
 * # Failure
 * A task which fails while holding a mutex will unlock the mutex as it
 * unwinds.
 */
struct mutex { priv sem: sem<~[waitqueue]>; }

/// Create a new mutex, with one associated condvar.
fn mutex() -> mutex { mutex_with_condvars(1) }
/**
 * Create a new mutex, with a specified number of associated condvars. This
 * will allow calling wait_on/signal_on/broadcast_on with condvar IDs between
 * 0 and num_condvars-1. (If num_condvars is 0, lock_cond will be allowed but
 * any operations on the condvar will fail.)
 */
fn mutex_with_condvars(num_condvars: uint) -> mutex {
    mutex { sem: new_sem_and_signal(1, num_condvars) }
}

impl &mutex {
    /// Create a new handle to the mutex.
    fn clone() -> mutex { mutex { sem: sem((*self.sem).clone()) } }

    /// Run a function with ownership of the mutex.
    fn lock<U>(blk: fn() -> U) -> U { (&self.sem).access(blk) }

    /// Run a function with ownership of the mutex and a handle to a condvar.
    fn lock_cond<U>(blk: fn(c: &condvar) -> U) -> U {
        (&self.sem).access_cond(blk)
    }
}

/****************************************************************************
 * Reader-writer locks
 ****************************************************************************/

// NB: Wikipedia - Readers-writers_problem#The_third_readers-writers_problem

#[doc(hidden)]
struct rwlock_inner {
    read_mode:  bool;
    read_count: uint;
}

/**
 * A blocking, no-starvation, reader-writer lock with an associated condvar.
 *
 * # Failure
 * A task which fails while holding an rwlock will unlock the rwlock as it
 * unwinds.
 */
struct rwlock {
    /* priv */ order_lock:  semaphore;
    /* priv */ access_lock: sem<~[waitqueue]>;
    /* priv */ state:       Exclusive<rwlock_inner>;
}

/// Create a new rwlock, with one associated condvar.
fn rwlock() -> rwlock { rwlock_with_condvars(1) }

/**
 * Create a new rwlock, with a specified number of associated condvars.
 * Similar to mutex_with_condvars.
 */
fn rwlock_with_condvars(num_condvars: uint) -> rwlock {
    rwlock { order_lock: semaphore(1),
             access_lock: new_sem_and_signal(1, num_condvars),
             state: exclusive(rwlock_inner { read_mode:  false,
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
        let mut release = none;
        unsafe {
            do task::unkillable {
                do (&self.order_lock).access {
                    let mut first_reader = false;
                    do self.state.with |state| {
                        first_reader = (state.read_count == 0);
                        state.read_count += 1;
                    }
                    if first_reader {
                        (&self.access_lock).acquire();
                        do self.state.with |state| {
                            // Must happen *after* getting access_lock. If
                            // this is set while readers are waiting, but
                            // while a writer holds the lock, the writer will
                            // be confused if they downgrade-then-unlock.
                            state.read_mode = true;
                        }
                    }
                }
                release = some(rwlock_release_read(self));
            }
        }
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
                do (&self.access_lock).access {
                    (&self.order_lock).release();
                    task::rekillable(blk)
                }
            }
        }
    }

    /**
     * As write(), but also with a handle to a condvar. Waiting on this
     * condvar will allow readers and writers alike to take the rwlock before
     * the waiting task is signalled. (Note: a writer that waited and then
     * was signalled might reacquire the lock before other waiting writers.)
     */
    fn write_cond<U>(blk: fn(c: &condvar) -> U) -> U {
        // NB: You might think I should thread the order_lock into the cond
        // wait call, so that it gets waited on before access_lock gets
        // reacquired upon being woken up. However, (a) this would be not
        // pleasant to implement (and would mandate a new 'rw_cond' type) and
        // (b) I think violating no-starvation in that case is appropriate.
        unsafe {
            do task::unkillable {
                (&self.order_lock).acquire();
                do (&self.access_lock).access_cond |cond| {
                    (&self.order_lock).release();
                    do task::rekillable { blk(cond) }
                }
            }
        }
    }

    /**
     * As write(), but with the ability to atomically 'downgrade' the lock;
     * i.e., to become a reader without letting other writers get the lock in
     * the meantime (such as unlocking and then re-locking as a reader would
     * do). The block takes a "write mode token" argument, which can be
     * transformed into a "read mode token" by calling downgrade(). Example:
     * ~~~
     * do lock.write_downgrade |write_mode| {
     *     do (&write_mode).write_cond |condvar| {
     *         ... exclusive access ...
     *     }
     *     let read_mode = lock.downgrade(write_mode);
     *     do (&read_mode).read {
     *         ... shared access ...
     *     }
     * }
     * ~~~
     */
    fn write_downgrade<U>(blk: fn(+rwlock_write_mode) -> U) -> U {
        // Implementation slightly different from the slicker 'write's above.
        // The exit path is conditional on whether the caller downgrades.
        let mut _release = none;
        unsafe {
            do task::unkillable {
                (&self.order_lock).acquire();
                (&self.access_lock).acquire();
                (&self.order_lock).release();
            }
            _release = some(rwlock_release_downgrade(self));
        }
        blk(rwlock_write_mode { lock: self })
    }

    /// To be called inside of the write_downgrade block.
    fn downgrade(+token: rwlock_write_mode) -> rwlock_read_mode {
        if !ptr::ref_eq(self, token.lock) {
            fail ~"Can't downgrade() with a different rwlock's write_mode!";
        }
        unsafe {
            do task::unkillable {
                let mut first_reader = false;
                do self.state.with |state| {
                    assert !state.read_mode;
                    state.read_mode = true;
                    first_reader = (state.read_count == 0);
                    state.read_count += 1;
                }
                if !first_reader {
                    // Guaranteed not to let another writer in, because
                    // another reader was holding the order_lock. Hence they
                    // must be the one to get the access_lock (because all
                    // access_locks are acquired with order_lock held).
                    (&self.access_lock).release();
                }
            }
        }
        rwlock_read_mode { lock: token.lock }
    }
}

// FIXME(#3136) should go inside of read()
#[doc(hidden)]
struct rwlock_release_read {
    lock: &rwlock;
    new(lock: &rwlock) { self.lock = lock; }
    drop unsafe {
        do task::unkillable {
            let mut last_reader = false;
            do self.lock.state.with |state| {
                assert state.read_mode;
                assert state.read_count > 0;
                state.read_count -= 1;
                if state.read_count == 0 {
                    last_reader = true;
                    state.read_mode = false;
                }
            }
            if last_reader {
                (&self.lock.access_lock).release();
            }
        }
    }
}

// FIXME(#3136) should go inside of downgrade()
#[doc(hidden)]
struct rwlock_release_downgrade {
    lock: &rwlock;
    new(lock: &rwlock) { self.lock = lock; }
    drop unsafe {
        do task::unkillable {
            let mut writer_or_last_reader = false;
            do self.lock.state.with |state| {
                if state.read_mode {
                    assert state.read_count > 0;
                    state.read_count -= 1;
                    if state.read_count == 0 {
                        // Case 1: Writer downgraded & was the last reader
                        writer_or_last_reader = true;
                        state.read_mode = false;
                    } else {
                        // Case 2: Writer downgraded & was not the last reader
                    }
                } else {
                    // Case 3: Writer did not downgrade
                    writer_or_last_reader = true;
                }
            }
            if writer_or_last_reader {
                (&self.lock.access_lock).release();
            }
        }
    }
}

/// The "write permission" token used for rwlock.write_downgrade().
struct rwlock_write_mode { /* priv */ lock: &rwlock; drop { } }
/// The "read permission" token used for rwlock.write_downgrade().
struct rwlock_read_mode  { priv lock: &rwlock; drop { } }

impl &rwlock_write_mode {
    /// Access the pre-downgrade rwlock in write mode.
    fn write<U>(blk: fn() -> U) -> U { blk() }
    /// Access the pre-downgrade rwlock in write mode with a condvar.
    fn write_cond<U>(blk: fn(c: &condvar) -> U) -> U {
        blk(&condvar { sem: &self.lock.access_lock })
    }
}
impl &rwlock_read_mode {
    /// Access the post-downgrade rwlock in read mode.
    fn read<U>(blk: fn() -> U) -> U { blk() }
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
        let s = ~semaphore(1);
        s.acquire();
        s.release();
        s.acquire();
    }
    #[test]
    fn test_sem_basic() {
        let s = ~semaphore(1);
        do s.access { }
    }
    #[test]
    fn test_sem_as_mutex() {
        let s = ~semaphore(1);
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
        let s = ~semaphore(0);
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
        let s = ~semaphore(0);
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
        let s = ~semaphore(2);
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
            let s = ~semaphore(1);
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
        let m = ~mutex();
        let m2 = ~m.clone();
        let sharedstate = ~0;
        let ptr = ptr::addr_of(*sharedstate);
        do task::spawn {
            let sharedstate: &mut int =
                unsafe { unsafe::reinterpret_cast(ptr) };
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
        let m = ~mutex();

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
        let m = ~mutex();
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
        let m = ~mutex();
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
        let m = ~mutex();
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
        let m = ~mutex();
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
            assert !woken;
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_killed_broadcast() {
        let m = ~mutex();
        let m2 = ~m.clone();
        let (c,p) = pipes::stream();

        let result: result::result<(),()> = do task::try {
            let mut sibling_convos = ~[];
            for 2.times {
                let (c,p) = pipes::stream();
                let c = ~mut some(c);
                vec::push(sibling_convos, p);
                let mi = ~m2.clone();
                // spawn sibling task
                do task::spawn { // linked
                    do mi.lock_cond |cond| {
                        let c = option::swap_unwrap(c);
                        c.send(()); // tell sibling to go ahead
                        let _z = send_on_failure(c);
                        cond.wait(); // block forever
                    }
                }
            }
            for vec::each(sibling_convos) |p| {
                let _ = p.recv(); // wait for sibling to get in the mutex
            }
            do m2.lock { }
            c.send(sibling_convos); // let parent wait on all children
            fail;
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        for vec::each(p.recv()) |p| { p.recv(); } // wait on all its siblings
        do m.lock_cond |cond| {
            let woken = cond.broadcast();
            assert woken == 0;
        }
        struct send_on_failure {
            c: pipes::chan<()>;
            new(+c: pipes::chan<()>) { self.c = c; }
            drop { self.c.send(()); }
        }
    }
    #[test]
    fn test_mutex_cond_signal_on_0() {
        // Tests that signal_on(0) is equivalent to signal().
        let m = ~mutex();
        do m.lock_cond |cond| {
            let m2 = ~m.clone();
            do task::spawn {
                do m2.lock_cond |cond| {
                    cond.signal_on(0);
                }
            }
            cond.wait();
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_different_conds() {
        let result = do task::try {
            let m = ~mutex_with_condvars(2);
            let m2 = ~m.clone();
            let (c,p) = pipes::stream();
            do task::spawn {
                do m2.lock_cond |cond| {
                    c.send(());
                    cond.wait_on(1);
                }
            }
            let _ = p.recv();
            do m.lock_cond |cond| {
                if !cond.signal_on(0) {
                    fail; // success; punt sibling awake.
                }
            }
        };
        assert result.is_err();
    }
    #[test] #[ignore(cfg(windows))]
    fn test_mutex_no_condvars() {
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.wait(); }
        };
        assert result.is_err();
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.signal(); }
        };
        assert result.is_err();
        let result = do task::try {
            let m = ~mutex_with_condvars(0);
            do m.lock_cond |cond| { cond.broadcast(); }
        };
        assert result.is_err();
    }
    /************************************************************************
     * Reader/writer lock tests
     ************************************************************************/
    #[cfg(test)]
    enum rwlock_mode { read, write, downgrade, downgrade_read }
    #[cfg(test)]
    fn lock_rwlock_in_mode(x: &rwlock, mode: rwlock_mode, blk: fn()) {
        match mode {
            read => x.read(blk),
            write => x.write(blk),
            downgrade =>
                do x.write_downgrade |mode| {
                    // FIXME(#2282)
                    let mode = unsafe { unsafe::transmute_region(&mode) };
                    mode.write(blk);
                },
            downgrade_read =>
                do x.write_downgrade |mode| {
                    let mode = x.downgrade(mode);
                    // FIXME(#2282)
                    let mode = unsafe { unsafe::transmute_region(&mode) };
                    mode.read(blk);
                },
        }
    }
    #[cfg(test)]
    fn test_rwlock_exclusion(+x: ~rwlock, mode1: rwlock_mode,
                             mode2: rwlock_mode) {
        // Test mutual exclusion between readers and writers. Just like the
        // mutex mutual exclusion test, a ways above.
        let (c,p) = pipes::stream();
        let x2 = ~x.clone();
        let sharedstate = ~0;
        let ptr = ptr::addr_of(*sharedstate);
        do task::spawn {
            let sharedstate: &mut int =
                unsafe { unsafe::reinterpret_cast(ptr) };
            access_shared(sharedstate, x2, mode1, 10);
            c.send(());
        }
        access_shared(sharedstate, x, mode2, 10);
        let _ = p.recv();

        assert *sharedstate == 20;

        fn access_shared(sharedstate: &mut int, x: &rwlock, mode: rwlock_mode,
                         n: uint) {
            for n.times {
                do lock_rwlock_in_mode(x, mode) {
                    let oldval = *sharedstate;
                    task::yield();
                    *sharedstate = oldval + 1;
                }
            }
        }
    }
    #[test]
    fn test_rwlock_readers_wont_modify_the_data() {
        test_rwlock_exclusion(~rwlock(), read, write);
        test_rwlock_exclusion(~rwlock(), write, read);
        test_rwlock_exclusion(~rwlock(), read, downgrade);
        test_rwlock_exclusion(~rwlock(), downgrade, read);
    }
    #[test]
    fn test_rwlock_writers_and_writers() {
        test_rwlock_exclusion(~rwlock(), write, write);
        test_rwlock_exclusion(~rwlock(), write, downgrade);
        test_rwlock_exclusion(~rwlock(), downgrade, write);
        test_rwlock_exclusion(~rwlock(), downgrade, downgrade);
    }
    #[cfg(test)]
    fn test_rwlock_handshake(+x: ~rwlock, mode1: rwlock_mode,
                             mode2: rwlock_mode, make_mode2_go_first: bool) {
        // Much like sem_multi_resource.
        let x2 = ~x.clone();
        let (c1,p1) = pipes::stream();
        let (c2,p2) = pipes::stream();
        do task::spawn {
            if !make_mode2_go_first {
                let _ = p2.recv(); // parent sends to us once it locks, or ...
            }
            do lock_rwlock_in_mode(x2, mode2) {
                if make_mode2_go_first {
                    c1.send(()); // ... we send to it once we lock
                }
                let _ = p2.recv();
                c1.send(());
            }
        }
        if make_mode2_go_first {
            let _ = p1.recv(); // child sends to us once it locks, or ...
        }
        do lock_rwlock_in_mode(x, mode1) {
            if !make_mode2_go_first {
                c2.send(()); // ... we send to it once we lock
            }
            c2.send(());
            let _ = p1.recv();
        }
    }
    #[test]
    fn test_rwlock_readers_and_readers() {
        test_rwlock_handshake(~rwlock(), read, read, false);
        // The downgrader needs to get in before the reader gets in, otherwise
        // they cannot end up reading at the same time.
        test_rwlock_handshake(~rwlock(), downgrade_read, read, false);
        test_rwlock_handshake(~rwlock(), read, downgrade_read, true);
        // Two downgrade_reads can never both end up reading at the same time.
    }
    #[test]
    fn test_rwlock_downgrade_unlock() {
        // Tests that downgrade can unlock the lock in both modes
        let x = ~rwlock();
        do lock_rwlock_in_mode(x, downgrade) { }
        test_rwlock_handshake(x, read, read, false);
        let y = ~rwlock();
        do lock_rwlock_in_mode(y, downgrade_read) { }
        test_rwlock_exclusion(y, write, write);
    }
    #[test]
    fn test_rwlock_read_recursive() {
        let x = ~rwlock();
        do x.read { do x.read { } }
    }
    #[test]
    fn test_rwlock_cond_wait() {
        // As test_mutex_cond_wait above.
        let x = ~rwlock();

        // Child wakes up parent
        do x.write_cond |cond| {
            let x2 = ~x.clone();
            do task::spawn {
                do x2.write_cond |cond| {
                    let woken = cond.signal();
                    assert woken;
                }
            }
            cond.wait();
        }
        // Parent wakes up child
        let (chan,port) = pipes::stream();
        let x3 = ~x.clone();
        do task::spawn {
            do x3.write_cond |cond| {
                chan.send(());
                cond.wait();
                chan.send(());
            }
        }
        let _ = port.recv(); // Wait until child gets in the rwlock
        do x.read { } // Must be able to get in as a reader in the meantime
        do x.write_cond |cond| { // Or as another writer
            let woken = cond.signal();
            assert woken;
        }
        let _ = port.recv(); // Wait until child wakes up
        do x.read { } // Just for good measure
    }
    #[cfg(test)]
    fn test_rwlock_cond_broadcast_helper(num_waiters: uint, dg1: bool,
                                         dg2: bool) {
        // Much like the mutex broadcast test. Downgrade-enabled.
        fn lock_cond(x: &rwlock, downgrade: bool, blk: fn(c: &condvar)) {
            if downgrade {
                do x.write_downgrade |mode| {
                    // FIXME(#2282)
                    let mode = unsafe { unsafe::transmute_region(&mode) };
                    mode.write_cond(blk)
                }
            } else {
                x.write_cond(blk)
            }
        }
        let x = ~rwlock();
        let mut ports = ~[];

        for num_waiters.times {
            let xi = ~x.clone();
            let (chan, port) = pipes::stream();
            vec::push(ports, port);
            do task::spawn {
                do lock_cond(xi, dg1) |cond| {
                    chan.send(());
                    cond.wait();
                    chan.send(());
                }
            }
        }

        // wait until all children get in the mutex
        for ports.each |port| { let _ = port.recv(); }
        do lock_cond(x, dg2) |cond| {
            let num_woken = cond.broadcast();
            assert num_woken == num_waiters;
        }
        // wait until all children wake up
        for ports.each |port| { let _ = port.recv(); }
    }
    #[test]
    fn test_rwlock_cond_broadcast() {
        test_rwlock_cond_broadcast_helper(0, true, true);
        test_rwlock_cond_broadcast_helper(0, true, false);
        test_rwlock_cond_broadcast_helper(0, false, true);
        test_rwlock_cond_broadcast_helper(0, false, false);
        test_rwlock_cond_broadcast_helper(12, true, true);
        test_rwlock_cond_broadcast_helper(12, true, false);
        test_rwlock_cond_broadcast_helper(12, false, true);
        test_rwlock_cond_broadcast_helper(12, false, false);
    }
    #[cfg(test)] #[ignore(cfg(windows))]
    fn rwlock_kill_helper(mode1: rwlock_mode, mode2: rwlock_mode) {
        // Mutex must get automatically unlocked if failed/killed within.
        let x = ~rwlock();
        let x2 = ~x.clone();

        let result: result::result<(),()> = do task::try {
            do lock_rwlock_in_mode(x2, mode1) {
                fail;
            }
        };
        assert result.is_err();
        // child task must have finished by the time try returns
        do lock_rwlock_in_mode(x, mode2) { }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_reader_killed_writer() { rwlock_kill_helper(read, write); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_writer_killed_reader() { rwlock_kill_helper(write,read ); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_reader_killed_reader() { rwlock_kill_helper(read, read ); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_writer_killed_writer() { rwlock_kill_helper(write,write); }
    #[test] #[ignore(cfg(windows))]
    fn test_rwlock_kill_downgrader() {
        rwlock_kill_helper(downgrade, read);
        rwlock_kill_helper(read, downgrade);
        rwlock_kill_helper(downgrade, write);
        rwlock_kill_helper(write, downgrade);
        rwlock_kill_helper(downgrade_read, read);
        rwlock_kill_helper(read, downgrade_read);
        rwlock_kill_helper(downgrade_read, write);
        rwlock_kill_helper(write, downgrade_read);
        rwlock_kill_helper(downgrade_read, downgrade);
        rwlock_kill_helper(downgrade_read, downgrade);
        rwlock_kill_helper(downgrade, downgrade_read);
        rwlock_kill_helper(downgrade, downgrade_read);
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rwlock_downgrade_cant_swap() {
        // Tests that you can't downgrade with a different rwlock's token.
        let x = ~rwlock();
        let y = ~rwlock();
        do x.write_downgrade |xwrite| {
            let mut xopt = some(xwrite);
            do y.write_downgrade |_ywrite| {
                y.downgrade(option::swap_unwrap(&mut xopt));
                error!("oops, y.downgrade(x) should have failed!");
            }
        }
    }
}
