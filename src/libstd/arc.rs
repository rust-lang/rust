/**
 * Concurrency-enabled mechanisms for sharing mutable and/or immutable state
 * between tasks.
 */

import unsafe::{shared_mutable_state, clone_shared_mutable_state,
                get_shared_mutable_state, get_shared_immutable_state};
import sync::{condvar, mutex, rwlock};

export arc, clone, get;
export mutex_arc, rw_arc;

/****************************************************************************
 * Immutable ARC
 ****************************************************************************/

/// An atomically reference counted wrapper for shared immutable state.
struct arc<T: const send> { x: shared_mutable_state<T>; }

/// Create an atomically reference counted wrapper.
fn arc<T: const send>(+data: T) -> arc<T> {
    arc { x: unsafe { shared_mutable_state(data) } }
}

/**
 * Access the underlying data in an atomically reference counted
 * wrapper.
 */
fn get<T: const send>(rc: &arc<T>) -> &T {
    unsafe { get_shared_immutable_state(&rc.x) }
}

/**
 * Duplicate an atomically reference counted wrapper.
 *
 * The resulting two `arc` objects will point to the same underlying data
 * object. However, one of the `arc` objects can be sent to another task,
 * allowing them to share the underlying data.
 */
fn clone<T: const send>(rc: &arc<T>) -> arc<T> {
    arc { x: unsafe { clone_shared_mutable_state(&rc.x) } }
}

/****************************************************************************
 * Mutex protected ARC (unsafe)
 ****************************************************************************/

struct mutex_arc_inner<T: send> { lock: mutex; failed: bool; data: T; }
/// An ARC with mutable data protected by a blocking mutex.
struct mutex_arc<T: send> { x: shared_mutable_state<mutex_arc_inner<T>>; }

/// Create a mutex-protected ARC with the supplied data.
fn mutex_arc<T: send>(+user_data: T) -> mutex_arc<T> {
    let data = mutex_arc_inner {
        lock: mutex(), failed: false, data: user_data
    };
    mutex_arc { x: unsafe { shared_mutable_state(data) } }
}

impl<T: send> &mutex_arc<T> {
    /// Duplicate a mutex-protected ARC, as arc::clone.
    fn clone() -> mutex_arc<T> {
        // NB: Cloning the underlying mutex is not necessary. Its reference
        // count would be exactly the same as the shared state's.
        mutex_arc { x: unsafe { clone_shared_mutable_state(&self.x) } }
    }

    /**
     * Access the underlying mutable data with mutual exclusion from other
     * tasks. The argument closure will be run with the mutex locked; all
     * other tasks wishing to access the data will block until the closure
     * finishes running.
     *
     * The reason this function is 'unsafe' is because it is possible to
     * construct a circular reference among multiple ARCs by mutating the
     * underlying data. This creates potential for deadlock, but worse, this
     * will guarantee a memory leak of all involved ARCs. Using mutex ARCs
     * inside of other ARCs is safe in absence of circular references.
     *
     * If you wish to nest mutex_arcs, one strategy for ensuring safety at
     * runtime is to add a "nesting level counter" inside the stored data, and
     * when traversing the arcs, assert that they monotonically decrease.
     *
     * # Failure
     *
     * Failing while inside the ARC will unlock the ARC while unwinding, so
     * that other tasks won't block forever. It will also poison the ARC:
     * any tasks that subsequently try to access it (including those already
     * blocked on the mutex) will also fail immediately.
     */
    #[inline(always)]
    unsafe fn access<U>(blk: fn(x: &mut T) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        // Borrowck would complain about this if the function were not already
        // unsafe. See borrow_rwlock, far below.
        do (&state.lock).lock {
            check_poison(true, state.failed);
            state.failed = true;
            let result = blk(&mut state.data);
            state.failed = false;
            result
        }
    }
/* FIXME(#3145): Make this compile; borrowck doesn't like it..?
    /// As access(), but with a condvar, as sync::mutex.lock_cond().
    #[inline(always)]
    unsafe fn access_cond<U>(blk: fn(x: &mut T, condvar) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do (&state.lock).lock_cond |cond| {
            check_poison(true, state.failed);
            state.failed = true;
            let result = blk(&mut state.data, cond);
            state.failed = false;
            result
        }
    }
*/
}

// Common code for {mutex.access,rwlock.write}{,_cond}.
#[inline(always)]
fn check_poison(is_mutex: bool, failed: bool) {
    if failed {
        if is_mutex {
            fail ~"Poisoned mutex_arc - another task failed inside!";
        } else {
            fail ~"Poisoned rw_arc - another task failed inside!";
        }
    }
}

/****************************************************************************
 * R/W lock protected ARC
 ****************************************************************************/

struct rw_arc_inner<T: const send> { lock: rwlock; failed: bool; data: T; }
/**
 * A dual-mode ARC protected by a reader-writer lock. The data can be accessed
 * mutably or immutably, and immutably-accessing tasks may run concurrently.
 *
 * Unlike mutex_arcs, rw_arcs are safe, because they cannot be nested.
 */
struct rw_arc<T: const send> {
    x: shared_mutable_state<rw_arc_inner<T>>;
    mut cant_nest: ();
}

/// Create a reader/writer ARC with the supplied data.
fn rw_arc<T: const send>(+user_data: T) -> rw_arc<T> {
    let data = rw_arc_inner {
        lock: rwlock(), failed: false, data: user_data
    };
    rw_arc { x: unsafe { shared_mutable_state(data) }, cant_nest: () }
}

impl<T: const send> &rw_arc<T> {
    /// Duplicate a rwlock-protected ARC, as arc::clone.
    fn clone() -> rw_arc<T> {
        rw_arc { x: unsafe { clone_shared_mutable_state(&self.x) },
                 cant_nest: () }
    }

    /**
     * Access the underlying data mutably. Locks the rwlock in write mode;
     * other readers and writers will block.
     *
     * # Failure
     *
     * Failing while inside the ARC will unlock the ARC while unwinding, so
     * that other tasks won't block forever. As mutex_arc.access, it will also
     * poison the ARC, so subsequent readers and writers will both also fail.
     */
    #[inline(always)]
    fn write<U>(blk: fn(x: &mut T) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do borrow_rwlock(state).write {
            check_poison(false, state.failed);
            state.failed = true;
            let result = blk(&mut state.data);
            state.failed = false;
            result
        }
    }
/* FIXME(#3145): Make this compile; borrowck doesn't like it..?
    /// As write(), but with a condvar, as sync::rwlock.write_cond().
    #[inline(always)]
    fn write_cond<U>(blk: fn(x: &mut T, condvar) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do borrow_rwlock(state).write_cond |cond| {
            check_poison(false, state.failed);
            state.failed = true;
            let result = blk(&mut state.data, cond);
            state.failed = false;
            result
        }
    }
*/
    /**
     * Access the underlying data immutably. May run concurrently with other
     * reading tasks.
     *
     * # Failure
     *
     * Failing will unlock the ARC while unwinding. However, unlike all other
     * access modes, this will not poison the ARC.
     */
    fn read<U>(blk: fn(x: &T) -> U) -> U {
        let state = unsafe { get_shared_immutable_state(&self.x) };
        do (&state.lock).read {
            check_poison(false, state.failed);
            blk(&state.data)
        }
    }
}

// Borrowck rightly complains about immutably aliasing the rwlock in order to
// lock it. This wraps the unsafety, with the justification that the 'lock'
// field is never overwritten; only 'failed' and 'data'.
fn borrow_rwlock<T: const send>(state: &mut rw_arc_inner<T>) -> &rwlock {
    unsafe { unsafe::reinterpret_cast(&state.lock) }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    import comm::*;

    #[test]
    fn manually_share_arc() {
        let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = arc::arc(v);

        let p = port();
        let c = chan(p);

        do task::spawn() {
            let p = port();
            c.send(chan(p));

            let arc_v = p.recv();

            let v = *arc::get::<~[int]>(&arc_v);
            assert v[3] == 4;
        };

        let c = p.recv();
        c.send(arc::clone(&arc_v));

        assert (*arc::get(&arc_v))[2] == 3;

        log(info, arc_v);
    }

    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_mutex_arc_poison() {
        let arc = ~mutex_arc(1);
        let arc2 = ~arc.clone();
        do task::try {
            do arc2.access |one| {
                assert *one == 2;
            }
        };
        do arc.access |one| {
            assert *one == 1;
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_wr() {
        let arc = ~rw_arc(1);
        let arc2 = ~arc.clone();
        do task::try {
            do arc2.write |one| {
                assert *one == 2;
            }
        };
        do arc.read |one| {
            assert *one == 1;
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_ww() {
        let arc = ~rw_arc(1);
        let arc2 = ~arc.clone();
        do task::try {
            do arc2.write |one| {
                assert *one == 2;
            }
        };
        do arc.write |one| {
            assert *one == 1;
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_rr() {
        let arc = ~rw_arc(1);
        let arc2 = ~arc.clone();
        do task::try {
            do arc2.read |one| {
                assert *one == 2;
            }
        };
        do arc.read |one| {
            assert *one == 1;
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_rw() {
        let arc = ~rw_arc(1);
        let arc2 = ~arc.clone();
        do task::try {
            do arc2.read |one| {
                assert *one == 2;
            }
        };
        do arc.write |one| {
            assert *one == 1;
        }
    }

    #[test]
    fn test_rw_arc() {
        let arc = ~rw_arc(0);
        let arc2 = ~arc.clone();
        let (c,p) = pipes::stream();

        do task::spawn {
            do arc2.write |num| {
                for 10.times {
                    let tmp = *num;
                    *num = -1;
                    task::yield();
                    *num = tmp + 1;
                }
                c.send(());
            }
        }
        // Readers try to catch the writer in the act
        let mut children = ~[];
        for 5.times {
            let arc3 = ~arc.clone();
            do task::task().future_result(|+r| vec::push(children, r)).spawn {
                do arc3.read |num| {
                    assert *num >= 0;
                }
            }
        }
        // Wait for children to pass their asserts
        for vec::each(children) |r| { future::get(r); }
        // Wait for writer to finish
        p.recv();
        do arc.read |num| { assert *num == 10; }
    }
}
