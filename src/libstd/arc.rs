// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
/**
 * Concurrency-enabled mechanisms for sharing mutable and/or immutable state
 * between tasks.
 */

use sync;
use sync::{Mutex, mutex_with_condvars, RWlock, rwlock_with_condvars};

use core::cast;
use core::pipes;
use core::prelude::*;
use core::private::{SharedMutableState, shared_mutable_state};
use core::private::{clone_shared_mutable_state, unwrap_shared_mutable_state};
use core::private::{get_shared_mutable_state, get_shared_immutable_state};
use core::ptr;
use core::task;
use core::util;

/// As sync::condvar, a mechanism for unlock-and-descheduling and signalling.
pub struct Condvar { is_mutex: bool, failed: &mut bool, cond: &sync::Condvar }

impl &Condvar {
    /// Atomically exit the associated ARC and block until a signal is sent.
    #[inline(always)]
    fn wait() { self.wait_on(0) }
    /**
     * Atomically exit the associated ARC and block on a specified condvar
     * until a signal is sent on that same condvar (as sync::cond.wait_on).
     *
     * wait() is equivalent to wait_on(0).
     */
    #[inline(always)]
    fn wait_on(condvar_id: uint) {
        assert !*self.failed;
        self.cond.wait_on(condvar_id);
        // This is why we need to wrap sync::condvar.
        check_poison(self.is_mutex, *self.failed);
    }
    /// Wake up a blocked task. Returns false if there was no blocked task.
    #[inline(always)]
    fn signal() -> bool { self.signal_on(0) }
    /**
     * Wake up a blocked task on a specified condvar (as
     * sync::cond.signal_on). Returns false if there was no blocked task.
     */
    #[inline(always)]
    fn signal_on(condvar_id: uint) -> bool {
        assert !*self.failed;
        self.cond.signal_on(condvar_id)
    }
    /// Wake up all blocked tasks. Returns the number of tasks woken.
    #[inline(always)]
    fn broadcast() -> uint { self.broadcast_on(0) }
    /**
     * Wake up all blocked tasks on a specified condvar (as
     * sync::cond.broadcast_on). Returns Returns the number of tasks woken.
     */
    #[inline(always)]
    fn broadcast_on(condvar_id: uint) -> uint {
        assert !*self.failed;
        self.cond.broadcast_on(condvar_id)
    }
}

/****************************************************************************
 * Immutable ARC
 ****************************************************************************/

/// An atomically reference counted wrapper for shared immutable state.
struct ARC<T: Const Owned> { x: SharedMutableState<T> }

/// Create an atomically reference counted wrapper.
pub fn ARC<T: Const Owned>(data: T) -> ARC<T> {
    ARC { x: unsafe { shared_mutable_state(move data) } }
}

/**
 * Access the underlying data in an atomically reference counted
 * wrapper.
 */
pub fn get<T: Const Owned>(rc: &a/ARC<T>) -> &a/T {
    unsafe { get_shared_immutable_state(&rc.x) }
}

/**
 * Duplicate an atomically reference counted wrapper.
 *
 * The resulting two `arc` objects will point to the same underlying data
 * object. However, one of the `arc` objects can be sent to another task,
 * allowing them to share the underlying data.
 */
pub fn clone<T: Const Owned>(rc: &ARC<T>) -> ARC<T> {
    ARC { x: unsafe { clone_shared_mutable_state(&rc.x) } }
}

/**
 * Retrieve the data back out of the ARC. This function blocks until the
 * reference given to it is the last existing one, and then unwrap the data
 * instead of destroying it.
 *
 * If multiple tasks call unwrap, all but the first will fail. Do not call
 * unwrap from a task that holds another reference to the same ARC; it is
 * guaranteed to deadlock.
 */
fn unwrap<T: Const Owned>(rc: ARC<T>) -> T {
    let ARC { x: x } = move rc;
    unsafe { unwrap_shared_mutable_state(move x) }
}

impl<T: Const Owned> ARC<T>: Clone {
    fn clone(&self) -> ARC<T> {
        clone(self)
    }
}

/****************************************************************************
 * Mutex protected ARC (unsafe)
 ****************************************************************************/

#[doc(hidden)]
struct MutexARCInner<T: Owned> { lock: Mutex, failed: bool, data: T }
/// An ARC with mutable data protected by a blocking mutex.
struct MutexARC<T: Owned> { x: SharedMutableState<MutexARCInner<T>> }

/// Create a mutex-protected ARC with the supplied data.
pub fn MutexARC<T: Owned>(user_data: T) -> MutexARC<T> {
    mutex_arc_with_condvars(move user_data, 1)
}
/**
 * Create a mutex-protected ARC with the supplied data and a specified number
 * of condvars (as sync::mutex_with_condvars).
 */
pub fn mutex_arc_with_condvars<T: Owned>(user_data: T,
                                    num_condvars: uint) -> MutexARC<T> {
    let data =
        MutexARCInner { lock: mutex_with_condvars(num_condvars),
                          failed: false, data: move user_data };
    MutexARC { x: unsafe { shared_mutable_state(move data) } }
}

impl<T: Owned> MutexARC<T>: Clone {
    /// Duplicate a mutex-protected ARC, as arc::clone.
    fn clone(&self) -> MutexARC<T> {
        // NB: Cloning the underlying mutex is not necessary. Its reference
        // count would be exactly the same as the shared state's.
        MutexARC { x: unsafe { clone_shared_mutable_state(&self.x) } }
    }
}

impl<T: Owned> &MutexARC<T> {

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
            let _z = PoisonOnFail(&mut state.failed);
            blk(&mut state.data)
        }
    }
    /// As access(), but with a condvar, as sync::mutex.lock_cond().
    #[inline(always)]
    unsafe fn access_cond<U>(blk: fn(x: &x/mut T, c: &c/Condvar) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do (&state.lock).lock_cond |cond| {
            check_poison(true, state.failed);
            let _z = PoisonOnFail(&mut state.failed);
            blk(&mut state.data,
                &Condvar { is_mutex: true, failed: &mut state.failed,
                           cond: cond })
        }
    }
}

/**
 * Retrieves the data, blocking until all other references are dropped,
 * exactly as arc::unwrap.
 *
 * Will additionally fail if another task has failed while accessing the arc.
 */
// FIXME(#3724) make this a by-move method on the arc
pub fn unwrap_mutex_arc<T: Owned>(arc: MutexARC<T>) -> T {
    let MutexARC { x: x } = move arc;
    let inner = unsafe { unwrap_shared_mutable_state(move x) };
    let MutexARCInner { failed: failed, data: data, _ } = move inner;
    if failed {
        fail ~"Can't unwrap poisoned MutexARC - another task failed inside!"
    }
    move data
}

// Common code for {mutex.access,rwlock.write}{,_cond}.
#[inline(always)]
#[doc(hidden)]
fn check_poison(is_mutex: bool, failed: bool) {
    if failed {
        if is_mutex {
            fail ~"Poisoned MutexARC - another task failed inside!";
        } else {
            fail ~"Poisoned rw_arc - another task failed inside!";
        }
    }
}

#[doc(hidden)]
struct PoisonOnFail {
    failed: &mut bool,
}

impl PoisonOnFail : Drop {
    fn finalize(&self) {
        /* assert !*self.failed; -- might be false in case of cond.wait() */
        if task::failing() { *self.failed = true; }
    }
}

fn PoisonOnFail(failed: &r/mut bool) -> PoisonOnFail/&r {
    PoisonOnFail {
        failed: failed
    }
}

/****************************************************************************
 * R/W lock protected ARC
 ****************************************************************************/

#[doc(hidden)]
struct RWARCInner<T: Const Owned> { lock: RWlock, failed: bool, data: T }
/**
 * A dual-mode ARC protected by a reader-writer lock. The data can be accessed
 * mutably or immutably, and immutably-accessing tasks may run concurrently.
 *
 * Unlike mutex_arcs, rw_arcs are safe, because they cannot be nested.
 */
struct RWARC<T: Const Owned> {
    x: SharedMutableState<RWARCInner<T>>,
    mut cant_nest: ()
}

/// Create a reader/writer ARC with the supplied data.
pub fn RWARC<T: Const Owned>(user_data: T) -> RWARC<T> {
    rw_arc_with_condvars(move user_data, 1)
}
/**
 * Create a reader/writer ARC with the supplied data and a specified number
 * of condvars (as sync::rwlock_with_condvars).
 */
pub fn rw_arc_with_condvars<T: Const Owned>(user_data: T,
                                       num_condvars: uint) -> RWARC<T> {
    let data =
        RWARCInner { lock: rwlock_with_condvars(num_condvars),
                     failed: false, data: move user_data };
    RWARC { x: unsafe { shared_mutable_state(move data) }, cant_nest: () }
}

impl<T: Const Owned> RWARC<T> {
    /// Duplicate a rwlock-protected ARC, as arc::clone.
    fn clone(&self) -> RWARC<T> {
        RWARC { x: unsafe { clone_shared_mutable_state(&self.x) },
                cant_nest: () }
    }

}

impl<T: Const Owned> &RWARC<T> {
    /**
     * Access the underlying data mutably. Locks the rwlock in write mode;
     * other readers and writers will block.
     *
     * # Failure
     *
     * Failing while inside the ARC will unlock the ARC while unwinding, so
     * that other tasks won't block forever. As MutexARC.access, it will also
     * poison the ARC, so subsequent readers and writers will both also fail.
     */
    #[inline(always)]
    fn write<U>(blk: fn(x: &mut T) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do borrow_rwlock(state).write {
            check_poison(false, state.failed);
            let _z = PoisonOnFail(&mut state.failed);
            blk(&mut state.data)
        }
    }
    /// As write(), but with a condvar, as sync::rwlock.write_cond().
    #[inline(always)]
    fn write_cond<U>(blk: fn(x: &x/mut T, c: &c/Condvar) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do borrow_rwlock(state).write_cond |cond| {
            check_poison(false, state.failed);
            let _z = PoisonOnFail(&mut state.failed);
            blk(&mut state.data,
                &Condvar { is_mutex: false, failed: &mut state.failed,
                           cond: cond })
        }
    }
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

    /**
     * As write(), but with the ability to atomically 'downgrade' the lock.
     * See sync::rwlock.write_downgrade(). The RWWriteMode token must be used
     * to obtain the &mut T, and can be transformed into a RWReadMode token by
     * calling downgrade(), after which a &T can be obtained instead.
     * ~~~
     * do arc.write_downgrade |write_mode| {
     *     do (&write_mode).write_cond |state, condvar| {
     *         ... exclusive access with mutable state ...
     *     }
     *     let read_mode = arc.downgrade(write_mode);
     *     do (&read_mode).read |state| {
     *         ... shared access with immutable state ...
     *     }
     * }
     * ~~~
     */
    fn write_downgrade<U>(blk: fn(v: RWWriteMode<T>) -> U) -> U {
        let state = unsafe { get_shared_mutable_state(&self.x) };
        do borrow_rwlock(state).write_downgrade |write_mode| {
            check_poison(false, state.failed);
            blk(RWWriteMode((&mut state.data, move write_mode,
                              PoisonOnFail(&mut state.failed))))
        }
    }

    /// To be called inside of the write_downgrade block.
    fn downgrade(token: RWWriteMode/&a<T>) -> RWReadMode/&a<T> {
        // The rwlock should assert that the token belongs to us for us.
        let state = unsafe { get_shared_immutable_state(&self.x) };
        let RWWriteMode((data, t, _poison)) = move token;
        // Let readers in
        let new_token = (&state.lock).downgrade(move t);
        // Whatever region the input reference had, it will be safe to use
        // the same region for the output reference. (The only 'unsafe' part
        // of this cast is removing the mutability.)
        let new_data = unsafe { cast::transmute_immut(data) };
        // Downgrade ensured the token belonged to us. Just a sanity check.
        assert ptr::ref_eq(&state.data, new_data);
        // Produce new token
        RWReadMode((new_data, move new_token))
    }
}

/**
 * Retrieves the data, blocking until all other references are dropped,
 * exactly as arc::unwrap.
 *
 * Will additionally fail if another task has failed while accessing the arc
 * in write mode.
 */
// FIXME(#3724) make this a by-move method on the arc
pub fn unwrap_rw_arc<T: Const Owned>(arc: RWARC<T>) -> T {
    let RWARC { x: x, _ } = move arc;
    let inner = unsafe { unwrap_shared_mutable_state(move x) };
    let RWARCInner { failed: failed, data: data, _ } = move inner;
    if failed {
        fail ~"Can't unwrap poisoned RWARC - another task failed inside!"
    }
    move data
}

// Borrowck rightly complains about immutably aliasing the rwlock in order to
// lock it. This wraps the unsafety, with the justification that the 'lock'
// field is never overwritten; only 'failed' and 'data'.
#[doc(hidden)]
fn borrow_rwlock<T: Const Owned>(state: &r/mut RWARCInner<T>) -> &r/RWlock {
    unsafe { cast::transmute_immut(&mut state.lock) }
}

// FIXME (#3154) ice with struct/&<T> prevents these from being structs.

/// The "write permission" token used for RWARC.write_downgrade().
pub enum RWWriteMode<T: Const Owned> =
    (&mut T, sync::RWlockWriteMode, PoisonOnFail);
/// The "read permission" token used for RWARC.write_downgrade().
pub enum RWReadMode<T:Const Owned> = (&T, sync::RWlockReadMode);

impl<T: Const Owned> &RWWriteMode<T> {
    /// Access the pre-downgrade RWARC in write mode.
    fn write<U>(blk: fn(x: &mut T) -> U) -> U {
        match *self {
            RWWriteMode((data, ref token, _)) => {
                do token.write {
                    blk(data)
                }
            }
        }
    }
    /// Access the pre-downgrade RWARC in write mode with a condvar.
    fn write_cond<U>(blk: fn(x: &x/mut T, c: &c/Condvar) -> U) -> U {
        match *self {
            RWWriteMode((data, ref token, ref poison)) => {
                do token.write_cond |cond| {
                    let cvar = Condvar {
                        is_mutex: false, failed: poison.failed,
                        cond: cond };
                    blk(data, &cvar)
                }
            }
        }
    }
}

impl<T: Const Owned> &RWReadMode<T> {
    /// Access the post-downgrade rwlock in read mode.
    fn read<U>(blk: fn(x: &T) -> U) -> U {
        match *self {
            RWReadMode((data, ref token)) => {
                do token.read { blk(data) }
            }
        }
    }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    #[legacy_exports];

    use core::prelude::*;

    use arc::*;
    use arc;

    use core::oldcomm::*;
    use core::option::{Some, None};
    use core::option;
    use core::pipes;
    use core::task;
    use core::vec;

    #[test]
    fn manually_share_arc() {
        let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = arc::ARC(v);

        let (p, c) = pipes::stream();

        do task::spawn() |move c| {
            let p = pipes::PortSet();
            c.send(p.chan());

            let arc_v = p.recv();

            let v = *arc::get::<~[int]>(&arc_v);
            assert v[3] == 4;
        };

        let c = p.recv();
        c.send(arc::clone(&arc_v));

        assert (*arc::get(&arc_v))[2] == 3;

        log(info, arc_v);
    }

    #[test]
    fn test_mutex_arc_condvar() {
        let arc = ~MutexARC(false);
        let arc2 = ~arc.clone();
        let (p,c) = pipes::oneshot();
        let (c,p) = (~mut Some(move c), ~mut Some(move p));
        do task::spawn |move arc2, move p| {
            // wait until parent gets in
            pipes::recv_one(option::swap_unwrap(p));
            do arc2.access_cond |state, cond| {
                *state = true;
                cond.signal();
            }
        }
        do arc.access_cond |state, cond| {
            pipes::send_one(option::swap_unwrap(c), ());
            assert !*state;
            while !*state {
                cond.wait();
            }
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_arc_condvar_poison() {
        let arc = ~MutexARC(1);
        let arc2 = ~arc.clone();
        let (p, c) = pipes::stream();

        do task::spawn_unlinked |move arc2, move p| {
            let _ = p.recv();
            do arc2.access_cond |one, cond| {
                cond.signal();
                assert *one == 0; // Parent should fail when it wakes up.
            }
        }

        do arc.access_cond |one, cond| {
            c.send(());
            while *one == 1 {
                cond.wait();
            }
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_mutex_arc_poison() {
        let arc = ~MutexARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
            do arc2.access |one| {
                assert *one == 2;
            }
        };
        do arc.access |one| {
            assert *one == 1;
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_mutex_arc_unwrap_poison() {
        let arc = MutexARC(1);
        let arc2 = ~(&arc).clone();
        let (p, c) = pipes::stream();
        do task::spawn |move c, move arc2| {
            do arc2.access |one| {
                c.send(());
                assert *one == 2;
            }
        }
        let _ = p.recv();
        let one = unwrap_mutex_arc(move arc);
        assert one == 1;
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_wr() {
        let arc = ~RWARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
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
        let arc = ~RWARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
            do arc2.write |one| {
                assert *one == 2;
            }
        };
        do arc.write |one| {
            assert *one == 1;
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_dw() {
        let arc = ~RWARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
            do arc2.write_downgrade |write_mode| {
                do (&write_mode).write |one| {
                    assert *one == 2;
                }
            }
        };
        do arc.write |one| {
            assert *one == 1;
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_rr() {
        let arc = ~RWARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
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
        let arc = ~RWARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
            do arc2.read |one| {
                assert *one == 2;
            }
        };
        do arc.write |one| {
            assert *one == 1;
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_dr() {
        let arc = ~RWARC(1);
        let arc2 = ~arc.clone();
        do task::try |move arc2| {
            do arc2.write_downgrade |write_mode| {
                let read_mode = arc2.downgrade(move write_mode);
                do (&read_mode).read |one| {
                    assert *one == 2;
                }
            }
        };
        do arc.write |one| {
            assert *one == 1;
        }
    }
    #[test]
    fn test_rw_arc() {
        let arc = ~RWARC(0);
        let arc2 = ~arc.clone();
        let (p,c) = pipes::stream();

        do task::spawn |move arc2, move c| {
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
            do task::task().future_result(|+r| children.push(move r)).spawn
                |move arc3| {
                do arc3.read |num| {
                    assert *num >= 0;
                }
            }
        }

        // Wait for children to pass their asserts
        for vec::each(children) |r| { r.recv(); }

        // Wait for writer to finish
        p.recv();
        do arc.read |num| { assert *num == 10; }
    }
    #[test]
    fn test_rw_downgrade() {
        // (1) A downgrader gets in write mode and does cond.wait.
        // (2) A writer gets in write mode, sets state to 42, and does signal.
        // (3) Downgrader wakes, sets state to 31337.
        // (4) tells writer and all other readers to contend as it downgrades.
        // (5) Writer attempts to set state back to 42, while downgraded task
        //     and all reader tasks assert that it's 31337.
        let arc = ~RWARC(0);

        // Reader tasks
        let mut reader_convos = ~[];
        for 10.times {
            let ((rp1,rc1),(rp2,rc2)) = (pipes::stream(),pipes::stream());
            reader_convos.push((move rc1, move rp2));
            let arcn = ~arc.clone();
            do task::spawn |move rp1, move rc2, move arcn| {
                rp1.recv(); // wait for downgrader to give go-ahead
                do arcn.read |state| {
                    assert *state == 31337;
                    rc2.send(());
                }
            }
        }

        // Writer task
        let arc2 = ~arc.clone();
        let ((wp1,wc1),(wp2,wc2)) = (pipes::stream(),pipes::stream());
        do task::spawn |move arc2, move wc2, move wp1| {
            wp1.recv();
            do arc2.write_cond |state, cond| {
                assert *state == 0;
                *state = 42;
                cond.signal();
            }
            wp1.recv();
            do arc2.write |state| {
                // This shouldn't happen until after the downgrade read
                // section, and all other readers, finish.
                assert *state == 31337;
                *state = 42;
            }
            wc2.send(());
        }

        // Downgrader (us)
        do arc.write_downgrade |write_mode| {
            do (&write_mode).write_cond |state, cond| {
                wc1.send(()); // send to another writer who will wake us up
                while *state == 0 {
                    cond.wait();
                }
                assert *state == 42;
                *state = 31337;
                // send to other readers
                for vec::each(reader_convos) |x| {
                    match *x {
                        (ref rc, _) => rc.send(()),
                    }
                }
            }
            let read_mode = arc.downgrade(move write_mode);
            do (&read_mode).read |state| {
                // complete handshake with other readers
                for vec::each(reader_convos) |x| {
                    match *x {
                        (_, ref rp) => rp.recv(),
                    }
                }
                wc1.send(()); // tell writer to try again
                assert *state == 31337;
            }
        }

        wp2.recv(); // complete handshake with writer
    }
}
