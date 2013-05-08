// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**
 * Concurrency-enabled mechanisms for sharing mutable and/or immutable state
 * between tasks.
 */

use sync;
use sync::{Mutex, mutex_with_condvars, RWlock, rwlock_with_condvars};

use core::cast;
use core::unstable::{SharedMutableState, shared_mutable_state};
use core::unstable::{clone_shared_mutable_state};
use core::unstable::{get_shared_mutable_state, get_shared_immutable_state};
use core::ptr;
use core::task;

/// As sync::condvar, a mechanism for unlock-and-descheduling and signalling.
pub struct Condvar<'self> {
    is_mutex: bool,
    failed: &'self mut bool,
    cond: &'self sync::Condvar<'self>
}

pub impl<'self> Condvar<'self> {
    /// Atomically exit the associated ARC and block until a signal is sent.
    #[inline(always)]
    fn wait(&self) { self.wait_on(0) }

    /**
     * Atomically exit the associated ARC and block on a specified condvar
     * until a signal is sent on that same condvar (as sync::cond.wait_on).
     *
     * wait() is equivalent to wait_on(0).
     */
    #[inline(always)]
    fn wait_on(&self, condvar_id: uint) {
        assert!(!*self.failed);
        self.cond.wait_on(condvar_id);
        // This is why we need to wrap sync::condvar.
        check_poison(self.is_mutex, *self.failed);
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    #[inline(always)]
    fn signal(&self) -> bool { self.signal_on(0) }

    /**
     * Wake up a blocked task on a specified condvar (as
     * sync::cond.signal_on). Returns false if there was no blocked task.
     */
    #[inline(always)]
    fn signal_on(&self, condvar_id: uint) -> bool {
        assert!(!*self.failed);
        self.cond.signal_on(condvar_id)
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    #[inline(always)]
    fn broadcast(&self) -> uint { self.broadcast_on(0) }

    /**
     * Wake up all blocked tasks on a specified condvar (as
     * sync::cond.broadcast_on). Returns Returns the number of tasks woken.
     */
    #[inline(always)]
    fn broadcast_on(&self, condvar_id: uint) -> uint {
        assert!(!*self.failed);
        self.cond.broadcast_on(condvar_id)
    }
}

/****************************************************************************
 * Immutable ARC
 ****************************************************************************/

/// An atomically reference counted wrapper for shared immutable state.
struct ARC<T> { x: SharedMutableState<T> }

/// Create an atomically reference counted wrapper.
pub fn ARC<T:Const + Owned>(data: T) -> ARC<T> {
    ARC { x: unsafe { shared_mutable_state(data) } }
}

/**
 * Access the underlying data in an atomically reference counted
 * wrapper.
 */
pub fn get<'a, T:Const + Owned>(rc: &'a ARC<T>) -> &'a T {
    unsafe { get_shared_immutable_state(&rc.x) }
}

/**
 * Duplicate an atomically reference counted wrapper.
 *
 * The resulting two `arc` objects will point to the same underlying data
 * object. However, one of the `arc` objects can be sent to another task,
 * allowing them to share the underlying data.
 */
pub fn clone<T:Const + Owned>(rc: &ARC<T>) -> ARC<T> {
    ARC { x: unsafe { clone_shared_mutable_state(&rc.x) } }
}

impl<T:Const + Owned> Clone for ARC<T> {
    fn clone(&self) -> ARC<T> {
        clone(self)
    }
}

/****************************************************************************
 * Mutex protected ARC (unsafe)
 ****************************************************************************/

#[doc(hidden)]
struct MutexARCInner<T> { lock: Mutex, failed: bool, data: T }
/// An ARC with mutable data protected by a blocking mutex.
struct MutexARC<T> { x: SharedMutableState<MutexARCInner<T>> }

/// Create a mutex-protected ARC with the supplied data.
pub fn MutexARC<T:Owned>(user_data: T) -> MutexARC<T> {
    mutex_arc_with_condvars(user_data, 1)
}
/**
 * Create a mutex-protected ARC with the supplied data and a specified number
 * of condvars (as sync::mutex_with_condvars).
 */
pub fn mutex_arc_with_condvars<T:Owned>(user_data: T,
                                    num_condvars: uint) -> MutexARC<T> {
    let data =
        MutexARCInner { lock: mutex_with_condvars(num_condvars),
                          failed: false, data: user_data };
    MutexARC { x: unsafe { shared_mutable_state(data) } }
}

impl<T:Owned> Clone for MutexARC<T> {
    /// Duplicate a mutex-protected ARC, as arc::clone.
    fn clone(&self) -> MutexARC<T> {
        // NB: Cloning the underlying mutex is not necessary. Its reference
        // count would be exactly the same as the shared state's.
        MutexARC { x: unsafe { clone_shared_mutable_state(&self.x) } }
    }
}

pub impl<T:Owned> MutexARC<T> {

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
    unsafe fn access<U>(&self, blk: &fn(x: &mut T) -> U) -> U {
        let state = get_shared_mutable_state(&self.x);
        // Borrowck would complain about this if the function were
        // not already unsafe. See borrow_rwlock, far below.
        do (&(*state).lock).lock {
            check_poison(true, (*state).failed);
            let _z = PoisonOnFail(&mut (*state).failed);
            blk(&mut (*state).data)
        }
    }

    /// As access(), but with a condvar, as sync::mutex.lock_cond().
    #[inline(always)]
    unsafe fn access_cond<'x, 'c, U>(
        &self,
        blk: &fn(x: &'x mut T, c: &'c Condvar) -> U) -> U
    {
        let state = get_shared_mutable_state(&self.x);
        do (&(*state).lock).lock_cond |cond| {
            check_poison(true, (*state).failed);
            let _z = PoisonOnFail(&mut (*state).failed);
            blk(&mut (*state).data,
                &Condvar {is_mutex: true,
                          failed: &mut (*state).failed,
                          cond: cond })
        }
    }
}

// Common code for {mutex.access,rwlock.write}{,_cond}.
#[inline(always)]
#[doc(hidden)]
fn check_poison(is_mutex: bool, failed: bool) {
    if failed {
        if is_mutex {
            fail!(~"Poisoned MutexARC - another task failed inside!");
        } else {
            fail!(~"Poisoned rw_arc - another task failed inside!");
        }
    }
}

#[doc(hidden)]
struct PoisonOnFail {
    failed: *mut bool,
}

impl Drop for PoisonOnFail {
    fn finalize(&self) {
        unsafe {
            /* assert!(!*self.failed);
               -- might be false in case of cond.wait() */
            if task::failing() {
                *self.failed = true;
            }
        }
    }
}

fn PoisonOnFail<'r>(failed: &'r mut bool) -> PoisonOnFail {
    PoisonOnFail {
        failed: ptr::to_mut_unsafe_ptr(failed)
    }
}

/****************************************************************************
 * R/W lock protected ARC
 ****************************************************************************/

#[doc(hidden)]
struct RWARCInner<T> { lock: RWlock, failed: bool, data: T }
/**
 * A dual-mode ARC protected by a reader-writer lock. The data can be accessed
 * mutably or immutably, and immutably-accessing tasks may run concurrently.
 *
 * Unlike mutex_arcs, rw_arcs are safe, because they cannot be nested.
 */
#[mutable]
struct RWARC<T> {
    x: SharedMutableState<RWARCInner<T>>,
    cant_nest: ()
}

/// Create a reader/writer ARC with the supplied data.
pub fn RWARC<T:Const + Owned>(user_data: T) -> RWARC<T> {
    rw_arc_with_condvars(user_data, 1)
}
/**
 * Create a reader/writer ARC with the supplied data and a specified number
 * of condvars (as sync::rwlock_with_condvars).
 */
pub fn rw_arc_with_condvars<T:Const + Owned>(
    user_data: T,
    num_condvars: uint) -> RWARC<T>
{
    let data =
        RWARCInner { lock: rwlock_with_condvars(num_condvars),
                     failed: false, data: user_data };
    RWARC { x: unsafe { shared_mutable_state(data) }, cant_nest: () }
}

pub impl<T:Const + Owned> RWARC<T> {
    /// Duplicate a rwlock-protected ARC, as arc::clone.
    fn clone(&self) -> RWARC<T> {
        RWARC { x: unsafe { clone_shared_mutable_state(&self.x) },
                cant_nest: () }
    }

}

pub impl<T:Const + Owned> RWARC<T> {
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
    fn write<U>(&self, blk: &fn(x: &mut T) -> U) -> U {
        unsafe {
            let state = get_shared_mutable_state(&self.x);
            do (*borrow_rwlock(state)).write {
                check_poison(false, (*state).failed);
                let _z = PoisonOnFail(&mut (*state).failed);
                blk(&mut (*state).data)
            }
        }
    }
    /// As write(), but with a condvar, as sync::rwlock.write_cond().
    #[inline(always)]
    fn write_cond<'x, 'c, U>(&self,
                             blk: &fn(x: &'x mut T, c: &'c Condvar) -> U)
                          -> U {
        unsafe {
            let state = get_shared_mutable_state(&self.x);
            do (*borrow_rwlock(state)).write_cond |cond| {
                check_poison(false, (*state).failed);
                let _z = PoisonOnFail(&mut (*state).failed);
                blk(&mut (*state).data,
                    &Condvar {is_mutex: false,
                              failed: &mut (*state).failed,
                              cond: cond})
            }
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
    fn read<U>(&self, blk: &fn(x: &T) -> U) -> U {
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
    fn write_downgrade<U>(&self, blk: &fn(v: RWWriteMode<T>) -> U) -> U {
        unsafe {
            let state = get_shared_mutable_state(&self.x);
            do (*borrow_rwlock(state)).write_downgrade |write_mode| {
                check_poison(false, (*state).failed);
                blk(RWWriteMode {
                    data: &mut (*state).data,
                    token: write_mode,
                    poison: PoisonOnFail(&mut (*state).failed)
                })
            }
        }
    }

    /// To be called inside of the write_downgrade block.
    fn downgrade<'a>(&self, token: RWWriteMode<'a, T>) -> RWReadMode<'a, T> {
        // The rwlock should assert that the token belongs to us for us.
        let state = unsafe { get_shared_immutable_state(&self.x) };
        let RWWriteMode {
            data: data,
            token: t,
            poison: _poison
        } = token;
        // Let readers in
        let new_token = (&state.lock).downgrade(t);
        // Whatever region the input reference had, it will be safe to use
        // the same region for the output reference. (The only 'unsafe' part
        // of this cast is removing the mutability.)
        let new_data = unsafe { cast::transmute_immut(data) };
        // Downgrade ensured the token belonged to us. Just a sanity check.
        assert!(ptr::ref_eq(&state.data, new_data));
        // Produce new token
        RWReadMode {
            data: new_data,
            token: new_token,
        }
    }
}

// Borrowck rightly complains about immutably aliasing the rwlock in order to
// lock it. This wraps the unsafety, with the justification that the 'lock'
// field is never overwritten; only 'failed' and 'data'.
#[doc(hidden)]
fn borrow_rwlock<T:Const + Owned>(state: *const RWARCInner<T>) -> *RWlock {
    unsafe { cast::transmute(&const (*state).lock) }
}

/// The "write permission" token used for RWARC.write_downgrade().
pub struct RWWriteMode<'self, T> {
    data: &'self mut T,
    token: sync::RWlockWriteMode<'self>,
    poison: PoisonOnFail,
}

/// The "read permission" token used for RWARC.write_downgrade().
pub struct RWReadMode<'self, T> {
    data: &'self T,
    token: sync::RWlockReadMode<'self>,
}

pub impl<'self, T:Const + Owned> RWWriteMode<'self, T> {
    /// Access the pre-downgrade RWARC in write mode.
    fn write<U>(&mut self, blk: &fn(x: &mut T) -> U) -> U {
        match *self {
            RWWriteMode {
                data: &ref mut data,
                token: ref token,
                poison: _
            } => {
                do token.write {
                    blk(data)
                }
            }
        }
    }
    /// Access the pre-downgrade RWARC in write mode with a condvar.
    fn write_cond<'x, 'c, U>(&mut self,
                             blk: &fn(x: &'x mut T, c: &'c Condvar) -> U)
                          -> U {
        match *self {
            RWWriteMode {
                data: &ref mut data,
                token: ref token,
                poison: ref poison
            } => {
                do token.write_cond |cond| {
                    unsafe {
                        let cvar = Condvar {
                            is_mutex: false,
                            failed: &mut *poison.failed,
                            cond: cond
                        };
                        blk(data, &cvar)
                    }
                }
            }
        }
    }
}

pub impl<'self, T:Const + Owned> RWReadMode<'self, T> {
    /// Access the post-downgrade rwlock in read mode.
    fn read<U>(&self, blk: &fn(x: &T) -> U) -> U {
        match *self {
            RWReadMode {
                data: data,
                token: ref token
            } => {
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

    use arc::*;
    use arc;

    use core::cell::Cell;
    use core::task;
    use core::vec;

    #[test]
    fn manually_share_arc() {
        let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = arc::ARC(v);

        let (p, c) = comm::stream();

        do task::spawn() || {
            let p = comm::PortSet::new();
            c.send(p.chan());

            let arc_v = p.recv();

            let v = *arc::get::<~[int]>(&arc_v);
            assert!(v[3] == 4);
        };

        let c = p.recv();
        c.send(arc::clone(&arc_v));

        assert!((*arc::get(&arc_v))[2] == 3);

        info!(arc_v);
    }

    #[test]
    fn test_mutex_arc_condvar() {
        let arc = ~MutexARC(false);
        let arc2 = ~arc.clone();
        let (p,c) = comm::oneshot();
        let (c,p) = (Cell(c), Cell(p));
        do task::spawn || {
            // wait until parent gets in
            comm::recv_one(p.take());
            do arc2.access_cond |state, cond| {
                *state = true;
                cond.signal();
            }
        }
        do arc.access_cond |state, cond| {
            comm::send_one(c.take(), ());
            assert!(!*state);
            while !*state {
                cond.wait();
            }
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_arc_condvar_poison() {
        let arc = ~MutexARC(1);
        let arc2 = ~arc.clone();
        let (p, c) = comm::stream();

        do task::spawn_unlinked || {
            let _ = p.recv();
            do arc2.access_cond |one, cond| {
                cond.signal();
                // Parent should fail when it wakes up.
                assert!(*one == 0);
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
        do task::try || {
            do arc2.access |one| {
                assert!(*one == 2);
            }
        };
        do arc.access |one| {
            assert!(*one == 1);
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_wr() {
        let arc = ~RWARC(1);
        let arc2 = (*arc).clone();
        do task::try || {
            do arc2.write |one| {
                assert!(*one == 2);
            }
        };
        do arc.read |one| {
            assert!(*one == 1);
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_ww() {
        let arc = ~RWARC(1);
        let arc2 = (*arc).clone();
        do task::try || {
            do arc2.write |one| {
                assert!(*one == 2);
            }
        };
        do arc.write |one| {
            assert!(*one == 1);
        }
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_rw_arc_poison_dw() {
        let arc = ~RWARC(1);
        let arc2 = (*arc).clone();
        do task::try || {
            do arc2.write_downgrade |mut write_mode| {
                do write_mode.write |one| {
                    assert!(*one == 2);
                }
            }
        };
        do arc.write |one| {
            assert!(*one == 1);
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_rr() {
        let arc = ~RWARC(1);
        let arc2 = (*arc).clone();
        do task::try || {
            do arc2.read |one| {
                assert!(*one == 2);
            }
        };
        do arc.read |one| {
            assert!(*one == 1);
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_rw() {
        let arc = ~RWARC(1);
        let arc2 = (*arc).clone();
        do task::try || {
            do arc2.read |one| {
                assert!(*one == 2);
            }
        };
        do arc.write |one| {
            assert!(*one == 1);
        }
    }
    #[test] #[ignore(cfg(windows))]
    fn test_rw_arc_no_poison_dr() {
        let arc = ~RWARC(1);
        let arc2 = (*arc).clone();
        do task::try || {
            do arc2.write_downgrade |write_mode| {
                let read_mode = arc2.downgrade(write_mode);
                do (&read_mode).read |one| {
                    assert!(*one == 2);
                }
            }
        };
        do arc.write |one| {
            assert!(*one == 1);
        }
    }
    #[test]
    fn test_rw_arc() {
        let arc = ~RWARC(0);
        let arc2 = (*arc).clone();
        let (p,c) = comm::stream();

        do task::spawn || {
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
            let arc3 = (*arc).clone();
            do task::task().future_result(|+r| children.push(r)).spawn
                || {
                do arc3.read |num| {
                    assert!(*num >= 0);
                }
            }
        }

        // Wait for children to pass their asserts
        for vec::each(children) |r| { r.recv(); }

        // Wait for writer to finish
        p.recv();
        do arc.read |num| { assert!(*num == 10); }
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
            let ((rp1,rc1),(rp2,rc2)) = (comm::stream(),comm::stream());
            reader_convos.push((rc1, rp2));
            let arcn = (*arc).clone();
            do task::spawn || {
                rp1.recv(); // wait for downgrader to give go-ahead
                do arcn.read |state| {
                    assert!(*state == 31337);
                    rc2.send(());
                }
            }
        }

        // Writer task
        let arc2 = (*arc).clone();
        let ((wp1,wc1),(wp2,wc2)) = (comm::stream(),comm::stream());
        do task::spawn || {
            wp1.recv();
            do arc2.write_cond |state, cond| {
                assert!(*state == 0);
                *state = 42;
                cond.signal();
            }
            wp1.recv();
            do arc2.write |state| {
                // This shouldn't happen until after the downgrade read
                // section, and all other readers, finish.
                assert!(*state == 31337);
                *state = 42;
            }
            wc2.send(());
        }

        // Downgrader (us)
        do arc.write_downgrade |mut write_mode| {
            do write_mode.write_cond |state, cond| {
                wc1.send(()); // send to another writer who will wake us up
                while *state == 0 {
                    cond.wait();
                }
                assert!(*state == 42);
                *state = 31337;
                // send to other readers
                for vec::each(reader_convos) |x| {
                    match *x {
                        (ref rc, _) => rc.send(()),
                    }
                }
            }
            let read_mode = arc.downgrade(write_mode);
            do (&read_mode).read |state| {
                // complete handshake with other readers
                for vec::each(reader_convos) |x| {
                    match *x {
                        (_, ref rp) => rp.recv(),
                    }
                }
                wc1.send(()); // tell writer to try again
                assert!(*state == 31337);
            }
        }

        wp2.recv(); // complete handshake with writer
    }
}
