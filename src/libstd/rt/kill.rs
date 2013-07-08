// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Task death: asynchronous killing, linked failure, exit code propagation.

use cast;
use cell::Cell;
use option::{Option, Some, None};
use prelude::*;
use rt::task::Task;
use unstable::atomics::{AtomicUint, Acquire, SeqCst};
use unstable::sync::{UnsafeAtomicRcBox, LittleLock};
use util;

static KILLED_MSG: &'static str = "killed by linked failure";

// State values for the 'killed' and 'unkillable' atomic flags below.
static KILL_RUNNING:    uint = 0;
static KILL_KILLED:     uint = 1;
static KILL_UNKILLABLE: uint = 2;

// FIXME(#7544)(bblum): think about the cache efficiency of this
struct KillHandleInner {
    // Is the task running, blocked, or killed? Possible values:
    // * KILL_RUNNING    - Not unkillable, no kill pending.
    // * KILL_KILLED     - Kill pending.
    // * <ptr>           - A transmuted blocked ~Task pointer.
    // This flag is refcounted because it may also be referenced by a blocking
    // concurrency primitive, used to wake the task normally, whose reference
    // may outlive the handle's if the task is killed.
    killed: UnsafeAtomicRcBox<AtomicUint>,
    // Has the task deferred kill signals? This flag guards the above one.
    // Possible values:
    // * KILL_RUNNING    - Not unkillable, no kill pending.
    // * KILL_KILLED     - Kill pending.
    // * KILL_UNKILLABLE - Kill signals deferred.
    unkillable: AtomicUint,

    // Shared state between task and children for exit code propagation. These
    // are here so we can re-use the kill handle to implement watched children
    // tasks. Using a separate ARClike would introduce extra atomic adds/subs
    // into common spawn paths, so this is just for speed.

    // Locklessly accessed; protected by the enclosing refcount's barriers.
    any_child_failed: bool,
    // A lazy list, consuming which may unwrap() many child tombstones.
    child_tombstones: Option<~fn() -> bool>,
    // Protects multiple children simultaneously creating tombstones.
    graveyard_lock: LittleLock,
}

/// State shared between tasks used for task killing during linked failure.
#[deriving(Clone)]
pub struct KillHandle(UnsafeAtomicRcBox<KillHandleInner>);

/// Per-task state related to task death, killing, failure, etc.
pub struct Death {
    // Shared among this task, its watched children, and any linked tasks who
    // might kill it. This is optional so we can take it by-value at exit time.
    kill_handle:     Option<KillHandle>,
    // Handle to a watching parent, if we have one, for exit code propagation.
    watching_parent: Option<KillHandle>,
    // Action to be done with the exit code. If set, also makes the task wait
    // until all its watched children exit before collecting the status.
    on_exit:         Option<~fn(bool)>,
    // nesting level counter for task::unkillable calls (0 == killable).
    unkillable:      int,
    // nesting level counter for task::atomically calls (0 == can yield).
    wont_sleep:      int,
}

impl KillHandle {
    pub fn new() -> KillHandle {
        KillHandle(UnsafeAtomicRcBox::new(KillHandleInner {
            // Linked failure fields
            killed:     UnsafeAtomicRcBox::new(AtomicUint::new(KILL_RUNNING)),
            unkillable: AtomicUint::new(KILL_RUNNING),
            // Exit code propagation fields
            any_child_failed: false,
            child_tombstones: None,
            graveyard_lock:   LittleLock(),
        }))
    }

    // Will begin unwinding if a kill signal was received, unless already_failing.
    // This can't be used recursively, because a task which sees a KILLED
    // signal must fail immediately, which an already-unkillable task can't do.
    #[inline]
    pub fn inhibit_kill(&mut self, already_failing: bool) {
        let inner = unsafe { &mut *self.get() };
        // Expect flag to contain RUNNING. If KILLED, it should stay KILLED.
        // FIXME(#7544)(bblum): is it really necessary to prohibit double kill?
        match inner.unkillable.compare_and_swap(KILL_RUNNING, KILL_UNKILLABLE, SeqCst) {
            KILL_RUNNING    => { }, // normal case
            KILL_KILLED     => if !already_failing { fail!(KILLED_MSG) },
            _               => rtabort!("inhibit_kill: task already unkillable"),
        }
    }

    // Will begin unwinding if a kill signal was received, unless already_failing.
    #[inline]
    pub fn allow_kill(&mut self, already_failing: bool) {
        let inner = unsafe { &mut *self.get() };
        // Expect flag to contain UNKILLABLE. If KILLED, it should stay KILLED.
        // FIXME(#7544)(bblum): is it really necessary to prohibit double kill?
        match inner.unkillable.compare_and_swap(KILL_UNKILLABLE, KILL_RUNNING, SeqCst) {
            KILL_UNKILLABLE => { }, // normal case
            KILL_KILLED     => if !already_failing { fail!(KILLED_MSG) },
            _               => rtabort!("allow_kill: task already killable"),
        }
    }

    // Send a kill signal to the handle's owning task. Returns the task itself
    // if it was blocked and needs punted awake. To be called by other tasks.
    pub fn kill(&mut self) -> Option<~Task> {
        let inner = unsafe { &mut *self.get() };
        if inner.unkillable.swap(KILL_KILLED, SeqCst) == KILL_RUNNING {
            // Got in. Allowed to try to punt the task awake.
            let flag = unsafe { &mut *inner.killed.get() };
            match flag.swap(KILL_KILLED, SeqCst) {
                // Task either not blocked or already taken care of.
                KILL_RUNNING | KILL_KILLED => None,
                // Got ownership of the blocked task.
                task_ptr => Some(unsafe { cast::transmute(task_ptr) }),
            }
        } else {
            // Otherwise it was either unkillable or already killed. Somebody
            // else was here first who will deal with the kill signal.
            None
        }
    }

    #[inline]
    pub fn killed(&self) -> bool {
        // Called every context switch, so shouldn't report true if the task
        // is unkillable with a kill signal pending.
        let inner = unsafe { &*self.get() };
        let flag  = unsafe { &*inner.killed.get() };
        // FIXME(#6598): can use relaxed ordering (i think)
        flag.load(Acquire) == KILL_KILLED
    }

    pub fn notify_immediate_failure(&mut self) {
        // A benign data race may happen here if there are failing sibling
        // tasks that were also spawned-watched. The refcount's write barriers
        // in UnsafeAtomicRcBox ensure that this write will be seen by the
        // unwrapper/destructor, whichever task may unwrap it.
        unsafe { (*self.get()).any_child_failed = true; }
    }

    // For use when a task does not need to collect its children's exit
    // statuses, but the task has a parent which might want them.
    pub fn reparent_children_to(self, parent: &mut KillHandle) {
        // Optimistic path: If another child of the parent's already failed,
        // we don't need to worry about any of this.
        if unsafe { (*parent.get()).any_child_failed } {
            return;
        }

        // Try to see if all our children are gone already.
        match unsafe { self.try_unwrap() } {
            // Couldn't unwrap; children still alive. Reparent entire handle as
            // our own tombstone, to be unwrapped later.
            Left(this) => {
                let this = Cell::new(this); // :(
                do add_lazy_tombstone(parent) |other_tombstones| {
                    let this = Cell::new(this.take()); // :(
                    let others = Cell::new(other_tombstones); // :(
                    || {
                        // Prefer to check tombstones that were there first,
                        // being "more fair" at the expense of tail-recursion.
                        others.take().map_consume_default(true, |f| f()) && {
                            let mut inner = unsafe { this.take().unwrap() };
                            (!inner.any_child_failed) &&
                                inner.child_tombstones.take_map_default(true, |f| f())
                        }
                    }
                }
            }
            // Whether or not all children exited, one or more already failed.
            Right(KillHandleInner { any_child_failed: true, _ }) => {
                parent.notify_immediate_failure();
            }
            // All children exited, but some left behind tombstones that we
            // don't want to wait on now. Give them to our parent.
            Right(KillHandleInner { any_child_failed: false,
                                    child_tombstones: Some(f), _ }) => {
                let f = Cell::new(f); // :(
                do add_lazy_tombstone(parent) |other_tombstones| {
                    let f = Cell::new(f.take()); // :(
                    let others = Cell::new(other_tombstones); // :(
                    || {
                        // Prefer fairness to tail-recursion, as in above case.
                        others.take().map_consume_default(true, |f| f()) &&
                            f.take()()
                    }
                }
            }
            // All children exited, none failed. Nothing to do!
            Right(KillHandleInner { any_child_failed: false,
                                    child_tombstones: None, _ }) => { }
        }

        // NB: Takes a pthread mutex -- 'blk' not allowed to reschedule.
        #[inline]
        fn add_lazy_tombstone(parent: &mut KillHandle,
                              blk: &fn(Option<~fn() -> bool>) -> ~fn() -> bool) {

            let inner: &mut KillHandleInner = unsafe { &mut *parent.get() };
            unsafe {
                do inner.graveyard_lock.lock {
                    // Update the current "head node" of the lazy list.
                    inner.child_tombstones =
                        Some(blk(util::replace(&mut inner.child_tombstones, None)));
                }
            }
        }
    }
}

impl Death {
    pub fn new() -> Death {
        Death {
            kill_handle:     Some(KillHandle::new()),
            watching_parent: None,
            on_exit:         None,
            unkillable:      0,
            wont_sleep:      0,
        }
    }

    pub fn new_child(&self) -> Death {
        // FIXME(#7327)
        Death {
            kill_handle:     Some(KillHandle::new()),
            watching_parent: self.kill_handle.clone(),
            on_exit:         None,
            unkillable:      0,
            wont_sleep:      0,
        }
    }

    /// Collect failure exit codes from children and propagate them to a parent.
    pub fn collect_failure(&mut self, mut success: bool) {
        // This may run after the task has already failed, so even though the
        // task appears to need to be killed, the scheduler should not fail us
        // when we block to unwrap.
        // (XXX: Another less-elegant reason for doing this is so that the use
        // of the LittleLock in reparent_children_to doesn't need to access the
        // unkillable flag in the kill_handle, since we'll have removed it.)
        rtassert!(self.unkillable == 0);
        self.unkillable = 1;

        // Step 1. Decide if we need to collect child failures synchronously.
        do self.on_exit.take_map |on_exit| {
            if success {
                // We succeeded, but our children might not. Need to wait for them.
                let mut inner = unsafe { self.kill_handle.take_unwrap().unwrap() };
                if inner.any_child_failed {
                    success = false;
                } else {
                    // Lockless access to tombstones protected by unwrap barrier.
                    success = inner.child_tombstones.take_map_default(true, |f| f());
                }
            }
            on_exit(success);
        };

        // Step 2. Possibly alert possibly-watching parent to failure status.
        // Note that as soon as parent_handle goes out of scope, the parent
        // can successfully unwrap its handle and collect our reported status.
        do self.watching_parent.take_map |mut parent_handle| {
            if success {
                // Our handle might be None if we had an exit callback, and
                // already unwrapped it. But 'success' being true means no
                // child failed, so there's nothing to do (see below case).
                do self.kill_handle.take_map |own_handle| {
                    own_handle.reparent_children_to(&mut parent_handle);
                };
            } else {
                // Can inform watching parent immediately that we failed.
                // (Note the importance of non-failing tasks NOT writing
                // 'false', which could obscure another task's failure.)
                parent_handle.notify_immediate_failure();
            }
        };

        // Can't use allow_kill directly; that would require the kill handle.
        rtassert!(self.unkillable == 1);
        self.unkillable = 0;
    }

    /// Fails if a kill signal was received.
    #[inline]
    pub fn check_killed(&self) {
        match self.kill_handle {
            Some(ref kill_handle) =>
                // The task may be both unkillable and killed if it does some
                // synchronization during unwinding or cleanup (for example,
                // sending on a notify port). In that case failing won't help.
                if self.unkillable == 0 && kill_handle.killed() {
                    fail!(KILLED_MSG);
                },
            // This may happen during task death (see comments in collect_failure).
            None => rtassert!(self.unkillable > 0),
        }
    }

    /// Enter a possibly-nested unkillable section of code.
    /// All calls must be paired with a subsequent call to allow_kill.
    #[inline]
    pub fn inhibit_kill(&mut self, already_failing: bool) {
        if self.unkillable == 0 {
            rtassert!(self.kill_handle.is_some());
            self.kill_handle.get_mut_ref().inhibit_kill(already_failing);
        }
        self.unkillable += 1;
    }

    /// Exit a possibly-nested unkillable section of code.
    /// All calls must be paired with a preceding call to inhibit_kill.
    #[inline]
    pub fn allow_kill(&mut self, already_failing: bool) {
        rtassert!(self.unkillable != 0);
        self.unkillable -= 1;
        if self.unkillable == 0 {
            rtassert!(self.kill_handle.is_some());
            self.kill_handle.get_mut_ref().allow_kill(already_failing);
        }
    }

    /// Enter a possibly-nested "atomic" section of code. Just for assertions.
    /// All calls must be paired with a subsequent call to allow_yield.
    #[inline]
    pub fn inhibit_yield(&mut self) {
        self.wont_sleep += 1;
    }

    /// Exit a possibly-nested "atomic" section of code. Just for assertions.
    /// All calls must be paired with a preceding call to inhibit_yield.
    #[inline]
    pub fn allow_yield(&mut self) {
        rtassert!(self.wont_sleep != 0);
        self.wont_sleep -= 1;
    }

    /// Ensure that the task is allowed to become descheduled.
    #[inline]
    pub fn assert_may_sleep(&self) {
        if self.wont_sleep != 0 {
            rtabort!("illegal atomic-sleep: can't deschedule inside atomically()");
        }
    }
}

impl Drop for Death {
    fn drop(&self) {
        // Mustn't be in an atomic or unkillable section at task death.
        rtassert!(self.unkillable == 0);
        rtassert!(self.wont_sleep == 0);
    }
}

#[cfg(test)]
mod test {
    #[allow(unused_mut)];
    use rt::test::*;
    use super::*;
    use util;

    #[test]
    fn no_tombstone_success() {
        do run_in_newsched_task {
            // Tests case 4 of the 4-way match in reparent_children.
            let mut parent = KillHandle::new();
            let mut child  = KillHandle::new();

            // Without another handle to child, the try unwrap should succeed.
            child.reparent_children_to(&mut parent);
            let mut parent_inner = unsafe { parent.unwrap() };
            assert!(parent_inner.child_tombstones.is_none());
            assert!(parent_inner.any_child_failed == false);
        }
    }
    #[test]
    fn no_tombstone_failure() {
        do run_in_newsched_task {
            // Tests case 2 of the 4-way match in reparent_children.
            let mut parent = KillHandle::new();
            let mut child  = KillHandle::new();

            child.notify_immediate_failure();
            // Without another handle to child, the try unwrap should succeed.
            child.reparent_children_to(&mut parent);
            let mut parent_inner = unsafe { parent.unwrap() };
            assert!(parent_inner.child_tombstones.is_none());
            // Immediate failure should have been propagated.
            assert!(parent_inner.any_child_failed);
        }
    }
    #[test]
    fn no_tombstone_because_sibling_already_failed() {
        do run_in_newsched_task {
            // Tests "case 0, the optimistic path in reparent_children.
            let mut parent = KillHandle::new();
            let mut child1 = KillHandle::new();
            let mut child2 = KillHandle::new();
            let mut link   = child2.clone();

            // Should set parent's child_failed flag
            child1.notify_immediate_failure();
            child1.reparent_children_to(&mut parent);
            // Should bypass trying to unwrap child2 entirely.
            // Otherwise, due to 'link', it would try to tombstone.
            child2.reparent_children_to(&mut parent);
            // Should successfully unwrap even though 'link' is still alive.
            let mut parent_inner = unsafe { parent.unwrap() };
            assert!(parent_inner.child_tombstones.is_none());
            // Immediate failure should have been propagated by first child.
            assert!(parent_inner.any_child_failed);
            util::ignore(link);
        }
    }
    #[test]
    fn one_tombstone_success() {
        do run_in_newsched_task {
            let mut parent = KillHandle::new();
            let mut child  = KillHandle::new();
            let mut link   = child.clone();

            // Creates 1 tombstone. Existence of 'link' makes try-unwrap fail.
            child.reparent_children_to(&mut parent);
            // Let parent collect tombstones.
            util::ignore(link);
            // Must have created a tombstone
            let mut parent_inner = unsafe { parent.unwrap() };
            assert!(parent_inner.child_tombstones.take_unwrap()());
            assert!(parent_inner.any_child_failed == false);
        }
    }
    #[test]
    fn one_tombstone_failure() {
        do run_in_newsched_task {
            let mut parent = KillHandle::new();
            let mut child  = KillHandle::new();
            let mut link   = child.clone();

            // Creates 1 tombstone. Existence of 'link' makes try-unwrap fail.
            child.reparent_children_to(&mut parent);
            // Must happen after tombstone to not be immediately propagated.
            link.notify_immediate_failure();
            // Let parent collect tombstones.
            util::ignore(link);
            // Must have created a tombstone
            let mut parent_inner = unsafe { parent.unwrap() };
            // Failure must be seen in the tombstone.
            assert!(parent_inner.child_tombstones.take_unwrap()() == false);
            assert!(parent_inner.any_child_failed == false);
        }
    }
    #[test]
    fn two_tombstones_success() {
        do run_in_newsched_task {
            let mut parent = KillHandle::new();
            let mut middle = KillHandle::new();
            let mut child  = KillHandle::new();
            let mut link   = child.clone();

            child.reparent_children_to(&mut middle); // case 1 tombstone
            // 'middle' should try-unwrap okay, but still have to reparent.
            middle.reparent_children_to(&mut parent); // case 3 tombston
            // Let parent collect tombstones.
            util::ignore(link);
            // Must have created a tombstone
            let mut parent_inner = unsafe { parent.unwrap() };
            assert!(parent_inner.child_tombstones.take_unwrap()());
            assert!(parent_inner.any_child_failed == false);
        }
    }
    #[test]
    fn two_tombstones_failure() {
        do run_in_newsched_task {
            let mut parent = KillHandle::new();
            let mut middle = KillHandle::new();
            let mut child  = KillHandle::new();
            let mut link   = child.clone();

            child.reparent_children_to(&mut middle); // case 1 tombstone
            // Must happen after tombstone to not be immediately propagated.
            link.notify_immediate_failure();
            // 'middle' should try-unwrap okay, but still have to reparent.
            middle.reparent_children_to(&mut parent); // case 3 tombstone
            // Let parent collect tombstones.
            util::ignore(link);
            // Must have created a tombstone
            let mut parent_inner = unsafe { parent.unwrap() };
            // Failure must be seen in the tombstone.
            assert!(parent_inner.child_tombstones.take_unwrap()() == false);
            assert!(parent_inner.any_child_failed == false);
        }
    }
}
