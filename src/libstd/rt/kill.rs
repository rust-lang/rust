// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Task death: asynchronous killing, linked failure, exit code propagation.

This file implements two orthogonal building-blocks for communicating failure
between tasks. One is 'linked failure' or 'task killing', that is, a failing
task causing other tasks to fail promptly (even those that are blocked on
pipes or I/O). The other is 'exit code propagation', which affects the result
observed by the parent of a task::try task that itself spawns child tasks
(such as any #[test] function). In both cases the data structures live in
KillHandle.


I. Task killing.

The model for killing involves two atomic flags, the "kill flag" and the
"unkillable flag". Operations on the kill flag include:

- In the taskgroup code (task/spawn.rs), tasks store a clone of their
  KillHandle in their shared taskgroup. Another task in the group that fails
  will use that handle to call kill().
- When a task blocks, it turns its ~Task into a BlockedTask by storing a
  the transmuted ~Task pointer inside the KillHandle's kill flag. A task
  trying to block and a task trying to kill it can simultaneously access the
  kill flag, after which the task will get scheduled and fail (no matter who
  wins the race). Likewise, a task trying to wake a blocked task normally and
  a task trying to kill it can simultaneously access the flag; only one will
  get the task to reschedule it.

Operations on the unkillable flag include:

- When a task becomes unkillable, it swaps on the flag to forbid any killer
  from waking it up while it's blocked inside the unkillable section. If a
  kill was already pending, the task fails instead of becoming unkillable.
- When a task is done being unkillable, it restores the flag to the normal
  running state. If a kill was received-but-blocked during the unkillable
  section, the task fails at this later point.
- When a task tries to kill another task, before swapping on the kill flag, it
  first swaps on the unkillable flag, to see if it's "allowed" to wake up the
  task. If it isn't, the killed task will receive the signal when it becomes
  killable again. (Of course, a task trying to wake the task normally (e.g.
  sending on a channel) does not access the unkillable flag at all.)

Why do we not need acquire/release barriers on any of the kill flag swaps?
This is because barriers establish orderings between accesses on different
memory locations, but each kill-related operation is only a swap on a single
location, so atomicity is all that matters. The exception is kill(), which
does a swap on both flags in sequence. kill() needs no barriers because it
does not matter if its two accesses are seen reordered on another CPU: if a
killer does perform both writes, it means it saw a KILL_RUNNING in the
unkillable flag, which means an unkillable task will see KILL_KILLED and fail
immediately (rendering the subsequent write to the kill flag unnecessary).


II. Exit code propagation.

The basic model for exit code propagation, which is used with the "watched"
spawn mode (on by default for linked spawns, off for supervised and unlinked
spawns), is that a parent will wait for all its watched children to exit
before reporting whether it succeeded or failed. A watching parent will only
report success if it succeeded and all its children also reported success;
otherwise, it will report failure. This is most useful for writing test cases:

 ```
#[test]
fn test_something_in_another_task {
    do spawn {
        assert!(collatz_conjecture_is_false());
    }
}
 ```

Here, as the child task will certainly outlive the parent task, we might miss
the failure of the child when deciding whether or not the test case passed.
The watched spawn mode avoids this problem.

In order to propagate exit codes from children to their parents, any
'watching' parent must wait for all of its children to exit before it can
report its final exit status. We achieve this by using an UnsafeArc, using the
reference counting to track how many children are still alive, and using the
unwrap() operation in the parent's exit path to wait for all children to exit.
The UnsafeArc referred to here is actually the KillHandle itself.

This also works transitively, as if a "middle" watched child task is itself
watching a grandchild task, the "middle" task will do unwrap() on its own
KillHandle (thereby waiting for the grandchild to exit) before dropping its
reference to its watching parent (which will alert the parent).

While UnsafeArc::unwrap() accomplishes the synchronization, there remains the
matter of reporting the exit codes themselves. This is easiest when an exiting
watched task has no watched children of its own:

- If the task with no watched children exits successfully, it need do nothing.
- If the task with no watched children has failed, it sets a flag in the
  parent's KillHandle ("any_child_failed") to false. It then stays false forever.

However, if a "middle" watched task with watched children of its own exits
before its child exits, we need to ensure that the grandparent task may still
see a failure from the grandchild task. While we could achieve this by having
each intermediate task block on its handle, this keeps around the other resources
the task was using. To be more efficient, this is accomplished via "tombstones".

A tombstone is a closure, proc() -> bool, which will perform any waiting necessary
to collect the exit code of descendant tasks. In its environment is captured
the KillHandle of whichever task created the tombstone, and perhaps also any
tombstones that that task itself had, and finally also another tombstone,
effectively creating a lazy-list of heap closures.

When a child wishes to exit early and leave tombstones behind for its parent,
it must use a LittleLock (pthread mutex) to synchronize with any possible
sibling tasks which are trying to do the same thing with the same parent.
However, on the other side, when the parent is ready to pull on the tombstones,
it need not use this lock, because the unwrap() serves as a barrier that ensures
no children will remain with references to the handle.

The main logic for creating and assigning tombstones can be found in the
function reparent_children_to() in the impl for KillHandle.


IIA. Issues with exit code propagation.

There are two known issues with the current scheme for exit code propagation.

- As documented in issue #8136, the structure mandates the possibility for stack
  overflow when collecting tombstones that are very deeply nested. This cannot
  be avoided with the closure representation, as tombstones end up structured in
  a sort of tree. However, notably, the tombstones do not actually need to be
  collected in any particular order, and so a doubly-linked list may be used.
  However we do not do this yet because DList is in libextra.

- A discussion with Graydon made me realize that if we decoupled the exit code
  propagation from the parents-waiting action, this could result in a simpler
  implementation as the exit codes themselves would not have to be propagated,
  and could instead be propagated implicitly through the taskgroup mechanism
  that we already have. The tombstoning scheme would still be required. I have
  not implemented this because currently we can't receive a linked failure kill
  signal during the task cleanup activity, as that is currently "unkillable",
  and occurs outside the task's unwinder's "try" block, so would require some
  restructuring.

*/

use cast;
use option::{Option, Some, None};
use prelude::*;
use iter;
use task::TaskResult;
use rt::task::Task;
use unstable::atomics::{AtomicUint, SeqCst};
use unstable::sync::UnsafeArc;

/// A handle to a blocked task. Usually this means having the ~Task pointer by
/// ownership, but if the task is killable, a killer can steal it at any time.
pub enum BlockedTask {
    Owned(~Task),
    Shared(UnsafeArc<AtomicUint>),
}

/// Per-task state related to task death, killing, failure, etc.
pub struct Death {
    // Action to be done with the exit code. If set, also makes the task wait
    // until all its watched children exit before collecting the status.
    on_exit:         Option<proc(TaskResult)>,
    // nesting level counter for unstable::atomically calls (0 == can deschedule).
    priv wont_sleep:      int,
}

pub struct BlockedTaskIterator {
    priv inner: UnsafeArc<AtomicUint>,
}

impl Iterator<BlockedTask> for BlockedTaskIterator {
    fn next(&mut self) -> Option<BlockedTask> {
        Some(Shared(self.inner.clone()))
    }
}

impl BlockedTask {
    /// Returns Some if the task was successfully woken; None if already killed.
    pub fn wake(self) -> Option<~Task> {
        match self {
            Owned(task) => Some(task),
            Shared(arc) => unsafe {
                match (*arc.get()).swap(0, SeqCst) {
                    0 => None,
                    n => cast::transmute(n),
                }
            }
        }
    }

    /// Create a blocked task, unless the task was already killed.
    pub fn block(task: ~Task) -> BlockedTask {
        Owned(task)
    }

    /// Converts one blocked task handle to a list of many handles to the same.
    pub fn make_selectable(self, num_handles: uint)
        -> iter::Take<BlockedTaskIterator>
    {
        let arc = match self {
            Owned(task) => {
                let flag = unsafe { AtomicUint::new(cast::transmute(task)) };
                UnsafeArc::new(flag)
            }
            Shared(arc) => arc.clone(),
        };
        BlockedTaskIterator{ inner: arc }.take(num_handles)
    }

    // This assertion has two flavours because the wake involves an atomic op.
    // In the faster version, destructors will fail dramatically instead.
    #[inline] #[cfg(not(test))]
    pub fn assert_already_awake(self) { }
    #[inline] #[cfg(test)]
    pub fn assert_already_awake(self) { assert!(self.wake().is_none()); }

    /// Convert to an unsafe uint value. Useful for storing in a pipe's state flag.
    #[inline]
    pub unsafe fn cast_to_uint(self) -> uint {
        match self {
            Owned(task) => {
                let blocked_task_ptr: uint = cast::transmute(task);
                rtassert!(blocked_task_ptr & 0x1 == 0);
                blocked_task_ptr
            }
            Shared(arc) => {
                let blocked_task_ptr: uint = cast::transmute(~arc);
                rtassert!(blocked_task_ptr & 0x1 == 0);
                blocked_task_ptr | 0x1
            }
        }
    }

    /// Convert from an unsafe uint value. Useful for retrieving a pipe's state flag.
    #[inline]
    pub unsafe fn cast_from_uint(blocked_task_ptr: uint) -> BlockedTask {
        if blocked_task_ptr & 0x1 == 0 {
            Owned(cast::transmute(blocked_task_ptr))
        } else {
            let ptr: ~UnsafeArc<AtomicUint> = cast::transmute(blocked_task_ptr & !1);
            Shared(*ptr)
        }
    }
}

impl Death {
    pub fn new() -> Death {
        Death {
            on_exit:         None,
            wont_sleep:      0,
        }
    }

    /// Collect failure exit codes from children and propagate them to a parent.
    pub fn collect_failure(&mut self, result: TaskResult) {
        match self.on_exit.take() {
            Some(f) => f(result),
            None => {}
        }
    }

    /// Enter a possibly-nested "atomic" section of code. Just for assertions.
    /// All calls must be paired with a subsequent call to allow_deschedule.
    #[inline]
    pub fn inhibit_deschedule(&mut self) {
        self.wont_sleep += 1;
    }

    /// Exit a possibly-nested "atomic" section of code. Just for assertions.
    /// All calls must be paired with a preceding call to inhibit_deschedule.
    #[inline]
    pub fn allow_deschedule(&mut self) {
        rtassert!(self.wont_sleep != 0);
        self.wont_sleep -= 1;
    }

    /// Ensure that the task is allowed to become descheduled.
    #[inline]
    pub fn assert_may_sleep(&self) {
        if self.wont_sleep != 0 {
            rtabort!("illegal atomic-sleep: attempt to reschedule while \
                      using an Exclusive or LittleLock");
        }
    }
}

impl Drop for Death {
    fn drop(&mut self) {
        // Mustn't be in an atomic or unkillable section at task death.
        rtassert!(self.wont_sleep == 0);
    }
}

#[cfg(test)]
mod test {
    use rt::test::*;
    use super::*;

    // Task blocking tests

    #[test]
    fn block_and_wake() {
        do with_test_task |task| {
            BlockedTask::block(task).wake().unwrap()
        }
    }
}
