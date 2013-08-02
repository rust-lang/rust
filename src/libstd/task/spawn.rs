// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!**************************************************************************
 * Spawning & linked failure
 *
 * Several data structures are involved in task management to allow properly
 * propagating failure across linked/supervised tasks.
 *
 * (1) The "taskgroup_arc" is an unsafe::exclusive which contains a hashset of
 *     all tasks that are part of the group. Some tasks are 'members', which
 *     means if they fail, they will kill everybody else in the taskgroup.
 *     Other tasks are 'descendants', which means they will not kill tasks
 *     from this group, but can be killed by failing members.
 *
 *     A new one of these is created each spawn_linked or spawn_supervised.
 *
 * (2) The "tcb" is a per-task control structure that tracks a task's spawn
 *     configuration. It contains a reference to its taskgroup_arc, a
 *     reference to its node in the ancestor list (below), a flag for
 *     whether it's part of the 'main'/'root' taskgroup, and an optionally
 *     configured notification port. These are stored in TLS.
 *
 * (3) The "ancestor_list" is a cons-style list of unsafe::exclusives which
 *     tracks 'generations' of taskgroups -- a group's ancestors are groups
 *     which (directly or transitively) spawn_supervised-ed them. Each task
 *     is recorded in the 'descendants' of each of its ancestor groups.
 *
 *     Spawning a supervised task is O(n) in the number of generations still
 *     alive, and exiting (by success or failure) that task is also O(n).
 *
 * This diagram depicts the references between these data structures:
 *
 *          linked_________________________________
 *        ___/                   _________         \___
 *       /   \                  | group X |        /   \
 *      (  A  ) - - - - - - - > | {A,B} {}|< - - -(  B  )
 *       \___/                  |_________|        \___/
 *      unlinked
 *         |      __ (nil)
 *         |      //|                         The following code causes this:
 *         |__   //   /\         _________
 *        /   \ //    ||        | group Y |     fn taskA() {
 *       (  C  )- - - ||- - - > |{C} {D,E}|         spawn(taskB);
 *        \___/      /  \=====> |_________|         spawn_unlinked(taskC);
 *      supervise   /gen \                          ...
 *         |    __  \ 00 /                      }
 *         |    //|  \__/                       fn taskB() { ... }
 *         |__ //     /\         _________      fn taskC() {
 *        /   \/      ||        | group Z |         spawn_supervised(taskD);
 *       (  D  )- - - ||- - - > | {D} {E} |         ...
 *        \___/      /  \=====> |_________|     }
 *      supervise   /gen \                      fn taskD() {
 *         |    __  \ 01 /                          spawn_supervised(taskE);
 *         |    //|  \__/                           ...
 *         |__ //                _________      }
 *        /   \/                | group W |     fn taskE() { ... }
 *       (  E  )- - - - - - - > | {E}  {} |
 *        \___/                 |_________|
 *
 *        "tcb"               "taskgroup_arc"
 *             "ancestor_list"
 *
 ****************************************************************************/

#[doc(hidden)];

use prelude::*;

use cast::transmute;
use cast;
use cell::Cell;
use container::MutableMap;
use comm::{Chan, GenericChan, oneshot};
use hashmap::{HashSet, HashSetConsumeIterator};
use local_data;
use task::local_data_priv::{local_get, local_set, OldHandle};
use task::rt::rust_task;
use task::rt;
use task::{Failure, SingleThreaded};
use task::{Success, TaskOpts, TaskResult};
use task::unkillable;
use to_bytes::IterBytes;
use uint;
use util;
use unstable::sync::Exclusive;
use rt::{OldTaskContext, TaskContext, SchedulerContext, GlobalContext, context};
use rt::local::Local;
use rt::task::{Task, Sched};
use rt::kill::KillHandle;
use rt::sched::Scheduler;
use rt::uv::uvio::UvEventLoop;
use rt::thread::Thread;

#[cfg(test)] use task::default_task_opts;
#[cfg(test)] use comm;
#[cfg(test)] use task;

// Transitionary.
#[deriving(Eq)]
enum TaskHandle {
    OldTask(*rust_task),
    NewTask(KillHandle),
}

impl Clone for TaskHandle {
    fn clone(&self) -> TaskHandle {
        match *self {
            OldTask(x) => OldTask(x),
            NewTask(ref x) => NewTask(x.clone()),
        }
    }
}

impl IterBytes for TaskHandle {
    fn iter_bytes(&self, lsb0: bool, f: &fn(buf: &[u8]) -> bool) -> bool {
        match *self {
            OldTask(ref x) => x.iter_bytes(lsb0, f),
            NewTask(ref x) => x.iter_bytes(lsb0, f),
        }
    }
}

struct TaskSet(HashSet<TaskHandle>);

impl TaskSet {
    #[inline]
    fn new() -> TaskSet {
        TaskSet(HashSet::new())
    }
    #[inline]
    fn insert(&mut self, task: TaskHandle) {
        let didnt_overwrite = (**self).insert(task);
        assert!(didnt_overwrite);
    }
    #[inline]
    fn remove(&mut self, task: &TaskHandle) {
        let was_present = (**self).remove(task);
        assert!(was_present);
    }
    #[inline]
    fn consume(self) -> HashSetConsumeIterator<TaskHandle> {
        (*self).consume()
    }
}

// One of these per group of linked-failure tasks.
struct TaskGroupData {
    // All tasks which might kill this group. When this is empty, the group
    // can be "GC"ed (i.e., its link in the ancestor list can be removed).
    members:     TaskSet,
    // All tasks unidirectionally supervised by (directly or transitively)
    // tasks in this group.
    descendants: TaskSet,
}
type TaskGroupArc = Exclusive<Option<TaskGroupData>>;

type TaskGroupInner<'self> = &'self mut Option<TaskGroupData>;

// A taskgroup is 'dead' when nothing can cause it to fail; only members can.
fn taskgroup_is_dead(tg: &TaskGroupData) -> bool {
    tg.members.is_empty()
}

// A list-like structure by which taskgroups keep track of all ancestor groups
// which may kill them. Needed for tasks to be able to remove themselves from
// ancestor groups upon exit. The list has a node for each "generation", and
// ends either at the root taskgroup (which has no ancestors) or at a
// taskgroup which was spawned-unlinked. Tasks from intermediate generations
// have references to the middle of the list; when intermediate generations
// die, their node in the list will be collected at a descendant's spawn-time.
struct AncestorNode {
    // Since the ancestor list is recursive, we end up with references to
    // exclusives within other exclusives. This is dangerous business (if
    // circular references arise, deadlock and memory leaks are imminent).
    // Hence we assert that this counter monotonically decreases as we
    // approach the tail of the list.
    generation:     uint,
    // Handle to the tasks in the group of the current generation.
    parent_group:   TaskGroupArc,
    // Recursive rest of the list.
    ancestors:      AncestorList,
}

struct AncestorList(Option<Exclusive<AncestorNode>>);

// Accessors for taskgroup arcs and ancestor arcs that wrap the unsafety.
#[inline]
fn access_group<U>(x: &TaskGroupArc, blk: &fn(TaskGroupInner) -> U) -> U {
    unsafe {
        x.with(blk)
    }
}

#[inline]
fn access_ancestors<U>(x: &Exclusive<AncestorNode>,
                       blk: &fn(x: &mut AncestorNode) -> U) -> U {
    unsafe {
        x.with(blk)
    }
}

#[inline] #[cfg(test)]
fn check_generation(younger: uint, older: uint) { assert!(younger > older); }
#[inline] #[cfg(not(test))]
fn check_generation(_younger: uint, _older: uint) { }

#[inline] #[cfg(test)]
fn incr_generation(ancestors: &AncestorList) -> uint {
    ancestors.map_default(0, |arc| access_ancestors(arc, |a| a.generation+1))
}
#[inline] #[cfg(not(test))]
fn incr_generation(_ancestors: &AncestorList) -> uint { 0 }

// Iterates over an ancestor list.
// (1) Runs forward_blk on each ancestral taskgroup in the list
// (2) If forward_blk "break"s, runs optional bail_blk on all ancestral
//     taskgroups that forward_blk already ran on successfully (Note: bail_blk
//     is NOT called on the block that forward_blk broke on!).
// (3) As a bonus, coalesces away all 'dead' taskgroup nodes in the list.
fn each_ancestor(list:        &mut AncestorList,
                 bail_blk:    &fn(TaskGroupInner),
                 forward_blk: &fn(TaskGroupInner) -> bool)
              -> bool {
    // "Kickoff" call - there was no last generation.
    return !coalesce(list, bail_blk, forward_blk, uint::max_value);

    // Recursively iterates, and coalesces afterwards if needed. Returns
    // whether or not unwinding is needed (i.e., !successful iteration).
    fn coalesce(list:            &mut AncestorList,
                bail_blk:        &fn(TaskGroupInner),
                forward_blk:     &fn(TaskGroupInner) -> bool,
                last_generation: uint) -> bool {
        let (coalesce_this, early_break) =
            iterate(list, bail_blk, forward_blk, last_generation);
        // What should our next ancestor end up being?
        if coalesce_this.is_some() {
            // Needed coalesce. Our next ancestor becomes our old
            // ancestor's next ancestor. ("next = old_next->next;")
            *list = coalesce_this.unwrap();
        }
        return early_break;
    }

    // Returns an optional list-to-coalesce and whether unwinding is needed.
    // Option<ancestor_list>:
    //     Whether or not the ancestor taskgroup being iterated over is
    //     dead or not; i.e., it has no more tasks left in it, whether or not
    //     it has descendants. If dead, the caller shall coalesce it away.
    // bool:
    //     True if the supplied block did 'break', here or in any recursive
    //     calls. If so, must call the unwinder on all previous nodes.
    fn iterate(ancestors:       &mut AncestorList,
               bail_blk:        &fn(TaskGroupInner),
               forward_blk:     &fn(TaskGroupInner) -> bool,
               last_generation: uint)
            -> (Option<AncestorList>, bool) {
        // At each step of iteration, three booleans are at play which govern
        // how the iteration should behave.
        // 'nobe_is_dead' - Should the list should be coalesced at this point?
        //                  Largely unrelated to the other two.
        // 'need_unwind'  - Should we run the bail_blk at this point? (i.e.,
        //                  do_continue was false not here, but down the line)
        // 'do_continue'  - Did the forward_blk succeed at this point? (i.e.,
        //                  should we recurse? or should our callers unwind?)

        let forward_blk = Cell::new(forward_blk);

        // The map defaults to None, because if ancestors is None, we're at
        // the end of the list, which doesn't make sense to coalesce.
        do ancestors.map_default((None,false)) |ancestor_arc| {
            // NB: Takes a lock! (this ancestor node)
            do access_ancestors(ancestor_arc) |nobe| {
                // Argh, but we couldn't give it to coalesce() otherwise.
                let forward_blk = forward_blk.take();
                // Check monotonicity
                check_generation(last_generation, nobe.generation);
                /*##########################################################*
                 * Step 1: Look at this ancestor group (call iterator block).
                 *##########################################################*/
                let mut nobe_is_dead = false;
                let do_continue =
                    // NB: Takes a lock! (this ancestor node's parent group)
                    do access_group(&nobe.parent_group) |tg_opt| {
                        // Decide whether this group is dead. Note that the
                        // group being *dead* is disjoint from it *failing*.
                        nobe_is_dead = match *tg_opt {
                            Some(ref tg) => taskgroup_is_dead(tg),
                            None => nobe_is_dead
                        };
                        // Call iterator block. (If the group is dead, it's
                        // safe to skip it. This will leave our TaskHandle
                        // hanging around in the group even after it's freed,
                        // but that's ok because, by virtue of the group being
                        // dead, nobody will ever kill-all (foreach) over it.)
                        if nobe_is_dead { true } else { forward_blk(tg_opt) }
                    };
                /*##########################################################*
                 * Step 2: Recurse on the rest of the list; maybe coalescing.
                 *##########################################################*/
                // 'need_unwind' is only set if blk returned true above, *and*
                // the recursive call early-broke.
                let mut need_unwind = false;
                if do_continue {
                    // NB: Takes many locks! (ancestor nodes & parent groups)
                    need_unwind = coalesce(&mut nobe.ancestors, |tg| bail_blk(tg),
                                           forward_blk, nobe.generation);
                }
                /*##########################################################*
                 * Step 3: Maybe unwind; compute return info for our caller.
                 *##########################################################*/
                if need_unwind && !nobe_is_dead {
                    do access_group(&nobe.parent_group) |tg_opt| {
                        bail_blk(tg_opt)
                    }
                }
                // Decide whether our caller should unwind.
                need_unwind = need_unwind || !do_continue;
                // Tell caller whether or not to coalesce and/or unwind
                if nobe_is_dead {
                    // Swap the list out here; the caller replaces us with it.
                    let rest = util::replace(&mut nobe.ancestors,
                                             AncestorList(None));
                    (Some(rest), need_unwind)
                } else {
                    (None, need_unwind)
                }
            }
        }
    }
}

// One of these per task.
pub struct Taskgroup {
    // List of tasks with whose fates this one's is intertwined.
    tasks:      TaskGroupArc, // 'none' means the group has failed.
    // Lists of tasks who will kill us if they fail, but whom we won't kill.
    ancestors:  AncestorList,
    is_main:    bool,
    notifier:   Option<AutoNotify>,
}

impl Drop for Taskgroup {
    // Runs on task exit.
    fn drop(&self) {
        unsafe {
            // FIXME(#4330) Need self by value to get mutability.
            let this: &mut Taskgroup = transmute(self);

            // If we are failing, the whole taskgroup needs to die.
            do RuntimeGlue::with_task_handle_and_failing |me, failing| {
                if failing {
                    foreach x in this.notifier.mut_iter() {
                        x.failed = true;
                    }
                    // Take everybody down with us.
                    do access_group(&self.tasks) |tg| {
                        kill_taskgroup(tg, &me, self.is_main);
                    }
                } else {
                    // Remove ourselves from the group(s).
                    do access_group(&self.tasks) |tg| {
                        leave_taskgroup(tg, &me, true);
                    }
                }
                // It doesn't matter whether this happens before or after dealing
                // with our own taskgroup, so long as both happen before we die.
                // We remove ourself from every ancestor we can, so no cleanup; no
                // break.
                do each_ancestor(&mut this.ancestors, |_| {}) |ancestor_group| {
                    leave_taskgroup(ancestor_group, &me, false);
                    true
                };
            }
        }
    }
}

pub fn Taskgroup(tasks: TaskGroupArc,
       ancestors: AncestorList,
       is_main: bool,
       mut notifier: Option<AutoNotify>) -> Taskgroup {
    foreach x in notifier.mut_iter() {
        x.failed = false;
    }

    Taskgroup {
        tasks: tasks,
        ancestors: ancestors,
        is_main: is_main,
        notifier: notifier
    }
}

struct AutoNotify {
    notify_chan: Chan<TaskResult>,
    failed: bool,
}

impl Drop for AutoNotify {
    fn drop(&self) {
        let result = if self.failed { Failure } else { Success };
        self.notify_chan.send(result);
    }
}

fn AutoNotify(chan: Chan<TaskResult>) -> AutoNotify {
    AutoNotify {
        notify_chan: chan,
        failed: true // Un-set above when taskgroup successfully made.
    }
}

fn enlist_in_taskgroup(state: TaskGroupInner, me: TaskHandle,
                           is_member: bool) -> bool {
    let me = Cell::new(me); // :(
    // If 'None', the group was failing. Can't enlist.
    do state.map_mut_default(false) |group| {
        (if is_member {
            &mut group.members
        } else {
            &mut group.descendants
        }).insert(me.take());
        true
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn leave_taskgroup(state: TaskGroupInner, me: &TaskHandle,
                       is_member: bool) {
    let me = Cell::new(me); // :(
    // If 'None', already failing and we've already gotten a kill signal.
    do state.map_mut |group| {
        (if is_member {
            &mut group.members
        } else {
            &mut group.descendants
        }).remove(me.take());
    };
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn kill_taskgroup(state: TaskGroupInner, me: &TaskHandle, is_main: bool) {
    unsafe {
        // NB: We could do the killing iteration outside of the group arc, by
        // having "let mut newstate" here, swapping inside, and iterating
        // after. But that would let other exiting tasks fall-through and exit
        // while we were trying to kill them, causing potential
        // use-after-free. A task's presence in the arc guarantees it's alive
        // only while we hold the lock, so if we're failing, all concurrently
        // exiting tasks must wait for us. To do it differently, we'd have to
        // use the runtime's task refcounting, but that could leave task
        // structs around long after their task exited.
        let newstate = util::replace(state, None);
        // Might already be None, if Somebody is failing simultaneously.
        // That's ok; only one task needs to do the dirty work. (Might also
        // see 'None' if Somebody already failed and we got a kill signal.)
        if newstate.is_some() {
            let TaskGroupData { members: members, descendants: descendants } =
                newstate.unwrap();
            foreach sibling in members.consume() {
                // Skip self - killing ourself won't do much good.
                if &sibling != me {
                    RuntimeGlue::kill_task(sibling);
                }
            }
            foreach child in descendants.consume() {
                assert!(&child != me);
                RuntimeGlue::kill_task(child);
            }
            // Only one task should ever do this.
            if is_main {
                RuntimeGlue::kill_all_tasks(me);
            }
            // Do NOT restore state to Some(..)! It stays None to indicate
            // that the whole taskgroup is failing, to forbid new spawns.
        }
        // (note: multiple tasks may reach this point)
    }
}

// FIXME (#2912): Work around core-vs-coretest function duplication. Can't use
// a proper closure because the #[test]s won't understand. Have to fake it.
fn taskgroup_key() -> local_data::Key<@@mut Taskgroup> {
    unsafe { cast::transmute(-2) }
}

// Transitionary.
struct RuntimeGlue;
impl RuntimeGlue {
    unsafe fn kill_task(task: TaskHandle) {
        match task {
            OldTask(ptr) => rt::rust_task_kill_other(ptr),
            NewTask(handle) => {
                let mut handle = handle;
                do handle.kill().map_consume |killed_task| {
                    let killed_task = Cell::new(killed_task);
                    do Local::borrow::<Scheduler, ()> |sched| {
                        sched.enqueue_task(killed_task.take());
                    }
                };
            }
        }
    }

    unsafe fn kill_all_tasks(task: &TaskHandle) {
        match *task {
            OldTask(ptr) => rt::rust_task_kill_all(ptr),
            // FIXME(#7544): Remove the kill_all feature entirely once the
            // oldsched goes away.
            NewTask(ref _handle) => rtabort!("can't kill_all in newsched"),
        }
    }

    fn with_task_handle_and_failing(blk: &fn(TaskHandle, bool)) {
        match context() {
            OldTaskContext => unsafe {
                let me = rt::rust_get_task();
                blk(OldTask(me), rt::rust_task_is_unwinding(me))
            },
            TaskContext => unsafe {
                // Can't use safe borrow, because the taskgroup destructor needs to
                // access the scheduler again to send kill signals to other tasks.
                let me = Local::unsafe_borrow::<Task>();
                // FIXME(#7544): Get rid of this clone by passing by-ref.
                // Will probably have to wait until the old rt is gone.
                blk(NewTask((*me).death.kill_handle.get_ref().clone()),
                    (*me).unwinder.unwinding)
            },
            SchedulerContext | GlobalContext => rtabort!("task dying in bad context"),
        }
    }

    fn with_my_taskgroup<U>(blk: &fn(&Taskgroup) -> U) -> U {
        match context() {
            OldTaskContext => unsafe {
                let me = rt::rust_get_task();
                do local_get(OldHandle(me), taskgroup_key()) |g| {
                    match g {
                        None => {
                            // Main task, doing first spawn ever. Lazily initialise here.
                            let mut members = TaskSet::new();
                            members.insert(OldTask(me));
                            let tasks = Exclusive::new(Some(TaskGroupData {
                                members: members,
                                descendants: TaskSet::new(),
                            }));
                            // Main task/group has no ancestors, no notifier, etc.
                            let group = @@mut Taskgroup(tasks, AncestorList(None),
                                                        true, None);
                            local_set(OldHandle(me), taskgroup_key(), group);
                            blk(&**group)
                        }
                        Some(&group) => blk(&**group)
                    }
                }
            },
            TaskContext => unsafe {
                // Can't use safe borrow, because creating new hashmaps for the
                // tasksets requires an rng, which needs to borrow the sched.
                let me = Local::unsafe_borrow::<Task>();
                blk(match (*me).taskgroup {
                    None => {
                        // Main task, doing first spawn ever. Lazily initialize.
                        let mut members = TaskSet::new();
                        let my_handle = (*me).death.kill_handle.get_ref().clone();
                        members.insert(NewTask(my_handle));
                        let tasks = Exclusive::new(Some(TaskGroupData {
                            members: members,
                            descendants: TaskSet::new(),
                        }));
                        // FIXME(#7544): Remove the is_main flag entirely once
                        // the newsched goes away. The main taskgroup has no special
                        // behaviour.
                        let group = Taskgroup(tasks, AncestorList(None), false, None);
                        (*me).taskgroup = Some(group);
                        (*me).taskgroup.get_ref()
                    }
                    Some(ref group) => group,
                })
            },
            SchedulerContext | GlobalContext => rtabort!("spawning in bad context"),
        }
    }
}

fn gen_child_taskgroup(linked: bool, supervised: bool)
    -> (TaskGroupArc, AncestorList, bool) {
    do RuntimeGlue::with_my_taskgroup |spawner_group| {
        let ancestors = AncestorList(spawner_group.ancestors.map(|x| x.clone()));
        if linked {
            // Child is in the same group as spawner.
            // Child's ancestors are spawner's ancestors.
            // Propagate main-ness.
            (spawner_group.tasks.clone(), ancestors, spawner_group.is_main)
        } else {
            // Child is in a separate group from spawner.
            let g = Exclusive::new(Some(TaskGroupData {
                members:     TaskSet::new(),
                descendants: TaskSet::new(),
            }));
            let a = if supervised {
                let new_generation = incr_generation(&ancestors);
                assert!(new_generation < uint::max_value);
                // Child's ancestors start with the spawner.
                // Build a new node in the ancestor list.
                AncestorList(Some(Exclusive::new(AncestorNode {
                    generation: new_generation,
                    parent_group: spawner_group.tasks.clone(),
                    ancestors: ancestors,
                })))
            } else {
                // Child has no ancestors.
                AncestorList(None)
            };
            (g, a, false)
        }
    }
}

// Set up membership in taskgroup and descendantship in all ancestor
// groups. If any enlistment fails, Some task was already failing, so
// don't let the child task run, and undo every successful enlistment.
fn enlist_many(child: TaskHandle, child_arc: &TaskGroupArc,
               ancestors: &mut AncestorList) -> bool {
    // Join this taskgroup.
    let mut result = do access_group(child_arc) |child_tg| {
        enlist_in_taskgroup(child_tg, child.clone(), true) // member
    };
    if result {
        // Unwinding function in case any ancestral enlisting fails
        let bail: &fn(TaskGroupInner) = |tg| { leave_taskgroup(tg, &child, false) };
        // Attempt to join every ancestor group.
        result = do each_ancestor(ancestors, bail) |ancestor_tg| {
            // Enlist as a descendant, not as an actual member.
            // Descendants don't kill ancestor groups on failure.
            enlist_in_taskgroup(ancestor_tg, child.clone(), false)
        };
        // If any ancestor group fails, need to exit this group too.
        if !result {
            do access_group(child_arc) |child_tg| {
                leave_taskgroup(child_tg, &child, true); // member
            }
        }
    }
    result
}

pub fn spawn_raw(opts: TaskOpts, f: ~fn()) {
    match context() {
        OldTaskContext   => spawn_raw_oldsched(opts, f),
        TaskContext      => spawn_raw_newsched(opts, f),
        SchedulerContext => fail!("can't spawn from scheduler context"),
        GlobalContext    => fail!("can't spawn from global context"),
    }
}

fn spawn_raw_newsched(mut opts: TaskOpts, f: ~fn()) {
    use rt::sched::*;

    let child_data = Cell::new(gen_child_taskgroup(opts.linked, opts.supervised));
    let indestructible = opts.indestructible;

    let child_wrapper: ~fn() = || {
        // Child task runs this code.
        let child_data = Cell::new(child_data.take()); // :(
        let enlist_success = do Local::borrow::<Task, bool> |me| {
            let (child_tg, ancestors, is_main) = child_data.take();
            let mut ancestors = ancestors;
            // FIXME(#7544): Optimize out the xadd in this clone, somehow.
            let handle = me.death.kill_handle.get_ref().clone();
            // Atomically try to get into all of our taskgroups.
            if enlist_many(NewTask(handle), &child_tg, &mut ancestors) {
                // Got in. We can run the provided child body, and can also run
                // the taskgroup's exit-time-destructor afterward.
                me.taskgroup = Some(Taskgroup(child_tg, ancestors, is_main, None));
                true
            } else {
                false
            }
        };
        // Should be run after the local-borrowed task is returned.
        if enlist_success {
            if indestructible {
                do unkillable { f() }
            } else {
                f()
            }
        }
    };

    let mut task = unsafe {
        if opts.sched.mode != SingleThreaded {
            if opts.watched {
                Task::build_child(child_wrapper)
            } else {
                Task::build_root(child_wrapper)
            }
        } else {
            // Creating a 1:1 task:thread ...
            let sched = Local::unsafe_borrow::<Scheduler>();
            let sched_handle = (*sched).make_handle();

            // Create a new scheduler to hold the new task
            let new_loop = ~UvEventLoop::new();
            let mut new_sched = ~Scheduler::new_special(new_loop,
                                                        (*sched).work_queue.clone(),
                                                        (*sched).sleeper_list.clone(),
                                                        false,
                                                        Some(sched_handle));
            let mut new_sched_handle = new_sched.make_handle();

            // Allow the scheduler to exit when the pinned task exits
            new_sched_handle.send(Shutdown);

            // Pin the new task to the new scheduler
            let new_task = if opts.watched {
                Task::build_homed_child(child_wrapper, Sched(new_sched_handle))
            } else {
                Task::build_homed_root(child_wrapper, Sched(new_sched_handle))
            };

            // Create a task that will later be used to join with the new scheduler
            // thread when it is ready to terminate
            let (thread_port, thread_chan) = oneshot();
            let thread_port_cell = Cell::new(thread_port);
            let join_task = do Task::build_child() {
                rtdebug!("running join task");
                let thread_port = thread_port_cell.take();
                let thread: Thread = thread_port.recv();
                thread.join();
            };

            // Put the scheduler into another thread
            let new_sched_cell = Cell::new(new_sched);
            let orig_sched_handle_cell = Cell::new((*sched).make_handle());
            let join_task_cell = Cell::new(join_task);

            let thread = do Thread::start {
                let mut new_sched = new_sched_cell.take();
                let mut orig_sched_handle = orig_sched_handle_cell.take();
                let join_task = join_task_cell.take();

                let bootstrap_task = ~do Task::new_root(&mut new_sched.stack_pool) || {
                    rtdebug!("boostraping a 1:1 scheduler");
                };
                new_sched.bootstrap(bootstrap_task);

                rtdebug!("enqueing join_task");
                // Now tell the original scheduler to join with this thread
                // by scheduling a thread-joining task on the original scheduler
                orig_sched_handle.send(TaskFromFriend(join_task));

                // NB: We can't simply send a message from here to another task
                // because this code isn't running in a task and message passing doesn't
                // work outside of tasks. Hence we're sending a scheduler message
                // to execute a new task directly to a scheduler.
            };

            // Give the thread handle to the join task
            thread_chan.send(thread);

            // When this task is enqueued on the current scheduler it will then get
            // forwarded to the scheduler to which it is pinned
            new_task
        }
    };

    if opts.notify_chan.is_some() {
        let notify_chan = opts.notify_chan.take_unwrap();
        let notify_chan = Cell::new(notify_chan);
        let on_exit: ~fn(bool) = |success| {
            notify_chan.take().send(
                if success { Success } else { Failure }
            )
        };
        task.death.on_exit = Some(on_exit);
    }

    task.name = opts.name.take();
    rtdebug!("spawn calling run_task");
    Scheduler::run_task(task);

}

fn spawn_raw_oldsched(mut opts: TaskOpts, f: ~fn()) {

    let (child_tg, ancestors, is_main) =
        gen_child_taskgroup(opts.linked, opts.supervised);

    unsafe {
        let child_data = Cell::new((child_tg, ancestors, f));
        // Being killed with the unsafe task/closure pointers would leak them.
        do unkillable {
            let (child_tg, ancestors, f) = child_data.take(); // :(
            // Create child task.
            let new_task = match opts.sched.mode {
                DefaultScheduler => rt::new_task(),
                _ => new_task_in_sched()
            };
            assert!(!new_task.is_null());
            // Getting killed after here would leak the task.
            let child_wrapper = make_child_wrapper(new_task, child_tg,
                  ancestors, is_main, opts.notify_chan.take(), f);

            let closure = cast::transmute(&child_wrapper);

            // Getting killed between these two calls would free the child's
            // closure. (Reordering them wouldn't help - then getting killed
            // between them would leak.)
            rt::start_task(new_task, closure);
            cast::forget(child_wrapper);
        }
    }

    // This function returns a closure-wrapper that we pass to the child task.
    // (1) It sets up the notification channel.
    // (2) It attempts to enlist in the child's group and all ancestor groups.
    // (3a) If any of those fails, it leaves all groups, and does nothing.
    // (3b) Otherwise it builds a task control structure and puts it in TLS,
    // (4) ...and runs the provided body function.
    fn make_child_wrapper(child: *rust_task, child_arc: TaskGroupArc,
                          ancestors: AncestorList, is_main: bool,
                          notify_chan: Option<Chan<TaskResult>>,
                          f: ~fn())
                       -> ~fn() {
        let child_data = Cell::new((notify_chan, child_arc, ancestors));
        let result: ~fn() = || {
            let (notify_chan, child_arc, ancestors) = child_data.take(); // :(
            let mut ancestors = ancestors;
            // Child task runs this code.

            // Even if the below code fails to kick the child off, we must
            // send Something on the notify channel.

            let notifier = notify_chan.map_consume(|c| AutoNotify(c));

            if enlist_many(OldTask(child), &child_arc, &mut ancestors) {
                let group = @@mut Taskgroup(child_arc, ancestors, is_main, notifier);
                unsafe {
                    local_set(OldHandle(child), taskgroup_key(), group);
                }

                // Run the child's body.
                f();

                // TLS cleanup code will exit the taskgroup.
            }

            // Run the box annihilator.
            // FIXME #4428: Crashy.
            // unsafe { cleanup::annihilate(); }
        };
        return result;
    }

    fn new_task_in_sched() -> *rust_task {
        unsafe {
            let sched_id = rt::rust_new_sched(1);
            rt::rust_new_task_in_sched(sched_id)
        }
    }
}

#[test]
fn test_spawn_raw_simple() {
    let (po, ch) = stream();
    do spawn_raw(default_task_opts()) {
        ch.send(());
    }
    po.recv();
}

#[test]
#[ignore(cfg(windows))]
fn test_spawn_raw_unsupervise() {
    let opts = task::TaskOpts {
        linked: false,
        watched: false,
        notify_chan: None,
        .. default_task_opts()
    };
    do spawn_raw(opts) {
        fail!();
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_spawn_raw_notify_success() {
    let (notify_po, notify_ch) = comm::stream();

    let opts = task::TaskOpts {
        notify_chan: Some(notify_ch),
        .. default_task_opts()
    };
    do spawn_raw(opts) {
    }
    assert_eq!(notify_po.recv(), Success);
}

#[test]
#[ignore(cfg(windows))]
fn test_spawn_raw_notify_failure() {
    // New bindings for these
    let (notify_po, notify_ch) = comm::stream();

    let opts = task::TaskOpts {
        linked: false,
        watched: false,
        notify_chan: Some(notify_ch),
        .. default_task_opts()
    };
    do spawn_raw(opts) {
        fail!();
    }
    assert_eq!(notify_po.recv(), Failure);
}
