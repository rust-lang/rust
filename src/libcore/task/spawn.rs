// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

#[doc(hidden)]; // FIXME #3538

use cast::transmute;
use cast;
use cell::Cell;
use container::Map;
use comm::{Chan, GenericChan};
use prelude::*;
use ptr;
use hashmap::HashSet;
use task::local_data_priv::{local_get, local_set, OldHandle};
use task::rt::rust_task;
use task::rt;
use task::{Failure, ManualThreads, PlatformThread, SchedOpts, SingleThreaded};
use task::{Success, TaskOpts, TaskResult, ThreadPerCore, ThreadPerTask};
use task::{ExistingScheduler, SchedulerHandle};
use task::unkillable;
use uint;
use util;
use unstable::sync::{Exclusive, exclusive};

#[cfg(test)] use task::default_task_opts;

macro_rules! move_it (
    { $x:expr } => ( unsafe { let y = *ptr::to_unsafe_ptr(&($x)); y } )
)

type TaskSet = HashSet<*rust_task>;

fn new_taskset() -> TaskSet {
    HashSet::new()
}
fn taskset_insert(tasks: &mut TaskSet, task: *rust_task) {
    let didnt_overwrite = tasks.insert(task);
    assert!(didnt_overwrite);
}
fn taskset_remove(tasks: &mut TaskSet, task: *rust_task) {
    let was_present = tasks.remove(&task);
    assert!(was_present);
}
pub fn taskset_each(tasks: &TaskSet, blk: &fn(v: *rust_task) -> bool) -> bool {
    tasks.each(|k| blk(*k))
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
    (&const tg.members).is_empty()
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
    // FIXME(#3068): Make the generation counter togglable with #[cfg(debug)].
    generation:     uint,
    // Should really be a non-option. This way appeases borrowck.
    parent_group:   Option<TaskGroupArc>,
    // Recursive rest of the list.
    ancestors:      AncestorList,
}

struct AncestorList(Option<Exclusive<AncestorNode>>);

// Accessors for taskgroup arcs and ancestor arcs that wrap the unsafety.
#[inline(always)]
fn access_group<U>(x: &TaskGroupArc, blk: &fn(TaskGroupInner) -> U) -> U {
    x.with(blk)
}

#[inline(always)]
fn access_ancestors<U>(x: &Exclusive<AncestorNode>,
                       blk: &fn(x: &mut AncestorNode) -> U) -> U {
    x.with(blk)
}

// Iterates over an ancestor list.
// (1) Runs forward_blk on each ancestral taskgroup in the list
// (2) If forward_blk "break"s, runs optional bail_blk on all ancestral
//     taskgroups that forward_blk already ran on successfully (Note: bail_blk
//     is NOT called on the block that forward_blk broke on!).
// (3) As a bonus, coalesces away all 'dead' taskgroup nodes in the list.
// FIXME(#2190): Change Option<@fn(...)> to Option<&fn(...)>, to save on
// allocations. Once that bug is fixed, changing the sigil should suffice.
fn each_ancestor(list:        &mut AncestorList,
                 bail_opt:    Option<@fn(TaskGroupInner)>,
                 forward_blk: &fn(TaskGroupInner) -> bool)
              -> bool {
    // "Kickoff" call - there was no last generation.
    return !coalesce(list, bail_opt, forward_blk, uint::max_value);

    // Recursively iterates, and coalesces afterwards if needed. Returns
    // whether or not unwinding is needed (i.e., !successful iteration).
    fn coalesce(list:            &mut AncestorList,
                bail_opt:        Option<@fn(TaskGroupInner)>,
                forward_blk:     &fn(TaskGroupInner) -> bool,
                last_generation: uint) -> bool {
        // Need to swap the list out to use it, to appease borrowck.
        let tmp_list = util::replace(&mut *list, AncestorList(None));
        let (coalesce_this, early_break) =
            iterate(&tmp_list, bail_opt, forward_blk, last_generation);
        // What should our next ancestor end up being?
        if coalesce_this.is_some() {
            // Needed coalesce. Our next ancestor becomes our old
            // ancestor's next ancestor. ("next = old_next->next;")
            *list = coalesce_this.unwrap();
        } else {
            // No coalesce; restore from tmp. ("next = old_next;")
            *list = tmp_list;
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
    fn iterate(ancestors:       &AncestorList,
               bail_opt:        Option<@fn(TaskGroupInner)>,
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

        // The map defaults to None, because if ancestors is None, we're at
        // the end of the list, which doesn't make sense to coalesce.
        return do (**ancestors).map_default((None,false)) |ancestor_arc| {
            // NB: Takes a lock! (this ancestor node)
            do access_ancestors(ancestor_arc) |nobe| {
                // Check monotonicity
                assert!(last_generation > nobe.generation);
                /*##########################################################*
                 * Step 1: Look at this ancestor group (call iterator block).
                 *##########################################################*/
                let mut nobe_is_dead = false;
                let do_continue =
                    // NB: Takes a lock! (this ancestor node's parent group)
                    do with_parent_tg(&mut nobe.parent_group) |tg_opt| {
                        // Decide whether this group is dead. Note that the
                        // group being *dead* is disjoint from it *failing*.
                        nobe_is_dead = match *tg_opt {
                            Some(ref tg) => taskgroup_is_dead(tg),
                            None => nobe_is_dead
                        };
                        // Call iterator block. (If the group is dead, it's
                        // safe to skip it. This will leave our *rust_task
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
                    need_unwind = coalesce(&mut nobe.ancestors, bail_opt,
                                           forward_blk, nobe.generation);
                }
                /*##########################################################*
                 * Step 3: Maybe unwind; compute return info for our caller.
                 *##########################################################*/
                if need_unwind && !nobe_is_dead {
                    for bail_opt.each |bail_blk| {
                        do with_parent_tg(&mut nobe.parent_group) |tg_opt| {
                            (*bail_blk)(tg_opt)
                        }
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
        };

        // Wrapper around exclusive::with that appeases borrowck.
        fn with_parent_tg<U>(parent_group: &mut Option<TaskGroupArc>,
                             blk: &fn(TaskGroupInner) -> U) -> U {
            // If this trips, more likely the problem is 'blk' failed inside.
            let tmp_arc = parent_group.swap_unwrap();
            let result = do access_group(&tmp_arc) |tg_opt| { blk(tg_opt) };
            *parent_group = Some(tmp_arc);
            result
        }
    }
}

// One of these per task.
struct TCB {
    me:         *rust_task,
    // List of tasks with whose fates this one's is intertwined.
    tasks:      TaskGroupArc, // 'none' means the group has failed.
    // Lists of tasks who will kill us if they fail, but whom we won't kill.
    ancestors:  AncestorList,
    is_main:    bool,
    notifier:   Option<AutoNotify>,
}

impl Drop for TCB {
    // Runs on task exit.
    fn finalize(&self) {
        unsafe {
            let this: &mut TCB = transmute(self);

            // If we are failing, the whole taskgroup needs to die.
            if rt::rust_task_is_unwinding(self.me) {
                for this.notifier.each_mut |x| {
                    x.failed = true;
                }
                // Take everybody down with us.
                do access_group(&self.tasks) |tg| {
                    kill_taskgroup(tg, self.me, self.is_main);
                }
            } else {
                // Remove ourselves from the group(s).
                do access_group(&self.tasks) |tg| {
                    leave_taskgroup(tg, self.me, true);
                }
            }
            // It doesn't matter whether this happens before or after dealing
            // with our own taskgroup, so long as both happen before we die.
            // We remove ourself from every ancestor we can, so no cleanup; no
            // break.
            for each_ancestor(&mut this.ancestors, None) |ancestor_group| {
                leave_taskgroup(ancestor_group, self.me, false);
            };
        }
    }
}

fn TCB(me: *rust_task,
       tasks: TaskGroupArc,
       ancestors: AncestorList,
       is_main: bool,
       mut notifier: Option<AutoNotify>) -> TCB {
    for notifier.each_mut |x| {
        x.failed = false;
    }

    TCB {
        me: me,
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
    fn finalize(&self) {
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

fn enlist_in_taskgroup(state: TaskGroupInner, me: *rust_task,
                           is_member: bool) -> bool {
    let newstate = util::replace(&mut *state, None);
    // If 'None', the group was failing. Can't enlist.
    if newstate.is_some() {
        let mut group = newstate.unwrap();
        taskset_insert(if is_member {
            &mut group.members
        } else {
            &mut group.descendants
        }, me);
        *state = Some(group);
        true
    } else {
        false
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn leave_taskgroup(state: TaskGroupInner, me: *rust_task,
                       is_member: bool) {
    let newstate = util::replace(&mut *state, None);
    // If 'None', already failing and we've already gotten a kill signal.
    if newstate.is_some() {
        let mut group = newstate.unwrap();
        taskset_remove(if is_member {
            &mut group.members
        } else {
            &mut group.descendants
        }, me);
        *state = Some(group);
    }
}

// NB: Runs in destructor/post-exit context. Can't 'fail'.
fn kill_taskgroup(state: TaskGroupInner, me: *rust_task, is_main: bool) {
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
            let group = newstate.unwrap();
            for taskset_each(&group.members) |sibling| {
                // Skip self - killing ourself won't do much good.
                if sibling != me {
                    rt::rust_task_kill_other(sibling);
                }
            }
            for taskset_each(&group.descendants) |child| {
                assert!(child != me);
                rt::rust_task_kill_other(child);
            }
            // Only one task should ever do this.
            if is_main {
                rt::rust_task_kill_all(me);
            }
            // Do NOT restore state to Some(..)! It stays None to indicate
            // that the whole taskgroup is failing, to forbid new spawns.
        }
        // (note: multiple tasks may reach this point)
    }
}

// FIXME (#2912): Work around core-vs-coretest function duplication. Can't use
// a proper closure because the #[test]s won't understand. Have to fake it.
macro_rules! taskgroup_key (
    // Use a "code pointer" value that will never be a real code pointer.
    () => (cast::transmute((-2 as uint, 0u)))
)

fn gen_child_taskgroup(linked: bool, supervised: bool)
    -> (TaskGroupArc, AncestorList, bool) {
    unsafe {
        let spawner = rt::rust_get_task();
        /*##################################################################*
         * Step 1. Get spawner's taskgroup info.
         *##################################################################*/
        let spawner_group: @@mut TCB =
            match local_get(OldHandle(spawner), taskgroup_key!()) {
                None => {
                    // Main task, doing first spawn ever. Lazily initialise
                    // here.
                    let mut members = new_taskset();
                    taskset_insert(&mut members, spawner);
                    let tasks = exclusive(Some(TaskGroupData {
                        members: members,
                        descendants: new_taskset(),
                    }));
                    // Main task/group has no ancestors, no notifier, etc.
                    let group = @@mut TCB(spawner,
                                          tasks,
                                          AncestorList(None),
                                          true,
                                          None);
                    local_set(OldHandle(spawner), taskgroup_key!(), group);
                    group
                }
                Some(group) => group
            };
        let spawner_group: &mut TCB = *spawner_group;

        /*##################################################################*
         * Step 2. Process spawn options for child.
         *##################################################################*/
        return if linked {
            // Child is in the same group as spawner.
            let g = spawner_group.tasks.clone();
            // Child's ancestors are spawner's ancestors.
            let a = share_ancestors(&mut spawner_group.ancestors);
            // Propagate main-ness.
            (g, a, spawner_group.is_main)
        } else {
            // Child is in a separate group from spawner.
            let g = exclusive(Some(TaskGroupData {
                members:     new_taskset(),
                descendants: new_taskset(),
            }));
            let a = if supervised {
                // Child's ancestors start with the spawner.
                let old_ancestors =
                    share_ancestors(&mut spawner_group.ancestors);
                // FIXME(#3068) - The generation counter is only used for a
                // debug assertion, but initialising it requires locking a
                // mutex. Hence it should be enabled only in debug builds.
                let new_generation =
                    match *old_ancestors {
                        Some(ref arc) => {
                            access_ancestors(arc, |a| a.generation+1)
                        }
                        None => 0 // the actual value doesn't really matter.
                    };
                assert!(new_generation < uint::max_value);
                // Build a new node in the ancestor list.
                AncestorList(Some(exclusive(AncestorNode {
                    generation: new_generation,
                    parent_group: Some(spawner_group.tasks.clone()),
                    ancestors: old_ancestors,
                })))
            } else {
                // Child has no ancestors.
                AncestorList(None)
            };
            (g, a, false)
        };
    }

    fn share_ancestors(ancestors: &mut AncestorList) -> AncestorList {
        // Appease the borrow-checker. Really this wants to be written as:
        // match ancestors
        //    Some(ancestor_arc) { ancestor_list(Some(ancestor_arc.clone())) }
        //    None               { ancestor_list(None) }
        let tmp = util::replace(&mut **ancestors, None);
        if tmp.is_some() {
            let ancestor_arc = tmp.unwrap();
            let result = ancestor_arc.clone();
            **ancestors = Some(ancestor_arc);
            AncestorList(Some(result))
        } else {
            AncestorList(None)
        }
    }
}

pub fn spawn_raw(opts: TaskOpts, f: ~fn()) {
    use rt::*;

    match context() {
        OldTaskContext => {
            spawn_raw_oldsched(opts, f)
        }
        TaskContext => {
            spawn_raw_newsched(opts, f)
        }
        SchedulerContext => {
            fail!("can't spawn from scheduler context")
        }
        GlobalContext => {
            fail!("can't spawn from global context")
        }
    }
}

fn spawn_raw_newsched(_opts: TaskOpts, f: ~fn()) {
    use rt::sched::*;

    let mut sched = local_sched::take();
    let task = ~Coroutine::new(&mut sched.stack_pool, f);
    sched.schedule_new_task(task);
}

fn spawn_raw_oldsched(mut opts: TaskOpts, f: ~fn()) {

    let (child_tg, ancestors, is_main) =
        gen_child_taskgroup(opts.linked, opts.supervised);

    unsafe {
        let child_data = Cell((child_tg, ancestors, f));
        // Being killed with the unsafe task/closure pointers would leak them.
        do unkillable {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let (child_tg, ancestors, f) = child_data.take();
            // Create child task.
            let new_task = match opts.sched.mode {
                DefaultScheduler => rt::new_task(),
                _ => new_task_in_sched(opts.sched)
            };
            assert!(!new_task.is_null());
            // Getting killed after here would leak the task.
            let notify_chan = if opts.notify_chan.is_none() {
                None
            } else {
                Some(opts.notify_chan.swap_unwrap())
            };

            let child_wrapper = make_child_wrapper(new_task, child_tg,
                  ancestors, is_main, notify_chan, f);

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
        let child_data = Cell((child_arc, ancestors));
        let result: ~fn() = || {
            // Agh. Get move-mode items into the closure. FIXME (#2829)
            let mut (child_arc, ancestors) = child_data.take();
            // Child task runs this code.

            // Even if the below code fails to kick the child off, we must
            // send Something on the notify channel.

            //let mut notifier = None;//notify_chan.map(|c| AutoNotify(c));
            let notifier = match notify_chan {
                Some(ref notify_chan_value) => {
                    let moved_ncv = move_it!(*notify_chan_value);
                    Some(AutoNotify(moved_ncv))
                }
                _ => None
            };

            if enlist_many(child, &child_arc, &mut ancestors) {
                let group = @@mut TCB(child,
                                      child_arc,
                                      ancestors,
                                      is_main,
                                      notifier);
                unsafe {
                    local_set(OldHandle(child), taskgroup_key!(), group);
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

        // Set up membership in taskgroup and descendantship in all ancestor
        // groups. If any enlistment fails, Some task was already failing, so
        // don't let the child task run, and undo every successful enlistment.
        fn enlist_many(child: *rust_task, child_arc: &TaskGroupArc,
                       ancestors: &mut AncestorList) -> bool {
            // Join this taskgroup.
            let mut result =
                do access_group(child_arc) |child_tg| {
                    enlist_in_taskgroup(child_tg, child, true) // member
                };
            if result {
                // Unwinding function in case any ancestral enlisting fails
                let bail: @fn(TaskGroupInner) = |tg| {
                    leave_taskgroup(tg, child, false)
                };
                // Attempt to join every ancestor group.
                result =
                    each_ancestor(ancestors, Some(bail), |ancestor_tg| {
                        // Enlist as a descendant, not as an actual member.
                        // Descendants don't kill ancestor groups on failure.
                        enlist_in_taskgroup(ancestor_tg, child, false)
                    });
                // If any ancestor group fails, need to exit this group too.
                if !result {
                    do access_group(child_arc) |child_tg| {
                        leave_taskgroup(child_tg, child, true); // member
                    }
                }
            }
            result
        }
    }

    fn new_task_in_sched(opts: SchedOpts) -> *rust_task {
        if opts.foreign_stack_size != None {
            fail!("foreign_stack_size scheduler option unimplemented");
        }

        let num_threads = match opts.mode {
            DefaultScheduler
            | CurrentScheduler
            | ExistingScheduler(*)
            | PlatformThread => 0u, /* Won't be used */
            SingleThreaded => 1u,
            ThreadPerCore => unsafe { rt::rust_num_threads() },
            ThreadPerTask => {
                fail!("ThreadPerTask scheduling mode unimplemented")
            }
            ManualThreads(threads) => {
                if threads == 0u {
                    fail!("can not create a scheduler with no threads");
                }
                threads
            }
        };

        unsafe {
            let sched_id = match opts.mode {
                CurrentScheduler => rt::rust_get_sched_id(),
                ExistingScheduler(SchedulerHandle(id)) => id,
                PlatformThread => rt::rust_osmain_sched_id(),
                _ => rt::rust_new_sched(num_threads)
            };
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
        notify_chan: Some(notify_ch),
        .. default_task_opts()
    };
    do spawn_raw(opts) {
        fail!();
    }
    assert_eq!(notify_po.recv(), Failure);
}
