// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! One of the major goals behind this channel implementation is to work
//! seamlessly on and off the runtime. This also means that the code isn't
//! littered with "if is_green() { ... } else { ... }". Right now, the rest of
//! the runtime isn't quite ready to for this abstraction to be done very
//! nicely, so the conditional "if green" blocks are all contained in this inner
//! module.
//!
//! The goal of this module is to mirror what the runtime "should be", not the
//! state that it is currently in today. You'll notice that there is no mention
//! of schedulers or is_green inside any of the channel code, it is currently
//! entirely contained in this one module.
//!
//! In the ideal world, nothing in this module exists and it is all implemented
//! elsewhere in the runtime (in the proper location). All of this code is
//! structured in order to easily refactor this to the correct location whenever
//! we have the trait objects in place to serve as the boundary of the
//! abstraction.

use iter::{range, Iterator};
use ops::Drop;
use option::{Some, None, Option};
use rt::local::Local;
use rt::sched::{SchedHandle, Scheduler, TaskFromFriend};
use rt::thread::Thread;
use rt;
use unstable::mutex::Mutex;
use unstable::sync::UnsafeArc;

// A task handle is a method of waking up a blocked task. The handle itself
// is completely opaque and only has a wake() method defined on it. This
// method will wake the method regardless of the context of the thread which
// is currently calling wake().
//
// This abstraction should be able to be created when putting a task to
// sleep. This should basically be a method on whatever the local Task is,
// consuming the local Task.

pub struct TaskHandle {
    priv inner: TaskRepr
}
enum TaskRepr {
    Green(rt::BlockedTask, *mut SchedHandle),
    Native(NativeWakeupStyle),
}
enum NativeWakeupStyle {
    ArcWakeup(UnsafeArc<Mutex>),    // shared mutex to synchronize on
    LocalWakeup(*mut Mutex),        // synchronize on the task-local mutex
}

impl TaskHandle {
    // Signal that this handle should be woken up. The `can_resched`
    // argument indicates whether the current task could possibly be
    // rescheduled or not. This does not have a lot of meaning for the
    // native case, but for an M:N case it indicates whether a context
    // switch can happen or not.
    pub fn wake(self, can_resched: bool) {
        match self.inner {
            Green(task, handle) => {
                // If we have a local scheduler, then use that to run the
                // blocked task, otherwise we can use the handle to send the
                // task back to its home.
                if rt::in_green_task_context() {
                    if can_resched {
                        task.wake().map(Scheduler::run_task);
                    } else {
                        let mut s: ~Scheduler = Local::take();
                        s.enqueue_blocked_task(task);
                        Local::put(s);
                    }
                } else {
                    let task = match task.wake() {
                        Some(task) => task, None => return
                    };
                    // XXX: this is not an easy section of code to refactor.
                    //      If this handle is owned by the Task (which it
                    //      should be), then this would be a use-after-free
                    //      because once the task is pushed onto the message
                    //      queue, the handle is gone.
                    //
                    //      Currently the handle is instead owned by the
                    //      Port/Chan pair, which means that because a
                    //      channel is invoking this method the handle will
                    //      continue to stay alive for the entire duration
                    //      of this method. This will require thought when
                    //      moving the handle into the task.
                    unsafe { (*handle).send(TaskFromFriend(task)) }
                }
            }

            // Note that there are no use-after-free races in this code. In
            // the arc-case, we own the lock, and in the local case, we're
            // using a lock so it's guranteed that they aren't running while
            // we hold the lock.
            Native(ArcWakeup(lock)) => {
                unsafe {
                    let lock = lock.get();
                    (*lock).lock();
                    (*lock).signal();
                    (*lock).unlock();
                }
            }
            Native(LocalWakeup(lock)) => {
                unsafe {
                    (*lock).lock();
                    (*lock).signal();
                    (*lock).unlock();
                }
            }
        }
    }

    // Trashes handle to this task. This ensures that necessary memory is
    // deallocated, and there may be some extra assertions as well.
    pub fn trash(self) {
        match self.inner {
            Green(task, _) => task.assert_already_awake(),
            Native(..) => {}
        }
    }
}

// This structure is an abstraction of what should be stored in the local
// task itself. This data is currently stored inside of each channel, but
// this should rather be stored in each task (and channels will still
// continue to lazily initialize this data).

pub struct TaskData {
    priv handle: Option<SchedHandle>,
    priv lock: Mutex,
}

impl TaskData {
    pub fn new() -> TaskData {
        TaskData {
            handle: None,
            lock: unsafe { Mutex::empty() },
        }
    }
}

impl Drop for TaskData {
    fn drop(&mut self) {
        unsafe { self.lock.destroy() }
    }
}

// Now this is the really fun part. This is where all the M:N/1:1-agnostic
// along with recv/select-agnostic blocking information goes. A "blocking
// context" is really just a stack-allocated structure (which is probably
// fine to be a stack-trait-object).
//
// This has some particularly strange interfaces, but the reason for all
// this is to support selection/recv/1:1/M:N all in one bundle.

pub struct BlockingContext<'a> {
    priv inner: BlockingRepr<'a>
}

enum BlockingRepr<'a> {
    GreenBlock(rt::BlockedTask, &'a mut Scheduler),
    NativeBlock(Option<UnsafeArc<Mutex>>),
}

impl<'a> BlockingContext<'a> {
    // Creates one blocking context. The data provided should in theory be
    // acquired from the local task, but it is instead acquired from the
    // channel currently.
    //
    // This function will call `f` with a blocking context, plus the data
    // that it is given. This function will then return whether this task
    // should actually go to sleep or not. If `true` is returned, then this
    // function does not return until someone calls `wake()` on the task.
    // If `false` is returned, then this function immediately returns.
    //
    // # Safety note
    //
    // Note that this stack closure may not be run on the same stack as when
    // this function was called. This means that the environment of this
    // stack closure could be unsafely aliased. This is currently prevented
    // through the guarantee that this function will never return before `f`
    // finishes executing.
    pub fn one(data: &mut TaskData,
               f: |BlockingContext, &mut TaskData| -> bool) {
        if rt::in_green_task_context() {
            let sched: ~Scheduler = Local::take();
            sched.deschedule_running_task_and_then(|sched, task| {
                let ctx = BlockingContext { inner: GreenBlock(task, sched) };
                // no need to do something on success/failure other than
                // returning because the `block` function for a BlockingContext
                // takes care of reawakening itself if the blocking procedure
                // fails. If this function is successful, then we're already
                // blocked, and if it fails, the task will already be
                // rescheduled.
                f(ctx, data);
            });
        } else {
            unsafe { data.lock.lock(); }
            let ctx = BlockingContext { inner: NativeBlock(None) };
            if f(ctx, data) {
                unsafe { data.lock.wait(); }
            }
            unsafe { data.lock.unlock(); }
        }
    }

    // Creates many blocking contexts. The intended use case for this
    // function is selection over a number of ports. This will create `amt`
    // blocking contexts, yielding them to `f` in turn. If `f` returns
    // false, then this function aborts and returns immediately. If `f`
    // repeatedly returns `true` `amt` times, then this function will block.
    pub fn many(amt: uint, f: |BlockingContext| -> bool) {
        if rt::in_green_task_context() {
            let sched: ~Scheduler = Local::take();
            sched.deschedule_running_task_and_then(|sched, task| {
                for handle in task.make_selectable(amt) {
                    let ctx = BlockingContext {
                        inner: GreenBlock(handle, sched)
                    };
                    // see comment above in `one` for why no further action is
                    // necessary here
                    if !f(ctx) { break }
                }
            });
        } else {
            // In the native case, our decision to block must be shared
            // amongst all of the channels. It may be possible to
            // stack-allocate this mutex (instead of putting it in an
            // UnsafeArc box), but for now in order to prevent
            // use-after-free trivially we place this into a box and then
            // pass that around.
            unsafe {
                let mtx = UnsafeArc::new(Mutex::new());
                (*mtx.get()).lock();
                let success = range(0, amt).all(|_| {
                    f(BlockingContext {
                        inner: NativeBlock(Some(mtx.clone()))
                    })
                });
                if success {
                    (*mtx.get()).wait();
                }
                (*mtx.get()).unlock();
            }
        }
    }

    // This function will consume this BlockingContext, and optionally block
    // if according to the atomic `decision` function. The semantics of this
    // functions are:
    //
    //  * `slot` is required to be a `None`-slot (which is owned by the
    //    channel)
    //  * The `slot` will be filled in with a blocked version of the current
    //    task (with `wake`-ability if this function is successful).
    //  * If the `decision` function returns true, then this function
    //    immediately returns having relinquished ownership of the task.
    //  * If the `decision` function returns false, then the `slot` is reset
    //    to `None` and the task is re-scheduled if necessary (remember that
    //    the task will not resume executing before the outer `one` or
    //    `many` function has returned. This function is expected to have a
    //    release memory fence in order for the modifications of `to_wake` to be
    //    visible to other tasks. Code which attempts to read `to_wake` should
    //    have an acquiring memory fence to guarantee that this write is
    //    visible.
    //
    // This function will return whether the blocking occurred or not.
    pub fn block(self,
                 data: &mut TaskData,
                 slot: &mut Option<TaskHandle>,
                 decision: || -> bool) -> bool {
        assert!(slot.is_none());
        match self.inner {
            GreenBlock(task, sched) => {
                if data.handle.is_none() {
                    data.handle = Some(sched.make_handle());
                }
                let handle = data.handle.get_mut_ref() as *mut SchedHandle;
                *slot = Some(TaskHandle { inner: Green(task, handle) });

                if !decision() {
                    match slot.take_unwrap().inner {
                        Green(task, _) => sched.enqueue_blocked_task(task),
                        Native(..) => unreachable!()
                    }
                    false
                } else {
                    true
                }
            }
            NativeBlock(shared) => {
                *slot = Some(TaskHandle {
                    inner: Native(match shared {
                        Some(arc) => ArcWakeup(arc),
                        None => LocalWakeup(&mut data.lock as *mut Mutex),
                    })
                });

                if !decision() {
                    *slot = None;
                    false
                } else {
                    true
                }
            }
        }
    }
}

// Agnostic method of forcing a yield of the current task
pub fn yield_now() {
    if rt::in_green_task_context() {
        let sched: ~Scheduler = Local::take();
        sched.yield_now();
    } else {
        Thread::yield_now();
    }
}

// Agnostic method of "maybe yielding" in order to provide fairness
pub fn maybe_yield() {
    if rt::in_green_task_context() {
        let sched: ~Scheduler = Local::take();
        sched.maybe_yield();
    } else {
        // the OS decides fairness, nothing for us to do.
    }
}
