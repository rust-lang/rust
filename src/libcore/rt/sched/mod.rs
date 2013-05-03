// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
use sys;
use cast::transmute;

use super::work_queue::WorkQueue;
use super::stack::{StackPool, StackSegment};
use super::rtio::{EventLoop, EventLoopObject};
use super::context::Context;
use super::local_services::LocalServices;
use cell::Cell;

#[cfg(test)] use super::uvio::UvEventLoop;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use int;

// A more convenient name for external callers, e.g. `local_sched::take()`
pub mod local_sched;

/// The Scheduler is responsible for coordinating execution of Tasks
/// on a single thread. When the scheduler is running it is owned by
/// thread local storage and the running task is owned by the
/// scheduler.
pub struct Scheduler {
    task_queue: WorkQueue<~Task>,
    stack_pool: StackPool,
    /// The event loop used to drive the scheduler and perform I/O
    event_loop: ~EventLoopObject,
    /// The scheduler's saved context.
    /// Always valid when a task is executing, otherwise not
    priv saved_context: Context,
    /// The currently executing task
    current_task: Option<~Task>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    priv cleanup_job: Option<CleanupJob>
}

// XXX: Some hacks to put a &fn in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = sys::Closure;
trait ClosureConverter {
    fn from_fn(&fn(~Task)) -> Self;
    fn to_fn(self) -> &fn(~Task);
}
impl ClosureConverter for UnsafeTaskReceiver {
    fn from_fn(f: &fn(~Task)) -> UnsafeTaskReceiver { unsafe { transmute(f) } }
    fn to_fn(self) -> &fn(~Task) { unsafe { transmute(self) } }
}

enum CleanupJob {
    DoNothing,
    GiveTask(~Task, UnsafeTaskReceiver)
}

pub impl Scheduler {

    fn in_task_context(&self) -> bool { self.current_task.is_some() }

    fn new(event_loop: ~EventLoopObject) -> Scheduler {

        // Lazily initialize the global state, currently the scheduler TLS key
        unsafe { rust_initialize_global_state(); }
        extern {
            fn rust_initialize_global_state();
        }

        Scheduler {
            event_loop: event_loop,
            task_queue: WorkQueue::new(),
            stack_pool: StackPool::new(),
            saved_context: Context::empty(),
            current_task: None,
            cleanup_job: None
        }
    }

    // XXX: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.
    fn run(~self) -> ~Scheduler {
        assert!(!self.in_task_context());

        // Give ownership of the scheduler (self) to the thread
        local_sched::put(self);

        unsafe {
            let scheduler = local_sched::unsafe_borrow();
            fn run_scheduler_once() {
                let scheduler = local_sched::take();
                if scheduler.resume_task_from_queue() {
                    // Ok, a task ran. Nice! We'll do it again later
                    do local_sched::borrow |scheduler| {
                        scheduler.event_loop.callback(run_scheduler_once);
                    }
                }
            }

            scheduler.event_loop.callback(run_scheduler_once);
            scheduler.event_loop.run();
        }

        return local_sched::take();
    }

    // * Scheduler-context operations

    fn resume_task_from_queue(~self) -> bool {
        assert!(!self.in_task_context());

        let mut self = self;
        match self.task_queue.pop_front() {
            Some(task) => {
                self.resume_task_immediately(task);
                return true;
            }
            None => {
                rtdebug!("no tasks in queue");
                local_sched::put(self);
                return false;
            }
        }
    }

    // * Task-context operations

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    fn terminate_current_task(~self) {
        let mut self = self;
        assert!(self.in_task_context());

        rtdebug!("ending running task");

        do self.deschedule_running_task_and_then |dead_task| {
            let dead_task = Cell(dead_task);
            do local_sched::borrow |sched| {
                dead_task.take().recycle(&mut sched.stack_pool);
            }
        }

        abort!("control reached end of task");
    }

    fn schedule_new_task(~self, task: ~Task) {
        let mut self = self;
        assert!(self.in_task_context());

        do self.switch_running_tasks_and_then(task) |last_task| {
            let last_task = Cell(last_task);
            do local_sched::borrow |sched| {
                sched.task_queue.push_front(last_task.take());
            }
        }
    }

    // Core scheduling ops

    fn resume_task_immediately(~self, task: ~Task) {
        let mut self = self;
        assert!(!self.in_task_context());

        rtdebug!("scheduling a task");

        // Store the task in the scheduler so it can be grabbed later
        self.current_task = Some(task);
        self.enqueue_cleanup_job(DoNothing);

        local_sched::put(self);

        // Take pointers to both the task and scheduler's saved registers.
        unsafe {
            let sched = local_sched::unsafe_borrow();
            let (sched_context, _, next_task_context) = sched.get_contexts();
            let next_task_context = next_task_context.unwrap();
            // Context switch to the task, restoring it's registers
            // and saving the scheduler's
            Context::swap(sched_context, next_task_context);

            let sched = local_sched::unsafe_borrow();
            // The running task should have passed ownership elsewhere
            assert!(sched.current_task.is_none());

            // Running tasks may have asked us to do some cleanup
            sched.run_cleanup_job();
        }
    }

    /// Block a running task, context switch to the scheduler, then pass the
    /// blocked task to a closure.
    ///
    /// # Safety note
    ///
    /// The closure here is a *stack* closure that lives in the
    /// running task.  It gets transmuted to the scheduler's lifetime
    /// and called while the task is blocked.
    fn deschedule_running_task_and_then(~self, f: &fn(~Task)) {
        let mut self = self;
        assert!(self.in_task_context());

        rtdebug!("blocking task");

        let blocked_task = self.current_task.swap_unwrap();
        let f_fake_region = unsafe { transmute::<&fn(~Task), &fn(~Task)>(f) };
        let f_opaque = ClosureConverter::from_fn(f_fake_region);
        self.enqueue_cleanup_job(GiveTask(blocked_task, f_opaque));

        local_sched::put(self);

        let sched = unsafe { local_sched::unsafe_borrow() };
        let (sched_context, last_task_context, _) = sched.get_contexts();
        let last_task_context = last_task_context.unwrap();
        Context::swap(last_task_context, sched_context);

        // We could be executing in a different thread now
        let sched = unsafe { local_sched::unsafe_borrow() };
        sched.run_cleanup_job();
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this, e.g. if there are
    /// pending I/O events it would be a bad idea.
    fn switch_running_tasks_and_then(~self, next_task: ~Task, f: &fn(~Task)) {
        let mut self = self;
        assert!(self.in_task_context());

        rtdebug!("switching tasks");

        let old_running_task = self.current_task.swap_unwrap();
        let f_fake_region = unsafe { transmute::<&fn(~Task), &fn(~Task)>(f) };
        let f_opaque = ClosureConverter::from_fn(f_fake_region);
        self.enqueue_cleanup_job(GiveTask(old_running_task, f_opaque));
        self.current_task = Some(next_task);

        local_sched::put(self);

        unsafe {
            let sched = local_sched::unsafe_borrow();
            let (_, last_task_context, next_task_context) = sched.get_contexts();
            let last_task_context = last_task_context.unwrap();
            let next_task_context = next_task_context.unwrap();
            Context::swap(last_task_context, next_task_context);

            // We could be executing in a different thread now
            let sched = local_sched::unsafe_borrow();
            sched.run_cleanup_job();
        }
    }



    // * Other stuff

    fn enqueue_cleanup_job(&mut self, job: CleanupJob) {
        assert!(self.cleanup_job.is_none());
        self.cleanup_job = Some(job);
    }

    fn run_cleanup_job(&mut self) {
        rtdebug!("running cleanup job");

        assert!(self.cleanup_job.is_some());

        let cleanup_job = self.cleanup_job.swap_unwrap();
        match cleanup_job {
            DoNothing => { }
            GiveTask(task, f) => (f.to_fn())(task)
        }
    }

    /// Get mutable references to all the contexts that may be involved in a
    /// context switch.
    ///
    /// Returns (the scheduler context, the optional context of the
    /// task in the cleanup list, the optional context of the task in
    /// the current task slot).  When context switching to a task,
    /// callers should first arrange for that task to be located in the
    /// Scheduler's current_task slot and set up the
    /// post-context-switch cleanup job.
    fn get_contexts<'a>(&'a mut self) -> (&'a mut Context,
                                          Option<&'a mut Context>,
                                          Option<&'a mut Context>) {
        let last_task = match self.cleanup_job {
            Some(GiveTask(~ref task, _)) => {
                Some(task)
            }
            Some(DoNothing) => {
                None
            }
            None => fail!(fmt!("all context switches should have a cleanup job"))
        };
        // XXX: Pattern matching mutable pointers above doesn't work
        // because borrowck thinks the three patterns are conflicting
        // borrows
        unsafe {
            let last_task = transmute::<Option<&Task>, Option<&mut Task>>(last_task);
            let last_task_context = match last_task {
                Some(ref t) => Some(&mut t.saved_context), None => None
            };
            let next_task_context = match self.current_task {
                Some(ref mut t) => Some(&mut t.saved_context), None => None
            };
            // XXX: These transmutes can be removed after snapshot
            return (transmute(&mut self.saved_context),
                    last_task_context,
                    transmute(next_task_context));
        }
    }
}

static TASK_MIN_STACK_SIZE: uint = 10000000; // XXX: Too much stack

pub struct Task {
    /// The segment of stack on which the task is currently running or,
    /// if the task is blocked, on which the task will resume execution
    priv current_stack_segment: StackSegment,
    /// These are always valid when the task is not running, unless
    /// the task is dead
    priv saved_context: Context,
    /// The heap, GC, unwinding, local storage, logging
    local_services: LocalServices
}

pub impl Task {
    fn new(stack_pool: &mut StackPool, start: ~fn()) -> Task {
        Task::with_local(stack_pool, LocalServices::new(), start)
    }

    fn with_local(stack_pool: &mut StackPool,
                  local_services: LocalServices,
                  start: ~fn()) -> Task {
        let start = Task::build_start_wrapper(start);
        let mut stack = stack_pool.take_segment(TASK_MIN_STACK_SIZE);
        // NB: Context holds a pointer to that ~fn
        let initial_context = Context::new(start, &mut stack);
        return Task {
            current_stack_segment: stack,
            saved_context: initial_context,
            local_services: local_services
        };
    }

    priv fn build_start_wrapper(start: ~fn()) -> ~fn() {
        // XXX: The old code didn't have this extra allocation
        let wrapper: ~fn() = || {
            // This is the first code to execute after the initial
            // context switch to the task. The previous context may
            // have asked us to do some cleanup.
            unsafe {
                let sched = local_sched::unsafe_borrow();
                sched.run_cleanup_job();

                let sched = local_sched::unsafe_borrow();
                let task = sched.current_task.get_mut_ref();
                // FIXME #6141: shouldn't neet to put `start()` in another closure
                task.local_services.run(||start());
            }

            let sched = local_sched::take();
            sched.terminate_current_task();
        };
        return wrapper;
    }

    /// Destroy the task and try to reuse its components
    fn recycle(~self, stack_pool: &mut StackPool) {
        match self {
            ~Task {current_stack_segment, _} => {
                stack_pool.give_segment(current_stack_segment);
            }
        }
    }
}

#[test]
fn test_simple_scheduling() {
    do run_in_bare_thread {
        let mut task_ran = false;
        let task_ran_ptr: *mut bool = &mut task_ran;

        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            unsafe { *task_ran_ptr = true; }
        };
        sched.task_queue.push_back(task);
        sched.run();
        assert!(task_ran);
    }
}

#[test]
fn test_several_tasks() {
    do run_in_bare_thread {
        let total = 10;
        let mut task_count = 0;
        let task_count_ptr: *mut int = &mut task_count;

        let mut sched = ~UvEventLoop::new_scheduler();
        for int::range(0, total) |_| {
            let task = ~do Task::new(&mut sched.stack_pool) {
                unsafe { *task_count_ptr = *task_count_ptr + 1; }
            };
            sched.task_queue.push_back(task);
        }
        sched.run();
        assert!(task_count == total);
    }
}

#[test]
fn test_swap_tasks_then() {
    do run_in_bare_thread {
        let mut count = 0;
        let count_ptr: *mut int = &mut count;

        let mut sched = ~UvEventLoop::new_scheduler();
        let task1 = ~do Task::new(&mut sched.stack_pool) {
            unsafe { *count_ptr = *count_ptr + 1; }
            let mut sched = local_sched::take();
            let task2 = ~do Task::new(&mut sched.stack_pool) {
                unsafe { *count_ptr = *count_ptr + 1; }
            };
            // Context switch directly to the new task
            do sched.switch_running_tasks_and_then(task2) |task1| {
                let task1 = Cell(task1);
                do local_sched::borrow |sched| {
                    sched.task_queue.push_front(task1.take());
                }
            }
            unsafe { *count_ptr = *count_ptr + 1; }
        };
        sched.task_queue.push_back(task1);
        sched.run();
        assert!(count == 3);
    }
}

#[bench] #[test] #[ignore(reason = "long test")]
fn test_run_a_lot_of_tasks_queued() {
    do run_in_bare_thread {
        static MAX: int = 1000000;
        let mut count = 0;
        let count_ptr: *mut int = &mut count;

        let mut sched = ~UvEventLoop::new_scheduler();

        let start_task = ~do Task::new(&mut sched.stack_pool) {
            run_task(count_ptr);
        };
        sched.task_queue.push_back(start_task);
        sched.run();

        assert!(count == MAX);

        fn run_task(count_ptr: *mut int) {
            do local_sched::borrow |sched| {
                let task = ~do Task::new(&mut sched.stack_pool) {
                    unsafe {
                        *count_ptr = *count_ptr + 1;
                        if *count_ptr != MAX {
                            run_task(count_ptr);
                        }
                    }
                };
                sched.task_queue.push_back(task);
            }
        };
    }
}

#[test]
fn test_block_task() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            let sched = local_sched::take();
            assert!(sched.in_task_context());
            do sched.deschedule_running_task_and_then() |task| {
                let task = Cell(task);
                do local_sched::borrow |sched| {
                    assert!(!sched.in_task_context());
                    sched.task_queue.push_back(task.take());
                }
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}
