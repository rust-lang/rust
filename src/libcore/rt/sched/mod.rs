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

#[cfg(test)] use super::uvio::UvEventLoop;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use int;

mod local;

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
    priv current_task: Option<~Task>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    priv cleanup_job: Option<CleanupJob>
}

// XXX: Some hacks to put a &fn in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = sys::Closure;
trait HackAroundBorrowCk {
    fn from_fn(&fn(&mut Scheduler, ~Task)) -> Self;
    fn to_fn(self) -> &fn(&mut Scheduler, ~Task);
}
impl HackAroundBorrowCk for UnsafeTaskReceiver {
    fn from_fn(f: &fn(&mut Scheduler, ~Task)) -> UnsafeTaskReceiver {
        unsafe { transmute(f) }
    }
    fn to_fn(self) -> &fn(&mut Scheduler, ~Task) {
        unsafe { transmute(self) }
    }
}

enum CleanupJob {
    DoNothing,
    RescheduleTask(~Task),
    RecycleTask(~Task),
    GiveTask(~Task, UnsafeTaskReceiver)
}

pub impl Scheduler {

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
        local::put(self);

        do Scheduler::local |scheduler| {
            fn run_scheduler_once() {
                do Scheduler::local |scheduler| {
                    if scheduler.resume_task_from_queue() {
                        // Ok, a task ran. Nice! We'll do it again later
                        scheduler.event_loop.callback(run_scheduler_once);
                    }
                }
            }

            scheduler.event_loop.callback(run_scheduler_once);
            scheduler.event_loop.run();
        }

        return local::take();
    }

    fn local(f: &fn(&mut Scheduler)) {
        unsafe { local::borrow(f) }
    }

    // * Scheduler-context operations

    fn resume_task_from_queue(&mut self) -> bool {
        assert!(!self.in_task_context());

        let mut self = self;
        match self.task_queue.pop_front() {
            Some(task) => {
                self.resume_task_immediately(task);
                return true;
            }
            None => {
                rtdebug!("no tasks in queue");
                return false;
            }
        }
    }

    fn resume_task_immediately(&mut self, task: ~Task) {
        assert!(!self.in_task_context());

        rtdebug!("scheduling a task");

        // Store the task in the scheduler so it can be grabbed later
        self.current_task = Some(task);
        self.enqueue_cleanup_job(DoNothing);

        // Take pointers to both the task and scheduler's saved registers.
        {
            let (sched_context, _, next_task_context) = self.get_contexts();
            let next_task_context = next_task_context.unwrap();
            // Context switch to the task, restoring it's registers
            // and saving the scheduler's
            Context::swap(sched_context, next_task_context);
        }

        // The running task should have passed ownership elsewhere
        assert!(self.current_task.is_none());

        // Running tasks may have asked us to do some cleanup
        self.run_cleanup_job();
    }


    // * Task-context operations

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    fn terminate_current_task(&mut self) {
        assert!(self.in_task_context());

        rtdebug!("ending running task");

        let dead_task = self.current_task.swap_unwrap();
        self.enqueue_cleanup_job(RecycleTask(dead_task));
        {
            let (sched_context, last_task_context, _) = self.get_contexts();
            let last_task_context = last_task_context.unwrap();
            Context::swap(last_task_context, sched_context);
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
    fn deschedule_running_task_and_then(&mut self, f: &fn(&mut Scheduler, ~Task)) {
        assert!(self.in_task_context());

        rtdebug!("blocking task");

        let blocked_task = self.current_task.swap_unwrap();
        let f_fake_region = unsafe {
            transmute::<&fn(&mut Scheduler, ~Task), &fn(&mut Scheduler, ~Task)>(f)
        };
        let f_opaque = HackAroundBorrowCk::from_fn(f_fake_region);
        self.enqueue_cleanup_job(GiveTask(blocked_task, f_opaque));
        {
            let (sched_context, last_task_context, _) = self.get_contexts();
            let last_task_context = last_task_context.unwrap();
            Context::swap(last_task_context, sched_context);
        }

        // We could be executing in a different thread now
        do Scheduler::local |sched| {
            sched.run_cleanup_job();
        }
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this, e.g. if there are
    /// pending I/O events it would be a bad idea.
    fn resume_task_from_running_task_direct(&mut self, next_task: ~Task) {
        assert!(self.in_task_context());

        rtdebug!("switching tasks");

        let old_running_task = self.current_task.swap_unwrap();
        self.enqueue_cleanup_job(RescheduleTask(old_running_task));
        self.current_task = Some(next_task);
        {
            let (_, last_task_context, next_task_context) = self.get_contexts();
            let last_task_context = last_task_context.unwrap();
            let next_task_context = next_task_context.unwrap();
            Context::swap(last_task_context, next_task_context);
        }

        // We could be executing in a different thread now
        do Scheduler::local |sched| {
            sched.run_cleanup_job();
        }
    }

    // * Other stuff

    fn in_task_context(&self) -> bool { self.current_task.is_some() }

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
            RescheduleTask(task) => {
                // NB: Pushing to the *front* of the queue
                self.task_queue.push_front(task);
            }
            RecycleTask(task) => task.recycle(&mut self.stack_pool),
            GiveTask(task, f) => (f.to_fn())(self, task)
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
    fn get_contexts(&mut self) -> (&'self mut Context,
                                   Option<&'self mut Context>,
                                   Option<&'self mut Context>) {
        let last_task = match self.cleanup_job {
            Some(RescheduleTask(~ref task)) |
            Some(RecycleTask(~ref task)) |
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
        let last_task = unsafe { transmute::<Option<&Task>, Option<&mut Task>>(last_task) };
        let last_task_context = match last_task {
            Some(ref t) => Some(&mut t.saved_context), None => None
        };
        let next_task_context = match self.current_task {
            Some(ref mut t) => Some(&mut t.saved_context), None => None
        };
        return (&mut self.saved_context,
                last_task_context,
                next_task_context);
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
}

pub impl Task {
    fn new(stack_pool: &mut StackPool, start: ~fn()) -> Task {
        let start = Task::build_start_wrapper(start);
        let mut stack = stack_pool.take_segment(TASK_MIN_STACK_SIZE);
        // NB: Context holds a pointer to that ~fn
        let initial_context = Context::new(start, &mut stack);
        return Task {
            current_stack_segment: stack,
            saved_context: initial_context,
        };
    }

    priv fn build_start_wrapper(start: ~fn()) -> ~fn() {
        // XXX: The old code didn't have this extra allocation
        let wrapper: ~fn() = || {
            // This is the first code to execute after the initial
            // context switch to the task. The previous context may
            // have asked us to do some cleanup.
            do Scheduler::local |sched| {
                sched.run_cleanup_job();
            }

            start();

            do Scheduler::local |sched| {
                sched.terminate_current_task();
            }
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
fn test_swap_tasks() {
    do run_in_bare_thread {
        let mut count = 0;
        let count_ptr: *mut int = &mut count;

        let mut sched = ~UvEventLoop::new_scheduler();
        let task1 = ~do Task::new(&mut sched.stack_pool) {
            unsafe { *count_ptr = *count_ptr + 1; }
            do Scheduler::local |sched| {
                let task2 = ~do Task::new(&mut sched.stack_pool) {
                    unsafe { *count_ptr = *count_ptr + 1; }
                };
                // Context switch directly to the new task
                sched.resume_task_from_running_task_direct(task2);
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
            do Scheduler::local |sched| {
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

#[bench] #[test] #[ignore(reason = "too much stack allocation")]
fn test_run_a_lot_of_tasks_direct() {
    do run_in_bare_thread {
        static MAX: int = 100000;
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
            do Scheduler::local |sched| {
                let task = ~do Task::new(&mut sched.stack_pool) {
                    unsafe {
                        *count_ptr = *count_ptr + 1;
                        if *count_ptr != MAX {
                            run_task(count_ptr);
                        }
                    }
                };
                // Context switch directly to the new task
                sched.resume_task_from_running_task_direct(task);
            }
        };
    }
}

#[test]
fn test_block_task() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            do Scheduler::local |sched| {
                assert!(sched.in_task_context());
                do sched.deschedule_running_task_and_then() |sched, task| {
                    assert!(!sched.in_task_context());
                    sched.task_queue.push_back(task);
                }
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}
