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
use libc::c_void;
use ptr::mut_null;

use super::work_queue::WorkQueue;
use super::stack::{StackPool, StackSegment};
use super::rtio::{EventLoop, EventLoopObject};
use super::context::Context;
use tls = super::thread_local_storage;

#[cfg(test)] use super::uvio::UvEventLoop;
#[cfg(test)] use unstable::run_in_bare_thread;
#[cfg(test)] use int;

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
    /// A queue of jobs to perform immediately upon return from task
    /// context to scheduler context.
    /// XXX: This probably should be a single cleanup action and it
    /// should run after a context switch, not on return from the
    /// scheduler
    priv cleanup_jobs: ~[CleanupJob]
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
            cleanup_jobs: ~[]
        }
    }

    // XXX: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.
    fn run(~self) -> ~Scheduler {
        assert!(!self.in_task_context());

        // Give ownership of the scheduler (self) to the thread
        do self.install |scheduler| {
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
    }

    fn install(~self, f: &fn(&mut Scheduler)) -> ~Scheduler {
        let mut tlsched = ThreadLocalScheduler::new();
        tlsched.put_scheduler(self);
        {
            let sched = tlsched.get_scheduler();
            f(sched);
        }
        return tlsched.take_scheduler();
    }

    fn local(f: &fn(&mut Scheduler)) {
        let mut tlsched = ThreadLocalScheduler::new();
        f(tlsched.get_scheduler());
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
        self.swap_in_task();
        // The running task should have passed ownership elsewhere
        assert!(self.current_task.is_none());

        // Running tasks may have asked us to do some cleanup
        self.run_cleanup_jobs();
    }


    // * Task-context operations

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    fn terminate_current_task(&mut self) {
        assert!(self.in_task_context());

        rtdebug!("ending running task");

        let dead_task = self.current_task.swap_unwrap();
        self.enqueue_cleanup_job(RecycleTask(dead_task));
        let dead_task = self.task_from_last_cleanup_job();
        self.swap_out_task(dead_task);
    }

    /// Block a running task, context switch to the scheduler, then pass the
    /// blocked task to a closure.
    ///
    /// # Safety note
    ///
    /// The closure here is a *stack* closure that lives in the
    /// running task.  It gets transmuted to the scheduler's lifetime
    /// and called while the task is blocked.
    fn block_running_task_and_then(&mut self, f: &fn(&mut Scheduler, ~Task)) {
        assert!(self.in_task_context());

        rtdebug!("blocking task");

        let blocked_task = self.current_task.swap_unwrap();
        let f_fake_region = unsafe {
            transmute::<&fn(&mut Scheduler, ~Task), &fn(&mut Scheduler, ~Task)>(f)
        };
        let f_opaque = HackAroundBorrowCk::from_fn(f_fake_region);
        self.enqueue_cleanup_job(GiveTask(blocked_task, f_opaque));
        let blocked_task = self.task_from_last_cleanup_job();

        self.swap_out_task(blocked_task);
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this, e.g. if there are
    /// pending I/O events it would be a bad idea.
    fn resume_task_from_running_task_direct(&mut self, next_task: ~Task) {
        assert!(self.in_task_context());

        rtdebug!("switching tasks");

        let old_running_task = self.current_task.swap_unwrap();
        self.enqueue_cleanup_job(RescheduleTask(old_running_task));
        let old_running_task = self.task_from_last_cleanup_job();

        self.current_task = Some(next_task);
        self.swap_in_task_from_running_task(old_running_task);
    }


    // * Context switching

    // NB: When switching to a task callers are expected to first set
    // self.running_task. When switching away from a task likewise move
    // out of the self.running_task

    priv fn swap_in_task(&mut self) {
        // Take pointers to both the task and scheduler's saved registers.
        let running_task: &~Task = self.current_task.get_ref();
        let task_context = &running_task.saved_context;
        let scheduler_context = &mut self.saved_context;

        // Context switch to the task, restoring it's registers
        // and saving the scheduler's
        Context::swap(scheduler_context, task_context);
    }

    priv fn swap_out_task(&mut self, running_task: &mut Task) {
        let task_context = &mut running_task.saved_context;
        let scheduler_context = &self.saved_context;
        Context::swap(task_context, scheduler_context);
    }

    priv fn swap_in_task_from_running_task(&mut self, running_task: &mut Task) {
        let running_task_context = &mut running_task.saved_context;
        let next_context = &self.current_task.get_ref().saved_context;
        Context::swap(running_task_context, next_context);
    }


    // * Other stuff

    fn in_task_context(&self) -> bool { self.current_task.is_some() }

    fn enqueue_cleanup_job(&mut self, job: CleanupJob) {
        self.cleanup_jobs.unshift(job);
    }

    fn run_cleanup_jobs(&mut self) {
        assert!(!self.in_task_context());
        rtdebug!("running cleanup jobs");

        while !self.cleanup_jobs.is_empty() {
            match self.cleanup_jobs.pop() {
                RescheduleTask(task) => {
                    // NB: Pushing to the *front* of the queue
                    self.task_queue.push_front(task);
                }
                RecycleTask(task) => task.recycle(&mut self.stack_pool),
                GiveTask(task, f) => (f.to_fn())(self, task)
            }
        }
    }

    // XXX: Hack. This should return &'self mut but I don't know how to
    // make the borrowcheck happy
    fn task_from_last_cleanup_job(&mut self) -> &mut Task {
        assert!(!self.cleanup_jobs.is_empty());
        let last_job: &'self mut CleanupJob = &mut self.cleanup_jobs[0];
        let last_task: &'self Task = match last_job {
            &RescheduleTask(~ref task) => task,
            &RecycleTask(~ref task) => task,
            &GiveTask(~ref task, _) => task,
        };
        // XXX: Pattern matching mutable pointers above doesn't work
        // because borrowck thinks the three patterns are conflicting
        // borrows
        return unsafe { transmute::<&Task, &mut Task>(last_task) };
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
            start();

            let mut sched = ThreadLocalScheduler::new();
            let sched = sched.get_scheduler();
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

// NB: This is a type so we can use make use of the &self region.
struct ThreadLocalScheduler(tls::Key);

impl ThreadLocalScheduler {
    fn new() -> ThreadLocalScheduler {
        unsafe {
            // NB: This assumes that the TLS key has been created prior.
            // Currently done in rust_start.
            let key: *mut c_void = rust_get_sched_tls_key();
            let key: &mut tls::Key = transmute(key);
            ThreadLocalScheduler(*key)
        }
    }

    fn put_scheduler(&mut self, scheduler: ~Scheduler) {
        unsafe {
            let key = match self { &ThreadLocalScheduler(key) => key };
            let value: *mut c_void = transmute::<~Scheduler, *mut c_void>(scheduler);
            tls::set(key, value);
        }
    }

    fn get_scheduler(&mut self) -> &'self mut Scheduler {
        unsafe {
            let key = match self { &ThreadLocalScheduler(key) => key };
            let mut value: *mut c_void = tls::get(key);
            assert!(value.is_not_null());
            {
                let value_ptr = &mut value;
                let sched: &mut ~Scheduler = {
                    transmute::<&mut *mut c_void, &mut ~Scheduler>(value_ptr)
                };
                let sched: &mut Scheduler = &mut **sched;
                return sched;
            }
        }
    }

    fn take_scheduler(&mut self) -> ~Scheduler {
        unsafe {
            let key = match self { &ThreadLocalScheduler(key) => key };
            let value: *mut c_void = tls::get(key);
            assert!(value.is_not_null());
            let sched = transmute(value);
            tls::set(key, mut_null());
            return sched;
        }
    }
}

extern {
    fn rust_get_sched_tls_key() -> *mut c_void;
}

#[test]
fn thread_local_scheduler_smoke_test() {
    let scheduler = ~UvEventLoop::new_scheduler();
    let mut tls_scheduler = ThreadLocalScheduler::new();
    tls_scheduler.put_scheduler(scheduler);
    {
        let _scheduler = tls_scheduler.get_scheduler();
    }
    let _scheduler = tls_scheduler.take_scheduler();
}

#[test]
fn thread_local_scheduler_two_instances() {
    let scheduler = ~UvEventLoop::new_scheduler();
    let mut tls_scheduler = ThreadLocalScheduler::new();
    tls_scheduler.put_scheduler(scheduler);
    {

        let _scheduler = tls_scheduler.get_scheduler();
    }
    {
        let scheduler = tls_scheduler.take_scheduler();
        tls_scheduler.put_scheduler(scheduler);
    }

    let mut tls_scheduler = ThreadLocalScheduler::new();
    {
        let _scheduler = tls_scheduler.get_scheduler();
    }
    let _scheduler = tls_scheduler.take_scheduler();
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
                do sched.block_running_task_and_then() |sched, task| {
                    assert!(!sched.in_task_context());
                    sched.task_queue.push_back(task);
                }
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}
