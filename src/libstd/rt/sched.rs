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
use cell::Cell;
use clone::Clone;

use super::sleeper_list::SleeperList;
use super::work_queue::WorkQueue;
use super::stack::{StackPool, StackSegment};
use super::rtio::{EventLoop, EventLoopObject, RemoteCallbackObject};
use super::context::Context;
use super::task::Task;
use super::message_queue::MessageQueue;
use rt::local_ptr;
use rt::local::Local;
use rt::rtio::RemoteCallback;
use rt::metrics::SchedMetrics;

/// The Scheduler is responsible for coordinating execution of Coroutines
/// on a single thread. When the scheduler is running it is owned by
/// thread local storage and the running task is owned by the
/// scheduler.
///
/// XXX: This creates too many callbacks to run_sched_once, resulting
/// in too much allocation and too many events.
pub struct Scheduler {
    /// A queue of available work. Under a work-stealing policy there
    /// is one per Scheduler.
    priv work_queue: WorkQueue<~Coroutine>,
    /// The queue of incoming messages from other schedulers.
    /// These are enqueued by SchedHandles after which a remote callback
    /// is triggered to handle the message.
    priv message_queue: MessageQueue<SchedMessage>,
    /// A shared list of sleeping schedulers. We'll use this to wake
    /// up schedulers when pushing work onto the work queue.
    priv sleeper_list: SleeperList,
    /// Indicates that we have previously pushed a handle onto the
    /// SleeperList but have not yet received the Wake message.
    /// Being `true` does not necessarily mean that the scheduler is
    /// not active since there are multiple event sources that may
    /// wake the scheduler. It just prevents the scheduler from pushing
    /// multiple handles onto the sleeper list.
    priv sleepy: bool,
    /// A flag to indicate we've received the shutdown message and should
    /// no longer try to go to sleep, but exit instead.
    no_sleep: bool,
    stack_pool: StackPool,
    /// The event loop used to drive the scheduler and perform I/O
    event_loop: ~EventLoopObject,
    /// The scheduler's saved context.
    /// Always valid when a task is executing, otherwise not
    priv saved_context: Context,
    /// The currently executing task
    current_task: Option<~Coroutine>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    priv cleanup_job: Option<CleanupJob>,
    metrics: SchedMetrics
}

pub struct SchedHandle {
    priv remote: ~RemoteCallbackObject,
    priv queue: MessageQueue<SchedMessage>
}

pub struct Coroutine {
    /// The segment of stack on which the task is currently running or,
    /// if the task is blocked, on which the task will resume execution
    priv current_stack_segment: StackSegment,
    /// These are always valid when the task is not running, unless
    /// the task is dead
    priv saved_context: Context,
    /// The heap, GC, unwinding, local storage, logging
    task: ~Task
}

pub enum SchedMessage {
    Wake,
    Shutdown
}

enum CleanupJob {
    DoNothing,
    GiveTask(~Coroutine, UnsafeTaskReceiver)
}

pub impl Scheduler {

    fn in_task_context(&self) -> bool { self.current_task.is_some() }

    fn new(event_loop: ~EventLoopObject,
           work_queue: WorkQueue<~Coroutine>,
           sleeper_list: SleeperList)
        -> Scheduler {

        // Lazily initialize the runtime TLS key
        local_ptr::init_tls_key();

        Scheduler {
            sleeper_list: sleeper_list,
            message_queue: MessageQueue::new(),
            sleepy: false,
            no_sleep: false,
            event_loop: event_loop,
            work_queue: work_queue,
            stack_pool: StackPool::new(),
            saved_context: Context::empty(),
            current_task: None,
            cleanup_job: None,
            metrics: SchedMetrics::new()
        }
    }

    // XXX: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.
    fn run(~self) -> ~Scheduler {
        assert!(!self.in_task_context());

        let mut self_sched = self;

        // Always run through the scheduler loop at least once so that
        // we enter the sleep state and can then be woken up by other
        // schedulers.
        self_sched.event_loop.callback(Scheduler::run_sched_once);

        unsafe {
            let event_loop: *mut ~EventLoopObject = {
                let event_loop: *mut ~EventLoopObject = &mut self_sched.event_loop;
                event_loop
            };

            // Give ownership of the scheduler (self) to the thread
            Local::put(self_sched);

            (*event_loop).run();
        }

        let sched = Local::take::<Scheduler>();
        // XXX: Reenable this once we're using a per-task queue. With a shared
        // queue this is not true
        //assert!(sched.work_queue.is_empty());
        rtdebug!("scheduler metrics: %s\n", sched.metrics.to_str());
        return sched;
    }

    fn run_sched_once() {

        let mut sched = Local::take::<Scheduler>();
        sched.metrics.turns += 1;

        // First, check the message queue for instructions.
        // XXX: perf. Check for messages without atomics.
        // It's ok if we miss messages occasionally, as long as
        // we sync and check again before sleeping.
        if sched.interpret_message_queue() {
            // We performed a scheduling action. There may be other work
            // to do yet, so let's try again later.
            let mut sched = Local::take::<Scheduler>();
            sched.metrics.messages_received += 1;
            sched.event_loop.callback(Scheduler::run_sched_once);
            Local::put(sched);
            return;
        }

        // Now, look in the work queue for tasks to run
        let sched = Local::take::<Scheduler>();
        if sched.resume_task_from_queue() {
            // We performed a scheduling action. There may be other work
            // to do yet, so let's try again later.
            let mut sched = Local::take::<Scheduler>();
            sched.metrics.tasks_resumed_from_queue += 1;
            sched.event_loop.callback(Scheduler::run_sched_once);
            Local::put(sched);
            return;
        }

        // If we got here then there was no work to do.
        // Generate a SchedHandle and push it to the sleeper list so
        // somebody can wake us up later.
        rtdebug!("no work to do");
        let mut sched = Local::take::<Scheduler>();
        sched.metrics.wasted_turns += 1;
        if !sched.sleepy && !sched.no_sleep {
            rtdebug!("sleeping");
            sched.metrics.sleepy_times += 1;
            sched.sleepy = true;
            let handle = sched.make_handle();
            sched.sleeper_list.push(handle);
        } else {
            rtdebug!("not sleeping");
        }
        Local::put(sched);
    }

    fn make_handle(&mut self) -> SchedHandle {
        let remote = self.event_loop.remote_callback(Scheduler::run_sched_once);

        return SchedHandle {
            remote: remote,
            queue: self.message_queue.clone()
        };
    }

    /// Schedule a task to be executed later.
    ///
    /// Pushes the task onto the work stealing queue and tells the event loop
    /// to run it later. Always use this instead of pushing to the work queue
    /// directly.
    fn enqueue_task(&mut self, task: ~Coroutine) {
        self.work_queue.push(task);
        self.event_loop.callback(Scheduler::run_sched_once);

        // We've made work available. Notify a sleeping scheduler.
        // XXX: perf. Check for a sleeper without synchronizing memory.
        // It's not critical that we always find it.
        // XXX: perf. If there's a sleeper then we might as well just send
        // it the task directly instead of pushing it to the
        // queue. That is essentially the intent here and it is less
        // work.
        match self.sleeper_list.pop() {
            Some(handle) => {
                let mut handle = handle;
                handle.send(Wake)
            }
            None => (/* pass */)
        }
    }

    // * Scheduler-context operations

    fn interpret_message_queue(~self) -> bool {
        assert!(!self.in_task_context());

        rtdebug!("looking for scheduler messages");

        let mut this = self;
        match this.message_queue.pop() {
            Some(Wake) => {
                rtdebug!("recv Wake message");
                this.sleepy = false;
                Local::put(this);
                return true;
            }
            Some(Shutdown) => {
                rtdebug!("recv Shutdown message");
                if this.sleepy {
                    // There may be an outstanding handle on the sleeper list.
                    // Pop them all to make sure that's not the case.
                    loop {
                        match this.sleeper_list.pop() {
                            Some(handle) => {
                                let mut handle = handle;
                                handle.send(Wake);
                            }
                            None => break
                        }
                    }
                }
                // No more sleeping. After there are no outstanding event loop
                // references we will shut down.
                this.no_sleep = true;
                this.sleepy = false;
                Local::put(this);
                return true;
            }
            None => {
                Local::put(this);
                return false;
            }
        }
    }

    fn resume_task_from_queue(~self) -> bool {
        assert!(!self.in_task_context());

        rtdebug!("looking in work queue for task to schedule");

        let mut this = self;
        match this.work_queue.pop() {
            Some(task) => {
                rtdebug!("resuming task from work queue");
                this.resume_task_immediately(task);
                return true;
            }
            None => {
                rtdebug!("no tasks in queue");
                Local::put(this);
                return false;
            }
        }
    }

    // * Task-context operations

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    fn terminate_current_task(~self) {
        assert!(self.in_task_context());

        rtdebug!("ending running task");

        do self.deschedule_running_task_and_then |sched, dead_task| {
            let dead_task = Cell(dead_task);
            dead_task.take().recycle(&mut sched.stack_pool);
        }

        abort!("control reached end of task");
    }

    fn schedule_new_task(~self, task: ~Coroutine) {
        assert!(self.in_task_context());

        do self.switch_running_tasks_and_then(task) |sched, last_task| {
            let last_task = Cell(last_task);
            sched.enqueue_task(last_task.take());
        }
    }

    fn schedule_task(~self, task: ~Coroutine) {
        assert!(self.in_task_context());

        do self.switch_running_tasks_and_then(task) |sched, last_task| {
            let last_task = Cell(last_task);
            sched.enqueue_task(last_task.take());
        }
    }

    // Core scheduling ops

    fn resume_task_immediately(~self, task: ~Coroutine) {
        let mut this = self;
        assert!(!this.in_task_context());

        rtdebug!("scheduling a task");
        this.metrics.context_switches_sched_to_task += 1;

        // Store the task in the scheduler so it can be grabbed later
        this.current_task = Some(task);
        this.enqueue_cleanup_job(DoNothing);

        Local::put(this);

        // Take pointers to both the task and scheduler's saved registers.
        unsafe {
            let sched = Local::unsafe_borrow::<Scheduler>();
            let (sched_context, _, next_task_context) = (*sched).get_contexts();
            let next_task_context = next_task_context.unwrap();
            // Context switch to the task, restoring it's registers
            // and saving the scheduler's
            Context::swap(sched_context, next_task_context);

            let sched = Local::unsafe_borrow::<Scheduler>();
            // The running task should have passed ownership elsewhere
            assert!((*sched).current_task.is_none());

            // Running tasks may have asked us to do some cleanup
            (*sched).run_cleanup_job();
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
    ///
    /// This passes a Scheduler pointer to the fn after the context switch
    /// in order to prevent that fn from performing further scheduling operations.
    /// Doing further scheduling could easily result in infinite recursion.
    fn deschedule_running_task_and_then(~self, f: &fn(&mut Scheduler, ~Coroutine)) {
        let mut this = self;
        assert!(this.in_task_context());

        rtdebug!("blocking task");
        this.metrics.context_switches_task_to_sched += 1;

        unsafe {
            let blocked_task = this.current_task.swap_unwrap();
            let f_fake_region = transmute::<&fn(&mut Scheduler, ~Coroutine),
                                            &fn(&mut Scheduler, ~Coroutine)>(f);
            let f_opaque = ClosureConverter::from_fn(f_fake_region);
            this.enqueue_cleanup_job(GiveTask(blocked_task, f_opaque));
        }

        Local::put(this);

        unsafe {
            let sched = Local::unsafe_borrow::<Scheduler>();
            let (sched_context, last_task_context, _) = (*sched).get_contexts();
            let last_task_context = last_task_context.unwrap();
            Context::swap(last_task_context, sched_context);

            // We could be executing in a different thread now
            let sched = Local::unsafe_borrow::<Scheduler>();
            (*sched).run_cleanup_job();
        }
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this, e.g. if there are
    /// pending I/O events it would be a bad idea.
    fn switch_running_tasks_and_then(~self, next_task: ~Coroutine,
                                     f: &fn(&mut Scheduler, ~Coroutine)) {
        let mut this = self;
        assert!(this.in_task_context());

        rtdebug!("switching tasks");
        this.metrics.context_switches_task_to_task += 1;

        let old_running_task = this.current_task.swap_unwrap();
        let f_fake_region = unsafe {
            transmute::<&fn(&mut Scheduler, ~Coroutine),
                        &fn(&mut Scheduler, ~Coroutine)>(f)
        };
        let f_opaque = ClosureConverter::from_fn(f_fake_region);
        this.enqueue_cleanup_job(GiveTask(old_running_task, f_opaque));
        this.current_task = Some(next_task);

        Local::put(this);

        unsafe {
            let sched = Local::unsafe_borrow::<Scheduler>();
            let (_, last_task_context, next_task_context) = (*sched).get_contexts();
            let last_task_context = last_task_context.unwrap();
            let next_task_context = next_task_context.unwrap();
            Context::swap(last_task_context, next_task_context);

            // We could be executing in a different thread now
            let sched = Local::unsafe_borrow::<Scheduler>();
            (*sched).run_cleanup_job();
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
            None => fail!("all context switches should have a cleanup job")
        };
        // XXX: Pattern matching mutable pointers above doesn't work
        // because borrowck thinks the three patterns are conflicting
        // borrows
        unsafe {
            let last_task = transmute::<Option<&Coroutine>, Option<&mut Coroutine>>(last_task);
            let last_task_context = match last_task {
                Some(t) => Some(&mut t.saved_context), None => None
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

impl SchedHandle {
    pub fn send(&mut self, msg: SchedMessage) {
        self.queue.push(msg);
        self.remote.fire();
    }
}

pub impl Coroutine {
    fn new(stack_pool: &mut StackPool, start: ~fn()) -> Coroutine {
        Coroutine::with_task(stack_pool, ~Task::new(), start)
    }

    fn with_task(stack_pool: &mut StackPool,
                  task: ~Task,
                  start: ~fn()) -> Coroutine {

        static MIN_STACK_SIZE: uint = 10000000; // XXX: Too much stack

        let start = Coroutine::build_start_wrapper(start);
        let mut stack = stack_pool.take_segment(MIN_STACK_SIZE);
        // NB: Context holds a pointer to that ~fn
        let initial_context = Context::new(start, &mut stack);
        return Coroutine {
            current_stack_segment: stack,
            saved_context: initial_context,
            task: task
        };
    }

    priv fn build_start_wrapper(start: ~fn()) -> ~fn() {
        // XXX: The old code didn't have this extra allocation
        let start_cell = Cell(start);
        let wrapper: ~fn() = || {
            // This is the first code to execute after the initial
            // context switch to the task. The previous context may
            // have asked us to do some cleanup.
            unsafe {
                let sched = Local::unsafe_borrow::<Scheduler>();
                (*sched).run_cleanup_job();

                let sched = Local::unsafe_borrow::<Scheduler>();
                let task = (*sched).current_task.get_mut_ref();
                // FIXME #6141: shouldn't neet to put `start()` in another closure
                let start_cell = Cell(start_cell.take());
                do task.task.run {
                    // N.B. Removing `start` from the start wrapper closure
                    // by emptying a cell is critical for correctness. The ~Task
                    // pointer, and in turn the closure used to initialize the first
                    // call frame, is destroyed in scheduler context, not task context.
                    // So any captured closures must not contain user-definable dtors
                    // that expect to be in task context. By moving `start` out of
                    // the closure, all the user code goes out of scope while
                    // the task is still running.
                    let start = start_cell.take();
                    start();
                };
            }

            let sched = Local::take::<Scheduler>();
            sched.terminate_current_task();
        };
        return wrapper;
    }

    /// Destroy the task and try to reuse its components
    fn recycle(~self, stack_pool: &mut StackPool) {
        match self {
            ~Coroutine {current_stack_segment, _} => {
                stack_pool.give_segment(current_stack_segment);
            }
        }
    }
}

// XXX: Some hacks to put a &fn in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = sys::Closure;
trait ClosureConverter {
    fn from_fn(&fn(&mut Scheduler, ~Coroutine)) -> Self;
    fn to_fn(self) -> &fn(&mut Scheduler, ~Coroutine);
}
impl ClosureConverter for UnsafeTaskReceiver {
    fn from_fn(f: &fn(&mut Scheduler, ~Coroutine)) -> UnsafeTaskReceiver { unsafe { transmute(f) } }
    fn to_fn(self) -> &fn(&mut Scheduler, ~Coroutine) { unsafe { transmute(self) } }
}

#[cfg(test)]
mod test {
    use int;
    use cell::Cell;
    use unstable::run_in_bare_thread;
    use task::spawn;
    use rt::local::Local;
    use rt::test::*;
    use super::*;
    use rt::thread::Thread;

    #[test]
    fn test_simple_scheduling() {
        do run_in_bare_thread {
            let mut task_ran = false;
            let task_ran_ptr: *mut bool = &mut task_ran;

            let mut sched = ~new_test_uv_sched();
            let task = ~do Coroutine::new(&mut sched.stack_pool) {
                unsafe { *task_ran_ptr = true; }
            };
            sched.enqueue_task(task);
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

            let mut sched = ~new_test_uv_sched();
            for int::range(0, total) |_| {
                let task = ~do Coroutine::new(&mut sched.stack_pool) {
                    unsafe { *task_count_ptr = *task_count_ptr + 1; }
                };
                sched.enqueue_task(task);
            }
            sched.run();
            assert_eq!(task_count, total);
        }
    }

    #[test]
    fn test_swap_tasks_then() {
        do run_in_bare_thread {
            let mut count = 0;
            let count_ptr: *mut int = &mut count;

            let mut sched = ~new_test_uv_sched();
            let task1 = ~do Coroutine::new(&mut sched.stack_pool) {
                unsafe { *count_ptr = *count_ptr + 1; }
                let mut sched = Local::take::<Scheduler>();
                let task2 = ~do Coroutine::new(&mut sched.stack_pool) {
                    unsafe { *count_ptr = *count_ptr + 1; }
                };
                // Context switch directly to the new task
                do sched.switch_running_tasks_and_then(task2) |sched, task1| {
                    let task1 = Cell(task1);
                    sched.enqueue_task(task1.take());
                }
                unsafe { *count_ptr = *count_ptr + 1; }
            };
            sched.enqueue_task(task1);
            sched.run();
            assert_eq!(count, 3);
        }
    }

    #[bench] #[test] #[ignore(reason = "long test")]
    fn test_run_a_lot_of_tasks_queued() {
        do run_in_bare_thread {
            static MAX: int = 1000000;
            let mut count = 0;
            let count_ptr: *mut int = &mut count;

            let mut sched = ~new_test_uv_sched();

            let start_task = ~do Coroutine::new(&mut sched.stack_pool) {
                run_task(count_ptr);
            };
            sched.enqueue_task(start_task);
            sched.run();

            assert_eq!(count, MAX);

            fn run_task(count_ptr: *mut int) {
                do Local::borrow::<Scheduler, ()> |sched| {
                    let task = ~do Coroutine::new(&mut sched.stack_pool) {
                        unsafe {
                            *count_ptr = *count_ptr + 1;
                            if *count_ptr != MAX {
                                run_task(count_ptr);
                            }
                        }
                    };
                    sched.enqueue_task(task);
                }
            };
        }
    }

    #[test]
    fn test_block_task() {
        do run_in_bare_thread {
            let mut sched = ~new_test_uv_sched();
            let task = ~do Coroutine::new(&mut sched.stack_pool) {
                let sched = Local::take::<Scheduler>();
                assert!(sched.in_task_context());
                do sched.deschedule_running_task_and_then() |sched, task| {
                    let task = Cell(task);
                    assert!(!sched.in_task_context());
                    sched.enqueue_task(task.take());
                }
            };
            sched.enqueue_task(task);
            sched.run();
        }
    }

    #[test]
    fn test_io_callback() {
        // This is a regression test that when there are no schedulable tasks
        // in the work queue, but we are performing I/O, that once we do put
        // something in the work queue again the scheduler picks it up and doesn't
        // exit before emptying the work queue
        do run_in_newsched_task {
            do spawn {
                let sched = Local::take::<Scheduler>();
                do sched.deschedule_running_task_and_then |sched, task| {
                    let task = Cell(task);
                    do sched.event_loop.callback_ms(10) {
                        rtdebug!("in callback");
                        let mut sched = Local::take::<Scheduler>();
                        sched.enqueue_task(task.take());
                        Local::put(sched);
                    }
                }
            }
        }
    }

    #[test]
    fn handle() {
        use rt::comm::*;

        do run_in_bare_thread {
            let (port, chan) = oneshot::<()>();
            let port_cell = Cell(port);
            let chan_cell = Cell(chan);
            let mut sched1 = ~new_test_uv_sched();
            let handle1 = sched1.make_handle();
            let handle1_cell = Cell(handle1);
            let task1 = ~do Coroutine::new(&mut sched1.stack_pool) {
                chan_cell.take().send(());
            };
            sched1.enqueue_task(task1);

            let mut sched2 = ~new_test_uv_sched();
            let task2 = ~do Coroutine::new(&mut sched2.stack_pool) {
                port_cell.take().recv();
                // Release the other scheduler's handle so it can exit
                handle1_cell.take();
            };
            sched2.enqueue_task(task2);

            let sched1_cell = Cell(sched1);
            let _thread1 = do Thread::start {
                let sched1 = sched1_cell.take();
                sched1.run();
            };

            let sched2_cell = Cell(sched2);
            let _thread2 = do Thread::start {
                let sched2 = sched2_cell.take();
                sched2.run();
            };
        }
    }

    #[test]
    fn multithreading() {
        use rt::comm::*;
        use iter::Times;
        use vec::OwnedVector;
        use container::Container;

        do run_in_mt_newsched_task {
            let mut ports = ~[];
            for 10.times {
                let (port, chan) = oneshot();
                let chan_cell = Cell(chan);
                do spawntask_later {
                    chan_cell.take().send(());
                }
                ports.push(port);
            }

            while !ports.is_empty() {
                ports.pop().recv();
            }
        }
    }

    #[test]
    fn thread_ring() {
        use rt::comm::*;
        use comm::{GenericPort, GenericChan};

        do run_in_mt_newsched_task {
            let (end_port, end_chan) = oneshot();

            let n_tasks = 10;
            let token = 2000;

            let mut (p, ch1) = stream();
            ch1.send((token, end_chan));
            let mut i = 2;
            while i <= n_tasks {
                let (next_p, ch) = stream();
                let imm_i = i;
                let imm_p = p;
                do spawntask_random {
                    roundtrip(imm_i, n_tasks, &imm_p, &ch);
                };
                p = next_p;
                i += 1;
            }
            let imm_p = p;
            let imm_ch = ch1;
            do spawntask_random {
                roundtrip(1, n_tasks, &imm_p, &imm_ch);
            }

            end_port.recv();
        }

        fn roundtrip(id: int, n_tasks: int,
                     p: &Port<(int, ChanOne<()>)>, ch: &Chan<(int, ChanOne<()>)>) {
            while (true) {
                match p.recv() {
                    (1, end_chan) => {
                        debug!("%d\n", id);
                        end_chan.send(());
                        return;
                    }
                    (token, end_chan) => {
                        debug!("thread: %d   got token: %d", id, token);
                        ch.send((token - 1, end_chan));
                        if token <= n_tasks {
                            return;
                        }
                    }
                }
            }
        }

    }

    #[test]
    fn start_closure_dtor() {
        use ops::Drop;

        // Regression test that the `start` task entrypoint can contain dtors
        // that use task resources
        do run_in_newsched_task {
            struct S { field: () }

            impl Drop for S {
                fn finalize(&self) {
                    let _foo = @0;
                }
            }

            let s = S { field: () };

            do spawntask {
                let _ss = &s;
            }
        }        
    }
}
