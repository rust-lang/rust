// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::{Option, Some, None};
use cast::{transmute, transmute_mut_region, transmute_mut_unsafe};
use clone::Clone;
use unstable::raw;
use super::sleeper_list::SleeperList;
use super::stack::{StackPool};
use super::rtio::EventLoop;
use super::context::Context;
use super::task::{Task, AnySched, Sched};
use super::message_queue::MessageQueue;
use rt::kill::BlockedTask;
use rt::deque;
use rt::local_ptr;
use rt::local::Local;
use rt::rtio::{RemoteCallback, PausibleIdleCallback, Callback};
use borrow::{to_uint};
use cell::Cell;
use rand::{XorShiftRng, Rng, Rand};
use iter::range;
use unstable::mutex::Mutex;
use vec::{OwnedVector};


/// A scheduler is responsible for coordinating the execution of Tasks
/// on a single thread. The scheduler runs inside a slightly modified
/// Rust Task. When not running this task is stored in the scheduler
/// struct. The scheduler struct acts like a baton, all scheduling
/// actions are transfers of the baton.
///
/// XXX: This creates too many callbacks to run_sched_once, resulting
/// in too much allocation and too many events.
pub struct Scheduler {
    /// There are N work queues, one per scheduler.
    work_queue: deque::Worker<~Task>,
    /// Work queues for the other schedulers. These are created by
    /// cloning the core work queues.
    work_queues: ~[deque::Stealer<~Task>],
    /// The queue of incoming messages from other schedulers.
    /// These are enqueued by SchedHandles after which a remote callback
    /// is triggered to handle the message.
    message_queue: MessageQueue<SchedMessage>,
    /// A shared list of sleeping schedulers. We'll use this to wake
    /// up schedulers when pushing work onto the work queue.
    sleeper_list: SleeperList,
    /// Indicates that we have previously pushed a handle onto the
    /// SleeperList but have not yet received the Wake message.
    /// Being `true` does not necessarily mean that the scheduler is
    /// not active since there are multiple event sources that may
    /// wake the scheduler. It just prevents the scheduler from pushing
    /// multiple handles onto the sleeper list.
    sleepy: bool,
    /// A flag to indicate we've received the shutdown message and should
    /// no longer try to go to sleep, but exit instead.
    no_sleep: bool,
    stack_pool: StackPool,
    /// The scheduler runs on a special task. When it is not running
    /// it is stored here instead of the work queue.
    sched_task: Option<~Task>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    cleanup_job: Option<CleanupJob>,
    /// Should this scheduler run any task, or only pinned tasks?
    run_anything: bool,
    /// If the scheduler shouldn't run some tasks, a friend to send
    /// them to.
    friend_handle: Option<SchedHandle>,
    /// A fast XorShift rng for scheduler use
    rng: XorShiftRng,
    /// A toggleable idle callback
    idle_callback: Option<~PausibleIdleCallback>,
    /// A countdown that starts at a random value and is decremented
    /// every time a yield check is performed. When it hits 0 a task
    /// will yield.
    yield_check_count: uint,
    /// A flag to tell the scheduler loop it needs to do some stealing
    /// in order to introduce randomness as part of a yield
    steal_for_yield: bool,

    // n.b. currently destructors of an object are run in top-to-bottom in order
    //      of field declaration. Due to its nature, the pausible idle callback
    //      must have some sort of handle to the event loop, so it needs to get
    //      destroyed before the event loop itself. For this reason, we destroy
    //      the event loop last to ensure that any unsafe references to it are
    //      destroyed before it's actually destroyed.

    /// The event loop used to drive the scheduler and perform I/O
    event_loop: ~EventLoop,
}

/// An indication of how hard to work on a given operation, the difference
/// mainly being whether memory is synchronized or not
#[deriving(Eq)]
enum EffortLevel {
    DontTryTooHard,
    GiveItYourBest
}

static MAX_YIELD_CHECKS: uint = 200;

fn reset_yield_check(rng: &mut XorShiftRng) -> uint {
    let r: uint = Rand::rand(rng);
    r % MAX_YIELD_CHECKS + 1
}

impl Scheduler {

    // * Initialization Functions

    pub fn new(event_loop: ~EventLoop,
               work_queue: deque::Worker<~Task>,
               work_queues: ~[deque::Stealer<~Task>],
               sleeper_list: SleeperList)
        -> Scheduler {

        Scheduler::new_special(event_loop, work_queue,
                               work_queues,
                               sleeper_list, true, None)

    }

    pub fn new_special(event_loop: ~EventLoop,
                       work_queue: deque::Worker<~Task>,
                       work_queues: ~[deque::Stealer<~Task>],
                       sleeper_list: SleeperList,
                       run_anything: bool,
                       friend: Option<SchedHandle>)
        -> Scheduler {

        let mut sched = Scheduler {
            sleeper_list: sleeper_list,
            message_queue: MessageQueue::new(),
            sleepy: false,
            no_sleep: false,
            event_loop: event_loop,
            work_queue: work_queue,
            work_queues: work_queues,
            stack_pool: StackPool::new(),
            sched_task: None,
            cleanup_job: None,
            run_anything: run_anything,
            friend_handle: friend,
            rng: new_sched_rng(),
            idle_callback: None,
            yield_check_count: 0,
            steal_for_yield: false
        };

        sched.yield_check_count = reset_yield_check(&mut sched.rng);

        return sched;
    }

    // XXX: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.

    // Take a main task to run, and a scheduler to run it in. Create a
    // scheduler task and bootstrap into it.
    pub fn bootstrap(mut ~self, task: ~Task) {

        // Build an Idle callback.
        let cb = ~SchedRunner as ~Callback;
        self.idle_callback = Some(self.event_loop.pausible_idle_callback(cb));

        // Initialize the TLS key.
        local_ptr::init();

        // Create a task for the scheduler with an empty context.
        let sched_task = ~Task::new_sched_task();

        // Now that we have an empty task struct for the scheduler
        // task, put it in TLS.
        Local::put(sched_task);

        // Before starting our first task, make sure the idle callback
        // is active. As we do not start in the sleep state this is
        // important.
        self.idle_callback.get_mut_ref().resume();

        // Now, as far as all the scheduler state is concerned, we are
        // inside the "scheduler" context. So we can act like the
        // scheduler and resume the provided task.
        self.resume_task_immediately(task);

        // Now we are back in the scheduler context, having
        // successfully run the input task. Start by running the
        // scheduler. Grab it out of TLS - performing the scheduler
        // action will have given it away.
        let sched: ~Scheduler = Local::take();

        rtdebug!("starting scheduler {}", sched.sched_id());
        sched.run();

        // Close the idle callback.
        let mut sched: ~Scheduler = Local::take();
        sched.idle_callback.take();
        // Make one go through the loop to run the close callback.
        sched.run();

        // Now that we are done with the scheduler, clean up the
        // scheduler task. Do so by removing it from TLS and manually
        // cleaning up the memory it uses. As we didn't actually call
        // task.run() on the scheduler task we never get through all
        // the cleanup code it runs.
        let mut stask: ~Task = Local::take();

        rtdebug!("stopping scheduler {}", stask.sched.get_ref().sched_id());

        // Should not have any messages
        let message = stask.sched.get_mut_ref().message_queue.pop();
        rtassert!(message.is_none());

        stask.destroyed = true;
    }

    // This does not return a scheduler, as the scheduler is placed
    // inside the task.
    pub fn run(mut ~self) {

        // This is unsafe because we need to place the scheduler, with
        // the event_loop inside, inside our task. But we still need a
        // mutable reference to the event_loop to give it the "run"
        // command.
        unsafe {
            let event_loop: *mut ~EventLoop = &mut self.event_loop;

            // Our scheduler must be in the task before the event loop
            // is started.
            let self_sched = Cell::new(self);
            Local::borrow(|stask: &mut Task| {
                stask.sched = Some(self_sched.take());
            });

            (*event_loop).run();
        }
    }

    // * Execution Functions - Core Loop Logic

    // The model for this function is that you continue through it
    // until you either use the scheduler while performing a schedule
    // action, in which case you give it away and return early, or
    // you reach the end and sleep. In the case that a scheduler
    // action is performed the loop is evented such that this function
    // is called again.
    fn run_sched_once() {

        // When we reach the scheduler context via the event loop we
        // already have a scheduler stored in our local task, so we
        // start off by taking it. This is the only path through the
        // scheduler where we get the scheduler this way.
        let mut sched: ~Scheduler = Local::take();

        // Assume that we need to continue idling unless we reach the
        // end of this function without performing an action.
        sched.idle_callback.get_mut_ref().resume();

        // First we check for scheduler messages, these are higher
        // priority than regular tasks.
        let sched = match sched.interpret_message_queue(DontTryTooHard) {
            Some(sched) => sched,
            None => return
        };

        // This helper will use a randomized work-stealing algorithm
        // to find work.
        let sched = match sched.do_work() {
            Some(sched) => sched,
            None => return
        };

        // Now, before sleeping we need to find out if there really
        // were any messages. Give it your best!
        let mut sched = match sched.interpret_message_queue(GiveItYourBest) {
            Some(sched) => sched,
            None => return
        };

        // If we got here then there was no work to do.
        // Generate a SchedHandle and push it to the sleeper list so
        // somebody can wake us up later.
        if !sched.sleepy && !sched.no_sleep {
            rtdebug!("scheduler has no work to do, going to sleep");
            sched.sleepy = true;
            let handle = sched.make_handle();
            sched.sleeper_list.push(handle);
            // Since we are sleeping, deactivate the idle callback.
            sched.idle_callback.get_mut_ref().pause();
        } else {
            rtdebug!("not sleeping, already doing so or no_sleep set");
            // We may not be sleeping, but we still need to deactivate
            // the idle callback.
            sched.idle_callback.get_mut_ref().pause();
        }

        // Finished a cycle without using the Scheduler. Place it back
        // in TLS.
        Local::put(sched);
    }

    // This function returns None if the scheduler is "used", or it
    // returns the still-available scheduler. At this point all
    // message-handling will count as a turn of work, and as a result
    // return None.
    fn interpret_message_queue(mut ~self, effort: EffortLevel) -> Option<~Scheduler> {

        let msg = if effort == DontTryTooHard {
            // Do a cheap check that may miss messages
            self.message_queue.casual_pop()
        } else {
            self.message_queue.pop()
        };

        match msg {
            Some(PinnedTask(task)) => {
                let mut task = task;
                task.give_home(Sched(self.make_handle()));
                self.resume_task_immediately(task);
                return None;
            }
            Some(TaskFromFriend(task)) => {
                rtdebug!("got a task from a friend. lovely!");
                self.process_task(task, Scheduler::resume_task_immediately_cl);
                return None;
            }
            Some(RunOnce(task)) => {
                // bypass the process_task logic to force running this task once
                // on this home scheduler. This is often used for I/O (homing).
                Scheduler::resume_task_immediately_cl(self, task);
                return None;
            }
            Some(Wake) => {
                self.sleepy = false;
                Local::put(self);
                return None;
            }
            Some(Shutdown) => {
                rtdebug!("shutting down");
                if self.sleepy {
                    // There may be an outstanding handle on the
                    // sleeper list.  Pop them all to make sure that's
                    // not the case.
                    loop {
                        match self.sleeper_list.pop() {
                            Some(handle) => {
                                let mut handle = handle;
                                handle.send(Wake);
                            }
                            None => break
                        }
                    }
                }
                // No more sleeping. After there are no outstanding
                // event loop references we will shut down.
                self.no_sleep = true;
                self.sleepy = false;
                Local::put(self);
                return None;
            }
            None => {
                return Some(self);
            }
        }
    }

    fn do_work(mut ~self) -> Option<~Scheduler> {
        rtdebug!("scheduler calling do work");
        match self.find_work() {
            Some(task) => {
                rtdebug!("found some work! processing the task");
                self.process_task(task, Scheduler::resume_task_immediately_cl);
                return None;
            }
            None => {
                rtdebug!("no work was found, returning the scheduler struct");
                return Some(self);
            }
        }
    }

    // Workstealing: In this iteration of the runtime each scheduler
    // thread has a distinct work queue. When no work is available
    // locally, make a few attempts to steal work from the queues of
    // other scheduler threads. If a few steals fail we end up in the
    // old "no work" path which is fine.

    // First step in the process is to find a task. This function does
    // that by first checking the local queue, and if there is no work
    // there, trying to steal from the remote work queues.
    fn find_work(&mut self) -> Option<~Task> {
        rtdebug!("scheduler looking for work");
        if !self.steal_for_yield {
            match self.work_queue.pop() {
                Some(task) => {
                    rtdebug!("found a task locally");
                    return Some(task)
                }
                None => {
                    rtdebug!("scheduler trying to steal");
                    return self.try_steals();
                }
            }
        } else {
            // During execution of the last task, it performed a 'yield',
            // so we're doing some work stealing in order to introduce some
            // scheduling randomness. Otherwise we would just end up popping
            // that same task again. This is pretty lame and is to work around
            // the problem that work stealing is not designed for 'non-strict'
            // (non-fork-join) task parallelism.
            self.steal_for_yield = false;
            match self.try_steals() {
                Some(task) => {
                    rtdebug!("stole a task after yielding");
                    return Some(task);
                }
                None => {
                    rtdebug!("did not steal a task after yielding");
                    // Back to business
                    return self.find_work();
                }
            }
        }
    }

    // Try stealing from all queues the scheduler knows about. This
    // naive implementation can steal from our own queue or from other
    // special schedulers.
    fn try_steals(&mut self) -> Option<~Task> {
        let work_queues = &mut self.work_queues;
        let len = work_queues.len();
        let start_index = self.rng.gen_range(0, len);
        for index in range(0, len).map(|i| (i + start_index) % len) {
            match work_queues[index].steal() {
                deque::Data(task) => {
                    rtdebug!("found task by stealing");
                    return Some(task)
                }
                _ => ()
            }
        };
        rtdebug!("giving up on stealing");
        return None;
    }

    // * Task Routing Functions - Make sure tasks send up in the right
    // place.

    fn process_task(mut ~self, mut task: ~Task, schedule_fn: SchedulingFn) {
        rtdebug!("processing a task");

        let home = task.take_unwrap_home();
        match home {
            Sched(home_handle) => {
                if home_handle.sched_id != self.sched_id() {
                    rtdebug!("sending task home");
                    task.give_home(Sched(home_handle));
                    Scheduler::send_task_home(task);
                    Local::put(self);
                } else {
                    rtdebug!("running task here");
                    task.give_home(Sched(home_handle));
                    schedule_fn(self, task);
                }
            }
            AnySched if self.run_anything => {
                rtdebug!("running anysched task here");
                task.give_home(AnySched);
                schedule_fn(self, task);
            }
            AnySched => {
                rtdebug!("sending task to friend");
                task.give_home(AnySched);
                self.send_to_friend(task);
                Local::put(self);
            }
        }
    }

    fn send_task_home(task: ~Task) {
        let mut task = task;
        let mut home = task.take_unwrap_home();
        match home {
            Sched(ref mut home_handle) => {
                home_handle.send(PinnedTask(task));
            }
            AnySched => {
                        rtabort!("error: cannot send anysched task home");
            }
        }
    }

    /// Take a non-homed task we aren't allowed to run here and send
    /// it to the designated friend scheduler to execute.
    fn send_to_friend(&mut self, task: ~Task) {
        rtdebug!("sending a task to friend");
        match self.friend_handle {
            Some(ref mut handle) => {
                handle.send(TaskFromFriend(task));
            }
            None => {
                rtabort!("tried to send task to a friend but scheduler has no friends");
            }
        }
    }

    /// Schedule a task to be executed later.
    ///
    /// Pushes the task onto the work stealing queue and tells the
    /// event loop to run it later. Always use this instead of pushing
    /// to the work queue directly.
    pub fn enqueue_task(&mut self, task: ~Task) {

        // We push the task onto our local queue clone.
        self.work_queue.push(task);
        self.idle_callback.get_mut_ref().resume();

        // We've made work available. Notify a
        // sleeping scheduler.

        match self.sleeper_list.casual_pop() {
            Some(handle) => {
                let mut handle = handle;
                handle.send(Wake)
            }
            None => { (/* pass */) }
        };
    }

    /// As enqueue_task, but with the possibility for the blocked task to
    /// already have been killed.
    pub fn enqueue_blocked_task(&mut self, blocked_task: BlockedTask) {
        blocked_task.wake().map(|task| self.enqueue_task(task));
    }

    // * Core Context Switching Functions

    // The primary function for changing contexts. In the current
    // design the scheduler is just a slightly modified GreenTask, so
    // all context swaps are from Task to Task. The only difference
    // between the various cases is where the inputs come from, and
    // what is done with the resulting task. That is specified by the
    // cleanup function f, which takes the scheduler and the
    // old task as inputs.

    pub fn change_task_context(mut ~self,
                               next_task: ~Task,
                               f: |&mut Scheduler, ~Task|) {
        // The current task is grabbed from TLS, not taken as an input.
        // Doing an unsafe_take to avoid writing back a null pointer -
        // We're going to call `put` later to do that.
        let current_task: ~Task = unsafe { Local::unsafe_take() };

        // Check that the task is not in an atomically() section (e.g.,
        // holding a pthread mutex, which could deadlock the scheduler).
        current_task.death.assert_may_sleep();

        // These transmutes do something fishy with a closure.
        let f_fake_region = unsafe {
            transmute::<|&mut Scheduler, ~Task|,
                        |&mut Scheduler, ~Task|>(f)
        };
        let f_opaque = ClosureConverter::from_fn(f_fake_region);

        // The current task is placed inside an enum with the cleanup
        // function. This enum is then placed inside the scheduler.
        self.cleanup_job = Some(CleanupJob::new(current_task, f_opaque));

        // The scheduler is then placed inside the next task.
        let mut next_task = next_task;
        next_task.sched = Some(self);

        // However we still need an internal mutable pointer to the
        // original task. The strategy here was "arrange memory, then
        // get pointers", so we crawl back up the chain using
        // transmute to eliminate borrowck errors.
        unsafe {

            let sched: &mut Scheduler =
                transmute_mut_region(*next_task.sched.get_mut_ref());

            let current_task: &mut Task = match sched.cleanup_job {
                Some(CleanupJob { task: ref task, .. }) => {
                    let task_ptr: *~Task = task;
                    transmute_mut_region(*transmute_mut_unsafe(task_ptr))
                }
                None => {
                    rtabort!("no cleanup job");
                }
            };

            let (current_task_context, next_task_context) =
                Scheduler::get_contexts(current_task, next_task);

            // Done with everything - put the next task in TLS. This
            // works because due to transmute the borrow checker
            // believes that we have no internal pointers to
            // next_task.
            Local::put(next_task);

            // The raw context swap operation. The next action taken
            // will be running the cleanup job from the context of the
            // next task.
            Context::swap(current_task_context, next_task_context);
        }

        // When the context swaps back to this task we immediately
        // run the cleanup job, as expected by the previously called
        // swap_contexts function.
        unsafe {
            let task: *mut Task = Local::unsafe_borrow();
            (*task).sched.get_mut_ref().run_cleanup_job();

            // See the comments in switch_running_tasks_and_then for why a lock
            // is acquired here. This is the resumption points and the "bounce"
            // that it is referring to.
            (*task).nasty_deschedule_lock.lock();
            (*task).nasty_deschedule_lock.unlock();
        }
    }

    // Returns a mutable reference to both contexts involved in this
    // swap. This is unsafe - we are getting mutable internal
    // references to keep even when we don't own the tasks. It looks
    // kinda safe because we are doing transmutes before passing in
    // the arguments.
    pub fn get_contexts<'a>(current_task: &mut Task, next_task: &mut Task) ->
        (&'a mut Context, &'a mut Context) {
        let current_task_context =
            &mut current_task.coroutine.get_mut_ref().saved_context;
        let next_task_context =
                &mut next_task.coroutine.get_mut_ref().saved_context;
        unsafe {
            (transmute_mut_region(current_task_context),
             transmute_mut_region(next_task_context))
        }
    }

    // * Context Swapping Helpers - Here be ugliness!

    pub fn resume_task_immediately(~self, task: ~Task) {
        self.change_task_context(task, |sched, stask| {
            sched.sched_task = Some(stask);
        })
    }

    fn resume_task_immediately_cl(sched: ~Scheduler,
                                  task: ~Task) {
        sched.resume_task_immediately(task)
    }


    pub fn resume_blocked_task_immediately(~self, blocked_task: BlockedTask) {
        match blocked_task.wake() {
            Some(task) => { self.resume_task_immediately(task); }
            None => Local::put(self)
        };
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
    ///
    /// Note that if the closure provided relinquishes ownership of the
    /// BlockedTask, then it is possible for the task to resume execution before
    /// the closure has finished executing. This would naturally introduce a
    /// race if the closure and task shared portions of the environment.
    ///
    /// This situation is currently prevented, or in other words it is
    /// guaranteed that this function will not return before the given closure
    /// has returned.
    pub fn deschedule_running_task_and_then(mut ~self,
                                            f: |&mut Scheduler, BlockedTask|) {
        // Trickier - we need to get the scheduler task out of self
        // and use it as the destination.
        let stask = self.sched_task.take_unwrap();
        // Otherwise this is the same as below.
        self.switch_running_tasks_and_then(stask, f);
    }

    pub fn switch_running_tasks_and_then(~self, next_task: ~Task,
                                         f: |&mut Scheduler, BlockedTask|) {
        // And here comes one of the sad moments in which a lock is used in a
        // core portion of the rust runtime. As always, this is highly
        // undesirable, so there's a good reason behind it.
        //
        // There is an excellent outline of the problem in issue #8132, and it's
        // summarized in that `f` is executed on a sched task, but its
        // environment is on the previous task. If `f` relinquishes ownership of
        // the BlockedTask, then it may introduce a race where `f` is using the
        // environment as well as the code after the 'deschedule' block.
        //
        // The solution we have chosen to adopt for now is to acquire a
        // task-local lock around this block. The resumption of the task in
        // context switching will bounce on the lock, thereby waiting for this
        // block to finish, eliminating the race mentioned above.
        //
        // To actually maintain a handle to the lock, we use an unsafe pointer
        // to it, but we're guaranteed that the task won't exit until we've
        // unlocked the lock so there's no worry of this memory going away.
        self.change_task_context(next_task, |sched, mut task| {
            let lock: *mut Mutex = &mut task.nasty_deschedule_lock;
            unsafe { (*lock).lock() }
            f(sched, BlockedTask::block(task));
            unsafe { (*lock).unlock() }
        })
    }

    fn switch_task(sched: ~Scheduler, task: ~Task) {
        sched.switch_running_tasks_and_then(task, |sched, last_task| {
            sched.enqueue_blocked_task(last_task);
        });
    }

    // * Task Context Helpers

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    pub fn terminate_current_task(mut ~self) {
        // Similar to deschedule running task and then, but cannot go through
        // the task-blocking path. The task is already dying.
        let stask = self.sched_task.take_unwrap();
        self.change_task_context(stask, |sched, mut dead_task| {
            let coroutine = dead_task.coroutine.take_unwrap();
            coroutine.recycle(&mut sched.stack_pool);
        })
    }

    pub fn run_task(task: ~Task) {
        let sched: ~Scheduler = Local::take();
        sched.process_task(task, Scheduler::switch_task);
    }

    pub fn run_task_later(next_task: ~Task) {
        let next_task = Cell::new(next_task);
        Local::borrow(|sched: &mut Scheduler| {
            sched.enqueue_task(next_task.take());
        });
    }

    /// Yield control to the scheduler, executing another task. This is guaranteed
    /// to introduce some amount of randomness to the scheduler. Currently the
    /// randomness is a result of performing a round of work stealing (which
    /// may end up stealing from the current scheduler).
    pub fn yield_now(mut ~self) {
        self.yield_check_count = reset_yield_check(&mut self.rng);
        // Tell the scheduler to start stealing on the next iteration
        self.steal_for_yield = true;
        self.deschedule_running_task_and_then(|sched, task| {
            sched.enqueue_blocked_task(task);
        })
    }

    pub fn maybe_yield(mut ~self) {
        // The number of times to do the yield check before yielding, chosen arbitrarily.
        rtassert!(self.yield_check_count > 0);
        self.yield_check_count -= 1;
        if self.yield_check_count == 0 {
            self.yield_now();
        } else {
            Local::put(self);
        }
    }


    // * Utility Functions

    pub fn sched_id(&self) -> uint { to_uint(self) }

    pub fn run_cleanup_job(&mut self) {
        let cleanup_job = self.cleanup_job.take_unwrap();
        cleanup_job.run(self);
    }

    pub fn make_handle(&mut self) -> SchedHandle {
        let remote = self.event_loop.remote_callback(~SchedRunner as ~Callback);

        return SchedHandle {
            remote: remote,
            queue: self.message_queue.clone(),
            sched_id: self.sched_id()
        };
    }
}

// Supporting types

type SchedulingFn = extern "Rust" fn (~Scheduler, ~Task);

pub enum SchedMessage {
    Wake,
    Shutdown,
    PinnedTask(~Task),
    TaskFromFriend(~Task),
    RunOnce(~Task),
}

pub struct SchedHandle {
    priv remote: ~RemoteCallback,
    priv queue: MessageQueue<SchedMessage>,
    sched_id: uint
}

impl SchedHandle {
    pub fn send(&mut self, msg: SchedMessage) {
        self.queue.push(msg);
        self.remote.fire();
    }
}

struct SchedRunner;

impl Callback for SchedRunner {
    fn call(&mut self) {
        Scheduler::run_sched_once();
    }
}

struct CleanupJob {
    task: ~Task,
    f: UnsafeTaskReceiver
}

impl CleanupJob {
    pub fn new(task: ~Task, f: UnsafeTaskReceiver) -> CleanupJob {
        CleanupJob {
            task: task,
            f: f
        }
    }

    pub fn run(self, sched: &mut Scheduler) {
        let CleanupJob { task: task, f: f } = self;
        f.to_fn()(sched, task)
    }
}

// XXX: Some hacks to put a || closure in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = raw::Closure;
trait ClosureConverter {
    fn from_fn(|&mut Scheduler, ~Task|) -> Self;
    fn to_fn(self) -> |&mut Scheduler, ~Task|;
}
impl ClosureConverter for UnsafeTaskReceiver {
    fn from_fn(f: |&mut Scheduler, ~Task|) -> UnsafeTaskReceiver {
        unsafe { transmute(f) }
    }
    fn to_fn(self) -> |&mut Scheduler, ~Task| { unsafe { transmute(self) } }
}

// On unix, we read randomness straight from /dev/urandom, but the
// default constructor of an XorShiftRng does this via io::fs, which
// relies on the scheduler existing, so we have to manually load
// randomness. Windows has its own C API for this, so we don't need to
// worry there.
#[cfg(windows)]
fn new_sched_rng() -> XorShiftRng {
    XorShiftRng::new()
}
#[cfg(unix)]
fn new_sched_rng() -> XorShiftRng {
    use libc;
    use mem;
    use c_str::ToCStr;
    use vec::MutableVector;
    use iter::Iterator;
    use rand::SeedableRng;

    let fd = "/dev/urandom".with_c_str(|name| {
        unsafe { libc::open(name, libc::O_RDONLY, 0) }
    });
    if fd == -1 {
        rtabort!("could not open /dev/urandom for reading.")
    }

    let mut seeds = [0u32, .. 4];
    let size = mem::size_of_val(&seeds);
    loop {
        let nbytes = seeds.as_mut_buf(|buf, _| {
            unsafe {
                libc::read(fd,
                           buf as *mut libc::c_void,
                           size as libc::size_t)
            }
        });
        rtassert!(nbytes as uint == size);

        if !seeds.iter().all(|x| *x == 0) {
            break;
        }
    }

    unsafe {libc::close(fd);}

    SeedableRng::from_seed(seeds)
}

#[cfg(test)]
mod test {
    use prelude::*;
    use rt::test::*;
    use unstable::run_in_bare_thread;
    use borrow::to_uint;
    use rt::sched::{Scheduler};
    use cell::Cell;
    use rt::deque::BufferPool;
    use rt::thread::Thread;
    use rt::task::{Task, Sched};
    use rt::basic;
    use rt::util;
    use option::{Some};
    use rt::task::UnwindResult;

    #[test]
    fn trivial_run_in_newsched_task_test() {
        let mut task_ran = false;
        let task_ran_ptr: *mut bool = &mut task_ran;
        do run_in_newsched_task || {
            unsafe { *task_ran_ptr = true };
            rtdebug!("executed from the new scheduler")
        }
        assert!(task_ran);
    }

    #[test]
    fn multiple_task_test() {
        let total = 10;
        let mut task_run_count = 0;
        let task_run_count_ptr: *mut uint = &mut task_run_count;
        do run_in_newsched_task || {
            for _ in range(0u, total) {
                do spawntask || {
                    unsafe { *task_run_count_ptr = *task_run_count_ptr + 1};
                }
            }
        }
        assert!(task_run_count == total);
    }

    #[test]
    fn multiple_task_nested_test() {
        let mut task_run_count = 0;
        let task_run_count_ptr: *mut uint = &mut task_run_count;
        do run_in_newsched_task || {
            do spawntask || {
                unsafe { *task_run_count_ptr = *task_run_count_ptr + 1 };
                do spawntask || {
                    unsafe { *task_run_count_ptr = *task_run_count_ptr + 1 };
                    do spawntask || {
                        unsafe { *task_run_count_ptr = *task_run_count_ptr + 1 };
                    }
                }
            }
        }
        assert!(task_run_count == 3);
    }

    // Confirm that a sched_id actually is the uint form of the
    // pointer to the scheduler struct.
    #[test]
    fn simple_sched_id_test() {
        do run_in_bare_thread {
            let sched = ~new_test_uv_sched();
            assert!(to_uint(sched) == sched.sched_id());
        }
    }

    // Compare two scheduler ids that are different, this should never
    // fail but may catch a mistake someday.
    #[test]
    fn compare_sched_id_test() {
        do run_in_bare_thread {
            let sched_one = ~new_test_uv_sched();
            let sched_two = ~new_test_uv_sched();
            assert!(sched_one.sched_id() != sched_two.sched_id());
        }
    }


    // A very simple test that confirms that a task executing on the
    // home scheduler notices that it is home.
    #[test]
    fn test_home_sched() {
        do run_in_bare_thread {
            let mut task_ran = false;
            let task_ran_ptr: *mut bool = &mut task_ran;

            let mut sched = ~new_test_uv_sched();
            let sched_handle = sched.make_handle();

            let mut task = ~do Task::new_root_homed(&mut sched.stack_pool, None,
                                                Sched(sched_handle)) {
                unsafe { *task_ran_ptr = true };
                assert!(Task::on_appropriate_sched());
            };

            let on_exit: proc(UnwindResult) = proc(exit_status) {
                rtassert!(exit_status.is_success())
            };
            task.death.on_exit = Some(on_exit);

            sched.bootstrap(task);
        }
    }

    // An advanced test that checks all four possible states that a
    // (task,sched) can be in regarding homes.

    #[test]
    fn test_schedule_home_states() {
        use rt::sleeper_list::SleeperList;
        use rt::sched::Shutdown;
        use borrow;
        use rt::comm::*;

        do run_in_bare_thread {

            let sleepers = SleeperList::new();
            let mut pool = BufferPool::new();
            let (normal_worker, normal_stealer) = pool.deque();
            let (special_worker, special_stealer) = pool.deque();
            let queues = ~[normal_stealer, special_stealer];

            // Our normal scheduler
            let mut normal_sched = ~Scheduler::new(
                basic::event_loop(),
                normal_worker,
                queues.clone(),
                sleepers.clone());

            let normal_handle = Cell::new(normal_sched.make_handle());

            let friend_handle = normal_sched.make_handle();

            // Our special scheduler
            let mut special_sched = ~Scheduler::new_special(
                basic::event_loop(),
                special_worker,
                queues.clone(),
                sleepers.clone(),
                false,
                Some(friend_handle));

            let special_handle = Cell::new(special_sched.make_handle());

            let t1_handle = special_sched.make_handle();
            let t4_handle = special_sched.make_handle();

            // Four test tasks:
            //   1) task is home on special
            //   2) task not homed, sched doesn't care
            //   3) task not homed, sched requeues
            //   4) task not home, send home

            let task1 = ~do Task::new_root_homed(&mut special_sched.stack_pool, None,
                                                 Sched(t1_handle)) || {
                rtassert!(Task::on_appropriate_sched());
            };
            rtdebug!("task1 id: **{}**", borrow::to_uint(task1));

            let task2 = ~do Task::new_root(&mut normal_sched.stack_pool, None) {
                rtassert!(Task::on_appropriate_sched());
            };

            let task3 = ~do Task::new_root(&mut normal_sched.stack_pool, None) {
                rtassert!(Task::on_appropriate_sched());
            };

            let task4 = ~do Task::new_root_homed(&mut special_sched.stack_pool, None,
                                                 Sched(t4_handle)) {
                rtassert!(Task::on_appropriate_sched());
            };
            rtdebug!("task4 id: **{}**", borrow::to_uint(task4));

            let task1 = Cell::new(task1);
            let task2 = Cell::new(task2);
            let task3 = Cell::new(task3);
            let task4 = Cell::new(task4);

            // Signal from the special task that we are done.
            let (port, chan) = oneshot::<()>();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            let normal_task = ~do Task::new_root(&mut normal_sched.stack_pool, None) {
                rtdebug!("*about to submit task2*");
                Scheduler::run_task(task2.take());
                rtdebug!("*about to submit task4*");
                Scheduler::run_task(task4.take());
                rtdebug!("*normal_task done*");
                port.take().recv();
                let mut nh = normal_handle.take();
                nh.send(Shutdown);
                let mut sh = special_handle.take();
                sh.send(Shutdown);
            };

            rtdebug!("normal task: {}", borrow::to_uint(normal_task));

            let special_task = ~do Task::new_root(&mut special_sched.stack_pool, None) {
                rtdebug!("*about to submit task1*");
                Scheduler::run_task(task1.take());
                rtdebug!("*about to submit task3*");
                Scheduler::run_task(task3.take());
                rtdebug!("*done with special_task*");
                chan.take().send(());
            };

            rtdebug!("special task: {}", borrow::to_uint(special_task));

            let special_sched = Cell::new(special_sched);
            let normal_sched = Cell::new(normal_sched);
            let special_task = Cell::new(special_task);
            let normal_task = Cell::new(normal_task);

            let normal_thread = do Thread::start {
                normal_sched.take().bootstrap(normal_task.take());
                rtdebug!("finished with normal_thread");
            };

            let special_thread = do Thread::start {
                special_sched.take().bootstrap(special_task.take());
                rtdebug!("finished with special_sched");
            };

            normal_thread.join();
            special_thread.join();
        }
    }

    #[test]
    fn test_stress_schedule_task_states() {
        if util::limit_thread_creation_due_to_osx_and_valgrind() { return; }
        let n = stress_factor() * 120;
        for _ in range(0, n as int) {
            test_schedule_home_states();
        }
    }

    #[test]
    fn test_io_callback() {
        use io::timer;

        // This is a regression test that when there are no schedulable tasks
        // in the work queue, but we are performing I/O, that once we do put
        // something in the work queue again the scheduler picks it up and doesn't
        // exit before emptying the work queue
        do run_in_uv_task {
            do spawntask {
                timer::sleep(10);
            }
        }
    }

    #[test]
    fn handle() {
        use rt::comm::*;

        do run_in_bare_thread {
            let (port, chan) = oneshot::<()>();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            let thread_one = do Thread::start {
                let chan = Cell::new(chan.take());
                do run_in_newsched_task_core {
                    chan.take().send(());
                }
            };

            let thread_two = do Thread::start {
                let port = Cell::new(port.take());
                do run_in_newsched_task_core {
                    port.take().recv();
                }
            };

            thread_two.join();
            thread_one.join();
        }
    }

    // A regression test that the final message is always handled.
    // Used to deadlock because Shutdown was never recvd.
    #[test]
    fn no_missed_messages() {
        use rt::sleeper_list::SleeperList;
        use rt::stack::StackPool;
        use rt::sched::{Shutdown, TaskFromFriend};

        do run_in_bare_thread {
            stress_factor().times(|| {
                let sleepers = SleeperList::new();
                let mut pool = BufferPool::new();
                let (worker, stealer) = pool.deque();

                let mut sched = ~Scheduler::new(
                    basic::event_loop(),
                    worker,
                    ~[stealer],
                    sleepers.clone());

                let mut handle = sched.make_handle();

                let sched = Cell::new(sched);

                let thread = do Thread::start {
                    let mut sched = sched.take();
                    let bootstrap_task =
                        ~Task::new_root(&mut sched.stack_pool,
                                        None,
                                        proc()());
                    sched.bootstrap(bootstrap_task);
                };

                let mut stack_pool = StackPool::new();
                let task = ~Task::new_root(&mut stack_pool, None, proc()());
                handle.send(TaskFromFriend(task));

                handle.send(Shutdown);
                drop(handle);

                thread.join();
            })
        }
    }

    #[test]
    fn multithreading() {
        use rt::comm::*;
        use num::Times;
        use vec::OwnedVector;
        use container::Container;

        do run_in_mt_newsched_task {
            let mut ports = ~[];
            10.times(|| {
                let (port, chan) = oneshot();
                let chan_cell = Cell::new(chan);
                do spawntask_later {
                    chan_cell.take().send(());
                }
                ports.push(port);
            });

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

            let (p, ch1) = stream();
            let mut p = p;
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
                                debug!("{}\n", id);
                                end_chan.send(());
                                return;
                    }
                    (token, end_chan) => {
                        debug!("thread: {}   got token: {}", id, token);
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

        // Regression test that the `start` task entrypoint can
        // contain dtors that use task resources
        do run_in_newsched_task {
            struct S { field: () }

            impl Drop for S {
                fn drop(&mut self) {
                    let _foo = @0;
                }
            }

            let s = S { field: () };

            do spawntask {
                let _ss = &s;
            }
        }
    }

    // FIXME: #9407: xfail-test
    fn dont_starve_1() {
        use rt::comm::oneshot;

        stress_factor().times(|| {
            do run_in_mt_newsched_task {
                let (port, chan) = oneshot();

                // This task should not be able to starve the sender;
                // The sender should get stolen to another thread.
                do spawntask {
                    while !port.peek() { }
                }

                chan.send(());
            }
        })
    }

    #[test]
    fn dont_starve_2() {
        use rt::comm::oneshot;

        stress_factor().times(|| {
            do run_in_newsched_task {
                let (port, chan) = oneshot();
                let (_port2, chan2) = stream();

                // This task should not be able to starve the other task.
                // The sends should eventually yield.
                do spawntask {
                    while !port.peek() {
                        chan2.send(());
                    }
                }

                chan.send(());
            }
        })
    }

    // Regression test for a logic bug that would cause single-threaded schedulers
    // to sleep forever after yielding and stealing another task.
    #[test]
    fn single_threaded_yield() {
        use task::{spawn, spawn_sched, SingleThreaded, deschedule};
        use num::Times;

        do spawn_sched(SingleThreaded) {
            5.times(|| { deschedule(); })
        }
        do spawn { }
        do spawn { }
    }
}
