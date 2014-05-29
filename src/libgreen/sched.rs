// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::rt::local::Local;
use std::rt::rtio::{RemoteCallback, PausableIdleCallback, Callback, EventLoop};
use std::rt::task::BlockedTask;
use std::rt::task::Task;
use std::sync::deque;
use std::unstable::mutex::NativeMutex;
use std::raw;

use std::rand::{XorShiftRng, Rng, Rand};

use TaskState;
use context::Context;
use coroutine::Coroutine;
use sleeper_list::SleeperList;
use stack::StackPool;
use task::{TypeSched, GreenTask, HomeSched, AnySched};
use msgq = message_queue;

/// A scheduler is responsible for coordinating the execution of Tasks
/// on a single thread. The scheduler runs inside a slightly modified
/// Rust Task. When not running this task is stored in the scheduler
/// struct. The scheduler struct acts like a baton, all scheduling
/// actions are transfers of the baton.
///
/// FIXME: This creates too many callbacks to run_sched_once, resulting
/// in too much allocation and too many events.
pub struct Scheduler {
    /// ID number of the pool that this scheduler is a member of. When
    /// reawakening green tasks, this is used to ensure that tasks aren't
    /// reawoken on the wrong pool of schedulers.
    pub pool_id: uint,
    /// The pool of stacks that this scheduler has cached
    pub stack_pool: StackPool,
    /// Bookkeeping for the number of tasks which are currently running around
    /// inside this pool of schedulers
    pub task_state: TaskState,
    /// There are N work queues, one per scheduler.
    work_queue: deque::Worker<Box<GreenTask>>,
    /// Work queues for the other schedulers. These are created by
    /// cloning the core work queues.
    work_queues: Vec<deque::Stealer<Box<GreenTask>>>,
    /// The queue of incoming messages from other schedulers.
    /// These are enqueued by SchedHandles after which a remote callback
    /// is triggered to handle the message.
    message_queue: msgq::Consumer<SchedMessage>,
    /// Producer used to clone sched handles from
    message_producer: msgq::Producer<SchedMessage>,
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
    /// The scheduler runs on a special task. When it is not running
    /// it is stored here instead of the work queue.
    sched_task: Option<Box<GreenTask>>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    cleanup_job: Option<CleanupJob>,
    /// If the scheduler shouldn't run some tasks, a friend to send
    /// them to.
    friend_handle: Option<SchedHandle>,
    /// Should this scheduler run any task, or only pinned tasks?
    run_anything: bool,
    /// A fast XorShift rng for scheduler use
    rng: XorShiftRng,
    /// A togglable idle callback
    idle_callback: Option<Box<PausableIdleCallback:Send>>,
    /// A countdown that starts at a random value and is decremented
    /// every time a yield check is performed. When it hits 0 a task
    /// will yield.
    yield_check_count: uint,
    /// A flag to tell the scheduler loop it needs to do some stealing
    /// in order to introduce randomness as part of a yield
    steal_for_yield: bool,

    // n.b. currently destructors of an object are run in top-to-bottom in order
    //      of field declaration. Due to its nature, the pausable idle callback
    //      must have some sort of handle to the event loop, so it needs to get
    //      destroyed before the event loop itself. For this reason, we destroy
    //      the event loop last to ensure that any unsafe references to it are
    //      destroyed before it's actually destroyed.

    /// The event loop used to drive the scheduler and perform I/O
    pub event_loop: Box<EventLoop:Send>,
}

/// An indication of how hard to work on a given operation, the difference
/// mainly being whether memory is synchronized or not
#[deriving(Eq)]
enum EffortLevel {
    DontTryTooHard,
    GiveItYourBest
}

static MAX_YIELD_CHECKS: uint = 20000;

fn reset_yield_check(rng: &mut XorShiftRng) -> uint {
    let r: uint = Rand::rand(rng);
    r % MAX_YIELD_CHECKS + 1
}

impl Scheduler {

    // * Initialization Functions

    pub fn new(pool_id: uint,
               event_loop: Box<EventLoop:Send>,
               work_queue: deque::Worker<Box<GreenTask>>,
               work_queues: Vec<deque::Stealer<Box<GreenTask>>>,
               sleeper_list: SleeperList,
               state: TaskState)
        -> Scheduler {

        Scheduler::new_special(pool_id, event_loop, work_queue, work_queues,
                               sleeper_list, true, None, state)

    }

    pub fn new_special(pool_id: uint,
                       event_loop: Box<EventLoop:Send>,
                       work_queue: deque::Worker<Box<GreenTask>>,
                       work_queues: Vec<deque::Stealer<Box<GreenTask>>>,
                       sleeper_list: SleeperList,
                       run_anything: bool,
                       friend: Option<SchedHandle>,
                       state: TaskState)
        -> Scheduler {

        let (consumer, producer) = msgq::queue();
        let mut sched = Scheduler {
            pool_id: pool_id,
            sleeper_list: sleeper_list,
            message_queue: consumer,
            message_producer: producer,
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
            steal_for_yield: false,
            task_state: state,
        };

        sched.yield_check_count = reset_yield_check(&mut sched.rng);

        return sched;
    }

    // FIXME: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.

    // Take a main task to run, and a scheduler to run it in. Create a
    // scheduler task and bootstrap into it.
    pub fn bootstrap(mut ~self) {

        // Build an Idle callback.
        let cb = box SchedRunner as Box<Callback:Send>;
        self.idle_callback = Some(self.event_loop.pausable_idle_callback(cb));

        // Create a task for the scheduler with an empty context.
        let sched_task = GreenTask::new_typed(Some(Coroutine::empty()),
                                              TypeSched);

        // Before starting our first task, make sure the idle callback
        // is active. As we do not start in the sleep state this is
        // important.
        self.idle_callback.get_mut_ref().resume();

        // Now, as far as all the scheduler state is concerned, we are inside
        // the "scheduler" context. The scheduler immediately hands over control
        // to the event loop, and this will only exit once the event loop no
        // longer has any references (handles or I/O objects).
        rtdebug!("starting scheduler {}", self.sched_id());
        let mut sched_task = self.run(sched_task);

        // Close the idle callback.
        let mut sched = sched_task.sched.take_unwrap();
        sched.idle_callback.take();
        // Make one go through the loop to run the close callback.
        let mut stask = sched.run(sched_task);

        // Now that we are done with the scheduler, clean up the
        // scheduler task. Do so by removing it from TLS and manually
        // cleaning up the memory it uses. As we didn't actually call
        // task.run() on the scheduler task we never get through all
        // the cleanup code it runs.
        rtdebug!("stopping scheduler {}", stask.sched.get_ref().sched_id());

        // Should not have any messages
        let message = stask.sched.get_mut_ref().message_queue.pop();
        rtassert!(match message { msgq::Empty => true, _ => false });

        stask.task.get_mut_ref().destroyed = true;
    }

    // This does not return a scheduler, as the scheduler is placed
    // inside the task.
    pub fn run(mut ~self, stask: Box<GreenTask>) -> Box<GreenTask> {

        // This is unsafe because we need to place the scheduler, with
        // the event_loop inside, inside our task. But we still need a
        // mutable reference to the event_loop to give it the "run"
        // command.
        unsafe {
            let event_loop: *mut Box<EventLoop:Send> = &mut self.event_loop;
            // Our scheduler must be in the task before the event loop
            // is started.
            stask.put_with_sched(self);
            (*event_loop).run();
        }

        //  This is a serious code smell, but this function could be done away
        //  with if necessary. The ownership of `stask` was transferred into
        //  local storage just before the event loop ran, so it is possible to
        //  transmute `stask` as a uint across the running of the event loop to
        //  re-acquire ownership here.
        //
        // This would involve removing the Task from TLS, removing the runtime,
        // forgetting the runtime, and then putting the task into `stask`. For
        // now, because we have `GreenTask::convert`, I chose to take this
        // method for cleanliness. This function is *not* a fundamental reason
        // why this function should exist.
        GreenTask::convert(Local::take())
    }

    // * Execution Functions - Core Loop Logic

    // This function is run from the idle callback on the uv loop, indicating
    // that there are no I/O events pending. When this function returns, we will
    // fall back to epoll() in the uv event loop, waiting for more things to
    // happen. We may come right back off epoll() if the idle callback is still
    // active, in which case we're truly just polling to see if I/O events are
    // complete.
    //
    // The model for this function is to execute as much work as possible while
    // still fairly considering I/O tasks. Falling back to epoll() frequently is
    // often quite expensive, so we attempt to avoid it as much as possible. If
    // we have any active I/O on the event loop, then we're forced to fall back
    // to epoll() in order to provide fairness, but as long as we're doing work
    // and there's no active I/O, we can continue to do work.
    //
    // If we try really hard to do some work, but no work is available to be
    // done, then we fall back to epoll() to block this thread waiting for more
    // work (instead of busy waiting).
    fn run_sched_once(mut ~self, stask: Box<GreenTask>) {
        // Make sure that we're not lying in that the `stask` argument is indeed
        // the scheduler task for this scheduler.
        assert!(self.sched_task.is_none());

        // Assume that we need to continue idling unless we reach the
        // end of this function without performing an action.
        self.idle_callback.get_mut_ref().resume();

        // First we check for scheduler messages, these are higher
        // priority than regular tasks.
        let (mut sched, mut stask, mut did_work) =
            self.interpret_message_queue(stask, DontTryTooHard);

        // After processing a message, we consider doing some more work on the
        // event loop. The "keep going" condition changes after the first
        // iteration becase we don't want to spin here infinitely.
        //
        // Once we start doing work we can keep doing work so long as the
        // iteration does something. Note that we don't want to starve the
        // message queue here, so each iteration when we're done working we
        // check the message queue regardless of whether we did work or not.
        let mut keep_going = !did_work || !sched.event_loop.has_active_io();
        while keep_going {
            let (a, b, c) = match sched.do_work(stask) {
                (sched, task, false) => {
                    sched.interpret_message_queue(task, GiveItYourBest)
                }
                (sched, task, true) => {
                    let (sched, task, _) =
                        sched.interpret_message_queue(task, GiveItYourBest);
                    (sched, task, true)
                }
            };
            sched = a;
            stask = b;
            did_work = c;

            // We only keep going if we managed to do something productive and
            // also don't have any active I/O. If we didn't do anything, we
            // should consider going to sleep, and if we have active I/O we need
            // to poll for completion.
            keep_going = did_work && !sched.event_loop.has_active_io();
        }

        // If we ever did some work, then we shouldn't put our scheduler
        // entirely to sleep just yet. Leave the idle callback active and fall
        // back to epoll() to see what's going on.
        if did_work {
            return stask.put_with_sched(sched);
        }

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
        stask.put_with_sched(sched);
    }

    // This function returns None if the scheduler is "used", or it
    // returns the still-available scheduler. At this point all
    // message-handling will count as a turn of work, and as a result
    // return None.
    fn interpret_message_queue(mut ~self, stask: Box<GreenTask>,
                               effort: EffortLevel)
            -> (Box<Scheduler>, Box<GreenTask>, bool)
    {

        let msg = if effort == DontTryTooHard {
            self.message_queue.casual_pop()
        } else {
            // When popping our message queue, we could see an "inconsistent"
            // state which means that we *should* be able to pop data, but we
            // are unable to at this time. Our options are:
            //
            //  1. Spin waiting for data
            //  2. Ignore this and pretend we didn't find a message
            //
            // If we choose route 1, then if the pusher in question is currently
            // pre-empted, we're going to take up our entire time slice just
            // spinning on this queue. If we choose route 2, then the pusher in
            // question is still guaranteed to make a send() on its async
            // handle, so we will guaranteed wake up and see its message at some
            // point.
            //
            // I have chosen to take route #2.
            match self.message_queue.pop() {
                msgq::Data(t) => Some(t),
                msgq::Empty | msgq::Inconsistent => None
            }
        };

        match msg {
            Some(PinnedTask(task)) => {
                let mut task = task;
                task.give_home(HomeSched(self.make_handle()));
                let (sched, task) = self.resume_task_immediately(stask, task);
                (sched, task, true)
            }
            Some(TaskFromFriend(task)) => {
                rtdebug!("got a task from a friend. lovely!");
                let (sched, task) =
                    self.process_task(stask, task,
                                      Scheduler::resume_task_immediately_cl);
                (sched, task, true)
            }
            Some(RunOnce(task)) => {
                // bypass the process_task logic to force running this task once
                // on this home scheduler. This is often used for I/O (homing).
                let (sched, task) = self.resume_task_immediately(stask, task);
                (sched, task, true)
            }
            Some(Wake) => {
                self.sleepy = false;
                (self, stask, true)
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
                (self, stask, true)
            }
            Some(NewNeighbor(neighbor)) => {
                self.work_queues.push(neighbor);
                (self, stask, false)
            }
            None => (self, stask, false)
        }
    }

    fn do_work(mut ~self, stask: Box<GreenTask>)
               -> (Box<Scheduler>, Box<GreenTask>, bool) {
        rtdebug!("scheduler calling do work");
        match self.find_work() {
            Some(task) => {
                rtdebug!("found some work! running the task");
                let (sched, task) =
                    self.process_task(stask, task,
                                      Scheduler::resume_task_immediately_cl);
                (sched, task, true)
            }
            None => {
                rtdebug!("no work was found, returning the scheduler struct");
                (self, stask, false)
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
    fn find_work(&mut self) -> Option<Box<GreenTask>> {
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
    fn try_steals(&mut self) -> Option<Box<GreenTask>> {
        let work_queues = &mut self.work_queues;
        let len = work_queues.len();
        let start_index = self.rng.gen_range(0, len);
        for index in range(0, len).map(|i| (i + start_index) % len) {
            match work_queues.get_mut(index).steal() {
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

    fn process_task(mut ~self,
                    cur: Box<GreenTask>,
                    mut next: Box<GreenTask>,
                    schedule_fn: SchedulingFn)
                    -> (Box<Scheduler>, Box<GreenTask>) {
        rtdebug!("processing a task");

        match next.take_unwrap_home() {
            HomeSched(home_handle) => {
                if home_handle.sched_id != self.sched_id() {
                    rtdebug!("sending task home");
                    next.give_home(HomeSched(home_handle));
                    Scheduler::send_task_home(next);
                    (self, cur)
                } else {
                    rtdebug!("running task here");
                    next.give_home(HomeSched(home_handle));
                    schedule_fn(self, cur, next)
                }
            }
            AnySched if self.run_anything => {
                rtdebug!("running anysched task here");
                next.give_home(AnySched);
                schedule_fn(self, cur, next)
            }
            AnySched => {
                rtdebug!("sending task to friend");
                next.give_home(AnySched);
                self.send_to_friend(next);
                (self, cur)
            }
        }
    }

    fn send_task_home(task: Box<GreenTask>) {
        let mut task = task;
        match task.take_unwrap_home() {
            HomeSched(mut home_handle) => home_handle.send(PinnedTask(task)),
            AnySched => rtabort!("error: cannot send anysched task home"),
        }
    }

    /// Take a non-homed task we aren't allowed to run here and send
    /// it to the designated friend scheduler to execute.
    fn send_to_friend(&mut self, task: Box<GreenTask>) {
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
    pub fn enqueue_task(&mut self, task: Box<GreenTask>) {

        // We push the task onto our local queue clone.
        assert!(!task.is_sched());
        self.work_queue.push(task);
        match self.idle_callback {
            Some(ref mut idle) => idle.resume(),
            None => {} // allow enqueuing before the scheduler starts
        }

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

    // * Core Context Switching Functions

    // The primary function for changing contexts. In the current
    // design the scheduler is just a slightly modified GreenTask, so
    // all context swaps are from GreenTask to GreenTask. The only difference
    // between the various cases is where the inputs come from, and
    // what is done with the resulting task. That is specified by the
    // cleanup function f, which takes the scheduler and the
    // old task as inputs.

    pub fn change_task_context(mut ~self,
                               current_task: Box<GreenTask>,
                               mut next_task: Box<GreenTask>,
                               f: |&mut Scheduler, Box<GreenTask>|)
                               -> Box<GreenTask> {
        let f_opaque = ClosureConverter::from_fn(f);

        let current_task_dupe = &*current_task as *GreenTask;

        // The current task is placed inside an enum with the cleanup
        // function. This enum is then placed inside the scheduler.
        self.cleanup_job = Some(CleanupJob::new(current_task, f_opaque));

        // The scheduler is then placed inside the next task.
        next_task.sched = Some(self);

        // However we still need an internal mutable pointer to the
        // original task. The strategy here was "arrange memory, then
        // get pointers", so we crawl back up the chain using
        // transmute to eliminate borrowck errors.
        unsafe {

            let sched: &mut Scheduler =
                mem::transmute(&**next_task.sched.get_mut_ref());

            let current_task: &mut GreenTask = match sched.cleanup_job {
                Some(CleanupJob { task: ref mut task, .. }) => &mut **task,
                None => rtabort!("no cleanup job")
            };

            let (current_task_context, next_task_context) =
                Scheduler::get_contexts(current_task, next_task);

            // Done with everything - put the next task in TLS. This
            // works because due to transmute the borrow checker
            // believes that we have no internal pointers to
            // next_task.
            mem::forget(next_task);

            // The raw context swap operation. The next action taken
            // will be running the cleanup job from the context of the
            // next task.
            Context::swap(current_task_context, next_task_context);
        }

        // When the context swaps back to this task we immediately
        // run the cleanup job, as expected by the previously called
        // swap_contexts function.
        let mut current_task: Box<GreenTask> = unsafe {
            mem::transmute(current_task_dupe)
        };
        current_task.sched.get_mut_ref().run_cleanup_job();

        // See the comments in switch_running_tasks_and_then for why a lock
        // is acquired here. This is the resumption points and the "bounce"
        // that it is referring to.
        unsafe {
            let _guard = current_task.nasty_deschedule_lock.lock();
        }
        return current_task;
    }

    // Returns a mutable reference to both contexts involved in this
    // swap. This is unsafe - we are getting mutable internal
    // references to keep even when we don't own the tasks. It looks
    // kinda safe because we are doing transmutes before passing in
    // the arguments.
    pub fn get_contexts<'a>(current_task: &mut GreenTask,
                            next_task: &mut GreenTask)
        -> (&'a mut Context, &'a mut Context)
    {
        let current_task_context =
            &mut current_task.coroutine.get_mut_ref().saved_context;
        let next_task_context =
                &mut next_task.coroutine.get_mut_ref().saved_context;
        unsafe {
            (mem::transmute(current_task_context),
             mem::transmute(next_task_context))
        }
    }

    // * Context Swapping Helpers - Here be ugliness!

    pub fn resume_task_immediately(~self,
                                   cur: Box<GreenTask>,
                                   next: Box<GreenTask>)
                                   -> (Box<Scheduler>, Box<GreenTask>) {
        assert!(cur.is_sched());
        let mut cur = self.change_task_context(cur, next, |sched, stask| {
            assert!(sched.sched_task.is_none());
            sched.sched_task = Some(stask);
        });
        (cur.sched.take_unwrap(), cur)
    }

    fn resume_task_immediately_cl(sched: Box<Scheduler>,
                                  cur: Box<GreenTask>,
                                  next: Box<GreenTask>)
                                  -> (Box<Scheduler>, Box<GreenTask>) {
        sched.resume_task_immediately(cur, next)
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
                                            cur: Box<GreenTask>,
                                            f: |&mut Scheduler, BlockedTask|) {
        // Trickier - we need to get the scheduler task out of self
        // and use it as the destination.
        let stask = self.sched_task.take_unwrap();
        // Otherwise this is the same as below.
        self.switch_running_tasks_and_then(cur, stask, f)
    }

    pub fn switch_running_tasks_and_then(~self,
                                         cur: Box<GreenTask>,
                                         next: Box<GreenTask>,
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
        // fail!("should never return!");
        //
        // To actually maintain a handle to the lock, we use an unsafe pointer
        // to it, but we're guaranteed that the task won't exit until we've
        // unlocked the lock so there's no worry of this memory going away.
        let cur = self.change_task_context(cur, next, |sched, mut task| {
            let lock: *mut NativeMutex = &mut task.nasty_deschedule_lock;
            unsafe {
                let _guard = (*lock).lock();
                f(sched, BlockedTask::block(task.swap()));
            }
        });
        cur.put();
    }

    fn switch_task(sched: Box<Scheduler>,
                   cur: Box<GreenTask>,
                   next: Box<GreenTask>)
                   -> (Box<Scheduler>, Box<GreenTask>) {
        let mut cur = sched.change_task_context(cur, next, |sched, last_task| {
            if last_task.is_sched() {
                assert!(sched.sched_task.is_none());
                sched.sched_task = Some(last_task);
            } else {
                sched.enqueue_task(last_task);
            }
        });
        (cur.sched.take_unwrap(), cur)
    }

    // * Task Context Helpers

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    pub fn terminate_current_task(mut ~self, cur: Box<GreenTask>) -> ! {
        // Similar to deschedule running task and then, but cannot go through
        // the task-blocking path. The task is already dying.
        let stask = self.sched_task.take_unwrap();
        let _cur = self.change_task_context(cur, stask, |sched, mut dead_task| {
            let coroutine = dead_task.coroutine.take_unwrap();
            coroutine.recycle(&mut sched.stack_pool);
            sched.task_state.decrement();
        });
        fail!("should never return!");
    }

    pub fn run_task(~self, cur: Box<GreenTask>, next: Box<GreenTask>) {
        let (sched, task) =
            self.process_task(cur, next, Scheduler::switch_task);
        task.put_with_sched(sched);
    }

    pub fn run_task_later(mut cur: Box<GreenTask>, next: Box<GreenTask>) {
        let mut sched = cur.sched.take_unwrap();
        sched.enqueue_task(next);
        cur.put_with_sched(sched);
    }

    /// Yield control to the scheduler, executing another task. This is guaranteed
    /// to introduce some amount of randomness to the scheduler. Currently the
    /// randomness is a result of performing a round of work stealing (which
    /// may end up stealing from the current scheduler).
    pub fn yield_now(mut ~self, cur: Box<GreenTask>) {
        // Async handles trigger the scheduler by calling yield_now on the local
        // task, which eventually gets us to here. See comments in SchedRunner
        // for more info on this.
        if cur.is_sched() {
            assert!(self.sched_task.is_none());
            self.run_sched_once(cur);
        } else {
            self.yield_check_count = reset_yield_check(&mut self.rng);
            // Tell the scheduler to start stealing on the next iteration
            self.steal_for_yield = true;
            let stask = self.sched_task.take_unwrap();
            let cur = self.change_task_context(cur, stask, |sched, task| {
                sched.enqueue_task(task);
            });
            cur.put()
        }
    }

    pub fn maybe_yield(mut ~self, cur: Box<GreenTask>) {
        // It's possible for sched tasks to possibly call this function, and it
        // just means that they're likely sending on channels (which
        // occasionally call this function). Sched tasks follow different paths
        // when executing yield_now(), which may possibly trip the assertion
        // below. For this reason, we just have sched tasks bail out soon.
        //
        // Sched tasks have no need to yield anyway because as soon as they
        // return they'll yield to other threads by falling back to the event
        // loop. Additionally, we completely control sched tasks, so we can make
        // sure that they never execute more than enough code.
        if cur.is_sched() {
            return cur.put_with_sched(self)
        }

        // The number of times to do the yield check before yielding, chosen
        // arbitrarily.
        rtassert!(self.yield_check_count > 0);
        self.yield_check_count -= 1;
        if self.yield_check_count == 0 {
            self.yield_now(cur);
        } else {
            cur.put_with_sched(self);
        }
    }


    // * Utility Functions

    pub fn sched_id(&self) -> uint { self as *Scheduler as uint }

    pub fn run_cleanup_job(&mut self) {
        let cleanup_job = self.cleanup_job.take_unwrap();
        cleanup_job.run(self)
    }

    pub fn make_handle(&mut self) -> SchedHandle {
        let remote = self.event_loop.remote_callback(box SchedRunner);

        return SchedHandle {
            remote: remote,
            queue: self.message_producer.clone(),
            sched_id: self.sched_id()
        }
    }
}

// Supporting types

type SchedulingFn = fn(Box<Scheduler>, Box<GreenTask>, Box<GreenTask>)
                       -> (Box<Scheduler>, Box<GreenTask>);

pub enum SchedMessage {
    Wake,
    Shutdown,
    NewNeighbor(deque::Stealer<Box<GreenTask>>),
    PinnedTask(Box<GreenTask>),
    TaskFromFriend(Box<GreenTask>),
    RunOnce(Box<GreenTask>),
}

pub struct SchedHandle {
    remote: Box<RemoteCallback:Send>,
    queue: msgq::Producer<SchedMessage>,
    pub sched_id: uint
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
        // In theory, this function needs to invoke the `run_sched_once`
        // function on the scheduler. Sadly, we have no context here, except for
        // knowledge of the local `Task`. In order to avoid a call to
        // `GreenTask::convert`, we just call `yield_now` and the scheduler will
        // detect when a sched task performs a yield vs a green task performing
        // a yield (and act accordingly).
        //
        // This function could be converted to `GreenTask::convert` if
        // absolutely necessary, but for cleanliness it is much better to not
        // use the conversion function.
        let task: Box<Task> = Local::take();
        task.yield_now();
    }
}

struct CleanupJob {
    task: Box<GreenTask>,
    f: UnsafeTaskReceiver
}

impl CleanupJob {
    pub fn new(task: Box<GreenTask>, f: UnsafeTaskReceiver) -> CleanupJob {
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

// FIXME: Some hacks to put a || closure in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = raw::Closure;
trait ClosureConverter {
    fn from_fn(|&mut Scheduler, Box<GreenTask>|) -> Self;
    fn to_fn(self) -> |&mut Scheduler, Box<GreenTask>|;
}
impl ClosureConverter for UnsafeTaskReceiver {
    fn from_fn(f: |&mut Scheduler, Box<GreenTask>|) -> UnsafeTaskReceiver {
        unsafe { mem::transmute(f) }
    }
    fn to_fn(self) -> |&mut Scheduler, Box<GreenTask>| {
        unsafe { mem::transmute(self) }
    }
}

// On unix, we read randomness straight from /dev/urandom, but the
// default constructor of an XorShiftRng does this via io::fs, which
// relies on the scheduler existing, so we have to manually load
// randomness. Windows has its own C API for this, so we don't need to
// worry there.
#[cfg(windows)]
fn new_sched_rng() -> XorShiftRng {
    use std::rand::OSRng;
    match OSRng::new() {
        Ok(mut r) => r.gen(),
        Err(e) => {
            rtabort!("sched: failed to create seeded RNG: {}", e)
        }
    }
}
#[cfg(unix)]
fn new_sched_rng() -> XorShiftRng {
    use libc;
    use std::mem;
    use std::rand::SeedableRng;

    let fd = "/dev/urandom".with_c_str(|name| {
        unsafe { libc::open(name, libc::O_RDONLY, 0) }
    });
    if fd == -1 {
        rtabort!("could not open /dev/urandom for reading.")
    }

    let mut seeds = [0u32, .. 4];
    let size = mem::size_of_val(&seeds);
    loop {
        let nbytes = unsafe {
            libc::read(fd,
                       seeds.as_mut_ptr() as *mut libc::c_void,
                       size as libc::size_t)
        };
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
    use rustuv;

    use std::task::TaskOpts;
    use std::rt::task::Task;
    use std::rt::local::Local;

    use {TaskState, PoolConfig, SchedPool};
    use basic;
    use sched::{TaskFromFriend, PinnedTask};
    use task::{GreenTask, HomeSched, AnySched};

    fn pool() -> SchedPool {
        SchedPool::new(PoolConfig {
            threads: 1,
            event_loop_factory: basic::event_loop,
        })
    }

    fn run(f: proc():Send) {
        let mut pool = pool();
        pool.spawn(TaskOpts::new(), f);
        pool.shutdown();
    }

    fn sched_id() -> uint {
        let mut task = Local::borrow(None::<Task>);
        match task.maybe_take_runtime::<GreenTask>() {
            Some(green) => {
                let ret = green.sched.get_ref().sched_id();
                task.put_runtime(green);
                return ret;
            }
            None => fail!()
        }
    }

    #[test]
    fn trivial_run_in_newsched_task_test() {
        let mut task_ran = false;
        let task_ran_ptr: *mut bool = &mut task_ran;
        run(proc() {
            unsafe { *task_ran_ptr = true };
            rtdebug!("executed from the new scheduler")
        });
        assert!(task_ran);
    }

    #[test]
    fn multiple_task_test() {
        let total = 10;
        let mut task_run_count = 0;
        let task_run_count_ptr: *mut uint = &mut task_run_count;
        // with only one thread this is safe to run in without worries of
        // contention.
        run(proc() {
            for _ in range(0u, total) {
                spawn(proc() {
                    unsafe { *task_run_count_ptr = *task_run_count_ptr + 1};
                });
            }
        });
        assert!(task_run_count == total);
    }

    #[test]
    fn multiple_task_nested_test() {
        let mut task_run_count = 0;
        let task_run_count_ptr: *mut uint = &mut task_run_count;
        run(proc() {
            spawn(proc() {
                unsafe { *task_run_count_ptr = *task_run_count_ptr + 1 };
                spawn(proc() {
                    unsafe { *task_run_count_ptr = *task_run_count_ptr + 1 };
                    spawn(proc() {
                        unsafe { *task_run_count_ptr = *task_run_count_ptr + 1 };
                    })
                })
            })
        });
        assert!(task_run_count == 3);
    }

    // A very simple test that confirms that a task executing on the
    // home scheduler notices that it is home.
    #[test]
    fn test_home_sched() {
        let mut pool = pool();

        let (dtx, drx) = channel();
        {
            let (tx, rx) = channel();
            let mut handle1 = pool.spawn_sched();
            let mut handle2 = pool.spawn_sched();

            handle1.send(TaskFromFriend(pool.task(TaskOpts::new(), proc() {
                tx.send(sched_id());
            })));
            let sched1_id = rx.recv();

            let mut task = pool.task(TaskOpts::new(), proc() {
                assert_eq!(sched_id(), sched1_id);
                dtx.send(());
            });
            task.give_home(HomeSched(handle1));
            handle2.send(TaskFromFriend(task));
        }
        drx.recv();

        pool.shutdown();
    }

    // An advanced test that checks all four possible states that a
    // (task,sched) can be in regarding homes.

    #[test]
    fn test_schedule_home_states() {
        use sleeper_list::SleeperList;
        use super::{Shutdown, Scheduler, SchedHandle};
        use std::rt::thread::Thread;
        use std::sync::deque::BufferPool;

        Thread::start(proc() {
            let sleepers = SleeperList::new();
            let pool = BufferPool::new();
            let (normal_worker, normal_stealer) = pool.deque();
            let (special_worker, special_stealer) = pool.deque();
            let queues = vec![normal_stealer, special_stealer];
            let (_p, state) = TaskState::new();

            // Our normal scheduler
            let mut normal_sched = box Scheduler::new(
                1,
                basic::event_loop(),
                normal_worker,
                queues.clone(),
                sleepers.clone(),
                state.clone());

            let normal_handle = normal_sched.make_handle();
            let friend_handle = normal_sched.make_handle();

            // Our special scheduler
            let mut special_sched = box Scheduler::new_special(
                1,
                basic::event_loop(),
                special_worker,
                queues.clone(),
                sleepers.clone(),
                false,
                Some(friend_handle),
                state);

            let special_handle = special_sched.make_handle();

            let t1_handle = special_sched.make_handle();
            let t4_handle = special_sched.make_handle();

            // Four test tasks:
            //   1) task is home on special
            //   2) task not homed, sched doesn't care
            //   3) task not homed, sched requeues
            //   4) task not home, send home

            // Grab both the scheduler and the task from TLS and check if the
            // task is executing on an appropriate scheduler.
            fn on_appropriate_sched() -> bool {
                use task::{TypeGreen, TypeSched, HomeSched};
                let task = GreenTask::convert(Local::take());
                let sched_id = task.sched.get_ref().sched_id();
                let run_any = task.sched.get_ref().run_anything;
                let ret = match task.task_type {
                    TypeGreen(Some(AnySched)) => {
                        run_any
                    }
                    TypeGreen(Some(HomeSched(SchedHandle {
                        sched_id: ref id,
                        ..
                    }))) => {
                        *id == sched_id
                    }
                    TypeGreen(None) => { fail!("task without home"); }
                    TypeSched => { fail!("expected green task"); }
                };
                task.put();
                ret
            }

            let task1 = GreenTask::new_homed(&mut special_sched.stack_pool,
                                             None, HomeSched(t1_handle), proc() {
                rtassert!(on_appropriate_sched());
            });

            let task2 = GreenTask::new(&mut normal_sched.stack_pool, None, proc() {
                rtassert!(on_appropriate_sched());
            });

            let task3 = GreenTask::new(&mut normal_sched.stack_pool, None, proc() {
                rtassert!(on_appropriate_sched());
            });

            let task4 = GreenTask::new_homed(&mut special_sched.stack_pool,
                                             None, HomeSched(t4_handle), proc() {
                rtassert!(on_appropriate_sched());
            });

            // Signal from the special task that we are done.
            let (tx, rx) = channel::<()>();

            fn run(next: Box<GreenTask>) {
                let mut task = GreenTask::convert(Local::take());
                let sched = task.sched.take_unwrap();
                sched.run_task(task, next)
            }

            let normal_task = GreenTask::new(&mut normal_sched.stack_pool, None, proc() {
                run(task2);
                run(task4);
                rx.recv();
                let mut nh = normal_handle;
                nh.send(Shutdown);
                let mut sh = special_handle;
                sh.send(Shutdown);
            });
            normal_sched.enqueue_task(normal_task);

            let special_task = GreenTask::new(&mut special_sched.stack_pool, None, proc() {
                run(task1);
                run(task3);
                tx.send(());
            });
            special_sched.enqueue_task(special_task);

            let normal_sched = normal_sched;
            let normal_thread = Thread::start(proc() { normal_sched.bootstrap() });

            let special_sched = special_sched;
            let special_thread = Thread::start(proc() { special_sched.bootstrap() });

            normal_thread.join();
            special_thread.join();
        }).join();
    }

    //#[test]
    //fn test_stress_schedule_task_states() {
    //    if util::limit_thread_creation_due_to_osx_and_valgrind() { return; }
    //    let n = stress_factor() * 120;
    //    for _ in range(0, n as int) {
    //        test_schedule_home_states();
    //    }
    //}

    #[test]
    fn test_io_callback() {
        use std::io::timer;

        let mut pool = SchedPool::new(PoolConfig {
            threads: 2,
            event_loop_factory: rustuv::event_loop,
        });

        // This is a regression test that when there are no schedulable tasks in
        // the work queue, but we are performing I/O, that once we do put
        // something in the work queue again the scheduler picks it up and
        // doesn't exit before emptying the work queue
        pool.spawn(TaskOpts::new(), proc() {
            spawn(proc() {
                timer::sleep(10);
            });
        });

        pool.shutdown();
    }

    #[test]
    fn wakeup_across_scheds() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();

        let mut pool1 = pool();
        let mut pool2 = pool();

        pool1.spawn(TaskOpts::new(), proc() {
            let id = sched_id();
            tx1.send(());
            rx2.recv();
            assert_eq!(id, sched_id());
        });

        pool2.spawn(TaskOpts::new(), proc() {
            let id = sched_id();
            rx1.recv();
            assert_eq!(id, sched_id());
            tx2.send(());
        });

        pool1.shutdown();
        pool2.shutdown();
    }

    // A regression test that the final message is always handled.
    // Used to deadlock because Shutdown was never recvd.
    #[test]
    fn no_missed_messages() {
        let mut pool = pool();

        let task = pool.task(TaskOpts::new(), proc()());
        pool.spawn_sched().send(TaskFromFriend(task));

        pool.shutdown();
    }

    #[test]
    fn multithreading() {
        run(proc() {
            let mut rxs = vec![];
            for _ in range(0, 10) {
                let (tx, rx) = channel();
                spawn(proc() {
                    tx.send(());
                });
                rxs.push(rx);
            }

            loop {
                match rxs.pop() {
                    Some(rx) => rx.recv(),
                    None => break,
                }
            }
        });
    }

     #[test]
    fn thread_ring() {
        run(proc() {
            let (end_tx, end_rx) = channel();

            let n_tasks = 10;
            let token = 2000;

            let (tx1, mut rx) = channel();
            tx1.send((token, end_tx));
            let mut i = 2;
            while i <= n_tasks {
                let (tx, next_rx) = channel();
                let imm_i = i;
                let imm_rx = rx;
                spawn(proc() {
                    roundtrip(imm_i, n_tasks, &imm_rx, &tx);
                });
                rx = next_rx;
                i += 1;
            }
            let rx = rx;
            spawn(proc() {
                roundtrip(1, n_tasks, &rx, &tx1);
            });

            end_rx.recv();
        });

        fn roundtrip(id: int, n_tasks: int,
                     rx: &Receiver<(int, Sender<()>)>,
                     tx: &Sender<(int, Sender<()>)>) {
            loop {
                match rx.recv() {
                    (1, end_tx) => {
                        debug!("{}\n", id);
                        end_tx.send(());
                        return;
                    }
                    (token, end_tx) => {
                        debug!("thread: {}   got token: {}", id, token);
                        tx.send((token - 1, end_tx));
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
        // Regression test that the `start` task entrypoint can
        // contain dtors that use task resources
        run(proc() {
            struct S { field: () }

            impl Drop for S {
                fn drop(&mut self) {
                    let _foo = box 0;
                }
            }

            let s = S { field: () };

            spawn(proc() {
                let _ss = &s;
            });
        });
    }

    #[test]
    fn dont_starve_1() {
        let mut pool = SchedPool::new(PoolConfig {
            threads: 2, // this must be > 1
            event_loop_factory: basic::event_loop,
        });
        pool.spawn(TaskOpts::new(), proc() {
            let (tx, rx) = channel();

            // This task should not be able to starve the sender;
            // The sender should get stolen to another thread.
            spawn(proc() {
                while rx.try_recv().is_err() { }
            });

            tx.send(());
        });
        pool.shutdown();
    }

    #[test]
    fn dont_starve_2() {
        run(proc() {
            let (tx1, rx1) = channel();
            let (tx2, _rx2) = channel();

            // This task should not be able to starve the other task.
            // The sends should eventually yield.
            spawn(proc() {
                while rx1.try_recv().is_err() {
                    tx2.send(());
                }
            });

            tx1.send(());
        });
    }

    // Regression test for a logic bug that would cause single-threaded
    // schedulers to sleep forever after yielding and stealing another task.
    #[test]
    fn single_threaded_yield() {
        use std::task::deschedule;
        run(proc() {
            for _ in range(0, 5) { deschedule(); }
        });
    }

    #[test]
    fn test_spawn_sched_blocking() {
        use std::unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;

        // Testing that a task in one scheduler can block in foreign code
        // without affecting other schedulers
        for _ in range(0, 20) {
            let mut pool = pool();
            let (start_tx, start_rx) = channel();
            let (fin_tx, fin_rx) = channel();

            let mut handle = pool.spawn_sched();
            handle.send(PinnedTask(pool.task(TaskOpts::new(), proc() {
                unsafe {
                    let guard = LOCK.lock();

                    start_tx.send(());
                    guard.wait();   // block the scheduler thread
                    guard.signal(); // let them know we have the lock
                }

                fin_tx.send(());
            })));
            drop(handle);

            let mut handle = pool.spawn_sched();
            handle.send(PinnedTask(pool.task(TaskOpts::new(), proc() {
                // Wait until the other task has its lock
                start_rx.recv();

                fn pingpong(po: &Receiver<int>, ch: &Sender<int>) {
                    let mut val = 20;
                    while val > 0 {
                        val = po.recv();
                        let _ = ch.send_opt(val - 1);
                    }
                }

                let (setup_tx, setup_rx) = channel();
                let (parent_tx, parent_rx) = channel();
                spawn(proc() {
                    let (child_tx, child_rx) = channel();
                    setup_tx.send(child_tx);
                    pingpong(&child_rx, &parent_tx);
                });

                let child_tx = setup_rx.recv();
                child_tx.send(20);
                pingpong(&parent_rx, &child_tx);
                unsafe {
                    let guard = LOCK.lock();
                    guard.signal();   // wakeup waiting scheduler
                    guard.wait();     // wait for them to grab the lock
                }
            })));
            drop(handle);

            fin_rx.recv();
            pool.shutdown();
        }
        unsafe { LOCK.destroy(); }
    }
}
