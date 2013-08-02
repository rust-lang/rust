// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use either::{Left, Right};
use option::{Option, Some, None};
use cast::{transmute, transmute_mut_region, transmute_mut_unsafe};
use clone::Clone;
use unstable::raw;

use super::sleeper_list::SleeperList;
use super::work_queue::WorkQueue;
use super::stack::{StackPool};
use super::rtio::{EventLoop, EventLoopObject, RemoteCallbackObject};
use super::context::Context;
use super::task::{Task, AnySched, Sched};
use super::message_queue::MessageQueue;
use rt::kill::BlockedTask;
use rt::local_ptr;
use rt::local::Local;
use rt::rtio::RemoteCallback;
use rt::metrics::SchedMetrics;
use borrow::{to_uint};
use cell::Cell;

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
    work_queue: WorkQueue<~Task>,
    /// The queue of incoming messages from other schedulers.
    /// These are enqueued by SchedHandles after which a remote callback
    /// is triggered to handle the message.
    priv message_queue: MessageQueue<SchedMessage>,
    /// A shared list of sleeping schedulers. We'll use this to wake
    /// up schedulers when pushing work onto the work queue.
    sleeper_list: SleeperList,
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
    /// The scheduler runs on a special task.
    sched_task: Option<~Task>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    priv cleanup_job: Option<CleanupJob>,
    metrics: SchedMetrics,
    /// Should this scheduler run any task, or only pinned tasks?
    run_anything: bool,
    /// If the scheduler shouldn't run some tasks, a friend to send
    /// them to.
    friend_handle: Option<SchedHandle>
}

pub struct SchedHandle {
    priv remote: ~RemoteCallbackObject,
    priv queue: MessageQueue<SchedMessage>,
    sched_id: uint
}

pub enum SchedMessage {
    Wake,
    Shutdown,
    PinnedTask(~Task),
    TaskFromFriend(~Task)
}

enum CleanupJob {
    DoNothing,
    GiveTask(~Task, UnsafeTaskReceiver)
}

impl Scheduler {

    pub fn sched_id(&self) -> uint { to_uint(self) }

    pub fn new(event_loop: ~EventLoopObject,
               work_queue: WorkQueue<~Task>,
               sleeper_list: SleeperList)
        -> Scheduler {

        Scheduler::new_special(event_loop, work_queue, sleeper_list, true, None)

    }

    // When you create a scheduler it isn't yet "in" a task, so the
    // task field is None.
    pub fn new_special(event_loop: ~EventLoopObject,
                       work_queue: WorkQueue<~Task>,
                       sleeper_list: SleeperList,
                       run_anything: bool,
                       friend: Option<SchedHandle>)
        -> Scheduler {

        Scheduler {
            sleeper_list: sleeper_list,
            message_queue: MessageQueue::new(),
            sleepy: false,
            no_sleep: false,
            event_loop: event_loop,
            work_queue: work_queue,
            stack_pool: StackPool::new(),
            sched_task: None,
            cleanup_job: None,
            metrics: SchedMetrics::new(),
            run_anything: run_anything,
            friend_handle: friend
        }
    }

    // XXX: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.

    // Take a main task to run, and a scheduler to run it in. Create a
    // scheduler task and bootstrap into it.
    pub fn bootstrap(~self, task: ~Task) {

        // Initialize the TLS key.
        local_ptr::init_tls_key();

        // Create a task for the scheduler with an empty context.
        let sched_task = ~Task::new_sched_task();

        // Now that we have an empty task struct for the scheduler
        // task, put it in TLS.
        Local::put::(sched_task);

        // Now, as far as all the scheduler state is concerned, we are
        // inside the "scheduler" context. So we can act like the
        // scheduler and resume the provided task.
        self.resume_task_immediately(task);

        // Now we are back in the scheduler context, having
        // successfully run the input task. Start by running the
        // scheduler. Grab it out of TLS - performing the scheduler
        // action will have given it away.
        let sched = Local::take::<Scheduler>();

        rtdebug!("starting scheduler %u", sched.sched_id());

        sched.run();

        // Now that we are done with the scheduler, clean up the
        // scheduler task. Do so by removing it from TLS and manually
        // cleaning up the memory it uses. As we didn't actually call
        // task.run() on the scheduler task we never get through all
        // the cleanup code it runs.
        let mut stask = Local::take::<Task>();

        rtdebug!("stopping scheduler %u", stask.sched.get_ref().sched_id());

        stask.destroyed = true;
    }

    // This does not return a scheduler, as the scheduler is placed
    // inside the task.
    pub fn run(~self) {

        let mut self_sched = self;

        // Always run through the scheduler loop at least once so that
        // we enter the sleep state and can then be woken up by other
        // schedulers.
        self_sched.event_loop.callback(Scheduler::run_sched_once);

        // This is unsafe because we need to place the scheduler, with
        // the event_loop inside, inside our task. But we still need a
        // mutable reference to the event_loop to give it the "run"
        // command.
        unsafe {
            let event_loop: *mut ~EventLoopObject = &mut self_sched.event_loop;

            // Our scheduler must be in the task before the event loop
            // is started.
            let self_sched = Cell::new(self_sched);
            do Local::borrow::<Task,()> |stask| {
                stask.sched = Some(self_sched.take());
            };

            (*event_loop).run();
        }
    }

    // One iteration of the scheduler loop, always run at least once.

    // The model for this function is that you continue through it
    // until you either use the scheduler while performing a schedule
    // action, in which case you give it away and do not return, or
    // you reach the end and sleep. In the case that a scheduler
    // action is performed the loop is evented such that this function
    // is called again.
    fn run_sched_once() {

        // When we reach the scheduler context via the event loop we
        // already have a scheduler stored in our local task, so we
        // start off by taking it. This is the only path through the
        // scheduler where we get the scheduler this way.
        let sched = Local::take::<Scheduler>();

        // Our first task is to read mail to see if we have important
        // messages.

        // 1) A wake message is easy, mutate sched struct and return
        //    it.
        // 2) A shutdown is also easy, shutdown.
        // 3) A pinned task - we resume immediately and do not return
        //    here.
        // 4) A message from another scheduler with a non-homed task
        //    to run here.

        let result = sched.interpret_message_queue();
        let sched = match result {
            Some(sched) => {
                // We did not resume a task, so we returned.
                sched
            }
            None => {
                return;
            }
        };

        // Second activity is to try resuming a task from the queue.

        let result = sched.resume_task_from_queue();
        let mut sched = match result {
            Some(sched) => {
                // Failed to dequeue a task, so we return.
                sched
            }
            None => {
                return;
            }
        };

        // If we got here then there was no work to do.
        // Generate a SchedHandle and push it to the sleeper list so
        // somebody can wake us up later.
        sched.metrics.wasted_turns += 1;
        if !sched.sleepy && !sched.no_sleep {
            rtdebug!("scheduler has no work to do, going to sleep");
            sched.metrics.sleepy_times += 1;
            sched.sleepy = true;
            let handle = sched.make_handle();
            sched.sleeper_list.push(handle);
        } else {
            rtdebug!("not sleeping, already doing so or no_sleep set");
        }

        // Finished a cycle without using the Scheduler. Place it back
        // in TLS.
        Local::put(sched);
    }

    pub fn make_handle(&mut self) -> SchedHandle {
        let remote = self.event_loop.remote_callback(Scheduler::run_sched_once);

        return SchedHandle {
            remote: remote,
            queue: self.message_queue.clone(),
            sched_id: self.sched_id()
        };
    }

    /// Schedule a task to be executed later.
    ///
    /// Pushes the task onto the work stealing queue and tells the
    /// event loop to run it later. Always use this instead of pushing
    /// to the work queue directly.
    pub fn enqueue_task(&mut self, task: ~Task) {

        let this = self;

        // We push the task onto our local queue clone.
        this.work_queue.push(task);
        this.event_loop.callback(Scheduler::run_sched_once);

        // We've made work available. Notify a
        // sleeping scheduler.

        // XXX: perf. Check for a sleeper without
        // synchronizing memory.  It's not critical
        // that we always find it.

        // XXX: perf. If there's a sleeper then we
        // might as well just send it the task
        // directly instead of pushing it to the
        // queue. That is essentially the intent here
        // and it is less work.
        match this.sleeper_list.pop() {
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
        do blocked_task.wake().map_consume |task| {
            self.enqueue_task(task);
        };
    }

    // * Scheduler-context operations

    // This function returns None if the scheduler is "used", or it
    // returns the still-available scheduler.
    fn interpret_message_queue(~self) -> Option<~Scheduler> {

        let mut this = self;
        match this.message_queue.pop() {
            Some(PinnedTask(task)) => {
                let mut task = task;
                task.give_home(Sched(this.make_handle()));
                this.resume_task_immediately(task);
                return None;
            }
            Some(TaskFromFriend(task)) => {
                return this.sched_schedule_task(task);
            }
            Some(Wake) => {
                this.sleepy = false;
                return Some(this);
            }
            Some(Shutdown) => {
                if this.sleepy {
                    // There may be an outstanding handle on the
                    // sleeper list.  Pop them all to make sure that's
                    // not the case.
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
                // No more sleeping. After there are no outstanding
                // event loop references we will shut down.
                this.no_sleep = true;
                this.sleepy = false;
                // YYY: Does a shutdown count as a "use" of the
                // scheduler? This seems to work - so I'm leaving it
                // this way despite not having a solid rational for
                // why I should return the scheduler here.
                return Some(this);
            }
            None => {
                return Some(this);
            }
        }
    }

    /// Given an input Coroutine sends it back to its home scheduler.
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
        match self.friend_handle {
            Some(ref mut handle) => {
                handle.send(TaskFromFriend(task));
            }
            None => {
                rtabort!("tried to send task to a friend but scheduler has no friends");
            }
        }
    }

    // Resume a task from the queue - but also take into account that
    // it might not belong here.

    // If we perform a scheduler action we give away the scheduler ~
    // pointer, if it is still available we return it.

    fn resume_task_from_queue(~self) -> Option<~Scheduler> {

        let mut this = self;

        match this.work_queue.pop() {
            Some(task) => {
                let mut task = task;
                let home = task.take_unwrap_home();
                match home {
                    Sched(home_handle) => {
                        if home_handle.sched_id != this.sched_id() {
                            task.give_home(Sched(home_handle));
                            Scheduler::send_task_home(task);
                            return Some(this);
                        } else {
                            task.give_home(Sched(home_handle));
                            this.resume_task_immediately(task);
                            return None;
                        }
                    }
                    AnySched if this.run_anything => {
                        task.give_home(AnySched);
                        this.resume_task_immediately(task);
                        return None;
                    }
                    AnySched => {
                        task.give_home(AnySched);
                        this.send_to_friend(task);
                        return Some(this);
                    }
                }
            }
            None => {
                return Some(this);
            }
        }
    }

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    pub fn terminate_current_task(~self) {
        // Similar to deschedule running task and then, but cannot go through
        // the task-blocking path. The task is already dying.
        let mut this = self;
        let stask = this.sched_task.take_unwrap();
        do this.change_task_context(stask) |sched, mut dead_task| {
            let coroutine = dead_task.coroutine.take_unwrap();
            coroutine.recycle(&mut sched.stack_pool);
        }
    }

    // Scheduling a task requires a few checks to make sure the task
    // ends up in the appropriate location. The run_anything flag on
    // the scheduler and the home on the task need to be checked. This
    // helper performs that check. It takes a function that specifies
    // how to queue the the provided task if that is the correct
    // action. This is a "core" function that requires handling the
    // returned Option correctly.

    pub fn schedule_task(~self, task: ~Task,
                         schedule_fn: ~fn(sched: ~Scheduler, task: ~Task))
        -> Option<~Scheduler> {

        // is the task home?
        let is_home = task.is_home_no_tls(&self);

        // does the task have a home?
        let homed = task.homed();

        let mut this = self;

        if is_home || (!homed && this.run_anything) {
            // here we know we are home, execute now OR we know we
            // aren't homed, and that this sched doesn't care
            rtdebug!("task: %u is on ok sched, executing", to_uint(task));
            schedule_fn(this, task);
            return None;
        } else if !homed && !this.run_anything {
            // the task isn't homed, but it can't be run here
            this.send_to_friend(task);
            return Some(this);
        } else {
            // task isn't home, so don't run it here, send it home
            Scheduler::send_task_home(task);
            return Some(this);
        }
    }

    // There are two contexts in which schedule_task can be called:
    // inside the scheduler, and inside a task. These contexts handle
    // executing the task slightly differently. In the scheduler
    // context case we want to receive the scheduler as an input, and
    // manually deal with the option. In the task context case we want
    // to use TLS to find the scheduler, and deal with the option
    // inside the helper.

    pub fn sched_schedule_task(~self, task: ~Task) -> Option<~Scheduler> {
        do self.schedule_task(task) |sched, next_task| {
            sched.resume_task_immediately(next_task);
        }
    }

    // Task context case - use TLS.
    pub fn run_task(task: ~Task) {
        let sched = Local::take::<Scheduler>();
        let opt = do sched.schedule_task(task) |sched, next_task| {
            do sched.switch_running_tasks_and_then(next_task) |sched, last_task| {
                sched.enqueue_blocked_task(last_task);
            }
        };
        opt.map_consume(Local::put);
    }

    // The primary function for changing contexts. In the current
    // design the scheduler is just a slightly modified GreenTask, so
    // all context swaps are from Task to Task. The only difference
    // between the various cases is where the inputs come from, and
    // what is done with the resulting task. That is specified by the
    // cleanup function f, which takes the scheduler and the
    // old task as inputs.

    pub fn change_task_context(~self,
                               next_task: ~Task,
                               f: &fn(&mut Scheduler, ~Task)) {
        let mut this = self;

        // The current task is grabbed from TLS, not taken as an input.
        let current_task: ~Task = Local::take::<Task>();

        // These transmutes do something fishy with a closure.
        let f_fake_region = unsafe {
            transmute::<&fn(&mut Scheduler, ~Task),
                        &fn(&mut Scheduler, ~Task)>(f)
        };
        let f_opaque = ClosureConverter::from_fn(f_fake_region);

        // The current task is placed inside an enum with the cleanup
        // function. This enum is then placed inside the scheduler.
        this.enqueue_cleanup_job(GiveTask(current_task, f_opaque));

        // The scheduler is then placed inside the next task.
        let mut next_task = next_task;
        next_task.sched = Some(this);

        // However we still need an internal mutable pointer to the
        // original task. The strategy here was "arrange memory, then
        // get pointers", so we crawl back up the chain using
        // transmute to eliminate borrowck errors.
        unsafe {

            let sched: &mut Scheduler =
                transmute_mut_region(*next_task.sched.get_mut_ref());

            let current_task: &mut Task = match sched.cleanup_job {
                Some(GiveTask(ref task, _)) => {
                    transmute_mut_region(*transmute_mut_unsafe(task))
                }
                Some(DoNothing) => {
                    rtabort!("no next task");
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
            let sched = Local::unsafe_borrow::<Scheduler>();
            (*sched).run_cleanup_job();

            // Must happen after running the cleanup job (of course).
            let task = Local::unsafe_borrow::<Task>();
            (*task).death.check_killed();
        }
    }

    // Old API for task manipulation implemented over the new core
    // function.

    pub fn resume_task_immediately(~self, task: ~Task) {
        do self.change_task_context(task) |sched, stask| {
            sched.sched_task = Some(stask);
        }
    }

    pub fn resume_blocked_task_immediately(~self, blocked_task: BlockedTask) {
        match blocked_task.wake() {
            Some(task) => self.resume_task_immediately(task),
            None => Local::put(self),
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
    pub fn deschedule_running_task_and_then(~self, f: &fn(&mut Scheduler, BlockedTask)) {
        // Trickier - we need to get the scheduler task out of self
        // and use it as the destination.
        let mut this = self;
        let stask = this.sched_task.take_unwrap();
        // Otherwise this is the same as below.
        this.switch_running_tasks_and_then(stask, f);
    }

    pub fn switch_running_tasks_and_then(~self, next_task: ~Task,
                                         f: &fn(&mut Scheduler, BlockedTask)) {
        // This is where we convert the BlockedTask-taking closure into one
        // that takes just a Task, and is aware of the block-or-killed protocol.
        do self.change_task_context(next_task) |sched, task| {
            // Task might need to receive a kill signal instead of blocking.
            // We can call the "and_then" only if it blocks successfully.
            match BlockedTask::try_block(task) {
                Left(killed_task) => sched.enqueue_task(killed_task),
                Right(blocked_task) => f(sched, blocked_task),
            }
        }
    }

    // A helper that looks up the scheduler and runs a task later by
    // enqueuing it.
    pub fn run_task_later(next_task: ~Task) {
        // We aren't performing a scheduler operation, so we want to
        // put the Scheduler back when we finish.
        let next_task = Cell::new(next_task);
        do Local::borrow::<Scheduler,()> |sched| {
            sched.enqueue_task(next_task.take());
        };
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

    pub fn enqueue_cleanup_job(&mut self, job: CleanupJob) {
        self.cleanup_job = Some(job);
    }

    pub fn run_cleanup_job(&mut self) {
        rtdebug!("running cleanup job");
        let cleanup_job = self.cleanup_job.take_unwrap();
        match cleanup_job {
            DoNothing => { }
            GiveTask(task, f) => f.to_fn()(self, task)
        }
    }

}

// The cases for the below function.
enum ResumeAction {
    SendHome,
    Requeue,
    ResumeNow,
    Homeless
}

impl SchedHandle {
    pub fn send(&mut self, msg: SchedMessage) {
        self.queue.push(msg);
        self.remote.fire();
    }
}

// XXX: Some hacks to put a &fn in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = raw::Closure;
trait ClosureConverter {
    fn from_fn(&fn(&mut Scheduler, ~Task)) -> Self;
    fn to_fn(self) -> &fn(&mut Scheduler, ~Task);
}
impl ClosureConverter for UnsafeTaskReceiver {
    fn from_fn(f: &fn(&mut Scheduler, ~Task)) -> UnsafeTaskReceiver {
        unsafe { transmute(f) }
    }
    fn to_fn(self) -> &fn(&mut Scheduler, ~Task) { unsafe { transmute(self) } }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use rt::test::*;
    use unstable::run_in_bare_thread;
    use borrow::to_uint;
    use rt::local::*;
    use rt::sched::{Scheduler};
    use cell::Cell;
    use rt::thread::Thread;
    use rt::task::{Task, Sched};
    use option::{Some};

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
            foreach _ in range(0u, total) {
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

            let mut task = ~do Task::new_root_homed(&mut sched.stack_pool,
                                                Sched(sched_handle)) {
                unsafe { *task_ran_ptr = true };
                assert!(Task::on_appropriate_sched());
            };

            let on_exit: ~fn(bool) = |exit_status| rtassert!(exit_status);
            task.death.on_exit = Some(on_exit);

            sched.bootstrap(task);
        }
    }

    // An advanced test that checks all four possible states that a
    // (task,sched) can be in regarding homes.

    #[test]
    fn test_schedule_home_states() {

        use rt::uv::uvio::UvEventLoop;
        use rt::sleeper_list::SleeperList;
        use rt::work_queue::WorkQueue;
        use rt::sched::Shutdown;
        use borrow;
        use rt::comm::*;

        do run_in_bare_thread {

            let sleepers = SleeperList::new();
            let work_queue = WorkQueue::new();

            // Our normal scheduler
            let mut normal_sched = ~Scheduler::new(
                ~UvEventLoop::new(),
                work_queue.clone(),
                sleepers.clone());

            let normal_handle = Cell::new(normal_sched.make_handle());

            let friend_handle = normal_sched.make_handle();

            // Our special scheduler
            let mut special_sched = ~Scheduler::new_special(
                ~UvEventLoop::new(),
                work_queue.clone(),
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

            let task1 = ~do Task::new_root_homed(&mut special_sched.stack_pool,
                                                 Sched(t1_handle)) || {
                rtassert!(Task::on_appropriate_sched());
            };
            rtdebug!("task1 id: **%u**", borrow::to_uint(task1));

            let task2 = ~do Task::new_root(&mut normal_sched.stack_pool) {
                rtassert!(Task::on_appropriate_sched());
            };

            let task3 = ~do Task::new_root(&mut normal_sched.stack_pool) {
                rtassert!(Task::on_appropriate_sched());
            };

            let task4 = ~do Task::new_root_homed(&mut special_sched.stack_pool,
                                                 Sched(t4_handle)) {
                rtassert!(Task::on_appropriate_sched());
            };
            rtdebug!("task4 id: **%u**", borrow::to_uint(task4));

            let task1 = Cell::new(task1);
            let task2 = Cell::new(task2);
            let task3 = Cell::new(task3);
            let task4 = Cell::new(task4);

            // Signal from the special task that we are done.
            let (port, chan) = oneshot::<()>();
            let port = Cell::new(port);
            let chan = Cell::new(chan);

            let normal_task = ~do Task::new_root(&mut normal_sched.stack_pool) {
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

            rtdebug!("normal task: %u", borrow::to_uint(normal_task));

            let special_task = ~do Task::new_root(&mut special_sched.stack_pool) {
                rtdebug!("*about to submit task1*");
                Scheduler::run_task(task1.take());
                rtdebug!("*about to submit task3*");
                Scheduler::run_task(task3.take());
                rtdebug!("*done with special_task*");
                chan.take().send(());
            };

            rtdebug!("special task: %u", borrow::to_uint(special_task));

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
        let n = stress_factor() * 120;
        foreach _ in range(0, n as int) {
            test_schedule_home_states();
        }
    }

    #[test]
    fn test_io_callback() {
        // This is a regression test that when there are no schedulable tasks
        // in the work queue, but we are performing I/O, that once we do put
        // something in the work queue again the scheduler picks it up and doesn't
        // exit before emptying the work queue
        do run_in_newsched_task {
            do spawntask {
                let sched = Local::take::<Scheduler>();
                do sched.deschedule_running_task_and_then |sched, task| {
                    let task = Cell::new(task);
                    do sched.event_loop.callback_ms(10) {
                        rtdebug!("in callback");
                        let mut sched = Local::take::<Scheduler>();
                        sched.enqueue_blocked_task(task.take());
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

    #[test]
    fn multithreading() {
        use rt::comm::*;
        use iter::Times;
        use vec::OwnedVector;
        use container::Container;

        do run_in_mt_newsched_task {
            let mut ports = ~[];
            do 10.times {
                let (port, chan) = oneshot();
                let chan_cell = Cell::new(chan);
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

        // Regression test that the `start` task entrypoint can
        // contain dtors that use task resources
        do run_in_newsched_task {
            struct S { field: () }

            impl Drop for S {
                fn drop(&self) {
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
