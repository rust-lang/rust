// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Language-level runtime services that should reasonably expected
//! to be available 'everywhere'. Local heaps, GC, unwinding,
//! local storage, and logging. Even a 'freestanding' Rust would likely want
//! to implement this.

use super::local_heap::LocalHeap;

use prelude::*;

use borrow;
use cast::transmute;
use cell::Cell;
use cleanup;
use libc::{c_void, uintptr_t, c_char, size_t};
use local_data;
use option::{Option, Some, None};
use rt::borrowck::BorrowRecord;
use rt::borrowck;
use rt::context::Context;
use rt::context;
use rt::env;
use rt::io::Writer;
use rt::kill::Death;
use rt::local::Local;
use rt::logging::StdErrLogger;
use rt::sched::{Scheduler, SchedHandle};
use rt::stack::{StackSegment, StackPool};
use send_str::SendStr;
use task::spawn::Taskgroup;
use unstable::finally::Finally;

// The Task struct represents all state associated with a rust
// task. There are at this point two primary "subtypes" of task,
// however instead of using a subtype we just have a "task_type" field
// in the struct. This contains a pointer to another struct that holds
// the type-specific state.

pub struct Task {
    heap: LocalHeap,
    priv gc: GarbageCollector,
    storage: LocalStorage,
    logger: StdErrLogger,
    unwinder: Unwinder,
    taskgroup: Option<Taskgroup>,
    death: Death,
    destroyed: bool,
    name: Option<SendStr>,
    coroutine: Option<Coroutine>,
    sched: Option<~Scheduler>,
    task_type: TaskType,
    // Dynamic borrowck debugging info
    borrow_list: Option<~[BorrowRecord]>,
    stdout_handle: Option<~Writer>,
}

pub enum TaskType {
    GreenTask(Option<SchedHome>),
    SchedTask
}

/// A coroutine is nothing more than a (register context, stack) pair.
pub struct Coroutine {
    /// The segment of stack on which the task is currently running or
    /// if the task is blocked, on which the task will resume
    /// execution.
    priv current_stack_segment: StackSegment,
    /// Always valid if the task is alive and not running.
    saved_context: Context
}

/// Some tasks have a dedicated home scheduler that they must run on.
pub enum SchedHome {
    AnySched,
    Sched(SchedHandle)
}

pub struct GarbageCollector;
pub struct LocalStorage(Option<local_data::Map>);

/// Represents the reason for the current unwinding process
pub enum UnwindResult {
    /// The task is ending successfully
    Success,

    /// The Task is failing with reason `UnwindReason`
    Failure(UnwindReason),
}

impl UnwindResult {
    /// Returns `true` if this `UnwindResult` is a failure
    #[inline]
    pub fn is_failure(&self) -> bool {
        match *self {
            Success => false,
            Failure(_) => true
        }
    }

    /// Returns `true` if this `UnwindResult` is a success
    #[inline]
    pub fn is_success(&self) -> bool {
        match *self {
            Success => true,
            Failure(_) => false
        }
    }
}

/// Represents the cause of a task failure
#[deriving(ToStr)]
pub enum UnwindReason {
    /// Failed with a string message
    UnwindReasonStr(SendStr),

    /// Failed with an `~Any`
    UnwindReasonAny(~Any),

    /// Failed because of linked failure
    UnwindReasonLinked
}

pub struct Unwinder {
    unwinding: bool,
    cause: Option<UnwindReason>
}

impl Unwinder {
    fn to_unwind_result(&mut self) -> UnwindResult {
        if self.unwinding {
            Failure(self.cause.take().unwrap())
        } else {
            Success
        }
    }
}

impl Task {

    // A helper to build a new task using the dynamically found
    // scheduler and task. Only works in GreenTask context.
    pub fn build_homed_child(stack_size: Option<uint>, f: ~fn(), home: SchedHome) -> ~Task {
        let f = Cell::new(f);
        let home = Cell::new(home);
        do Local::borrow |running_task: &mut Task| {
            let mut sched = running_task.sched.take_unwrap();
            let new_task = ~running_task.new_child_homed(&mut sched.stack_pool,
                                                         stack_size,
                                                         home.take(),
                                                         f.take());
            running_task.sched = Some(sched);
            new_task
        }
    }

    pub fn build_child(stack_size: Option<uint>, f: ~fn()) -> ~Task {
        Task::build_homed_child(stack_size, f, AnySched)
    }

    pub fn build_homed_root(stack_size: Option<uint>, f: ~fn(), home: SchedHome) -> ~Task {
        let f = Cell::new(f);
        let home = Cell::new(home);
        do Local::borrow |running_task: &mut Task| {
            let mut sched = running_task.sched.take_unwrap();
            let new_task = ~Task::new_root_homed(&mut sched.stack_pool,
                                                 stack_size,
                                                 home.take(),
                                                 f.take());
            running_task.sched = Some(sched);
            new_task
        }
    }

    pub fn build_root(stack_size: Option<uint>, f: ~fn()) -> ~Task {
        Task::build_homed_root(stack_size, f, AnySched)
    }

    pub fn new_sched_task() -> Task {
        Task {
            heap: LocalHeap::new(),
            gc: GarbageCollector,
            storage: LocalStorage(None),
            logger: StdErrLogger::new(),
            unwinder: Unwinder { unwinding: false, cause: None },
            taskgroup: None,
            death: Death::new(),
            destroyed: false,
            coroutine: Some(Coroutine::empty()),
            name: None,
            sched: None,
            task_type: SchedTask,
            borrow_list: None,
            stdout_handle: None,
        }
    }

    pub fn new_root(stack_pool: &mut StackPool,
                    stack_size: Option<uint>,
                    start: ~fn()) -> Task {
        Task::new_root_homed(stack_pool, stack_size, AnySched, start)
    }

    pub fn new_child(&mut self,
                     stack_pool: &mut StackPool,
                     stack_size: Option<uint>,
                     start: ~fn()) -> Task {
        self.new_child_homed(stack_pool, stack_size, AnySched, start)
    }

    pub fn new_root_homed(stack_pool: &mut StackPool,
                          stack_size: Option<uint>,
                          home: SchedHome,
                          start: ~fn()) -> Task {
        Task {
            heap: LocalHeap::new(),
            gc: GarbageCollector,
            storage: LocalStorage(None),
            logger: StdErrLogger::new(),
            unwinder: Unwinder { unwinding: false, cause: None },
            taskgroup: None,
            death: Death::new(),
            destroyed: false,
            name: None,
            coroutine: Some(Coroutine::new(stack_pool, stack_size, start)),
            sched: None,
            task_type: GreenTask(Some(home)),
            borrow_list: None,
            stdout_handle: None,
        }
    }

    pub fn new_child_homed(&mut self,
                           stack_pool: &mut StackPool,
                           stack_size: Option<uint>,
                           home: SchedHome,
                           start: ~fn()) -> Task {
        Task {
            heap: LocalHeap::new(),
            gc: GarbageCollector,
            storage: LocalStorage(None),
            logger: StdErrLogger::new(),
            unwinder: Unwinder { unwinding: false, cause: None },
            taskgroup: None,
            // FIXME(#7544) make watching optional
            death: self.death.new_child(),
            destroyed: false,
            name: None,
            coroutine: Some(Coroutine::new(stack_pool, stack_size, start)),
            sched: None,
            task_type: GreenTask(Some(home)),
            borrow_list: None,
            stdout_handle: None,
        }
    }

    pub fn give_home(&mut self, new_home: SchedHome) {
        match self.task_type {
            GreenTask(ref mut home) => {
                *home = Some(new_home);
            }
            SchedTask => {
                rtabort!("type error: used SchedTask as GreenTask");
            }
        }
    }

    pub fn take_unwrap_home(&mut self) -> SchedHome {
        match self.task_type {
            GreenTask(ref mut home) => {
                let out = home.take_unwrap();
                return out;
            }
            SchedTask => {
                rtabort!("type error: used SchedTask as GreenTask");
            }
        }
    }

    pub fn run(&mut self, f: &fn()) {
        rtdebug!("run called on task: {}", borrow::to_uint(self));

        // The only try/catch block in the world. Attempt to run the task's
        // client-specified code and catch any failures.
        do self.unwinder.try {

            // Run the task main function, then do some cleanup.
            do f.finally {

                // First, destroy task-local storage. This may run user dtors.
                //
                // FIXME #8302: Dear diary. I'm so tired and confused.
                // There's some interaction in rustc between the box
                // annihilator and the TLS dtor by which TLS is
                // accessed from annihilated box dtors *after* TLS is
                // destroyed. Somehow setting TLS back to null, as the
                // old runtime did, makes this work, but I don't currently
                // understand how. I would expect that, if the annihilator
                // reinvokes TLS while TLS is uninitialized, that
                // TLS would be reinitialized but never destroyed,
                // but somehow this works. I have no idea what's going
                // on but this seems to make things magically work. FML.
                //
                // (added after initial comment) A possible interaction here is
                // that the destructors for the objects in TLS themselves invoke
                // TLS, or possibly some destructors for those objects being
                // annihilated invoke TLS. Sadly these two operations seemed to
                // be intertwined, and miraculously work for now...
                self.storage.take();

                // Destroy remaining boxes. Also may run user dtors.
                unsafe { cleanup::annihilate(); }

                // Finally flush and destroy any output handles which the task
                // owns. There are no boxes here, and no user destructors should
                // run after this any more.
                match self.stdout_handle.take() {
                    Some(handle) => {
                        let mut handle = handle;
                        handle.flush();
                    }
                    None => {}
                }
            }
        }

        // Cleanup the dynamic borrowck debugging info
        borrowck::clear_task_borrow_list();

        // NB. We pass the taskgroup into death so that it can be dropped while
        // the unkillable counter is set. This is necessary for when the
        // taskgroup destruction code drops references on KillHandles, which
        // might require using unkillable (to synchronize with an unwrapper).
        self.death.collect_failure(self.unwinder.to_unwind_result(), self.taskgroup.take());
        self.destroyed = true;
    }

    // New utility functions for homes.

    pub fn is_home_no_tls(&self, sched: &~Scheduler) -> bool {
        match self.task_type {
            GreenTask(Some(AnySched)) => { false }
            GreenTask(Some(Sched(SchedHandle { sched_id: ref id, _}))) => {
                *id == sched.sched_id()
            }
            GreenTask(None) => {
                rtabort!("task without home");
            }
            SchedTask => {
                // Awe yea
                rtabort!("type error: expected: GreenTask, found: SchedTask");
            }
        }
    }

    pub fn homed(&self) -> bool {
        match self.task_type {
            GreenTask(Some(AnySched)) => { false }
            GreenTask(Some(Sched(SchedHandle { _ }))) => { true }
            GreenTask(None) => {
                rtabort!("task without home");
            }
            SchedTask => {
                rtabort!("type error: expected: GreenTask, found: SchedTask");
            }
        }
    }

    // Grab both the scheduler and the task from TLS and check if the
    // task is executing on an appropriate scheduler.
    pub fn on_appropriate_sched() -> bool {
        do Local::borrow |task: &mut Task| {
            let sched_id = task.sched.get_ref().sched_id();
            let sched_run_anything = task.sched.get_ref().run_anything;
            match task.task_type {
                GreenTask(Some(AnySched)) => {
                    rtdebug!("anysched task in sched check ****");
                    sched_run_anything
                }
                GreenTask(Some(Sched(SchedHandle { sched_id: ref id, _ }))) => {
                    rtdebug!("homed task in sched check ****");
                    *id == sched_id
                }
                GreenTask(None) => {
                    rtabort!("task without home");
                }
                SchedTask => {
                    rtabort!("type error: expected: GreenTask, found: SchedTask");
                }
            }
        }
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        rtdebug!("called drop for a task: {}", borrow::to_uint(self));
        rtassert!(self.destroyed);
    }
}

// Coroutines represent nothing more than a context and a stack
// segment.

impl Coroutine {

    pub fn new(stack_pool: &mut StackPool, stack_size: Option<uint>, start: ~fn()) -> Coroutine {
        let stack_size = match stack_size {
            Some(size) => size,
            None => env::min_stack()
        };
        let start = Coroutine::build_start_wrapper(start);
        let mut stack = stack_pool.take_segment(stack_size);
        let initial_context = Context::new(start, &mut stack);
        Coroutine {
            current_stack_segment: stack,
            saved_context: initial_context
        }
    }

    pub fn empty() -> Coroutine {
        Coroutine {
            current_stack_segment: StackSegment::new(0),
            saved_context: Context::empty()
        }
    }

    fn build_start_wrapper(start: ~fn()) -> ~fn() {
        let start_cell = Cell::new(start);
        let wrapper: ~fn() = || {
            // First code after swap to this new context. Run our
            // cleanup job.
            unsafe {

                // Again - might work while safe, or it might not.
                do Local::borrow |sched: &mut Scheduler| {
                    sched.run_cleanup_job();
                }

                // To call the run method on a task we need a direct
                // reference to it. The task is in TLS, so we can
                // simply unsafe_borrow it to get this reference. We
                // need to still have the task in TLS though, so we
                // need to unsafe_borrow.
                let task: *mut Task = Local::unsafe_borrow();

                do (*task).run {
                    // N.B. Removing `start` from the start wrapper
                    // closure by emptying a cell is critical for
                    // correctness. The ~Task pointer, and in turn the
                    // closure used to initialize the first call
                    // frame, is destroyed in the scheduler context,
                    // not task context. So any captured closures must
                    // not contain user-definable dtors that expect to
                    // be in task context. By moving `start` out of
                    // the closure, all the user code goes our of
                    // scope while the task is still running.
                    let start = start_cell.take();
                    start();
                };
            }

            // We remove the sched from the Task in TLS right now.
            let sched: ~Scheduler = Local::take();
            // ... allowing us to give it away when performing a
            // scheduling operation.
            sched.terminate_current_task()
        };
        return wrapper;
    }

    /// Destroy coroutine and try to reuse stack segment.
    pub fn recycle(self, stack_pool: &mut StackPool) {
        match self {
            Coroutine { current_stack_segment, _ } => {
                stack_pool.give_segment(current_stack_segment);
            }
        }
    }

}


// Just a sanity check to make sure we are catching a Rust-thrown exception
static UNWIND_TOKEN: uintptr_t = 839147;

impl Unwinder {
    pub fn try(&mut self, f: &fn()) {
        use unstable::raw::Closure;

        unsafe {
            let closure: Closure = transmute(f);
            let code = transmute(closure.code);
            let env = transmute(closure.env);

            let token = rust_try(try_fn, code, env);
            assert!(token == 0 || token == UNWIND_TOKEN);
        }

        extern fn try_fn(code: *c_void, env: *c_void) {
            unsafe {
                let closure: Closure = Closure {
                    code: transmute(code),
                    env: transmute(env),
                };
                let closure: &fn() = transmute(closure);
                closure();
            }
        }

        extern {
            #[rust_stack]
            fn rust_try(f: extern "C" fn(*c_void, *c_void),
                        code: *c_void,
                        data: *c_void) -> uintptr_t;
        }
    }

    pub fn begin_unwind(&mut self, cause: UnwindReason) -> ! {
        #[fixed_stack_segment]; #[inline(never)];

        self.unwinding = true;
        self.cause = Some(cause);
        unsafe {
            rust_begin_unwind(UNWIND_TOKEN);
            return transmute(());
        }
        extern {
            fn rust_begin_unwind(token: uintptr_t);
        }
    }
}

/// This function is invoked from rust's current __morestack function. Segmented
/// stacks are currently not enabled as segmented stacks, but rather one giant
/// stack segment. This means that whenever we run out of stack, we want to
/// truly consider it to be stack overflow rather than allocating a new stack.
#[no_mangle]      // - this is called from C code
#[no_split_stack] // - it would be sad for this function to trigger __morestack
#[doc(hidden)] // XXX: this function shouldn't have to be `pub` to get exported
               //      so it can be linked against, we should have a better way
               //      of specifying that.
pub extern "C" fn rust_stack_exhausted() {
    use rt::in_green_task_context;
    use rt::task::Task;
    use rt::local::Local;
    use unstable::intrinsics;

    unsafe {
        // We're calling this function because the stack just ran out. We need
        // to call some other rust functions, but if we invoke the functions
        // right now it'll just trigger this handler being called again. In
        // order to alleviate this, we move the stack limit to be inside of the
        // red zone that was allocated for exactly this reason.
        let limit = context::get_sp_limit();
        context::record_sp_limit(limit - context::RED_ZONE / 2);

        // This probably isn't the best course of action. Ideally one would want
        // to unwind the stack here instead of just aborting the entire process.
        // This is a tricky problem, however. There's a few things which need to
        // be considered:
        //
        //  1. We're here because of a stack overflow, yet unwinding will run
        //     destructors and hence arbitrary code. What if that code overflows
        //     the stack? One possibility is to use the above allocation of an
        //     extra 10k to hope that we don't hit the limit, and if we do then
        //     abort the whole program. Not the best, but kind of hard to deal
        //     with unless we want to switch stacks.
        //
        //  2. LLVM will optimize functions based on whether they can unwind or
        //     not. It will flag functions with 'nounwind' if it believes that
        //     the function cannot trigger unwinding, but if we do unwind on
        //     stack overflow then it means that we could unwind in any function
        //     anywhere. We would have to make sure that LLVM only places the
        //     nounwind flag on functions which don't call any other functions.
        //
        //  3. The function that overflowed may have owned arguments. These
        //     arguments need to have their destructors run, but we haven't even
        //     begun executing the function yet, so unwinding will not run the
        //     any landing pads for these functions. If this is ignored, then
        //     the arguments will just be leaked.
        //
        // Exactly what to do here is a very delicate topic, and is possibly
        // still up in the air for what exactly to do. Some relevant issues:
        //
        //  #3555 - out-of-stack failure leaks arguments
        //  #3695 - should there be a stack limit?
        //  #9855 - possible strategies which could be taken
        //  #9854 - unwinding on windows through __morestack has never worked
        //  #2361 - possible implementation of not using landing pads

        if in_green_task_context() {
            do Local::borrow |task: &mut Task| {
                let n = task.name.as_ref().map(|n| n.as_slice()).unwrap_or("<unnamed>");

                // See the message below for why this is not emitted to the
                // task's logger. This has the additional conundrum of the
                // logger may not be initialized just yet, meaning that an FFI
                // call would happen to initialized it (calling out to libuv),
                // and the FFI call needs 2MB of stack when we just ran out.
                rterrln!("task '{}' has overflowed its stack", n);
            }
        } else {
            rterrln!("stack overflow in non-task context");
        }

        intrinsics::abort();
    }
}

/// This is the entry point of unwinding for things like lang items and such.
/// The arguments are normally generated by the compiler, and need to
/// have static lifetimes.
pub fn begin_unwind(msg: *c_char, file: *c_char, line: size_t) -> ! {
    use c_str::CString;
    use cast::transmute;

    #[inline]
    fn static_char_ptr(p: *c_char) -> &'static str {
        let s = unsafe { CString::new(p, false) };
        match s.as_str() {
            Some(s) => unsafe { transmute::<&str, &'static str>(s) },
            None => rtabort!("message wasn't utf8?")
        }
    }

    let msg = static_char_ptr(msg);
    let file = static_char_ptr(file);

    begin_unwind_reason(UnwindReasonStr(msg.into_send_str()), file, line as uint)
}

/// This is the entry point of unwinding for fail!() and assert!().
pub fn begin_unwind_reason(reason: UnwindReason, file: &'static str, line: uint) -> ! {
    use rt::in_green_task_context;
    use rt::task::Task;
    use rt::local::Local;
    use str::Str;
    use unstable::intrinsics;

    unsafe {
        // Be careful not to allocate in this block, if we're failing we may
        // have been failing due to a lack of memory in the first place...

        let task: *mut Task;

        {
            let msg = match reason {
                UnwindReasonStr(ref s) => s.as_slice(),
                UnwindReasonAny(_)     => "~Any",
                UnwindReasonLinked     => "linked failure",
            };

            if !in_green_task_context() {
                rterrln!("failed in non-task context at '{}', {}:{}",
                        msg, file, line);
                intrinsics::abort();
            }

            task = Local::unsafe_borrow();
            let n = (*task).name.as_ref().map(|n| n.as_slice()).unwrap_or("<unnamed>");

            // XXX: this should no get forcibly printed to the console, this should
            //      either be sent to the parent task (ideally), or get printed to
            //      the task's logger. Right now the logger is actually a uvio
            //      instance, which uses unkillable blocks internally for various
            //      reasons. This will cause serious trouble if the task is failing
            //      due to mismanagment of its own kill flag, so calling our own
            //      logger in its current state is a bit of a problem.

            rterrln!("task '{}' failed at '{}', {}:{}", n, msg, file, line);

            if (*task).unwinder.unwinding {
                rtabort!("unwinding again");
            }
        }

        (*task).unwinder.begin_unwind(reason);
    }
}

#[cfg(test)]
mod test {
    use rt::test::*;

    #[test]
    fn local_heap() {
        do run_in_newsched_task() {
            let a = @5;
            let b = a;
            assert!(*a == 5);
            assert!(*b == 5);
        }
    }

    #[test]
    fn tls() {
        use local_data;
        do run_in_newsched_task() {
            local_data_key!(key: @~str)
            local_data::set(key, @~"data");
            assert!(*local_data::get(key, |k| k.map(|k| *k)).unwrap() == ~"data");
            local_data_key!(key2: @~str)
            local_data::set(key2, @~"data");
            assert!(*local_data::get(key2, |k| k.map(|k| *k)).unwrap() == ~"data");
        }
    }

    #[test]
    fn unwind() {
        do run_in_newsched_task() {
            let result = spawntask_try(||());
            rtdebug!("trying first assert");
            assert!(result.is_ok());
            let result = spawntask_try(|| fail!());
            rtdebug!("trying second assert");
            assert!(result.is_err());
        }
    }

    #[test]
    fn rng() {
        do run_in_uv_task() {
            use rand::{rng, Rng};
            let mut r = rng();
            let _ = r.next_u32();
        }
    }

    #[test]
    fn logging() {
        do run_in_uv_task() {
            info!("here i am. logging in a newsched task");
        }
    }

    #[test]
    fn comm_oneshot() {
        use comm::*;

        do run_in_newsched_task {
            let (port, chan) = oneshot();
            chan.send(10);
            assert!(port.recv() == 10);
        }
    }

    #[test]
    fn comm_stream() {
        use comm::*;

        do run_in_newsched_task() {
            let (port, chan) = stream();
            chan.send(10);
            assert!(port.recv() == 10);
        }
    }

    #[test]
    fn comm_shared_chan() {
        use comm::*;

        do run_in_newsched_task() {
            let (port, chan) = stream();
            let chan = SharedChan::new(chan);
            chan.send(10);
            assert!(port.recv() == 10);
        }
    }

    #[test]
    fn linked_failure() {
        do run_in_newsched_task() {
            let res = do spawntask_try {
                spawntask_random(|| fail!());
            };
            assert!(res.is_err());
        }
    }

    #[test]
    fn heap_cycles() {
        use option::{Option, Some, None};

        do run_in_newsched_task {
            struct List {
                next: Option<@mut List>,
            }

            let a = @mut List { next: None };
            let b = @mut List { next: Some(a) };

            a.next = Some(b);
        }
    }
}
