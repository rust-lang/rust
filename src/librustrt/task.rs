// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Language-level runtime services that should reasonably expected
//! to be available 'everywhere'. Unwinding, local storage, and logging.
//! Even a 'freestanding' Rust would likely want to implement this.

pub use self::BlockedTask::*;
use self::TaskState::*;

use alloc::arc::Arc;
use alloc::boxed::Box;
use core::any::Any;
use core::atomic::{AtomicUint, SeqCst};
use core::iter::Take;
use core::kinds::marker;
use core::mem;
use core::prelude::{Clone, Drop, Err, Iterator, None, Ok, Option, Send, Some};
use core::prelude::{drop};

use bookkeeping;
use mutex::NativeMutex;
use local::Local;
use thread::{mod, Thread};
use stack;
use unwind;
use unwind::Unwinder;
use collections::str::SendStr;

/// State associated with Rust tasks.
///
/// This structure is currently undergoing major changes, and is
/// likely to be move/be merged with a `Thread` structure.
pub struct Task {
    pub unwinder: Unwinder,
    pub death: Death,
    pub name: Option<SendStr>,

    state: TaskState,
    lock: NativeMutex,       // native synchronization
    awoken: bool,            // used to prevent spurious wakeups

    // This field holds the known bounds of the stack in (lo, hi) form. Not all
    // native tasks necessarily know their precise bounds, hence this is
    // optional.
    stack_bounds: (uint, uint),

    stack_guard: uint
}

// Once a task has entered the `Armed` state it must be destroyed via `drop`,
// and no other method. This state is used to track this transition.
#[deriving(PartialEq)]
enum TaskState {
    New,
    Armed,
    Destroyed,
}

pub struct TaskOpts {
    /// Invoke this procedure with the result of the task when it finishes.
    pub on_exit: Option<proc(Result): Send>,
    /// A name for the task-to-be, for identification in panic messages
    pub name: Option<SendStr>,
    /// The size of the stack for the spawned task
    pub stack_size: Option<uint>,
}

/// Indicates the manner in which a task exited.
///
/// A task that completes without panicking is considered to exit successfully.
///
/// If you wish for this result's delivery to block until all
/// children tasks complete, recommend using a result future.
pub type Result = ::core::result::Result<(), Box<Any + Send>>;

/// A handle to a blocked task. Usually this means having the Box<Task>
/// pointer by ownership, but if the task is killable, a killer can steal it
/// at any time.
pub enum BlockedTask {
    Owned(Box<Task>),
    Shared(Arc<AtomicUint>),
}

/// Per-task state related to task death, killing, panic, etc.
pub struct Death {
    pub on_exit: Option<proc(Result):Send>,
    marker: marker::NoCopy,
}

pub struct BlockedTasks {
    inner: Arc<AtomicUint>,
}

impl Task {
    /// Creates a new uninitialized task.
    pub fn new(stack_bounds: Option<(uint, uint)>, stack_guard: Option<uint>) -> Task {
        Task {
            unwinder: Unwinder::new(),
            death: Death::new(),
            state: New,
            name: None,
            lock: unsafe { NativeMutex::new() },
            awoken: false,
            // these *should* get overwritten
            stack_bounds: stack_bounds.unwrap_or((0, 0)),
            stack_guard: stack_guard.unwrap_or(0)
        }
    }

    pub fn spawn(opts: TaskOpts, f: proc():Send) {
        let TaskOpts { name, stack_size, on_exit } = opts;

        let mut task = box Task::new(None, None);
        task.name = name;
        task.death.on_exit = on_exit;

        // FIXME: change this back after moving rustrt into std
        // let stack = stack_size.unwrap_or(rt::min_stack());
        let stack = stack_size.unwrap_or(2 * 1024 * 1024);

        // Note that this increment must happen *before* the spawn in order to
        // guarantee that if this task exits it will always end up waiting for
        // the spawned task to exit.
        let token = bookkeeping::increment();

        // Spawning a new OS thread guarantees that __morestack will never get
        // triggered, but we must manually set up the actual stack bounds once
        // this function starts executing. This raises the lower limit by a bit
        // because by the time that this function is executing we've already
        // consumed at least a little bit of stack (we don't know the exact byte
        // address at which our stack started).
        Thread::spawn_stack(stack, proc() {
            let something_around_the_top_of_the_stack = 1;
            let addr = &something_around_the_top_of_the_stack as *const int;
            let my_stack = addr as uint;
            unsafe {
                stack::record_os_managed_stack_bounds(my_stack - stack + 1024,
                                                      my_stack);
            }
            task.stack_guard = thread::current_guard_page();
            task.stack_bounds = (my_stack - stack + 1024, my_stack);

            let mut f = Some(f);
            drop(task.run(|| { f.take().unwrap()() }).destroy());
            drop(token);
        })
    }

    /// Consumes ownership of a task, runs some code, and returns the task back.
    ///
    /// This function can be used as an emulated "try/catch" to interoperate
    /// with the rust runtime at the outermost boundary. It is not possible to
    /// use this function in a nested fashion (a try/catch inside of another
    /// try/catch). Invoking this function is quite cheap.
    ///
    /// If the closure `f` succeeds, then the returned task can be used again
    /// for another invocation of `run`. If the closure `f` panics then `self`
    /// will be internally destroyed along with all of the other associated
    /// resources of this task. The `on_exit` callback is invoked with the
    /// cause of panic (not returned here). This can be discovered by querying
    /// `is_destroyed()`.
    ///
    /// Note that it is possible to view partial execution of the closure `f`
    /// because it is not guaranteed to run to completion, but this function is
    /// guaranteed to return if it panicks. Care should be taken to ensure that
    /// stack references made by `f` are handled appropriately.
    ///
    /// It is invalid to call this function with a task that has been previously
    /// destroyed via a failed call to `run`.
    pub fn run(mut self: Box<Task>, f: ||) -> Box<Task> {
        assert!(!self.is_destroyed(), "cannot re-use a destroyed task");

        // First, make sure that no one else is in TLS. This does not allow
        // recursive invocations of run(). If there's no one else, then
        // relinquish ownership of ourselves back into TLS.
        if Local::exists(None::<Task>) {
            panic!("cannot run a task recursively inside another");
        }
        self.state = Armed;
        Local::put(self);

        // There are two primary reasons that general try/catch is unsafe. The
        // first is that we do not support nested try/catch. The above check for
        // an existing task in TLS is sufficient for this invariant to be
        // upheld. The second is that unwinding while unwinding is not defined.
        // We take care of that by having an 'unwinding' flag in the task
        // itself. For these reasons, this unsafety should be ok.
        let result = unsafe { unwind::try(f) };

        // After running the closure given return the task back out if it ran
        // successfully, or clean up the task if it panicked.
        let task: Box<Task> = Local::take();
        match result {
            Ok(()) => task,
            Err(cause) => { task.cleanup(Err(cause)) }
        }
    }

    /// Destroy all associated resources of this task.
    ///
    /// This function will perform any necessary clean up to prepare the task
    /// for destruction. It is required that this is called before a `Task`
    /// falls out of scope.
    ///
    /// The returned task cannot be used for running any more code, but it may
    /// be used to extract the runtime as necessary.
    pub fn destroy(self: Box<Task>) -> Box<Task> {
        if self.is_destroyed() {
            self
        } else {
            self.cleanup(Ok(()))
        }
    }

    /// Cleans up a task, processing the result of the task as appropriate.
    ///
    /// This function consumes ownership of the task, deallocating it once it's
    /// done being processed. It is assumed that TLD and the local heap have
    /// already been destroyed and/or annihilated.
    fn cleanup(mut self: Box<Task>, result: Result) -> Box<Task> {
        // After taking care of the data above, we need to transmit the result
        // of this task.
        let what_to_do = self.death.on_exit.take();
        Local::put(self);

        // FIXME: this is running in a seriously constrained context. If this
        //        allocates TLD then it will likely abort the runtime. Similarly,
        //        if this panics, this will also likely abort the runtime.
        //
        //        This closure is currently limited to a channel send via the
        //        standard library's task interface, but this needs
        //        reconsideration to whether it's a reasonable thing to let a
        //        task to do or not.
        match what_to_do {
            Some(f) => { f(result) }
            None => { drop(result) }
        }

        // Now that we're done, we remove the task from TLS and flag it for
        // destruction.
        let mut task: Box<Task> = Local::take();
        task.state = Destroyed;
        return task;
    }

    /// Queries whether this can be destroyed or not.
    pub fn is_destroyed(&self) -> bool { self.state == Destroyed }

    /// Deschedules the current task, invoking `f` `amt` times. It is not
    /// recommended to use this function directly, but rather communication
    /// primitives in `std::comm` should be used.
    //
    // This function gets a little interesting. There are a few safety and
    // ownership violations going on here, but this is all done in the name of
    // shared state. Additionally, all of the violations are protected with a
    // mutex, so in theory there are no races.
    //
    // The first thing we need to do is to get a pointer to the task's internal
    // mutex. This address will not be changing (because the task is allocated
    // on the heap). We must have this handle separately because the task will
    // have its ownership transferred to the given closure. We're guaranteed,
    // however, that this memory will remain valid because *this* is the current
    // task's execution thread.
    //
    // The next weird part is where ownership of the task actually goes. We
    // relinquish it to the `f` blocking function, but upon returning this
    // function needs to replace the task back in TLS. There is no communication
    // from the wakeup thread back to this thread about the task pointer, and
    // there's really no need to. In order to get around this, we cast the task
    // to a `uint` which is then used at the end of this function to cast back
    // to a `Box<Task>` object. Naturally, this looks like it violates
    // ownership semantics in that there may be two `Box<Task>` objects.
    //
    // The fun part is that the wakeup half of this implementation knows to
    // "forget" the task on the other end. This means that the awakening half of
    // things silently relinquishes ownership back to this thread, but not in a
    // way that the compiler can understand. The task's memory is always valid
    // for both tasks because these operations are all done inside of a mutex.
    //
    // You'll also find that if blocking fails (the `f` function hands the
    // BlockedTask back to us), we will `mem::forget` the handles. The
    // reasoning for this is the same logic as above in that the task silently
    // transfers ownership via the `uint`, not through normal compiler
    // semantics.
    //
    // On a mildly unrelated note, it should also be pointed out that OS
    // condition variables are susceptible to spurious wakeups, which we need to
    // be ready for. In order to accommodate for this fact, we have an extra
    // `awoken` field which indicates whether we were actually woken up via some
    // invocation of `reawaken`. This flag is only ever accessed inside the
    // lock, so there's no need to make it atomic.
    pub fn deschedule(mut self: Box<Task>,
                      times: uint,
                      f: |BlockedTask| -> ::core::result::Result<(), BlockedTask>) {
        unsafe {
            let me = &mut *self as *mut Task;
            let task = BlockedTask::block(self);

            if times == 1 {
                let guard = (*me).lock.lock();
                (*me).awoken = false;
                match f(task) {
                    Ok(()) => {
                        while !(*me).awoken {
                            guard.wait();
                        }
                    }
                    Err(task) => { mem::forget(task.wake()); }
                }
            } else {
                let iter = task.make_selectable(times);
                let guard = (*me).lock.lock();
                (*me).awoken = false;

                // Apply the given closure to all of the "selectable tasks",
                // bailing on the first one that produces an error. Note that
                // care must be taken such that when an error is occurred, we
                // may not own the task, so we may still have to wait for the
                // task to become available. In other words, if task.wake()
                // returns `None`, then someone else has ownership and we must
                // wait for their signal.
                match iter.map(f).filter_map(|a| a.err()).next() {
                    None => {}
                    Some(task) => {
                        match task.wake() {
                            Some(task) => {
                                mem::forget(task);
                                (*me).awoken = true;
                            }
                            None => {}
                        }
                    }
                }
                while !(*me).awoken {
                    guard.wait();
                }
            }
            // put the task back in TLS, and everything is as it once was.
            Local::put(mem::transmute(me));
        }
    }

    /// Wakes up a previously blocked task. This function can only be
    /// called on tasks that were previously blocked in `deschedule`.
    //
    // See the comments on `deschedule` for why the task is forgotten here, and
    // why it's valid to do so.
    pub fn reawaken(mut self: Box<Task>) {
        unsafe {
            let me = &mut *self as *mut Task;
            mem::forget(self);
            let guard = (*me).lock.lock();
            (*me).awoken = true;
            guard.signal();
        }
    }

    /// Yields control of this task to another task. This function will
    /// eventually return, but possibly not immediately. This is used as an
    /// opportunity to allow other tasks a chance to run.
    pub fn yield_now() {
        Thread::yield_now();
    }

    /// Returns the stack bounds for this task in (lo, hi) format. The stack
    /// bounds may not be known for all tasks, so the return value may be
    /// `None`.
    pub fn stack_bounds(&self) -> (uint, uint) {
        self.stack_bounds
    }

    /// Returns the stack guard for this task, if known.
    pub fn stack_guard(&self) -> Option<uint> {
        if self.stack_guard != 0 {
            Some(self.stack_guard)
        } else {
            None
        }
    }

    /// Consume this task, flagging it as a candidate for destruction.
    ///
    /// This function is required to be invoked to destroy a task. A task
    /// destroyed through a normal drop will abort.
    pub fn drop(mut self) {
        self.state = Destroyed;
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        rtdebug!("called drop for a task: {}", self as *mut Task as uint);
        rtassert!(self.state != Armed);
    }
}

impl TaskOpts {
    pub fn new() -> TaskOpts {
        TaskOpts { on_exit: None, name: None, stack_size: None }
    }
}

impl Iterator<BlockedTask> for BlockedTasks {
    fn next(&mut self) -> Option<BlockedTask> {
        Some(Shared(self.inner.clone()))
    }
}

impl BlockedTask {
    /// Returns Some if the task was successfully woken; None if already killed.
    pub fn wake(self) -> Option<Box<Task>> {
        match self {
            Owned(task) => Some(task),
            Shared(arc) => {
                match arc.swap(0, SeqCst) {
                    0 => None,
                    n => Some(unsafe { mem::transmute(n) }),
                }
            }
        }
    }

    /// Reawakens this task if ownership is acquired. If finer-grained control
    /// is desired, use `wake` instead.
    pub fn reawaken(self) {
        self.wake().map(|t| t.reawaken());
    }

    // This assertion has two flavours because the wake involves an atomic op.
    // In the faster version, destructors will panic dramatically instead.
    #[cfg(not(test))] pub fn trash(self) { }
    #[cfg(test)]      pub fn trash(self) { assert!(self.wake().is_none()); }

    /// Create a blocked task, unless the task was already killed.
    pub fn block(task: Box<Task>) -> BlockedTask {
        Owned(task)
    }

    /// Converts one blocked task handle to a list of many handles to the same.
    pub fn make_selectable(self, num_handles: uint) -> Take<BlockedTasks> {
        let arc = match self {
            Owned(task) => {
                let flag = unsafe { AtomicUint::new(mem::transmute(task)) };
                Arc::new(flag)
            }
            Shared(arc) => arc.clone(),
        };
        BlockedTasks{ inner: arc }.take(num_handles)
    }

    /// Convert to an unsafe uint value. Useful for storing in a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_to_uint(self) -> uint {
        match self {
            Owned(task) => {
                let blocked_task_ptr: uint = mem::transmute(task);
                rtassert!(blocked_task_ptr & 0x1 == 0);
                blocked_task_ptr
            }
            Shared(arc) => {
                let blocked_task_ptr: uint = mem::transmute(box arc);
                rtassert!(blocked_task_ptr & 0x1 == 0);
                blocked_task_ptr | 0x1
            }
        }
    }

    /// Convert from an unsafe uint value. Useful for retrieving a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_from_uint(blocked_task_ptr: uint) -> BlockedTask {
        if blocked_task_ptr & 0x1 == 0 {
            Owned(mem::transmute(blocked_task_ptr))
        } else {
            let ptr: Box<Arc<AtomicUint>> =
                mem::transmute(blocked_task_ptr & !1);
            Shared(*ptr)
        }
    }
}

impl Death {
    pub fn new() -> Death {
        Death { on_exit: None, marker: marker::NoCopy }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::prelude::*;
    use std::task;
    use unwind;

    #[test]
    fn unwind() {
        let result = task::try(proc()());
        rtdebug!("trying first assert");
        assert!(result.is_ok());
        let result = task::try::<()>(proc() panic!());
        rtdebug!("trying second assert");
        assert!(result.is_err());
    }

    #[test]
    fn rng() {
        use std::rand::{StdRng, Rng};
        let mut r = StdRng::new().ok().unwrap();
        let _ = r.next_u32();
    }

    #[test]
    fn comm_stream() {
        let (tx, rx) = channel();
        tx.send(10i);
        assert!(rx.recv() == 10);
    }

    #[test]
    fn comm_shared_chan() {
        let (tx, rx) = channel();
        tx.send(10i);
        assert!(rx.recv() == 10);
    }

    #[test]
    #[should_fail]
    fn test_begin_unwind() {
        use unwind::begin_unwind;
        begin_unwind("cause", &(file!(), line!()))
    }

    #[test]
    fn drop_new_task_ok() {
        drop(Task::new(None, None));
    }

    // Task blocking tests

    #[test]
    fn block_and_wake() {
        let task = box Task::new(None, None);
        let task = BlockedTask::block(task).wake().unwrap();
        task.drop();
    }
}
