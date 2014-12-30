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

use any::Any;
use boxed::Box;
use sync::Arc;
use sync::atomic::{AtomicUint, SeqCst};
use iter::{IteratorExt, Take};
use kinds::marker;
use mem;
use ops::FnMut;
use core::prelude::{Clone, Drop, Err, Iterator, None, Ok, Option, Send, Some};
use core::prelude::{drop};
use str::SendStr;
use thunk::Thunk;

use rt;
use rt::mutex::NativeMutex;
use rt::local::Local;
use rt::thread::{mod, Thread};
use sys_common::stack;
use rt::unwind;
use rt::unwind::Unwinder;

/// State associated with Rust threads
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
    // native threads necessarily know their precise bounds, hence this is
    // optional.
    stack_bounds: (uint, uint),

    stack_guard: uint
}

// Once a thread has entered the `Armed` state it must be destroyed via `drop`,
// and no other method. This state is used to track this transition.
#[deriving(PartialEq)]
enum TaskState {
    New,
    Armed,
    Destroyed,
}

pub struct TaskOpts {
    /// Invoke this procedure with the result of the thread when it finishes.
    pub on_exit: Option<Thunk<Result>>,
    /// A name for the thread-to-be, for identification in panic messages
    pub name: Option<SendStr>,
    /// The size of the stack for the spawned thread
    pub stack_size: Option<uint>,
}

/// Indicates the manner in which a thread exited.
///
/// A thread that completes without panicking is considered to exit successfully.
///
/// If you wish for this result's delivery to block until all
/// children threads complete, recommend using a result future.
pub type Result = ::core::result::Result<(), Box<Any + Send>>;

/// A handle to a blocked thread. Usually this means having the Box<Task>
/// pointer by ownership, but if the thread is killable, a killer can steal it
/// at any time.
pub enum BlockedTask {
    Owned(Box<Task>),
    Shared(Arc<AtomicUint>),
}

/// Per-thread state related to thread death, killing, panic, etc.
pub struct Death {
    pub on_exit: Option<Thunk<Result>>,
}

pub struct BlockedTasks {
    inner: Arc<AtomicUint>,
}

impl Task {
    /// Creates a new uninitialized thread.
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

    pub fn spawn<F>(opts: TaskOpts, f: F)
        where F : FnOnce(), F : Send
    {
        Task::spawn_thunk(opts, Thunk::new(f))
    }

    fn spawn_thunk(opts: TaskOpts, f: Thunk) {
        let TaskOpts { name, stack_size, on_exit } = opts;

        let mut task = box Task::new(None, None);
        task.name = name;
        task.death.on_exit = on_exit;

        let stack = stack_size.unwrap_or(rt::min_stack());

        // Spawning a new OS thread guarantees that __morestack will never get
        // triggered, but we must manually set up the actual stack bounds once
        // this function starts executing. This raises the lower limit by a bit
        // because by the time that this function is executing we've already
        // consumed at least a little bit of stack (we don't know the exact byte
        // address at which our stack started).
        Thread::spawn_stack(stack, move|| {
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
            drop(task.run(|| { f.take().unwrap().invoke(()) }).destroy());
        })
    }

    /// Consumes ownership of a thread, runs some code, and returns the thread back.
    ///
    /// This function can be used as an emulated "try/catch" to interoperate
    /// with the rust runtime at the outermost boundary. It is not possible to
    /// use this function in a nested fashion (a try/catch inside of another
    /// try/catch). Invoking this function is quite cheap.
    ///
    /// If the closure `f` succeeds, then the returned thread can be used again
    /// for another invocation of `run`. If the closure `f` panics then `self`
    /// will be internally destroyed along with all of the other associated
    /// resources of this thread. The `on_exit` callback is invoked with the
    /// cause of panic (not returned here). This can be discovered by querying
    /// `is_destroyed()`.
    ///
    /// Note that it is possible to view partial execution of the closure `f`
    /// because it is not guaranteed to run to completion, but this function is
    /// guaranteed to return if it panicks. Care should be taken to ensure that
    /// stack references made by `f` are handled appropriately.
    ///
    /// It is invalid to call this function with a thread that has been previously
    /// destroyed via a failed call to `run`.
    pub fn run(mut self: Box<Task>, f: ||) -> Box<Task> {
        assert!(!self.is_destroyed(), "cannot re-use a destroyed thread");

        // First, make sure that no one else is in TLS. This does not allow
        // recursive invocations of run(). If there's no one else, then
        // relinquish ownership of ourselves back into TLS.
        if Local::exists(None::<Task>) {
            panic!("cannot run a thread recursively inside another");
        }
        self.state = Armed;
        Local::put(self);

        // There are two primary reasons that general try/catch is unsafe. The
        // first is that we do not support nested try/catch. The above check for
        // an existing thread in TLS is sufficient for this invariant to be
        // upheld. The second is that unwinding while unwinding is not defined.
        // We take care of that by having an 'unwinding' flag in the thread
        // itself. For these reasons, this unsafety should be ok.
        let result = unsafe { unwind::try(f) };

        // After running the closure given return the thread back out if it ran
        // successfully, or clean up the thread if it panicked.
        let task: Box<Task> = Local::take();
        match result {
            Ok(()) => task,
            Err(cause) => { task.cleanup(Err(cause)) }
        }
    }

    /// Destroy all associated resources of this thread.
    ///
    /// This function will perform any necessary clean up to prepare the thread
    /// for destruction. It is required that this is called before a `Task`
    /// falls out of scope.
    ///
    /// The returned thread cannot be used for running any more code, but it may
    /// be used to extract the runtime as necessary.
    pub fn destroy(self: Box<Task>) -> Box<Task> {
        if self.is_destroyed() {
            self
        } else {
            self.cleanup(Ok(()))
        }
    }

    /// Cleans up a thread, processing the result of the thread as appropriate.
    ///
    /// This function consumes ownership of the thread, deallocating it once it's
    /// done being processed. It is assumed that TLD and the local heap have
    /// already been destroyed and/or annihilated.
    fn cleanup(mut self: Box<Task>, result: Result) -> Box<Task> {
        // After taking care of the data above, we need to transmit the result
        // of this thread.
        let what_to_do = self.death.on_exit.take();
        Local::put(self);

        // FIXME: this is running in a seriously constrained context. If this
        //        allocates TLD then it will likely abort the runtime. Similarly,
        //        if this panics, this will also likely abort the runtime.
        //
        //        This closure is currently limited to a channel send via the
        //        standard library's thread interface, but this needs
        //        reconsideration to whether it's a reasonable thing to let a
        //        thread to do or not.
        match what_to_do {
            Some(f) => { f.invoke(result) }
            None => { drop(result) }
        }

        // Now that we're done, we remove the thread from TLS and flag it for
        // destruction.
        let mut task: Box<Task> = Local::take();
        task.state = Destroyed;
        return task;
    }

    /// Queries whether this can be destroyed or not.
    pub fn is_destroyed(&self) -> bool { self.state == Destroyed }

    /// Deschedules the current thread, invoking `f` `amt` times. It is not
    /// recommended to use this function directly, but rather communication
    /// primitives in `std::comm` should be used.
    //
    // This function gets a little interesting. There are a few safety and
    // ownership violations going on here, but this is all done in the name of
    // shared state. Additionally, all of the violations are protected with a
    // mutex, so in theory there are no races.
    //
    // The first thing we need to do is to get a pointer to the thread's internal
    // mutex. This address will not be changing (because the thread is allocated
    // on the heap). We must have this handle separately because the thread will
    // have its ownership transferred to the given closure. We're guaranteed,
    // however, that this memory will remain valid because *this* is the current
    // thread's execution thread.
    //
    // The next weird part is where ownership of the thread actually goes. We
    // relinquish it to the `f` blocking function, but upon returning this
    // function needs to replace the thread back in TLS. There is no communication
    // from the wakeup thread back to this thread about the thread pointer, and
    // there's really no need to. In order to get around this, we cast the thread
    // to a `uint` which is then used at the end of this function to cast back
    // to a `Box<Task>` object. Naturally, this looks like it violates
    // ownership semantics in that there may be two `Box<Task>` objects.
    //
    // The fun part is that the wakeup half of this implementation knows to
    // "forget" the thread on the other end. This means that the awakening half of
    // things silently relinquishes ownership back to this thread, but not in a
    // way that the compiler can understand. The thread's memory is always valid
    // for both threads because these operations are all done inside of a mutex.
    //
    // You'll also find that if blocking fails (the `f` function hands the
    // BlockedTask back to us), we will `mem::forget` the handles. The
    // reasoning for this is the same logic as above in that the thread silently
    // transfers ownership via the `uint`, not through normal compiler
    // semantics.
    //
    // On a mildly unrelated note, it should also be pointed out that OS
    // condition variables are susceptible to spurious wakeups, which we need to
    // be ready for. In order to accommodate for this fact, we have an extra
    // `awoken` field which indicates whether we were actually woken up via some
    // invocation of `reawaken`. This flag is only ever accessed inside the
    // lock, so there's no need to make it atomic.
    pub fn deschedule<F>(mut self: Box<Task>, times: uint, mut f: F) where
        F: FnMut(BlockedTask) -> ::core::result::Result<(), BlockedTask>,
    {
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

                // Apply the given closure to all of the "selectable threads",
                // bailing on the first one that produces an error. Note that
                // care must be taken such that when an error is occurred, we
                // may not own the thread, so we may still have to wait for the
                // thread to become available. In other words, if thread.wake()
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
            // put the thread back in TLS, and everything is as it once was.
            Local::put(mem::transmute(me));
        }
    }

    /// Wakes up a previously blocked thread. This function can only be
    /// called on threads that were previously blocked in `deschedule`.
    //
    // See the comments on `deschedule` for why the thread is forgotten here, and
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

    /// Yields control of this thread to another thread. This function will
    /// eventually return, but possibly not immediately. This is used as an
    /// opportunity to allow other threads a chance to run.
    pub fn yield_now() {
        Thread::yield_now();
    }

    /// Returns the stack bounds for this thread in (lo, hi) format. The stack
    /// bounds may not be known for all threads, so the return value may be
    /// `None`.
    pub fn stack_bounds(&self) -> (uint, uint) {
        self.stack_bounds
    }

    /// Returns the stack guard for this thread, if known.
    pub fn stack_guard(&self) -> Option<uint> {
        if self.stack_guard != 0 {
            Some(self.stack_guard)
        } else {
            None
        }
    }

    /// Consume this thread, flagging it as a candidate for destruction.
    ///
    /// This function is required to be invoked to destroy a thread. A thread
    /// destroyed through a normal drop will abort.
    pub fn drop(mut self) {
        self.state = Destroyed;
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        rtdebug!("called drop for a thread: {}", self as *mut Task as uint);
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
    /// Returns Some if the thread was successfully woken; None if already killed.
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

    /// Reawakens this thread if ownership is acquired. If finer-grained control
    /// is desired, use `wake` instead.
    pub fn reawaken(self) {
        self.wake().map(|t| t.reawaken());
    }

    // This assertion has two flavours because the wake involves an atomic op.
    // In the faster version, destructors will panic dramatically instead.
    #[cfg(not(test))] pub fn trash(self) { }
    #[cfg(test)]      pub fn trash(self) { assert!(self.wake().is_none()); }

    /// Create a blocked thread, unless the thread was already killed.
    pub fn block(task: Box<Task>) -> BlockedTask {
        Owned(task)
    }

    /// Converts one blocked thread handle to a list of many handles to the same.
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
        Death { on_exit: None }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use prelude::*;
    use task;
    use rt::unwind;

    #[test]
    fn unwind() {
        let result = task::try(move|| ());
        rtdebug!("trying first assert");
        assert!(result.is_ok());
        let result = task::try(move|| -> () panic!());
        rtdebug!("trying second assert");
        assert!(result.is_err());
    }

    #[test]
    fn rng() {
        use rand::{StdRng, Rng};
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
        use rt::unwind::begin_unwind;
        begin_unwind("cause", &(file!(), line!()))
    }

    #[test]
    fn drop_new_task_ok() {
        drop(Task::new(None, None));
    }

    // Thread blocking tests

    #[test]
    fn block_and_wake() {
        let task = box Task::new(None, None);
        let task = BlockedTask::block(task).wake().unwrap();
        task.drop();
    }
}
