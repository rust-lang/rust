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
//! to be available 'everywhere'. Local heaps, GC, unwinding,
//! local storage, and logging. Even a 'freestanding' Rust would likely want
//! to implement this.

use core::prelude::*;

use alloc::arc::Arc;
use alloc::boxed::{BoxAny, Box};
use core::any::Any;
use core::atomics::{AtomicUint, SeqCst};
use core::iter::Take;
use core::kinds::marker;
use core::mem;
use core::raw;

use local_data;
use Runtime;
use local::Local;
use local_heap::LocalHeap;
use rtio::LocalIo;
use unwind;
use unwind::Unwinder;
use collections::str::SendStr;

/// State associated with Rust tasks.
///
/// Rust tasks are primarily built with two separate components. One is this
/// structure which handles standard services such as TLD, unwinding support,
/// naming of a task, etc. The second component is the runtime of this task, a
/// `Runtime` trait object.
///
/// The `Runtime` object instructs this task how it can perform critical
/// operations such as blocking, rescheduling, I/O constructors, etc. The two
/// halves are separately owned, but one is often found contained in the other.
/// A task's runtime can be reflected upon with the `maybe_take_runtime` method,
/// and otherwise its ownership is managed with `take_runtime` and
/// `put_runtime`.
///
/// In general, this structure should not be used. This is meant to be an
/// unstable internal detail of the runtime itself. From time-to-time, however,
/// it is useful to manage tasks directly. An example of this would be
/// interoperating with the Rust runtime from FFI callbacks or such. For this
/// reason, there are two methods of note with the `Task` structure.
///
/// * `run` - This function will execute a closure inside the context of a task.
///           Failure is caught and handled via the task's on_exit callback. If
///           this fails, the task is still returned, but it can no longer be
///           used, it is poisoned.
///
/// * `destroy` - This is a required function to call to destroy a task. If a
///               task falls out of scope without calling `destroy`, its
///               destructor bomb will go off, aborting the process.
///
/// With these two methods, tasks can be re-used to execute code inside of its
/// context while having a point in the future where destruction is allowed.
/// More information can be found on these specific methods.
///
/// # Example
///
/// ```no_run
/// extern crate native;
/// use std::uint;
/// # fn main() {
///
/// // Create a task using a native runtime
/// let task = native::task::new((0, uint::MAX));
///
/// // Run some code, catching any possible failures
/// let task = task.run(|| {
///     // Run some code inside this task
///     println!("Hello with a native runtime!");
/// });
///
/// // Run some code again, catching the failure
/// let task = task.run(|| {
///     fail!("oh no, what to do!");
/// });
///
/// // Now that the task is failed, it can never be used again
/// assert!(task.is_destroyed());
///
/// // Deallocate the resources associated with this task
/// task.destroy();
/// # }
/// ```
pub struct Task {
    pub heap: LocalHeap,
    pub gc: GarbageCollector,
    pub storage: LocalStorage,
    pub unwinder: Unwinder,
    pub death: Death,
    pub destroyed: bool,
    pub name: Option<SendStr>,

    imp: Option<Box<Runtime + Send>>,
}

pub struct TaskOpts {
    /// Invoke this procedure with the result of the task when it finishes.
    pub on_exit: Option<proc(Result): Send>,
    /// A name for the task-to-be, for identification in failure messages
    pub name: Option<SendStr>,
    /// The size of the stack for the spawned task
    pub stack_size: Option<uint>,
}

/// Indicates the manner in which a task exited.
///
/// A task that completes without failing is considered to exit successfully.
///
/// If you wish for this result's delivery to block until all
/// children tasks complete, recommend using a result future.
pub type Result = ::core::result::Result<(), Box<Any + Send>>;

pub struct GarbageCollector;
pub struct LocalStorage(pub Option<local_data::Map>);

/// A handle to a blocked task. Usually this means having the Box<Task>
/// pointer by ownership, but if the task is killable, a killer can steal it
/// at any time.
pub enum BlockedTask {
    Owned(Box<Task>),
    Shared(Arc<AtomicUint>),
}

/// Per-task state related to task death, killing, failure, etc.
pub struct Death {
    pub on_exit: Option<proc(Result):Send>,
    marker: marker::NoCopy,
}

pub struct BlockedTasks {
    inner: Arc<AtomicUint>,
}

impl Task {
    /// Creates a new uninitialized task.
    ///
    /// This method cannot be used to immediately invoke `run` because the task
    /// itself will likely require a runtime to be inserted via `put_runtime`.
    ///
    /// Note that you likely don't want to call this function, but rather the
    /// task creation functions through libnative or libgreen.
    pub fn new() -> Task {
        Task {
            heap: LocalHeap::new(),
            gc: GarbageCollector,
            storage: LocalStorage(None),
            unwinder: Unwinder::new(),
            death: Death::new(),
            destroyed: false,
            name: None,
            imp: None,
        }
    }

    /// Consumes ownership of a task, runs some code, and returns the task back.
    ///
    /// This function can be used as an emulated "try/catch" to interoperate
    /// with the rust runtime at the outermost boundary. It is not possible to
    /// use this function in a nested fashion (a try/catch inside of another
    /// try/catch). Invoking this function is quite cheap.
    ///
    /// If the closure `f` succeeds, then the returned task can be used again
    /// for another invocation of `run`. If the closure `f` fails then `self`
    /// will be internally destroyed along with all of the other associated
    /// resources of this task. The `on_exit` callback is invoked with the
    /// cause of failure (not returned here). This can be discovered by querying
    /// `is_destroyed()`.
    ///
    /// Note that it is possible to view partial execution of the closure `f`
    /// because it is not guaranteed to run to completion, but this function is
    /// guaranteed to return if it fails. Care should be taken to ensure that
    /// stack references made by `f` are handled appropriately.
    ///
    /// It is invalid to call this function with a task that has been previously
    /// destroyed via a failed call to `run`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// extern crate native;
    /// use std::uint;
    /// # fn main() {
    ///
    /// // Create a new native task
    /// let task = native::task::new((0, uint::MAX));
    ///
    /// // Run some code once and then destroy this task
    /// task.run(|| {
    ///     println!("Hello with a native runtime!");
    /// }).destroy();
    /// # }
    /// ```
    pub fn run(~self, f: ||) -> Box<Task> {
        assert!(!self.is_destroyed(), "cannot re-use a destroyed task");

        // First, make sure that no one else is in TLS. This does not allow
        // recursive invocations of run(). If there's no one else, then
        // relinquish ownership of ourselves back into TLS.
        if Local::exists(None::<Task>) {
            fail!("cannot run a task recursively inside another");
        }
        Local::put(self);

        // There are two primary reasons that general try/catch is unsafe. The
        // first is that we do not support nested try/catch. The above check for
        // an existing task in TLS is sufficient for this invariant to be
        // upheld. The second is that unwinding while unwinding is not defined.
        // We take care of that by having an 'unwinding' flag in the task
        // itself. For these reasons, this unsafety should be ok.
        let result = unsafe { unwind::try(f) };

        // After running the closure given return the task back out if it ran
        // successfully, or clean up the task if it failed.
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
    pub fn destroy(~self) -> Box<Task> {
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
    fn cleanup(~self, result: Result) -> Box<Task> {
        // The first thing to do when cleaning up is to deallocate our local
        // resources, such as TLD and GC data.
        //
        // FIXME: there are a number of problems with this code
        //
        // 1. If any TLD object fails destruction, then all of TLD will leak.
        //    This appears to be a consequence of #14875.
        //
        // 2. Failing during GC annihilation aborts the runtime #14876.
        //
        // 3. Setting a TLD key while destroying TLD or while destroying GC will
        //    abort the runtime #14807.
        //
        // 4. Invoking GC in GC destructors will abort the runtime #6996.
        //
        // 5. The order of destruction of TLD and GC matters, but either way is
        //    susceptible to leaks (see 3/4) #8302.
        //
        // That being said, there are a few upshots to this code
        //
        // 1. If TLD destruction fails, heap destruction will be attempted.
        //    There is a test for this at fail-during-tld-destroy.rs. Sadly the
        //    other way can't be tested due to point 2 above. Note that we must
        //    immortalize the heap first because if any deallocations are
        //    attempted while TLD is being dropped it will attempt to free the
        //    allocation from the wrong heap (because the current one has been
        //    replaced).
        //
        // 2. One failure in destruction is tolerable, so long as the task
        //    didn't originally fail while it was running.
        //
        // And with all that in mind, we attempt to clean things up!
        let mut task = self.run(|| {
            let mut task = Local::borrow(None::<Task>);
            let tld = {
                let &LocalStorage(ref mut optmap) = &mut task.storage;
                optmap.take()
            };
            let mut heap = mem::replace(&mut task.heap, LocalHeap::new());
            unsafe { heap.immortalize() }
            drop(task);

            // First, destroy task-local storage. This may run user dtors.
            drop(tld);

            // Destroy remaining boxes. Also may run user dtors.
            drop(heap);
        });

        // If the above `run` block failed, then it must be the case that the
        // task had previously succeeded. This also means that the code below
        // was recursively run via the `run` method invoking this method. In
        // this case, we just make sure the world is as we thought, and return.
        if task.is_destroyed() {
            rtassert!(result.is_ok())
            return task
        }

        // After taking care of the data above, we need to transmit the result
        // of this task.
        let what_to_do = task.death.on_exit.take();
        Local::put(task);

        // FIXME: this is running in a seriously constrained context. If this
        //        allocates GC or allocates TLD then it will likely abort the
        //        runtime. Similarly, if this fails, this will also likely abort
        //        the runtime.
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
        task.destroyed = true;
        return task;
    }

    /// Queries whether this can be destroyed or not.
    pub fn is_destroyed(&self) -> bool { self.destroyed }

    /// Inserts a runtime object into this task, transferring ownership to the
    /// task. It is illegal to replace a previous runtime object in this task
    /// with this argument.
    pub fn put_runtime(&mut self, ops: Box<Runtime + Send>) {
        assert!(self.imp.is_none());
        self.imp = Some(ops);
    }

    /// Removes the runtime from this task, transferring ownership to the
    /// caller.
    pub fn take_runtime(&mut self) -> Box<Runtime + Send> {
        assert!(self.imp.is_some());
        self.imp.take().unwrap()
    }

    /// Attempts to extract the runtime as a specific type. If the runtime does
    /// not have the provided type, then the runtime is not removed. If the
    /// runtime does have the specified type, then it is removed and returned
    /// (transfer of ownership).
    ///
    /// It is recommended to only use this method when *absolutely necessary*.
    /// This function may not be available in the future.
    pub fn maybe_take_runtime<T: 'static>(&mut self) -> Option<Box<T>> {
        // This is a terrible, terrible function. The general idea here is to
        // take the runtime, cast it to Box<Any>, check if it has the right
        // type, and then re-cast it back if necessary. The method of doing
        // this is pretty sketchy and involves shuffling vtables of trait
        // objects around, but it gets the job done.
        //
        // FIXME: This function is a serious code smell and should be avoided at
        //      all costs. I have yet to think of a method to avoid this
        //      function, and I would be saddened if more usage of the function
        //      crops up.
        unsafe {
            let imp = self.imp.take_unwrap();
            let vtable = mem::transmute::<_, &raw::TraitObject>(&imp).vtable;
            match imp.wrap().downcast::<T>() {
                Ok(t) => Some(t),
                Err(t) => {
                    let data = mem::transmute::<_, raw::TraitObject>(t).data;
                    let obj: Box<Runtime + Send> =
                        mem::transmute(raw::TraitObject {
                            vtable: vtable,
                            data: data,
                        });
                    self.put_runtime(obj);
                    None
                }
            }
        }
    }

    /// Spawns a sibling to this task. The newly spawned task is configured with
    /// the `opts` structure and will run `f` as the body of its code.
    pub fn spawn_sibling(mut ~self, opts: TaskOpts, f: proc(): Send) {
        let ops = self.imp.take_unwrap();
        ops.spawn_sibling(self, opts, f)
    }

    /// Deschedules the current task, invoking `f` `amt` times. It is not
    /// recommended to use this function directly, but rather communication
    /// primitives in `std::comm` should be used.
    pub fn deschedule(mut ~self, amt: uint,
                      f: |BlockedTask| -> ::core::result::Result<(), BlockedTask>) {
        let ops = self.imp.take_unwrap();
        ops.deschedule(amt, self, f)
    }

    /// Wakes up a previously blocked task, optionally specifying whether the
    /// current task can accept a change in scheduling. This function can only
    /// be called on tasks that were previously blocked in `deschedule`.
    pub fn reawaken(mut ~self) {
        let ops = self.imp.take_unwrap();
        ops.reawaken(self);
    }

    /// Yields control of this task to another task. This function will
    /// eventually return, but possibly not immediately. This is used as an
    /// opportunity to allow other tasks a chance to run.
    pub fn yield_now(mut ~self) {
        let ops = self.imp.take_unwrap();
        ops.yield_now(self);
    }

    /// Similar to `yield_now`, except that this function may immediately return
    /// without yielding (depending on what the runtime decides to do).
    pub fn maybe_yield(mut ~self) {
        let ops = self.imp.take_unwrap();
        ops.maybe_yield(self);
    }

    /// Acquires a handle to the I/O factory that this task contains, normally
    /// stored in the task's runtime. This factory may not always be available,
    /// which is why the return type is `Option`
    pub fn local_io<'a>(&'a mut self) -> Option<LocalIo<'a>> {
        self.imp.get_mut_ref().local_io()
    }

    /// Returns the stack bounds for this task in (lo, hi) format. The stack
    /// bounds may not be known for all tasks, so the return value may be
    /// `None`.
    pub fn stack_bounds(&self) -> (uint, uint) {
        self.imp.get_ref().stack_bounds()
    }

    /// Returns whether it is legal for this task to block the OS thread that it
    /// is running on.
    pub fn can_block(&self) -> bool {
        self.imp.get_ref().can_block()
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        rtdebug!("called drop for a task: {}", self as *mut Task as uint);
        rtassert!(self.destroyed);
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
    // In the faster version, destructors will fail dramatically instead.
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
    use std::gc::{Gc, GC};

    #[test]
    fn local_heap() {
        let a = box(GC) 5i;
        let b = a;
        assert!(*a == 5);
        assert!(*b == 5);
    }

    #[test]
    fn tls() {
        local_data_key!(key: Gc<String>)
        key.replace(Some(box(GC) "data".to_string()));
        assert_eq!(key.get().unwrap().as_slice(), "data");
        local_data_key!(key2: Gc<String>)
        key2.replace(Some(box(GC) "data".to_string()));
        assert_eq!(key2.get().unwrap().as_slice(), "data");
    }

    #[test]
    fn unwind() {
        let result = task::try(proc()());
        rtdebug!("trying first assert");
        assert!(result.is_ok());
        let result = task::try::<()>(proc() fail!());
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
    fn heap_cycles() {
        use std::cell::RefCell;

        struct List {
            next: Option<Gc<RefCell<List>>>,
        }

        let a = box(GC) RefCell::new(List { next: None });
        let b = box(GC) RefCell::new(List { next: Some(a) });

        {
            let mut a = a.borrow_mut();
            a.next = Some(b);
        }
    }

    #[test]
    #[should_fail]
    fn test_begin_unwind() {
        use std::rt::unwind::begin_unwind;
        begin_unwind("cause", file!(), line!())
    }

    // Task blocking tests

    #[test]
    fn block_and_wake() {
        let task = box Task::new();
        let mut task = BlockedTask::block(task).wake().unwrap();
        task.destroyed = true;
    }
}
