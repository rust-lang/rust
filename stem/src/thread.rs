use crate::errors::Errno;
use crate::stack::{Stack, StackSpec};
use abi::wait::{WaitKind, WaitSpec};
use alloc::boxed::Box;

pub type ThreadId = u64;

pub fn spawn(entry: extern "C" fn() -> !) -> Result<ThreadId, Errno> {
    let stack = Stack::alloc_growing_stack(StackSpec::default())?;
    spawn_on(stack, entry)
}

pub fn spawn_on(stack: Stack, entry: extern "C" fn() -> !) -> Result<ThreadId, Errno> {
    crate::syscall::spawn_thread(entry as usize, 0, &stack).map(|id| id as ThreadId)
}

pub fn spawn_with_stack(stack: Stack, entry: extern "C" fn() -> !) -> Result<ThreadId, Errno> {
    spawn_on(stack, entry)
}

pub fn spawn_with_arg(entry: extern "C" fn(usize) -> !, arg: usize) -> Result<ThreadId, Errno> {
    let stack = Stack::alloc_growing_stack(StackSpec::default())?;
    crate::syscall::spawn_thread(entry as usize, arg, &stack).map(|id| id as ThreadId)
}

/// Spawn a thread with an explicit initial TLS base (e.g. FS_BASE on x86_64).
///
/// The kernel writes `tls_base` into the hardware TLS register before the
/// new thread is first scheduled, giving the thread a deterministic TLS
/// starting state without requiring a separate `SYS_TASK_SET_TLS_BASE`
/// syscall on the thread's own entry path.
///
/// Pass `0` for `tls_base` to leave the register in its default initial state
/// (equivalent to [`spawn_with_arg`]).
pub fn spawn_with_tls(
    entry: extern "C" fn(usize) -> !,
    arg: usize,
    tls_base: usize,
) -> Result<ThreadId, Errno> {
    let stack = Stack::alloc_growing_stack(StackSpec::default())?;
    crate::syscall::spawn_thread_ex(entry as usize, arg, &stack, tls_base, 0)
        .map(|id| id as ThreadId)
}

pub fn yield_now() {
    crate::syscall::yield_now();
}

/// Block until the specified task exits, returning its exit code.
pub fn wait(tid: ThreadId) -> Result<i32, Errno> {
    crate::syscall::task_wait(tid)
}

pub fn set_priority(tid: ThreadId, priority: usize) -> Result<(), Errno> {
    crate::syscall::set_priority(tid, priority)
}

/// Set the calling thread's human-readable name.
///
/// The name is stored in the kernel's thread record and is visible via
/// `/proc/<pid>/task/<tid>/name`.  Names longer than 31 bytes are
/// silently truncated by the kernel.
pub fn set_name(name: &[u8]) -> Result<(), Errno> {
    crate::syscall::task_set_name(name)
}

/// A first-class handle for observing a task's exit via `wait_many`.
///
/// Placing a `TaskExitWatch` in a `wait_many` call eliminates the need for
/// a dedicated polling loop around [`wait()`]. The result's `value` field
/// carries the exit code of the exited task.
///
/// # Example
///
/// ```ignore
/// let tid = stem::thread::spawn_with_arg(worker, arg)?;
/// let watcher = TaskExitWatch::new(tid);
///
/// let specs = [watcher.wait_spec(0)];
/// let mut results = [WaitResult::default()];
/// stem::syscall::wait_many(&specs, &mut results, None)?;
/// let exit_code = results[0].value as i32;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskExitWatch {
    tid: ThreadId,
}

impl TaskExitWatch {
    /// Create a watch for the given thread ID.
    pub fn new(tid: ThreadId) -> Self {
        Self { tid }
    }

    /// Returns the thread ID being watched.
    pub fn tid(&self) -> ThreadId {
        self.tid
    }

    /// Returns a [`WaitSpec`] that fires when the watched task exits.
    ///
    /// Pass the returned spec to `stem::syscall::wait_many` alongside any other
    /// specs (ports, graph watches, timeouts, etc.) to multiplex without blocking.
    /// When the spec fires, `WaitResult::value` contains the task's exit code and
    /// `WaitResult::flags` has [`abi::wait::ready::EXITED`](abi::wait::ready::EXITED) set.
    pub fn wait_spec(&self, token: u64) -> WaitSpec {
        WaitSpec {
            kind: WaitKind::TaskExit as u32,
            flags: 0,
            object: self.tid,
            token,
        }
    }
}

struct BoxWrapper<F> {
    f: F,
}

extern "C" fn generic_thread_trampoline<F>(arg: usize) -> !
where
    F: FnOnce() + Send + 'static,
{
    // Reconstruct the Box and take ownership of the closure
    let b = unsafe { Box::from_raw(arg as *mut BoxWrapper<F>) };

    // Execute the closure
    (b.f)();

    // Exit securely with status 0 upon successful completion
    crate::syscall::exit(0);
}

/// A handle to a spawned background thread.
#[derive(Debug)]
pub struct JoinHandle {
    tid: ThreadId,
}

impl JoinHandle {
    /// Wait for the thread to exit and return its exit code.
    pub fn join(self) -> Result<i32, Errno> {
        wait(self.tid)
    }

    /// Returns the thread ID.
    pub fn tid(&self) -> ThreadId {
        self.tid
    }

    /// Detach the thread, relinquishing the ability to join it.
    ///
    /// After calling `detach`, the thread runs independently: its resources
    /// are reclaimed automatically when it exits.  Attempting to join a
    /// detached TID via [`wait`] will return `EINVAL`.
    ///
    /// This method consumes the handle to make the detachment explicit.
    pub fn detach(self) {
        // Nothing to do on the caller side; the kernel flag is already set
        // if the thread was spawned detached.  For handles created via
        // spawn_task (which are joinable by default) we simply drop the
        // handle without joining — the thread continues running.
        core::mem::forget(self);
    }
}

/// Spawns a new thread, executing the given closure.
///
/// This provides an ergonomic, `std::thread`-like API for starting background tasks.
/// The closure will execute on a dynamically allocated stack and exit with code 0 natively.
/// The returned [`JoinHandle`] can be used to wait for the thread to finish.
pub fn spawn_task<F>(f: F) -> Result<JoinHandle, Errno>
where
    F: FnOnce() + Send + 'static,
{
    let b = Box::new(BoxWrapper { f });
    let ptr = Box::into_raw(b) as usize;

    let stack = Stack::alloc_growing_stack(StackSpec::default())?;

    let tid = crate::syscall::spawn_thread(
        generic_thread_trampoline::<F> as *const () as usize,
        ptr,
        &stack,
    )
    .map(|id| id as ThreadId)?;

    Ok(JoinHandle { tid })
}

/// Spawns a new **detached** thread, executing the given closure.
///
/// A detached thread cannot be joined: its kernel resources are reclaimed
/// automatically when it exits.  Use this when you want fire-and-forget
/// background work without holding a [`JoinHandle`].
pub fn spawn_task_detached<F>(f: F) -> Result<ThreadId, Errno>
where
    F: FnOnce() + Send + 'static,
{
    let b = Box::new(BoxWrapper { f });
    let ptr = Box::into_raw(b) as usize;

    let stack = Stack::alloc_growing_stack(StackSpec::default())?;

    crate::syscall::spawn_thread_ex(
        generic_thread_trampoline::<F> as *const () as usize,
        ptr,
        &stack,
        0,
        abi::types::spawn_thread_flags::DETACHED,
    )
    .map(|id| id as ThreadId)
}
