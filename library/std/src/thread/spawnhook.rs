use crate::io;
use crate::sync::RwLock;
use crate::thread::Thread;

static SPAWN_HOOKS: RwLock<
    Vec<&'static (dyn Fn(&Thread) -> io::Result<Box<dyn FnOnce() + Send>> + Sync)>,
> = RwLock::new(Vec::new());

/// Registers a function to run for every new thread spawned.
///
/// The hook is executed in the parent thread, and returns a function
/// that will be executed in the new thread.
///
/// The hook is called with the `Thread` handle for the new thread.
///
/// If the hook returns an `Err`, thread spawning is aborted. In that case, the
/// function used to spawn the thread (e.g. `std::thread::spawn`) will return
/// the error returned by the hook.
///
/// Hooks can only be added, not removed.
///
/// The hooks will run in order, starting with the most recently added.
///
/// # Usage
///
/// ```
/// #![feature(thread_spawn_hook)]
///
/// std::thread::add_spawn_hook(|_| {
///     ..; // This will run in the parent (spawning) thread.
///     Ok(move || {
///         ..; // This will run it the child (spawned) thread.
///     })
/// });
/// ```
///
/// # Example
///
/// A spawn hook can be used to initialize thread locals from the parent thread:
///
/// ```
/// #![feature(thread_spawn_hook)]
///
/// use std::cell::Cell;
///
/// thread_local! {
///     static X: Cell<u32> = Cell::new(0);
/// }
///
/// std::thread::add_spawn_hook(|_| {
///     // Get the value of X in the spawning thread.
///     let value = X.get();
///     // Set the value of X in the newly spawned thread.
///     Ok(move || {
///         X.set(value);
///     })
/// });
///
/// X.set(123);
///
/// std::thread::spawn(|| {
///     assert_eq!(X.get(), 123);
/// }).join().unwrap();
/// ```
#[unstable(feature = "thread_spawn_hook", issue = "none")]
pub fn add_spawn_hook<F, G>(hook: F)
where
    F: 'static + Sync + Fn(&Thread) -> io::Result<G>,
    G: 'static + Send + FnOnce(),
{
    SPAWN_HOOKS.write().unwrap_or_else(|e| e.into_inner()).push(Box::leak(Box::new(
        move |thread: &Thread| -> io::Result<_> {
            let f: Box<dyn FnOnce() + Send> = Box::new(hook(thread)?);
            Ok(f)
        },
    )));
}

/// Runs all the spawn hooks.
///
/// Called on the parent thread.
///
/// Returns the functions to be called on the newly spawned thread.
pub(super) fn run_spawn_hooks(thread: &Thread) -> io::Result<Vec<Box<dyn FnOnce() + Send>>> {
    SPAWN_HOOKS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .rev()
        .map(|hook| hook(thread))
        .collect()
}
