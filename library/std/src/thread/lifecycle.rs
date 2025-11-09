//! The inner logic for thread spawning and joining.

use super::current::set_current;
use super::id::ThreadId;
use super::scoped::ScopeData;
use super::thread::Thread;
use super::{Result, spawnhook};
use crate::cell::UnsafeCell;
use crate::marker::PhantomData;
use crate::mem::{ManuallyDrop, MaybeUninit};
use crate::sync::Arc;
use crate::sync::atomic::{Atomic, AtomicUsize, Ordering};
use crate::sys::thread as imp;
use crate::sys_common::{AsInner, IntoInner};
use crate::{env, io, panic};

#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub(super) unsafe fn spawn_unchecked<'scope, F, T>(
    name: Option<String>,
    stack_size: Option<usize>,
    no_hooks: bool,
    scope_data: Option<Arc<ScopeData>>,
    f: F,
) -> io::Result<JoinInner<'scope, T>>
where
    F: FnOnce() -> T,
    F: Send,
    T: Send,
{
    let stack_size = stack_size.unwrap_or_else(|| {
        static MIN: Atomic<usize> = AtomicUsize::new(0);

        match MIN.load(Ordering::Relaxed) {
            0 => {}
            n => return n - 1,
        }

        let amt = env::var_os("RUST_MIN_STACK")
            .and_then(|s| s.to_str().and_then(|s| s.parse().ok()))
            .unwrap_or(imp::DEFAULT_MIN_STACK_SIZE);

        // 0 is our sentinel value, so ensure that we'll never see 0 after
        // initialization has run
        MIN.store(amt + 1, Ordering::Relaxed);
        amt
    });

    let id = ThreadId::new();
    let thread = Thread::new(id, name);

    let hooks = if no_hooks {
        spawnhook::ChildSpawnHooks::default()
    } else {
        spawnhook::run_spawn_hooks(&thread)
    };

    let my_packet: Arc<Packet<'scope, T>> =
        Arc::new(Packet { scope: scope_data, result: UnsafeCell::new(None), _marker: PhantomData });
    let their_packet = my_packet.clone();

    // Pass `f` in `MaybeUninit` because actually that closure might *run longer than the lifetime of `F`*.
    // See <https://github.com/rust-lang/rust/issues/101983> for more details.
    // To prevent leaks we use a wrapper that drops its contents.
    #[repr(transparent)]
    struct MaybeDangling<T>(MaybeUninit<T>);
    impl<T> MaybeDangling<T> {
        fn new(x: T) -> Self {
            MaybeDangling(MaybeUninit::new(x))
        }
        fn into_inner(self) -> T {
            // Make sure we don't drop.
            let this = ManuallyDrop::new(self);
            // SAFETY: we are always initialized.
            unsafe { this.0.assume_init_read() }
        }
    }
    impl<T> Drop for MaybeDangling<T> {
        fn drop(&mut self) {
            // SAFETY: we are always initialized.
            unsafe { self.0.assume_init_drop() };
        }
    }

    let f = MaybeDangling::new(f);

    // The entrypoint of the Rust thread, after platform-specific thread
    // initialization is done.
    let rust_start = move || {
        let f = f.into_inner();
        let try_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            crate::sys::backtrace::__rust_begin_short_backtrace(|| hooks.run());
            crate::sys::backtrace::__rust_begin_short_backtrace(f)
        }));
        // SAFETY: `their_packet` as been built just above and moved by the
        // closure (it is an Arc<...>) and `my_packet` will be stored in the
        // same `JoinInner` as this closure meaning the mutation will be
        // safe (not modify it and affect a value far away).
        unsafe { *their_packet.result.get() = Some(try_result) };
        // Here `their_packet` gets dropped, and if this is the last `Arc` for that packet that
        // will call `decrement_num_running_threads` and therefore signal that this thread is
        // done.
        drop(their_packet);
        // Here, the lifetime `'scope` can end. `main` keeps running for a bit
        // after that before returning itself.
    };

    if let Some(scope_data) = &my_packet.scope {
        scope_data.increment_num_running_threads();
    }

    // SAFETY: dynamic size and alignment of the Box remain the same. See below for why the
    // lifetime change is justified.
    let rust_start = unsafe {
        Box::from_raw(Box::into_raw(Box::new(rust_start)) as *mut (dyn FnOnce() + Send + 'static))
    };

    let init = Box::new(ThreadInit { handle: thread.clone(), rust_start });

    Ok(JoinInner {
        // SAFETY:
        //
        // `imp::Thread::new` takes a closure with a `'static` lifetime, since it's passed
        // through FFI or otherwise used with low-level threading primitives that have no
        // notion of or way to enforce lifetimes.
        //
        // As mentioned in the `Safety` section of this function's documentation, the caller of
        // this function needs to guarantee that the passed-in lifetime is sufficiently long
        // for the lifetime of the thread.
        //
        // Similarly, the `sys` implementation must guarantee that no references to the closure
        // exist after the thread has terminated, which is signaled by `Thread::join`
        // returning.
        native: unsafe { imp::Thread::new(stack_size, init)? },
        thread,
        packet: my_packet,
    })
}

/// The data passed to the spawned thread for thread initialization. Any thread
/// implementation should start a new thread by calling .init() on this before
/// doing anything else to ensure the current thread is properly initialized and
/// the global allocator works.
pub(crate) struct ThreadInit {
    pub handle: Thread,
    pub rust_start: Box<dyn FnOnce() + Send>,
}

impl ThreadInit {
    /// Initialize the 'current thread' mechanism on this thread, returning the
    /// Rust entry point.
    pub fn init(self: Box<Self>) -> Box<dyn FnOnce() + Send> {
        // Set the current thread before any (de)allocations on the global allocator occur,
        // so that it may call std::thread::current() in its implementation. This is also
        // why we take Box<Self>, to ensure the Box is not destroyed until after this point.
        // Cloning the handle does not invoke the global allocator, it is an Arc.
        if let Err(_thread) = set_current(self.handle.clone()) {
            // The current thread should not have set yet. Use an abort to save binary size (see #123356).
            rtabort!("current thread handle already set during thread spawn");
        }

        if let Some(name) = self.handle.cname() {
            imp::set_name(name);
        }

        self.rust_start
    }
}

// This packet is used to communicate the return value between the spawned
// thread and the rest of the program. It is shared through an `Arc` and
// there's no need for a mutex here because synchronization happens with `join()`
// (the caller will never read this packet until the thread has exited).
//
// An Arc to the packet is stored into a `JoinInner` which in turns is placed
// in `JoinHandle`.
struct Packet<'scope, T> {
    scope: Option<Arc<ScopeData>>,
    result: UnsafeCell<Option<Result<T>>>,
    _marker: PhantomData<Option<&'scope ScopeData>>,
}

// Due to the usage of `UnsafeCell` we need to manually implement Sync.
// The type `T` should already always be Send (otherwise the thread could not
// have been created) and the Packet is Sync because all access to the
// `UnsafeCell` synchronized (by the `join()` boundary), and `ScopeData` is Sync.
unsafe impl<'scope, T: Send> Sync for Packet<'scope, T> {}

impl<'scope, T> Drop for Packet<'scope, T> {
    fn drop(&mut self) {
        // If this packet was for a thread that ran in a scope, the thread
        // panicked, and nobody consumed the panic payload, we make sure
        // the scope function will panic.
        let unhandled_panic = matches!(self.result.get_mut(), Some(Err(_)));
        // Drop the result without causing unwinding.
        // This is only relevant for threads that aren't join()ed, as
        // join() will take the `result` and set it to None, such that
        // there is nothing left to drop here.
        // If this panics, we should handle that, because we're outside the
        // outermost `catch_unwind` of our thread.
        // We just abort in that case, since there's nothing else we can do.
        // (And even if we tried to handle it somehow, we'd also need to handle
        // the case where the panic payload we get out of it also panics on
        // drop, and so on. See issue #86027.)
        if let Err(_) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            *self.result.get_mut() = None;
        })) {
            rtabort!("thread result panicked on drop");
        }
        // Book-keeping so the scope knows when it's done.
        if let Some(scope) = &self.scope {
            // Now that there will be no more user code running on this thread
            // that can use 'scope, mark the thread as 'finished'.
            // It's important we only do this after the `result` has been dropped,
            // since dropping it might still use things it borrowed from 'scope.
            scope.decrement_num_running_threads(unhandled_panic);
        }
    }
}

/// Inner representation for JoinHandle
pub(super) struct JoinInner<'scope, T> {
    native: imp::Thread,
    thread: Thread,
    packet: Arc<Packet<'scope, T>>,
}

impl<'scope, T> JoinInner<'scope, T> {
    pub(super) fn is_finished(&self) -> bool {
        Arc::strong_count(&self.packet) == 1
    }

    pub(super) fn thread(&self) -> &Thread {
        &self.thread
    }

    pub(super) fn join(mut self) -> Result<T> {
        self.native.join();
        Arc::get_mut(&mut self.packet)
            // FIXME(fuzzypixelz): returning an error instead of panicking here
            // would require updating the documentation of
            // `std::thread::Result`; currently we can return `Err` if and only
            // if the thread had panicked.
            .expect("threads should not terminate unexpectedly")
            .result
            .get_mut()
            .take()
            .unwrap()
    }
}

impl<T> AsInner<imp::Thread> for JoinInner<'static, T> {
    fn as_inner(&self) -> &imp::Thread {
        &self.native
    }
}

impl<T> IntoInner<imp::Thread> for JoinInner<'static, T> {
    fn into_inner(self) -> imp::Thread {
        self.native
    }
}
