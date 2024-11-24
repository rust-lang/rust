use super::{Thread, ThreadId};
use crate::mem::ManuallyDrop;
use crate::ptr;
use crate::sys::thread_local::local_pointer;

const NONE: *mut () = ptr::null_mut();
const BUSY: *mut () = ptr::without_provenance_mut(1);
const DESTROYED: *mut () = ptr::without_provenance_mut(2);

local_pointer! {
    static CURRENT;
}

/// Persistent storage for the thread ID.
///
/// We store the thread ID so that it never gets destroyed during the lifetime
/// of a thread, either using `#[thread_local]` or multiple `local_pointer!`s.
mod id {
    use super::*;

    cfg_if::cfg_if! {
        if #[cfg(target_thread_local)] {
            use crate::cell::Cell;

            #[thread_local]
            static ID: Cell<Option<ThreadId>> = Cell::new(None);

            pub(super) const CHEAP: bool = true;

            pub(super) fn get() -> Option<ThreadId> {
                ID.get()
            }

            pub(super) fn set(id: ThreadId) {
                ID.set(Some(id))
            }
        } else if #[cfg(target_pointer_width = "16")] {
            local_pointer! {
                static ID0;
                static ID16;
                static ID32;
                static ID48;
            }

            pub(super) const CHEAP: bool = false;

            pub(super) fn get() -> Option<ThreadId> {
                let id0 = ID0.get().addr() as u64;
                let id16 = ID16.get().addr() as u64;
                let id32 = ID32.get().addr() as u64;
                let id48 = ID48.get().addr() as u64;
                ThreadId::from_u64((id48 << 48) + (id32 << 32) + (id16 << 16) + id0)
            }

            pub(super) fn set(id: ThreadId) {
                let val = id.as_u64().get();
                ID0.set(ptr::without_provenance_mut(val as usize));
                ID16.set(ptr::without_provenance_mut((val >> 16) as usize));
                ID32.set(ptr::without_provenance_mut((val >> 32) as usize));
                ID48.set(ptr::without_provenance_mut((val >> 48) as usize));
            }
        } else if #[cfg(target_pointer_width = "32")] {
            local_pointer! {
                static ID0;
                static ID32;
            }

            pub(super) const CHEAP: bool = false;

            pub(super) fn get() -> Option<ThreadId> {
                let id0 = ID0.get().addr() as u64;
                let id32 = ID32.get().addr() as u64;
                ThreadId::from_u64((id32 << 32) + id0)
            }

            pub(super) fn set(id: ThreadId) {
                let val = id.as_u64().get();
                ID0.set(ptr::without_provenance_mut(val as usize));
                ID32.set(ptr::without_provenance_mut((val >> 32) as usize));
            }
        } else {
            local_pointer! {
                static ID;
            }

            pub(super) const CHEAP: bool = true;

            pub(super) fn get() -> Option<ThreadId> {
                let id = ID.get().addr() as u64;
                ThreadId::from_u64(id)
            }

            pub(super) fn set(id: ThreadId) {
                let val = id.as_u64().get();
                ID.set(ptr::without_provenance_mut(val as usize));
            }
        }
    }

    #[inline]
    pub(super) fn get_or_init() -> ThreadId {
        get().unwrap_or_else(
            #[cold]
            || {
                let id = ThreadId::new();
                id::set(id);
                id
            },
        )
    }
}

/// Tries to set the thread handle for the current thread. Fails if a handle was
/// already set or if the thread ID of `thread` would change an already-set ID.
pub(crate) fn set_current(thread: Thread) -> Result<(), Thread> {
    if CURRENT.get() != NONE {
        return Err(thread);
    }

    match id::get() {
        Some(id) if id == thread.id() => {}
        None => id::set(thread.id()),
        _ => return Err(thread),
    }

    // Make sure that `crate::rt::thread_cleanup` will be run, which will
    // call `drop_current`.
    crate::sys::thread_local::guard::enable();
    CURRENT.set(thread.into_raw().cast_mut());
    Ok(())
}

/// Gets the id of the thread that invokes it.
///
/// This function will always succeed, will always return the same value for
/// one thread and is guaranteed not to call the global allocator.
#[inline]
pub(crate) fn current_id() -> ThreadId {
    // If accessing the persistant thread ID takes multiple TLS accesses, try
    // to retrieve it from the current thread handle, which will only take one
    // TLS access.
    if !id::CHEAP {
        let current = CURRENT.get();
        if current > DESTROYED {
            unsafe {
                let current = ManuallyDrop::new(Thread::from_raw(current));
                return current.id();
            }
        }
    }

    id::get_or_init()
}

/// Gets a handle to the thread that invokes it, if the handle has been initialized.
pub(crate) fn try_current() -> Option<Thread> {
    let current = CURRENT.get();
    if current > DESTROYED {
        unsafe {
            let current = ManuallyDrop::new(Thread::from_raw(current));
            Some((*current).clone())
        }
    } else {
        None
    }
}

/// Gets a handle to the thread that invokes it. If the handle stored in thread-
/// local storage was already destroyed, this creates a new unnamed temporary
/// handle to allow thread parking in nearly all situations.
pub(crate) fn current_or_unnamed() -> Thread {
    let current = CURRENT.get();
    if current > DESTROYED {
        unsafe {
            let current = ManuallyDrop::new(Thread::from_raw(current));
            (*current).clone()
        }
    } else if current == DESTROYED {
        Thread::new_unnamed(id::get_or_init())
    } else {
        init_current(current)
    }
}

/// Gets a handle to the thread that invokes it.
///
/// # Examples
///
/// Getting a handle to the current thread with `thread::current()`:
///
/// ```
/// use std::thread;
///
/// let handler = thread::Builder::new()
///     .name("named thread".into())
///     .spawn(|| {
///         let handle = thread::current();
///         assert_eq!(handle.name(), Some("named thread"));
///     })
///     .unwrap();
///
/// handler.join().unwrap();
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn current() -> Thread {
    let current = CURRENT.get();
    if current > DESTROYED {
        unsafe {
            let current = ManuallyDrop::new(Thread::from_raw(current));
            (*current).clone()
        }
    } else {
        init_current(current)
    }
}

#[cold]
fn init_current(current: *mut ()) -> Thread {
    if current == NONE {
        CURRENT.set(BUSY);
        // If the thread ID was initialized already, use it.
        let id = id::get_or_init();
        let thread = Thread::new_unnamed(id);

        // Make sure that `crate::rt::thread_cleanup` will be run, which will
        // call `drop_current`.
        crate::sys::thread_local::guard::enable();
        CURRENT.set(thread.clone().into_raw().cast_mut());
        thread
    } else if current == BUSY {
        // BUSY exists solely for this check, but as it is in the slow path, the
        // extra TLS write above shouldn't matter. The alternative is nearly always
        // a stack overflow.

        // If you came across this message, contact the author of your allocator.
        // If you are said author: A surprising amount of functions inside the
        // standard library (e.g. `Mutex`, `thread_local!`, `File` when using long
        // paths, even `panic!` when using unwinding), need memory allocation, so
        // you'll get circular dependencies all over the place when using them.
        // I (joboet) highly recommend using only APIs from core in your allocator
        // and implementing your own system abstractions. Still, if you feel that
        // a particular API should be entirely allocation-free, feel free to open
        // an issue on the Rust repository, we'll see what we can do.
        rtabort!(
            "\n
            Attempted to access thread-local data while allocating said data.\n
            Do not access functions that allocate in the global allocator!\n
            This is a bug in the global allocator.\n
        "
        )
    } else {
        debug_assert_eq!(current, DESTROYED);
        panic!(
            "use of std::thread::current() is not possible after the thread's
         local data has been destroyed"
        )
    }
}

/// This should be run in [`crate::rt::thread_cleanup`] to reset the thread
/// handle.
pub(crate) fn drop_current() {
    let current = CURRENT.get();
    if current > DESTROYED {
        unsafe {
            CURRENT.set(DESTROYED);
            drop(Thread::from_raw(current));
        }
    }
}
