//! Support for Windows TLS destructors.
//!
//! Windows has an API to provide a destructor for a FLS (fiber local storage) variable,
//! which behaves similarly to a TLS variable for our purpose [1].
//!
//! All TLS destructors are tracked by *us*, not the Windows runtime.
//! This means that we have a global list of destructors for
//! each TLS key or variable that we know about.
//!
//! [1]: https://devblogs.microsoft.com/oldnewthing/20191011-00/?p=102989

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering, fence};

use crate::cell::Cell;
use crate::ptr;
use crate::sys::c::{self, FLS_OUT_OF_INDEXES};

pub type Key = u32;

unsafe fn create(dtor: c::PFLS_CALLBACK_FUNCTION) -> Key {
    let key_result = unsafe { c::FlsAlloc(dtor) };

    if key_result == c::FLS_OUT_OF_INDEXES {
        rtabort!("out of FLS keys");
    }

    key_result
}

unsafe fn set(key: Key, ptr: *const c_void) {
    let result = unsafe { c::FlsSetValue(key, ptr) };

    if result == c::FALSE {
        rtabort!("failed to set FLS value");
    }
}

fn is_thread_a_fiber() -> bool {
    let res = unsafe { c::IsThreadAFiber() };
    res == c::TRUE
}

static KEY: AtomicU32 = AtomicU32::new(FLS_OUT_OF_INDEXES);

/// Used to track whether we are currently in the critical section of `enable`.
/// For miri, these atomic operations cause synchronization that can mask user bugs,
/// and they are not needed as `atexit` is anyway not supported, so we can skip them.
struct EnableGuard;
static AT_EXIT_HOOK_CALLED: AtomicBool = AtomicBool::new(false);
static ACTIVE_ENABLE_CALLS: AtomicU32 = AtomicU32::new(0);

impl EnableGuard {
    // Mark the start of an `enable` call, returning whether the `atexit` hook has already been called or not.
    fn new() -> (Self, bool) {
        if cfg!(miri) {
            return (Self, false);
        }
        ACTIVE_ENABLE_CALLS.fetch_add(1, Ordering::Relaxed);

        // Both `new` and `start_exit` publish state to one atomic and inspect the other.
        // `AcqRel` is insufficient because neither read is required to observe the other's publication,
        // so we could create the guard but `start_exit` would not see any active enable calls.
        // `SeqCst` ensures that there's a single global order between the publish and check,
        // so at least one side must observe the other and bail.
        fence(Ordering::SeqCst);

        let at_exit_called = AT_EXIT_HOOK_CALLED.load(Ordering::Relaxed);

        (Self, at_exit_called)
    }

    /// Mark the start of process exit, returning whether we should free the FLS key or not.
    fn start_exit() -> bool {
        // After this hook starts, new destructor registration will be skipped,
        // causing TLS destructors initialized after this point to leak.
        if AT_EXIT_HOOK_CALLED.swap(true, Ordering::Relaxed) {
            // Cleanup already started, there is nothing else to do.
            return false;
        }

        fence(Ordering::SeqCst);

        let any_active_enabled_called = ACTIVE_ENABLE_CALLS.load(Ordering::Relaxed) != 0;

        if any_active_enabled_called {
            // If another thread is currently in `enable`, it may already have loaded this key and may be about to call `FlsSetValue`.
            // So we must *not* call free the FLS key.
            //
            // During real process exit this is harmless because the `cleanup` hook is always available,
            // and the FLS callback will be triggered normally by the OS.
            //
            // During DLL unload, the unloader cannot safely have threads running code from the DLL except for the destructors,
            // so there must not be any `enable` calls active anyway.
            return false;
        }

        return true;
    }
}

#[cfg(not(miri))]
impl Drop for EnableGuard {
    fn drop(&mut self) {
        ACTIVE_ENABLE_CALLS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Set up the current thread to invoke `cleanup` when it finishes.
pub fn enable() {
    let registered = if cfg!(target_thread_local) {
        #[thread_local]
        static REGISTERED: Cell<bool> = Cell::new(false);
        REGISTERED.replace(true)
    } else {
        // `#[thread_local]` is unavailable on windows-gnu (`target_thread_local` is off),
        // but setting the FLS key's value is about as expensive as `TlsGet`, so we don't bother tracking registration separately.
        false
    };

    if !registered {
        // We are in a critical section where we are trying to register a destructor for the current thread.
        // We need to avoid racing with the `atexit` hook that frees the FLS slot, which would cause us to call `FlsSetValue` on a freed key,
        // or calling `atexit` during process shutdown, which would cause a deadlock.
        let (_guard, at_exit_called) = EnableGuard::new();

        if at_exit_called {
            // We are exiting and don't want to race with the `atexit` hook, so we won't be able to run the destructors for this thread.
            return;
        }

        let current_key = KEY.load(Ordering::Acquire);

        // If we already allocated a key, we only need to set it to a non-null value so that the destructors hook is run for this thread.
        let key = if current_key != FLS_OUT_OF_INDEXES {
            current_key
        } else {
            // Otherwise, we try to allocate a key.
            let new_key = unsafe { create(Some(cleanup)) };

            // Now we need to set this key to be used by everyone else.
            // If we won the race, our key is the right one and we can set it to non-null value.
            // If we lost, we'll use the winning key and free our losing key.
            match KEY.compare_exchange(current_key, new_key, Ordering::Release, Ordering::Acquire) {
                Ok(_) => {
                    // If the current DLL is unloaded, the registered `cleanup` hook will not be available later during thread exit,
                    // triggering a `STATUS_ACCESS_VIOLATION`. To avoid this, we use the `atexit` hook, which is called during DLL unload
                    // to manually free the FLS slot, triggering the destructors.
                    //
                    // However, calling `atexit` during process exit can cause a deadlock.
                    // In a Rust binary, `enable` is called during the main thread startup and before any user code,
                    // and we checked using `at_exit_called` that we aren't in process shutdown.
                    //
                    // In a Rust DLL, dynamic unloading can only happen safely when no other threads are
                    // concurrently executing Rust code, so if we are here we cannot be unloading yet.
                    //
                    // If a main non-Rust binary is exiting, it must not be trigger the `enable` guard
                    // for the first time during process shutdown.
                    //
                    // Miri has no DLL unloading so we can skip this step here.
                    if !cfg!(miri) {
                        if cleanup_is_unloadable() {
                            let res = unsafe { c::atexit(free_fls_key_at_exit) };
                            if res != 0 {
                                rtabort!("failed to register fls atexit hook");
                            }
                        }
                    }

                    new_key
                }
                Err(other_key) => {
                    unsafe { c::FlsFree(new_key) };
                    other_key
                }
            }
        };

        // Setting the key's value to non-zero will cause the dtor callback to be called when the thread exits.
        unsafe { set(key, ptr::without_provenance(1)) };
    }
}

/// Checks if `cleanup` is in a different module from the main executable,
/// using `GetModuleHandleExW(FLAG_FROM_ADDRESS, cleanup) != GetModuleHandleW(ptr::null())`.
///
/// If `cleanup` lives in the main executable, its code cannot be unmapped
/// before process exit, so no unload hook is needed.
///
/// If it lives in a DLL, the DLL may be unloaded while the process keeps
/// running, so the FLS callback must be unregistered before that image is
/// unmapped.
///
/// On failure, return true, which assumes it can be unloaded.
fn cleanup_is_unloadable() -> bool {
    // Get a handle to the module of `cleanup`.
    let cleanup_module = {
        let mut handle: c::HMODULE = ptr::null_mut();

        let res = unsafe {
            c::GetModuleHandleExW(
                c::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | c::GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                cleanup as *const () as c::PCWSTR,
                &mut handle,
            )
        };

        if res == c::FALSE || handle.is_null() {
            return true;
        }

        handle
    };

    // Get a handle to the file used to create the calling process (.exe file).
    let main_exe_module = unsafe { c::GetModuleHandleW(ptr::null()) };

    if main_exe_module.is_null() {
        return true;
    }

    cleanup_module != main_exe_module
}

extern "C" fn free_fls_key_at_exit() {
    // The main purpose of this hook is to free the FLS slot during DLL unload.
    // However, this hook will also be called during normal process exit, while other Rust threads are still running,
    // so we must be careful to avoid races with `enable`.
    let should_free_key = EnableGuard::start_exit();
    if !should_free_key {
        return;
    }

    let current_key = KEY.swap(c::FLS_OUT_OF_INDEXES, Ordering::AcqRel);
    if current_key != c::FLS_OUT_OF_INDEXES {
        // Calling `FlsFree` will cause the OS to call the `cleanup` hook, in the current thread, *for each thread* (or fiber) with a value in this FLS slot.
        // `cleanup` is safe to run repeatedly: it only drains the current thread's TLS destructor list, and we check that we are not running in a fiber before doing so.
        // We only call this when no `enable` call is active, so it cannot race with `FlsSetValue` using this key.
        // Destructors of thread locals in other threads will not run and therefore leak, which is allowed since we are exiting or unloading.
        unsafe { c::FlsFree(current_key) };
    }
}

unsafe extern "system" fn cleanup(_ptr: *const c_void) {
    // Avoid running the hook if we are in a fiber.
    // This will cause destructors of thread locals to not run, leaking them.
    // Thread-local runtime state will not be cleaned.
    //
    // We need to verify that we won't run the destructors *before* the thread exits,
    // but if the fiber that registered the callback is deleted, the thread might still be running other fibers.
    //
    // By checking that we are not running in a fiber here, we are guaranteed that the hook is only running during the thread's exit.
    // See also the `fiber_does_not_trigger_dtor` test.
    if is_thread_a_fiber() {
        return;
    }

    unsafe {
        #[cfg(target_thread_local)]
        super::super::destructors::run();
        #[cfg(not(target_thread_local))]
        super::super::key::run_dtors();
    }

    crate::rt::thread_cleanup();
}
