//! Test for `gettid` and similar functions for retrieving an OS thread ID.
//@ revisions: with_isolation without_isolation
//@ [without_isolation] compile-flags: -Zmiri-disable-isolation

#![feature(linkage)]

fn gettid() -> u64 {
    cfg_if::cfg_if! {
        if #[cfg(any(target_os = "android", target_os = "linux"))] {
            gettid_linux_like()
        } else if #[cfg(target_os = "nto")] {
            unsafe { libc::gettid() as u64 }
        } else if #[cfg(target_os = "openbsd")] {
            unsafe { libc::getthrid() as u64 }
        } else if #[cfg(target_os = "freebsd")] {
            unsafe { libc::pthread_getthreadid_np() as u64 }
        } else if #[cfg(target_os = "netbsd")] {
            unsafe { libc::_lwp_self() as u64 }
        } else if #[cfg(any(target_os = "solaris", target_os = "illumos"))] {
            // On Solaris and Illumos, the `pthread_t` is the OS TID.
            unsafe { libc::pthread_self() as u64 }
        } else if #[cfg(target_vendor = "apple")] {
            let mut id = 0u64;
            let status: libc::c_int = unsafe { libc::pthread_threadid_np(0, &mut id) };
            assert_eq!(status, 0);
            id
        } else if #[cfg(windows)] {
            use windows_sys::Win32::System::Threading::GetCurrentThreadId;
            unsafe { GetCurrentThreadId() as u64 }
        } else {
            compile_error!("platform has no gettid")
        }
    }
}

/// Test the libc function, the syscall, and the extern symbol.
#[cfg(any(target_os = "android", target_os = "linux"))]
fn gettid_linux_like() -> u64 {
    unsafe extern "C" {
        #[linkage = "extern_weak"]
        static gettid: Option<unsafe extern "C" fn() -> libc::pid_t>;
    }

    let from_libc = unsafe { libc::gettid() as u64 };
    let from_sys = unsafe { libc::syscall(libc::SYS_gettid) as u64 };
    let from_static = unsafe { gettid.unwrap()() as u64 };

    assert_eq!(from_libc, from_sys);
    assert_eq!(from_libc, from_static);

    from_libc
}

/// Specific platforms can query the tid of arbitrary threads from a `pthread_t` / `HANDLE`
#[cfg(any(target_vendor = "apple", windows))]
mod queried {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::{ptr, thread};

    use super::*;

    static SPAWNED_TID: AtomicU64 = AtomicU64::new(0);
    static CAN_JOIN: AtomicBool = AtomicBool::new(false);

    /// Save this thread's TID, give the spawning thread a chance to query it separately before
    /// being joined.
    fn thread_body() {
        SPAWNED_TID.store(gettid(), Ordering::Relaxed);

        // Spin until the main thread has a chance to read this thread's ID
        while !CAN_JOIN.load(Ordering::Relaxed) {
            thread::yield_now();
        }
    }

    /// Spawn a thread, query then return its TID.
    #[cfg(target_vendor = "apple")]
    fn spawn_update_join() -> u64 {
        extern "C" fn thread_start(_data: *mut c_void) -> *mut c_void {
            thread_body();
            ptr::null_mut()
        }

        let mut t: libc::pthread_t = 0;
        let mut spawned_tid_from_handle = 0u64;

        unsafe {
            let res = libc::pthread_create(&mut t, ptr::null(), thread_start, ptr::null_mut());
            assert_eq!(res, 0, "failed to spawn thread");

            let res = libc::pthread_threadid_np(t, &mut spawned_tid_from_handle);
            assert_eq!(res, 0, "failed to query thread ID");
            CAN_JOIN.store(true, Ordering::Relaxed);

            let res = libc::pthread_join(t, ptr::null_mut());
            assert_eq!(res, 0, "failed to join thread");

            // Apple also has two documented return values for invalid threads and null pointers
            let res = libc::pthread_threadid_np(libc::pthread_t::MAX, &mut 0);
            assert_eq!(res, libc::ESRCH, "expected ESRCH for invalid TID");
            let res = libc::pthread_threadid_np(0, ptr::null_mut());
            assert_eq!(res, libc::EINVAL, "invalid EINVAL for a null pointer");
        }

        spawned_tid_from_handle
    }

    /// Spawn a thread, query then return its TID.
    #[cfg(windows)]
    fn spawn_update_join() -> u64 {
        use windows_sys::Win32::Foundation::WAIT_FAILED;
        use windows_sys::Win32::System::Threading::{
            CreateThread, GetThreadId, INFINITE, WaitForSingleObject,
        };

        extern "system" fn thread_start(_data: *mut c_void) -> u32 {
            thread_body();
            0
        }

        let spawned_tid_from_handle;
        let mut tid_at_spawn = 0u32;

        unsafe {
            let handle =
                CreateThread(ptr::null(), 0, Some(thread_start), ptr::null(), 0, &mut tid_at_spawn);
            assert!(!handle.is_null(), "failed to spawn thread");

            spawned_tid_from_handle = GetThreadId(handle);
            assert_ne!(spawned_tid_from_handle, 0, "failed to query thread ID");
            CAN_JOIN.store(true, Ordering::Relaxed);

            let res = WaitForSingleObject(handle, INFINITE);
            assert_ne!(res, WAIT_FAILED, "failed to join thread");
        }

        // Windows also indirectly returns the TID from `CreateThread`, ensure that matches up.
        assert_eq!(spawned_tid_from_handle, tid_at_spawn);

        spawned_tid_from_handle.into()
    }

    pub fn check() {
        let spawned_tid_from_handle = spawn_update_join();
        let spawned_tid_from_self = SPAWNED_TID.load(Ordering::Relaxed);
        let current_tid = gettid();

        // Ensure that we got a different thread ID.
        assert_ne!(spawned_tid_from_handle, current_tid);

        // Ensure that we get the same result from `gettid` and from querying a thread's handle
        assert_eq!(spawned_tid_from_handle, spawned_tid_from_self);
    }
}

fn main() {
    let tid = gettid();

    std::thread::spawn(move || {
        assert_ne!(gettid(), tid);
    });

    // Test that in isolation mode a deterministic value will be returned.
    // The value is not important, we only care that whatever the value is,
    // won't change from execution to execution.
    if cfg!(with_isolation) {
        if cfg!(target_os = "linux") {
            // Linux starts the TID at the PID, which is 1000.
            assert_eq!(tid, 1000);
        } else {
            // Other platforms start counting from 0.
            assert_eq!(tid, 0);
        }
    }

    // On Linux and NetBSD, the first TID is the PID.
    #[cfg(any(target_os = "linux", target_os = "netbsd"))]
    assert_eq!(tid, unsafe { libc::getpid() } as u64);

    #[cfg(any(target_vendor = "apple", windows))]
    queried::check();
}
