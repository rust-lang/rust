//@ ignore-target: illumos solaris wasm
//@ revisions: with_isolation without_isolation
//@ [without_isolation] compile-flags: -Zmiri-disable-isolation

// This test is based on `getpid.rs`

fn gettid() -> u64 {
    cfg_if::cfg_if! {
        if #[cfg(any(target_os = "android", target_os = "linux", target_os = "nto"))] {
            (unsafe { libc::gettid() }) as u64
        } else if #[cfg(target_os = "openbsd")] {
            (unsafe { libc::getthrid() }) as u64
        } else if #[cfg(target_os = "freebsd")] {
            (unsafe { libc::pthread_getthreadid_np() }) as u64
        } else if #[cfg(target_os = "netbsd")] {
            (unsafe { libc::_lwp_self() }) as u64
        } else if #[cfg(target_vendor = "apple")] {
            let mut id = 0u64;
            let status: libc::c_int = unsafe { libc::pthread_threadid_np(0, &mut id) };
            assert_eq!(status, 0);
            id
        } else if #[cfg(windows)] {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadId};
            (unsafe { GetThreadId(GetCurrentThread()) }) as u64
        } else {
            compile_error!("platform has no gettid")
        }
    }
}

/// Specific platforms can query the tid of arbitrary threads; test that here.
#[cfg(any(target_vendor = "apple", windows))]
mod queried {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::{ptr, thread, time};

    use super::*;

    static SPAWNED_TID: AtomicU64 = AtomicU64::new(0);
    static CAN_JOIN: AtomicBool = AtomicBool::new(false);

    #[cfg(unix)]
    extern "C" fn thread_start(_data: *mut c_void) -> *mut c_void {
        thread_body();
        ptr::null_mut()
    }

    #[cfg(windows)]
    extern "system" fn thread_start(_data: *mut c_void) -> u32 {
        thread_body();
        0
    }

    fn thread_body() {
        SPAWNED_TID.store(gettid(), Ordering::Relaxed);
        let sleep_duration = time::Duration::from_millis(10);

        // Spin until the main thread has a chance to read this thread's ID
        while !CAN_JOIN.load(Ordering::Relaxed) {
            thread::sleep(sleep_duration);
        }
    }

    #[cfg(unix)]
    fn spawn_update_join() -> u64 {
        let mut t: libc::pthread_t = 0;
        let mut spawned_tid_from_handle = 0u64;

        unsafe {
            let res = libc::pthread_create(&mut t, ptr::null(), thread_start, ptr::null_mut());
            assert_eq!(res, 0);

            let res = libc::pthread_threadid_np(t, &mut spawned_tid_from_handle);
            assert_eq!(res, 0);
            CAN_JOIN.store(true, Ordering::Relaxed);

            let res = libc::pthread_join(t, ptr::null_mut());
            assert_eq!(res, 0);
        }

        spawned_tid_from_handle
    }

    #[cfg(windows)]
    fn spawn_update_join() -> u64 {
        use windows_sys::Win32::Foundation::WAIT_FAILED;
        use windows_sys::Win32::System::Threading::{
            CreateThread, GetThreadId, INFINITE, WaitForSingleObject,
        };

        let spawned_tid_from_handle;
        let mut tid_at_spawn = 0u32;

        unsafe {
            let handle =
                CreateThread(ptr::null(), 0, Some(thread_start), ptr::null(), 0, &mut tid_at_spawn);
            assert!(!handle.is_null());

            spawned_tid_from_handle = GetThreadId(handle);
            assert_ne!(spawned_tid_from_handle, 0);
            CAN_JOIN.store(true, Ordering::Relaxed);

            let res = WaitForSingleObject(handle, INFINITE);
            assert_ne!(res, WAIT_FAILED);
        }

        assert_eq!(spawned_tid_from_handle, tid_at_spawn);

        spawned_tid_from_handle.into()
    }

    pub fn check() {
        let spawned_tid_from_handle = spawn_update_join();
        assert_ne!(spawned_tid_from_handle, 0);
        assert_ne!(spawned_tid_from_handle, gettid());
        assert_eq!(spawned_tid_from_handle, SPAWNED_TID.load(Ordering::Relaxed));
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
