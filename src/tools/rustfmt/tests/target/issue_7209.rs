//@ rustfmt-unstable: true

fn gettid() -> u64 {
    cfg_select! {
        any(target_os = "android", target_os = "linux") => gettid_linux_like(),
        any(target_os = "nto", target_os = "qnx") => unsafe { libc::gettid() as u64 },
        target_os = "openbsd" => unsafe { libc::getthrid() as u64 },
        target_os = "freebsd" => unsafe { libc::pthread_getthreadid_np() as u64 },
        target_os = "netbsd" => unsafe { libc::_lwp_self() as u64 },
        any(target_os = "solaris", target_os = "illumos") => {
            // On Solaris and Illumos, the `pthread_t` is the OS TID.
            unsafe { libc::pthread_self() as u64 }
        }
        target_vendor = "apple" => {
            let mut id = 0u64;
            let status: libc::c_int = unsafe { libc::pthread_threadid_np(0, &mut id) };
            assert_eq!(status, 0);
            id
        }
        windows => {
            use windows_sys::Win32::System::Threading::GetCurrentThreadId;
            unsafe { GetCurrentThreadId() as u64 }
        }
        _ => {
            compile_error!("platform has no gettid")
        }
    }
}
