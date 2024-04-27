//@ignore-target-windows: No libc on Windows
use std::ffi::CStr;
#[cfg(not(target_os = "freebsd"))]
use std::ffi::CString;
use std::thread;

fn main() {
    let long_name = std::iter::once("test_named_thread_truncation")
        .chain(std::iter::repeat(" yada").take(100))
        .collect::<String>();

    fn set_thread_name(name: &CStr) -> i32 {
        #[cfg(target_os = "linux")]
        return unsafe { libc::pthread_setname_np(libc::pthread_self(), name.as_ptr().cast()) };
        #[cfg(target_os = "freebsd")]
        unsafe {
            // pthread_set_name_np does not return anything
            libc::pthread_set_name_np(libc::pthread_self(), name.as_ptr().cast());
            return 0;
        };
        #[cfg(target_os = "macos")]
        return unsafe { libc::pthread_setname_np(name.as_ptr().cast()) };
    }

    let result = thread::Builder::new().name(long_name.clone()).spawn(move || {
        // Rust remembers the full thread name itself.
        assert_eq!(thread::current().name(), Some(long_name.as_str()));

        // But the system is limited -- make sure we successfully set a truncation.
        let mut buf = vec![0u8; long_name.len() + 1];
        #[cfg(not(target_os = "freebsd"))]
        unsafe {
            libc::pthread_getname_np(libc::pthread_self(), buf.as_mut_ptr().cast(), buf.len())
        };
        #[cfg(target_os = "freebsd")]
        unsafe {
            libc::pthread_get_name_np(libc::pthread_self(), buf.as_mut_ptr().cast(), buf.len())
        };
        let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
        assert!(cstr.to_bytes().len() >= 15, "name is too short: len={}", cstr.to_bytes().len()); // POSIX seems to promise at least 15 chars
        assert!(long_name.as_bytes().starts_with(cstr.to_bytes()));

        // Also test directly calling pthread_setname to check its return value.
        assert_eq!(set_thread_name(&cstr), 0);
        // But with a too long name it should fail (except on FreeBSD where the
        // function has no return, hence cannot indicate failure).
        #[cfg(not(target_os = "freebsd"))]
        assert_ne!(set_thread_name(&CString::new(long_name).unwrap()), 0);
    });
    result.unwrap().join().unwrap();
}
