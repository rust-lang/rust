//@ignore-target: windows # No pthreads on Windows
use std::ffi::CStr;
#[cfg(not(target_os = "freebsd"))]
use std::ffi::CString;
use std::thread;

fn main() {
    // The short name should be shorter than 16 bytes which POSIX promises
    // for thread names. The length includes a null terminator.
    let short_name = "test_named".to_owned();
    let long_name = std::iter::once("test_named_thread_truncation")
        .chain(std::iter::repeat(" yada").take(100))
        .collect::<String>();

    fn set_thread_name(name: &CStr) -> i32 {
        cfg_if::cfg_if! {
            if #[cfg(any(target_os = "linux", target_os = "illumos", target_os = "solaris"))] {
                unsafe { libc::pthread_setname_np(libc::pthread_self(), name.as_ptr().cast()) }
            } else if #[cfg(target_os = "freebsd")] {
                // pthread_set_name_np does not return anything
                unsafe { libc::pthread_set_name_np(libc::pthread_self(), name.as_ptr().cast()) };
                0
            } else if #[cfg(target_os = "macos")] {
                unsafe { libc::pthread_setname_np(name.as_ptr().cast()) }
            } else {
                compile_error!("set_thread_name not supported for this OS")
            }
        }
    }

    fn get_thread_name(name: &mut [u8]) -> i32 {
        cfg_if::cfg_if! {
            if #[cfg(any(
                target_os = "linux",
                target_os = "illumos",
                target_os = "solaris",
                target_os = "macos"
            ))] {
                unsafe {
                    libc::pthread_getname_np(libc::pthread_self(), name.as_mut_ptr().cast(), name.len())
                }
            } else if #[cfg(target_os = "freebsd")] {
                // pthread_get_name_np does not return anything
                unsafe {
                    libc::pthread_get_name_np(libc::pthread_self(), name.as_mut_ptr().cast(), name.len())
                };
                0
            } else {
                compile_error!("get_thread_name not supported for this OS")
            }
        }
    }

    thread::Builder::new()
        .name(short_name.clone())
        .spawn(move || {
            // Rust remembers the full thread name itself.
            assert_eq!(thread::current().name(), Some(short_name.as_str()));

            // Note that glibc requires 15 bytes long buffer exculding a null terminator.
            // Otherwise, `pthread_getname_np` returns an error.
            let mut buf = vec![0u8; short_name.len().max(15) + 1];
            assert_eq!(get_thread_name(&mut buf), 0);

            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            // POSIX seems to promise at least 15 chars excluding a null terminator.
            assert_eq!(short_name.as_bytes(), cstr.to_bytes());

            // Also test directly calling pthread_setname to check its return value.
            assert_eq!(set_thread_name(&cstr), 0);

            // For glibc used by linux-gnu there should be a failue,
            // if a shorter than 16 bytes buffer is provided, even if that would be
            // large enough for the thread name.
            #[cfg(target_os = "linux")]
            assert_eq!(get_thread_name(&mut buf[..15]), libc::ERANGE);
        })
        .unwrap()
        .join()
        .unwrap();

    thread::Builder::new()
        .name(long_name.clone())
        .spawn(move || {
            // Rust remembers the full thread name itself.
            assert_eq!(thread::current().name(), Some(long_name.as_str()));

            // But the system is limited -- make sure we successfully set a truncation.
            // Note that there's no specific to glibc buffer requirement, since the value
            // `long_name` is longer than 16 bytes including a null terminator.
            let mut buf = vec![0u8; long_name.len() + 1];
            assert_eq!(get_thread_name(&mut buf), 0);

            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            // POSIX seems to promise at least 15 chars excluding a null terminator.
            assert!(
                cstr.to_bytes().len() >= 15,
                "name is too short: len={}",
                cstr.to_bytes().len()
            );
            assert!(long_name.as_bytes().starts_with(cstr.to_bytes()));

            // Also test directly calling pthread_setname to check its return value.
            assert_eq!(set_thread_name(&cstr), 0);

            // But with a too long name it should fail (except on FreeBSD where the
            // function has no return, hence cannot indicate failure).
            #[cfg(not(target_os = "freebsd"))]
            assert_ne!(set_thread_name(&CString::new(long_name).unwrap()), 0);
        })
        .unwrap()
        .join()
        .unwrap();
}
