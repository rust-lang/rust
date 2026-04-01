//@ignore-target: windows # No pthreads on Windows
//@ignore-target: android # No pthread_{get,set}_name on Android
use std::ffi::{CStr, CString};
use std::thread;

const MAX_THREAD_NAME_LEN: usize = {
    cfg_if::cfg_if! {
        if #[cfg(any(target_os = "linux"))] {
            16
        } else if #[cfg(any(target_os = "illumos", target_os = "solaris"))] {
            32
        } else if #[cfg(target_os = "macos")] {
            libc::MAXTHREADNAMESIZE // 64, at the time of writing
        } else if #[cfg(target_os = "freebsd")] {
            usize::MAX // as far as I can tell
        } else {
            panic!()
        }
    }
};

fn main() {
    // The short name should be shorter than 16 bytes which POSIX promises
    // for thread names. The length includes a null terminator.
    let short_name = "test_named".to_owned();
    let long_name = std::iter::once("test_named_thread_truncation")
        .chain(std::iter::repeat(" yada").take(100))
        .collect::<String>();

    fn set_thread_name(name: &CStr) -> i32 {
        cfg_if::cfg_if! {
            if #[cfg(any(
                target_os = "linux",
                target_os = "freebsd",
                target_os = "illumos",
                target_os = "solaris"
            ))] {
                unsafe { libc::pthread_setname_np(libc::pthread_self(), name.as_ptr().cast()) }
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
                target_os = "freebsd",
                target_os = "illumos",
                target_os = "solaris",
                target_os = "macos"
            ))] {
                unsafe {
                    libc::pthread_getname_np(libc::pthread_self(), name.as_mut_ptr().cast(), name.len())
                }
            } else {
                compile_error!("get_thread_name not supported for this OS")
            }
        }
    }

    // Set name via Rust API, get it via pthreads.
    let long_name2 = long_name.clone();
    thread::Builder::new()
        .name(long_name.clone())
        .spawn(move || {
            let mut buf = vec![0u8; long_name2.len() + 1];
            assert_eq!(get_thread_name(&mut buf), 0);
            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            let truncated_name = &long_name2[..long_name2.len().min(MAX_THREAD_NAME_LEN - 1)];
            assert_eq!(cstr.to_bytes(), truncated_name.as_bytes());
        })
        .unwrap()
        .join()
        .unwrap();

    // Set name via pthread and get it again (short name).
    thread::Builder::new()
        .spawn(move || {
            // Set short thread name.
            let cstr = CString::new(short_name.clone()).unwrap();
            assert!(cstr.to_bytes_with_nul().len() <= MAX_THREAD_NAME_LEN); // this should fit
            assert_eq!(set_thread_name(&cstr), 0);

            // Now get it again, in various ways.

            // POSIX seems to promise at least 15 chars excluding a null terminator.
            let mut buf = vec![0u8; short_name.len().max(15) + 1];
            assert_eq!(get_thread_name(&mut buf), 0);
            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            assert_eq!(cstr.to_bytes(), short_name.as_bytes());

            // Test what happens when the buffer is shorter than 16, but still long enough.
            let res = get_thread_name(&mut buf[..15]);
            cfg_if::cfg_if! {
                if #[cfg(target_os = "linux")] {
                    // For glibc used by linux-gnu there should be a failue,
                    // if a shorter than 16 bytes buffer is provided, even if that would be
                    // large enough for the thread name.
                    assert_eq!(res, libc::ERANGE);
                } else {
                    // Everywhere else, this should work.
                    assert_eq!(res, 0);
                    // POSIX seems to promise at least 15 chars excluding a null terminator.
                    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
                    assert_eq!(short_name.as_bytes(), cstr.to_bytes());
                }
            }

            // Test what happens when the buffer is too short even for the short name.
            let res = get_thread_name(&mut buf[..4]);
            cfg_if::cfg_if! {
                if #[cfg(any(target_os = "freebsd", target_os = "macos"))] {
                    // On macOS and FreeBSD it's not an error for the buffer to be
                    // too short for the thread name -- they truncate instead.
                    assert_eq!(res, 0);
                    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
                    assert_eq!(cstr.to_bytes_with_nul().len(), 4);
                    assert!(short_name.as_bytes().starts_with(cstr.to_bytes()));
                } else {
                    // The rest should give an error.
                    assert_eq!(res, libc::ERANGE);
                }
            }

            // Test zero-sized buffer.
            let res = get_thread_name(&mut []);
            cfg_if::cfg_if! {
                if #[cfg(any(target_os = "freebsd", target_os = "macos"))] {
                    // On macOS and FreeBSD it's not an error for the buffer to be
                    // too short for the thread name -- even with size 0.
                    assert_eq!(res, 0);
                } else {
                    // The rest should give an error.
                    assert_eq!(res, libc::ERANGE);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();

    // Set name via pthread and get it again (long name).
    thread::Builder::new()
        .spawn(move || {
            // Set full thread name.
            let cstr = CString::new(long_name.clone()).unwrap();
            let res = set_thread_name(&cstr);
            cfg_if::cfg_if! {
                if #[cfg(target_os = "freebsd")] {
                    // Names of all size are supported.
                    assert!(cstr.to_bytes_with_nul().len() <= MAX_THREAD_NAME_LEN);
                    assert_eq!(res, 0);
                } else if #[cfg(target_os = "macos")] {
                    // Name is too long.
                    assert!(cstr.to_bytes_with_nul().len() > MAX_THREAD_NAME_LEN);
                    assert_eq!(res, libc::ENAMETOOLONG);
                } else {
                    // Name is too long.
                    assert!(cstr.to_bytes_with_nul().len() > MAX_THREAD_NAME_LEN);
                    assert_eq!(res, libc::ERANGE);
                }
            }
            // Set the longest name we can.
            let truncated_name = &long_name[..long_name.len().min(MAX_THREAD_NAME_LEN - 1)];
            let cstr = CString::new(truncated_name).unwrap();
            assert_eq!(set_thread_name(&cstr), 0);

            // Now get it again, in various ways.

            // This name should round-trip properly.
            let mut buf = vec![0u8; long_name.len() + 1];
            assert_eq!(get_thread_name(&mut buf), 0);
            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            assert_eq!(cstr.to_bytes(), truncated_name.as_bytes());

            // Test what happens when our buffer is just one byte too small.
            let res = get_thread_name(&mut buf[..truncated_name.len()]);
            cfg_if::cfg_if! {
                if #[cfg(any(target_os = "freebsd", target_os = "macos"))] {
                    // On macOS and FreeBSD it's not an error for the buffer to be
                    // too short for the thread name -- they truncate instead.
                    assert_eq!(res, 0);
                    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
                    assert_eq!(cstr.to_bytes(), &truncated_name.as_bytes()[..(truncated_name.len() - 1)]);
                } else {
                    // The rest should give an error.
                    assert_eq!(res, libc::ERANGE);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();

    // Now set the name for a non-existing thread and verify error codes.
    let invalid_thread = 0xdeadbeef;
    let error = {
        cfg_if::cfg_if! {
            if #[cfg(target_os = "linux")] {
                libc::ENOENT
            } else {
                libc::ESRCH
            }
        }
    };

    #[cfg(not(target_os = "macos"))]
    {
        // macOS has no `setname` function accepting a thread id as the first argument.
        let res = unsafe { libc::pthread_setname_np(invalid_thread, [0].as_ptr()) };
        assert_eq!(res, error);
    }

    let mut buf = [0; 64];
    let res = unsafe { libc::pthread_getname_np(invalid_thread, buf.as_mut_ptr(), buf.len()) };
    assert_eq!(res, error);
}
