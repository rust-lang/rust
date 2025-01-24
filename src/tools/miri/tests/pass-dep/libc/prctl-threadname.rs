//@only-target: android  # Miri supports prctl for Android only
use std::ffi::{CStr, CString};
use std::thread;

// The Linux kernel all names 16 bytes long including the null terminator.
const MAX_THREAD_NAME_LEN: usize = 16;

fn main() {
    // The short name should be shorter than 16 bytes which POSIX promises
    // for thread names. The length includes a null terminator.
    let short_name = "test_named".to_owned();
    let long_name = std::iter::once("test_named_thread_truncation")
        .chain(std::iter::repeat(" yada").take(100))
        .collect::<String>();

    fn set_thread_name(name: &CStr) -> i32 {
        unsafe { libc::prctl(libc::PR_SET_NAME, name.as_ptr().cast::<libc::c_char>()) }
    }

    fn get_thread_name(name: &mut [u8]) -> i32 {
        assert!(name.len() >= MAX_THREAD_NAME_LEN);
        unsafe { libc::prctl(libc::PR_GET_NAME, name.as_mut_ptr().cast::<libc::c_char>()) }
    }

    // Set name via Rust API, get it via prctl.
    let long_name2 = long_name.clone();
    thread::Builder::new()
        .name(long_name.clone())
        .spawn(move || {
            let mut buf = vec![0u8; MAX_THREAD_NAME_LEN];
            assert_eq!(get_thread_name(&mut buf), 0);
            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            let truncated_name = &long_name2[..long_name2.len().min(MAX_THREAD_NAME_LEN - 1)];
            assert_eq!(cstr.to_bytes(), truncated_name.as_bytes());
        })
        .unwrap()
        .join()
        .unwrap();

    // Set name via prctl and get it again (short name).
    thread::Builder::new()
        .spawn(move || {
            // Set short thread name.
            let cstr = CString::new(short_name.clone()).unwrap();
            assert!(cstr.to_bytes_with_nul().len() <= MAX_THREAD_NAME_LEN); // this should fit
            assert_eq!(set_thread_name(&cstr), 0);

            let mut buf = vec![0u8; MAX_THREAD_NAME_LEN];
            assert_eq!(get_thread_name(&mut buf), 0);
            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            assert_eq!(cstr.to_bytes(), short_name.as_bytes());
        })
        .unwrap()
        .join()
        .unwrap();

    // Set name via prctl and get it again (long name).
    thread::Builder::new()
        .spawn(move || {
            // Set full thread name.
            let cstr = CString::new(long_name.clone()).unwrap();
            assert!(cstr.to_bytes_with_nul().len() > MAX_THREAD_NAME_LEN);
            // Names are truncated by the Linux kernel.
            assert_eq!(set_thread_name(&cstr), 0);

            let mut buf = vec![0u8; MAX_THREAD_NAME_LEN];
            assert_eq!(get_thread_name(&mut buf), 0);
            let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
            assert_eq!(cstr.to_bytes(), &long_name.as_bytes()[..(MAX_THREAD_NAME_LEN - 1)]);
        })
        .unwrap()
        .join()
        .unwrap();
}
