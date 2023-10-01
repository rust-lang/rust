use std::ffi::OsString;
use std::path::PathBuf;

use super::miri_extern;

pub fn host_to_target_path(path: OsString) -> PathBuf {
    use std::ffi::{CStr, CString};

    // Once into_encoded_bytes is stable we can use it here.
    // (Unstable features would need feature flags in each test...)
    let path = CString::new(path.into_string().unwrap()).unwrap();
    let mut out = Vec::with_capacity(1024);

    unsafe {
        let ret =
            miri_extern::miri_host_to_target_path(path.as_ptr(), out.as_mut_ptr(), out.capacity());
        assert_eq!(ret, 0);
        // Here we panic if it's not UTF-8... but that is hard to avoid with OsStr APIs.
        let out = CStr::from_ptr(out.as_ptr()).to_str().unwrap();
        PathBuf::from(out)
    }
}

pub fn tmp() -> PathBuf {
    let path =
        std::env::var_os("MIRI_TEMP").unwrap_or_else(|| std::env::temp_dir().into_os_string());
    // These are host paths. We need to convert them to the target.
    host_to_target_path(path)
}
