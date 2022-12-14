use std::borrow::Cow::{Borrowed, Owned};
use std::ffi::CStr;
use std::os::raw::c_char;

#[test]
fn to_str() {
    let data = b"123\xE2\x80\xA6\0";
    let ptr = data.as_ptr() as *const c_char;
    unsafe {
        assert_eq!(CStr::from_ptr(ptr).to_str(), Ok("123…"));
        assert_eq!(CStr::from_ptr(ptr).to_string_lossy(), Borrowed("123…"));
    }
    let data = b"123\xE2\0";
    let ptr = data.as_ptr() as *const c_char;
    unsafe {
        assert!(CStr::from_ptr(ptr).to_str().is_err());
        assert_eq!(CStr::from_ptr(ptr).to_string_lossy(), Owned::<str>(format!("123\u{FFFD}")));
    }
}
