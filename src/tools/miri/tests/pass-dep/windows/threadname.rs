//@only-target: windows # this directly tests windows-only functions
//@run-native

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Foundation::LocalFree;
use windows_sys::Win32::System::Threading::{
    GetCurrentThread, GetThreadDescription, SetThreadDescription,
};

fn to_u16s<S: AsRef<OsStr>>(s: S) -> Vec<u16> {
    let mut result: Vec<_> = s.as_ref().encode_wide().collect();
    result.push(0);
    result
}

fn main() {
    unsafe {
        let name = c"mythreadname";

        let utf16 = to_u16s(name.to_str().unwrap());
        SetThreadDescription(GetCurrentThread(), utf16.as_ptr());

        let mut ptr = core::ptr::null_mut::<u16>();
        let result = GetThreadDescription(GetCurrentThread(), &mut ptr);
        assert!(result >= 0);
        let name_gotten = String::from_utf16_lossy({
            let mut len = 0;
            while *ptr.add(len) != 0 {
                len += 1;
            }
            core::slice::from_raw_parts(ptr, len)
        });
        assert_eq!(name_gotten, name.to_str().unwrap());
        let r = LocalFree(ptr.cast());
        assert!(r.is_null());
    }
}
