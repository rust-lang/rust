//@only-target: windows # this directly tests windows-only functions
//@run-native

use core::ffi::c_void;
use core::mem::size_of_val;
use core::ptr::null_mut;

use windows_sys::Win32::Security::Cryptography::{
    BCRYPT_RNG_ALG_HANDLE, BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom, ProcessPrng,
};

// FIXME: `windows-sys` declares the return type as bool. The SDK declares it as BOOLEAN (u8).
// https://github.com/microsoft/windows-rs/issues/5437
#[link(name = "advapi32")]
unsafe extern "system" {
    #[link_name = "SystemFunction036"]
    fn RtlGenRandom(buffer: *mut c_void, len: u32) -> u8;
}

fn main() {
    let mut key = [0u8; 24];
    let len: u32 = size_of_val(&key).try_into().unwrap();
    let ret = unsafe {
        BCryptGenRandom(null_mut(), key.as_mut_ptr(), len, BCRYPT_USE_SYSTEM_PREFERRED_RNG)
    };
    // NTSTATUS codes use the high bit to indicate an error
    assert!(ret >= 0);

    let ret = unsafe { BCryptGenRandom(BCRYPT_RNG_ALG_HANDLE, key.as_mut_ptr(), len, 0) };
    assert!(ret >= 0);

    let ret = unsafe { RtlGenRandom(key.as_mut_ptr().cast(), len) };
    // RtlGenRandom returns a BOOLEAN where 0 indicates an error
    assert_ne!(ret, 0);

    let len = key.len();
    let ret = unsafe { ProcessPrng(key.as_mut_ptr(), len) };
    // ProcessPrng is documented as always returning `TRUE`.
    // https://learn.microsoft.com/en-us/windows/win32/seccng/processprng#return-value
    assert_eq!(ret, 1);
}
