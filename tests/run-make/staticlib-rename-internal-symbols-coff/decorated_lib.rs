#![crate_type = "staticlib"]
#![feature(abi_vectorcall)]

#[no_mangle]
pub extern "C" fn decorated_cdecl(a: i32, b: i32) -> i32 {
    a + b
}

// stdcall/fastcall are decorated only on x86; elsewhere they are deprecated into "C".
#[cfg(target_arch = "x86")]
#[no_mangle]
pub extern "stdcall" fn decorated_stdcall(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(target_arch = "x86")]
#[no_mangle]
pub extern "fastcall" fn decorated_fastcall(a: i32, b: i32) -> i32 {
    a + b
}

// vectorcall is x86/x86_64-only and MSVC-only (GCC does not support it). A single `u64`
// argument is 8 bytes on both x86 (4-byte pointer width, rounded up) and x86_64, so its `@@N`
// decoration suffix is a stable `@@8` regardless of target.
#[cfg(all(target_env = "msvc", any(target_arch = "x86", target_arch = "x86_64")))]
#[no_mangle]
pub extern "vectorcall" fn decorated_vectorcall(x: u64) -> u64 {
    x
}

fn internal_decorated_helper() -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn decorated_calls_internal() -> i32 {
    internal_decorated_helper() + 1
}
