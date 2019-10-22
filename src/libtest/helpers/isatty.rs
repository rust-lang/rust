//! Helper module which provides a function to test
//! if stdout is a tty.

#[cfg(any(
    target_os = "cloudabi",
    all(target_arch = "wasm32", not(target_os = "emscripten")),
    all(target_vendor = "fortanix", target_env = "sgx")
))]
pub fn stdout_isatty() -> bool {
    // FIXME: Implement isatty on SGX
    false
}
#[cfg(unix)]
pub fn stdout_isatty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 }
}
#[cfg(windows)]
pub fn stdout_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    type LPDWORD = *mut u32;
    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: LPDWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}
