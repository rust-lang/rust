//@ run-pass
//@ compile-flags: -Ctarget-feature=-crt-static -Zexport-executable-symbols
//@ ignore-wasm
//@ ignore-cross-compile
//@ edition: 2024

// Regression test for <https://github.com/rust-lang/rust/issues/101610>.

#![feature(rustc_private)]

#[unsafe(no_mangle)]
fn hack() -> u64 {
    998244353
}

fn main() {
    #[cfg(unix)]
    unsafe {
        extern crate libc;
        let handle = libc::dlopen(std::ptr::null(), libc::RTLD_NOW);
        let ptr = libc::dlsym(handle, c"hack".as_ptr());
        let ptr: Option<unsafe fn() -> u64> = std::mem::transmute(ptr);
        if let Some(f) = ptr {
            assert!(f() == 998244353);
            println!("symbol `hack` is found successfully");
        } else {
            panic!("symbol `hack` is not found");
        }
    }
    #[cfg(windows)]
    unsafe {
        type PCSTR = *const u8;
        type HMODULE = *mut core::ffi::c_void;
        type FARPROC = Option<unsafe extern "system" fn() -> isize>;
        #[link(name = "kernel32", kind = "raw-dylib")]
        unsafe extern "system" {
            fn GetModuleHandleA(lpmodulename: PCSTR) -> HMODULE;
            fn GetProcAddress(hmodule: HMODULE, lpprocname: PCSTR) -> FARPROC;
        }
        let handle = GetModuleHandleA(std::ptr::null_mut());
        let ptr = GetProcAddress(handle, b"hack\0".as_ptr());
        let ptr: Option<unsafe fn() -> u64> = std::mem::transmute(ptr);
        if let Some(f) = ptr {
            assert!(f() == 998244353);
            println!("symbol `hack` is found successfully");
        } else {
            panic!("symbol `hack` is not found");
        }
    }
}
