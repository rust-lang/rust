/// Verify that we can load a cdylib, call functions from it, and then unload it.

#[cfg(windows)]
mod libloading {
    type BOOL = i32;
    type DWORD = u32;
    type HANDLE = isize;
    pub type HMODULE = isize;
    type FARPROC = *mut core::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(filename: *const u16, file: HANDLE, flags: DWORD) -> HMODULE;
        fn FreeLibrary(module: HMODULE) -> BOOL;
        fn GetProcAddress(module: HMODULE, procname: *const u8) -> FARPROC;
    }

    fn wide_null(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(core::iter::once(0)).collect()
    }

    fn ansi_null(s: &str) -> Vec<u8> {
        assert!(!s.as_bytes().contains(&0), "symbol name must not contain interior NUL");
        s.as_bytes().iter().copied().chain(core::iter::once(0)).collect()
    }

    pub fn load(lib_name: &str) -> Option<HMODULE> {
        let filename = wide_null(&format!("{}.dll", lib_name));
        let handle = unsafe { LoadLibraryExW(filename.as_ptr(), 0, 0) };
        if handle == 0 { None } else { Some(handle) }
    }

    pub fn get_symbol(handle: HMODULE, name: &str) -> Option<FARPROC> {
        let symbol_name = ansi_null(name);
        let symbol = unsafe { GetProcAddress(handle, symbol_name.as_ptr()) };
        if symbol.is_null() { None } else { Some(symbol) }
    }

    pub fn unload(handle: HMODULE) {
        unsafe {
            FreeLibrary(handle);
        }
    }
}

#[cfg(unix)]
mod libloading {
    use std::ffi::{c_char, c_int, c_void};

    const RTLD_NOW: c_int = 2;

    fn cstr_null(s: &str) -> Vec<c_char> {
        assert!(!s.as_bytes().contains(&0), "string must not contain interior NUL");
        s.as_bytes().iter().copied().chain(core::iter::once(0)).map(|b| b as c_char).collect()
    }

    pub fn load(lib_name: &str) -> Option<*mut c_void> {
        #[cfg(target_os = "macos")]
        let filename = cstr_null(&format!("lib{}.dylib", lib_name));

        #[cfg(not(target_os = "macos"))]
        let filename = cstr_null(&format!("./lib{}.so", lib_name));

        let handle = unsafe { dlopen(filename.as_ptr(), RTLD_NOW) };
        if handle.is_null() { None } else { Some(handle) }
    }

    pub fn get_symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let symbol_name = cstr_null(name);
        let symbol = unsafe { dlsym(handle, symbol_name.as_ptr()) };
        if symbol.is_null() { None } else { Some(symbol) }
    }

    pub fn unload(handle: *mut c_void) {
        unsafe {
            dlclose(handle);
        }
    }

    #[cfg_attr(any(target_os = "linux", target_os = "android"), link(name = "dl"))]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: core::ffi::c_int) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> core::ffi::c_int;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

type ExternFn = unsafe extern "C" fn(u32, u32) -> u32;

fn main() {
    let foo = libloading::load("foo").expect("Failed to load library");
    println!("loaded library");

    let extern_fn_1 = libloading::get_symbol(foo, "extern_fn_1").expect("Failed to find symbol");
    let extern_fn_1: ExternFn = unsafe { std::mem::transmute(extern_fn_1) };
    let result = unsafe { extern_fn_1(2, 3) };
    println!("result of extern_fn_1(2, 3): {}", result);

    let extern_fn_2 = libloading::get_symbol(foo, "extern_fn_2").expect("Failed to find symbol");
    let extern_fn_2: ExternFn = unsafe { std::mem::transmute(extern_fn_2) };
    let result = unsafe { extern_fn_2(2, 3) };
    println!("result of extern_fn_2(2, 3): {}", result);

    libloading::unload(foo);
    println!("unloaded library");
}
