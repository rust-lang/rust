#[cfg(bin_type)]
pub fn crate_type() -> &'static str { "bin" }

#[cfg(lib_type)]
pub fn crate_type() -> &'static str { "lib" }

#[cfg(dylib_type)]
pub fn crate_type() -> &'static str { "dylib" }

#[cfg(rlib_type)]
pub fn crate_type() -> &'static str { "rlib" }

// (The cases below are not used in any of these tests yet. But it might be good
// to think about adding them.)

#[cfg(staticlib_type)]
pub extern "C" fn crate_type() -> *const u8 { "staticlib\0".as_ptr() }

#[cfg(cdylib_type)]
pub extern "C" fn crate_type() -> *const u8 { "cdylib\0".as_ptr() }
