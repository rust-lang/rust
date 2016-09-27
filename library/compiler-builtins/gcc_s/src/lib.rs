#![feature(drop_types_in_const)]

extern crate libloading;

use std::sync::{Once, ONCE_INIT};

use libloading::Library;

static mut GCC_S: Option<Library> = None;

#[cfg(not(windows))]
fn gcc_s() -> &'static Library {
    #[cfg(not(target_os = "macos"))]
    const LIBGCC_S: &'static str = "libgcc_s.so.1";

    #[cfg(target_os = "macos")]
    const LIBGCC_S: &'static str = "libgcc_s.1.dylib";

    unsafe {
        static INIT: Once = ONCE_INIT;

        INIT.call_once(|| {
            GCC_S = Some(Library::new(LIBGCC_S).unwrap());
        });
        GCC_S.as_ref().unwrap()
    }
}

#[cfg(windows)]
pub fn get(_sym: &str) -> Option<usize> {
    None
}

#[cfg(not(windows))]
pub fn get(sym: &str) -> Option<usize> {
    unsafe {
        gcc_s().get(sym.as_bytes()).ok().map(|s| *s)
    }
}
