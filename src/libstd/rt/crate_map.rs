// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast;
use option::{Some, None, Option};
use ptr::RawPtr;
use rt::rtio::EventLoop;

// Need to tell the linker on OS X to not barf on undefined symbols
// and instead look them up at runtime, which we need to resolve
// the crate_map properly.
#[cfg(target_os = "macos")]
#[link_args = "-Wl,-U,__rust_crate_map_toplevel"]
extern {}

pub struct CrateMap<'a> {
    version: i32,
    event_loop_factory: Option<fn() -> ~EventLoop>,
}

// When working on android, apparently weak symbols don't work so well for
// finding the crate map, and neither does dlopen + dlsym. This is mainly a
// problem when integrating a shared library with an existing application.
// Standalone binaries do not appear to have this problem. The reasons are a
// little mysterious, and more information can be found in #11731.
//
// For now we provide a way to tell libstd about the crate map manually that's
// checked before the normal weak symbol/dlopen paths. In theory this is useful
// on other platforms where our dlopen/weak linkage strategy mysteriously fails
// but the crate map can be specified manually.
static mut MANUALLY_PROVIDED_CRATE_MAP: *CrateMap<'static> =
                                                    0 as *CrateMap<'static>;
#[no_mangle]
#[cfg(not(test))]
pub extern fn rust_set_crate_map(map: *CrateMap<'static>) {
    unsafe { MANUALLY_PROVIDED_CRATE_MAP = map; }
}

fn manual_crate_map() -> Option<&'static CrateMap<'static>> {
    unsafe {
        if MANUALLY_PROVIDED_CRATE_MAP.is_null() {
            None
        } else {
            Some(cast::transmute(MANUALLY_PROVIDED_CRATE_MAP))
        }
    }
}

#[cfg(not(windows))]
pub fn get_crate_map() -> Option<&'static CrateMap<'static>> {
    extern {
        #[crate_map]
        static CRATE_MAP: CrateMap<'static>;
    }

    manual_crate_map().or_else(|| {
        let ptr: (*CrateMap) = &'static CRATE_MAP;
        if ptr.is_null() {
            None
        } else {
            Some(&'static CRATE_MAP)
        }
    })
}

#[cfg(windows)]
pub fn get_crate_map() -> Option<&'static CrateMap<'static>> {
    use c_str::ToCStr;
    use unstable::dynamic_lib::dl;

    match manual_crate_map() {
        Some(cm) => return Some(cm),
        None => {}
    }

    let sym = unsafe {
        let module = dl::open_internal();
        let rust_crate_map_toplevel = if cfg!(target_arch = "x86") {
            "__rust_crate_map_toplevel"
        } else {
            "_rust_crate_map_toplevel"
        };
        let sym = rust_crate_map_toplevel.with_c_str(|buf| {
            dl::symbol(module, buf)
        });
        dl::close(module);
        sym
    };
    let ptr: (*CrateMap) = sym as *CrateMap;
    if ptr.is_null() {
        return None;
    } else {
        unsafe {
            return Some(cast::transmute(sym));
        }
    }
}
