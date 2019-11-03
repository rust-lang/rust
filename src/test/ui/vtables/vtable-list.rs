// For windows, linux, android, macos and ios only

// run-pass

// ignore-cloudabi
// ignore-dragonfly
// ignore-emscripten
// ignore-freebsd
// ignore-haiku
// ignore-netbsd
// ignore-openbsd
// ignore-solaris
// ignore-sgx

#![feature(ptr_offset_from)]
#![feature(raw)]

use std::{mem::transmute, raw::TraitObject, slice};

#[allow(improper_ctypes)]
fn vtables() -> &'static [&'static ()] {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios"))]
    extern "C" {
        #[cfg_attr(
            any(target_os = "linux", target_os = "android"),
            link_name = "__start___rust_vtables"
        )]
        #[cfg_attr(
            any(target_os = "macos", target_os = "ios"),
            link_name = "\u{1}section$start$__DATA$__rust_vtables"
        )]
        static START: [&'static (); 0];
        #[cfg_attr(
            any(target_os = "linux", target_os = "android"),
            link_name = "__stop___rust_vtables"
        )]
        #[cfg_attr(
            any(target_os = "macos", target_os = "ios"),
            link_name = "\u{1}section$end$__DATA$__rust_vtables"
        )]
        static END: [&'static (); 0];
    }

    #[cfg(target_os = "windows")]
    {
        #[link_section = ".rdata.__rust_vtables$A"]
        static START: [&'static (); 0] = [];
        #[link_section = ".rdata.__rust_vtables$C"]
        static END: [&'static (); 0] = [];
    }

    unsafe {
        let (start_ptr, end_ptr) = (&START as *const &'static (), &END as *const &'static ());
        slice::from_raw_parts(start_ptr, end_ptr.offset_from(start_ptr) as usize)
    }
}

trait Trait {}
struct Foo;
impl Trait for Foo {}

fn main() {
    let vtable: &'static () = unsafe { &*transmute::<&dyn Trait, TraitObject>(&Foo).vtable };
    vtables().iter().find(|&&x| x as *const () == vtable as *const ()).unwrap();
}
