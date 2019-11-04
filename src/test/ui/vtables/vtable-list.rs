// Only Windows, Linux, Android, macOS and iOS have this implemented for now.

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

#![feature(core_intrinsics)]
#![feature(ptr_offset_from)]
#![feature(raw)]
#![feature(test)]

use std::intrinsics::type_id;
use std::mem::transmute;
use std::raw::TraitObject;
use std::slice;

#[derive(Copy, Clone)]
#[repr(C, packed)]
struct Record {
    type_id: u64,
    vtable: &'static (),
}

fn vtables() -> &'static [Record] {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios"))]
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(
            any(target_os = "linux", target_os = "android"),
            link_name = "__start___rust_vtables"
        )]
        #[cfg_attr(
            any(target_os = "macos", target_os = "ios"),
            link_name = "\u{1}section$start$__DATA$__rust_vtables"
        )]
        static START: [Record; 0];
        #[cfg_attr(
            any(target_os = "linux", target_os = "android"),
            link_name = "__stop___rust_vtables"
        )]
        #[cfg_attr(
            any(target_os = "macos", target_os = "ios"),
            link_name = "\u{1}section$end$__DATA$__rust_vtables"
        )]
        static END: [Record; 0];
    }

    #[cfg(target_os = "windows")]
    {
        #[link_section = ".rdata.__rust_vtables$A"]
        static START: [Record; 0] = [];
        #[link_section = ".rdata.__rust_vtables$C"]
        static END: [Record; 0] = [];
    }

    unsafe {
        let (start_ptr, end_ptr) = (&START as *const Record, &END as *const Record);
        slice::from_raw_parts(start_ptr, end_ptr.offset_from(start_ptr) as usize)
    }
}

trait Trait {}
struct Struct;
impl Trait for Struct {}

#[inline(never)]
fn foo() {
    let a: &dyn Trait = &Struct;
    let b: &(dyn Trait + Send) = &Struct;
    std::hint::black_box((a, b));
}

fn main() {
    let vtable: &'static () = unsafe { &*transmute::<&dyn Trait, TraitObject>(&Struct).vtable };
    let type_id = unsafe { type_id::<dyn Trait>() };
    let count = vtables().iter().filter(|&&record| record.type_id == type_id).count();
    assert_ne!(count, 0, "The vtable record for dyn Trait is missing");
    assert_eq!(count, 1, "Duplicate vtable records found for dyn Trait");
    let record = vtables().iter().find(|&&record| record.vtable as *const () == vtable).unwrap();
    assert_eq!(
        record.vtable as *const (), vtable,
        "The vtable for Struct as dyn Trait is incorrect"
    );
    foo();
}
