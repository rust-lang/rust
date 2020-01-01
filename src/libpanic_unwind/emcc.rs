//! Unwinding for *emscripten* target.
//!
//! Whereas Rust's usual unwinding implementation for Unix platforms
//! calls into the libunwind APIs directly, on Emscripten we instead
//! call into the C++ unwinding APIs. This is just an expedience since
//! Emscripten's runtime always implements those APIs and does not
//! implement libunwind.

#![allow(private_no_mangle_fns)]

use alloc::boxed::Box;
use core::any::Any;
use core::mem;
use core::ptr;
use libc::{self, c_int};
use unwind as uw;

// This matches the layout of std::type_info in C++
#[repr(C)]
struct TypeInfo {
    vtable: *const usize,
    name: *const u8,
}
unsafe impl Sync for TypeInfo {}

extern "C" {
    // The leading `\x01` byte here is actually a magical signal to LLVM to
    // *not* apply any other mangling like prefixing with a `_` character.
    //
    // This symbol is the vtable used by C++'s `std::type_info`. Objects of type
    // `std::type_info`, type descriptors, have a pointer to this table. Type
    // descriptors are referenced by the C++ EH structures defined above and
    // that we construct below.
    //
    // Note that the real size is larger than 3 usize, but we only need our
    // vtable to point to the third element.
    #[link_name = "\x01_ZTVN10__cxxabiv117__class_type_infoE"]
    static CLASS_TYPE_INFO_VTABLE: [usize; 3];
}

// std::type_info for a rust_panic class
#[lang = "eh_catch_typeinfo"]
static EXCEPTION_TYPE_INFO: TypeInfo = TypeInfo {
    // Normally we would use .as_ptr().add(2) but this doesn't work in a const context.
    vtable: unsafe { &CLASS_TYPE_INFO_VTABLE[2] },
    // This intentionally doesn't use the normal name mangling scheme because
    // we don't want C++ to be able to produce or catch Rust panics.
    name: b"rust_panic\0".as_ptr(),
};

pub fn payload() -> *mut u8 {
    ptr::null_mut()
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    assert!(!ptr.is_null());
    let adjusted_ptr = __cxa_begin_catch(ptr as *mut libc::c_void);
    let ex = ptr::read(adjusted_ptr as *mut _);
    __cxa_end_catch();
    ex
}

pub unsafe fn panic(data: Box<dyn Any + Send>) -> u32 {
    let sz = mem::size_of_val(&data);
    let exception = __cxa_allocate_exception(sz);
    if exception == ptr::null_mut() {
        return uw::_URC_FATAL_PHASE1_ERROR as u32;
    }
    ptr::write(exception as *mut _, data);
    __cxa_throw(exception as *mut _, &EXCEPTION_TYPE_INFO, ptr::null_mut());
}

#[lang = "eh_personality"]
#[no_mangle]
unsafe extern "C" fn rust_eh_personality(
    version: c_int,
    actions: uw::_Unwind_Action,
    exception_class: uw::_Unwind_Exception_Class,
    exception_object: *mut uw::_Unwind_Exception,
    context: *mut uw::_Unwind_Context,
) -> uw::_Unwind_Reason_Code {
    __gxx_personality_v0(version, actions, exception_class, exception_object, context)
}

extern "C" {
    fn __cxa_allocate_exception(thrown_size: libc::size_t) -> *mut libc::c_void;
    fn __cxa_begin_catch(thrown_exception: *mut libc::c_void) -> *mut libc::c_void;
    fn __cxa_end_catch();
    fn __cxa_throw(
        thrown_exception: *mut libc::c_void,
        tinfo: *const TypeInfo,
        dest: *mut libc::c_void,
    ) -> !;
    fn __gxx_personality_v0(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        exception_object: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context,
    ) -> uw::_Unwind_Reason_Code;
}
