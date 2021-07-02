//! Unwinding for *emscripten* target.
//!
//! Whereas Rust's usual unwinding implementation for Unix platforms
//! calls into the libunwind APIs directly, on Emscripten we instead
//! call into the C++ unwinding APIs. This is just an expedience since
//! Emscripten's runtime always implements those APIs and does not
//! implement libunwind.

use libc::c_int;
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

#[rustc_std_internal_symbol]
unsafe extern "C" fn rust_eh_personality(
    _version: c_int,
    _actions: uw::_Unwind_Action,
    _exception_class: uw::_Unwind_Exception_Class,
    _exception_object: *mut uw::_Unwind_Exception,
    _context: *mut uw::_Unwind_Context,
) -> uw::_Unwind_Reason_Code {
    crate::do_abort();
}
