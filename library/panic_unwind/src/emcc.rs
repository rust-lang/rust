//! Unwinding for *emscripten* target.
//!
//! Whereas Rust's usual unwinding implementation for Unix platforms
//! calls into the libunwind APIs directly, on Emscripten we instead
//! call into the C++ unwinding APIs. This is just an expedience since
//! Emscripten's runtime always implements those APIs and does not
//! implement libunwind.

use alloc::boxed::Box;
use core::any::Any;
use core::sync::atomic::{AtomicBool, Ordering};
use core::{intrinsics, mem, ptr};

use unwind as uw;

// This matches the layout of std::type_info in C++
#[repr(C)]
struct TypeInfo {
    vtable: *const usize,
    name: *const u8,
}
unsafe impl Sync for TypeInfo {}

unsafe extern "C" {
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

// NOTE(nbdd0121): The `canary` field is part of stable ABI.
#[repr(C)]
struct Exception {
    // See `gcc.rs` on why this is present. We already have a static here so just use it.
    canary: *const TypeInfo,

    // This is necessary because C++ code can capture our exception with
    // std::exception_ptr and rethrow it multiple times, possibly even in
    // another thread.
    caught: AtomicBool,

    // This needs to be an Option because the object's lifetime follows C++
    // semantics: when catch_unwind moves the Box out of the exception it must
    // still leave the exception object in a valid state because its destructor
    // is still going to be called by __cxa_end_catch.
    data: Option<Box<dyn Any + Send>>,
}

pub(crate) unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    // intrinsics::try actually gives us a pointer to this structure.
    #[repr(C)]
    struct CatchData {
        ptr: *mut u8,
        is_rust_panic: bool,
    }
    let catch_data = &*(ptr as *mut CatchData);

    let adjusted_ptr = __cxa_begin_catch(catch_data.ptr as *mut libc::c_void) as *mut Exception;
    if !catch_data.is_rust_panic {
        super::__rust_foreign_exception();
    }

    let canary = (&raw const (*adjusted_ptr).canary).read();
    if !ptr::eq(canary, &EXCEPTION_TYPE_INFO) {
        super::__rust_foreign_exception();
    }

    let was_caught = (*adjusted_ptr).caught.swap(true, Ordering::Relaxed);
    if was_caught {
        // Since cleanup() isn't allowed to panic, we just abort instead.
        intrinsics::abort();
    }
    let out = (*adjusted_ptr).data.take().unwrap();
    __cxa_end_catch();
    out
}

pub(crate) unsafe fn panic(data: Box<dyn Any + Send>) -> u32 {
    let exception = __cxa_allocate_exception(mem::size_of::<Exception>()) as *mut Exception;
    if exception.is_null() {
        return uw::_URC_FATAL_PHASE1_ERROR as u32;
    }
    ptr::write(
        exception,
        Exception {
            canary: &EXCEPTION_TYPE_INFO,
            caught: AtomicBool::new(false),
            data: Some(data),
        },
    );
    __cxa_throw(exception as *mut _, &EXCEPTION_TYPE_INFO, exception_cleanup);
}

extern "C" fn exception_cleanup(ptr: *mut libc::c_void) -> *mut libc::c_void {
    unsafe {
        if let Some(b) = (ptr as *mut Exception).read().data {
            drop(b);
            super::__rust_drop_panic();
        }
        ptr
    }
}

unsafe extern "C" {
    fn __cxa_allocate_exception(thrown_size: libc::size_t) -> *mut libc::c_void;
    fn __cxa_begin_catch(thrown_exception: *mut libc::c_void) -> *mut libc::c_void;
    fn __cxa_end_catch();
    fn __cxa_throw(
        thrown_exception: *mut libc::c_void,
        tinfo: *const TypeInfo,
        dest: extern "C" fn(*mut libc::c_void) -> *mut libc::c_void,
    ) -> !;
}
