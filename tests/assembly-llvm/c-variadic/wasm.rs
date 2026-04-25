//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: WASM32 WASM64
//@ [WASM32] compile-flags: -Copt-level=3 -Zmerge-functions=disabled --target wasm32-unknown-unknown
//@ [WASM32] needs-llvm-components: webassembly
//@ [WASM64] compile-flags: -Copt-level=3 -Zmerge-functions=disabled --target wasm64-unknown-unknown
//@ [WASM64] needs-llvm-components: webassembly
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
unsafe impl VaArgSafe for i128 {}
unsafe impl VaArgSafe for f64 {}
unsafe impl<T> VaArgSafe for *const T {}

#[repr(transparent)]
struct VaListInner {
    ptr: *const c_void,
}

#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomData<&'a mut ()>,
}

#[rustc_intrinsic]
#[rustc_nounwind]
pub const unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaList<'_>) -> T;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // WASM32-LABEL: read_f64:
    // WASM32: local.get 0
    // WASM32-NEXT: local.get 0
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: i32.const 7
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.const -8
    // WASM32-NEXT: i32.and
    // WASM32-NEXT: local.tee 1
    // WASM32-NEXT: i32.const 8
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.store 0
    // WASM32-NEXT: local.get 1
    // WASM32-NEXT: f64.load 0
    // WASM32-NEXT: end_function
    //
    // WASM64-LABEL: read_f64:
    // WASM64: local.get 0
    // WASM64-NEXT: local.get 0
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: i64.const 7
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.const -8
    // WASM64-NEXT: i64.and
    // WASM64-NEXT: local.tee 1
    // WASM64-NEXT: i64.const 8
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.store 0
    // WASM64-NEXT: local.get 1
    // WASM64-NEXT: f64.load 0
    // WASM64-NEXT: end_function
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // WASM32-LABEL: read_i32:
    // WASM32: local.get 0
    // WASM32-NEXT: local.get 0
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: local.tee 1
    // WASM32-NEXT: i32.const 4
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.store 0
    // WASM32-NEXT: local.get 1
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: end_function
    //
    // WASM64-LABEL: read_i32:
    // WASM64: local.get 0
    // WASM64-NEXT: local.get 0
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: local.tee 1
    // WASM64-NEXT: i64.const 4
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.store 0
    // WASM64-NEXT: local.get 1
    // WASM64-NEXT: i32.load 0
    // WASM64-NEXT: end_function
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // WASM32-LABEL: read_ptr:
    // WASM32: local.get 0
    // WASM32-NEXT: local.get 0
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: local.tee 1
    // WASM32-NEXT: i32.const 4
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.store 0
    // WASM32-NEXT: local.get 1
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: end_function
    //
    // WASM64-LABEL: read_ptr:
    // WASM64: local.get 0
    // WASM64-NEXT: local.get 0
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: i64.const 7
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.const -8
    // WASM64-NEXT: i64.and
    // WASM64-NEXT: local.tee 1
    // WASM64-NEXT: i64.const 8
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.store 0
    // WASM64-NEXT: local.get 1
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: end_function
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // WASM32-LABEL: read_i64:
    // WASM32: local.get 0
    // WASM32-NEXT: local.get 0
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: i32.const 7
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.const -8
    // WASM32-NEXT: i32.and
    // WASM32-NEXT: local.tee 1
    // WASM32-NEXT: i32.const 8
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.store 0
    // WASM32-NEXT: local.get 1
    // WASM32-NEXT: i64.load 0
    // WASM32-NEXT: end_function
    //
    // WASM64-LABEL: read_i64:
    // WASM64: local.get 0
    // WASM64-NEXT: local.get 0
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: i64.const 7
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.const -8
    // WASM64-NEXT: i64.and
    // WASM64-NEXT: local.tee 1
    // WASM64-NEXT: i64.const 8
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.store 0
    // WASM64-NEXT: local.get 1
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: end_function
    va_arg(ap)
}

// Clang and Rustc use a different ABI for i128 on wasm32, and LLVM optimizes differently if we use
// a mutable reference instead of just a pointer. With this setup we match the equivalent Clang
// input exactly.
#[unsafe(no_mangle)]
unsafe extern "C" fn read_i128(out: *mut i128, ap: *mut VaList<'_>) {
    // WASM32-LABEL: read_i128:
    // WASM32: local.get 1
    // WASM32-NEXT: local.get 1
    // WASM32-NEXT: i32.load 0
    // WASM32-NEXT: i32.const 15
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.const -16
    // WASM32-NEXT: i32.and
    // WASM32-NEXT: local.tee 2
    // WASM32-NEXT: i32.const 16
    // WASM32-NEXT: i32.add
    // WASM32-NEXT: i32.store 0
    // WASM32-NEXT: local.get 0
    // WASM32-NEXT: local.get 2
    // WASM32-NEXT: i64.load 0
    // WASM32-NEXT: i64.store 0
    // WASM32-NEXT: local.get 0
    // WASM32-NEXT: local.get 2
    // WASM32-NEXT: i64.load 8
    // WASM32-NEXT: i64.store 8
    // WASM32-NEXT: end_function
    //
    // WASM64-LABEL: read_i128:
    // WASM64: local.get 1
    // WASM64-NEXT: local.get 1
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: i64.const 15
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.const -16
    // WASM64-NEXT: i64.and
    // WASM64-NEXT: local.tee 2
    // WASM64-NEXT: i64.const 16
    // WASM64-NEXT: i64.add
    // WASM64-NEXT: i64.store 0
    // WASM64-NEXT: local.get 0
    // WASM64-NEXT: local.get 2
    // WASM64-NEXT: i64.load 0
    // WASM64-NEXT: i64.store 0
    // WASM64-NEXT: local.get 0
    // WASM64-NEXT: local.get 2
    // WASM64-NEXT: i64.load 8
    // WASM64-NEXT: i64.store 8
    // WASM64-NEXT: end_function
    *out = va_arg(mem::transmute(ap));
}
