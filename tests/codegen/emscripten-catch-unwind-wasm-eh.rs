//@ compile-flags: -Copt-level=3 --target wasm32-unknown-emscripten -Z emscripten-wasm-eh
//@ needs-llvm-components: webassembly

// Emscripten catch_unwind using wasm exceptions

#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

impl<T> Copy for *mut T {}

#[rustc_intrinsic]
fn size_of<T>() -> usize {
    loop {}
}
#[rustc_intrinsic]
unsafe fn catch_unwind(
    try_fn: fn(_: *mut u8),
    data: *mut u8,
    catch_fn: fn(_: *mut u8, _: *mut u8),
) -> i32;

// CHECK-LABEL: @ptr_size
#[no_mangle]
pub fn ptr_size() -> usize {
    // CHECK: ret [[PTR_SIZE:.*]]
    size_of::<*mut u8>()
}

// CHECK-LABEL: @test_catch_unwind
#[no_mangle]
pub unsafe fn test_catch_unwind(
    try_fn: fn(_: *mut u8),
    data: *mut u8,
    catch_fn: fn(_: *mut u8, _: *mut u8),
) -> i32 {
    // CHECK: start:
    // CHECK: invoke void %try_fn(ptr %data)
    // CHECK:         to label %__rust_try.exit unwind label %catchswitch.i
    // CHECK:   catchswitch.i:                                    ; preds = %start
    // CHECK:   %catchswitch1.i = catchswitch within none [label %catchpad.i] unwind to caller

    // CHECK: catchpad.i:                                       ; preds = %catchswitch.i
    // CHECK:   %catchpad2.i = catchpad within %catchswitch1.i [ptr null]
    // CHECK:   %0 = tail call ptr @llvm.wasm.get.exception(token %catchpad2.i)
    // CHECK:   %1 = tail call i32 @llvm.wasm.get.ehselector(token %catchpad2.i)
    // CHECK:   call void %catch_fn(ptr %data, ptr %0) [ "funclet"(token %catchpad2.i) ]
    // CHECK:   catchret from %catchpad2.i to label %__rust_try.exit

    // CHECK: __rust_try.exit:                                  ; preds = %start, %catchpad.i
    // CHECK:   %common.ret.op.i = phi i32 [ 0, %start ], [ 1, %catchpad.i ]
    // CHECK:   ret i32 %common.ret.op.i

    catch_unwind(try_fn, data, catch_fn)
}
