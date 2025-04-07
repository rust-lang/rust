//@ compile-flags: -Copt-level=3 --target wasm32-unknown-emscripten
//@ needs-llvm-components: webassembly

// Emscripten has its own unique implementation of catch_unwind (in `codegen_emcc_try`),
// make sure it generates something reasonable.

#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}
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
    // CHECK: [[ALLOCA:%.*]] = alloca

    // CHECK: catch.i:
    // CHECK: [[LANDINGPAD:%.*]] = landingpad
    // CHECK: [[EXCEPTION:%.*]] = extractvalue {{.*}} [[LANDINGPAD]], 0
    // CHECK: [[SELECTOR:%.*]] = extractvalue {{.*}} [[LANDINGPAD]], 1

    // CHECK: [[IS_RUST_EXN:%.*]] = icmp eq {{.*}}[[SELECTOR]]
    // CHECK: [[IS_RUST_EXN_I8:%.*]] = zext i1 [[IS_RUST_EXN]] to i8

    // CHECK: store ptr [[EXCEPTION]], ptr [[ALLOCA]]
    // CHECK: [[IS_RUST_SLOT:%.*]] = getelementptr inbounds{{( nuw)?}} i8, ptr [[ALLOCA]], [[PTR_SIZE]]
    // CHECK: store i8 [[IS_RUST_EXN_I8]], ptr [[IS_RUST_SLOT]]

    // CHECK: call void %catch_fn(ptr %data, ptr nonnull [[ALLOCA]])

    catch_unwind(try_fn, data, catch_fn)
}
