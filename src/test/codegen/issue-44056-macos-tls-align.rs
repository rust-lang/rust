// ignore-tidy-linelength
// only-macos
// no-system-llvm
// compile-flags: -O

#![crate_type = "rlib"]
#![feature(thread_local)]

// CHECK: @STATIC_VAR_1 = thread_local local_unnamed_addr global <{ [32 x i8] }> zeroinitializer, section "__DATA,__thread_bss", align 4
#[no_mangle]
#[thread_local]
static mut STATIC_VAR_1: [u32; 8] = [0; 8];

// CHECK: @STATIC_VAR_2 = thread_local local_unnamed_addr global <{ [32 x i8] }> <{{[^>]*}}>, section "__DATA,__thread_data", align 4
#[no_mangle]
#[thread_local]
static mut STATIC_VAR_2: [u32; 8] = [4; 8];

#[no_mangle]
pub unsafe fn f(x: &mut [u32; 8]) {
    std::mem::swap(x, &mut STATIC_VAR_1)
}

#[no_mangle]
pub unsafe fn g(x: &mut [u32; 8]) {
    std::mem::swap(x, &mut STATIC_VAR_2)
}
