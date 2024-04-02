//@ assembly-output: emit-asm
//@ only-x86
// Force frame pointers to make ASM more consistent between targets
//@ compile-flags: -O -C force-frame-pointers
//@ filecheck-flags: --implicit-check-not fld --implicit-check-not fst
//@ revisions: unix windows
//@[unix] ignore-windows
//@[windows] only-windows

#![crate_type = "lib"]
#![feature(f16, f128)]

// Tests that returning `f32` and `f64` with the "Rust" ABI on 32-bit x86 doesn't use the x87
// floating point stack, as loading and storing `f32`s and `f64`s to and from the x87 stack quietens
// signalling NaNs.

// Returning individual floats

// CHECK-LABEL: return_f32:
#[no_mangle]
pub fn return_f32(x: f32) -> f32 {
    // CHECK: movl {{.*}}(%ebp), %eax
    // CHECK-NOT: ax
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f64:
#[no_mangle]
pub fn return_f64(x: f64) -> f64 {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL:.*]]
    // CHECK-NEXT: movsd %[[VAL]], (%[[PTR]])
    // CHECK: retl
    x
}

// Returning scalar pairs containing floats

// CHECK-LABEL: return_f32_f32:
#[no_mangle]
pub fn return_f32_f32(x: (f32, f32)) -> (f32, f32) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movss %[[VAL2]], 4(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f64_f64:
#[no_mangle]
pub fn return_f64_f64(x: (f64, f64)) -> (f64, f64) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+12]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movsd %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movsd %[[VAL2]], 8(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f32_f64:
#[no_mangle]
pub fn return_f32_f64(x: (f32, f64)) -> (f32, f64) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movsd %[[VAL2]], {{4|8}}(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f64_f32:
#[no_mangle]
pub fn return_f64_f32(x: (f64, f32)) -> (f64, f32) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+12]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movsd %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movss %[[VAL2]], 8(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f32_other:
#[no_mangle]
pub fn return_f32_other(x: (f32, usize)) -> (f32, usize) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movl %[[VAL2]], 4(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f64_other:
#[no_mangle]
pub fn return_f64_other(x: (f64, usize)) -> (f64, usize) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+12]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movsd %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movl %[[VAL2]], 8(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_other_f32:
#[no_mangle]
pub fn return_other_f32(x: (usize, f32)) -> (usize, f32) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movl %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movss %[[VAL2]], 4(%[[PTR]])
    // CHECK: retl
    x
}

// CHECK-LABEL: return_other_f64:
#[no_mangle]
pub fn return_other_f64(x: (usize, f64)) -> (usize, f64) {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movl %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movsd %[[VAL2]], {{4|8}}(%[[PTR]])
    // CHECK: retl
    x
}

// Calling functions returning floats

// CHECK-LABEL: call_f32:
#[no_mangle]
pub unsafe fn call_f32(x: &mut f32) {
    extern "Rust" {
        fn get_f32() -> f32;
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f32
    // CHECK-NEXT: movl %eax, (%[[PTR]])
    *x = get_f32();
}

// CHECK-LABEL: call_f64:
#[no_mangle]
pub unsafe fn call_f64(x: &mut f64) {
    extern "Rust" {
        fn get_f64() -> f64;
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f64
    // CHECK: movsd {{.*}}(%{{ebp|esp}}), %[[VAL:.*]]
    // CHECK-NEXT: movsd %[[VAL:.*]], (%[[PTR]])
    *x = get_f64();
}

// Calling functions returning scalar pairs containing floats

// CHECK-LABEL: call_f32_f32:
#[no_mangle]
pub unsafe fn call_f32_f32(x: &mut (f32, f32)) {
    extern "Rust" {
        fn get_f32_f32() -> (f32, f32);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f32_f32
    // CHECK: movss [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movss %[[VAL2]], 4(%[[PTR]])
    *x = get_f32_f32();
}

// CHECK-LABEL: call_f64_f64:
#[no_mangle]
pub unsafe fn call_f64_f64(x: &mut (f64, f64)) {
    extern "Rust" {
        fn get_f64_f64() -> (f64, f64);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f64_f64
    // unix: movsd [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // unix-NEXT: movsd [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // windows: movsd (%esp), %[[VAL1:.*]]
    // windows-NEXT: movsd 8(%esp), %[[VAL2:.*]]
    // CHECK-NEXT: movsd %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movsd %[[VAL2]], 8(%[[PTR]])
    *x = get_f64_f64();
}

// CHECK-LABEL: call_f32_f64:
#[no_mangle]
pub unsafe fn call_f32_f64(x: &mut (f32, f64)) {
    extern "Rust" {
        fn get_f32_f64() -> (f32, f64);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f32_f64
    // unix: movss [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // unix-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // windows: movss (%esp), %[[VAL1:.*]]
    // windows-NEXT: movsd 8(%esp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // unix-NEXT: movsd %[[VAL2]], 4(%[[PTR]])
    // windows-NEXT: movsd %[[VAL2]], 8(%[[PTR]])
    *x = get_f32_f64();
}

// CHECK-LABEL: call_f64_f32:
#[no_mangle]
pub unsafe fn call_f64_f32(x: &mut (f64, f32)) {
    extern "Rust" {
        fn get_f64_f32() -> (f64, f32);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f64_f32
    // unix: movsd [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // unix-NEXT: movss [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // windows: movsd (%esp), %[[VAL1:.*]]
    // windows-NEXT: movss 8(%esp), %[[VAL2:.*]]
    // CHECK-NEXT: movsd %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movss %[[VAL2]], 8(%[[PTR]])
    *x = get_f64_f32();
}

// CHECK-LABEL: call_f32_other:
#[no_mangle]
pub unsafe fn call_f32_other(x: &mut (f32, usize)) {
    extern "Rust" {
        fn get_f32_other() -> (f32, usize);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f32_other
    // CHECK: movss [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movl %[[VAL2]], 4(%[[PTR]])
    *x = get_f32_other();
}

// CHECK-LABEL: call_f64_other:
#[no_mangle]
pub unsafe fn call_f64_other(x: &mut (f64, usize)) {
    extern "Rust" {
        fn get_f64_other() -> (f64, usize);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_f64_other
    // unix: movsd [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // unix-NEXT: movl [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // windows: movsd (%esp), %[[VAL1:.*]]
    // windows-NEXT: movl 8(%esp), %[[VAL2:.*]]
    // CHECK-NEXT: movsd %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movl %[[VAL2]], 8(%[[PTR]])
    *x = get_f64_other();
}

// CHECK-LABEL: call_other_f32:
#[no_mangle]
pub unsafe fn call_other_f32(x: &mut (usize, f32)) {
    extern "Rust" {
        fn get_other_f32() -> (usize, f32);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_other_f32
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movl %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movss %[[VAL2]], 4(%[[PTR]])
    *x = get_other_f32();
}

// CHECK-LABEL: call_other_f64:
#[no_mangle]
pub unsafe fn call_other_f64(x: &mut (usize, f64)) {
    extern "Rust" {
        fn get_other_f64() -> (usize, f64);
    }
    // CHECK: movl {{.*}}(%ebp), %[[PTR:.*]]
    // CHECK: calll {{()|_}}get_other_f64
    // unix: movl [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // unix-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // windows: movl (%esp), %[[VAL1:.*]]
    // windows-NEXT: movsd 8(%esp), %[[VAL2:.*]]
    // CHECK-NEXT: movl %[[VAL1]], (%[[PTR]])
    // unix-NEXT: movsd %[[VAL2]], 4(%[[PTR]])
    // windows-NEXT: movsd %[[VAL2]], 8(%[[PTR]])
    *x = get_other_f64();
}

// The "C" ABI for `f16` and `f128` on x86 has never used the x87 floating point stack. Do some
// basic checks to ensure this remains the case for the "Rust" ABI.

// CHECK-LABEL: return_f16:
#[no_mangle]
pub fn return_f16(x: f16) -> f16 {
    // CHECK: pinsrw $0, {{.*}}(%ebp), %xmm0
    // CHECK-NOT: xmm0
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f128:
#[no_mangle]
pub fn return_f128(x: f128) -> f128 {
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+16]](%ebp), %[[VAL4:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+12]](%ebp), %[[VAL3:.*]]
    // CHECK-NEXT: movl %[[VAL4:.*]] 12(%[[PTR]])
    // CHECK-NEXT: movl %[[VAL3:.*]] 8(%[[PTR]])
    // CHECK-NEXT: movl %[[VAL2:.*]] 4(%[[PTR]])
    // CHECK-NEXT: movl %[[VAL1:.*]] (%[[PTR]])
    // CHECK: retl
    x
}
