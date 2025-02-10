//@ assembly-output: emit-asm
//@ revisions: sse nosse
//@[sse] compile-flags: --target i686-unknown-linux-gnu
//@[sse] needs-llvm-components: x86
// We make SSE available but don't use it for the ABI.
//@[nosse] compile-flags: --target i586-unknown-linux-gnu -Ctarget-feature=+sse2 -Ctarget-cpu=pentium4
//@[nosse] needs-llvm-components: x86

// Force frame pointers to make ASM more consistent between targets
//@ compile-flags: -O -C force-frame-pointers
//@ filecheck-flags: --implicit-check-not fld --implicit-check-not fst

#![feature(f16, f128)]
#![feature(no_core, lang_items, rustc_attrs, repr_simd)]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

impl Copy for f16 {}
impl Copy for f32 {}
impl Copy for f64 {}
impl Copy for f128 {}
impl Copy for usize {}

// Tests that returning `f32` and `f64` with the "Rust" ABI on 32-bit x86 doesn't use the x87
// floating point stack, as loading and storing `f32`s and `f64`s to and from the x87 stack quietens
// signalling NaNs.

// Returning individual floats

// CHECK-LABEL: return_f32:
#[no_mangle]
pub fn return_f32(x: f32) -> f32 {
    // sse: movss {{.*}}(%ebp), %xmm0
    // nosse: movl {{.*}}(%ebp), %eax
    // CHECK-NOT: ax
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f64:
#[no_mangle]
pub fn return_f64(x: f64) -> f64 {
    // nosse: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // nosse-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL:.*]]
    // nosse-NEXT: movsd %[[VAL]], (%[[PTR]])
    // sse: movsd {{.*}}(%ebp), %xmm0
    // sse-NOT: ax
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
    // sse-NEXT: movss %xmm0, (%[[PTR]])
    // nosse-NEXT: movl %eax, (%[[PTR]])
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
    // sse: movlps %xmm0, (%[[PTR]])
    // nosse: movsd {{.*}}(%{{ebp|esp}}), %[[VAL:.*]]
    // nosse-NEXT: movsd %[[VAL:.*]], (%[[PTR]])
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
    // CHECK: movsd [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
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
    // CHECK: movss [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movss %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movsd %[[VAL2]], 4(%[[PTR]])
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
    // CHECK: movsd [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movss [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
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
    // CHECK: movsd [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movl [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
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
    // CHECK: movl [[#%d,OFFSET:]](%ebp), %[[VAL1:.*]]
    // CHECK-NEXT: movsd [[#%d,OFFSET+4]](%ebp), %[[VAL2:.*]]
    // CHECK-NEXT: movl %[[VAL1]], (%[[PTR]])
    // CHECK-NEXT: movsd %[[VAL2]], 4(%[[PTR]])
    *x = get_other_f64();
}

// The "C" ABI for `f16` and `f128` on x86 has never used the x87 floating point stack. Do some
// basic checks to ensure this remains the case for the "Rust" ABI.

// CHECK-LABEL: return_f16:
#[no_mangle]
pub fn return_f16(x: f16) -> f16 {
    // CHECK: pushl %ebp
    // CHECK: movl %esp, %ebp
    // nosse: movzwl 8(%ebp), %eax
    // sse: pinsrw $0, 8(%ebp), %xmm0
    // CHECK: popl %ebp
    // CHECK: retl
    x
}

// CHECK-LABEL: return_f128:
#[no_mangle]
pub fn return_f128(x: f128) -> f128 {
    // CHECK: pushl %ebp
    // sse: movaps [[#%d,OFFSET:]](%ebp), %xmm0
    // nosse: movl [[#%d,OFFSET:]](%ebp), %[[PTR:.*]]
    // nosse-NEXT: movl [[#%d,OFFSET+4]](%ebp), %[[VAL1:.*]]
    // nosse-NEXT: movl [[#%d,OFFSET+8]](%ebp), %[[VAL2:.*]]
    // nosse-NEXT: movl [[#%d,OFFSET+12]](%ebp), %[[VAL3:.*]]
    // nosse-NEXT: movl [[#%d,OFFSET+16]](%ebp), %[[VAL4:.*]]
    // nosse-NEXT: movl %[[VAL4:.*]] 12(%[[PTR]])
    // nosse-NEXT: movl %[[VAL3:.*]] 8(%[[PTR]])
    // nosse-NEXT: movl %[[VAL2:.*]] 4(%[[PTR]])
    // nosse-NEXT: movl %[[VAL1:.*]] (%[[PTR]])
    // CHECK: popl %ebp
    // CHECK: retl
    x
}
