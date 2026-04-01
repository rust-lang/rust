// Test for -Z large_data_threshold=...
// This test verifies that with the medium code model, data above the threshold
// is placed in large data sections (.ldata, .lbss, .lrodata).
//@ assembly-output: emit-asm
//@ compile-flags: -Ccode-model=medium -Zlarge-data-threshold=4
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "drop_in_place"]
fn drop_in_place<T>(_: *mut T) {}

#[used]
#[no_mangle]
// U is below the threshold, should be in .data
static mut U: u16 = 123;

#[used]
#[no_mangle]
// V is below the threshold, should be in .bss
static mut V: u16 = 0;

#[used]
#[no_mangle]
// W is at the threshold, should be in .data
static mut W: u32 = 123;

#[used]
#[no_mangle]
// X is at the threshold, should be in .bss
static mut X: u32 = 0;

#[used]
#[no_mangle]
// Y is over the threshold, should be in .ldata
static mut Y: u64 = 123;

#[used]
#[no_mangle]
// Z is over the threshold, should be in .lbss
static mut Z: u64 = 0;

// CHECK: .section .data.U,
// CHECK-NOT: .section
// CHECK: U:
// CHECK: .section .bss.V,
// CHECK-NOT: .section
// CHECK: V:
// CHECK: .section .data.W,
// CHECK-NOT: .section
// CHECK: W:
// CHECK: .section .bss.X,
// CHECK-NOT: .section
// CHECK: X:
// CHECK: .section .ldata.Y,
// CHECK-NOT: .section
// CHECK: Y:
// CHECK: .section .lbss.Z,
// CHECK-NOT: .section
// CHECK: Z:
