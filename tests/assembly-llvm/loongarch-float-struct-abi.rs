//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target loongarch64-unknown-linux-gnu
//@ needs-llvm-components: loongarch

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C, align(64))]
struct Aligned(f64);

#[repr(C)]
struct Padded(u8, Aligned);

#[repr(C, packed)]
struct Packed(u8, f32);

impl Copy for Aligned {}
impl Copy for Padded {}
impl Copy for Packed {}

extern "C" {
    fn take_padded(x: Padded);
    fn get_padded() -> Padded;
    fn take_packed(x: Packed);
    fn get_packed() -> Packed;
}

// CHECK-LABEL: pass_padded
#[unsafe(no_mangle)]
extern "C" fn pass_padded(out: &mut Padded, x: Padded) {
    // CHECK: st.b $a1, $a0, 0
    // CHECK-NEXT: fst.d $fa0, $a0, 64
    // CHECK-NEXT: ret
    *out = x;
}

// CHECK-LABEL: ret_padded
#[unsafe(no_mangle)]
extern "C" fn ret_padded(x: &Padded) -> Padded {
    // CHECK: fld.d $fa0, $a0, 64
    // CHECK-NEXT: ld.b $a0, $a0, 0
    // CHECK-NEXT: ret
    *x
}

#[unsafe(no_mangle)]
extern "C" fn call_padded(x: &Padded) {
    // CHECK: fld.d $fa0, $a0, 64
    // CHECK-NEXT: ld.b $a0, $a0, 0
    // CHECK-NEXT: pcaddu18i $t8, %call36(take_padded)
    // CHECK-NEXT: jr $t8
    unsafe {
        take_padded(*x);
    }
}

#[unsafe(no_mangle)]
extern "C" fn receive_padded(out: &mut Padded) {
    // CHECK: addi.d $sp, $sp, -16
    // CHECK-NEXT: .cfi_def_cfa_offset 16
    // CHECK-NEXT: st.d $ra, $sp, [[#%d,RA_SPILL:]]
    // CHECK-NEXT: st.d [[TEMP:.*]], $sp, [[#%d,TEMP_SPILL:]]
    // CHECK-NEXT: .cfi_offset 1, [[#%d,RA_SPILL - 16]]
    // CHECK-NEXT: .cfi_offset [[#%d,TEMP_NUM:]], [[#%d,TEMP_SPILL - 16]]
    // CHECK-NEXT: move [[TEMP]], $a0
    // CHECK-NEXT: pcaddu18i $ra, %call36(get_padded)
    // CHECK-NEXT: jirl $ra, $ra, 0
    // CHECK-NEXT: st.b $a0, [[TEMP]], 0
    // CHECK-NEXT: fst.d $fa0, [[TEMP]], 64
    // CHECK-NEXT: ld.d [[TEMP]], $sp, [[#%d,TEMP_SPILL]]
    // CHECK-NEXT: ld.d $ra, $sp, [[#%d,RA_SPILL]]
    // CHECK: addi.d $sp, $sp, 16
    // CHECK: ret
    unsafe {
        *out = get_padded();
    }
}

// CHECK-LABEL: pass_packed
#[unsafe(no_mangle)]
extern "C" fn pass_packed(out: &mut Packed, x: Packed) {
    // CHECK: st.b $a1, $a0, 0
    // CHECK-NEXT: fst.s $fa0, $a0, 1
    // CHECK-NEXT: ret
    *out = x;
}

// CHECK-LABEL: ret_packed
#[unsafe(no_mangle)]
extern "C" fn ret_packed(x: &Packed) -> Packed {
    // CHECK: fld.s $fa0, $a0, 1
    // CHECK-NEXT: ld.b $a0, $a0, 0
    // CHECK-NEXT: ret
    *x
}

#[unsafe(no_mangle)]
extern "C" fn call_packed(x: &Packed) {
    // CHECK: fld.s $fa0, $a0, 1
    // CHECK-NEXT: ld.b $a0, $a0, 0
    // CHECK-NEXT: pcaddu18i $t8, %call36(take_packed)
    // CHECK-NEXT: jr $t8
    unsafe {
        take_packed(*x);
    }
}

#[unsafe(no_mangle)]
extern "C" fn receive_packed(out: &mut Packed) {
    // CHECK: addi.d $sp, $sp, -16
    // CHECK-NEXT: .cfi_def_cfa_offset 16
    // CHECK-NEXT: st.d $ra, $sp, [[#%d,RA_SPILL:]]
    // CHECK-NEXT: st.d [[TEMP:.*]], $sp, [[#%d,TEMP_SPILL:]]
    // CHECK-NEXT: .cfi_offset 1, [[#%d,RA_SPILL - 16]]
    // CHECK-NEXT: .cfi_offset [[#%d,TEMP_NUM:]], [[#%d,TEMP_SPILL - 16]]
    // CHECK-NEXT: move [[TEMP]], $a0
    // CHECK-NEXT: pcaddu18i $ra, %call36(get_packed)
    // CHECK-NEXT: jirl $ra, $ra, 0
    // CHECK-NEXT: st.b $a0, [[TEMP]], 0
    // CHECK-NEXT: fst.s $fa0, [[TEMP]], 1
    // CHECK-NEXT: ld.d [[TEMP]], $sp, [[#%d,TEMP_SPILL]]
    // CHECK-NEXT: ld.d $ra, $sp, [[#%d,RA_SPILL]]
    // CHECK: addi.d $sp, $sp, 16
    // CHECK: ret
    unsafe {
        *out = get_packed();
    }
}
