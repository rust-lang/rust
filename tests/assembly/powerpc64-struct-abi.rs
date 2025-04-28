//@ add-core-stubs
//@ revisions: elfv1-be elfv2-be elfv2-le aix
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@[elfv1-be] compile-flags: --target powerpc64-unknown-linux-gnu
//@[elfv1-be] needs-llvm-components: powerpc
//@[elfv2-be] compile-flags: --target powerpc64-unknown-linux-musl
//@[elfv2-be] needs-llvm-components: powerpc
//@[elfv2-le] compile-flags: --target powerpc64le-unknown-linux-gnu -C target-cpu=pwr8
//@[elfv2-le] needs-llvm-components: powerpc
//@[aix] compile-flags: --target powerpc64-ibm-aix
//@[aix] needs-llvm-components: powerpc
//@[elfv1-be] filecheck-flags: --check-prefix be
//@[elfv2-be] filecheck-flags: --check-prefix be
//@[elfv1-be] filecheck-flags: --check-prefix elf
//@[elfv2-be] filecheck-flags: --check-prefix elf
//@[elfv2-le] filecheck-flags: --check-prefix elf

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

impl Copy for FiveU32s {}
impl Copy for FiveU16s {}
impl Copy for ThreeU8s {}

#[repr(C)]
struct FiveU32s(u32, u32, u32, u32, u32);

#[repr(C)]
struct FiveU16s(u16, u16, u16, u16, u16);

#[repr(C)]
struct ThreeU8s(u8, u8, u8);

// CHECK-LABEL: read_large
// aix: lwz [[REG1:.*]], 16(4)
// aix-NEXT: lxv{{d2x|w4x}} 0, 0, 4
// aix-NEXT: stw [[REG1]], 16(3)
// aix-NEXT: stxv{{d2x|w4x}} 0, 0, 3
// be: lwz [[REG1:.*]], 16(4)
// be-NEXT: stw [[REG1]], 16(3)
// be-NEXT: ld [[REG2:.*]], 8(4)
// be-NEXT: ld [[REG3:.*]], 0(4)
// be-NEXT: std [[REG2]], 8(3)
// be-NEXT: std [[REG3]], 0(3)
// elfv2-le: lxvd2x [[REG1:.*]], 0, 4
// elfv2-le-NEXT: lwz [[REG2:.*]], 16(4)
// elfv2-le-NEXT: stw [[REG2]], 16(3)
// elfv2-le-NEXT: stxvd2x [[REG1]], 0, 3
// CHECK-NEXT: blr
#[no_mangle]
extern "C" fn read_large(x: &FiveU32s) -> FiveU32s {
    *x
}

// CHECK-LABEL: read_medium
// aix: lhz [[REG1:.*]], 8(4)
// aix-NEXT: ld [[REG2:.*]], 0(4)
// aix-NEXT: sth [[REG1]], 8(3)
// aix-NEXT: std [[REG2]], 0(3)
// elfv1-be: lhz [[REG1:.*]], 8(4)
// elfv1-be-NEXT: ld [[REG2:.*]], 0(4)
// elfv1-be-NEXT: sth [[REG1]], 8(3)
// elfv1-be-NEXT: std [[REG2]], 0(3)
// elfv2-be: lhz [[REG1:.*]], 8(3)
// elfv2-be-NEXT: ld 3, 0(3)
// elfv2-be-NEXT: sldi 4, [[REG1]], 48
// elfv2-le: ld [[REG1:.*]], 0(3)
// elfv2-le-NEXT: lhz 4, 8(3)
// elfv2-le-NEXT: mr 3, [[REG1]]
// CHECK-NEXT: blr
#[no_mangle]
extern "C" fn read_medium(x: &FiveU16s) -> FiveU16s {
    *x
}

// CHECK-LABEL: read_small
// aix: lbz [[REG1:.*]], 2(4)
// aix-NEXT: lhz [[REG2:.*]], 0(4)
// aix-NEXT: stb [[REG1]], 2(3)
// aix-NEXT: sth [[REG2]], 0(3)
// elfv1-be: lbz [[REG1:.*]], 2(4)
// elfv1-be-NEXT: lhz [[REG2:.*]], 0(4)
// elfv1-be-NEXT: stb [[REG1]], 2(3)
// elfv1-be-NEXT: sth [[REG2]], 0(3)
// elfv2-be: lhz [[REG1:.*]], 0(3)
// elfv2-be-NEXT: lbz 3, 2(3)
// elfv2-be-NEXT: rldimi 3, [[REG1]], 8, 0
// elfv2-le: lbz [[REG1:.*]], 2(3)
// elfv2-le-NEXT: lhz 3, 0(3)
// elfv2-le-NEXT: rldimi 3, [[REG1]], 16, 0
// CHECK-NEXT: blr
#[no_mangle]
extern "C" fn read_small(x: &ThreeU8s) -> ThreeU8s {
    *x
}

// CHECK-LABEL: write_large
// aix: std 3, 48(1)
// aix-NEXT: rldicl [[REG1:.*]], 5, 32, 32
// aix-NEXT: std 5, 64(1)
// aix-NEXT: std 4, 56(1)
// aix-NEXT: stw [[REG1]], 16(6)
// aix-NEXT: addi [[REG2:.*]], 1, 48
// aix-NEXT: lxv{{d2x|w4x}} 0, 0, [[REG2]]
// aix-NEXT: stxv{{d2x|w4x}} 0, 0, 6
// elf: std 3, 0(6)
// be-NEXT: rldicl [[REG1:.*]], 5, 32, 32
// elf-NEXT: std 4, 8(6)
// be-NEXT: stw [[REG1]], 16(6)
// elfv2-le-NEXT: stw 5, 16(6)
// CHECK-NEXT: blr
#[no_mangle]
extern "C" fn write_large(x: FiveU32s, dest: &mut FiveU32s) {
    *dest = x;
}

// CHECK-LABEL: write_medium
// aix: std 4, 56(1)
// aix-NEXT: rldicl [[REG1:.*]], 4, 16, 48
// aix-NEXT: std 3, 48(1)
// aix-NEXT: std 3, 0(5)
// aix-NEXT: sth [[REG1]], 8(5)
// elf: std 3, 0(5)
// be-NEXT: rldicl [[REG1:.*]], 4, 16, 48
// be-NEXT: sth [[REG1]], 8(5)
// elfv2-le-NEXT: sth 4, 8(5)
// CHECK-NEXT: blr
#[no_mangle]
extern "C" fn write_medium(x: FiveU16s, dest: &mut FiveU16s) {
    *dest = x;
}

// CHECK-LABEL: write_small
// aix: std 3, 48(1)
// aix-NEXT: rldicl [[REG1:.*]], 3, 16, 48
// aix-NEXT: sth 3, 0(4)
// aix-NEXT: lbz 3, 50(1)
// aix-NEXT: stb [[REG1]], 2(4)
// be: stb 3, 2(4)
// be-NEXT: srwi [[REG1:.*]], 3, 8
// be-NEXT: sth [[REG1]], 0(4)
// The order these instructions are emitted in changed in LLVM 18.
// elfv2-le-DAG: sth 3, 0(4)
// elfv2-le-DAG: srwi [[REG1:.*]], 3, 16
// elfv2-le-NEXT: stb [[REG1]], 2(4)
// CHECK-NEXT: blr
#[no_mangle]
extern "C" fn write_small(x: ThreeU8s, dest: &mut ThreeU8s) {
    *dest = x;
}
