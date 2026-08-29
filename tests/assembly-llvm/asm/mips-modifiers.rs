//@ add-minicore
//@ revisions: mips32 mips32el mips32r6 mips32r6el mips64 mips64el mips64r6 mips64r6el
//@ assembly-output: emit-asm
//@[mips32] compile-flags: --target mips-unknown-linux-gnu -Ctarget-feature=+mips32r5
//@[mips32] needs-llvm-components: mips
//@[mips32el] compile-flags: --target mipsel-unknown-linux-gnu -Ctarget-feature=+mips32r5
//@[mips32el] needs-llvm-components: mips
//@[mips32r6] compile-flags: --target mipsisa32r6-unknown-linux-gnu
//@[mips32r6] needs-llvm-components: mips
//@[mips32r6el] compile-flags: --target mipsisa32r6el-unknown-linux-gnu
//@[mips32r6el] needs-llvm-components: mips
//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64 -Ctarget-feature=+mips64r5
//@[mips64] needs-llvm-components: mips
//@[mips64el] compile-flags: --target mips64el-unknown-linux-gnuabi64 -Ctarget-feature=+mips64r5
//@[mips64el] needs-llvm-components: mips
//@[mips64r6] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64
//@[mips64r6] needs-llvm-components: mips
//@[mips64r6el] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64
//@[mips64r6el] needs-llvm-components: mips
//@ compile-flags: -Copt-level=3 -C panic=abort
//@ compile-flags: -Zmerge-functions=disabled
//@ compile-flags: -Ctarget-feature=+fp64,+msa

#![feature(no_core, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register)]

extern crate minicore;
use minicore::*;

macro_rules! check {
    ($func:ident $modifier:literal $reg:ident $mov:literal) => {
        // -Copt-level=3 and extern "C" guarantee that the selected register is always w0
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $func() {
            asm!(concat!($mov, " {0:", $modifier, "}, {0:", $modifier, "}"), out($reg) _);
        }
    };
}

// CHECK-LABEL: freg_w:
// CHECK: #APP
// CHECK: move.v $w0, $w0
// CHECK: #NO_APP
check!(freg_w "w" freg "move.v");
