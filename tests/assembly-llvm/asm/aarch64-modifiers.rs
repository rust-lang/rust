//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 -C panic=abort
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ compile-flags: -Zmerge-functions=disabled
//@ needs-llvm-components: aarch64

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register)]

extern crate minicore;
use minicore::*;

macro_rules! check {
    ($func:ident $reg:ident $code:literal) => {
        // -Copt-level=3 and extern "C" guarantee that the selected register is always r0/s0/d0/q0
        #[no_mangle]
        pub unsafe extern "C" fn $func() -> i32 {
            let y;
            asm!($code, out($reg) y);
            y
        }
    };
}

// CHECK-LABEL: reg:
// CHECK: //APP
// CHECK: mov x0, x0
// CHECK: //NO_APP
check!(reg reg "mov {0}, {0}");

// CHECK-LABEL: reg_w:
// CHECK: //APP
// CHECK: mov w0, w0
// CHECK: //NO_APP
check!(reg_w reg "mov {0:w}, {0:w}");

// CHECK-LABEL: reg_x:
// CHECK: //APP
// CHECK: mov x0, x0
// CHECK: //NO_APP
check!(reg_x reg "mov {0:x}, {0:x}");

// CHECK-LABEL: vreg:
// CHECK: //APP
// CHECK: add v0.4s, v0.4s, v0.4s
// CHECK: //NO_APP
check!(vreg vreg "add {0}.4s, {0}.4s, {0}.4s");

// CHECK-LABEL: vreg_b:
// CHECK: //APP
// CHECK: ldr b0, [x0]
// CHECK: //NO_APP
check!(vreg_b vreg "ldr {:b}, [x0]");

// CHECK-LABEL: vreg_h:
// CHECK: //APP
// CHECK: ldr h0, [x0]
// CHECK: //NO_APP
check!(vreg_h vreg "ldr {:h}, [x0]");

// CHECK-LABEL: vreg_s:
// CHECK: //APP
// CHECK: ldr s0, [x0]
// CHECK: //NO_APP
check!(vreg_s vreg "ldr {:s}, [x0]");

// CHECK-LABEL: vreg_d:
// CHECK: //APP
// CHECK: ldr d0, [x0]
// CHECK: //NO_APP
check!(vreg_d vreg "ldr {:d}, [x0]");

// CHECK-LABEL: vreg_q:
// CHECK: //APP
// CHECK: ldr q0, [x0]
// CHECK: //NO_APP
check!(vreg_q vreg "ldr {:q}, [x0]");

// CHECK-LABEL: vreg_v:
// CHECK: //APP
// CHECK: add v0.4s, v0.4s, v0.4s
// CHECK: //NO_APP
check!(vreg_v vreg "add {0:v}.4s, {0:v}.4s, {0:v}.4s");

// CHECK-LABEL: vreg_low16:
// CHECK: //APP
// CHECK: add v0.4s, v0.4s, v0.4s
// CHECK: //NO_APP
check!(vreg_low16 vreg_low16 "add {0}.4s, {0}.4s, {0}.4s");

// CHECK-LABEL: vreg_low16_b:
// CHECK: //APP
// CHECK: ldr b0, [x0]
// CHECK: //NO_APP
check!(vreg_low16_b vreg_low16 "ldr {:b}, [x0]");

// CHECK-LABEL: vreg_low16_h:
// CHECK: //APP
// CHECK: ldr h0, [x0]
// CHECK: //NO_APP
check!(vreg_low16_h vreg_low16 "ldr {:h}, [x0]");

// CHECK-LABEL: vreg_low16_s:
// CHECK: //APP
// CHECK: ldr s0, [x0]
// CHECK: //NO_APP
check!(vreg_low16_s vreg_low16 "ldr {:s}, [x0]");

// CHECK-LABEL: vreg_low16_d:
// CHECK: //APP
// CHECK: ldr d0, [x0]
// CHECK: //NO_APP
check!(vreg_low16_d vreg_low16 "ldr {:d}, [x0]");

// CHECK-LABEL: vreg_low16_q:
// CHECK: //APP
// CHECK: ldr q0, [x0]
// CHECK: //NO_APP
check!(vreg_low16_q vreg_low16 "ldr {:q}, [x0]");

// CHECK-LABEL: vreg_low16_v:
// CHECK: //APP
// CHECK: add v0.4s, v0.4s, v0.4s
// CHECK: //NO_APP
check!(vreg_low16_v vreg_low16 "add {0:v}.4s, {0:v}.4s, {0:v}.4s");
