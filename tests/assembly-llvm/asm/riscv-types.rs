//@ add-core-stubs
//@ revisions: riscv64 riscv32 riscv64-zfhmin riscv32-zfhmin riscv64-zfh riscv32-zfh
//@ assembly-output: emit-asm

//@[riscv64] compile-flags: --target riscv64imac-unknown-none-elf
//@[riscv64] needs-llvm-components: riscv

//@[riscv32] compile-flags: --target riscv32imac-unknown-none-elf
//@[riscv32] needs-llvm-components: riscv

//@[riscv64-zfhmin] compile-flags: --target riscv64imac-unknown-none-elf --cfg riscv64
//@[riscv64-zfhmin] needs-llvm-components: riscv
//@[riscv64-zfhmin] compile-flags: -C target-feature=+zfhmin
//@[riscv64-zfhmin] filecheck-flags: --check-prefix riscv64

//@[riscv32-zfhmin] compile-flags: --target riscv32imac-unknown-none-elf
//@[riscv32-zfhmin] needs-llvm-components: riscv
//@[riscv32-zfhmin] compile-flags: -C target-feature=+zfhmin

//@[riscv64-zfh] compile-flags: --target riscv64imac-unknown-none-elf --cfg riscv64
//@[riscv64-zfh] needs-llvm-components: riscv
//@[riscv64-zfh] compile-flags: -C target-feature=+zfh
//@[riscv64-zfh] filecheck-flags: --check-prefix riscv64 --check-prefix zfhmin

//@[riscv32-zfh] compile-flags: --target riscv32imac-unknown-none-elf
//@[riscv32-zfh] needs-llvm-components: riscv
//@[riscv32-zfh] compile-flags: -C target-feature=+zfh
//@[riscv32-zfh] filecheck-flags: --check-prefix zfhmin

//@ compile-flags: -C target-feature=+d
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, f16)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register)]

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// CHECK-LABEL: sym_fn:
// CHECK: #APP
// CHECK: call extern_func
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: #APP
// CHECK: auipc t0, %pcrel_hi(extern_static)
// CHECK: lb t0, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})(t0)
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("lb t0, {}", sym extern_static);
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " {}, {}"), out($class) y, in($class) x);
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i8 i8 reg "mv");

// CHECK-LABEL: reg_f16:
// CHECK: #APP
// CHECK: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_f16 f16 reg "mv");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i16 i16 reg "mv");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i32 i32 reg "mv");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_f32 f32 reg "mv");

// riscv64-LABEL: reg_i64:
// riscv64: #APP
// riscv64: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// riscv64: #NO_APP
#[cfg(riscv64)]
check!(reg_i64 i64 reg "mv");

// riscv64-LABEL: reg_f64:
// riscv64: #APP
// riscv64: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// riscv64: #NO_APP
#[cfg(riscv64)]
check!(reg_f64 f64 reg "mv");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: mv {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr ptr reg "mv");

// CHECK-LABEL: freg_f16:
// zfhmin-NOT: or
// CHECK: #APP
// CHECK: fmv.s f{{[a-z0-9]+}}, f{{[a-z0-9]+}}
// CHECK: #NO_APP
// zfhmin-NOT: or
check!(freg_f16 f16 freg "fmv.s");

// CHECK-LABEL: freg_f32:
// CHECK: #APP
// CHECK: fmv.s f{{[a-z0-9]+}}, f{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f32 f32 freg "fmv.s");

// CHECK-LABEL: freg_f64:
// CHECK: #APP
// CHECK: fmv.d f{{[a-z0-9]+}}, f{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f64 f64 freg "fmv.d");

// CHECK-LABEL: a0_i8:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check_reg!(a0_i8 i8 "a0" "mv");

// CHECK-LABEL: a0_i16:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check_reg!(a0_i16 i16 "a0" "mv");

// CHECK-LABEL: a0_f16:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check_reg!(a0_f16 f16 "a0" "mv");

// CHECK-LABEL: a0_i32:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check_reg!(a0_i32 i32 "a0" "mv");

// CHECK-LABEL: a0_f32:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check_reg!(a0_f32 f32 "a0" "mv");

// riscv64-LABEL: a0_i64:
// riscv64: #APP
// riscv64: mv a0, a0
// riscv64: #NO_APP
#[cfg(riscv64)]
check_reg!(a0_i64 i64 "a0" "mv");

// riscv64-LABEL: a0_f64:
// riscv64: #APP
// riscv64: mv a0, a0
// riscv64: #NO_APP
#[cfg(riscv64)]
check_reg!(a0_f64 f64 "a0" "mv");

// CHECK-LABEL: a0_ptr:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check_reg!(a0_ptr ptr "a0" "mv");

// CHECK-LABEL: fa0_f16:
// zfhmin-NOT: or
// CHECK: #APP
// CHECK: fmv.s fa0, fa0
// CHECK: #NO_APP
// zfhmin-NOT: or
check_reg!(fa0_f16 f16 "fa0" "fmv.s");

// CHECK-LABEL: fa0_f32:
// CHECK: #APP
// CHECK: fmv.s fa0, fa0
// CHECK: #NO_APP
check_reg!(fa0_f32 f32 "fa0" "fmv.s");

// CHECK-LABEL: fa0_f64:
// CHECK: #APP
// CHECK: fmv.d fa0, fa0
// CHECK: #NO_APP
check_reg!(fa0_f64 f64 "fa0" "fmv.d");
