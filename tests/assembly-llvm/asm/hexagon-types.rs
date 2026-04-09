//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target hexagon-unknown-linux-musl
//@ compile-flags: -C target-feature=+hvx-length128b
//@ compile-flags: -Zmerge-functions=disabled
//@ needs-llvm-components: hexagon

#![feature(no_core, repr_simd, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *const i32;

#[repr(simd)]
pub struct i32x32([i32; 32]); // 1024-bit HVX vector (128B mode)
impl Copy for i32x32 {}

#[repr(simd)]
pub struct i32x64([i32; 64]); // 2048-bit HVX vector pair (128B mode)
impl Copy for i32x64 {}

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!("{} = {}", out($class) y, in($class) x);
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($reg, " = ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: sym_static:
// CHECK: InlineAsm Start
// CHECK: r0 = {{#+}}extern_static
// CHECK: InlineAsm End
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("r0 = #{}", sym extern_static);
}

// CHECK-LABEL: sym_fn:
// CHECK: InlineAsm Start
// CHECK: r0 = {{#+}}extern_func
// CHECK: InlineAsm End
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("r0 = #{}", sym extern_func);
}

// This is a test for multi-instruction packets,
// which require the escaped braces.
//
// CHECK-LABEL: packet:
// CHECK: InlineAsm Start
// CHECK: {
// CHECK:   r{{[0-9]+}} = r0
// CHECK:   memw(r1{{(\+#0)?}}) = r{{[0-9]+}}
// CHECK: }
// CHECK: InlineAsm End
#[no_mangle]
pub unsafe fn packet() {
    let val = 1024;
    asm!("{{
        {} = r0
        memw(r1) = {}
    }}", out(reg) _, in(reg) &val);
}

// CHECK-LABEL: reg_ptr:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_ptr ptr reg);

// CHECK-LABEL: reg_f32:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_f32 f32 reg);

// CHECK-LABEL: reg_i32:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_i32 i32 reg);

// CHECK-LABEL: reg_i8:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_i8 i8 reg);

// CHECK-LABEL: reg_i16:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_i16 i16 reg);

// CHECK-LABEL: r0_ptr:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_ptr ptr "r0");

// CHECK-LABEL: r0_f32:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_f32 f32 "r0");

// CHECK-LABEL: r0_i32:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_i32 i32 "r0");

// CHECK-LABEL: r0_i8:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_i8 i8 "r0");

// CHECK-LABEL: r0_i16:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_i16 i16 "r0");

// ===== Register pair (reg_pair) tests =====

// CHECK-LABEL: reg_pair_i64:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}}:{{[0-9]+}} = combine(r{{[0-9]+}},r{{[0-9]+}})
// CHECK: InlineAsm End
check!(reg_pair_i64 i64 reg_pair);

// CHECK-LABEL: reg_pair_f64:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}}:{{[0-9]+}} = combine(r{{[0-9]+}},r{{[0-9]+}})
// CHECK: InlineAsm End
check!(reg_pair_f64 f64 reg_pair);

// CHECK-LABEL: r1_0_i64:
// CHECK: InlineAsm Start
// CHECK: r1:0 = combine(r1,r0)
// CHECK: InlineAsm End
check_reg!(r1_0_i64 i64 "r1:0");

// CHECK-LABEL: r1_0_f64:
// CHECK: InlineAsm Start
// CHECK: r1:0 = combine(r1,r0)
// CHECK: InlineAsm End
check_reg!(r1_0_f64 f64 "r1:0");

// ===== HVX vector register (vreg) tests =====

// CHECK-LABEL: vreg_i32x32:
// CHECK: InlineAsm Start
// CHECK: v{{[0-9]+}} = v{{[0-9]+}}
// CHECK: InlineAsm End
check!(vreg_i32x32 i32x32 vreg);

// CHECK-LABEL: v0_i32x32:
// CHECK: InlineAsm Start
// CHECK: v0 = v0
// CHECK: InlineAsm End
check_reg!(v0_i32x32 i32x32 "v0");

// ===== HVX vector pair (vreg_pair) tests =====

// CHECK-LABEL: vreg_pair_i32x64:
// CHECK: InlineAsm Start
// CHECK: v{{[0-9]+}}:{{[0-9]+}} = vcombine(r{{[0-9]+}},r{{[0-9]+}})
// CHECK: InlineAsm End
check!(vreg_pair_i32x64 i32x64 vreg_pair);

// CHECK-LABEL: v1_0_i32x64:
// CHECK: InlineAsm Start
// CHECK: v1:0 = vcombine(r1,r0)
// CHECK: InlineAsm End
check_reg!(v1_0_i32x64 i32x64 "v1:0");
