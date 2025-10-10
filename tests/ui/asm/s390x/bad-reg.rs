//@ add-core-stubs
//@ needs-asm-support
//@ revisions: s390x s390x_vector s390x_vector_stable
//@[s390x] compile-flags: --target s390x-unknown-linux-gnu -C target-feature=-vector
//@[s390x] needs-llvm-components: systemz
//@[s390x_vector] compile-flags: --target s390x-unknown-linux-gnu -C target-feature=+vector
//@[s390x_vector] needs-llvm-components: systemz
//@[s390x_vector_stable] compile-flags: --target s390x-unknown-linux-gnu -C target-feature=+vector
//@[s390x_vector_stable] needs-llvm-components: systemz

#![crate_type = "rlib"]
#![feature(no_core, repr_simd)]
#![cfg_attr(not(s390x_vector_stable), feature(asm_experimental_reg))]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i64x2([i64; 2]);

impl Copy for i64x2 {}

fn f() {
    let mut x = 0;
    let mut b = 0u8;
    let mut v = i64x2([0; 2]);
    unsafe {
        // Unsupported registers
        asm!("", out("r11") _);
        //~^ ERROR invalid register `r11`: The frame pointer cannot be used as an operand for inline asm
        asm!("", out("r15") _);
        //~^ ERROR invalid register `r15`: The stack pointer cannot be used as an operand for inline asm
        asm!("", out("c0") _);
        //~^ ERROR invalid register `c0`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c1") _);
        //~^ ERROR invalid register `c1`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c2") _);
        //~^ ERROR invalid register `c2`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c3") _);
        //~^ ERROR invalid register `c3`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c4") _);
        //~^ ERROR invalid register `c4`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c5") _);
        //~^ ERROR invalid register `c5`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c6") _);
        //~^ ERROR invalid register `c6`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c7") _);
        //~^ ERROR invalid register `c7`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c8") _);
        //~^ ERROR invalid register `c8`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c9") _);
        //~^ ERROR invalid register `c9`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c10") _);
        //~^ ERROR invalid register `c10`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c11") _);
        //~^ ERROR invalid register `c11`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c12") _);
        //~^ ERROR invalid register `c12`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c13") _);
        //~^ ERROR invalid register `c13`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c14") _);
        //~^ ERROR invalid register `c14`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("c15") _);
        //~^ ERROR invalid register `c15`: control registers are reserved by the kernel and cannot be used as operands for inline asm
        asm!("", out("a0") _);
        //~^ ERROR invalid register `a0`: a0 and a1 are reserved for system use and cannot be used as operands for inline asm
        asm!("", out("a1") _);
        //~^ ERROR invalid register `a1`: a0 and a1 are reserved for system use and cannot be used as operands for inline asm

        // vreg
        asm!("", out("v0") _); // always ok
        asm!("", in("v0") v); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `i64x2` cannot be used with this register class in stable [E0658]
        asm!("", out("v0") v); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `i64x2` cannot be used with this register class in stable [E0658]
        asm!("", in("v0") x); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `i32` cannot be used with this register class in stable [E0658]
        asm!("", out("v0") x); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `i32` cannot be used with this register class in stable [E0658]
        asm!("", in("v0") b);
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector]~^^ ERROR type `u8` cannot be used with this register class
        //[s390x_vector_stable]~^^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `u8` cannot be used with this register class
        asm!("", out("v0") b);
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector]~^^ ERROR type `u8` cannot be used with this register class
        //[s390x_vector_stable]~^^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `u8` cannot be used with this register class
        asm!("/* {} */", in(vreg) v); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `i64x2` cannot be used with this register class in stable [E0658]
        asm!("/* {} */", in(vreg) x); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `i32` cannot be used with this register class in stable [E0658]
        asm!("/* {} */", in(vreg) b);
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector]~^^ ERROR type `u8` cannot be used with this register class
        //[s390x_vector_stable]~^^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]
        //[s390x_vector_stable]~| ERROR type `u8` cannot be used with this register class
        asm!("/* {} */", out(vreg) _); // requires vector & asm_experimental_reg
        //[s390x]~^ ERROR register class `vreg` requires the `vector` target feature
        //[s390x_vector_stable]~^^ ERROR register class `vreg` can only be used as a clobber in stable [E0658]

        // Clobber-only registers
        // areg
        asm!("", out("a2") _); // ok
        asm!("", in("a2") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("a2") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(areg) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(areg) _);
        //~^ ERROR can only be used as a clobber

        // Overlapping registers
        // vreg/freg
        asm!("", out("v0") _, out("f0") _);
        //~^ ERROR register `f0` conflicts with register `v0`
        asm!("", out("v1") _, out("f1") _);
        //~^ ERROR register `f1` conflicts with register `v1`
        asm!("", out("v2") _, out("f2") _);
        //~^ ERROR register `f2` conflicts with register `v2`
        asm!("", out("v3") _, out("f3") _);
        //~^ ERROR register `f3` conflicts with register `v3`
        asm!("", out("v4") _, out("f4") _);
        //~^ ERROR register `f4` conflicts with register `v4`
        asm!("", out("v5") _, out("f5") _);
        //~^ ERROR register `f5` conflicts with register `v5`
        asm!("", out("v6") _, out("f6") _);
        //~^ ERROR register `f6` conflicts with register `v6`
        asm!("", out("v7") _, out("f7") _);
        //~^ ERROR register `f7` conflicts with register `v7`
        asm!("", out("v8") _, out("f8") _);
        //~^ ERROR register `f8` conflicts with register `v8`
        asm!("", out("v9") _, out("f9") _);
        //~^ ERROR register `f9` conflicts with register `v9`
        asm!("", out("v10") _, out("f10") _);
        //~^ ERROR register `f10` conflicts with register `v10`
        asm!("", out("v11") _, out("f11") _);
        //~^ ERROR register `f11` conflicts with register `v11`
        asm!("", out("v12") _, out("f12") _);
        //~^ ERROR register `f12` conflicts with register `v12`
        asm!("", out("v13") _, out("f13") _);
        //~^ ERROR register `f13` conflicts with register `v13`
        asm!("", out("v14") _, out("f14") _);
        //~^ ERROR register `f14` conflicts with register `v14`
        asm!("", out("v15") _, out("f15") _);
        //~^ ERROR register `f15` conflicts with register `v15`
        // no %f16
        asm!("", out("v16") _, out("f16") _);
        //~^ ERROR invalid register `f16`: unknown register
    }
}
