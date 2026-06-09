//@ add-minicore
//@ revisions: powerpc powerpc64 powerpc64le aix64 powerpcspe
//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//@[powerpc64le] needs-llvm-components: powerpc
//@[aix64] compile-flags: --target powerpc64-ibm-aix
//@[aix64] needs-llvm-components: powerpc
//@[powerpcspe] compile-flags: --target powerpc-unknown-linux-gnuspe
//@[powerpcspe] needs-llvm-components: powerpc
//@ ignore-backends: gcc
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core, repr_simd, asm_experimental_arch)]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i32x4([i32; 4]);
#[repr(simd)]
pub struct i64x2([i64; 2]);

impl Copy for i32x4 {}
impl Copy for i64x2 {}

fn f() {
    let mut x = 0;
    let mut v32x4 = i32x4([0; 4]);
    let mut v64x2 = i64x2([0; 2]);
    unsafe {
        // Unsupported registers
        asm!("", out("sp") _);
        //~^ ERROR invalid register `sp`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("r2") _);
        //~^ ERROR invalid register `r2`: r2 is a system reserved register and cannot be used as an operand for inline asm
        asm!("", out("r13") _);
        //~^ ERROR cannot use register `r13`: r13 is a reserved register on this target
        asm!("", out("r29") _);
        //[powerpc,powerpcspe]~^ ERROR cannot use register `r29`: r29 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("r30") _);
        //~^ ERROR invalid register `r30`: r30 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("fp") _);
        //~^ ERROR invalid register `fp`: the frame pointer cannot be used as an operand for inline asm
        asm!("", out("vrsave") _);
        //~^ ERROR invalid register `vrsave`: the vrsave register cannot be used as an operand for inline asm

        // vreg
        asm!("", out("v0") _); // always ok
        asm!("", in("v0") v32x4); // requires altivec
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        asm!("", out("v0") v32x4); // requires altivec
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        asm!("", in("v0") v64x2); // requires vsx
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        //[powerpc64]~^^ ERROR `vsx` target feature is not enabled
        asm!("", out("v0") v64x2); // requires vsx
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        //[powerpc64]~^^ ERROR `vsx` target feature is not enabled
        asm!("", in("v0") x); // FIXME: should be ok if vsx is available
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        //[powerpc64,powerpc64le,aix64]~^^ ERROR type `i32` cannot be used with this register class
        asm!("", out("v0") x); // FIXME: should be ok if vsx is available
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        //[powerpc64,powerpc64le,aix64]~^^ ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(vreg) v32x4); // requires altivec
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        asm!("/* {} */", in(vreg) v64x2); // requires vsx
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        //[powerpc64]~^^ ERROR `vsx` target feature is not enabled
        asm!("/* {} */", in(vreg) x); // FIXME: should be ok if vsx is available
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        //[powerpc64,powerpc64le,aix64]~^^ ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(vreg) _); // requires altivec
        //[powerpc,powerpcspe]~^ ERROR register class `vreg` requires at least one of the following target features: altivec, vsx
        // v20-v31 (vs52-vs63) are reserved on AIX with vec-default ABI (this ABI is not currently used in Rust's builtin AIX targets).
        asm!("", out("v20") _);
        asm!("", out("v21") _);
        asm!("", out("v22") _);
        asm!("", out("v23") _);
        asm!("", out("v24") _);
        asm!("", out("v25") _);
        asm!("", out("v26") _);
        asm!("", out("v27") _);
        asm!("", out("v28") _);
        asm!("", out("v29") _);
        asm!("", out("v30") _);
        asm!("", out("v31") _);


        // vsreg
        asm!("", out("vs0") _); // always ok
        asm!("", in("vs0") v32x4); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        asm!("", out("vs0") v32x4); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        asm!("", in("vs0") v64x2); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        asm!("", out("vs0") v64x2); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        asm!("", in("vs0") x); // FIXME: should be ok if vsx is available
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        //[powerpc64le,aix64]~^^ ERROR type `i32` cannot be used with this register class
        asm!("", out("vs0") x); // FIXME: should be ok if vsx is available
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        //[powerpc64le,aix64]~^^ ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(vsreg) v32x4); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        asm!("/* {} */", in(vsreg) v64x2); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        asm!("/* {} */", in(vsreg) x); // FIXME: should be ok if vsx is available
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature
        //[powerpc64le,aix64]~^^ ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(vsreg) _); // requires vsx
        //[powerpc,powerpcspe,powerpc64]~^ ERROR register class `vsreg` requires the `vsx` target feature

        // v20-v31 (vs52-vs63) are reserved on AIX with vec-default ABI (this ABI is not currently used in Rust's builtin AIX targets).
        asm!("", out("vs52") _);
        asm!("", out("vs53") _);
        asm!("", out("vs54") _);
        asm!("", out("vs55") _);
        asm!("", out("vs56") _);
        asm!("", out("vs57") _);
        asm!("", out("vs58") _);
        asm!("", out("vs59") _);
        asm!("", out("vs60") _);
        asm!("", out("vs61") _);
        asm!("", out("vs62") _);
        asm!("", out("vs63") _);

        // Clobber-only registers
        // cr
        asm!("", out("cr") _); // ok
        asm!("", in("cr") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("cr") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(cr) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(cr) _);
        //~^ ERROR can only be used as a clobber
        // ctr
        asm!("", out("ctr") _); // ok
        asm!("", in("ctr") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("ctr") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(ctr) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(ctr) _);
        //~^ ERROR can only be used as a clobber
        // lr
        asm!("", out("lr") _); // ok
        asm!("", in("lr") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("lr") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(lr) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(lr) _);
        //~^ ERROR can only be used as a clobber
        // xer
        asm!("", out("xer") _); // ok
        asm!("", in("xer") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("xer") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(xer) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(xer) _);
        //~^ ERROR can only be used as a clobber

        // Overlapping-only registers
        asm!("", out("cr") _, out("cr0") _);
        //~^ ERROR register `cr0` conflicts with register `cr`
        asm!("", out("cr") _, out("cr1") _);
        //~^ ERROR register `cr1` conflicts with register `cr`
        asm!("", out("cr") _, out("cr2") _);
        //~^ ERROR register `cr2` conflicts with register `cr`
        asm!("", out("cr") _, out("cr3") _);
        //~^ ERROR register `cr3` conflicts with register `cr`
        asm!("", out("cr") _, out("cr4") _);
        //~^ ERROR register `cr4` conflicts with register `cr`
        asm!("", out("cr") _, out("cr5") _);
        //~^ ERROR register `cr5` conflicts with register `cr`
        asm!("", out("cr") _, out("cr6") _);
        //~^ ERROR register `cr6` conflicts with register `cr`
        asm!("", out("cr") _, out("cr7") _);
        //~^ ERROR register `cr7` conflicts with register `cr`
        asm!("", out("f0") _, out("v0") _); // ok
        asm!("", out("f0") _, out("vs0") _);
        //~^ ERROR register `vs0` conflicts with register `f0`
        asm!("", out("f1") _, out("vs1") _);
        //~^ ERROR register `vs1` conflicts with register `f1`
        asm!("", out("f2") _, out("vs2") _);
        //~^ ERROR register `vs2` conflicts with register `f2`
        asm!("", out("f3") _, out("vs3") _);
        //~^ ERROR register `vs3` conflicts with register `f3`
        asm!("", out("f4") _, out("vs4") _);
        //~^ ERROR register `vs4` conflicts with register `f4`
        asm!("", out("f5") _, out("vs5") _);
        //~^ ERROR register `vs5` conflicts with register `f5`
        asm!("", out("f6") _, out("vs6") _);
        //~^ ERROR register `vs6` conflicts with register `f6`
        asm!("", out("f7") _, out("vs7") _);
        //~^ ERROR register `vs7` conflicts with register `f7`
        asm!("", out("f8") _, out("vs8") _);
        //~^ ERROR register `vs8` conflicts with register `f8`
        asm!("", out("f9") _, out("vs9") _);
        //~^ ERROR register `vs9` conflicts with register `f9`
        asm!("", out("f10") _, out("vs10") _);
        //~^ ERROR register `vs10` conflicts with register `f10`
        asm!("", out("f11") _, out("vs11") _);
        //~^ ERROR register `vs11` conflicts with register `f11`
        asm!("", out("f12") _, out("vs12") _);
        //~^ ERROR register `vs12` conflicts with register `f12`
        asm!("", out("f13") _, out("vs13") _);
        //~^ ERROR register `vs13` conflicts with register `f13`
        asm!("", out("f14") _, out("vs14") _);
        //~^ ERROR register `vs14` conflicts with register `f14`
        asm!("", out("f15") _, out("vs15") _);
        //~^ ERROR register `vs15` conflicts with register `f15`
        asm!("", out("f16") _, out("vs16") _);
        //~^ ERROR register `vs16` conflicts with register `f16`
        asm!("", out("f17") _, out("vs17") _);
        //~^ ERROR register `vs17` conflicts with register `f17`
        asm!("", out("f18") _, out("vs18") _);
        //~^ ERROR register `vs18` conflicts with register `f18`
        asm!("", out("f19") _, out("vs19") _);
        //~^ ERROR register `vs19` conflicts with register `f19`
        asm!("", out("f20") _, out("vs20") _);
        //~^ ERROR register `vs20` conflicts with register `f20`
        asm!("", out("f21") _, out("vs21") _);
        //~^ ERROR register `vs21` conflicts with register `f21`
        asm!("", out("f22") _, out("vs22") _);
        //~^ ERROR register `vs22` conflicts with register `f22`
        asm!("", out("f23") _, out("vs23") _);
        //~^ ERROR register `vs23` conflicts with register `f23`
        asm!("", out("f24") _, out("vs24") _);
        //~^ ERROR register `vs24` conflicts with register `f24`
        asm!("", out("f25") _, out("vs25") _);
        //~^ ERROR register `vs25` conflicts with register `f25`
        asm!("", out("f26") _, out("vs26") _);
        //~^ ERROR register `vs26` conflicts with register `f26`
        asm!("", out("f27") _, out("vs27") _);
        //~^ ERROR register `vs27` conflicts with register `f27`
        asm!("", out("f28") _, out("vs28") _);
        //~^ ERROR register `vs28` conflicts with register `f28`
        asm!("", out("f29") _, out("vs29") _);
        //~^ ERROR register `vs29` conflicts with register `f29`
        asm!("", out("f30") _, out("vs30") _);
        //~^ ERROR register `vs30` conflicts with register `f30`
        asm!("", out("f31") _, out("vs31") _);
        //~^ ERROR register `vs31` conflicts with register `f31`
        asm!("", out("vs32") _, out("v0") _);
        //~^ ERROR register `v0` conflicts with register `vs32`
        asm!("", out("vs33") _, out("v1") _);
        //~^ ERROR register `v1` conflicts with register `vs33`
        asm!("", out("vs34") _, out("v2") _);
        //~^ ERROR register `v2` conflicts with register `vs34`
        asm!("", out("vs35") _, out("v3") _);
        //~^ ERROR register `v3` conflicts with register `vs35`
        asm!("", out("vs36") _, out("v4") _);
        //~^ ERROR register `v4` conflicts with register `vs36`
        asm!("", out("vs37") _, out("v5") _);
        //~^ ERROR register `v5` conflicts with register `vs37`
        asm!("", out("vs38") _, out("v6") _);
        //~^ ERROR register `v6` conflicts with register `vs38`
        asm!("", out("vs39") _, out("v7") _);
        //~^ ERROR register `v7` conflicts with register `vs39`
        asm!("", out("vs40") _, out("v8") _);
        //~^ ERROR register `v8` conflicts with register `vs40`
        asm!("", out("vs41") _, out("v9") _);
        //~^ ERROR register `v9` conflicts with register `vs41`
        asm!("", out("vs42") _, out("v10") _);
        //~^ ERROR register `v10` conflicts with register `vs42`
        asm!("", out("vs43") _, out("v11") _);
        //~^ ERROR register `v11` conflicts with register `vs43`
        asm!("", out("vs44") _, out("v12") _);
        //~^ ERROR register `v12` conflicts with register `vs44`
        asm!("", out("vs45") _, out("v13") _);
        //~^ ERROR register `v13` conflicts with register `vs45`
        asm!("", out("vs46") _, out("v14") _);
        //~^ ERROR register `v14` conflicts with register `vs46`
        asm!("", out("vs47") _, out("v15") _);
        //~^ ERROR register `v15` conflicts with register `vs47`
        asm!("", out("vs48") _, out("v16") _);
        //~^ ERROR register `v16` conflicts with register `vs48`
        asm!("", out("vs49") _, out("v17") _);
        //~^ ERROR register `v17` conflicts with register `vs49`
        asm!("", out("vs50") _, out("v18") _);
        //~^ ERROR register `v18` conflicts with register `vs50`
        asm!("", out("vs51") _, out("v19") _);
        //~^ ERROR register `v19` conflicts with register `vs51`
        asm!("", out("vs52") _, out("v20") _);
        //~^ ERROR register `v20` conflicts with register `vs52`
        asm!("", out("vs53") _, out("v21") _);
        //~^ ERROR register `v21` conflicts with register `vs53`
        asm!("", out("vs54") _, out("v22") _);
        //~^ ERROR register `v22` conflicts with register `vs54`
        asm!("", out("vs55") _, out("v23") _);
        //~^ ERROR register `v23` conflicts with register `vs55`
        asm!("", out("vs56") _, out("v24") _);
        //~^ ERROR register `v24` conflicts with register `vs56`
        asm!("", out("vs57") _, out("v25") _);
        //~^ ERROR register `v25` conflicts with register `vs57`
        asm!("", out("vs58") _, out("v26") _);
        //~^ ERROR register `v26` conflicts with register `vs58`
        asm!("", out("vs59") _, out("v27") _);
        //~^ ERROR register `v27` conflicts with register `vs59`
        asm!("", out("vs60") _, out("v28") _);
        //~^ ERROR register `v28` conflicts with register `vs60`
        asm!("", out("vs61") _, out("v29") _);
        //~^ ERROR register `v29` conflicts with register `vs61`
        asm!("", out("vs62") _, out("v30") _);
        //~^ ERROR register `v30` conflicts with register `vs62`
        asm!("", out("vs63") _, out("v31") _);
        //~^ ERROR register `v31` conflicts with register `vs63`

        // powerpc-*spe target specific tests
        asm!("", out("spe_acc") _);
        //[aix64,powerpc,powerpc64,powerpc64le]~^ ERROR cannot use register `spe_acc`: spe_acc is only available on spe targets
        asm!("/* {} */", out(spe_acc) _);
        //~^ ERROR can only be used as a clobber
    }
}
