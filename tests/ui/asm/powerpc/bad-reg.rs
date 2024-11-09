//@ revisions: powerpc powerpc64 powerpc64le aix64
//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//@[powerpc64le] needs-llvm-components: powerpc
//@[aix64] compile-flags: --target powerpc64-ibm-aix
//@[aix64] needs-llvm-components: powerpc
//@ needs-asm-support

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items, asm_experimental_arch)]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

impl Copy for i32 {}

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

fn f() {
    let mut x = 0;
    unsafe {
        // Unsupported registers
        asm!("", out("sp") _);
        //~^ ERROR invalid register `sp`: the stack pointer cannot be used as an operand for inline asm
        asm!("", out("r2") _);
        //~^ ERROR invalid register `r2`: r2 is a system reserved register and cannot be used as an operand for inline asm
        asm!("", out("r13") _);
        //~^ ERROR cannot use register `r13`: r13 is a reserved register on this target
        asm!("", out("r29") _);
        //~^ ERROR invalid register `r29`: r29 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("r30") _);
        //~^ ERROR invalid register `r30`: r30 is used internally by LLVM and cannot be used as an operand for inline asm
        asm!("", out("fp") _);
        //~^ ERROR invalid register `fp`: the frame pointer cannot be used as an operand for inline asm
        asm!("", out("lr") _);
        //~^ ERROR invalid register `lr`: the link register cannot be used as an operand for inline asm
        asm!("", out("ctr") _);
        //~^ ERROR invalid register `ctr`: the counter register cannot be used as an operand for inline asm
        asm!("", out("vrsave") _);
        //~^ ERROR invalid register `vrsave`: the vrsave register cannot be used as an operand for inline asm
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
        // vreg
        asm!("", out("v0") _); // ok
        // FIXME: will be supported in the subsequent patch: https://github.com/rust-lang/rust/pull/131551
        asm!("", in("v0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("", out("v0") x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", in(vreg) x);
        //~^ ERROR can only be used as a clobber
        //~| ERROR type `i32` cannot be used with this register class
        asm!("/* {} */", out(vreg) _);
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
    }
}
