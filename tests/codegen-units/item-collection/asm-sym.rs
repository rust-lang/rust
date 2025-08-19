//@ needs-asm-support
//@ compile-flags: -Ccodegen-units=1 --crate-type=lib

#[inline(always)]
pub unsafe fn f() {
    //~ MONO_ITEM static f::S @@ asm_sym-cgu.0[External]
    static S: usize = 1;
    //~ MONO_ITEM fn f::fun @@ asm_sym-cgu.0[External]
    #[inline(never)]
    fn fun() {}
    core::arch::asm!("/* {0} {1} */", sym S, sym fun);
}

//~ MONO_ITEM fn g @@ asm_sym-cgu.0[External]
#[inline(never)]
pub unsafe fn g() {
    //~ MONO_ITEM static g::S @@ asm_sym-cgu.0[Internal]
    static S: usize = 2;
    //~ MONO_ITEM fn g::fun @@ asm_sym-cgu.0[Internal]
    #[inline(never)]
    fn fun() {}
    core::arch::asm!("/* {0} {1} */", sym S, sym fun);
}
