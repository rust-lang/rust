//@ build-fail
//@ revisions: legacy verbose-legacy v0
//@ compile-flags: --crate-name=a -Z unstable-options
//@ [legacy] compile-flags: -Csymbol-mangling-version=legacy
//@ [verbose-legacy] compile-flags: -Csymbol-mangling-version=legacy -Zverbose-internals
//@ [v0] compile-flags: -Csymbol-mangling-version=v0
//@ normalize-stderr: "h[[:xdigit:]]{16}" -> "h[HASH]"
//@ [v0] normalize-stderr: "\[[[:xdigit:]]{16}\]" -> "[HASH]"

#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(f128)]
#![feature(f16)]

pub fn b() {
    struct Type<T: ?Sized>(T);

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b16Type$LT$bool$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<bool>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<bool>)
    //[v0]~^^^^ ERROR symbol-name(_RMNvCsCRATE_HASH_1a1bINtB<REF>_4TypebE)
    //[v0]~| ERROR ::b::Type<bool>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<bool>>)
    impl Type<bool> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b16Type$LT$char$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<char>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<char>)
    //[v0]~^^^^ ERROR symbol-name(_RMs_NvCsCRATE_HASH_1a1bINtB<REF>_4TypecE)
    //[v0]~| ERROR ::b::Type<char>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<char>>)
    impl Type<char> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b14Type$LT$i8$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<i8>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<i8>)
    //[v0]~^^^^ ERROR symbol-name(_RMs0_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeaE)
    //[v0]~| ERROR ::b::Type<i8>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<i8>>)
    impl Type<i8> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$i16$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<i16>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<i16>)
    //[v0]~^^^^ ERROR symbol-name(_RMs1_NvCsCRATE_HASH_1a1bINtB<REF>_4TypesE)
    //[v0]~| ERROR ::b::Type<i16>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<i16>>)
    impl Type<i16> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$i32$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<i32>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<i32>)
    //[v0]~^^^^ ERROR symbol-name(_RMs2_NvCsCRATE_HASH_1a1bINtB<REF>_4TypelE)
    //[v0]~| ERROR ::b::Type<i32>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<i32>>)
    impl Type<i32> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$i64$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<i64>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<i64>)
    //[v0]~^^^^ ERROR symbol-name(_RMs3_NvCsCRATE_HASH_1a1bINtB<REF>_4TypexE)
    //[v0]~| ERROR ::b::Type<i64>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<i64>>)
    impl Type<i64> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b14Type$LT$u8$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<u8>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<u8>)
    //[v0]~^^^^ ERROR symbol-name(_RMs4_NvCsCRATE_HASH_1a1bINtB<REF>_4TypehE)
    //[v0]~| ERROR ::b::Type<u8>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<u8>>)
    impl Type<u8> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$u16$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<u16>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<u16>)
    //[v0]~^^^^ ERROR symbol-name(_RMs5_NvCsCRATE_HASH_1a1bINtB<REF>_4TypetE)
    //[v0]~| ERROR ::b::Type<u16>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<u16>>)
    impl Type<u16> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$u32$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<u32>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<u32>)
    //[v0]~^^^^ ERROR symbol-name(_RMs6_NvCsCRATE_HASH_1a1bINtB<REF>_4TypemE)
    //[v0]~| ERROR ::b::Type<u32>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<u32>>)
    impl Type<u32> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$u64$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<u64>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<u64>)
    //[v0]~^^^^ ERROR symbol-name(_RMs7_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeyE)
    //[v0]~| ERROR ::b::Type<u64>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<u64>>)
    impl Type<u64> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$f16$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<f16>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<f16>)
    //[v0]~^^^^ ERROR symbol-name(_RMs8_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeC3f16E)
    //[v0]~| ERROR ::b::Type<f16>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<f16>>)
    impl Type<f16> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$f32$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<f32>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<f32>)
    //[v0]~^^^^ ERROR symbol-name(_RMs9_NvCsCRATE_HASH_1a1bINtB<REF>_4TypefE)
    //[v0]~| ERROR ::b::Type<f32>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<f32>>)
    impl Type<f32> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$f64$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<f64>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<f64>)
    //[v0]~^^^^ ERROR symbol-name(_RMsa_NvCsCRATE_HASH_1a1bINtB<REF>_4TypedE)
    //[v0]~| ERROR ::b::Type<f64>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<f64>>)
    impl Type<f64> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b16Type$LT$f128$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<f128>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<f128>)
    //[v0]~^^^^ ERROR symbol-name(_RMsb_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeC4f128E)
    //[v0]~| ERROR ::b::Type<f128>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<f128>>)
    impl Type<f128> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b15Type$LT$str$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<str>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<str>)
    //[v0]~^^^^ ERROR symbol-name(_RMsc_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeeE)
    //[v0]~| ERROR ::b::Type<str>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<str>>)
    impl Type<str> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b17Type$LT$$u21$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<!>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<!>)
    //[v0]~^^^^ ERROR symbol-name(_RMsd_NvCsCRATE_HASH_1a1bINtB<REF>_4TypezE)
    //[v0]~| ERROR ::b::Type<!>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<!>>)
    impl Type<!> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b20Type$LT$$LP$$RP$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<()>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<()>)
    //[v0]~^^^^ ERROR symbol-name(_RMse_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeuE)
    //[v0]~| ERROR ::b::Type<()>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<()>>)
    impl Type<()> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b25Type$LT$$LP$u8$C$$RP$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<(u8,)>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<(u8,)>)
    //[v0]~^^^^ ERROR symbol-name(_RMsf_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeThEE)
    //[v0]~| ERROR ::b::Type<(u8,)>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<(u8,)>>)
    impl Type<(u8,)> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b28Type$LT$$LP$u8$C$u16$RP$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<(u8,u16)>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<(u8,u16)>)
    //[v0]~^^^^ ERROR symbol-name(_RMsg_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeThtEE)
    //[v0]~| ERROR ::b::Type<(u8, u16)>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<(u8, u16)>>)
    impl Type<(u8, u16)> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b34Type$LT$$LP$u8$C$u16$C$u32$RP$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<(u8,u16,u32)>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<(u8,u16,u32)>)
    //[v0]~^^^^ ERROR symbol-name(_RMsh_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeThtmEE)
    //[v0]~| ERROR ::b::Type<(u8, u16, u32)>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<(u8, u16, u32)>>)
    impl Type<(u8, u16, u32)> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b28Type$LT$$BP$const$u20$u8$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<*const u8>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<*const u8>)
    //[v0]~^^^^ ERROR symbol-name(_RMsi_NvCsCRATE_HASH_1a1bINtB<REF>_4TypePhE)
    //[v0]~| ERROR ::b::Type<*const u8>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<*const u8>>)
    impl Type<*const u8> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b26Type$LT$$BP$mut$u20$u8$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<*mut u8>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<*mut u8>)
    //[v0]~^^^^ ERROR symbol-name(_RMsj_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeOhE)
    //[v0]~| ERROR ::b::Type<*mut u8>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<*mut u8>>)
    impl Type<*mut u8> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b19Type$LT$$RF$str$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<&str>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<&str>)
    //[v0]~^^^^ ERROR symbol-name(_RMsk_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeReE)
    //[v0]~| ERROR ::b::Type<&str>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<&str>>)
    impl Type<&str> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b27Type$LT$$RF$mut$u20$str$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<&mut str>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<&mut str>)
    //[v0]~^^^^ ERROR symbol-name(_RMsl_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeQeE)
    //[v0]~| ERROR ::b::Type<&mut str>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<&mut str>>)
    impl Type<&mut str> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b35Type$LT$$u5b$u8$u3b$$u20$0$u5d$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<[u8; 0]>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<[u8; 0]>)
    //[v0]~^^^^ ERROR symbol-name(_RMsm_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeAhj0_E)
    //[v0]~| ERROR ::b::Type<[u8; 0usize]>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<[u8; 0]>>)
    impl Type<[u8; 0]> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b22Type$LT$fn$LP$$RP$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<fn()>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<fn()>)
    //[v0]~^^^^ ERROR symbol-name(_RMsn_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeFEuE)
    //[v0]~| ERROR ::b::Type<fn()>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<fn()>>)
    impl Type<fn()> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b60Type$LT$unsafe$u20$extern$u20$$u22$C$u22$$u20$fn$LP$$RP$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<unsafe extern "C" fn()>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<unsafe extern "C" fn()>)
    //[v0]~^^^^ ERROR symbol-name(_RMso_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeFUKCEuE)
    //[v0]~| ERROR ::b::Type<unsafe extern "C" fn()>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<unsafe extern "C" fn()>>)
    impl Type<unsafe extern "C" fn()> {}

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b34Type$LT$$u5b$T$u3b$$u20$N$u5d$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<[T; N]>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<[T; N]>)
    //[v0]~^^^^ ERROR symbol-name(_RMsp_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeAppEB<REF>_)
    //[v0]~| ERROR ::b::Type<[_; _]>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<[_; _]>>)
    impl<const N: usize, T> Type<[T; N]> {}

    const ZERO: usize = 0;

    #[rustc_symbol_name]
    //[legacy,verbose-legacy]~^ ERROR symbol-name(_ZN1a1b35Type$LT$$u5b$u8$u3b$$u20$0$u5d$$GT$
    //[legacy,verbose-legacy]~| ERROR demangling(a::b::Type<[u8; 0]>::
    //[legacy,verbose-legacy]~| ERROR demangling-alt(a::b::Type<[u8; 0]>)
    //[v0]~^^^^ ERROR symbol-name(_RMsq_NvCsCRATE_HASH_1a1bINtB<REF>_4TypeAhj0_E)
    //[v0]~| ERROR ::b::Type<[u8; 0usize]>>)
    //[v0]~| ERROR demangling-alt(<a::b::Type<[u8; 0]>>)
    impl Type<[u8; ZERO]> {}
}

fn main() {}
