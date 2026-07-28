//! Test for splat symbol mangling.
//@ revisions: default legacy v0
//@ [default] compile-flags: -C opt-level=0
//@ [legacy] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=legacy
//@ [v0] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=v0
//@ build-fail

// CRATE_HASH normalization doesn't seem to work on some of these symbol logs
//@ normalize-stderr: "splat_mangling\[([0-9a-f]{16})\]::" -> "splat_mangling[CRATE_HASH]::"
//@ normalize-stderr: "alloc\[([0-9a-f]{16})\]::" -> "alloc[CRATE_HASH]::"
//@ normalize-stderr: "h([0-9a-f]{16})E\)" -> "hCRATE_HASHE)"
//@ normalize-stderr: "::h([0-9a-f]{16})\)" -> "::hCRATE_HASH)"

#![allow(incomplete_features)]
#![feature(splat, rustc_attrs)]

fn main() {
    struct Type<T: ?Sized>(T);

    // Single argument splat, with different numbers of arguments inside the splat
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main69Type$LT$fn$LP$$u23$$u5b$rustc_splat$u5d$$u20$$LP$u8$C$u32$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(#[rustc_splat] (u8,u32))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(#[rustc_splat] (u8,u32))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMNvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFwThmEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(#[splat] (u8, u32))>>)
    impl Type<fn(#[rustc_splat] (u8, u32))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main38Type$LT$fn$LP$$LP$u8$C$u32$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn((u8,u32))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn((u8,u32))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFThmEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn((u8, u32))>>)
    impl Type<fn((u8, u32))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main80Type$LT$fn$LP$$u23$$u5b$rustc_splat$u5d$$u20$$LP$$LP$u8$C$u32$RP$$C$$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(#[rustc_splat] ((u8,u32),))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(#[rustc_splat] ((u8,u32),))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs0_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFwTThmEEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(#[splat] ((u8, u32),))>>)
    impl Type<fn(#[rustc_splat] ((u8, u32),))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main49Type$LT$fn$LP$$LP$$LP$u8$C$u32$RP$$C$$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(((u8,u32),))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(((u8,u32),))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs1_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFTThmEEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(((u8, u32),))>>)
    impl Type<fn(((u8, u32),))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main83Type$LT$$BP$const$u20$fn$LP$$u23$$u5b$rustc_splat$u5d$$u20$$LP$u32$C$i8$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<*const fn(#[rustc_splat] (u32,i8))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<*const fn(#[rustc_splat] (u32,i8))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs2_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypePFwTmaEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<*const fn(#[splat] (u32, i8))>>)
    impl Type<*const fn(#[rustc_splat] (u32, i8))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main52Type$LT$$BP$const$u20$fn$LP$$LP$u32$C$i8$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<*const fn((u32,i8))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<*const fn((u32,i8))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs3_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypePFTmaEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<*const fn((u32, i8))>>)
    impl Type<*const fn((u32, i8))> {}

    // Multi-argument, leading splat
    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main75Type$LT$fn$LP$$u23$$u5b$rustc_splat$u5d$$u20$$LP$u32$C$i8$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(#[rustc_splat] (u32,i8),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(#[rustc_splat] (u32,i8),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs4_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFwTmaEdEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(#[splat] (u32, i8), f64)>>)
    impl Type<fn(#[rustc_splat] (u32, i8), f64)> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main44Type$LT$fn$LP$$LP$u32$C$i8$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn((u32,i8),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn((u32,i8),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs5_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFTmaEdEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn((u32, i8), f64)>>)
    impl Type<fn((u32, i8), f64)> {}

    // Multi-argument, trailing splat
    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main90Type$LT$$BP$mut$u20$fn$LP$u32$C$i8$C$$u23$$u5b$rustc_splat$u5d$$u20$$LP$f64$C$$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<*mut fn(u32,i8,#[rustc_splat] (f64,))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<*mut fn(u32,i8,#[rustc_splat] (f64,))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs6_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeOFmawTdEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<*mut fn(u32, i8, #[splat] (f64,))>>)
    impl Type<*mut fn(u32, i8, #[rustc_splat] (f64,))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main59Type$LT$$BP$mut$u20$fn$LP$u32$C$i8$C$$LP$f64$C$$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<*mut fn(u32,i8,(f64,))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<*mut fn(u32,i8,(f64,))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs7_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeOFmaTdEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<*mut fn(u32, i8, (f64,))>>)
    impl Type<*mut fn(u32, i8, (f64,))> {}

    // Multi-argument, middle splat
    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main93Type$LT$$RF$fn$LP$u32$C$$u23$$u5b$rustc_splat$u5d$$u20$$LP$i8$C$f32$C$usize$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<&fn(u32,#[rustc_splat] (i8,f32,usize),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<&fn(u32,#[rustc_splat] (i8,f32,usize),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs8_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeRFmwTafjEdEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<&fn(u32, #[splat] (i8, f32, usize), f64)>>)
    impl Type<&fn(u32, #[rustc_splat] (i8, f32, usize), f64)> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main62Type$LT$$RF$fn$LP$u32$C$$LP$i8$C$f32$C$usize$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<&fn(u32,(i8,f32,usize),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<&fn(u32,(i8,f32,usize),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs9_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeRFmTafjEdEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<&fn(u32, (i8, f32, usize), f64)>>)
    impl Type<&fn(u32, (i8, f32, usize), f64)> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main85Type$LT$$RF$mut$u20$fn$LP$u32$C$$u23$$u5b$rustc_splat$u5d$$u20$$LP$$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<&mut fn(u32,#[rustc_splat] (),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<&mut fn(u32,#[rustc_splat] (),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMsa_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeQFmwudEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<&mut fn(u32, #[splat] (), f64)>>)
    impl Type<&mut fn(u32, #[rustc_splat] (), f64)> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main54Type$LT$$RF$mut$u20$fn$LP$u32$C$$LP$$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<&mut fn(u32,(),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<&mut fn(u32,(),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMsb_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeQFmudEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<&mut fn(u32, (), f64)>>)
    impl Type<&mut fn(u32, (), f64)> {}

    // Splats within splats
    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main152Type$LT$alloc..boxed..Box$LT$fn$LP$u32$C$$u23$$u5b$rustc_splat$u5d$$u20$$LP$fn$LP$$u23$$u5b$rustc_splat$u5d$$u20$$LP$$RP$$RP$$C$i8$RP$$C$f64$RP$$GT$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,#[rustc_splat] (fn(#[rustc_splat] ()),i8),f64)>>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,#[rustc_splat] (fn(#[rustc_splat] ()),i8),f64)>>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMsc_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeINtNtCsCRATE_HASH_5alloc5boxed3BoxFmwTFwuEuaEdEuEE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<alloc::boxed::Box<fn(u32, #[splat] (fn(#[splat] ()), i8), f64)>>>)
    impl Type<Box<fn(u32, #[rustc_splat] (fn(#[rustc_splat] ()), i8), f64)>> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main121Type$LT$alloc..boxed..Box$LT$fn$LP$u32$C$$LP$fn$LP$$u23$$u5b$rustc_splat$u5d$$u20$$LP$$RP$$RP$$C$i8$RP$$C$f64$RP$$GT$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,(fn(#[rustc_splat] ()),i8),f64)>>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,(fn(#[rustc_splat] ()),i8),f64)>>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMsd_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeINtNtCsCRATE_HASH_5alloc5boxed3BoxFmTFwuEuaEdEuEE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<alloc::boxed::Box<fn(u32, (fn(#[splat] ()), i8), f64)>>>)
    impl Type<Box<fn(u32, (fn(#[rustc_splat] ()), i8), f64)>> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main121Type$LT$alloc..boxed..Box$LT$fn$LP$u32$C$$u23$$u5b$rustc_splat$u5d$$u20$$LP$fn$LP$$LP$$RP$$RP$$C$i8$RP$$C$f64$RP$$GT$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,#[rustc_splat] (fn(()),i8),f64)>>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,#[rustc_splat] (fn(()),i8),f64)>>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMse_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeINtNtCsCRATE_HASH_5alloc5boxed3BoxFmwTFuEuaEdEuEE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<alloc::boxed::Box<fn(u32, #[splat] (fn(()), i8), f64)>>>)
    impl Type<Box<fn(u32, #[rustc_splat] (fn(()), i8), f64)>> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main90Type$LT$alloc..boxed..Box$LT$fn$LP$u32$C$$LP$fn$LP$$LP$$RP$$RP$$C$i8$RP$$C$f64$RP$$GT$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,(fn(()),i8),f64)>>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<alloc::boxed::Box<fn(u32,(fn(()),i8),f64)>>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMsf_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeINtNtCsCRATE_HASH_5alloc5boxed3BoxFmTFuEuaEdEuEE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<alloc::boxed::Box<fn(u32, (fn(()), i8), f64)>>>)
    impl Type<Box<fn(u32, (fn(()), i8), f64)>> {}
}
