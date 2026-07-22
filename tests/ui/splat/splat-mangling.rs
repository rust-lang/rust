//! Test for splat symbol mangling.
//@ revisions: default legacy v0
//@ [default] compile-flags: -C opt-level=0
//@ [legacy] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=legacy
//@ [v0] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=v0
//@ build-fail

// CRATE_HASH normalization doesn't seem to work on some of these symbol logs
//@ normalize-stderr: "splat_mangling\[([0-9a-f]{16})\]::" -> "splat_mangling[CRATE_HASH]::"
//@ normalize-stderr: "h([0-9a-f]{16})E\)" -> "hCRATE_HASHE)"
//@ normalize-stderr: "::h([0-9a-f]{16})\)" -> "::hCRATE_HASH)"

#![allow(incomplete_features)]
#![feature(splat, rustc_attrs)]

fn main() {
    struct Type<T: ?Sized>(T);

    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    // FIXME(splat, legacy mangling): the first comma is in the wrong place
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main66Type$LT$fn$LP$$C$$u20$$u23$$u5b$splat$u5d$$LP$u8$C$u32$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(, #[splat](u8,u32))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(, #[splat](u8,u32))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMNvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFwThmEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(#[splat] (u8, u32))>>)
    impl Type<fn(#[splat] (u8, u32))> {}

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
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main77Type$LT$fn$LP$$C$$u20$$u23$$u5b$splat$u5d$$LP$$LP$u8$C$u32$RP$$C$$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(, #[splat]((u8,u32),))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(, #[splat]((u8,u32),))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs0_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFwTThmEEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(#[splat] ((u8, u32),))>>)
    impl Type<fn(#[splat] ((u8, u32),))> {}

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
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main80Type$LT$$BP$const$u20$fn$LP$$C$$u20$$u23$$u5b$splat$u5d$$LP$u32$C$i8$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<*const fn(, #[splat](u32,i8))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<*const fn(, #[splat](u32,i8))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs2_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypePFwTmaEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<*const fn(#[splat] (u32, i8))>>)
    impl Type<*const fn(#[splat] (u32, i8))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main52Type$LT$$BP$const$u20$fn$LP$$LP$u32$C$i8$RP$$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<*const fn((u32,i8))>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<*const fn((u32,i8))>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs3_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypePFTmaEEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<*const fn((u32, i8))>>)
    impl Type<*const fn((u32, i8))> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main72Type$LT$fn$LP$$C$$u20$$u23$$u5b$splat$u5d$$LP$u32$C$i8$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn(, #[splat](u32,i8),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn(, #[splat](u32,i8),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs4_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFwTmaEdEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn(#[splat] (u32, i8), f64)>>)
    impl Type<fn(#[splat] (u32, i8), f64)> {}

    #[rustfmt::skip]
    #[rustc_dump_symbol_name]
           //[legacy]~^ ERROR symbol-name(_ZN14splat_mangling4main44Type$LT$fn$LP$$LP$u32$C$i8$RP$$C$f64$RP$$GT
           //[legacy]~| ERROR demangling(splat_mangling::main::Type<fn((u32,i8),f64)>::
           //[legacy]~| ERROR demangling-alt(splat_mangling::main::Type<fn((u32,i8),f64)>)
    //[v0,default]~^^^^ ERROR symbol-name(_RMs5_NvCsCRATE_HASH_14splat_mangling4mainINtB<REF>_4TypeFTmaEdEuE)
       //[v0,default]~| ERROR demangling(<splat_mangling[
       //[v0,default]~| ERROR demangling-alt(<splat_mangling::main::Type<fn((u32, i8), f64)>>)
    impl Type<fn((u32, i8), f64)> {}
}
