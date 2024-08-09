//@ build-fail
//@ revisions: legacy v0
//@ compile-flags: --crate-name=c
//@[legacy]compile-flags: -C symbol-mangling-version=legacy -Z unstable-options
//@    [v0]compile-flags: -C symbol-mangling-version=v0
//@[legacy]normalize-stderr-test: "h[[:xdigit:]]{16}" -> "h[HASH]"
//@    [v0]normalize-stderr-test: "c\[.*?\]" -> "c[HASH]"
#![feature(rustc_attrs)]

pub struct Unsigned<const F: u8>;

impl Unsigned<11> {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name(_RNvMCs
    //[v0]~| ERROR demangling(<c[
    //[v0]~| ERROR demangling-alt(<c::Unsigned<11>>::f)
    //[legacy]~^^^^ ERROR symbol-name(_ZN1c17Unsigned$LT$_$GT$
    //[legacy]~|    ERROR demangling(c::Unsigned<_>::f::
    //[legacy]~|    ERROR demangling-alt(c::Unsigned<_>::f)
    fn f() {}
}

pub struct Signed<const F: i16>;

impl Signed<-152> {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name(_RNvMs_Cs
    //[v0]~| ERROR demangling(<c[
    //[v0]~| ERROR demangling-alt(<c::Signed<-152>>::f)
    //[legacy]~^^^^ ERROR symbol-name(_ZN1c15Signed$LT$_$GT$
    //[legacy]~|    ERROR demangling(c::Signed<_>::f::
    //[legacy]~|    ERROR demangling-alt(c::Signed<_>::f)
    fn f() {}
}

pub struct Bool<const F: bool>;

impl Bool<true> {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name(_RNvMs0_Cs
    //[v0]~| ERROR demangling(<c[
    //[v0]~| ERROR demangling-alt(<c::Bool<true>>::f)
    //[legacy]~^^^^ ERROR symbol-name(_ZN1c13Bool$LT$_$GT$
    //[legacy]~|    ERROR demangling(c::Bool<_>::f::
    //[legacy]~|    ERROR demangling-alt(c::Bool<_>::f)
    fn f() {}
}

pub struct Char<const F: char>;

impl Char<'∂'> {
    #[rustc_symbol_name]
    //[v0]~^ ERROR symbol-name(_RNvMs1_Cs
    //[v0]~| ERROR demangling(<c[
    //[v0]~| ERROR demangling-alt(<c::Char<'∂'>>::f)
    //[legacy]~^^^^ ERROR symbol-name(_ZN1c13Char$LT$_$GT$
    //[legacy]~|    ERROR demangling(c::Char<_>::f::
    //[legacy]~|    ERROR demangling-alt(c::Char<_>::f)
    fn f() {}
}

fn main() {}
