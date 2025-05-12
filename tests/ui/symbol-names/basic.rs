//@ build-fail
//@ revisions: legacy v0
//@[legacy]compile-flags: -Z unstable-options -C symbol-mangling-version=legacy
//@[v0]compile-flags: -C symbol-mangling-version=v0

#![feature(rustc_attrs)]

#[rustc_symbol_name]
//[legacy]~^ ERROR symbol-name(_ZN5basic4main
//[legacy]~| ERROR demangling(basic::main
//[legacy]~| ERROR demangling-alt(basic::main)
 //[v0]~^^^^ ERROR symbol-name(_RNv
    //[v0]~| ERROR demangling(basic[
    //[v0]~| ERROR demangling-alt(basic::main)
#[rustc_def_path]
//[legacy]~^ ERROR def-path(main)
   //[v0]~^^ ERROR def-path(main)
fn main() {
}
