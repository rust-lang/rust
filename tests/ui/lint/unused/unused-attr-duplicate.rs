// Tests for repeating attribute warnings.
//@ aux-build:lint_unused_extern_crate.rs
//@ compile-flags:--test
// Not tested due to extra requirements:
// - panic_handler: needs extra setup
// - target_feature: platform-specific
// - link_section: platform-specific
// - proc_macro, proc_macro_derive, proc_macro_attribute: needs to be a
//   proc-macro, and have special handling for mixing.
// - unstable attributes (not going to bother)
// - no_main: extra setup
#![deny(unused_attributes)]
#![crate_name = "unused_attr_duplicate"]
#![crate_name = "unused_attr_duplicate2"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
#![recursion_limit = "128"]
#![recursion_limit = "256"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
#![type_length_limit = "1048576"]
#![type_length_limit = "1"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
#![no_std]
#![no_std] //~ ERROR unused attribute
#![no_implicit_prelude]
#![no_implicit_prelude] //~ ERROR unused attribute
#![windows_subsystem = "console"]
#![windows_subsystem = "windows"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
#![no_builtins]
#![no_builtins] //~ ERROR unused attribute

#[no_link]
#[no_link] //~ ERROR unused attribute
extern crate lint_unused_extern_crate;

#[macro_use]
#[macro_use] //~ ERROR unused attribute
pub mod m {
    #[macro_export]
    #[macro_export] //~ ERROR unused attribute
    macro_rules! foo {
        () => {};
    }
}

#[path = "auxiliary/lint_unused_extern_crate.rs"]
#[path = "bar.rs"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
pub mod from_path;

#[test]
#[ignore]
#[ignore = "some text"] //~ ERROR unused attribute
#[should_panic]
#[should_panic(expected = "values don't match")] //~ ERROR unused attribute
//~^ WARN this was previously accepted
fn t1() {}

#[must_use]
#[must_use = "some message"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
// No warnings for #[repr], would require more logic.
#[repr(C)]
#[repr(C)]
#[non_exhaustive]
#[non_exhaustive] //~ ERROR unused attribute
pub struct X;

#[automatically_derived]
#[automatically_derived] //~ ERROR unused attribute
impl X {}

#[inline(always)]
#[inline(never)] //~ ERROR unused attribute
//~^ WARN this was previously accepted
#[cold]
#[cold] //~ ERROR unused attribute
#[track_caller]
#[track_caller] //~ ERROR unused attribute
pub fn xyz() {}

// No warnings for #[link], would require more logic.
#[link(name = "rust_test_helpers", kind = "static")]
#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    #[link_name = "this_does_not_exist"] //~ ERROR unused attribute
    //~^ WARN this was previously accepted
    #[link_name = "rust_dbg_extern_identity_u32"]
    pub fn name_in_rust(v: u32) -> u32;
}

#[export_name = "exported_symbol_name"] //~ ERROR unused attribute
//~^ WARN this was previously accepted
#[export_name = "exported_symbol_name2"]
pub fn export_test() {}

#[no_mangle]
#[no_mangle] //~ ERROR unused attribute
pub fn no_mangle_test() {}

#[used]
#[used] //~ ERROR unused attribute
static FOO: u32 = 0;

#[link_section = ".text"]
//~^ ERROR unused attribute
//~| WARN this was previously accepted
#[link_section = ".bss"]
pub extern "C" fn example() {}

fn main() {}
