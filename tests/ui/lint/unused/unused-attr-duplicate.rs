// Tests for repeating attribute warnings.
//@ aux-build:lint_unused_extern_crate.rs
//@ compile-flags:--test -W repeated-reprs
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
#![crate_name = "unused_attr_duplicate2"]
//~^ ERROR duplicate `crate_name` attributes
//~| WARN this was previously accepted
//~| ERROR duplicate `crate_name` attributes
//~| WARN this was previously accepted
// FIXME(jdonszelmann) this error is given twice now. I'll look at this in the future
#![recursion_limit = "128"]
#![recursion_limit = "256"]
//~^ ERROR duplicate `recursion_limit` attributes
//~| WARN this was previously accepted
#![type_length_limit = "1048576"]
#![type_length_limit = "1"]
//~^ ERROR duplicate `type_length_limit` attributes
//~| WARN this was previously accepted
#![no_std]
#![no_std] //~ ERROR duplicate `no_std` attributes
#![no_implicit_prelude]
#![no_implicit_prelude] //~ ERROR duplicate `no_implicit_prelude` attributes
#![windows_subsystem = "console"]
#![windows_subsystem = "windows"]
//~^ ERROR duplicate `windows_subsystem` attributes
//~| WARN this was previously accepted
#![no_builtins]
#![no_builtins] //~ ERROR duplicate `no_builtins` attributes

#[no_link]
//~^ ERROR use of deprecated `no_link` attribute
//~| WARN this was previously accepted by the compiler
#[no_link] //~ ERROR duplicate `no_link` attributes
extern crate lint_unused_extern_crate;

#[macro_use]
#[macro_use] //~ ERROR duplicate `macro_use` attributes
pub mod m {
    #[macro_export]
    #[macro_export] //~ ERROR duplicate `macro_export` attributes
    macro_rules! foo {
        () => {};
    }
}

#[path = "auxiliary/lint_unused_extern_crate.rs"]
#[path = "bar.rs"] //~ ERROR duplicate `path` attributes
//~^ WARN this was previously accepted
pub mod from_path;

#[test]
#[ignore]
#[ignore = "some text"]
//~^ ERROR duplicate `ignore` attributes
#[should_panic]
#[should_panic(expected = "values don't match")]
//~^ ERROR duplicate `should_panic` attributes
//~| WARN this was previously accepted
fn t1() {}

#[must_use]
#[must_use = "some message"]
//~^ ERROR duplicate `must_use` attributes
//~| WARN this was previously accepted
#[repr(C)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
#[repr(C)]
#[non_exhaustive]
#[non_exhaustive] //~ ERROR duplicate `non_exhaustive` attributes
pub struct X;

trait Trait {}

#[automatically_derived]
#[automatically_derived]
//~^ ERROR duplicate `automatically_derived` attributes
impl Trait for X {}

#[inline(always)]
#[inline(never)] //~ ERROR duplicate `inline` attributes
//~^ WARN this was previously accepted
#[cold]
#[cold] //~ ERROR duplicate `cold` attributes
#[track_caller]
#[track_caller] //~ ERROR duplicate `track_caller` attributes
pub fn xyz() {}

// No warnings for #[link], would require more logic.
#[link(name = "rust_test_helpers", kind = "static")]
#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    #[link_name = "this_does_not_exist"]
    #[link_name = "rust_dbg_extern_identity_u32"]
    //~^ ERROR duplicate `link_name` attributes
    //~| WARN this was previously accepted
    pub fn name_in_rust(v: u32) -> u32;
}

#[export_name = "exported_symbol_name"]
#[export_name = "exported_symbol_name2"]
//~^ ERROR duplicate `export_name` attributes
//~| WARN this was previously accepted
pub fn export_test() {}

#[no_mangle]
#[no_mangle] //~ ERROR duplicate `no_mangle` attributes
pub fn no_mangle_test() {}

#[used]
#[used] //~ ERROR duplicate `used` attributes
static FOO: u32 = 0;

#[link_section = "__TEXT,__text"]
#[link_section = "__DATA,__mod_init_func"]
//~^ ERROR duplicate `link_section` attributes
//~| WARN this was previously accepted
pub extern "C" fn example() {}

fn main() {}
