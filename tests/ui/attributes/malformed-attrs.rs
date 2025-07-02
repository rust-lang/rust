// This file contains a bunch of malformed attributes.
// We enable a bunch of features to not get feature-gate errs in this test.
#![feature(rustc_attrs)]
#![feature(rustc_allow_const_fn_unstable)]
#![feature(allow_internal_unstable)]
#![feature(fn_align)]
#![feature(optimize_attribute)]
#![feature(dropck_eyepatch)]
#![feature(export_stable)]
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]
#![feature(ffi_const, ffi_pure)]
#![feature(coverage_attribute)]
#![feature(no_sanitize)]
#![feature(marker_trait_attr)]
#![feature(thread_local)]
#![feature(must_not_suspend)]
#![feature(coroutines)]
#![feature(linkage)]
#![feature(cfi_encoding, extern_types)]
#![feature(patchable_function_entry)]
#![feature(omit_gdb_pretty_printer_section)]
#![feature(fundamental)]


#![omit_gdb_pretty_printer_section = 1]
//~^ ERROR malformed `omit_gdb_pretty_printer_section` attribute input

#![windows_subsystem]
//~^ ERROR malformed

#[unsafe(export_name)]
//~^ ERROR malformed
#[rustc_allow_const_fn_unstable]
//~^ ERROR `rustc_allow_const_fn_unstable` expects a list of feature names
#[allow_internal_unstable]
//~^ ERROR `allow_internal_unstable` expects a list of feature names
#[rustc_confusables]
//~^ ERROR malformed
#[deprecated = 5]
//~^ ERROR malformed
#[doc]
//~^ ERROR valid forms for the attribute are
//~| WARN this was previously accepted by the compiler
#[rustc_macro_transparency]
//~^ ERROR malformed
#[repr]
//~^ ERROR malformed
#[rustc_as_ptr = 5]
//~^ ERROR malformed
#[inline = 5]
//~^ ERROR valid forms for the attribute are
//~| WARN this was previously accepted by the compiler
#[align]
//~^ ERROR malformed
#[optimize]
//~^ ERROR malformed
#[cold = 1]
//~^ ERROR malformed
#[must_use()]
//~^ ERROR valid forms for the attribute are
#[no_mangle = 1]
//~^ ERROR malformed
#[unsafe(naked())]
//~^ ERROR malformed
#[track_caller()]
//~^ ERROR malformed
#[export_name()]
//~^ ERROR malformed
#[used()]
//~^ ERROR malformed
#[crate_name]
//~^ ERROR malformed
#[doc]
//~^ ERROR valid forms for the attribute are
//~| WARN this was previously accepted by the compiler
#[target_feature]
//~^ ERROR malformed
#[export_stable = 1]
//~^ ERROR malformed
#[link]
//~^ ERROR attribute must be of the form
//~| WARN this was previously accepted by the compiler
#[link_name]
//~^ ERROR malformed
#[link_section]
//~^ ERROR malformed
#[coverage]
//~^ ERROR malformed `coverage` attribute input
#[no_sanitize]
//~^ ERROR malformed
#[ignore()]
//~^ ERROR valid forms for the attribute are
//~| WARN this was previously accepted by the compiler
#[no_implicit_prelude = 23]
//~^ ERROR malformed
#[proc_macro = 18]
//~^ ERROR malformed
//~| ERROR the `#[proc_macro]` attribute is only usable with crates of the `proc-macro` crate type
#[cfg]
//~^ ERROR is not followed by parentheses
#[cfg_attr]
//~^ ERROR malformed
#[instruction_set]
//~^ ERROR malformed
#[patchable_function_entry]
//~^ ERROR malformed
fn test() {
    #[coroutine = 63] || {}
    //~^ ERROR malformed `coroutine` attribute input
    //~| ERROR mismatched types [E0308]
}

#[proc_macro_attribute = 19]
//~^ ERROR malformed
//~| ERROR the `#[proc_macro_attribute]` attribute is only usable with crates of the `proc-macro` crate type
#[must_use = 1]
//~^ ERROR malformed
fn test2() { }

#[proc_macro_derive]
//~^ ERROR malformed `proc_macro_derive` attribute
//~| ERROR the `#[proc_macro_derive]` attribute is only usable with crates of the `proc-macro` crate type
pub fn test3() {}

#[rustc_layout_scalar_valid_range_start]
//~^ ERROR malformed
#[rustc_layout_scalar_valid_range_end]
//~^ ERROR malformed
#[must_not_suspend()]
//~^ ERROR malformed
#[cfi_encoding]
//~^ ERROR malformed
struct Test;

#[diagnostic::on_unimplemented]
//~^ WARN missing options for `on_unimplemented` attribute
#[diagnostic::on_unimplemented = 1]
//~^ WARN malformed
trait Hey {
    #[type_const = 1]
    //~^ ERROR malformed
    const HEY: usize = 5;
}

struct Empty;
#[diagnostic::do_not_recommend()]
//~^ WARN does not expect any arguments
impl Hey for Empty {

}

#[marker = 3]
//~^ ERROR malformed
#[fundamental()]
//~^ ERROR malformed
trait EmptyTrait {

}


extern "C" {
    #[unsafe(ffi_pure = 1)]
    //~^ ERROR malformed
    #[link_ordinal]
    //~^ ERROR malformed
    pub fn baz();

    #[unsafe(ffi_const = 1)]
    //~^ ERROR malformed
    #[linkage]
    //~^ ERROR malformed
    pub fn bar();
}

#[allow]
//~^ ERROR malformed
#[expect]
//~^ ERROR malformed
#[warn]
//~^ ERROR malformed
#[deny]
//~^ ERROR malformed
#[forbid]
//~^ ERROR malformed
#[debugger_visualizer]
//~^ ERROR invalid argument
//~| ERROR malformed `debugger_visualizer` attribute input
#[automatically_derived = 18]
//~^ ERROR malformed
mod yooo {

}

#[non_exhaustive = 1]
//~^ ERROR malformed
enum Slenum {

}

#[thread_local()]
//~^ ERROR malformed
static mut TLS: u8 = 42;

#[no_link()]
//~^ ERROR malformed
#[macro_use = 1]
//~^ ERROR malformed
extern crate wloop;
//~^ ERROR can't find crate for `wloop` [E0463]

#[macro_export = 18]
//~^ ERROR malformed `macro_export` attribute input
#[allow_internal_unsafe = 1]
//~^ ERROR malformed
//~| ERROR allow_internal_unsafe side-steps the unsafe_code lint
macro_rules! slump {
    () => {}
}

fn main() {}
