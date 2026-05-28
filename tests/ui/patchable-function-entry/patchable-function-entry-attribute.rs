#![feature(patchable_function_entry)]
fn main() {}

#[patchable_function_entry(prefix_nops = 256, entry_nops = 0)]
//~^ ERROR malformed
pub fn too_high_pnops() {}

#[patchable_function_entry(prefix_nops = "stringvalue", entry_nops = 0)]
//~^ ERROR malformed
pub fn non_int_nop() {}

#[patchable_function_entry]
//~^ ERROR malformed `patchable_function_entry` attribute input
pub fn malformed_attribute() {}

#[patchable_function_entry(prefix_nops = 10, something = 0)]
//~^ ERROR malformed
pub fn unexpected_parameter_name() {}

#[patchable_function_entry()]
//~^ ERROR malformed
pub fn no_parameters_given() {}

#[patchable_function_entry(prefix_nops = 255, prefix_nops = 255)]
//~^ ERROR malformed
pub fn duplicate_parameter() {}
