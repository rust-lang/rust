#![feature(patchable_function_entry)]
fn main() {}

#[patchable_function_entry(prefix_nops = 256, entry_nops = 0)]//~error: Expected integer value between 0 and 255.
pub fn too_high_pnops() {}

#[patchable_function_entry(prefix_nops = "stringvalue", entry_nops = 0)]//~error: Expected integer value between 0 and 255.
pub fn non_int_nop() {}

#[patchable_function_entry]//~error: malformed `patchable_function_entry` attribute input
pub fn malformed_attribute() {}

#[patchable_function_entry(prefix_nops = 10, something = 0)]//~error: Unexpected parameter name. Allowed names: prefix_nops, entry_nops
pub fn unexpected_parameter_name() {}

#[patchable_function_entry()]//~error: Must specify at least one parameter.
pub fn no_parameters_given() {}
