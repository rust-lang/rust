#![feature(patchable_function_entry)]
fn main() {}

#[patchable_function_entry(prefix_nops = 256, entry_nops = 0)]//~error: integer value out of range
pub fn too_high_pnops() {}

#[patchable_function_entry(prefix_nops = "stringvalue", entry_nops = 0)]//~error: invalid literal value
pub fn non_int_nop() {}

#[patchable_function_entry]//~error: malformed `patchable_function_entry` attribute input
pub fn malformed_attribute() {}

#[patchable_function_entry(prefix_nops = 10, something = 0)]//~error: unexpected parameter name
pub fn unexpected_parameter_name() {}

#[patchable_function_entry()]//~error: must specify at least one parameter
pub fn no_parameters_given() {}
