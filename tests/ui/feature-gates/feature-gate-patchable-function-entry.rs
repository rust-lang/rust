#[patchable_function_entry(prefix_nops = 1, entry_nops = 1)]
//~^ ERROR: the `#[patchable_function_entry]` attribute is an experimental feature
fn main() {}
