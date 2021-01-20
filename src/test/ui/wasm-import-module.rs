#[link(name = "...", wasm_import_module)] //~ ERROR: must be of the form
extern {}

#[link(name = "...", wasm_import_module(x))] //~ ERROR: must be of the form
extern {}

#[link(name = "...", wasm_import_module())] //~ ERROR: must be of the form
extern {}

fn main() {}
