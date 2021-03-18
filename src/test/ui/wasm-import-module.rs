#[link(name = "...", wasm_import_module)] //~ ERROR: must be of the form
extern "C" {}

#[link(name = "...", wasm_import_module(x))] //~ ERROR: must be of the form
extern "C" {}

#[link(name = "...", wasm_import_module())] //~ ERROR: must be of the form
extern "C" {}

fn main() {}
