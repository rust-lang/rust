#![feature(link_cfg)]

#[link(name = "...", wasm_import_module)] //~ ERROR: must be of the form
extern "C" {}

#[link(name = "...", wasm_import_module(x))] //~ ERROR: must be of the form
extern "C" {}

#[link(name = "...", wasm_import_module())] //~ ERROR: must be of the form
extern "C" {}

#[link(wasm_import_module = "foo", name = "bar")] //~ ERROR: `wasm_import_module` is incompatible with other arguments
extern "C" {}

#[link(wasm_import_module = "foo", kind = "dylib")] //~ ERROR: `wasm_import_module` is incompatible with other arguments
extern "C" {}

#[link(wasm_import_module = "foo", cfg(false))] //~ ERROR: `wasm_import_module` is incompatible with other arguments
extern "C" {}

fn main() {}
