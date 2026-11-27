#![feature(link_cfg)]

#[link(name = "...", wasm_import_module)] //~ ERROR: malformed `link` attribute input
extern "C" {}

#[link(name = "...", wasm_import_module(x))] //~ ERROR: malformed `link` attribute input
extern "C" {}

#[link(name = "...", wasm_import_module())] //~ ERROR: malformed `link` attribute input
extern "C" {}

#[link(wasm_import_module = "foo", name = "bar")] //~ ERROR: `wasm_import_module` is incompatible with other arguments
extern "C" {}

#[link(wasm_import_module = "foo", kind = "dylib")] //~ ERROR: `wasm_import_module` is incompatible with other arguments
extern "C" {}

#[link(wasm_import_module = "foo", cfg(false))] //~ ERROR: `wasm_import_module` is incompatible with other arguments
extern "C" {}

fn main() {}
