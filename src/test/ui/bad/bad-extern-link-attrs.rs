#![feature(link_cfg)]

#[link()] //~ ERROR: specified without `name =
#[link(name = "")] //~ ERROR: with empty name
#[link(name = "foo")]
#[link(name = "foo", kind = "bar")] //~ ERROR: unknown kind
#[link] //~ ERROR #[link(...)] specified without arguments
#[link = "foo"] //~ ERROR #[link(...)] specified without arguments
#[link(name = "foo", name = "bar")] //~ ERROR #[link(...)] contains repeated `name` arguments
#[link(name = "foo", kind = "dylib", kind = "dylib")]
//~^ ERROR #[link(...)] contains repeated `kind` arguments
#[link(name = "foo", cfg(foo), cfg(bar))]
//~^ ERROR #[link(...)] contains repeated `cfg` arguments
#[link(wasm_import_module = "foo", wasm_import_module = "bar")]
//~^ ERROR #[link(...)] contains repeated `wasm_import_module` arguments
extern {}

fn main() {}
