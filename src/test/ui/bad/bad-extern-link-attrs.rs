#![deny(invalid_link_arguments)]

#![feature(link_cfg)]

#[link()] //~ ERROR: specified without `name =
#[link(name = "")] //~ ERROR: with empty name
#[link(name = "foo")]
#[link(name = "foo", kind = "bar")] //~ ERROR: unknown kind
#[link] //~ ERROR #[link(...)] requires an argument list
        //~^ this was previously accepted by the compiler
#[link = "foo"] //~ ERROR #[link(...)] requires an argument list
                //~^ this was previously accepted by the compiler
#[link(name = "foo", name = "bar")] //~ ERROR #[link(...)] should not contain repeated arguments
                                    //~^ this was previously accepted by the compiler
#[link(name = "foo", kind = "dylib", kind = "dylib")]
//~^ ERROR #[link(...)] should not contain repeated arguments
//~^^ this was previously accepted by the compiler
#[link(name = "foo", cfg(foo), cfg(bar))]
//~^ ERROR #[link(...)] should not contain repeated arguments
//~^^ this was previously accepted by the compiler
#[link(wasm_import_module = "foo", wasm_import_module = "bar")]
//~^ ERROR #[link(...)] should not contain repeated arguments
//~^^ this was previously accepted by the compiler
extern {}

fn main() {}
