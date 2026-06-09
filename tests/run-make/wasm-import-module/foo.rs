#![crate_type = "rlib"]
#![deny(warnings)]

#[link(wasm_import_module = "./dep")]
extern "C" {
    pub fn dep();
}
