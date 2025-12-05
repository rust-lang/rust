#![no_std]

#[link(wasm_import_module = "test")]
unsafe extern "C" {
    #[link_name = "close"]
    pub fn close(x: u32) -> u32;
}
