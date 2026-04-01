#![crate_type = "cdylib"]

mod a {
    #[link(wasm_import_module = "a")]
    extern "C" {
        pub fn foo();
    }
}

mod b {
    #[link(wasm_import_module = "b")]
    extern "C" {
        pub fn foo();
    }
}

#[no_mangle]
pub fn start() {
    unsafe {
        a::foo();
        b::foo();
    }
}
