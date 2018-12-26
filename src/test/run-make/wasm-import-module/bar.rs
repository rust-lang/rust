#![crate_type = "cdylib"]
#![deny(warnings)]

extern crate foo;

#[link(wasm_import_module = "./me")]
extern {
    #[link_name = "me_in_dep"]
    fn dep();
}

#[no_mangle]
pub extern fn foo() {
    unsafe {
        foo::dep();
        dep();
    }
}
