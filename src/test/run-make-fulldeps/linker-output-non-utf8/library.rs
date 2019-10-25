#![crate_type = "staticlib"]

extern "C" {
    fn this_symbol_not_defined();
}

#[no_mangle]
pub extern "C" fn foo() {
    unsafe { this_symbol_not_defined(); }
}
