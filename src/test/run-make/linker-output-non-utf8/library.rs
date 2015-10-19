#![crate_type = "dylib"]

extern "C" {
    fn this_symbol_not_defined();
}

pub extern "C" fn foo() {
    unsafe { this_symbol_not_defined(); }
}
