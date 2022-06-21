extern "C" {
    fn exported_symbol() -> i8;
}

#[no_mangle]
pub extern "C" fn call_exported_symbol() -> i8 {
    unsafe { exported_symbol() }
}
