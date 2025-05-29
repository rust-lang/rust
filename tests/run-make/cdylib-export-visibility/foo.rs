#![crate_type = "cdylib"]
#![feature(export_visibility)]

#[unsafe(no_mangle)]
unsafe extern "C" fn test_fn_no_attr() -> u32 {
    // Using `line!()` means that the functions return different results
    // and therefore identical code folding (ICF) in the linker won't apply.
    line!()
}

#[unsafe(no_mangle)]
#[export_visibility = "target_default"]
unsafe extern "C" fn test_fn_export_visibility_asks_for_target_default() -> u32 {
    line!()
}
