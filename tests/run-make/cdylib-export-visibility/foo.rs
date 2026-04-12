#![crate_type = "cdylib"]
#![feature(export_visibility)]
// `no_std` makes it slightly easier to run the test when cross-compiling.
// Ideally the test would use `no_core`, but `//@ add-minicore` doesn't seem
// to work...
#![no_std]
//@ edition: 2024

#[unsafe(no_mangle)]
unsafe extern "C" fn test_fn_no_export_visibility_attribute() -> u32 {
    // Using unique integer means that the functions return different results
    // and therefore identical code folding (ICF) in the linker won't apply.
    16 // line number;  can't use `line!` with `no_core`
}

#[unsafe(no_mangle)]
#[export_visibility = "target_default"]
unsafe extern "C" fn test_fn_export_visibility_asks_for_target_default() -> u32 {
    // Using unique integer means that the functions return different results
    // and therefore identical code folding (ICF) in the linker won't apply.
    24 // line number;  can't use `line!` with `no_core`
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // The infinite loop should never run - we only look at symbol visibilities
    // of `test_fn_...` above.
    loop {}
}
