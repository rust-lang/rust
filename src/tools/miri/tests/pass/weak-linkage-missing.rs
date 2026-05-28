#![feature(linkage)]

unsafe extern "C" {
    #[linkage = "extern_weak"]
    static test_symbol_that_does_not_exist: Option<unsafe extern "C" fn()>;
}

fn main() {
    unsafe { assert!(test_symbol_that_does_not_exist.is_none()) };
}
