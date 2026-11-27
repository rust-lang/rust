#![crate_type = "proc-macro"]

use proc_macro::TokenStream;

extern crate dep as _;
extern crate proc_macro;

#[proc_macro]
pub fn mymacro(input: TokenStream) -> TokenStream {
    extern "C" {
        static VERY_IMPORTANT_SYMBOL: u32;
    }

    // read the symbol otherwise the _linker_ may discard it
    // `#[used]` only preserves the symbol up to the compiler output (.o file)
    // which is then passed to the linker. the linker is free to discard the
    // `#[used]` symbol if it's not accessed/referred-to by other object
    // files (crates)
    let symbol_value = unsafe { core::ptr::addr_of!(VERY_IMPORTANT_SYMBOL).read_volatile() };

    // the exact logic here is unimportant for the rmake check
    let is_that_version = symbol_value == 12345;
    if is_that_version { input } else { panic!() }
}
