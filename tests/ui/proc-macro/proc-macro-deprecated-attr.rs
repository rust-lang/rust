// check-pass
// force-host
// no-prefer-dynamic

#![deny(deprecated)]

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
#[deprecated(since = "1.0.0", note = "test")]
pub fn test_compile_without_warning_with_deprecated(_: TokenStream) -> TokenStream {
    TokenStream::new()
}
