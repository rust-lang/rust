// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

struct Zeroable;

#[proc_macro_derive(Zeroable)]
pub fn derive_zeroable(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
  proc_macro::TokenStream::default()
}
