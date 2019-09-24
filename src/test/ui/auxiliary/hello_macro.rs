// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_hygiene, proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{TokenStream, quote};

// This macro is not very interesting, but it does contain delimited tokens with
// no content - `()` and `{}` - which has caused problems in the past.
// Also, it tests that we can escape `$` via `$$`.
#[proc_macro]
pub fn hello(_: TokenStream) -> TokenStream {
    quote!({
        fn hello() {}
        macro_rules! m { ($$($$t:tt)*) => { $$($$t)* } }
        m!(hello());
    })
}
