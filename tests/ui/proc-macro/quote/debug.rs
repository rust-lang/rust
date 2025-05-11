//@ check-pass
//@ force-host
//@ no-prefer-dynamic
//@ compile-flags: -Z unpretty=expanded
//@ needs-unwind compiling proc macros with panic=abort causes a warning
//@ edition: 2015
//
// This file is not actually used as a proc-macro - instead,
// it's just used to show the output of the `quote!` macro

#![feature(proc_macro_quote)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

fn main() {
    proc_macro::quote! {
        let hello = "world";
        let r#raw_ident = r#"raw"literal"#;
    }
}
