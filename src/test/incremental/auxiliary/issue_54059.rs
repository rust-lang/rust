// force-host
// no-prefer-dynamic

// check that having extern "C" functions in a proc macro doesn't crash.

#![crate_type="proc-macro"]
#![allow(non_snake_case)]

extern crate proc_macro;

macro_rules! proc_macro_tokenstream {
    () => {
        ::proc_macro::TokenStream
    };
}

macro_rules! proc_macro_expr_impl {
    ($(
        $( #[$attr:meta] )*
        pub fn $func:ident($input:ident: &str) -> String $body:block
    )+) => {
        $(
            // Parses an input that looks like:
            //
            // ```
            // #[allow(unused)]
            // enum ProcMacroHack {
            //     Input = (stringify!(ARGS), 0).1,
            // }
            // ```
            $( #[$attr] )*
            #[proc_macro_derive($func)]
            pub fn $func(input: proc_macro_tokenstream!()) -> proc_macro_tokenstream!() {
                unsafe { rust_dbg_extern_identity_u64(0); }
                panic!()
            }
        )+
    };
}

proc_macro_expr_impl! {
    pub fn base2_impl(input: &str) -> String {
        panic!()
    }
}

#[link(name="rust_test_helpers")]
extern "C" {
    pub fn rust_dbg_extern_identity_u64(v: u64) -> u64;
}
