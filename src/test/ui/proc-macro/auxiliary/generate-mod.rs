// run-pass
// force-host
// no-prefer-dynamic
// ignore-pass

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn check(_: TokenStream) -> TokenStream {
    "
    type Alias = FromOutside; // OK
    struct Outer;
    mod inner {
        type Alias = FromOutside; // `FromOutside` shouldn't be available from here
        type Inner = Outer; // `Outer` shouldn't be available from here
    }
    ".parse().unwrap()
}

#[proc_macro_attribute]
pub fn check_attr(_: TokenStream, _: TokenStream) -> TokenStream {
    "
    type AliasAttr = FromOutside; // OK
    struct OuterAttr;
    mod inner_attr {
        type Alias = FromOutside; // `FromOutside` shouldn't be available from here
        type Inner = OuterAttr; // `OuterAttr` shouldn't be available from here
    }
    ".parse().unwrap()
}

#[proc_macro_derive(CheckDerive)]
pub fn check_derive(_: TokenStream) -> TokenStream {
    "
    type AliasDerive = FromOutside; // OK
    struct OuterDerive;
    mod inner_derive {
        type Alias = FromOutside; // `FromOutside` shouldn't be available from here
        type Inner = OuterDerive; // `OuterDerive` shouldn't be available from here
    }
    ".parse().unwrap()
}

#[proc_macro_derive(CheckDeriveLint)]
pub fn check_derive_lint(_: TokenStream) -> TokenStream {
    "
    type AliasDeriveLint = FromOutside; // OK
    struct OuterDeriveLint;
    #[allow(proc_macro_derive_resolution_fallback)]
    mod inner_derive_lint {
        type Alias = FromOutside; // `FromOutside` shouldn't be available from here
        type Inner = OuterDeriveLint; // `OuterDeriveLint` shouldn't be available from here
    }
    ".parse().unwrap()
}
