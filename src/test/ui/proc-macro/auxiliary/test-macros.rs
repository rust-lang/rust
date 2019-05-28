// force-host
// no-prefer-dynamic

// Proc macros commonly used by tests.
// `panic`/`print` -> `panic_bang`/`print_bang` to avoid conflicts with standard macros.

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

// Macro that return empty token stream.

#[proc_macro]
pub fn empty(_: TokenStream) -> TokenStream {
    TokenStream::new()
}

#[proc_macro_attribute]
pub fn empty_attr(_: TokenStream, _: TokenStream) -> TokenStream {
    TokenStream::new()
}

#[proc_macro_derive(Empty, attributes(empty_helper))]
pub fn empty_derive(_: TokenStream) -> TokenStream {
    TokenStream::new()
}

// Macro that panics.

#[proc_macro]
pub fn panic_bang(_: TokenStream) -> TokenStream {
    panic!("panic-bang");
}

#[proc_macro_attribute]
pub fn panic_attr(_: TokenStream, _: TokenStream) -> TokenStream {
    panic!("panic-attr");
}

#[proc_macro_derive(Panic, attributes(panic_helper))]
pub fn panic_derive(_: TokenStream) -> TokenStream {
    panic!("panic-derive");
}

// Macros that return the input stream.

#[proc_macro]
pub fn identity(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn identity_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(Identity, attributes(identity_helper))]
pub fn identity_derive(input: TokenStream) -> TokenStream {
    input
}

// Macros that iterate and re-collect the input stream.

#[proc_macro]
pub fn recollect(input: TokenStream) -> TokenStream {
    input.into_iter().collect()
}

#[proc_macro_attribute]
pub fn recollect_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    input.into_iter().collect()
}

#[proc_macro_derive(Recollect, attributes(recollect_helper))]
pub fn recollect_derive(input: TokenStream) -> TokenStream {
    input.into_iter().collect()
}

// Macros that print their input in the original and re-collected forms (if they differ).

fn print_helper(input: TokenStream, kind: &str) -> TokenStream {
    let input_display = format!("{}", input);
    let input_debug = format!("{:#?}", input);
    let recollected = input.into_iter().collect();
    let recollected_display = format!("{}", recollected);
    let recollected_debug = format!("{:#?}", recollected);
    println!("PRINT-{} INPUT (DISPLAY): {}", kind, input_display);
    if recollected_display != input_display {
        println!("PRINT-{} RE-COLLECTED (DISPLAY): {}", kind, recollected_display);
    }
    println!("PRINT-{} INPUT (DEBUG): {}", kind, input_debug);
    if recollected_debug != input_debug {
        println!("PRINT-{} RE-COLLECTED (DEBUG): {}", kind, recollected_debug);
    }
    recollected
}

#[proc_macro]
pub fn print_bang(input: TokenStream) -> TokenStream {
    print_helper(input, "BANG")
}

#[proc_macro_attribute]
pub fn print_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    print_helper(input, "ATTR")
}

#[proc_macro_derive(Print, attributes(print_helper))]
pub fn print_derive(input: TokenStream) -> TokenStream {
    print_helper(input, "DERIVE")
}
