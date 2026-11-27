// Proc macros commonly used by tests.
// `panic`/`print` -> `panic_bang`/`print_bang` to avoid conflicts with standard macros.

extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree};

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
    print_helper_ext(input, kind, true)
}

fn deep_recollect(input: TokenStream) -> TokenStream {
    input.into_iter().map(|tree| {
        match tree {
            TokenTree::Group(group) => {
                let inner = deep_recollect(group.stream());
                let mut new_group = TokenTree::Group(
                    proc_macro::Group::new(group.delimiter(), inner)
                );
                new_group.set_span(group.span());
                new_group
            }
            _ => tree,
        }
    }).collect()
}

fn print_helper_ext(input: TokenStream, kind: &str, debug: bool) -> TokenStream {
    let input_display = format!("{}", input);
    let input_debug = format!("{:#?}", input);
    let recollected = input.clone().into_iter().collect();
    let recollected_display = format!("{}", recollected);
    let recollected_debug = format!("{:#?}", recollected);

    let deep_recollected = deep_recollect(input);
    let deep_recollected_display = format!("{}", deep_recollected);
    let deep_recollected_debug = format!("{:#?}", deep_recollected);



    println!("PRINT-{} INPUT (DISPLAY): {}", kind, input_display);
    if recollected_display != input_display {
        println!("PRINT-{} RE-COLLECTED (DISPLAY): {}", kind, recollected_display);
    }

    if deep_recollected_display != recollected_display {
        println!("PRINT-{} DEEP-RE-COLLECTED (DISPLAY): {}", kind, deep_recollected_display);
    }

    if debug {
        println!("PRINT-{} INPUT (DEBUG): {}", kind, input_debug);
        if recollected_debug != input_debug {
            println!("PRINT-{} RE-COLLECTED (DEBUG): {}", kind, recollected_debug);
        }
        if deep_recollected_debug != recollected_debug {
            println!("PRINT-{} DEEP-RE-COLLETED (DEBUG): {}", kind, deep_recollected_debug);
        }
    }
    recollected
}

#[proc_macro]
pub fn print_bang(input: TokenStream) -> TokenStream {
    print_helper(input, "BANG")
}

#[proc_macro]
pub fn print_bang_consume(input: TokenStream) -> TokenStream {
    print_helper(input, "BANG");
    TokenStream::new()
}

#[proc_macro_attribute]
pub fn print_attr(args: TokenStream, input: TokenStream) -> TokenStream {
    let debug = match &args.into_iter().collect::<Vec<_>>()[..] {
        [TokenTree::Ident(ident)] if ident.to_string() == "nodebug" => false,
        _ => true,
    };
    print_helper_ext(input, "ATTR", debug)
}

#[proc_macro_attribute]
pub fn print_attr_args(args: TokenStream, input: TokenStream) -> TokenStream {
    print_helper(args, "ATTR_ARGS");
    input
}

#[proc_macro_attribute]
pub fn print_target_and_args(args: TokenStream, input: TokenStream) -> TokenStream {
    print_helper(args, "ATTR_ARGS");
    print_helper(input.clone(), "ATTR");
    input
}

#[proc_macro_attribute]
pub fn print_target_and_args_consume(args: TokenStream, input: TokenStream) -> TokenStream {
    print_helper(args, "ATTR_ARGS");
    print_helper(input.clone(), "ATTR");
    TokenStream::new()
}

#[proc_macro_derive(Print, attributes(print_helper))]
pub fn print_derive(input: TokenStream) -> TokenStream {
    print_helper(input, "DERIVE");
    TokenStream::new()
}
