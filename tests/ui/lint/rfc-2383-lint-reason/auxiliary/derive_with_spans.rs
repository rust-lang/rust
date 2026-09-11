// Derive macros generating an `#[automatically_derived]` impl whose tokens carry
// call-site, mixed-site or def-site spans. Whatever spans (and with them, hygiene
// data) a proc-macro assigns to its output must not make the generated impl inherit
// `#[expect]` expectations from the derive input.

#![feature(proc_macro_def_site)]

extern crate proc_macro;

use proc_macro::{Group, Span, TokenStream, TokenTree};

fn struct_name(input: TokenStream) -> String {
    let mut tokens = input.into_iter();
    while let Some(tt) = tokens.next() {
        if let TokenTree::Ident(id) = &tt {
            if id.to_string() == "struct" {
                if let Some(TokenTree::Ident(name)) = tokens.next() {
                    return name.to_string();
                }
            }
        }
    }
    panic!("no struct in derive input");
}

// Respans everything except identifiers, which keep their default spans so that
// paths in the generated code still resolve at the call site.
fn respan(ts: TokenStream, span: Span) -> TokenStream {
    ts.into_iter()
        .map(|tt| match tt {
            TokenTree::Group(g) => {
                let mut new = Group::new(g.delimiter(), respan(g.stream(), span));
                new.set_span(span);
                TokenTree::Group(new)
            }
            TokenTree::Ident(id) => TokenTree::Ident(id),
            mut tt => {
                tt.set_span(span);
                tt
            }
        })
        .collect()
}

fn derive_impl(input: TokenStream, span: Span) -> TokenStream {
    let name = struct_name(input);
    let source = format!(
        "#[automatically_derived]
        impl Trait for {name} {{
            fn method(&self) {{ let unused_in_derived = 0; }}
        }}"
    );
    respan(source.parse().unwrap(), span)
}

#[proc_macro_derive(WithCallSite)]
pub fn with_call_site(input: TokenStream) -> TokenStream {
    derive_impl(input, Span::call_site())
}

#[proc_macro_derive(WithMixedSite)]
pub fn with_mixed_site(input: TokenStream) -> TokenStream {
    derive_impl(input, Span::mixed_site())
}

#[proc_macro_derive(WithDefSite)]
pub fn with_def_site(input: TokenStream) -> TokenStream {
    derive_impl(input, Span::def_site())
}
