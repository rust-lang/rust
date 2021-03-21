// force-host
// no-prefer-dynamic

#![feature(proc_macro_def_site)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_quote)]
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn resolve_located_at(input: TokenStream) -> TokenStream {
    match &*input.into_iter().collect::<Vec<_>>() {
        [a, b, ..] => {
            // The error is reported at input `a`.
            let diag = Diagnostic::error("expected error")
                .mark(Span::def_site().located_at(a.span()))
                .emit();

            // Resolves to `struct S;` at def site, but the error is reported at input `b`.
            let s = TokenTree::Ident(Ident::new("S", b.span().resolved_at(Span::def_site())));
            quote!({
                struct S;

                $s
            })
        }
        _ => panic!("unexpected input"),
    }
}
