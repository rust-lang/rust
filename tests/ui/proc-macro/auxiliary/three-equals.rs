// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_diagnostic, proc_macro_span, proc_macro_def_site)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Span, Diagnostic};

fn parse(input: TokenStream) -> Result<(), Diagnostic> {
    let mut count = 0;
    let mut last_span = Span::def_site();
    for tree in input {
        let span = tree.span();
        if count >= 3 {
            return Err(span.error(format!("expected EOF, found `{}`.", tree))
                           .span_note(last_span, "last good input was here")
                           .help("input must be: `===`"))
        }

        if let TokenTree::Punct(ref tt) = tree {
            if tt.as_char() == '=' {
                count += 1;
                last_span = span;
                continue
            }
        }
        return Err(span.error(format!("expected `=`, found `{}`.", tree)));
    }

    if count < 3 {
        return Err(Span::def_site()
                       .error(format!("found {} equal signs, need exactly 3", count))
                       .help("input must be: `===`"))
    }

    Ok(())
}

#[proc_macro]
pub fn three_equals(input: TokenStream) -> TokenStream {
    if let Err(diag) = parse(input) {
        diag.emit();
        return TokenStream::new();
    }

    "3".parse().unwrap()
}
