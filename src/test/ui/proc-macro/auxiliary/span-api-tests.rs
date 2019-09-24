// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(proc_macro_span)]

extern crate proc_macro;

use proc_macro::*;

// Re-emits the input tokens by parsing them from strings
#[proc_macro]
pub fn reemit(input: TokenStream) -> TokenStream {
    input.to_string().parse().unwrap()
}

#[proc_macro]
pub fn assert_fake_source_file(input: TokenStream) -> TokenStream {
    for tk in input {
        let source_file = tk.span().source_file();
        assert!(!source_file.is_real(), "Source file is real: {:?}", source_file);
    }

    "".parse().unwrap()
}

#[proc_macro]
pub fn assert_source_file(input: TokenStream) -> TokenStream {
    for tk in input {
        let source_file = tk.span().source_file();
        assert!(source_file.is_real(), "Source file is not real: {:?}", source_file);
    }

    "".parse().unwrap()
}

#[proc_macro]
pub fn macro_stringify(input: TokenStream) -> TokenStream {
    let mut tokens = input.into_iter();
    let first_span = tokens.next().expect("first token").span();
    let last_span = tokens.last().map(|x| x.span()).unwrap_or(first_span);
    let span = first_span.join(last_span).expect("joined span");
    let src = span.source_text().expect("source_text");
    TokenTree::Literal(Literal::string(&src)).into()
}
