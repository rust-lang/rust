// Proc macro helper for issue #89699, used by tests/ui/proc-macro/edition-gated-async-move-
// syntax-issue89699.rs, emitting an `async move` closure. This syntax is only available in
// editions 2018 and up, but is used in edition 2015 in the test.

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_attribute]
pub fn foo(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let tt = item.into_iter().next().unwrap();
    let sp = tt.span();
    let mut arg = TokenStream::new();
    let mut g = Group::new(Delimiter::Brace, TokenStream::new());
    g.set_span(sp);
    arg.extend([
        TokenTree::Ident(Ident::new("async", sp)),
        TokenTree::Ident(Ident::new("move", sp)),
        TokenTree::Group(g),
    ]);
    let mut body = TokenStream::new();
    body.extend([
        TokenTree::Ident(Ident::new("async_main", sp)),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, arg)),
    ]);

    let mut ret = TokenStream::new();
    ret.extend([
        TokenTree::Ident(Ident::new("fn", sp)),
        TokenTree::Ident(Ident::new("main", sp)),
        TokenTree::Group(Group::new(Delimiter::Parenthesis, TokenStream::new())),
        TokenTree::Group(Group::new(Delimiter::Brace, body)),
    ]);
    ret
}
