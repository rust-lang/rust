// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn bar(_input: TokenStream) -> TokenStream {
    let mut ret = Vec::<TokenTree>::new();
    ret.push(Ident::new("static", Span::call_site()).into());
    ret.push(Ident::new("FOO", Span::call_site()).into());
    ret.push(Punct::new(':', Spacing::Alone).into());
    ret.push(Punct::new('&', Spacing::Alone).into());
    ret.push(Punct::new('\'', Spacing::Joint).into());
    ret.push(Ident::new("static", Span::call_site()).into());
    ret.push(Ident::new("i32", Span::call_site()).into());
    ret.push(Punct::new('=', Spacing::Alone).into());
    ret.push(Punct::new('&', Spacing::Alone).into());
    ret.push(Literal::i32_unsuffixed(1).into());
    ret.push(Punct::new(';', Spacing::Alone).into());
    ret.into_iter().collect()
}
