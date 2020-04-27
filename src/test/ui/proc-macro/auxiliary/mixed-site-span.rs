// force-host
// no-prefer-dynamic

#![feature(proc_macro_hygiene)]
#![feature(proc_macro_quote)]

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn proc_macro_rules(input: TokenStream) -> TokenStream {
    if input.is_empty() {
        let id = |s| TokenTree::from(Ident::new(s, Span::mixed_site()));
        let item_def = id("ItemDef");
        let local_def = id("local_def");
        let item_use = id("ItemUse");
        let local_use = id("local_use");
        let mut single_quote = Punct::new('\'', Spacing::Joint);
        single_quote.set_span(Span::mixed_site());
        let label_use: TokenStream = [
            TokenTree::from(single_quote),
            id("label_use"),
        ].iter().cloned().collect();
        quote!(
            struct $item_def;
            let $local_def = 0;

            $item_use; // OK
            $local_use; // ERROR
            break $label_use; // ERROR
        )
    } else {
        let mut dollar_crate = input.into_iter().next().unwrap();
        dollar_crate.set_span(Span::mixed_site());
        quote!(
            type A = $dollar_crate::ItemUse;
        )
    }
}
