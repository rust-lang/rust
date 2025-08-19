#![feature(proc_macro_quote)]

extern crate proc_macro;
use proc_macro::*;


#[proc_macro]
pub fn proc_macro_item(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro]
pub fn proc_macro_rules(_input: TokenStream) -> TokenStream {
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
    let dollar_crate = id("$crate");
    quote!(
        use $dollar_crate::proc_macro_item as _; // OK
        type A = $dollar_crate::ItemUse; // ERROR

        struct $item_def;
        let $local_def = 0;

        $item_use; // OK
        $local_use; // ERROR
        break $label_use; // ERROR
    )
}

#[proc_macro]
pub fn with_crate(input: TokenStream) -> TokenStream {
    let mut input = input.into_iter();
    let TokenTree::Ident(mut krate) = input.next().unwrap() else { panic!("missing $crate") };
    let TokenTree::Ident(span) = input.next().unwrap() else { panic!("missing span") };
    let TokenTree::Ident(ident) = input.next().unwrap() else { panic!("missing ident") };

    match (krate.to_string().as_str(), span.to_string().as_str()) {
        ("$crate", "input") => {},
        (_, "input") => krate = Ident::new("$crate", krate.span()),

        ("$crate", "mixed") => krate.set_span(Span::mixed_site()),
        (_, "mixed") => krate = Ident::new("$crate", Span::mixed_site()),

        ("$crate", "call") => krate.set_span(Span::call_site()),
        (_, "call") => krate = Ident::new("$crate", Span::call_site()),

        (_, x) => panic!("bad span {}", x),
    }

    quote!(use $krate::$ident as _;)
}

#[proc_macro]
pub fn declare_macro(input: TokenStream) -> TokenStream {
    let mut input = input.into_iter();
    let TokenTree::Ident(mut krate) = input.next().unwrap() else { panic!("missing $crate") };
    let TokenTree::Ident(span) = input.next().unwrap() else { panic!("missing span") };
    let TokenTree::Ident(ident) = input.next().unwrap() else { panic!("missing ident") };


    match (krate.to_string().as_str(), span.to_string().as_str()) {
        ("$crate", "input") => {},
        (_, "input") => krate = Ident::new("$crate", krate.span()),

        ("$crate", "mixed") => krate.set_span(Span::mixed_site()),
        (_, "mixed") => krate = Ident::new("$crate", Span::mixed_site()),

        ("$crate", "call") => krate.set_span(Span::call_site()),
        (_, "call") => krate = Ident::new("$crate", Span::call_site()),

        (_, x) => panic!("bad span {}", x),
    }

    quote!(
        #[macro_export]
        macro_rules! $ident {
            ($$i:ident) => {
                use $krate::$$i as _;
            };
        }
    )
}
