#![feature(proc_macro_quote, proc_macro_span)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::eq_op)]
#![allow(clippy::literal_string_with_formatting_args)]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree, quote};

#[proc_macro_derive(DeriveSomething)]
pub fn derive(_: TokenStream) -> TokenStream {
    // Should not trigger `used_underscore_binding`
    let _inside_derive = 1;
    assert_eq!(_inside_derive, _inside_derive);

    let output = quote! {
        // Should not trigger `useless_attribute`
        #[allow(dead_code)]
        extern crate rustc_middle;
    };
    output
}

#[proc_macro_derive(ImplStructWithStdDisplay)]
pub fn derive_std(_: TokenStream) -> TokenStream {
    quote! {
        struct A {}
        impl ::std::fmt::Display for A {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "A")
            }
        }
    }
}

#[proc_macro_derive(FieldReassignWithDefault)]
pub fn derive_foo(_input: TokenStream) -> TokenStream {
    quote! {
        #[derive(Default)]
        struct A {
            pub i: i32,
            pub j: i64,
        }
        #[automatically_derived]
        fn lint() {
            let mut a: A = Default::default();
            a.i = 42;
            a;
        }
    }
}

#[proc_macro_derive(StructAUseSelf)]
pub fn derive_use_self(_input: TokenStream) -> proc_macro::TokenStream {
    quote! {
        struct A;
        impl A {
            fn new() -> A {
                A
            }
        }
    }
}

#[proc_macro_derive(ClippyMiniMacroTest)]
pub fn mini_macro(_: TokenStream) -> TokenStream {
    quote!(
        #[allow(unused)]
        fn needless_take_by_value(s: String) {
            println!("{}", s.len());
        }
        #[allow(unused)]
        fn needless_loop(items: &[u8]) {
            for i in 0..items.len() {
                println!("{}", items[i]);
            }
        }
        fn line_wrapper() {
            println!("{}", line!());
        }
    )
}

#[proc_macro_derive(ExtraLifetimeDerive)]
#[allow(unused)]
pub fn extra_lifetime(_input: TokenStream) -> TokenStream {
    quote!(
        pub struct ExtraLifetime;

        impl<'b> ExtraLifetime {
            pub fn something<'c>() -> Self {
                Self
            }
        }
    )
}

#[allow(unused)]
#[proc_macro_derive(ArithmeticDerive)]
pub fn arithmetic_derive(_: TokenStream) -> TokenStream {
    <TokenStream as FromIterator<TokenTree>>::from_iter([
        Ident::new("fn", Span::call_site()).into(),
        Ident::new("_foo", Span::call_site()).into(),
        Group::new(Delimiter::Parenthesis, TokenStream::new()).into(),
        Group::new(
            Delimiter::Brace,
            <TokenStream as FromIterator<TokenTree>>::from_iter([
                Ident::new("let", Span::call_site()).into(),
                Ident::new("mut", Span::call_site()).into(),
                Ident::new("_n", Span::call_site()).into(),
                Punct::new('=', Spacing::Alone).into(),
                Literal::i32_unsuffixed(9).into(),
                Punct::new(';', Spacing::Alone).into(),
                Ident::new("_n", Span::call_site()).into(),
                Punct::new('=', Spacing::Alone).into(),
                Literal::i32_unsuffixed(9).into(),
                Punct::new('/', Spacing::Alone).into(),
                Literal::i32_unsuffixed(2).into(),
                Punct::new(';', Spacing::Alone).into(),
                Ident::new("_n", Span::call_site()).into(),
                Punct::new('=', Spacing::Alone).into(),
                Punct::new('-', Spacing::Alone).into(),
                Ident::new("_n", Span::call_site()).into(),
                Punct::new(';', Spacing::Alone).into(),
            ]),
        )
        .into(),
    ])
}

#[allow(unused)]
#[proc_macro_derive(ShadowDerive)]
pub fn shadow_derive(_: TokenStream) -> TokenStream {
    <TokenStream as FromIterator<TokenTree>>::from_iter([
        Ident::new("fn", Span::call_site()).into(),
        Ident::new("_foo", Span::call_site()).into(),
        Group::new(Delimiter::Parenthesis, TokenStream::new()).into(),
        Group::new(
            Delimiter::Brace,
            <TokenStream as FromIterator<TokenTree>>::from_iter([
                Ident::new("let", Span::call_site()).into(),
                Ident::new("_x", Span::call_site()).into(),
                Punct::new('=', Spacing::Alone).into(),
                Literal::i32_unsuffixed(2).into(),
                Punct::new(';', Spacing::Alone).into(),
                Ident::new("let", Span::call_site()).into(),
                Ident::new("_x", Span::call_site()).into(),
                Punct::new('=', Spacing::Alone).into(),
                Ident::new("_x", Span::call_site()).into(),
                Punct::new(';', Spacing::Alone).into(),
            ]),
        )
        .into(),
    ])
}

#[proc_macro_derive(StructIgnoredUnitPattern)]
pub fn derive_ignored_unit_pattern(_: TokenStream) -> TokenStream {
    quote! {
        struct A;
        impl A {
            fn a(&self) -> Result<(), ()> {
                unimplemented!()
            }

            pub fn b(&self) {
                let _ = self.a().unwrap();
            }
        }
    }
}

#[proc_macro_derive(NonCanonicalClone)]
pub fn non_canonical_clone_derive(_: TokenStream) -> TokenStream {
    quote! {
        struct NonCanonicalClone;
        impl Clone for NonCanonicalClone {
            fn clone(&self) -> Self {
                todo!()
            }
        }
        impl Copy for NonCanonicalClone {}
    }
}

// Derive macro that generates the following but where all generated spans are set to the entire
// input span.
//
// ```
// #[allow(clippy::missing_const_for_fn)]
// fn check() {}
// ```
#[proc_macro_derive(AllowLintSameSpan)]
pub fn allow_lint_same_span_derive(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let first = iter.next().unwrap();
    let last = iter.last().unwrap();
    let span = first.span().join(last.span()).unwrap();
    let span_help = |mut t: TokenTree| -> TokenTree {
        t.set_span(span);
        t
    };
    // Generate the TokenStream but setting all the spans to the entire input span
    <TokenStream as FromIterator<TokenTree>>::from_iter([
        span_help(Punct::new('#', Spacing::Alone).into()),
        span_help(
            Group::new(
                Delimiter::Bracket,
                <TokenStream as FromIterator<TokenTree>>::from_iter([
                    Ident::new("allow", span).into(),
                    span_help(
                        Group::new(
                            Delimiter::Parenthesis,
                            <TokenStream as FromIterator<TokenTree>>::from_iter([
                                Ident::new("clippy", span).into(),
                                span_help(Punct::new(':', Spacing::Joint).into()),
                                span_help(Punct::new(':', Spacing::Alone).into()),
                                Ident::new("missing_const_for_fn", span).into(),
                            ]),
                        )
                        .into(),
                    ),
                ]),
            )
            .into(),
        ),
        Ident::new("fn", span).into(),
        Ident::new("check", span).into(),
        span_help(Group::new(Delimiter::Parenthesis, TokenStream::new()).into()),
        span_help(Group::new(Delimiter::Brace, TokenStream::new()).into()),
    ])
}
