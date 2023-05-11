//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.

//! This quasiquoter uses macros 2.0 hygiene to reliably access
//! items from `proc_macro`, to build a `proc_macro::TokenStream`.

use super::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

macro_rules! quote_tt {
    (($($t:tt)*)) => { Group::new(Delimiter::Parenthesis, quote!($($t)*)) };
    ([$($t:tt)*]) => { Group::new(Delimiter::Bracket, quote!($($t)*)) };
    ({$($t:tt)*}) => { Group::new(Delimiter::Brace, quote!($($t)*)) };
    (,) => { Punct::new(',', Spacing::Alone) };
    (.) => { Punct::new('.', Spacing::Alone) };
    (;) => { Punct::new(';', Spacing::Alone) };
    (!) => { Punct::new('!', Spacing::Alone) };
    (<) => { Punct::new('<', Spacing::Alone) };
    (>) => { Punct::new('>', Spacing::Alone) };
    (&) => { Punct::new('&', Spacing::Alone) };
    (=) => { Punct::new('=', Spacing::Alone) };
    ($i:ident) => { Ident::new(stringify!($i), Span::def_site()) };
}

macro_rules! quote_ts {
    ((@ $($t:tt)*)) => { $($t)* };
    (::) => {
        [
            TokenTree::from(Punct::new(':', Spacing::Joint)),
            TokenTree::from(Punct::new(':', Spacing::Alone)),
        ].iter()
            .cloned()
            .map(|mut x| {
                x.set_span(Span::def_site());
                x
            })
            .collect::<TokenStream>()
    };
    ($t:tt) => { TokenTree::from(quote_tt!($t)) };
}

/// Simpler version of the real `quote!` macro, implemented solely
/// through `macro_rules`, for bootstrapping the real implementation
/// (see the `quote` function), which does not have access to the
/// real `quote!` macro due to the `proc_macro` crate not being
/// able to depend on itself.
///
/// Note: supported tokens are a subset of the real `quote!`, but
/// unquoting is different: instead of `$x`, this uses `(@ expr)`.
macro_rules! quote {
    () => { TokenStream::new() };
    ($($t:tt)*) => {
        [
            $(TokenStream::from(quote_ts!($t)),)*
        ].iter().cloned().collect::<TokenStream>()
    };
}

/// Quote a `TokenStream` into a `TokenStream`.
/// This is the actual implementation of the `quote!()` proc macro.
///
/// It is loaded by the compiler in `register_builtin_macros`.
pub fn quote(stream: TokenStream) -> TokenStream {
    if stream.is_empty() {
        return quote!(super::TokenStream::new());
    }
    let proc_macro_crate = quote!(crate);
    let mut after_dollar = false;
    let tokens = stream
        .into_iter()
        .filter_map(|tree| {
            if after_dollar {
                after_dollar = false;
                match tree {
                    TokenTree::Ident(_) => {
                        return Some(quote!(Into::<super::TokenStream>::into(
                        Clone::clone(&(@ tree))),));
                    }
                    TokenTree::Punct(ref tt) if tt.as_char() == '$' => {}
                    _ => panic!("`$` must be followed by an ident or `$` in `quote!`"),
                }
            } else if let TokenTree::Punct(ref tt) = tree {
                if tt.as_char() == '$' {
                    after_dollar = true;
                    return None;
                }
            }

            Some(quote!(super::TokenStream::from((@ match tree {
                TokenTree::Punct(tt) => quote!(super::TokenTree::Punct(super::Punct::new(
                    (@ TokenTree::from(Literal::character(tt.as_char()))),
                    (@ match tt.spacing() {
                        Spacing::Alone => quote!(super::Spacing::Alone),
                        Spacing::Joint => quote!(super::Spacing::Joint),
                    }),
                ))),
                TokenTree::Group(tt) => quote!(super::TokenTree::Group(super::Group::new(
                    (@ match tt.delimiter() {
                        Delimiter::Parenthesis => quote!(super::Delimiter::Parenthesis),
                        Delimiter::Brace => quote!(super::Delimiter::Brace),
                        Delimiter::Bracket => quote!(super::Delimiter::Bracket),
                        Delimiter::None => quote!(super::Delimiter::None),
                    }),
                    (@ quote(tt.stream())),
                ))),
                TokenTree::Ident(tt) => quote!(super::TokenTree::Ident(super::Ident::new(
                    (@ TokenTree::from(Literal::string(&tt.to_string()))),
                    (@ quote_span(proc_macro_crate.clone(), tt.span())),
                ))),
                TokenTree::Literal(tt) => quote!(super::TokenTree::Literal({
                    let mut iter = (@ TokenTree::from(Literal::string(&tt.to_string())))
                        .parse::<super::TokenStream>()
                        .unwrap()
                        .into_iter();
                    if let (Some(super::TokenTree::Literal(mut lit)), None) =
                        (iter.next(), iter.next())
                    {
                        lit.set_span((@ quote_span(proc_macro_crate.clone(), tt.span())));
                        lit
                    } else {
                        unreachable!()
                    }
                }))
            })),))
        })
        .collect::<TokenStream>();

    if after_dollar {
        panic!("unexpected trailing `$` in `quote!`");
    }

    quote!([(@ tokens)].iter().cloned().collect::<super::TokenStream>())
}

/// Quote a `Span` into a `TokenStream`.
/// This is needed to implement a custom quoter.
pub fn quote_span(proc_macro_crate: TokenStream, span: Span) -> TokenStream {
    let id = span.save_span();
    quote!((@ proc_macro_crate ) ::Span::recover_proc_macro_span((@ TokenTree::from(Literal::usize_unsuffixed(id)))))
}
