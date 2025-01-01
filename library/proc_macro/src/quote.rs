//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.

//! This quasiquoter uses macros 2.0 hygiene to reliably access
//! items from `proc_macro`, to build a `proc_macro::TokenStream`.

use crate::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

macro_rules! minimal_quote_tt {
    (($($t:tt)*)) => { Group::new(Delimiter::Parenthesis, minimal_quote!($($t)*)) };
    ([$($t:tt)*]) => { Group::new(Delimiter::Bracket, minimal_quote!($($t)*)) };
    ({$($t:tt)*}) => { Group::new(Delimiter::Brace, minimal_quote!($($t)*)) };
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

macro_rules! minimal_quote_ts {
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
    ($t:tt) => { TokenTree::from(minimal_quote_tt!($t)) };
}

/// Simpler version of the real `quote!` macro, implemented solely
/// through `macro_rules`, for bootstrapping the real implementation
/// (see the `quote` function), which does not have access to the
/// real `quote!` macro due to the `proc_macro` crate not being
/// able to depend on itself.
///
/// Note: supported tokens are a subset of the real `quote!`, but
/// unquoting is different: instead of `$x`, this uses `(@ expr)`.
macro_rules! minimal_quote {
    () => { TokenStream::new() };
    ($($t:tt)*) => {
        [
            $(TokenStream::from(minimal_quote_ts!($t)),)*
        ].iter().cloned().collect::<TokenStream>()
    };
}

/// Quote a `TokenStream` into a `TokenStream`.
/// This is the actual implementation of the `quote!()` proc macro.
///
/// It is loaded by the compiler in `register_builtin_macros`.
#[unstable(feature = "proc_macro_quote", issue = "54722")]
pub fn quote(stream: TokenStream) -> TokenStream {
    if stream.is_empty() {
        return minimal_quote!(crate::TokenStream::new());
    }
    let proc_macro_crate = minimal_quote!(crate);
    let mut after_dollar = false;
    let tokens = stream
        .into_iter()
        .filter_map(|tree| {
            if after_dollar {
                after_dollar = false;
                match tree {
                    TokenTree::Ident(_) => {
                        return Some(minimal_quote!(Into::<crate::TokenStream>::into(
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

            Some(minimal_quote!(crate::TokenStream::from((@ match tree {
                TokenTree::Punct(tt) => minimal_quote!(crate::TokenTree::Punct(crate::Punct::new(
                    (@ TokenTree::from(Literal::character(tt.as_char()))),
                    (@ match tt.spacing() {
                        Spacing::Alone => minimal_quote!(crate::Spacing::Alone),
                        Spacing::Joint => minimal_quote!(crate::Spacing::Joint),
                    }),
                ))),
                TokenTree::Group(tt) => minimal_quote!(crate::TokenTree::Group(crate::Group::new(
                    (@ match tt.delimiter() {
                        Delimiter::Parenthesis => minimal_quote!(crate::Delimiter::Parenthesis),
                        Delimiter::Brace => minimal_quote!(crate::Delimiter::Brace),
                        Delimiter::Bracket => minimal_quote!(crate::Delimiter::Bracket),
                        Delimiter::None => minimal_quote!(crate::Delimiter::None),
                    }),
                    (@ quote(tt.stream())),
                ))),
                TokenTree::Ident(tt) => minimal_quote!(crate::TokenTree::Ident(crate::Ident::new(
                    (@ TokenTree::from(Literal::string(&tt.to_string()))),
                    (@ quote_span(proc_macro_crate.clone(), tt.span())),
                ))),
                TokenTree::Literal(tt) => minimal_quote!(crate::TokenTree::Literal({
                    let mut iter = (@ TokenTree::from(Literal::string(&tt.to_string())))
                        .parse::<crate::TokenStream>()
                        .unwrap()
                        .into_iter();
                    if let (Some(crate::TokenTree::Literal(mut lit)), None) =
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

    minimal_quote!([(@ tokens)].iter().cloned().collect::<crate::TokenStream>())
}

/// Quote a `Span` into a `TokenStream`.
/// This is needed to implement a custom quoter.
#[unstable(feature = "proc_macro_quote", issue = "54722")]
pub fn quote_span(proc_macro_crate: TokenStream, span: Span) -> TokenStream {
    let id = span.save_span();
    minimal_quote!((@ proc_macro_crate ) ::Span::recover_proc_macro_span((@ TokenTree::from(Literal::usize_unsuffixed(id)))))
}
