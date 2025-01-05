//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.

//! This quasiquoter uses macros 2.0 hygiene to reliably access
//! items from `proc_macro`, to build a `proc_macro::TokenStream`.

use crate::{
    Delimiter, Group, Ident, Literal, Punct, Spacing, Span, ToTokens, TokenStream, TokenTree,
};

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
        {
            let mut c = (
                TokenTree::from(Punct::new(':', Spacing::Joint)),
                TokenTree::from(Punct::new(':', Spacing::Alone))
            );
            c.0.set_span(Span::def_site());
            c.1.set_span(Span::def_site());
            [c.0, c.1].into_iter().collect::<TokenStream>()
        }
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
    ($($t:tt)*) => {
        {
            #[allow(unused_mut)] // In case the expansion is empty
            let mut ts = TokenStream::new();
            $(ToTokens::to_tokens(&minimal_quote_ts!($t), &mut ts);)*
            ts
        }
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

    let mut tokens = crate::TokenStream::new();
    for tree in stream {
        if after_dollar {
            after_dollar = false;
            match tree {
                TokenTree::Ident(_) => {
                    minimal_quote!(crate::ToTokens::to_tokens(&(@ tree), &mut ts);)
                        .to_tokens(&mut tokens);
                    continue;
                }
                TokenTree::Punct(ref tt) if tt.as_char() == '$' => {}
                _ => panic!("`$` must be followed by an ident or `$` in `quote!`"),
            }
        } else if let TokenTree::Punct(ref tt) = tree {
            if tt.as_char() == '$' {
                after_dollar = true;
                continue;
            }
        }

        match tree {
            TokenTree::Punct(tt) => {
                minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Punct(crate::Punct::new(
                    (@ TokenTree::from(Literal::character(tt.as_char()))),
                    (@ match tt.spacing() {
                        Spacing::Alone => minimal_quote!(crate::Spacing::Alone),
                        Spacing::Joint => minimal_quote!(crate::Spacing::Joint),
                    }),
                )), &mut ts);)
            }
            TokenTree::Group(tt) => {
                minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Group(crate::Group::new(
                    (@ match tt.delimiter() {
                        Delimiter::Parenthesis => minimal_quote!(crate::Delimiter::Parenthesis),
                        Delimiter::Brace => minimal_quote!(crate::Delimiter::Brace),
                        Delimiter::Bracket => minimal_quote!(crate::Delimiter::Bracket),
                        Delimiter::None => minimal_quote!(crate::Delimiter::None),
                    }),
                    (@ quote(tt.stream())),
                )), &mut ts);)
            }
            TokenTree::Ident(tt) => {
                minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Ident(crate::Ident::new(
                    (@ TokenTree::from(Literal::string(&tt.to_string()))),
                    (@ quote_span(proc_macro_crate.clone(), tt.span())),
                )), &mut ts);)
            }
            TokenTree::Literal(tt) => {
                minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Literal({
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
                }), &mut ts);)
            }
        }
        .to_tokens(&mut tokens);
    }
    if after_dollar {
        panic!("unexpected trailing `$` in `quote!`");
    }

    minimal_quote! {
        {
            let mut ts = crate::TokenStream::new();
            (@ tokens)
            ts
        }
    }
}

/// Quote a `Span` into a `TokenStream`.
/// This is needed to implement a custom quoter.
#[unstable(feature = "proc_macro_quote", issue = "54722")]
pub fn quote_span(proc_macro_crate: TokenStream, span: Span) -> TokenStream {
    let id = span.save_span();
    minimal_quote!((@ proc_macro_crate ) ::Span::recover_proc_macro_span((@ TokenTree::from(Literal::usize_unsuffixed(id)))))
}
