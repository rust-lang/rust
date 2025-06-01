//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.

//! This quasiquoter uses macros 2.0 hygiene to reliably access
//! items from `proc_macro`, to build a `proc_macro::TokenStream`.

use crate::{
    Delimiter, Group, Ident, Literal, Punct, Spacing, Span, ToTokens, TokenStream, TokenTree,
    token_stream,
};

// Helper for quote to parse TokenStream
struct LookaheadIter {
    iter: token_stream::IntoIter,
    tokens: [Option<TokenTree>; 2],
    pos: usize,
}

impl LookaheadIter {
    fn new(ts: TokenStream) -> Self {
        Self { iter: ts.into_iter(), tokens: [None, None], pos: 0 }
    }

    fn peek(&mut self) -> Option<TokenTree> {
        if self.pos == 2 {
            panic!("invalid peek")
        }

        self.tokens[self.pos] = self.iter.next();
        let token_opt = self.tokens[self.pos].clone();
        self.pos += 1;
        token_opt
    }

    fn next(&mut self) -> Option<TokenTree> {
        let token_opt = match self.pos {
            0 => self.iter.next(),
            1 => {
                self.pos = 0;
                self.tokens[0].take()
            }
            2 => {
                self.pos = 1;
                let token_opt = self.tokens[0].take();
                self.tokens[0] = self.tokens[1].take();
                token_opt
            }
            _ => panic!("invalid next"),
        };
        token_opt
    }

    fn nth(&mut self, n: usize) -> Option<TokenTree> {
        for _ in 0..n - 1 {
            self.next();
        }
        self.next()
    }
}

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
    (#) => { Punct::new('#', Spacing::Alone) };
    (|) => { Punct::new('|', Spacing::Alone) };
    (:) => { Punct::new(':', Spacing::Alone) };
    (*) => { Punct::new('*', Spacing::Alone) };
    (_) => { Ident::new("_", Span::def_site()) };
    ($i:ident) => { Ident::new(stringify!($i), Span::def_site()) };
    ($lit:literal) => { stringify!($lit).parse::<Literal>().unwrap() };
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
    (=>) => {
        {
            let mut c = (
                TokenTree::from(Punct::new('=', Spacing::Joint)),
                TokenTree::from(Punct::new('>', Spacing::Alone))
            );
            c.0.set_span(Span::def_site());
            c.1.set_span(Span::def_site());
            [c.0, c.1].into_iter().collect::<TokenStream>()
        }
    };
    (+=) => {
        {
            let mut c = (
                TokenTree::from(Punct::new('+', Spacing::Joint)),
                TokenTree::from(Punct::new('=', Spacing::Alone))
            );
            c.0.set_span(Span::def_site());
            c.1.set_span(Span::def_site());
            [c.0, c.1].into_iter().collect::<TokenStream>()
        }
    };
    (!=) => {
        {
            let mut c = (
                TokenTree::from(Punct::new('!', Spacing::Joint)),
                TokenTree::from(Punct::new('=', Spacing::Alone))
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
    let mut iter = LookaheadIter::new(stream);
    while let Some(tree) = iter.next() {
        if after_dollar {
            after_dollar = false;
            match tree {
                TokenTree::Group(inner) => {
                    let content = inner.stream();

                    let sep_opt: Option<TokenTree> = match (iter.peek(), iter.peek()) {
                        (Some(TokenTree::Punct(tt1)), Some(TokenTree::Punct(tt2)))
                            if tt1.spacing() == Spacing::Joint && tt2.as_char() == '*' =>
                        {
                            iter.nth(2);
                            Some(TokenTree::Punct(tt1))
                        }
                        (Some(TokenTree::Punct(tt1)), _) if tt1.as_char() == '*' => {
                            iter.next();
                            None
                        }
                        _ => {
                            panic!("`$(...)` must be followed by `*` in `quote!`");
                        }
                    };

                    let meta_vars = collect_meta_vars(content.clone());

                    let mut content_tokens = TokenStream::new();
                    minimal_quote!(
                        use crate::ext::*;
                    )
                    .to_tokens(&mut content_tokens);
                    if let Some(TokenTree::Punct(_)) = sep_opt {
                        minimal_quote!(
                            let mut _i = 0usize;
                        )
                        .to_tokens(&mut content_tokens);
                    }
                    minimal_quote!(
                        let has_iter = crate::ThereIsNoIteratorInRepetition;
                    )
                    .to_tokens(&mut content_tokens);
                    for meta_var in &meta_vars {
                        minimal_quote!(
                            #[allow(unused_mut)]
                            let (mut (@ meta_var), i) = (@ meta_var).quote_into_iter();
                            let has_iter = has_iter | i;
                        )
                        .to_tokens(&mut content_tokens);
                    }
                    minimal_quote!(
                        let _: crate::HasIterator = has_iter;
                    )
                    .to_tokens(&mut content_tokens);

                    let while_ident = TokenTree::Ident(Ident::new("while", Span::call_site()));
                    let true_literal = TokenTree::Ident(Ident::new("true", Span::call_site()));

                    let mut inner_tokens = TokenStream::new();
                    for meta_var in &meta_vars {
                        minimal_quote!(
                            let (@ meta_var) = match (@ meta_var).next() {
                                Some(_x) => crate::RepInterp(_x),
                                None => break,
                            };
                        )
                        .to_tokens(&mut inner_tokens);
                    }
                    if let Some(TokenTree::Punct(tt)) = sep_opt {
                        minimal_quote!(
                            if _i > 0 {
                                (@ minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Punct(crate::Punct::new(
                                    (@ TokenTree::from(Literal::character(tt.as_char()))),
                                    (@ minimal_quote!(crate::Spacing::Alone)),
                                )), &mut ts);))
                            }
                            _i += 1;
                        )
                        .to_tokens(&mut inner_tokens);
                    };
                    minimal_quote!(
                        (@ quote(content.clone())).to_tokens(&mut ts);
                    )
                    .to_tokens(&mut inner_tokens);
                    let while_block = TokenTree::Group(Group::new(Delimiter::Brace, inner_tokens));

                    content_tokens.extend(vec![while_ident, true_literal, while_block]);
                    let block = TokenTree::Group(Group::new(Delimiter::Brace, content_tokens));
                    minimal_quote!((@ block)).to_tokens(&mut tokens);

                    continue;
                }
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
                let literal = tt.to_string();
                let (literal, ctor) = if let Some(stripped) = literal.strip_prefix("r#") {
                    (stripped, minimal_quote!(crate::Ident::new_raw))
                } else {
                    (literal.as_str(), minimal_quote!(crate::Ident::new))
                };
                minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Ident((@ ctor)(
                    (@ TokenTree::from(Literal::string(literal))),
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

fn collect_meta_vars(stream: TokenStream) -> Vec<Ident> {
    fn helper(stream: TokenStream, out: &mut Vec<Ident>) {
        let mut iter = stream.into_iter().peekable();
        while let Some(tree) = iter.next() {
            match &tree {
                TokenTree::Punct(tt) if tt.as_char() == '$' => {
                    if let Some(TokenTree::Ident(id)) = iter.peek() {
                        out.push(id.clone());
                        iter.next();
                    }
                }
                TokenTree::Group(tt) => {
                    helper(tt.stream(), out);
                }
                _ => {}
            }
        }
    }

    let mut vars = Vec::new();
    helper(stream, &mut vars);
    vars
}

/// Quote a `Span` into a `TokenStream`.
/// This is needed to implement a custom quoter.
#[unstable(feature = "proc_macro_quote", issue = "54722")]
pub fn quote_span(proc_macro_crate: TokenStream, span: Span) -> TokenStream {
    let id = span.save_span();
    minimal_quote!((@ proc_macro_crate ) ::Span::recover_proc_macro_span((@ TokenTree::from(Literal::usize_unsuffixed(id)))))
}
