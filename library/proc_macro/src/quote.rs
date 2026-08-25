//! # Quasiquoter
//! This file contains the implementation internals of the quasiquoter provided by `quote!`.

//! This quasiquoter uses macros 2.0 hygiene to reliably access
//! items from `proc_macro`, to build a `proc_macro::TokenStream`.

use crate::{
    BitOr, Delimiter, Group, Ident, Literal, Punct, Spacing, Span, ToTokens, TokenStream, TokenTree,
};

#[doc(hidden)]
pub struct HasIterator; // True
#[doc(hidden)]
pub struct ThereIsNoIteratorInRepetition; // False

impl BitOr<ThereIsNoIteratorInRepetition> for ThereIsNoIteratorInRepetition {
    type Output = ThereIsNoIteratorInRepetition;
    fn bitor(self, _rhs: ThereIsNoIteratorInRepetition) -> ThereIsNoIteratorInRepetition {
        ThereIsNoIteratorInRepetition
    }
}

impl BitOr<ThereIsNoIteratorInRepetition> for HasIterator {
    type Output = HasIterator;
    fn bitor(self, _rhs: ThereIsNoIteratorInRepetition) -> HasIterator {
        HasIterator
    }
}

impl BitOr<HasIterator> for ThereIsNoIteratorInRepetition {
    type Output = HasIterator;
    fn bitor(self, _rhs: HasIterator) -> HasIterator {
        HasIterator
    }
}

impl BitOr<HasIterator> for HasIterator {
    type Output = HasIterator;
    fn bitor(self, _rhs: HasIterator) -> HasIterator {
        HasIterator
    }
}

/// Extension traits used by the implementation of `quote!`. These are defined
/// in separate traits, rather than as a single trait due to ambiguity issues.
///
/// These traits expose a `quote_into_iter` method which should allow calling
/// whichever impl happens to be applicable. Calling that method repeatedly on
/// the returned value should be idempotent.
#[doc(hidden)]
pub mod ext {
    use core::slice;
    use std::collections::btree_set::{self, BTreeSet};

    use super::{
        HasIterator as HasIter, RepInterp, ThereIsNoIteratorInRepetition as DoesNotHaveIter,
    };
    use crate::ToTokens;

    /// Extension trait providing the `quote_into_iter` method on iterators.
    #[doc(hidden)]
    pub trait RepIteratorExt: Iterator + Sized {
        fn quote_into_iter(self) -> (Self, HasIter) {
            (self, HasIter)
        }
    }

    impl<T: Iterator> RepIteratorExt for T {}

    /// Extension trait providing the `quote_into_iter` method for
    /// non-iterable types. These types interpolate the same value in each
    /// iteration of the repetition.
    #[doc(hidden)]
    pub trait RepToTokensExt {
        /// Pretend to be an iterator for the purposes of `quote_into_iter`.
        /// This allows repeated calls to `quote_into_iter` to continue
        /// correctly returning DoesNotHaveIter.
        fn next(&self) -> Option<&Self> {
            Some(self)
        }

        fn quote_into_iter(&self) -> (&Self, DoesNotHaveIter) {
            (self, DoesNotHaveIter)
        }
    }

    impl<T: ToTokens + ?Sized> RepToTokensExt for T {}

    /// Extension trait providing the `quote_into_iter` method for types that
    /// can be referenced as an iterator.
    #[doc(hidden)]
    pub trait RepAsIteratorExt<'q> {
        type Iter: Iterator;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter);
    }

    impl<'q, T: RepAsIteratorExt<'q> + ?Sized> RepAsIteratorExt<'q> for &T {
        type Iter = T::Iter;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            <T as RepAsIteratorExt>::quote_into_iter(*self)
        }
    }

    impl<'q, T: RepAsIteratorExt<'q> + ?Sized> RepAsIteratorExt<'q> for &mut T {
        type Iter = T::Iter;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            <T as RepAsIteratorExt>::quote_into_iter(*self)
        }
    }

    impl<'q, T: 'q> RepAsIteratorExt<'q> for [T] {
        type Iter = slice::Iter<'q, T>;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            (self.iter(), HasIter)
        }
    }

    impl<'q, T: 'q, const N: usize> RepAsIteratorExt<'q> for [T; N] {
        type Iter = slice::Iter<'q, T>;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            (self.iter(), HasIter)
        }
    }

    impl<'q, T: 'q> RepAsIteratorExt<'q> for Vec<T> {
        type Iter = slice::Iter<'q, T>;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            (self.iter(), HasIter)
        }
    }

    impl<'q, T: 'q> RepAsIteratorExt<'q> for BTreeSet<T> {
        type Iter = btree_set::Iter<'q, T>;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            (self.iter(), HasIter)
        }
    }

    impl<'q, T: RepAsIteratorExt<'q>> RepAsIteratorExt<'q> for RepInterp<T> {
        type Iter = T::Iter;

        fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
            self.0.quote_into_iter()
        }
    }
}

// Helper type used within interpolations to allow for repeated binding names.
// Implements the relevant traits, and exports a dummy `next()` method.
#[derive(Copy, Clone)]
#[doc(hidden)]
pub struct RepInterp<T>(pub T);

impl<T> RepInterp<T> {
    // This method is intended to look like `Iterator::next`, and is called when
    // a name is bound multiple times, as the previous binding will shadow the
    // original `Iterator` object. This allows us to avoid advancing the
    // iterator multiple times per iteration.
    pub fn next(self) -> Option<T> {
        Some(self.0)
    }
}

impl<T: Iterator> Iterator for RepInterp<T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T: ToTokens> ToTokens for RepInterp<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.to_tokens(tokens);
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
    let mut iter = stream.into_iter().peekable();
    while let Some(tree) = iter.next() {
        if after_dollar {
            after_dollar = false;
            match tree {
                TokenTree::Group(tt) => {
                    // Handles repetition by expanding `$( CONTENTS ) SEP_OPT *` to `{ REP_EXPANDED }`.
                    let contents = tt.stream();

                    // The `*` token is also consumed here.
                    let sep_opt: Option<Punct> = match (iter.next(), iter.peek()) {
                        (Some(TokenTree::Punct(sep)), Some(TokenTree::Punct(star)))
                            if sep.spacing() == Spacing::Joint && star.as_char() == '*' =>
                        {
                            iter.next();
                            Some(sep)
                        }
                        (Some(TokenTree::Punct(star)), _) if star.as_char() == '*' => None,
                        _ => panic!("`$(...)` must be followed by `*` in `quote!`"),
                    };

                    let mut rep_expanded = TokenStream::new();

                    // Append setup code for a `while`, where recursively quoted `CONTENTS`
                    // and `SEP_OPT` are repeatedly processed, to `REP_EXPANDED`.
                    let meta_vars = collect_meta_vars(contents.clone());
                    minimal_quote!(
                        use crate::ext::*;
                        (@ if sep_opt.is_some() {
                            minimal_quote!(let mut _i = 0usize;)
                        } else {
                            minimal_quote!(();)
                        })
                        let has_iter = crate::ThereIsNoIteratorInRepetition;
                    )
                    .to_tokens(&mut rep_expanded);
                    for meta_var in &meta_vars {
                        minimal_quote!(
                            #[allow(unused_mut)]
                            let (mut (@ meta_var), i) = (@ meta_var).quote_into_iter();
                            let has_iter = has_iter | i;
                        )
                        .to_tokens(&mut rep_expanded);
                    }
                    minimal_quote!(let _: crate::HasIterator = has_iter;)
                        .to_tokens(&mut rep_expanded);

                    // Append the `while` to `REP_EXPANDED`.
                    let mut while_body = TokenStream::new();
                    for meta_var in &meta_vars {
                        minimal_quote!(
                            let (@ meta_var) = match (@ meta_var).next() {
                                Some(_x) => crate::RepInterp(_x),
                                None => break,
                            };
                        )
                        .to_tokens(&mut while_body);
                    }
                    minimal_quote!(
                        (@ if let Some(sep) = sep_opt {
                            minimal_quote!(
                                if _i > 0 {
                                    (@ minimal_quote!(crate::ToTokens::to_tokens(&crate::TokenTree::Punct(crate::Punct::new(
                                        (@ TokenTree::from(Literal::character(sep.as_char()))),
                                        (@ minimal_quote!(crate::Spacing::Alone)),
                                    )), &mut ts);))
                                }
                                _i += 1;
                            )
                        } else {
                            minimal_quote!(();)
                        })
                        (@ quote(contents.clone())).to_tokens(&mut ts);
                    )
                        .to_tokens(&mut while_body);
                    rep_expanded.extend(vec![
                        TokenTree::Ident(Ident::new("while", Span::call_site())),
                        TokenTree::Ident(Ident::new("true", Span::call_site())),
                        TokenTree::Group(Group::new(Delimiter::Brace, while_body)),
                    ]);

                    minimal_quote!((@ TokenTree::Group(Group::new(Delimiter::Brace, rep_expanded)))).to_tokens(&mut tokens);
                    continue;
                }
                TokenTree::Ident(_) => {
                    minimal_quote!(crate::ToTokens::to_tokens(&(@ tree), &mut ts);)
                        .to_tokens(&mut tokens);
                    continue;
                }
                TokenTree::Punct(ref tt) if tt.as_char() == '$' => {}
                _ => panic!(
                    "`$` must be followed by an ident or `$` or a repetition group in `quote!`"
                ),
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

/// Helper function to support macro repetitions like `$( CONTENTS ) SEP_OPT *` in `quote!`.
/// Recursively collects all `Ident`s (meta-variables) that follow a `$`
/// from the given `CONTENTS` stream, preserving their order of appearance.
fn collect_meta_vars(content_stream: TokenStream) -> Vec<Ident> {
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
    helper(content_stream, &mut vars);
    vars
}

/// Quote a `Span` into a `TokenStream`.
/// This is needed to implement a custom quoter.
#[unstable(feature = "proc_macro_quote", issue = "54722")]
pub fn quote_span(proc_macro_crate: TokenStream, span: Span) -> TokenStream {
    let id = span.save_span();
    minimal_quote!((@ proc_macro_crate ) ::Span::recover_proc_macro_span((@ TokenTree::from(Literal::usize_unsuffixed(id)))))
}
