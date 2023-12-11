//! TokenStream implementation used by sysroot ABI

use tt::TokenTree;

#[derive(Debug, Clone)]
pub struct TokenStream<S> {
    pub(super) token_trees: Vec<TokenTree<S>>,
}

impl<S> Default for TokenStream<S> {
    fn default() -> Self {
        Self { token_trees: vec![] }
    }
}

impl<S> TokenStream<S> {
    pub(crate) fn new() -> Self {
        TokenStream { token_trees: vec![] }
    }

    pub(crate) fn with_subtree(subtree: tt::Subtree<S>) -> Self {
        if subtree.delimiter.kind != tt::DelimiterKind::Invisible {
            TokenStream { token_trees: vec![TokenTree::Subtree(subtree)] }
        } else {
            TokenStream { token_trees: subtree.token_trees }
        }
    }

    pub(crate) fn into_subtree(self, call_site: S) -> tt::Subtree<S>
    where
        S: Copy,
    {
        tt::Subtree {
            delimiter: tt::Delimiter {
                open: call_site,
                close: call_site,
                kind: tt::DelimiterKind::Invisible,
            },
            token_trees: self.token_trees,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.token_trees.is_empty()
    }
}

/// Creates a token stream containing a single token tree.
impl<S> From<TokenTree<S>> for TokenStream<S> {
    fn from(tree: TokenTree<S>) -> TokenStream<S> {
        TokenStream { token_trees: vec![tree] }
    }
}

/// Collects a number of token trees into a single stream.
impl<S> FromIterator<TokenTree<S>> for TokenStream<S> {
    fn from_iter<I: IntoIterator<Item = TokenTree<S>>>(trees: I) -> Self {
        trees.into_iter().map(TokenStream::from).collect()
    }
}

/// A "flattening" operation on token streams, collects token trees
/// from multiple token streams into a single stream.
impl<S> FromIterator<TokenStream<S>> for TokenStream<S> {
    fn from_iter<I: IntoIterator<Item = TokenStream<S>>>(streams: I) -> Self {
        let mut builder = TokenStreamBuilder::new();
        streams.into_iter().for_each(|stream| builder.push(stream));
        builder.build()
    }
}

impl<S> Extend<TokenTree<S>> for TokenStream<S> {
    fn extend<I: IntoIterator<Item = TokenTree<S>>>(&mut self, trees: I) {
        self.extend(trees.into_iter().map(TokenStream::from));
    }
}

impl<S> Extend<TokenStream<S>> for TokenStream<S> {
    fn extend<I: IntoIterator<Item = TokenStream<S>>>(&mut self, streams: I) {
        for item in streams {
            for tkn in item {
                match tkn {
                    tt::TokenTree::Subtree(subtree)
                        if subtree.delimiter.kind == tt::DelimiterKind::Invisible =>
                    {
                        self.token_trees.extend(subtree.token_trees);
                    }
                    _ => {
                        self.token_trees.push(tkn);
                    }
                }
            }
        }
    }
}

pub(super) struct TokenStreamBuilder<S> {
    acc: TokenStream<S>,
}

/// pub(super)lic implementation details for the `TokenStream` type, such as iterators.
pub(super) mod token_stream {

    use super::{TokenStream, TokenTree};

    /// An iterator over `TokenStream`'s `TokenTree`s.
    /// The iteration is "shallow", e.g., the iterator doesn't recurse into delimited groups,
    /// and returns whole groups as token trees.
    impl<S> IntoIterator for TokenStream<S> {
        type Item = TokenTree<S>;
        type IntoIter = std::vec::IntoIter<TokenTree<S>>;

        fn into_iter(self) -> Self::IntoIter {
            self.token_trees.into_iter()
        }
    }

    type LexError = String;

    /// Attempts to break the string into tokens and parse those tokens into a token stream.
    /// May fail for a number of reasons, for example, if the string contains unbalanced delimiters
    /// or characters not existing in the language.
    /// All tokens in the parsed stream get `Span::call_site()` spans.
    ///
    /// NOTE: some errors may cause panics instead of returning `LexError`. We reserve the right to
    /// change these errors into `LexError`s later.
    #[rustfmt::skip]
    impl<S: tt::Span> /*FromStr for*/ TokenStream<S> {
        // type Err = LexError;

        pub(crate) fn from_str(src: &str, call_site: S) -> Result<TokenStream<S>, LexError> {
            let subtree =
                mbe::parse_to_token_tree_static_span(call_site, src).ok_or("Failed to parse from mbe")?;

            Ok(TokenStream::with_subtree(subtree))
        }
    }

    impl<S> ToString for TokenStream<S> {
        fn to_string(&self) -> String {
            ::tt::pretty(&self.token_trees)
        }
    }

    fn subtree_replace_spans_with_call_site<S: Copy>(
        subtree: tt::Subtree<S>,
        call_site: S,
    ) -> tt::Subtree<S> {
        tt::Subtree {
            delimiter: tt::Delimiter { open: call_site, close: call_site, ..subtree.delimiter },
            token_trees: subtree
                .token_trees
                .into_iter()
                .map(|it| token_tree_replace_spans_with_call_site(it, call_site))
                .collect(),
        }
    }

    fn token_tree_replace_spans_with_call_site<S: Copy>(
        tt: tt::TokenTree<S>,
        call_site: S,
    ) -> tt::TokenTree<S> {
        match tt {
            tt::TokenTree::Leaf(leaf) => {
                tt::TokenTree::Leaf(leaf_replace_spans_with_call_site(leaf, call_site))
            }
            tt::TokenTree::Subtree(subtree) => {
                tt::TokenTree::Subtree(subtree_replace_spans_with_call_site(subtree, call_site))
            }
        }
    }

    fn leaf_replace_spans_with_call_site<S: Copy>(leaf: tt::Leaf<S>, call_site: S) -> tt::Leaf<S> {
        match leaf {
            tt::Leaf::Literal(lit) => tt::Leaf::Literal(tt::Literal { span: call_site, ..lit }),
            tt::Leaf::Punct(punct) => tt::Leaf::Punct(tt::Punct { span: call_site, ..punct }),
            tt::Leaf::Ident(ident) => tt::Leaf::Ident(tt::Ident { span: call_site, ..ident }),
        }
    }
}

impl<S> TokenStreamBuilder<S> {
    pub(super) fn new() -> TokenStreamBuilder<S> {
        TokenStreamBuilder { acc: TokenStream::new() }
    }

    pub(super) fn push(&mut self, stream: TokenStream<S>) {
        self.acc.extend(stream.into_iter())
    }

    pub(super) fn build(self) -> TokenStream<S> {
        self.acc
    }
}
