//! TokenStream implementation used by sysroot ABI

use crate::tt::{self, TokenTree};

#[derive(Debug, Default, Clone)]
pub struct TokenStream {
    pub token_trees: Vec<TokenTree>,
}

impl TokenStream {
    pub fn new() -> Self {
        TokenStream::default()
    }

    pub fn with_subtree(subtree: tt::Subtree) -> Self {
        if subtree.delimiter.kind != tt::DelimiterKind::Invisible {
            TokenStream { token_trees: vec![TokenTree::Subtree(subtree)] }
        } else {
            TokenStream { token_trees: subtree.token_trees }
        }
    }

    pub fn into_subtree(self) -> tt::Subtree {
        tt::Subtree { delimiter: tt::Delimiter::UNSPECIFIED, token_trees: self.token_trees }
    }

    pub fn is_empty(&self) -> bool {
        self.token_trees.is_empty()
    }
}

/// Creates a token stream containing a single token tree.
impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream { token_trees: vec![tree] }
    }
}

/// Collects a number of token trees into a single stream.
impl FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(trees: I) -> Self {
        trees.into_iter().map(TokenStream::from).collect()
    }
}

/// A "flattening" operation on token streams, collects token trees
/// from multiple token streams into a single stream.
impl FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
        let mut builder = TokenStreamBuilder::new();
        streams.into_iter().for_each(|stream| builder.push(stream));
        builder.build()
    }
}

impl Extend<TokenTree> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenTree>>(&mut self, trees: I) {
        self.extend(trees.into_iter().map(TokenStream::from));
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
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

pub struct TokenStreamBuilder {
    acc: TokenStream,
}

/// Public implementation details for the `TokenStream` type, such as iterators.
pub mod token_stream {
    use std::str::FromStr;

    use super::{tt, TokenStream, TokenTree};

    /// An iterator over `TokenStream`'s `TokenTree`s.
    /// The iteration is "shallow", e.g., the iterator doesn't recurse into delimited groups,
    /// and returns whole groups as token trees.
    impl IntoIterator for TokenStream {
        type Item = TokenTree;
        type IntoIter = std::vec::IntoIter<TokenTree>;

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
    impl FromStr for TokenStream {
        type Err = LexError;

        fn from_str(src: &str) -> Result<TokenStream, LexError> {
            let (subtree, _token_map) =
                mbe::parse_to_token_tree(src).ok_or("Failed to parse from mbe")?;

            let subtree = subtree_replace_token_ids_with_unspecified(subtree);
            Ok(TokenStream::with_subtree(subtree))
        }
    }

    impl ToString for TokenStream {
        fn to_string(&self) -> String {
            ::tt::pretty(&self.token_trees)
        }
    }

    fn subtree_replace_token_ids_with_unspecified(subtree: tt::Subtree) -> tt::Subtree {
        tt::Subtree {
            delimiter: tt::Delimiter {
                open: tt::TokenId::UNSPECIFIED,
                close: tt::TokenId::UNSPECIFIED,
                ..subtree.delimiter
            },
            token_trees: subtree
                .token_trees
                .into_iter()
                .map(token_tree_replace_token_ids_with_unspecified)
                .collect(),
        }
    }

    fn token_tree_replace_token_ids_with_unspecified(tt: tt::TokenTree) -> tt::TokenTree {
        match tt {
            tt::TokenTree::Leaf(leaf) => {
                tt::TokenTree::Leaf(leaf_replace_token_ids_with_unspecified(leaf))
            }
            tt::TokenTree::Subtree(subtree) => {
                tt::TokenTree::Subtree(subtree_replace_token_ids_with_unspecified(subtree))
            }
        }
    }

    fn leaf_replace_token_ids_with_unspecified(leaf: tt::Leaf) -> tt::Leaf {
        match leaf {
            tt::Leaf::Literal(lit) => {
                tt::Leaf::Literal(tt::Literal { span: tt::TokenId::unspecified(), ..lit })
            }
            tt::Leaf::Punct(punct) => {
                tt::Leaf::Punct(tt::Punct { span: tt::TokenId::unspecified(), ..punct })
            }
            tt::Leaf::Ident(ident) => {
                tt::Leaf::Ident(tt::Ident { span: tt::TokenId::unspecified(), ..ident })
            }
        }
    }
}

impl TokenStreamBuilder {
    pub(super) fn new() -> TokenStreamBuilder {
        TokenStreamBuilder { acc: TokenStream::new() }
    }

    pub(super) fn push(&mut self, stream: TokenStream) {
        self.acc.extend(stream.into_iter())
    }

    pub(super) fn build(self) -> TokenStream {
        self.acc
    }
}
