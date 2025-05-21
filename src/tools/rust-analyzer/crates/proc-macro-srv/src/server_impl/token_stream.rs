//! TokenStream implementation used by sysroot ABI

use proc_macro::bridge;

use crate::server_impl::{TopSubtree, delim_to_external, literal_kind_to_external};

#[derive(Clone)]
pub struct TokenStream<S> {
    pub(super) token_trees: Vec<tt::TokenTree<S>>,
}

// #[derive(Default)] would mean that `S: Default`.
impl<S> Default for TokenStream<S> {
    fn default() -> Self {
        Self { token_trees: Default::default() }
    }
}

impl<S: std::fmt::Debug + Copy> std::fmt::Debug for TokenStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenStream")
            .field("token_trees", &tt::TokenTreesView::new(&self.token_trees))
            .finish()
    }
}

impl<S: Copy> TokenStream<S> {
    pub(crate) fn with_subtree(subtree: TopSubtree<S>) -> Self {
        let delimiter_kind = subtree.top_subtree().delimiter.kind;
        let mut token_trees = subtree.0;
        if delimiter_kind == tt::DelimiterKind::Invisible {
            token_trees.remove(0);
        }
        TokenStream { token_trees }
    }

    pub(crate) fn into_subtree(mut self, call_site: S) -> TopSubtree<S>
    where
        S: Copy,
    {
        self.token_trees.insert(
            0,
            tt::TokenTree::Subtree(tt::Subtree {
                delimiter: tt::Delimiter {
                    open: call_site,
                    close: call_site,
                    kind: tt::DelimiterKind::Invisible,
                },
                len: self.token_trees.len() as u32,
            }),
        );
        TopSubtree(self.token_trees)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.token_trees.is_empty()
    }

    pub(crate) fn into_bridge(
        self,
        join_spans: &mut dyn FnMut(S, S) -> S,
    ) -> Vec<bridge::TokenTree<Self, S, intern::Symbol>> {
        let mut result = Vec::new();
        let mut iter = self.token_trees.into_iter();
        while let Some(tree) = iter.next() {
            match tree {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    result.push(bridge::TokenTree::Ident(bridge::Ident {
                        sym: ident.sym,
                        is_raw: ident.is_raw.yes(),
                        span: ident.span,
                    }))
                }
                // Note, we do not have to assemble our `-` punct and literal split into a single
                // negative bridge literal here. As the proc-macro docs state
                // > Literals created from negative numbers might not survive round-trips through
                // > TokenStream or strings and may be broken into two tokens (- and positive
                // > literal).
                tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => {
                    result.push(bridge::TokenTree::Literal(bridge::Literal {
                        span: lit.span,
                        kind: literal_kind_to_external(lit.kind),
                        symbol: lit.symbol,
                        suffix: lit.suffix,
                    }))
                }
                tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) => {
                    result.push(bridge::TokenTree::Punct(bridge::Punct {
                        ch: punct.char as u8,
                        joint: punct.spacing == tt::Spacing::Joint,
                        span: punct.span,
                    }))
                }
                tt::TokenTree::Subtree(subtree) => {
                    result.push(bridge::TokenTree::Group(bridge::Group {
                        delimiter: delim_to_external(subtree.delimiter),
                        stream: if subtree.len == 0 {
                            None
                        } else {
                            Some(TokenStream {
                                token_trees: iter.by_ref().take(subtree.usize_len()).collect(),
                            })
                        },
                        span: bridge::DelimSpan {
                            open: subtree.delimiter.open,
                            close: subtree.delimiter.close,
                            entire: join_spans(subtree.delimiter.open, subtree.delimiter.close),
                        },
                    }))
                }
            }
        }
        result
    }
}

pub(super) struct TokenStreamBuilder<S> {
    acc: TokenStream<S>,
}

/// pub(super)lic implementation details for the `TokenStream` type, such as iterators.
pub(super) mod token_stream_impls {

    use core::fmt;

    use super::{TokenStream, TopSubtree};

    /// Attempts to break the string into tokens and parse those tokens into a token stream.
    /// May fail for a number of reasons, for example, if the string contains unbalanced delimiters
    /// or characters not existing in the language.
    /// All tokens in the parsed stream get `Span::call_site()` spans.
    ///
    /// NOTE: some errors may cause panics instead of returning `LexError`. We reserve the right to
    /// change these errors into `LexError`s later.
    impl<S: Copy + fmt::Debug> TokenStream<S> {
        pub(crate) fn from_str(src: &str, call_site: S) -> Result<TokenStream<S>, String> {
            let subtree = syntax_bridge::parse_to_token_tree_static_span(
                span::Edition::CURRENT_FIXME,
                call_site,
                src,
            )
            .ok_or_else(|| format!("lexing error: {src}"))?;

            Ok(TokenStream::with_subtree(TopSubtree(subtree.0.into_vec())))
        }
    }

    #[allow(clippy::to_string_trait_impl)]
    impl<S> ToString for TokenStream<S> {
        fn to_string(&self) -> String {
            ::tt::pretty(&self.token_trees)
        }
    }
}

impl<S: Copy> TokenStreamBuilder<S> {
    pub(super) fn push(&mut self, stream: TokenStream<S>) {
        self.acc.token_trees.extend(stream.token_trees)
    }

    pub(super) fn build(self) -> TokenStream<S> {
        self.acc
    }
}

impl<S: Copy> Default for TokenStreamBuilder<S> {
    fn default() -> Self {
        Self { acc: TokenStream::default() }
    }
}
