use crate::parse::ParseSess;
use crate::session::Session;
use rustc_ast::token::{self, DelimToken, Nonterminal, Token};
use rustc_ast::tokenstream::CanSynthesizeMissingTokens;
use rustc_ast::tokenstream::{DelimSpan, TokenStream, TokenTree};
use rustc_data_structures::profiling::VerboseTimingGuard;
use std::path::{Path, PathBuf};

pub type NtToTokenstream = fn(&Nonterminal, &ParseSess, CanSynthesizeMissingTokens) -> TokenStream;

impl Session {
    pub fn timer<'a>(&'a self, what: &'static str) -> VerboseTimingGuard<'a> {
        self.prof.verbose_generic_activity(what)
    }
    pub fn time<R>(&self, what: &'static str, f: impl FnOnce() -> R) -> R {
        self.prof.verbose_generic_activity(what).run(f)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
pub enum NativeLibKind {
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC) included
    /// when linking a final binary, but not when archiving an rlib.
    StaticNoBundle,
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC) included
    /// when linking a final binary, but also included when archiving an rlib.
    StaticBundle,
    /// Dynamic library (e.g. `libfoo.so` on Linux)
    /// or an import library corresponding to a dynamic library (e.g. `foo.lib` on Windows/MSVC).
    Dylib,
    /// Dynamic library (e.g. `foo.dll` on Windows) without a corresponding import library.
    RawDylib,
    /// A macOS-specific kind of dynamic libraries.
    Framework,
    /// The library kind wasn't specified, `Dylib` is currently used as a default.
    Unspecified,
}

rustc_data_structures::impl_stable_hash_via_hash!(NativeLibKind);

/// A path that has been canonicalized along with its original, non-canonicalized form
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalizedPath {
    // Optional since canonicalization can sometimes fail
    canonicalized: Option<PathBuf>,
    original: PathBuf,
}

impl CanonicalizedPath {
    pub fn new(path: &Path) -> Self {
        Self { original: path.to_owned(), canonicalized: std::fs::canonicalize(path).ok() }
    }

    pub fn canonicalized(&self) -> &PathBuf {
        self.canonicalized.as_ref().unwrap_or(self.original())
    }

    pub fn original(&self) -> &PathBuf {
        &self.original
    }
}

// FIXME: Find a better spot for this - it needs to be accessible from `rustc_ast_lowering`,
// and needs to access `ParseSess
pub struct FlattenNonterminals<'a> {
    pub parse_sess: &'a ParseSess,
    pub synthesize_tokens: CanSynthesizeMissingTokens,
    pub nt_to_tokenstream: NtToTokenstream,
}

impl<'a> FlattenNonterminals<'a> {
    pub fn process_token_stream(&mut self, tokens: TokenStream) -> TokenStream {
        fn can_skip(stream: &TokenStream) -> bool {
            stream.trees().all(|tree| match tree {
                TokenTree::Token(token) => !matches!(token.kind, token::Interpolated(_)),
                TokenTree::Delimited(_, _, inner) => can_skip(&inner),
            })
        }

        if can_skip(&tokens) {
            return tokens;
        }

        tokens.into_trees().flat_map(|tree| self.process_token_tree(tree).into_trees()).collect()
    }

    pub fn process_token_tree(&mut self, tree: TokenTree) -> TokenStream {
        match tree {
            TokenTree::Token(token) => self.process_token(token),
            TokenTree::Delimited(span, delim, tts) => {
                TokenTree::Delimited(span, delim, self.process_token_stream(tts)).into()
            }
        }
    }

    pub fn process_token(&mut self, token: Token) -> TokenStream {
        match token.kind {
            token::Interpolated(nt) => {
                let tts = (self.nt_to_tokenstream)(&nt, self.parse_sess, self.synthesize_tokens);
                TokenTree::Delimited(
                    DelimSpan::from_single(token.span),
                    DelimToken::NoDelim,
                    self.process_token_stream(tts),
                )
                .into()
            }
            _ => TokenTree::Token(token).into(),
        }
    }
}
