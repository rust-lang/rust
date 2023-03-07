//! Rustc proc-macro server implementation with tt
//!
//! Based on idea from <https://github.com/fedochet/rust-proc-macro-expander>
//! The lib-proc-macro server backend is `TokenStream`-agnostic, such that
//! we could provide any TokenStream implementation.
//! The original idea from fedochet is using proc-macro2 as backend,
//! we use tt instead for better integration with RA.
//!
//! FIXME: No span and source file information is implemented yet

use super::proc_macro::bridge::{self, server};

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Bound;
use std::{ascii, vec::IntoIter};

use crate::tt;

type Group = tt::Subtree;
type TokenTree = tt::TokenTree;
type Punct = tt::Punct;
type Spacing = tt::Spacing;
type Literal = tt::Literal;
type Span = tt::TokenId;

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

#[derive(Clone)]
pub struct SourceFile {
    // FIXME stub
}

type Level = super::proc_macro::Level;
type LineColumn = super::proc_macro::LineColumn;

/// A structure representing a diagnostic message and associated children
/// messages.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    level: Level,
    message: String,
    spans: Vec<Span>,
    children: Vec<Diagnostic>,
}

impl Diagnostic {
    /// Creates a new diagnostic with the given `level` and `message`.
    pub fn new<T: Into<String>>(level: Level, message: T) -> Diagnostic {
        Diagnostic { level, message: message.into(), spans: vec![], children: vec![] }
    }
}

// Rustc Server Ident has to be `Copyable`
// We use a stub here for bypassing
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct IdentId(u32);

#[derive(Clone, Hash, Eq, PartialEq)]
struct IdentData(tt::Ident);

#[derive(Default)]
struct IdentInterner {
    idents: HashMap<IdentData, u32>,
    ident_data: Vec<IdentData>,
}

impl IdentInterner {
    fn intern(&mut self, data: &IdentData) -> u32 {
        if let Some(index) = self.idents.get(data) {
            return *index;
        }

        let index = self.idents.len() as u32;
        self.ident_data.push(data.clone());
        self.idents.insert(data.clone(), index);
        index
    }

    fn get(&self, index: u32) -> &IdentData {
        &self.ident_data[index as usize]
    }

    #[allow(unused)]
    fn get_mut(&mut self, index: u32) -> &mut IdentData {
        self.ident_data.get_mut(index as usize).expect("Should be consistent")
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
        type IntoIter = super::IntoIter<TokenTree>;

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
    fn new() -> TokenStreamBuilder {
        TokenStreamBuilder { acc: TokenStream::new() }
    }

    fn push(&mut self, stream: TokenStream) {
        self.acc.extend(stream.into_iter())
    }

    fn build(self) -> TokenStream {
        self.acc
    }
}

pub struct FreeFunctions;

#[derive(Clone)]
pub struct TokenStreamIter {
    trees: IntoIter<TokenTree>,
}

#[derive(Default)]
pub struct RustAnalyzer {
    ident_interner: IdentInterner,
    // FIXME: store span information here.
}

impl server::Types for RustAnalyzer {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type Group = Group;
    type Punct = Punct;
    type Ident = IdentId;
    type Literal = Literal;
    type SourceFile = SourceFile;
    type Diagnostic = Diagnostic;
    type Span = Span;
    type MultiSpan = Vec<Span>;
}

impl server::FreeFunctions for RustAnalyzer {
    fn track_env_var(&mut self, _var: &str, _value: Option<&str>) {
        // FIXME: track env var accesses
        // https://github.com/rust-lang/rust/pull/71858
    }
    fn track_path(&mut self, _path: &str) {}
}

impl server::TokenStream for RustAnalyzer {
    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }
    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        use std::str::FromStr;

        Self::TokenStream::from_str(src).expect("cannot parse string")
    }
    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        stream.to_string()
    }
    fn from_token_tree(
        &mut self,
        tree: bridge::TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>,
    ) -> Self::TokenStream {
        match tree {
            bridge::TokenTree::Group(group) => {
                let tree = TokenTree::from(group);
                Self::TokenStream::from_iter(vec![tree])
            }

            bridge::TokenTree::Ident(IdentId(index)) => {
                let IdentData(ident) = self.ident_interner.get(index).clone();
                let ident: tt::Ident = ident;
                let leaf = tt::Leaf::from(ident);
                let tree = TokenTree::from(leaf);
                Self::TokenStream::from_iter(vec![tree])
            }

            bridge::TokenTree::Literal(literal) => {
                let leaf = tt::Leaf::from(literal);
                let tree = TokenTree::from(leaf);
                Self::TokenStream::from_iter(vec![tree])
            }

            bridge::TokenTree::Punct(p) => {
                let leaf = tt::Leaf::from(p);
                let tree = TokenTree::from(leaf);
                Self::TokenStream::from_iter(vec![tree])
            }
        }
    }

    fn expand_expr(&mut self, self_: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        Ok(self_.clone())
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<bridge::TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>>,
    ) -> Self::TokenStream {
        let mut builder = TokenStreamBuilder::new();
        if let Some(base) = base {
            builder.push(base);
        }
        for tree in trees {
            builder.push(self.from_token_tree(tree));
        }
        builder.build()
    }

    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut builder = TokenStreamBuilder::new();
        if let Some(base) = base {
            builder.push(base);
        }
        for stream in streams {
            builder.push(stream);
        }
        builder.build()
    }

    fn into_trees(
        &mut self,
        stream: Self::TokenStream,
    ) -> Vec<bridge::TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>> {
        stream
            .into_iter()
            .map(|tree| match tree {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    bridge::TokenTree::Ident(IdentId(self.ident_interner.intern(&IdentData(ident))))
                }
                tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => bridge::TokenTree::Literal(lit),
                tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) => bridge::TokenTree::Punct(punct),
                tt::TokenTree::Subtree(subtree) => bridge::TokenTree::Group(subtree),
            })
            .collect()
    }
}

fn delim_to_internal(d: bridge::Delimiter) -> tt::Delimiter {
    let kind = match d {
        bridge::Delimiter::Parenthesis => tt::DelimiterKind::Parenthesis,
        bridge::Delimiter::Brace => tt::DelimiterKind::Brace,
        bridge::Delimiter::Bracket => tt::DelimiterKind::Bracket,
        bridge::Delimiter::None => tt::DelimiterKind::Invisible,
    };
    tt::Delimiter { open: tt::TokenId::unspecified(), close: tt::TokenId::unspecified(), kind }
}

fn delim_to_external(d: tt::Delimiter) -> bridge::Delimiter {
    match d.kind {
        tt::DelimiterKind::Parenthesis => bridge::Delimiter::Parenthesis,
        tt::DelimiterKind::Brace => bridge::Delimiter::Brace,
        tt::DelimiterKind::Bracket => bridge::Delimiter::Bracket,
        tt::DelimiterKind::Invisible => bridge::Delimiter::None,
    }
}

fn spacing_to_internal(spacing: bridge::Spacing) -> Spacing {
    match spacing {
        bridge::Spacing::Alone => Spacing::Alone,
        bridge::Spacing::Joint => Spacing::Joint,
    }
}

fn spacing_to_external(spacing: Spacing) -> bridge::Spacing {
    match spacing {
        Spacing::Alone => bridge::Spacing::Alone,
        Spacing::Joint => bridge::Spacing::Joint,
    }
}

impl server::Group for RustAnalyzer {
    fn new(
        &mut self,
        delimiter: bridge::Delimiter,
        stream: Option<Self::TokenStream>,
    ) -> Self::Group {
        Self::Group {
            delimiter: delim_to_internal(delimiter),
            token_trees: stream.unwrap_or_default().token_trees,
        }
    }
    fn delimiter(&mut self, group: &Self::Group) -> bridge::Delimiter {
        delim_to_external(group.delimiter)
    }

    // NOTE: Return value of do not include delimiter
    fn stream(&mut self, group: &Self::Group) -> Self::TokenStream {
        TokenStream { token_trees: group.token_trees.clone() }
    }

    fn span(&mut self, group: &Self::Group) -> Self::Span {
        group.delimiter.open
    }

    fn set_span(&mut self, group: &mut Self::Group, span: Self::Span) {
        group.delimiter.open = span;
    }

    fn span_open(&mut self, group: &Self::Group) -> Self::Span {
        group.delimiter.open
    }

    fn span_close(&mut self, group: &Self::Group) -> Self::Span {
        group.delimiter.close
    }
}

impl server::Punct for RustAnalyzer {
    fn new(&mut self, ch: char, spacing: bridge::Spacing) -> Self::Punct {
        tt::Punct {
            char: ch,
            spacing: spacing_to_internal(spacing),
            span: tt::TokenId::unspecified(),
        }
    }
    fn as_char(&mut self, punct: Self::Punct) -> char {
        punct.char
    }
    fn spacing(&mut self, punct: Self::Punct) -> bridge::Spacing {
        spacing_to_external(punct.spacing)
    }
    fn span(&mut self, punct: Self::Punct) -> Self::Span {
        punct.span
    }
    fn with_span(&mut self, punct: Self::Punct, span: Self::Span) -> Self::Punct {
        tt::Punct { span: span, ..punct }
    }
}

impl server::Ident for RustAnalyzer {
    fn new(&mut self, string: &str, span: Self::Span, is_raw: bool) -> Self::Ident {
        IdentId(self.ident_interner.intern(&IdentData(tt::Ident {
            text: if is_raw { ::tt::SmolStr::from_iter(["r#", string]) } else { string.into() },
            span,
        })))
    }

    fn span(&mut self, ident: Self::Ident) -> Self::Span {
        self.ident_interner.get(ident.0).0.span
    }
    fn with_span(&mut self, ident: Self::Ident, span: Self::Span) -> Self::Ident {
        let data = self.ident_interner.get(ident.0);
        let new = IdentData(tt::Ident { span: span, ..data.0.clone() });
        IdentId(self.ident_interner.intern(&new))
    }
}

impl server::Literal for RustAnalyzer {
    fn debug_kind(&mut self, _literal: &Self::Literal) -> String {
        // r-a: debug_kind and suffix are unsupported; corresponding client code has been changed to not call these.
        // They must still be present to be ABI-compatible and work with upstream proc_macro.
        "".to_owned()
    }
    fn from_str(&mut self, s: &str) -> Result<Self::Literal, ()> {
        Ok(Literal { text: s.into(), span: tt::TokenId::unspecified() })
    }
    fn symbol(&mut self, literal: &Self::Literal) -> String {
        literal.text.to_string()
    }
    fn suffix(&mut self, _literal: &Self::Literal) -> Option<String> {
        None
    }

    fn to_string(&mut self, literal: &Self::Literal) -> String {
        literal.to_string()
    }

    fn integer(&mut self, n: &str) -> Self::Literal {
        let n = match n.parse::<i128>() {
            Ok(n) => n.to_string(),
            Err(_) => n.parse::<u128>().unwrap().to_string(),
        };
        Literal { text: n.into(), span: tt::TokenId::unspecified() }
    }

    fn typed_integer(&mut self, n: &str, kind: &str) -> Self::Literal {
        macro_rules! def_suffixed_integer {
            ($kind:ident, $($ty:ty),*) => {
                match $kind {
                    $(
                        stringify!($ty) => {
                            let n: $ty = n.parse().unwrap();
                            format!(concat!("{}", stringify!($ty)), n)
                        }
                    )*
                    _ => unimplemented!("unknown args for typed_integer: n {}, kind {}", n, $kind),
                }
            }
        }

        let text = def_suffixed_integer! {kind, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize};

        Literal { text: text.into(), span: tt::TokenId::unspecified() }
    }

    fn float(&mut self, n: &str) -> Self::Literal {
        let n: f64 = n.parse().unwrap();
        let mut text = f64::to_string(&n);
        if !text.contains('.') {
            text += ".0"
        }
        Literal { text: text.into(), span: tt::TokenId::unspecified() }
    }

    fn f32(&mut self, n: &str) -> Self::Literal {
        let n: f32 = n.parse().unwrap();
        let text = format!("{n}f32");
        Literal { text: text.into(), span: tt::TokenId::unspecified() }
    }

    fn f64(&mut self, n: &str) -> Self::Literal {
        let n: f64 = n.parse().unwrap();
        let text = format!("{n}f64");
        Literal { text: text.into(), span: tt::TokenId::unspecified() }
    }

    fn string(&mut self, string: &str) -> Self::Literal {
        let mut escaped = String::new();
        for ch in string.chars() {
            escaped.extend(ch.escape_debug());
        }
        Literal { text: format!("\"{escaped}\"").into(), span: tt::TokenId::unspecified() }
    }

    fn character(&mut self, ch: char) -> Self::Literal {
        Literal { text: format!("'{ch}'").into(), span: tt::TokenId::unspecified() }
    }

    fn byte_string(&mut self, bytes: &[u8]) -> Self::Literal {
        let string = bytes
            .iter()
            .cloned()
            .flat_map(ascii::escape_default)
            .map(Into::<char>::into)
            .collect::<String>();

        Literal { text: format!("b\"{string}\"").into(), span: tt::TokenId::unspecified() }
    }

    fn span(&mut self, literal: &Self::Literal) -> Self::Span {
        literal.span
    }

    fn set_span(&mut self, literal: &mut Self::Literal, span: Self::Span) {
        literal.span = span;
    }

    fn subspan(
        &mut self,
        _literal: &Self::Literal,
        _start: Bound<usize>,
        _end: Bound<usize>,
    ) -> Option<Self::Span> {
        // FIXME handle span
        None
    }
}

impl server::SourceFile for RustAnalyzer {
    // FIXME these are all stubs
    fn eq(&mut self, _file1: &Self::SourceFile, _file2: &Self::SourceFile) -> bool {
        true
    }
    fn path(&mut self, _file: &Self::SourceFile) -> String {
        String::new()
    }
    fn is_real(&mut self, _file: &Self::SourceFile) -> bool {
        true
    }
}

impl server::Diagnostic for RustAnalyzer {
    fn new(&mut self, level: Level, msg: &str, spans: Self::MultiSpan) -> Self::Diagnostic {
        let mut diag = Diagnostic::new(level, msg);
        diag.spans = spans;
        diag
    }

    fn sub(
        &mut self,
        _diag: &mut Self::Diagnostic,
        _level: Level,
        _msg: &str,
        _spans: Self::MultiSpan,
    ) {
        // FIXME handle diagnostic
        //
    }

    fn emit(&mut self, _diag: Self::Diagnostic) {
        // FIXME handle diagnostic
        // diag.emit()
    }
}

impl server::Span for RustAnalyzer {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span.0)
    }
    fn def_site(&mut self) -> Self::Span {
        // MySpan(self.span_interner.intern(&MySpanData(Span::def_site())))
        // FIXME handle span
        tt::TokenId::unspecified()
    }
    fn call_site(&mut self) -> Self::Span {
        // MySpan(self.span_interner.intern(&MySpanData(Span::call_site())))
        // FIXME handle span
        tt::TokenId::unspecified()
    }
    fn source_file(&mut self, _span: Self::Span) -> Self::SourceFile {
        SourceFile {}
    }
    fn save_span(&mut self, _span: Self::Span) -> usize {
        // FIXME stub
        0
    }
    fn recover_proc_macro_span(&mut self, _id: usize) -> Self::Span {
        // FIXME stub
        tt::TokenId::unspecified()
    }
    /// Recent feature, not yet in the proc_macro
    ///
    /// See PR:
    /// https://github.com/rust-lang/rust/pull/55780
    fn source_text(&mut self, _span: Self::Span) -> Option<String> {
        None
    }

    fn parent(&mut self, _span: Self::Span) -> Option<Self::Span> {
        // FIXME handle span
        None
    }
    fn source(&mut self, span: Self::Span) -> Self::Span {
        // FIXME handle span
        span
    }
    fn start(&mut self, _span: Self::Span) -> LineColumn {
        // FIXME handle span
        LineColumn { line: 0, column: 0 }
    }
    fn end(&mut self, _span: Self::Span) -> LineColumn {
        // FIXME handle span
        LineColumn { line: 0, column: 0 }
    }
    fn join(&mut self, first: Self::Span, _second: Self::Span) -> Option<Self::Span> {
        // Just return the first span again, because some macros will unwrap the result.
        Some(first)
    }
    fn resolved_at(&mut self, _span: Self::Span, _at: Self::Span) -> Self::Span {
        // FIXME handle span
        tt::TokenId::unspecified()
    }

    fn mixed_site(&mut self) -> Self::Span {
        // FIXME handle span
        tt::TokenId::unspecified()
    }

    fn after(&mut self, _self_: Self::Span) -> Self::Span {
        tt::TokenId::unspecified()
    }

    fn before(&mut self, _self_: Self::Span) -> Self::Span {
        tt::TokenId::unspecified()
    }
}

impl server::MultiSpan for RustAnalyzer {
    fn new(&mut self) -> Self::MultiSpan {
        // FIXME handle span
        vec![]
    }

    fn push(&mut self, other: &mut Self::MultiSpan, span: Self::Span) {
        //TODP
        other.push(span)
    }
}

#[cfg(test)]
mod tests {
    use super::super::proc_macro::bridge::server::Literal;
    use super::*;

    #[test]
    fn test_ra_server_literals() {
        let mut srv = RustAnalyzer { ident_interner: IdentInterner::default() };
        assert_eq!(srv.integer("1234").text, "1234");

        assert_eq!(srv.typed_integer("12", "u8").text, "12u8");
        assert_eq!(srv.typed_integer("255", "u16").text, "255u16");
        assert_eq!(srv.typed_integer("1234", "u32").text, "1234u32");
        assert_eq!(srv.typed_integer("15846685", "u64").text, "15846685u64");
        assert_eq!(srv.typed_integer("15846685258", "u128").text, "15846685258u128");
        assert_eq!(srv.typed_integer("156788984", "usize").text, "156788984usize");
        assert_eq!(srv.typed_integer("127", "i8").text, "127i8");
        assert_eq!(srv.typed_integer("255", "i16").text, "255i16");
        assert_eq!(srv.typed_integer("1234", "i32").text, "1234i32");
        assert_eq!(srv.typed_integer("15846685", "i64").text, "15846685i64");
        assert_eq!(srv.typed_integer("15846685258", "i128").text, "15846685258i128");
        assert_eq!(srv.float("0").text, "0.0");
        assert_eq!(srv.float("15684.5867").text, "15684.5867");
        assert_eq!(srv.f32("15684.58").text, "15684.58f32");
        assert_eq!(srv.f64("15684.58").text, "15684.58f64");

        assert_eq!(srv.string("hello_world").text, "\"hello_world\"");
        assert_eq!(srv.character('c').text, "'c'");
        assert_eq!(srv.byte_string(b"1234586\x88").text, "b\"1234586\\x88\"");

        // u128::max
        assert_eq!(
            srv.integer("340282366920938463463374607431768211455").text,
            "340282366920938463463374607431768211455"
        );
        // i128::min
        assert_eq!(
            srv.integer("-170141183460469231731687303715884105728").text,
            "-170141183460469231731687303715884105728"
        );
    }

    #[test]
    fn test_ra_server_to_string() {
        let s = TokenStream {
            token_trees: vec![
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    text: "struct".into(),
                    span: tt::TokenId::unspecified(),
                })),
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    text: "T".into(),
                    span: tt::TokenId::unspecified(),
                })),
                tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter {
                        open: tt::TokenId::unspecified(),
                        close: tt::TokenId::unspecified(),
                        kind: tt::DelimiterKind::Brace,
                    },
                    token_trees: vec![],
                }),
            ],
        };

        assert_eq!(s.to_string(), "struct T {}");
    }

    #[test]
    fn test_ra_server_from_str() {
        use std::str::FromStr;
        let subtree_paren_a = tt::TokenTree::Subtree(tt::Subtree {
            delimiter: tt::Delimiter {
                open: tt::TokenId::unspecified(),
                close: tt::TokenId::unspecified(),
                kind: tt::DelimiterKind::Parenthesis,
            },
            token_trees: vec![tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                text: "a".into(),
                span: tt::TokenId::unspecified(),
            }))],
        });

        let t1 = TokenStream::from_str("(a)").unwrap();
        assert_eq!(t1.token_trees.len(), 1);
        assert_eq!(t1.token_trees[0], subtree_paren_a);

        let t2 = TokenStream::from_str("(a);").unwrap();
        assert_eq!(t2.token_trees.len(), 2);
        assert_eq!(t2.token_trees[0], subtree_paren_a);

        let underscore = TokenStream::from_str("_").unwrap();
        assert_eq!(
            underscore.token_trees[0],
            tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                text: "_".into(),
                span: tt::TokenId::unspecified(),
            }))
        );
    }
}
