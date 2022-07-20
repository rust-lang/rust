//! proc-macro server implementation
//!
//! Based on idea from <https://github.com/fedochet/rust-proc-macro-expander>
//! The lib-proc-macro server backend is `TokenStream`-agnostic, such that
//! we could provide any TokenStream implementation.
//! The original idea from fedochet is using proc-macro2 as backend,
//! we use tt instead for better integration with RA.
//!
//! FIXME: No span and source file information is implemented yet

use super::proc_macro::bridge::{self, server};

mod token_stream;
pub use token_stream::*;

use std::ascii;
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::Bound;

type Group = tt::Subtree;
type TokenTree = tt::TokenTree;
type Punct = tt::Punct;
type Spacing = tt::Spacing;
type Literal = tt::Literal;
type Span = tt::TokenId;

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

pub struct FreeFunctions;

#[derive(Default)]
pub struct RustAnalyzer {
    // FIXME: store span information here.
}

impl server::Types for RustAnalyzer {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type SourceFile = SourceFile;
    type MultiSpan = Vec<Span>;
    type Diagnostic = Diagnostic;
    type Span = Span;
    type Symbol = Symbol;
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
        tree: bridge::TokenTree<Self::TokenStream, Self::Span, Self::Ident, Self::Literal>,
    ) -> Self::TokenStream {
        match tree {
            bridge::TokenTree::Group(group) => {
                let group = Group {
                    delimiter: delim_to_internal(group.delimiter),
                    token_trees: match group.stream {
                        Some(stream) => stream.into_iter().collect(),
                        None => Vec::new(),
                    },
                };
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
                let punct = tt::Punct {
                    char: p.ch as char,
                    spacing: if p.joint { Spacing::Joint } else { Spacing::Alone },
                    id: p.span,
                };
                let leaf = tt::Leaf::from(punct);
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
        trees: Vec<bridge::TokenTree<Self::TokenStream, Self::Span, Self::Ident, Self::Literal>>,
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
    ) -> Vec<bridge::TokenTree<Self::TokenStream, Self::Span, Self::Ident, Self::Literal>> {
        stream
            .into_iter()
            .map(|tree| match tree {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    bridge::TokenTree::Ident(IdentId(self.ident_interner.intern(&IdentData(ident))))
                }
                tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => bridge::TokenTree::Literal(lit),
                tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) => {
                    bridge::TokenTree::Punct(bridge::Punct {
                        ch: punct.char as u8,
                        joint: punct.spacing == Spacing::Joint,
                        span: punct.id,
                    })
                }
                tt::TokenTree::Subtree(subtree) => bridge::TokenTree::Group(bridge::Group {
                    delimiter: delim_to_external(subtree.delimiter),
                    stream: if subtree.token_trees.is_empty() {
                        None
                    } else {
                        Some(subtree.token_trees.into_iter().collect())
                    },
                    span: bridge::DelimSpan::from_single(
                        subtree.delimiter.map_or(Span::unspecified(), |del| del.id),
                    ),
                }),
            })
            .collect()
    }
}

fn delim_to_internal(d: bridge::Delimiter) -> Option<tt::Delimiter> {
    let kind = match d {
        bridge::Delimiter::Parenthesis => tt::DelimiterKind::Parenthesis,
        bridge::Delimiter::Brace => tt::DelimiterKind::Brace,
        bridge::Delimiter::Bracket => tt::DelimiterKind::Bracket,
        bridge::Delimiter::None => return None,
    };
    Some(tt::Delimiter { id: tt::TokenId::unspecified(), kind })
}

fn delim_to_external(d: Option<tt::Delimiter>) -> bridge::Delimiter {
    match d.map(|it| it.kind) {
        Some(tt::DelimiterKind::Parenthesis) => bridge::Delimiter::Parenthesis,
        Some(tt::DelimiterKind::Brace) => bridge::Delimiter::Brace,
        Some(tt::DelimiterKind::Bracket) => bridge::Delimiter::Bracket,
        None => bridge::Delimiter::None,
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

impl server::Ident for RustAnalyzer {
    fn new(&mut self, string: &str, span: Self::Span, _is_raw: bool) -> Self::Ident {
        IdentId(self.ident_interner.intern(&IdentData(tt::Ident { text: string.into(), id: span })))
    }

    fn span(&mut self, ident: Self::Ident) -> Self::Span {
        self.ident_interner.get(ident.0).0.id
    }
    fn with_span(&mut self, ident: Self::Ident, span: Self::Span) -> Self::Ident {
        let data = self.ident_interner.get(ident.0);
        let new = IdentData(tt::Ident { id: span, ..data.0.clone() });
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
        Ok(Literal { text: s.into(), id: tt::TokenId::unspecified() })
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
        Literal { text: n.into(), id: tt::TokenId::unspecified() }
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

        Literal { text: text.into(), id: tt::TokenId::unspecified() }
    }

    fn float(&mut self, n: &str) -> Self::Literal {
        let n: f64 = n.parse().unwrap();
        let mut text = f64::to_string(&n);
        if !text.contains('.') {
            text += ".0"
        }
        Literal { text: text.into(), id: tt::TokenId::unspecified() }
    }

    fn f32(&mut self, n: &str) -> Self::Literal {
        let n: f32 = n.parse().unwrap();
        let text = format!("{}f32", n);
        Literal { text: text.into(), id: tt::TokenId::unspecified() }
    }

    fn f64(&mut self, n: &str) -> Self::Literal {
        let n: f64 = n.parse().unwrap();
        let text = format!("{}f64", n);
        Literal { text: text.into(), id: tt::TokenId::unspecified() }
    }

    fn string(&mut self, string: &str) -> Self::Literal {
        let mut escaped = String::new();
        for ch in string.chars() {
            escaped.extend(ch.escape_debug());
        }
        Literal { text: format!("\"{}\"", escaped).into(), id: tt::TokenId::unspecified() }
    }

    fn character(&mut self, ch: char) -> Self::Literal {
        Literal { text: format!("'{}'", ch).into(), id: tt::TokenId::unspecified() }
    }

    fn byte_string(&mut self, bytes: &[u8]) -> Self::Literal {
        let string = bytes
            .iter()
            .cloned()
            .flat_map(ascii::escape_default)
            .map(Into::<char>::into)
            .collect::<String>();

        Literal { text: format!("b\"{}\"", string).into(), id: tt::TokenId::unspecified() }
    }

    fn span(&mut self, literal: &Self::Literal) -> Self::Span {
        literal.id
    }

    fn set_span(&mut self, literal: &mut Self::Literal, span: Self::Span) {
        literal.id = span;
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

impl server::Server for RustAnalyzer {
    fn globals(&mut self) -> bridge::ExpnGlobals<Self::Span> {
        bridge::ExpnGlobals {
            def_site: Span::unspecified(),
            call_site: Span::unspecified(),
            mixed_site: Span::unspecified(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ra_server_to_string() {
        let s = TokenStream {
            token_trees: vec![
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    text: "struct".into(),
                    id: tt::TokenId::unspecified(),
                })),
                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    text: "T".into(),
                    id: tt::TokenId::unspecified(),
                })),
                tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: Some(tt::Delimiter {
                        id: tt::TokenId::unspecified(),
                        kind: tt::DelimiterKind::Brace,
                    }),
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
            delimiter: Some(tt::Delimiter {
                id: tt::TokenId::unspecified(),
                kind: tt::DelimiterKind::Parenthesis,
            }),
            token_trees: vec![tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                text: "a".into(),
                id: tt::TokenId::unspecified(),
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
                id: tt::TokenId::unspecified(),
            }))
        );
    }
}
