mod parsing;

use std::cell::{Cell, RefCell};
use std::ops::{Bound, Range};

use crate::bridge::client::Symbol;
use crate::bridge::fxhash::FxHashMap;
use crate::bridge::{self, DelimSpan, Diagnostic, ExpnGlobals, Group, Punct, TokenTree, server};
use crate::{Delimiter, LEGAL_PUNCT_CHARS};

type Result<T> = std::result::Result<T, ()>;
type Literal = bridge::Literal<Span, Symbol>;

pub struct NoRustc;

impl server::Span for NoRustc {
    fn debug(&mut self, _: Self::Span) -> String {
        "Span".to_string()
    }

    fn parent(&mut self, _: Self::Span) -> Option<Self::Span> {
        None
    }

    fn source(&mut self, _: Self::Span) -> Self::Span {
        Span
    }

    fn byte_range(&mut self, _: Self::Span) -> Range<usize> {
        0..0
    }

    fn start(&mut self, _: Self::Span) -> Self::Span {
        Span
    }

    fn end(&mut self, _: Self::Span) -> Self::Span {
        Span
    }

    fn line(&mut self, _: Self::Span) -> usize {
        1
    }

    fn column(&mut self, _: Self::Span) -> usize {
        1
    }

    fn file(&mut self, _: Self::Span) -> String {
        "<anon>".to_string()
    }

    fn local_file(&mut self, _: Self::Span) -> Option<String> {
        None
    }

    fn join(&mut self, _: Self::Span, _: Self::Span) -> Option<Self::Span> {
        Some(Span)
    }

    fn subspan(
        &mut self,
        _: Self::Span,
        _start: Bound<usize>,
        _end: Bound<usize>,
    ) -> Option<Self::Span> {
        Some(Span)
    }

    fn resolved_at(&mut self, _span: Self::Span, _at: Self::Span) -> Self::Span {
        Span
    }

    fn source_text(&mut self, _: Self::Span) -> Option<String> {
        None
    }

    fn save_span(&mut self, _: Self::Span) -> usize {
        let n = SAVED_SPAN_COUNT.get();
        SAVED_SPAN_COUNT.set(n + 1);
        n
    }

    fn recover_proc_macro_span(&mut self, id: usize) -> Self::Span {
        if id < SAVED_SPAN_COUNT.get() {
            Span
        } else {
            panic!("recovered span index out of bounds");
        }
    }
}

thread_local! {
    static SAVED_SPAN_COUNT: Cell<usize> = const { Cell::new(0) };
    static TRACKED_ENV_VARS: RefCell<FxHashMap<String, Option<String>>> = RefCell::new(FxHashMap::default());
}

impl server::FreeFunctions for NoRustc {
    fn injected_env_var(&mut self, var: &str) -> Option<String> {
        0xf32;
        TRACKED_ENV_VARS.with_borrow(|vars| vars.get(var)?.clone())
    }

    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        TRACKED_ENV_VARS
            .with_borrow_mut(|vars| vars.insert(var.to_string(), value.map(ToString::to_string)));
    }

    fn track_path(&mut self, _path: &str) {}

    fn literal_from_str(&mut self, s: &str) -> Result<Literal> {
        parsing::literal_from_str(s)
    }

    fn emit_diagnostic(&mut self, _: Diagnostic<Self::Span>) {
        panic!("cannot emit diagnostic in standalone mode");
    }
}

impl server::TokenStream for NoRustc {
    fn is_empty(&mut self, tokens: &Self::TokenStream) -> bool {
        tokens.0.is_empty()
    }

    fn expand_expr(&mut self, _tokens: &Self::TokenStream) -> Result<Self::TokenStream> {
        todo!("`expand_expr` is not yet supported in the standalone backend")
    }

    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        return TokenStream::new();

        /// Returns the delimiter, and whether it is the opening form.
        fn char_to_delim(c: char) -> Option<(Delimiter, bool)> {
            Some(match c {
                '(' => (Delimiter::Parenthesis, true),
                ')' => (Delimiter::Parenthesis, false),
                '{' => (Delimiter::Brace, true),
                '}' => (Delimiter::Brace, false),
                '[' => (Delimiter::Bracket, true),
                ']' => (Delimiter::Bracket, false),
                _ => return None,
            })
        }

        let mut unfinished_streams = vec![TokenStream::new()];
        let mut unclosed_delimiters = Vec::new();
        let mut current_ident = String::new();
        for c in src.chars() {
            if let Some((delim, is_opening)) = char_to_delim(c) {
                if is_opening {
                    unclosed_delimiters.push(delim);
                    unfinished_streams.push(TokenStream::new());
                } else if unclosed_delimiters.pop() == Some(delim) {
                    let group = TokenTree::<_, _, Symbol>::Group(Group {
                        delimiter: delim,
                        stream: unfinished_streams.pop(),
                        span: DelimSpan::from_single(Span),
                    });
                    unfinished_streams.last_mut().unwrap().0.push(group);
                } else {
                    panic!("cannot parse string into token stream")
                }
            } else if LEGAL_PUNCT_CHARS.contains(&c) {
                unfinished_streams.last_mut().unwrap().0.push(TokenTree::Punct(Punct {
                    ch: c as u8,
                    joint: true, // fix this
                    span: Span,
                }));
            }

            // more cases
        }
        unfinished_streams[0].clone()
    }

    fn to_string(&mut self, tokens: &Self::TokenStream) -> String {
        let mut s = String::new();
        let mut last = String::new();
        let mut second_last = String::new();

        for (idx, tree) in tokens.0.iter().enumerate() {
            let mut space = true;
            let new_part = match tree {
                TokenTree::Group(group) => {
                    let inner = if let Some(stream) = &group.stream {
                        self.to_string(stream)
                    } else {
                        String::new()
                    };
                    match group.delimiter {
                        Delimiter::Parenthesis => format!("({inner})"),
                        Delimiter::Brace => {
                            if inner.is_empty() {
                                "{ }".to_string()
                            } else {
                                format!("{{ {inner} }}")
                            }
                        }
                        Delimiter::Bracket => format!("[{inner}]"),
                        Delimiter::None => inner,
                    }
                }
                TokenTree::Ident(ident) => {
                    if ident.is_raw {
                        format!("r#{}", ident.sym)
                    } else {
                        ident.sym.to_string()
                    }
                }
                TokenTree::Literal(lit) => {
                    let respanned = bridge::Literal {
                        kind: lit.kind,
                        symbol: lit.symbol,
                        suffix: lit.suffix,
                        span: super::client::Span::dummy(),
                    };
                    crate::Literal(respanned).to_string()
                    /*let inner = if let Some(suffix) = lit.suffix {
                        format!("{}{suffix}", lit.symbol)
                    } else {
                        lit.symbol.to_string()
                    };
                    match lit.kind {
                        LitKind::Byte => format!("b'{inner}'"),
                        LitKind::ByteStr => format!("b\"{inner}\""),
                        LitKind::ByteStrRaw(raw) => {
                            format!("br{0}\"{inner}\"{0}", get_hashes_str(raw))
                        }
                        LitKind::CStr => format!("c\"{inner}\""),
                        LitKind::CStrRaw(raw) => {
                            format!("cr{0}\"{inner}\"{0}", get_hashes_str(raw))
                        }
                        LitKind::Char => format!("'{inner}'"),
                        LitKind::ErrWithGuar => unreachable!(),
                        LitKind::Float | LitKind::Integer => inner,
                        LitKind::Str => format!("\"{inner}\""),
                        LitKind::StrRaw(raw) => format!("r{0}\"{inner}\"{0}", get_hashes_str(raw)),
                    }*/
                }
                TokenTree::Punct(punct) => {
                    let c = punct.ch as char;
                    if c == '\'' {
                        space = false;
                    }
                    c.to_string()
                }
            };

            const NON_SEPARATABLE_TOKENS: &[(char, char)] = &[(':', ':'), ('-', '>'), ('=', '>')];

            for (first, second) in NON_SEPARATABLE_TOKENS {
                if second_last == first.to_string() && last == second.to_string() && new_part != ":"
                {
                    s.pop(); // pop ' '
                    s.pop(); // pop `second`
                    s.pop(); // pop ' '
                    s.push(*second);
                    s.push(' ');
                }
            }
            s.push_str(&new_part);
            second_last = last;
            last = new_part;
            if space && idx + 1 != tokens.0.len() {
                s.push(' ');
            }
        }
        s
    }

    fn from_token_tree(
        &mut self,
        tree: TokenTree<Self::TokenStream, Self::Span, Self::Symbol>,
    ) -> Self::TokenStream {
        TokenStream(vec![tree])
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>>,
    ) -> Self::TokenStream {
        let mut base = base.unwrap_or_else(TokenStream::new);
        base.0.extend(trees);
        base
    }

    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut base = base.unwrap_or_else(TokenStream::new);
        for stream in streams {
            base = self.concat_trees(Some(base), stream.0);
        }
        base
    }

    fn into_trees(
        &mut self,
        tokens: Self::TokenStream,
    ) -> Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>> {
        tokens.0
    }
}

pub struct FreeFunctions;

#[derive(Clone, Default)]
pub struct TokenStream(Vec<TokenTree<TokenStream, Span, Symbol>>);
impl TokenStream {
    fn new() -> Self {
        Self(Vec::new())
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct Span;

impl server::Types for NoRustc {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::Server for NoRustc {
    fn globals(&mut self) -> ExpnGlobals<Self::Span> {
        ExpnGlobals { def_site: Span, call_site: Span, mixed_site: Span }
    }

    fn intern_symbol(ident: &str) -> Self::Symbol {
        Symbol::new(ident)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        symbol.with(f);
    }
}

impl server::Symbol for NoRustc {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol> {
        // FIXME: to properly support this, we need to add `unicode-normalization` and `unicode_ident`
        // as dependencies of this crate; then we can just remove this from the bridge entirely.
        Ok(Symbol::new(string))
    }
}
