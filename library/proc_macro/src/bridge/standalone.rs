#![allow(unused_variables)]
#![warn(warnings)]
use std::cell::RefCell;
use std::ops::{Bound, Range};

use crate::{Delimiter, LEGAL_PUNCT_CHARS};
use crate::bridge::client::Symbol;
use crate::bridge::fxhash::FxHashMap;
use crate::bridge::{server, DelimSpan, Diagnostic, ExpnGlobals, Group, LitKind, Literal, Punct, TokenTree};

pub struct NoRustc;

impl server::Span for NoRustc {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{} bytes({}..{})", span.hi - span.lo, span.lo, span.hi)
    }

    fn parent(&mut self, span: Self::Span) -> Option<Self::Span> {
        todo!()
    }

    fn source(&mut self, span: Self::Span) -> Self::Span {
        todo!()
    }

    fn byte_range(&mut self, span: Self::Span) -> Range<usize> {
        span.lo as usize..span.hi as usize
    }

    fn start(&mut self, span: Self::Span) -> Self::Span {
        Span { lo: span.lo, hi: span.lo }
    }

    fn end(&mut self, span: Self::Span) -> Self::Span {
        Span { lo: span.hi, hi: span.hi }
    }

    fn line(&mut self, span: Self::Span) -> usize {
        todo!()
    }

    fn column(&mut self, span: Self::Span) -> usize {
        todo!()
    }

    fn file(&mut self, span: Self::Span) -> String {
        todo!()
    }

    fn local_file(&mut self, span: Self::Span) -> Option<String> {
        todo!()
    }

    fn join(&mut self, span: Self::Span, other: Self::Span) -> Option<Self::Span> {
        todo!()
    }

    fn subspan(
        &mut self,
        span: Self::Span,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
        let length = span.hi as usize - span.lo as usize;

        let start = match start {
            Bound::Included(lo) => lo,
            Bound::Excluded(lo) => lo.checked_add(1)?,
            Bound::Unbounded => 0,
        };

        let end = match end {
            Bound::Included(hi) => hi.checked_add(1)?,
            Bound::Excluded(hi) => hi,
            Bound::Unbounded => length,
        };

        // Bounds check the values, preventing addition overflow and OOB spans.
        if start > u32::MAX as usize
            || end > u32::MAX as usize
            || (u32::MAX - start as u32) < span.lo
            || (u32::MAX - end as u32) < span.lo
            || start >= end
            || end > length
        {
            return None;
        }

        let new_lo = span.lo + start as u32;
        let new_hi = span.lo + end as u32;
        Some(Span { lo: new_lo, hi: new_hi })
    }

    fn resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        todo!()
    }

    fn source_text(&mut self, span: Self::Span) -> Option<String> {
        todo!()
    }

    fn save_span(&mut self, span: Self::Span) -> usize {
        SAVED_SPANS.with_borrow_mut(|spans| {
            let idx = spans.len();
            spans.push(span);
            idx
        })
    }

    fn recover_proc_macro_span(&mut self, id: usize) -> Self::Span {
        SAVED_SPANS.with_borrow(|spans| spans[id])
    }
}

thread_local! {
    static SAVED_SPANS: RefCell<Vec<Span>> = const { RefCell::new(Vec::new()) };
    static TRACKED_ENV_VARS: RefCell<FxHashMap<String, Option<String>>> = RefCell::new(FxHashMap::default());
}

impl server::FreeFunctions for NoRustc {
    fn injected_env_var(&mut self, var: &str) -> Option<String> {
        TRACKED_ENV_VARS.with_borrow(|vars| vars.get(var)?.clone())
    }

    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        TRACKED_ENV_VARS
            .with_borrow_mut(|vars| vars.insert(var.to_string(), value.map(ToString::to_string)));
    }

    fn track_path(&mut self, _path: &str) {}

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span, Self::Symbol>, ()> {
        let mut chars = s.chars();
        let Some(first) = chars.next() else {
            return Err(());
        };
        br"";
        cr"";

        match first {
            'b' => todo!(),
            'c' => todo!(),
            'r' => todo!(),
            '0'..='9' | '-' => todo!(),
            '\'' => todo!(),
            '"' => todo!(),
            _ => Err(())
        }
    }

    fn emit_diagnostic(&mut self, diagnostic: Diagnostic<Self::Span>) {
        panic!("cannot emit diagnostic in standalone mode");
    }
}

impl server::TokenStream for NoRustc {
    fn is_empty(&mut self, tokens: &Self::TokenStream) -> bool {
        tokens.0.is_empty()
    }

    fn expand_expr(&mut self, tokens: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        todo!()
    }

    fn from_str(&mut self, src: &str) -> Self::TokenStream {
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
                        span: DelimSpan::from_single(Span::DUMMY)
                    });
                    unfinished_streams.last_mut().unwrap().0.push(group);
                } else {
                    panic!("cannot parse string into token stream")
                }
            } else if LEGAL_PUNCT_CHARS.contains(&c) {
                unfinished_streams.last_mut().unwrap().0.push(TokenTree::Punct(Punct {
                    ch: c as u8,
                    joint: false, // TODO
                    span: Span::DUMMY,
                }));
            }
            match c {
                _ => todo!(),
            }
        }
        unfinished_streams[0].clone()
    }

    fn to_string(&mut self, tokens: &Self::TokenStream) -> String {
        /*
        /// Returns a string containing exactly `num` '#' characters.
        /// Uses a 256-character source string literal which is always safe to
        /// index with a `u8` index.
        fn get_hashes_str(num: u8) -> &'static str {
            const HASHES: &str = "\
            ################################################################\
            ################################################################\
            ################################################################\
            ################################################################\
            ";
            const _: () = assert!(HASHES.len() == 256);
            &HASHES[..num as usize]
        }*/

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
                    let respanned = Literal {
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
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}
impl Span {
    pub const DUMMY: Self = Self { lo: 0, hi: 0 };
}

impl server::Types for NoRustc {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::Server for NoRustc {
    fn globals(&mut self) -> ExpnGlobals<Self::Span> {
        ExpnGlobals { def_site: Span::DUMMY, call_site: Span::DUMMY, mixed_site: Span::DUMMY }
    }

    fn intern_symbol(ident: &str) -> Self::Symbol {
        Symbol::new(ident)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        symbol.with(f);
    }
}

impl server::Symbol for NoRustc {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        todo!()
    }
}
