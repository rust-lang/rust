#![warn(warnings)]
use std::cell::{Cell, RefCell};
use std::ops::{Bound, Range, RangeBounds};

use crate::bridge::client::Symbol;
use crate::bridge::fxhash::FxHashMap;
use crate::bridge::{
    self, DelimSpan, Diagnostic, ExpnGlobals, Group, LitKind, Punct, TokenTree, server,
};
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

fn parse_maybe_raw_str(
    mut s: &str,
    raw_variant: fn(u8) -> LitKind,
    regular_variant: LitKind,
) -> Result<Literal> {
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
    }
    let mut hash_count = None;

    if s.starts_with('r') {
        s = s.strip_prefix('r').unwrap();
        let mut h = 0;
        for c in s.chars() {
            if c == '#' {
                if h == u8::MAX {
                    return Err(());
                }
                h += 1;
            } else {
                break;
            }
        }
        hash_count = Some(h);
        let hashes = get_hashes_str(h);
        s = s.strip_prefix(hashes).unwrap();
        s = s.strip_suffix(hashes).ok_or(())?;
    }
    let sym = parse_plain_str(s)?;

    Ok(make_literal(if let Some(h) = hash_count { raw_variant(h) } else { regular_variant }, sym))
}

fn parse_char(s: &str) -> Result<Literal> {
    if s.chars().count() == 1 { Ok(make_literal(LitKind::Char, Symbol::new(s))) } else { Err(()) }
}

fn parse_plain_str(mut s: &str) -> Result<Symbol> {
    s = s.strip_prefix("\"").ok_or(())?.strip_suffix('\"').ok_or(())?;
    Ok(Symbol::new(s))
}

const INT_SUFFIXES: &[&str] =
    &["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "u128", "i128"];
const FLOAT_SUFFIXES: &[&str] = &["f16", "f32", "f64", "f128"];

fn parse_numeral(mut s: &str) -> Result<Literal> {
    for suffix in INT_SUFFIXES {
        if s.ends_with(suffix) {
            return parse_integer(s);
        }
    }
    let is_negative = s.starts_with('-');
    let non_negative = s.strip_prefix('-').unwrap();
    if non_negative.starts_with("0b")
        || non_negative.starts_with("0o")
        || non_negative.starts_with("0x")
    {
        return parse_integer(s);
    }
    let (s, suffix) = strip_number_suffix(s, FLOAT_SUFFIXES);

    Ok(Literal { kind: LitKind::Float, symbol: todo!(), suffix, span: Span })
}

fn parse_integer(mut s: &str) -> Result<Literal> {
    let is_negative = s.starts_with('-');
    s = s.strip_prefix('-').unwrap();

    let (s, valid_chars) = if let Some(s) = s.strip_prefix("0b") {
        (s, '0'..='1')
    } else if let Some(s) = s.strip_prefix("0o") {
        (s, '0'..='7')
    } else if let Some(s) = s.strip_prefix("0x") {
        (s, '0'..='F')
    } else {
        (s, '0'..='9')
    };

    let (s, suffix) = strip_number_suffix(s, INT_SUFFIXES);

    let mut any_found = false;
    for c in s.chars() {
        if c == '_' {
            continue;
        }
        if valid_chars.contains(&c) {
            any_found = true;
            continue;
        }
        return Err(());
    }
    if !any_found {
        return Err(());
    }

    Ok(Literal { kind: LitKind::Integer, symbol: Symbol::new(s), suffix, span: Span })
}

fn strip_number_suffix<'a>(s: &'a str, suffixes: &[&str]) -> (&'a str, Option<Symbol>) {
    for suf in suffixes {
        if let Some(new_s) = s.strip_suffix(suf) {
            return (new_s, Some(Symbol::new(suf)));
        }
    }
    (s, None)
}

fn make_literal(kind: LitKind, symbol: Symbol) -> Literal {
    Literal { kind, symbol, suffix: None, span: Span }
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

    fn literal_from_str(&mut self, s: &str) -> Result<Literal> {
        let mut chars = s.chars();
        let Some(first) = chars.next() else {
            return Err(());
        };
        let rest = &s[1..];

        match first {
            'b' => {
                if chars.next() == Some('\'') {
                    parse_char(rest).map(|mut lit| {
                        lit.kind = LitKind::Byte;
                        lit
                    })
                } else {
                    parse_maybe_raw_str(rest, LitKind::ByteStrRaw, LitKind::ByteStr)
                }
            }
            'c' => parse_maybe_raw_str(rest, LitKind::CStrRaw, LitKind::CStr),
            'r' => parse_maybe_raw_str(rest, LitKind::StrRaw, LitKind::Str),
            '0'..='9' | '-' => parse_numeral(s),
            '\'' => parse_char(s),
            '"' => Ok(make_literal(LitKind::Str, parse_plain_str(s)?)),
            _ => Err(()),
        }
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
                    joint: todo!(),
                    span: Span,
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
        todo!()
    }
}
