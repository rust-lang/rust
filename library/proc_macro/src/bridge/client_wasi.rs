use std::fmt;
use std::ops::Bound;

pub(crate) use crate::bridge::Methods;
pub(crate) use crate::bridge::symbol::Symbol;
use crate::wasi_bindgen::exports::rust_lang::rust::custom_derive as cd;
pub use crate::wasi_bindgen::rust_lang::rust::host;

pub(crate) fn is_available() -> bool {
    // On wasm we currently don't support executing libproc_macro without the bridge (since it's
    // statically imported) -- unclear whether that can change in the future, but for now just
    // always return true.
    true
}

#[derive(Copy, Clone)]
pub(crate) struct Span {
    // Leak the handle because we need this to be Copy, which is stabilized.
    pub(crate) span: &'static host::Span,
}

impl Span {
    pub(crate) fn def_site() -> Span {
        Span { span: Box::leak(Box::new(host::Span::def_site())) }
    }

    pub(crate) fn call_site() -> Span {
        Span { span: Box::leak(Box::new(host::Span::call_site())) }
    }

    pub(crate) fn mixed_site() -> Span {
        Span { span: Box::leak(Box::new(host::Span::mixed_site())) }
    }

    pub(crate) fn start(&self) -> Span {
        Span { span: Box::leak(Box::new(self.span.start())) }
    }

    pub(crate) fn end(&self) -> Span {
        Span { span: Box::leak(Box::new(self.span.end())) }
    }

    pub(crate) fn join(self, other: Span) -> Option<Span> {
        Some(Span { span: Box::leak(Box::new(self.span.join(other.span)?)) })
    }

    pub(crate) fn resolved_at(self, at: Span) -> Span {
        Span { span: Box::leak(Box::new(self.span.resolved_at(at.span))) }
    }

    pub(crate) fn byte_range(self) -> std::ops::Range<usize> {
        self.span.byte_range_start() as usize..self.span.byte_range_end() as usize
    }

    pub(crate) fn parent(&self) -> Option<Span> {
        Some(Span { span: Box::leak(Box::new(self.span.parent()?)) })
    }

    pub(crate) fn source(&self) -> Span {
        Span { span: Box::leak(Box::new(self.span.source())) }
    }
}

fn to_wasm_range(b: Bound<usize>) -> host::RangeBound {
    match b {
        Bound::Included(v) => host::RangeBound { value: v as u64, bound: host::Bound::Included },
        Bound::Excluded(v) => host::RangeBound { value: v as u64, bound: host::Bound::Excluded },
        Bound::Unbounded => host::RangeBound { value: 0, bound: host::Bound::Unbounded },
    }
}

impl PartialEq for Span {
    fn eq(&self, other: &Span) -> bool {
        host::Span::is_same(&self.span, other.span)
    }
}

impl Eq for Span {}

impl From<host::Span> for Span {
    fn from(sp: host::Span) -> Span {
        Span { span: Box::leak(Box::new(sp)) }
    }
}

impl std::hash::Hash for Span {
    fn hash<H: std::hash::Hasher>(&self, _h: &mut H) {
        // Technically an empty impl would also be valid...
        todo!()
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.span.debug())
    }
}

pub(crate) struct TokenStream {
    handle: host::TokenStream,
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.handle.to_string())
    }
}

impl TokenStream {
    fn new() -> Self {
        TokenStream { handle: host::TokenStream::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.handle.is_empty()
    }

    pub(crate) fn expand_expr(&self) -> Result<TokenStream, ()> {
        Ok(Self { handle: self.handle.expand_expr()? })
    }

    pub(crate) fn concat_streams(base: Option<Self>, streams: Vec<TokenStream>) -> TokenStream {
        Self {
            handle: host::TokenStream::concat_streams(
                base.map(|b| b.handle),
                streams.into_iter().map(|s| s.handle).collect(),
            ),
        }
    }

    pub(crate) fn concat_trees(
        base: Option<Self>,
        streams: Vec<super::TokenTree<Self, Span, Symbol>>,
    ) -> TokenStream {
        Self {
            handle: host::TokenStream::concat_trees(
                base.map(|b| b.handle),
                streams.into_iter().map(guest_to_host_tt).collect(),
            ),
        }
    }

    pub(crate) fn into_trees(self) -> Vec<super::TokenTree<Self, Span, Symbol>> {
        self.handle.into_trees().into_iter().map(host_to_guest_tt).collect()
    }
}

fn guest_to_host_tt(tt: super::TokenTree<TokenStream, Span, Symbol>) -> host::TokenTree {
    match tt {
        super::TokenTree::Group(g) => host::TokenTree::Group(host::Group {
            delimiter: match g.delimiter {
                super::Delimiter::Parenthesis => host::Delimiter::Parenthesis,
                super::Delimiter::Brace => host::Delimiter::Brace,
                super::Delimiter::Bracket => host::Delimiter::Bracket,
                super::Delimiter::None => host::Delimiter::None,
            },
            stream: g.stream.map(|s| s.handle),
            span: host::DelimSpan {
                open: g.span.open.span.clone(),
                close: g.span.close.span.clone(),
                entire: g.span.entire.span.clone(),
            },
        }),
        super::TokenTree::Punct(p) => host::TokenTree::Punct(host::Punct {
            ch: p.ch,
            joint: p.joint,
            span: p.span.span.clone(),
        }),
        super::TokenTree::Ident(i) => host::TokenTree::Ident(host::Ident {
            sym: host::Symbol { s: i.sym.to_string() },
            is_raw: i.is_raw,
            span: i.span.span.clone(),
        }),
        super::TokenTree::Literal(l) => host::TokenTree::Literal(host::Literal {
            kind: l.kind.into(),
            span: l.span.span.clone(),
            suffix: l.suffix.map(|s| host::Symbol { s: s.to_string() }),
            symbol: host::Symbol { s: l.symbol.to_string() },
        }),
    }
}

fn host_to_guest_tt(tt: host::TokenTree) -> super::TokenTree<TokenStream, Span, Symbol> {
    match tt {
        host::TokenTree::Group(g) => super::TokenTree::Group(super::Group {
            delimiter: match g.delimiter {
                host::Delimiter::Parenthesis => super::Delimiter::Parenthesis,
                host::Delimiter::Brace => super::Delimiter::Brace,
                host::Delimiter::Bracket => super::Delimiter::Bracket,
                host::Delimiter::None => super::Delimiter::None,
            },
            stream: g.stream.map(|t| TokenStream { handle: t }),
            span: super::DelimSpan {
                open: g.span.open.into(),
                close: g.span.close.into(),
                entire: g.span.entire.into(),
            },
        }),
        host::TokenTree::Punct(p) => super::TokenTree::Punct(super::Punct {
            ch: p.ch,
            joint: p.joint,
            span: Span::from(p.span),
        }),
        host::TokenTree::Ident(i) => super::TokenTree::Ident(super::Ident {
            sym: Symbol::from(i.sym),
            is_raw: i.is_raw,
            span: i.span.into(),
        }),
        host::TokenTree::Literal(l) => super::TokenTree::Literal(super::Literal {
            kind: l.kind.into(),
            span: l.span.into(),
            suffix: l.suffix.map(Symbol::from),
            symbol: Symbol::from(l.symbol),
        }),
    }
}

impl From<super::LitKind> for host::LiteralKind {
    fn from(v: super::LitKind) -> Self {
        match v {
            super::LitKind::Byte => Self::Byte,
            super::LitKind::Char => Self::Char,
            super::LitKind::Integer => Self::Integer,
            super::LitKind::Float => Self::Float,
            super::LitKind::ByteStr => Self::ByteStr,
            super::LitKind::ByteStrRaw(n) => Self::ByteStrRaw(n),
            super::LitKind::Str => Self::Str,
            super::LitKind::StrRaw(n) => Self::StrRaw(n),
            super::LitKind::CStr => Self::CStr,
            super::LitKind::CStrRaw(n) => Self::CStrRaw(n),
            super::LitKind::ErrWithGuar => Self::ErrWithGuar,
        }
    }
}

impl From<host::LiteralKind> for super::LitKind {
    fn from(h: host::LiteralKind) -> Self {
        match h {
            host::LiteralKind::Byte => Self::Byte,
            host::LiteralKind::Char => Self::Char,
            host::LiteralKind::Integer => Self::Integer,
            host::LiteralKind::Float => Self::Float,
            host::LiteralKind::ByteStr => Self::ByteStr,
            host::LiteralKind::ByteStrRaw(n) => Self::ByteStrRaw(n),
            host::LiteralKind::Str => Self::Str,
            host::LiteralKind::StrRaw(n) => Self::StrRaw(n),
            host::LiteralKind::CStr => Self::CStr,
            host::LiteralKind::CStrRaw(n) => Self::CStrRaw(n),
            host::LiteralKind::ErrWithGuar => Self::ErrWithGuar,
        }
    }
}

impl From<host::Symbol> for Symbol {
    fn from(h: host::Symbol) -> Symbol {
        Symbol::new(&h.s)
    }
}

impl Clone for TokenStream {
    fn clone(&self) -> Self {
        Self { handle: self.handle.clone() }
    }
}

impl !Send for TokenStream {}
impl !Sync for TokenStream {}

#[derive(Copy, Clone)]
pub enum Client {
    Expand1 { expand: fn(crate::TokenStream) -> crate::TokenStream },

    Expand2 { expand: fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream },
}

impl Client {
    pub const fn expand1(expand: fn(crate::TokenStream) -> crate::TokenStream) -> Self {
        Self::Expand1 { expand }
    }

    pub const fn expand2(
        expand: fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream,
    ) -> Self {
        Self::Expand2 { expand }
    }
}

impl cd::GuestCustomDerive for Client {
    fn get_kind(&self) -> cd::DeriveKind {
        match self {
            Self::Expand1 { .. } => cd::DeriveKind::Expand1,
            Self::Expand2 { .. } => cd::DeriveKind::Expand2,
        }
    }

    fn expand1(&self, tokens: host::TokenStream) -> host::TokenStream {
        setup_hook();
        match self {
            Self::Expand1 { expand } => {
                let output = (*expand)(crate::TokenStream(Some(TokenStream { handle: tokens })));
                let output = output.0.unwrap_or_else(|| TokenStream::new());
                output.handle
            }
            Self::Expand2 { .. } => unreachable!("should not be called by host"),
        }
    }

    fn expand2(&self, a: host::TokenStream, b: host::TokenStream) -> host::TokenStream {
        setup_hook();

        match self {
            Self::Expand2 { expand } => {
                let output = (*expand)(
                    crate::TokenStream(Some(TokenStream { handle: a })),
                    crate::TokenStream(Some(TokenStream { handle: b })),
                );
                let output = output.0.unwrap_or_else(|| TokenStream::new());
                output.handle
            }
            Self::Expand1 { .. } => {
                unreachable!("should not be called by host")
            }
        }
    }
}

fn setup_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // If the host wants us to also emit it, run the default panic hook.
            if host::report_panic(info.payload_as_str()) {
                prev(info)
            }
        }));
    });
}

#[allow_internal_unstable(proc_macro_internals, staged_api)]
pub macro generate_export($list:ident) {
    use $crate::wasi_bindgen::exports::rust_lang::rust::custom_derive as cd;

    struct ExportAbi;

    impl cd::Guest for ExportAbi {
        type CustomDerive = $crate::bridge::client::Client;
        fn get_custom_derives() -> Vec<cd::CustomDerive> {
            $list.iter().map(|d| cd::CustomDerive::new(*d)).collect()
        }
    }

    $crate::wasi_bindgen::export!(ExportAbi with_types_in $crate::wasi_bindgen);
}

impl Methods {
    pub(crate) fn literal_from_str(s: &str) -> Result<super::Literal<Span, Symbol>, String> {
        match host::literal_from_str(s) {
            Ok(l) => Ok(super::Literal {
                kind: l.kind.into(),
                symbol: Symbol::from(l.symbol),
                suffix: l.suffix.map(Symbol::from),
                span: Span { span: Box::leak(Box::new(l.span)) },
            }),
            Err(e) => Err(e),
        }
    }

    pub(crate) fn symbol_normalize_and_validate_ident(s: &str) -> Result<Symbol, ()> {
        match host::symbol_normalize_and_validate_ident(s) {
            Ok(s) => Ok(Symbol::from(s)),
            Err(()) => Err(()),
        }
    }

    pub(crate) fn injected_env_var(var: &str) -> Option<String> {
        host::injected_env_var(var)
    }
    pub(crate) fn track_env_var(var: &str, value: Option<&str>) {
        host::track_env_var(var, value)
    }
    pub(crate) fn track_path(path: &str) {
        host::track_path(path)
    }

    pub(crate) fn span_save_span(span: Span) -> usize {
        span.span.save() as usize
    }
    pub(crate) fn span_recover_proc_macro_span(id: usize) -> Span {
        Span { span: Box::leak(Box::new(host::Span::recover(id as u64))) }
    }
    pub(crate) fn span_debug(span: Span) -> String {
        span.span.debug()
    }
    pub(crate) fn span_parent(span: Span) -> Option<Span> {
        span.parent()
    }
    pub(crate) fn span_source(span: Span) -> Span {
        span.source()
    }
    pub(crate) fn span_byte_range(span: Span) -> std::ops::Range<usize> {
        span.byte_range()
    }
    pub(crate) fn span_start(span: Span) -> Span {
        span.start()
    }
    pub(crate) fn span_end(span: Span) -> Span {
        span.end()
    }
    pub(crate) fn span_line(span: Span) -> usize {
        span.span.line() as usize
    }
    pub(crate) fn span_column(span: Span) -> usize {
        span.span.column() as usize
    }
    pub(crate) fn span_file(span: Span) -> String {
        span.span.file()
    }
    pub(crate) fn span_local_file(span: Span) -> Option<String> {
        span.span.local_file()
    }
    pub(crate) fn span_join(span: Span, other: Span) -> Option<Span> {
        span.join(other)
    }
    pub(crate) fn span_resolved_at(span: Span, at: Span) -> Span {
        span.resolved_at(at)
    }
    pub(crate) fn span_source_text(span: Span) -> Option<String> {
        span.span.source_text()
    }
    pub(crate) fn span_subspan(span: Span, start: Bound<usize>, end: Bound<usize>) -> Option<Span> {
        Some(Span {
            span: Box::leak(Box::new(span.span.subspan(to_wasm_range(start), to_wasm_range(end))?)),
        })
    }

    pub(crate) fn ts_concat_trees(
        base: Option<TokenStream>,
        trees: Vec<super::TokenTree<TokenStream, Span, Symbol>>,
    ) -> TokenStream {
        TokenStream::concat_trees(base, trees)
    }
    pub(crate) fn ts_concat_streams(
        base: Option<TokenStream>,
        trees: Vec<TokenStream>,
    ) -> TokenStream {
        TokenStream::concat_streams(base, trees)
    }
    pub(crate) fn ts_into_trees(
        base: TokenStream,
    ) -> Vec<super::TokenTree<TokenStream, Span, Symbol>> {
        base.into_trees()
    }
    pub(crate) fn ts_expand_expr(stream: &TokenStream) -> Result<TokenStream, ()> {
        stream.expand_expr()
    }
    pub(crate) fn ts_from_str(src: &str) -> Result<TokenStream, String> {
        Ok(TokenStream { handle: host::TokenStream::from_str(src)? })
    }
    pub(crate) fn ts_is_empty(stream: &TokenStream) -> bool {
        stream.is_empty()
    }
    pub(crate) fn ts_to_string(stream: &TokenStream) -> String {
        stream.handle.to_string()
    }

    pub(crate) fn ts_from_token_tree(
        tree: crate::bridge::TokenTree<TokenStream, Span, Symbol>,
    ) -> TokenStream {
        let handle = host::TokenStream::from_token_tree(match tree {
            crate::bridge::TokenTree::Group(g) => host::TokenTree::Group(host::Group {
                delimiter: match g.delimiter {
                    super::Delimiter::Parenthesis => host::Delimiter::Parenthesis,
                    super::Delimiter::Brace => host::Delimiter::Brace,
                    super::Delimiter::Bracket => host::Delimiter::Bracket,
                    super::Delimiter::None => host::Delimiter::None,
                },
                stream: g.stream.map(|s| s.handle),
                span: host::DelimSpan {
                    open: g.span.open.span.clone(),
                    close: g.span.close.span.clone(),
                    entire: g.span.entire.span.clone(),
                },
            }),
            crate::bridge::TokenTree::Punct(p) => host::TokenTree::Punct(host::Punct {
                ch: p.ch,
                joint: p.joint,
                span: p.span.span.clone(),
            }),
            crate::bridge::TokenTree::Ident(i) => host::TokenTree::Ident(host::Ident {
                sym: host::Symbol { s: i.sym.to_string() },
                is_raw: i.is_raw,
                span: i.span.span.clone(),
            }),
            crate::bridge::TokenTree::Literal(l) => host::TokenTree::Literal(host::Literal {
                kind: l.kind.into(),
                span: l.span.span.clone(),
                suffix: l.suffix.map(|s| host::Symbol { s: s.to_string() }),
                symbol: host::Symbol { s: l.symbol.to_string() },
            }),
        });
        TokenStream { handle }
    }
}
