use std::sync::Arc;

use rustc_ast::tokenstream;
use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_proc_macro::bridge;
use rustc_proc_macro::bridge::server::Server as _;
use rustc_session::Session;
use rustc_span::{Span, Symbol};

use crate::base::ExtCtxt;
use crate::proc_macro_server::Rustc;
use crate::wasm_proc_macro::generated::exports::rust_lang::rust::custom_derive::DeriveKind;
use crate::wasm_proc_macro::generated::rust_lang::rust::host;

#[allow(unreachable_pub)]
mod generated {
    wasmtime::component::bindgen!({
        path: "../../library/proc_macro/wasm-interface.wit",

        with: {
            "rust-lang:rust/host.token-stream": rustc_ast::tokenstream::TokenStream,
            "rust-lang:rust/host.diagnostic": rust_lang::rust::host::DiagnosticInner,
        },
    });
}

pub struct Ctx {
    ctx: wasmtime_wasi::WasiCtx,
    pub table: wasmtime_wasi::ResourceTable,

    // Outer Option is tracking whether a panic was stored during wasm execution
    // (from a panic hook).
    // Inner Option<String> is the panic message (which may not be available).
    stored_panic: Option<Option<String>>,
}

impl Default for Ctx {
    fn default() -> Self {
        Ctx {
            ctx: wasmtime_wasi::WasiCtx::builder().inherit_stdout().inherit_stderr().build(),
            table: Default::default(),
            stored_panic: None,
        }
    }
}

impl wasmtime_wasi::WasiView for Ctx {
    fn ctx(&mut self) -> wasmtime_wasi::WasiCtxView<'_> {
        wasmtime_wasi::WasiCtxView { ctx: &mut self.ctx, table: &mut self.table }
    }
}

trait ToInternal<T> {
    fn to_internal(self) -> T;
}

impl ToInternal<rustc_errors::Level> for generated::rust_lang::rust::host::Level {
    fn to_internal(self) -> rustc_errors::Level {
        match self {
            Self::Error => rustc_errors::Level::Error,
            Self::Warning => rustc_errors::Level::Warning,
            Self::Help => rustc_errors::Level::Help,
            Self::Note => rustc_errors::Level::Note,
        }
    }
}

impl ToInternal<rustc_proc_macro::Level> for generated::rust_lang::rust::host::Level {
    fn to_internal(self) -> rustc_proc_macro::Level {
        match self {
            Self::Error => rustc_proc_macro::Level::Error,
            Self::Warning => rustc_proc_macro::Level::Warning,
            Self::Help => rustc_proc_macro::Level::Help,
            Self::Note => rustc_proc_macro::Level::Note,
        }
    }
}

impl host::DiagnosticInner {
    fn to_internal(
        self,
        table: &mut wasmtime_wasi::ResourceTable,
    ) -> rustc_proc_macro::bridge::Diagnostic<Span> {
        rustc_proc_macro::bridge::Diagnostic {
            level: self.level.to_internal(),
            message: self.message,
            spans: self.spans.into_iter().map(|s| s.decode()).collect(),
            children: self
                .children
                .into_iter()
                .map(|c| {
                    let child = table.delete(c).unwrap();
                    child.to_internal(table)
                })
                .collect(),
        }
    }
}

impl ToInternal<host::LiteralKind> for rustc_proc_macro::bridge::LitKind {
    fn to_internal(self) -> host::LiteralKind {
        use host::LiteralKind;
        use rustc_proc_macro::bridge::LitKind;
        match self {
            LitKind::Byte => LiteralKind::Byte,
            LitKind::Char => LiteralKind::Char,
            LitKind::Integer => LiteralKind::Integer,
            LitKind::Float => LiteralKind::Float,
            LitKind::Str => LiteralKind::Str,
            LitKind::StrRaw(n) => LiteralKind::StrRaw(n),
            LitKind::ByteStr => LiteralKind::ByteStr,
            LitKind::ByteStrRaw(n) => LiteralKind::ByteStrRaw(n),
            LitKind::CStr => LiteralKind::CStr,
            LitKind::CStrRaw(n) => LiteralKind::CStrRaw(n),
            LitKind::ErrWithGuar => LiteralKind::ErrWithGuar,
        }
    }
}

impl ToInternal<rustc_proc_macro::bridge::LitKind> for host::LiteralKind {
    fn to_internal(self) -> rustc_proc_macro::bridge::LitKind {
        use host::LiteralKind;
        use rustc_proc_macro::bridge::LitKind;
        match self {
            LiteralKind::Byte => LitKind::Byte,
            LiteralKind::Char => LitKind::Char,
            LiteralKind::Integer => LitKind::Integer,
            LiteralKind::Float => LitKind::Float,
            LiteralKind::Str => LitKind::Str,
            LiteralKind::StrRaw(n) => LitKind::StrRaw(n),
            LiteralKind::ByteStr => LitKind::ByteStr,
            LiteralKind::ByteStrRaw(n) => LitKind::ByteStrRaw(n),
            LiteralKind::CStr => LitKind::CStr,
            LiteralKind::CStrRaw(n) => LitKind::CStrRaw(n),
            LiteralKind::ErrWithGuar => LitKind::ErrWithGuar,
        }
    }
}

trait ToWasmSpan {
    fn encode(self) -> WasmSpan;
}

impl ToWasmSpan for Span {
    fn encode(self) -> WasmSpan {
        WasmSpan { inner: self.encode_raw() }
    }
}

impl WasmSpan {
    fn decode(self) -> Span {
        Span::decode_raw(self.inner)
    }
}

impl generated::rust_lang::rust::host::Host for Ctx {
    fn emit_diagnostic(&mut self, diag: wasmtime::component::Resource<host::DiagnosticInner>) {
        let diagnostic = self.table.delete(diag).unwrap();
        DeriveExpandCtx::with(|(ecx, _)| {
            ecx.emit_diagnostic(diagnostic.to_internal(&mut self.table))
        })
    }

    fn literal_from_str(&mut self, s: String) -> Result<host::Literal, String> {
        DeriveExpandCtx::with(|(ecx, _)| {
            let literal = ecx.literal_from_str(&s)?;
            Ok(host::Literal {
                kind: literal.kind.to_internal(),
                symbol: literal.symbol.to_string(),
                suffix: literal.suffix.map(|s| s.to_string()),
                span: self.span_call_site(),
            })
        })
    }

    fn symbol_normalize_and_validate_ident(&mut self, string: String) -> Result<String, ()> {
        DeriveExpandCtx::with(|(ecx, _)| {
            let sym = ecx.symbol_normalize_and_validate_ident(&string)?;
            Ok(sym.to_string())
        })
    }

    fn injected_env_var(&mut self, var: String) -> Option<String> {
        DeriveExpandCtx::with(|(ecx, _)| ecx.injected_env_var(&var))
    }

    fn track_env_var(&mut self, var: String, value: Option<String>) {
        DeriveExpandCtx::with(|(ecx, _)| ecx.track_env_var(&var, value.as_deref()))
    }

    fn track_path(&mut self, path: String) {
        DeriveExpandCtx::with(|(ecx, _)| ecx.track_path(&path))
    }

    fn report_panic(&mut self, msg: Option<String>) -> bool {
        self.stored_panic = Some(msg);

        DeriveExpandCtx::with(|(ecx, _)| ecx.ecx.ecfg.proc_macro_backtrace)
    }

    fn span_is_same(&mut self, a: WasmSpan, b: WasmSpan) -> bool {
        a.decode() == b.decode()
    }

    fn span_subspan(
        &mut self,
        a: WasmSpan,
        start: host::RangeBound,
        end: host::RangeBound,
    ) -> Option<WasmSpan> {
        let span = a.decode();

        let start = match start.bound {
            host::Bound::Included => std::ops::Bound::Included(start.value as usize),
            host::Bound::Excluded => std::ops::Bound::Excluded(start.value as usize),
            host::Bound::Unbounded => std::ops::Bound::Unbounded,
        };
        let end = match end.bound {
            host::Bound::Included => std::ops::Bound::Included(end.value as usize),
            host::Bound::Excluded => std::ops::Bound::Excluded(end.value as usize),
            host::Bound::Unbounded => std::ops::Bound::Unbounded,
        };
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_subspan(span, start, end))?;
        Some(ret.encode())
    }

    fn span_def_site(&mut self) -> WasmSpan {
        DeriveExpandCtx::with(|(ecx, _)| ecx.def_site.encode())
    }

    fn span_call_site(&mut self) -> WasmSpan {
        DeriveExpandCtx::with(|(ecx, _)| ecx.call_site.encode())
    }

    fn span_mixed_site(&mut self) -> WasmSpan {
        DeriveExpandCtx::with(|(ecx, _)| ecx.mixed_site.encode())
    }

    fn span_line(&mut self, sp: WasmSpan) -> u64 {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_line(span)).try_into().unwrap()
    }

    fn span_column(&mut self, sp: WasmSpan) -> u64 {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_column(span).try_into().unwrap())
    }

    fn span_file(&mut self, sp: WasmSpan) -> String {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_file(span))
    }

    fn span_local_file(&mut self, sp: WasmSpan) -> Option<String> {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_local_file(span))
    }

    fn span_start(&mut self, sp: WasmSpan) -> WasmSpan {
        let span = sp.decode();
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_start(span));
        ret.encode()
    }

    fn span_end(&mut self, sp: WasmSpan) -> WasmSpan {
        let span = sp.decode();
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_end(span));
        ret.encode()
    }

    fn span_join(&mut self, this: WasmSpan, other: WasmSpan) -> Option<WasmSpan> {
        let this = this.decode();
        let other = other.decode();

        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_join(this, other))?;
        Some(ret.encode())
    }

    fn span_byte_range(&mut self, sp: WasmSpan) -> (u64, u64) {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| {
            let ret = ecx.span_byte_range(span);
            (ret.start as u64, ret.end as u64)
        })
    }

    fn span_parent(&mut self, sp: WasmSpan) -> Option<WasmSpan> {
        let span = sp.decode();
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_parent(span))?;
        Some(ret.encode())
    }

    fn span_source(&mut self, sp: WasmSpan) -> WasmSpan {
        let span = sp.decode();
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_source(span));
        ret.encode()
    }

    fn span_debug(&mut self, sp: WasmSpan) -> String {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_debug(span))
    }

    fn span_source_text(&mut self, sp: WasmSpan) -> Option<String> {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_source_text(span))
    }

    fn span_resolved_at(&mut self, this: WasmSpan, at: WasmSpan) -> WasmSpan {
        let this = this.decode();
        let at = at.decode();
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.span_resolved_at(this, at));
        ret.encode()
    }

    fn span_save_span(&mut self, sp: WasmSpan) -> u64 {
        let span = sp.decode();
        DeriveExpandCtx::with(|(ecx, _)| ecx.span_save_span(span)).try_into().unwrap()
    }

    fn span_recover_proc_macro_span(&mut self, id: u64) -> WasmSpan {
        let span = DeriveExpandCtx::with(|(ecx, _)| ecx.span_recover_proc_macro_span(id as usize));
        span.encode()
    }

    fn ts_drop(&mut self, ts: TokenStreamResource) {
        self.table.delete(ts).unwrap();
    }

    fn ts_is_empty(&mut self, input: TokenStreamResource) -> bool {
        let input = self.table.get(&input).unwrap();
        input.is_empty()
    }

    fn ts_concat_streams(
        &mut self,
        base: Option<TokenStreamResource>,
        streams: Vec<TokenStreamResource>,
    ) -> TokenStreamResource {
        let base = base.map(|b| self.table.delete(b).unwrap());
        let ret = DeriveExpandCtx::with(|(ecx, _)| {
            ecx.ts_concat_streams(
                base,
                streams.into_iter().map(|s| self.table.delete(s).unwrap()).collect(),
            )
        });
        self.table.push(ret).unwrap()
    }

    fn ts_from_token_tree(&mut self, tree: host::TokenTree) -> TokenStreamResource {
        let ret = DeriveExpandCtx::with(|(ecx, _)| {
            ecx.ts_from_token_tree(tree.to_internal(&mut self.table))
        });
        self.table.push(ret).unwrap()
    }

    fn ts_concat_trees(
        &mut self,
        base: Option<TokenStreamResource>,
        trees: Vec<generated::rust_lang::rust::host::TokenTree>,
    ) -> TokenStreamResource {
        let base = base.map(|b| self.table.delete(b).unwrap());
        let ret = DeriveExpandCtx::with(|(ecx, _)| {
            ecx.ts_concat_trees(
                base,
                trees.into_iter().map(|t| t.to_internal(&mut self.table)).collect(),
            )
        });
        self.table.push(ret).unwrap()
    }

    fn ts_clone(&mut self, input: TokenStreamResource) -> TokenStreamResource {
        let input = self.table.get(&input).unwrap();
        self.table.push(input.clone()).unwrap()
    }

    fn ts_expand_expr(&mut self, stream: TokenStreamResource) -> Result<TokenStreamResource, ()> {
        let stream = self.table.get(&stream).unwrap();
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.ts_expand_expr(stream))?;
        Ok(self.table.push(ret).unwrap())
    }

    fn ts_from_str(&mut self, src: String) -> Result<TokenStreamResource, String> {
        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.ts_from_str(&src))?;
        Ok(self.table.push(ret).unwrap())
    }

    fn ts_to_string(&mut self, input: TokenStreamResource) -> String {
        let input = self.table.get(&input).unwrap();
        DeriveExpandCtx::with(|(ecx, _)| ecx.ts_to_string(&input))
    }

    fn ts_into_trees(&mut self, stream: TokenStreamResource) -> Vec<host::TokenTree> {
        let stream = self.table.get(&stream).expect("valid resource");
        let stream = stream.clone();

        let ret = DeriveExpandCtx::with(|(ecx, _)| ecx.ts_into_trees(stream));
        ret.into_iter()
            .map(|tt| match tt {
                bridge::TokenTree::Group(group) => host::TokenTree::Group(host::Group {
                    delimiter: match group.delimiter {
                        rustc_proc_macro::Delimiter::Parenthesis => host::Delimiter::Parenthesis,
                        rustc_proc_macro::Delimiter::Brace => host::Delimiter::Brace,
                        rustc_proc_macro::Delimiter::Bracket => host::Delimiter::Bracket,
                        rustc_proc_macro::Delimiter::None => host::Delimiter::None,
                    },
                    stream: group.stream.map(|ts| self.table.push(ts).unwrap()),
                    span: host::DelimSpan {
                        open: group.span.open.encode(),
                        close: group.span.close.encode(),
                        entire: group.span.entire.encode(),
                    },
                }),
                bridge::TokenTree::Punct(punct) => host::TokenTree::Punct(host::Punct {
                    ch: punct.ch,
                    joint: punct.joint,
                    span: punct.span.encode(),
                }),
                bridge::TokenTree::Ident(ident) => host::TokenTree::Ident(host::Ident {
                    sym: ident.sym.to_string(),
                    is_raw: ident.is_raw,
                    span: ident.span.encode(),
                }),
                bridge::TokenTree::Literal(literal) => host::TokenTree::Literal(host::Literal {
                    kind: literal.kind.to_internal(),
                    symbol: literal.symbol.to_string(),
                    suffix: literal.suffix.map(|s| s.to_string()),
                    span: literal.span.encode(),
                }),
            })
            .collect()
    }
}

type TokenStreamResource = wasmtime::component::Resource<TokenStream>;
type WasmSpan = generated::rust_lang::rust::host::Span;
type Diagnostic = generated::rust_lang::rust::host::DiagnosticInner;

impl generated::rust_lang::rust::host::HostDiagnostic for Ctx {
    fn new(&mut self, diagnostic: Diagnostic) -> wasmtime::component::Resource<Diagnostic> {
        self.table.push(diagnostic).unwrap()
    }

    fn drop(&mut self, rep: wasmtime::component::Resource<Diagnostic>) -> wasmtime::Result<()> {
        self.table.delete(rep)?;
        Ok(())
    }
}

impl host::DelimSpan {
    fn to_internal(self) -> rustc_proc_macro::bridge::DelimSpan<Span> {
        rustc_proc_macro::bridge::DelimSpan {
            open: self.open.decode(),
            close: self.close.decode(),
            entire: self.entire.decode(),
        }
    }
}

impl host::Delimiter {
    fn to_internal(self) -> rustc_proc_macro::Delimiter {
        match self {
            host::Delimiter::Parenthesis => rustc_proc_macro::Delimiter::Parenthesis,
            host::Delimiter::Brace => rustc_proc_macro::Delimiter::Brace,
            host::Delimiter::Bracket => rustc_proc_macro::Delimiter::Bracket,
            host::Delimiter::None => rustc_proc_macro::Delimiter::None,
        }
    }
}

impl host::TokenTree {
    fn to_internal(
        self,
        table: &mut wasmtime_wasi::ResourceTable,
    ) -> bridge::TokenTree<TokenStream, Span, Symbol> {
        match self {
            host::TokenTree::Group(group) => bridge::TokenTree::Group(bridge::Group {
                delimiter: group.delimiter.to_internal(),
                stream: group.stream.map(|s| table.delete(s).unwrap()),
                span: group.span.to_internal(),
            }),
            host::TokenTree::Punct(punct) => bridge::TokenTree::Punct(bridge::Punct {
                ch: punct.ch,
                joint: punct.joint,
                span: punct.span.decode(),
            }),
            host::TokenTree::Ident(ident) => bridge::TokenTree::Ident(bridge::Ident {
                sym: Symbol::intern(&ident.sym),
                is_raw: ident.is_raw,
                span: ident.span.decode(),
            }),
            host::TokenTree::Literal(literal) => bridge::TokenTree::Literal(bridge::Literal {
                kind: literal.kind.to_internal(),
                symbol: Symbol::intern(&literal.symbol),
                suffix: literal.suffix.map(|s| Symbol::intern(&s)),
                span: literal.span.decode(),
            }),
        }
    }
}

impl generated::rust_lang::rust::host::HostTokenStream for Ctx {
    fn new(&mut self) -> TokenStreamResource {
        self.table.push(TokenStream::default()).unwrap()
    }

    fn drop(&mut self, r: TokenStreamResource) -> wasmtime::Result<()> {
        self.table.delete(r)?;
        Ok(())
    }
}

#[derive(Clone)]
pub enum RustcProcMacro {
    Dylib { client: rustc_proc_macro::bridge::client::Client },

    WasmExpand1 { client: crate::proc_macro::WasmExpand1 },

    WasmExpand2 { client: crate::proc_macro::WasmExpand2 },
}

type ErasedCtx<'a, 'b> = (Rustc<'a, 'b>, rustc_span::LocalExpnId);

/// Stores the context necessary to expand a derive proc macro via a query.
struct DeriveExpandCtx {
    /// Type-erased version of `&mut ExtCtxt`
    ctx: *mut (),
}

impl DeriveExpandCtx {
    /// Store the extension context and the client into the thread local value.
    /// It will be accessible via the `with` method while `f` is active.
    fn enter<F, R>(ecx: &mut ErasedCtx<'_, '_>, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // We need erasure to get rid of the lifetime
        let ctx = Self { ctx: ecx as *mut _ as *mut () };
        DERIVE_EXPAND_CTX.set(&ctx, f)
    }

    /// Accesses the thread local value of the derive expansion context.
    /// Must be called while the `enter` function is active.
    fn with<F, R>(f: F) -> R
    where
        F: for<'a, 'b, 'c> FnOnce(&'a mut ErasedCtx<'b, 'c>) -> R,
    {
        DERIVE_EXPAND_CTX.with(|ctx| {
            let ectx = {
                let casted = ctx.ctx.cast::<ErasedCtx<'_, '_>>();
                // SAFETY: We can only get the value from `with` while the `enter` function
                // is active (on the callstack), and that function's signature ensures that the
                // lifetime is valid.
                // If `with` is called at some other time, it will panic due to usage of
                // `scoped_tls::with`.
                unsafe { casted.as_mut().unwrap() }
            };

            f(ectx)
        })
    }
}

// When we invoke a query to expand a derive proc macro, we need to provide it with the expansion
// context and derive Client. We do that using a thread-local.
scoped_tls::scoped_thread_local!(static DERIVE_EXPAND_CTX: DeriveExpandCtx);

pub fn expand(
    engine: &wasmtime::Engine,
    component: &wasmtime::component::Component,
    ecx: &mut ExtCtxt<'_>,
    expn_id: rustc_span::LocalExpnId,
    derive_idx: usize,
    cb: impl FnOnce(
        &mut wasmtime::Store<Ctx>,
        wasmtime::component::ResourceAny,
        generated::exports::rust_lang::rust::custom_derive::GuestCustomDerive<'_>,
    ) -> wasmtime::Result<TokenStreamResource>,
) -> Result<tokenstream::TokenStream, Option<String>> {
    let mut linker = wasmtime::component::Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker).unwrap();
    generated::ProcMacro::add_to_linker::<_, wasmtime::component::HasSelf<_>>(
        &mut linker,
        |state| state,
    )
    .unwrap();

    let ctx = Ctx::default();

    let mut store = wasmtime::Store::new(&engine, ctx);
    let bindings = generated::ProcMacro::instantiate(&mut store, component, &linker).unwrap();

    let mut derives =
        bindings.rust_lang_rust_custom_derive().call_get_custom_derives(&mut store).unwrap();
    let derive = derives.remove(derive_idx);

    let ecx = Rustc::new(ecx);

    DeriveExpandCtx::enter(&mut (ecx, expn_id), || {
        match cb(&mut store, derive, bindings.rust_lang_rust_custom_derive().custom_derive()) {
            Ok(expanded) => Ok(store.data_mut().table.delete::<TokenStream>(expanded).unwrap()),
            Err(e) => {
                if let Some(msg) = store.data_mut().stored_panic.take() {
                    return Err(msg);
                }

                // If we didn't store a panic during execution then this must be something else.
                panic!("Failed to run wasm proc-macro: {:?}", e);
            }
        }
    })
}

pub enum WasmLoadError {
    FileRead(std::io::Error),
    Parse(wasmtime::Error),
    RuntimeFailed(wasmtime::Error),
}

pub fn load_wasm_macro(
    _sess: &Session,
    path: &std::path::Path,
) -> Result<Vec<RustcProcMacro>, WasmLoadError> {
    let code = std::fs::read(path).map_err(WasmLoadError::FileRead)?;
    let engine = wasmtime::Engine::default();
    let component =
        wasmtime::component::Component::new(&engine, code).map_err(WasmLoadError::Parse)?;
    let mut linker = wasmtime::component::Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker).unwrap();
    generated::ProcMacro::add_to_linker::<_, wasmtime::component::HasSelf<_>>(
        &mut linker,
        |state| state,
    )
    .unwrap();

    let ctx = Ctx::default();
    let mut store = wasmtime::Store::new(&engine, ctx);
    let bindings = generated::ProcMacro::instantiate(&mut store, &component, &linker).unwrap();
    let mut provided = vec![];

    let derives =
        bindings.rust_lang_rust_custom_derive().call_get_custom_derives(&mut store).unwrap();
    for (idx, derive) in derives.into_iter().enumerate() {
        let kind = bindings
            .rust_lang_rust_custom_derive()
            .custom_derive()
            .call_get_kind(&mut store, derive)
            .unwrap();
        let component = component.clone();
        let engine = engine.clone();
        match kind {
            DeriveKind::Expand1 => {
                provided.push(RustcProcMacro::WasmExpand1 {
                    client: IntoDynSyncSend(Arc::new(move |ecx, expn_id, stream| {
                        expand(&engine, &component, ecx, expn_id, idx, |store, derive, bindings| {
                            let input_handle = store.data_mut().table.push(stream).unwrap();
                            bindings.call_expand1(store, derive, input_handle)
                        })
                    })),
                });
            }

            DeriveKind::Expand2 => {
                provided.push(RustcProcMacro::WasmExpand2 {
                    client: IntoDynSyncSend(Arc::new(move |ecx, expn_id, stream_a, stream_b| {
                        expand(&engine, &component, ecx, expn_id, idx, |store, derive, bindings| {
                            let input_handle_a = store.data_mut().table.push(stream_a).unwrap();
                            let input_handle_b = store.data_mut().table.push(stream_b).unwrap();
                            bindings.call_expand2(store, derive, input_handle_a, input_handle_b)
                        })
                    })),
                });
            }
        }
    }
    Ok(provided)
}
