//! proc-macro server backend based on rust-analyzer's internal span representation
//! This backend is used solely by rust-analyzer as it ties into rust-analyzer internals.
//!
//! It is an unfortunate result of how the proc-macro API works that we need to look into the
//! concrete representation of the spans, and as such, RustRover cannot make use of this unless they
//! change their representation to be compatible with rust-analyzer's.
use std::{
    collections::{HashMap, HashSet},
    ops::{Bound, Range},
};

use intern::Symbol;
use rustc_proc_macro::bridge::server;
use span::{FIXUP_ERASED_FILE_AST_ID_MARKER, Span, TextRange, TextSize};

use crate::{
    ProcMacroClientHandle,
    bridge::{Diagnostic, ExpnGlobals, Literal, TokenTree},
    server_impl::literal_from_str,
};

pub struct RaSpanServer<'a> {
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_env_vars: HashMap<Box<str>, Option<Box<str>>>,
    // FIXME: Report this back to the caller to track as dependencies
    pub tracked_paths: HashSet<Box<str>>,
    pub call_site: Span,
    pub def_site: Span,
    pub mixed_site: Span,
    pub callback: Option<ProcMacroClientHandle<'a>>,
}

impl server::Server for RaSpanServer<'_> {
    type TokenStream = crate::token_stream::TokenStream<Span>;
    type Span = Span;
    type Symbol = Symbol;

    fn globals(&mut self) -> ExpnGlobals<Self::Span> {
        ExpnGlobals {
            def_site: self.def_site,
            call_site: self.call_site,
            mixed_site: self.mixed_site,
        }
    }

    fn intern_symbol(ident: &str) -> Self::Symbol {
        Symbol::intern(ident)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        f(symbol.as_str())
    }

    fn injected_env_var(&mut self, _: &str) -> Option<std::string::String> {
        None
    }

    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        self.tracked_env_vars.insert(var.into(), value.map(Into::into));
    }
    fn track_path(&mut self, path: &str) {
        self.tracked_paths.insert(path.into());
    }

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span>, ()> {
        literal_from_str(s, self.call_site)
    }

    fn emit_diagnostic(&mut self, _: Diagnostic<Self::Span>) {
        // FIXME handle diagnostic
    }

    fn ts_drop(&mut self, stream: Self::TokenStream) {
        drop(stream);
    }

    fn ts_clone(&mut self, stream: &Self::TokenStream) -> Self::TokenStream {
        stream.clone()
    }

    fn ts_is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }
    fn ts_from_str(&mut self, src: &str) -> Self::TokenStream {
        Self::TokenStream::from_str(src, self.call_site).unwrap_or_else(|e| {
            Self::TokenStream::from_str(
                &format!("compile_error!(\"failed to parse str to token stream: {e}\")"),
                self.call_site,
            )
            .unwrap()
        })
    }
    fn ts_to_string(&mut self, stream: &Self::TokenStream) -> String {
        stream.to_string()
    }

    fn ts_from_token_tree(&mut self, tree: TokenTree<Self::Span>) -> Self::TokenStream {
        Self::TokenStream::new(vec![tree])
    }

    fn ts_expand_expr(&mut self, self_: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        // FIXME: requires db, more importantly this requires name resolution so we would need to
        // eagerly expand this proc-macro, but we can't know that this proc-macro is eager until we
        // expand it ...
        // This calls for some kind of marker that a proc-macro wants to access this eager API,
        // otherwise we need to treat every proc-macro eagerly / or not support this.
        Ok(self_.clone())
    }

    fn ts_concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<TokenTree<Self::Span>>,
    ) -> Self::TokenStream {
        match base {
            Some(mut base) => {
                for tt in trees {
                    base.push_tree(tt);
                }
                base
            }
            None => Self::TokenStream::new(trees),
        }
    }

    fn ts_concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut stream = base.unwrap_or_default();
        for s in streams {
            stream.push_stream(s);
        }
        stream
    }

    fn ts_into_trees(&mut self, stream: Self::TokenStream) -> Vec<TokenTree<Self::Span>> {
        (*stream.0).clone()
    }

    fn span_debug(&mut self, span: Self::Span) -> String {
        format!("{:?}", span)
    }
    fn span_file(&mut self, span: Self::Span) -> String {
        self.callback.as_mut().map(|cb| cb.file(span.anchor.file_id.file_id())).unwrap_or_default()
    }
    fn span_local_file(&mut self, span: Self::Span) -> Option<String> {
        self.callback.as_mut().and_then(|cb| cb.local_file(span.anchor.file_id.file_id()))
    }
    fn span_save_span(&mut self, _span: Self::Span) -> usize {
        // FIXME, quote is incompatible with third-party tools
        // This is called by the quote proc-macro which is expanded when the proc-macro is compiled
        // As such, r-a will never observe this
        0
    }
    fn span_recover_proc_macro_span(&mut self, _id: usize) -> Self::Span {
        // FIXME, quote is incompatible with third-party tools
        // This is called by the expansion of quote!, r-a will observe this, but we don't have
        // access to the spans that were encoded
        self.call_site
    }
    /// Recent feature, not yet in the proc_macro
    ///
    /// See PR:
    /// https://github.com/rust-lang/rust/pull/55780
    fn span_source_text(&mut self, span: Self::Span) -> Option<String> {
        self.callback.as_mut()?.source_text(span)
    }

    fn span_parent(&mut self, _span: Self::Span) -> Option<Self::Span> {
        // FIXME requires db, looks up the parent call site
        None
    }
    fn span_source(&mut self, span: Self::Span) -> Self::Span {
        // FIXME requires db, returns the top level call site
        span
    }
    fn span_byte_range(&mut self, span: Self::Span) -> Range<usize> {
        if let Some(cb) = self.callback.as_mut() {
            return cb.byte_range(span);
        }
        Range { start: span.range.start().into(), end: span.range.end().into() }
    }
    fn span_join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        // We can't modify the span range for fixup spans, those are meaningful to fixup, so just
        // prefer the non-fixup span.
        if first.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return Some(second);
        }
        if second.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return Some(first);
        }
        // FIXME: Once we can talk back to the client, implement a "long join" request for anchors
        // that differ in [AstId]s as joining those spans requires resolving the AstIds.
        if first.anchor != second.anchor {
            return None;
        }
        // Differing context, we can't merge these so prefer the one that's root
        if first.ctx != second.ctx {
            if first.ctx.is_root() {
                return Some(second);
            } else if second.ctx.is_root() {
                return Some(first);
            }
        }
        Some(Span {
            range: first.range.cover(second.range),
            anchor: second.anchor,
            ctx: second.ctx,
        })
    }
    fn span_subspan(
        &mut self,
        span: Self::Span,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
        // We can't modify the span range for fixup spans, those are meaningful to fixup.
        if span.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return Some(span);
        }
        let length = span.range.len().into();

        let start: u32 = match start {
            Bound::Included(lo) => lo,
            Bound::Excluded(lo) => lo.checked_add(1)?,
            Bound::Unbounded => 0,
        }
        .try_into()
        .ok()?;

        let end: u32 = match end {
            Bound::Included(hi) => hi.checked_add(1)?,
            Bound::Excluded(hi) => hi,
            Bound::Unbounded => span.range.len().into(),
        }
        .try_into()
        .ok()?;

        // Bounds check the values, preventing addition overflow and OOB spans.
        let span_start = span.range.start().into();
        if (u32::MAX - start) < span_start
            || (u32::MAX - end) < span_start
            || start >= end
            || end > length
        {
            return None;
        }

        Some(Span {
            range: TextRange::new(TextSize::from(start), TextSize::from(end)) + span.range.start(),
            ..span
        })
    }

    fn span_resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        Span { ctx: at.ctx, ..span }
    }

    fn span_end(&mut self, span: Self::Span) -> Self::Span {
        // We can't modify the span range for fixup spans, those are meaningful to fixup.
        if span.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return span;
        }
        Span { range: TextRange::empty(span.range.end()), ..span }
    }

    fn span_start(&mut self, span: Self::Span) -> Self::Span {
        // We can't modify the span range for fixup spans, those are meaningful to fixup.
        if span.anchor.ast_id == FIXUP_ERASED_FILE_AST_ID_MARKER {
            return span;
        }
        Span { range: TextRange::empty(span.range.start()), ..span }
    }

    fn span_line(&mut self, span: Self::Span) -> usize {
        self.callback.as_mut().and_then(|cb| cb.line_column(span)).map_or(1, |(l, _)| l as usize)
    }

    fn span_column(&mut self, span: Self::Span) -> usize {
        self.callback.as_mut().and_then(|cb| cb.line_column(span)).map_or(1, |(_, c)| c as usize)
    }

    fn symbol_normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        // FIXME: nfc-normalize and validate idents
        Ok(<Self as server::Server>::intern_symbol(string))
    }
}
