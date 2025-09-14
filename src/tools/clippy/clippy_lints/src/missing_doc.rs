// Note: More specifically this lint is largely inspired (aka copied) from
// *rustc*'s
// [`missing_doc`].
//
// [`missing_doc`]: https://github.com/rust-lang/rust/blob/cf9cf7c923eb01146971429044f216a3ca905e06/compiler/rustc_lint/src/builtin.rs#L415
//

use clippy_config::Conf;
use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use clippy_utils::source::SpanRangeExt;
use rustc_ast::ast::MetaItemInner;
use rustc_hir as hir;
use rustc_hir::Attribute;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{AssocContainer, Visibility};
use rustc_session::impl_lint_pass;
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::symbol::kw;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Warns if there is missing documentation for any private documentable item.
    ///
    /// ### Why restrict this?
    /// Doc is good. *rustc* has a `MISSING_DOCS`
    /// allowed-by-default lint for
    /// public members, but has no way to enforce documentation of private items.
    /// This lint fixes that.
    #[clippy::version = "pre 1.29.0"]
    pub MISSING_DOCS_IN_PRIVATE_ITEMS,
    restriction,
    "detects missing documentation for private members"
}

macro_rules! note_prev_span_then_ret {
    ($prev_span:expr, $span:expr) => {{
        $prev_span = Some($span);
        return;
    }};
}

pub struct MissingDoc {
    /// Whether to **only** check for missing documentation in items visible within the current
    /// crate. For example, `pub(crate)` items.
    crate_items_only: bool,
    /// Whether to allow fields starting with an underscore to skip documentation requirements
    allow_unused: bool,
    /// Stack of whether #[doc(hidden)] is set
    /// at each level which has lint attributes.
    doc_hidden_stack: Vec<bool>,
    /// Used to keep tracking of the previous item, field or variants etc, to get the search span.
    prev_span: Option<Span>,
}

impl MissingDoc {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            crate_items_only: conf.missing_docs_in_crate_items,
            allow_unused: conf.missing_docs_allow_unused,
            doc_hidden_stack: vec![false],
            prev_span: None,
        }
    }

    fn doc_hidden(&self) -> bool {
        *self.doc_hidden_stack.last().expect("empty doc_hidden_stack")
    }

    fn has_include(meta: Option<&[MetaItemInner]>) -> bool {
        if let Some(list) = meta
            && let Some(meta) = list.first()
            && let Some(name) = meta.ident()
        {
            name.name == sym::include
        } else {
            false
        }
    }

    fn check_missing_docs_attrs(
        &self,
        cx: &LateContext<'_>,
        def_id: LocalDefId,
        attrs: &[Attribute],
        sp: Span,
        article: &'static str,
        desc: &'static str,
    ) {
        // If we're building a test harness, then warning about
        // documentation is probably not really relevant right now.
        if cx.sess().opts.test {
            return;
        }

        // `#[doc(hidden)]` disables missing_docs check.
        if self.doc_hidden() {
            return;
        }

        if sp.from_expansion() {
            return;
        }

        if self.crate_items_only && def_id != CRATE_DEF_ID {
            let vis = cx.tcx.visibility(def_id);
            if vis == Visibility::Public || vis != Visibility::Restricted(CRATE_DEF_ID.into()) {
                return;
            }
        } else if def_id != CRATE_DEF_ID && cx.effective_visibilities.is_exported(def_id) {
            return;
        }

        if let Some(parent_def_id) = cx.tcx.opt_parent(def_id.to_def_id())
            && let DefKind::AnonConst
            | DefKind::AssocConst
            | DefKind::AssocFn
            | DefKind::Closure
            | DefKind::Const
            | DefKind::Fn
            | DefKind::InlineConst
            | DefKind::Static { .. }
            | DefKind::SyntheticCoroutineBody = cx.tcx.def_kind(parent_def_id)
        {
            // Nested item has no generated documentation, so it doesn't need to be documented.
            return;
        }

        let has_doc = attrs
            .iter()
            .any(|a| a.doc_str().is_some() || Self::has_include(a.meta_item_list().as_deref()))
            || matches!(self.search_span(sp), Some(span) if span_to_snippet_contains_docs(cx, span));

        if !has_doc {
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                sp,
                format!("missing documentation for {article} {desc}"),
            );
        }
    }

    /// Return a span to search for doc comments manually.
    ///
    /// # Example
    /// ```ignore
    /// fn foo() { ... }
    /// ^^^^^^^^^^^^^^^^ prev_span
    ///                ↑
    /// |  search_span |
    /// ↓
    /// fn bar() { ... }
    /// ^^^^^^^^^^^^^^^^ cur_span
    /// ```
    fn search_span(&self, cur_span: Span) -> Option<Span> {
        let prev_span = self.prev_span?;
        let start_pos = if prev_span.contains(cur_span) {
            // In case when the prev_span is an entire struct, or enum,
            // and the current span is a field, or variant, we need to search from
            // the starting pos of the previous span.
            prev_span.lo()
        } else {
            prev_span.hi()
        };
        let search_span = cur_span.with_lo(start_pos).with_hi(cur_span.lo());
        Some(search_span)
    }
}

impl_lint_pass!(MissingDoc => [MISSING_DOCS_IN_PRIVATE_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for MissingDoc {
    fn check_attributes(&mut self, _: &LateContext<'tcx>, attrs: &'tcx [Attribute]) {
        let doc_hidden = self.doc_hidden() || is_doc_hidden(attrs);
        self.doc_hidden_stack.push(doc_hidden);
    }

    fn check_attributes_post(&mut self, _: &LateContext<'tcx>, _: &'tcx [Attribute]) {
        self.doc_hidden_stack.pop().expect("empty doc_hidden_stack");
    }

    fn check_crate(&mut self, cx: &LateContext<'tcx>) {
        let attrs = cx.tcx.hir_attrs(hir::CRATE_HIR_ID);
        self.check_missing_docs_attrs(cx, CRATE_DEF_ID, attrs, cx.tcx.def_span(CRATE_DEF_ID), "the", "crate");
    }

    fn check_crate_post(&mut self, _: &LateContext<'tcx>) {
        self.prev_span = None;
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::Item<'_>) {
        match it.kind {
            hir::ItemKind::Fn { ident, .. } => {
                // ignore main()
                if ident.name == sym::main {
                    let at_root = cx.tcx.local_parent(it.owner_id.def_id) == CRATE_DEF_ID;
                    if at_root {
                        note_prev_span_then_ret!(self.prev_span, it.span);
                    }
                }
            },
            hir::ItemKind::Const(ident, ..) => {
                if ident.name == kw::Underscore {
                    note_prev_span_then_ret!(self.prev_span, it.span);
                }
            },
            hir::ItemKind::Enum(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Static(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Union(..) => {},
            hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::GlobalAsm { .. }
            | hir::ItemKind::Impl { .. }
            | hir::ItemKind::Use(..) => note_prev_span_then_ret!(self.prev_span, it.span),
        }

        let (article, desc) = cx.tcx.article_and_description(it.owner_id.to_def_id());

        let attrs = cx.tcx.hir_attrs(it.hir_id());
        if !is_from_proc_macro(cx, it) {
            self.check_missing_docs_attrs(cx, it.owner_id.def_id, attrs, it.span, article, desc);
        }
        self.prev_span = Some(it.span);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx hir::TraitItem<'_>) {
        let (article, desc) = cx.tcx.article_and_description(trait_item.owner_id.to_def_id());

        let attrs = cx.tcx.hir_attrs(trait_item.hir_id());
        if !is_from_proc_macro(cx, trait_item) {
            self.check_missing_docs_attrs(cx, trait_item.owner_id.def_id, attrs, trait_item.span, article, desc);
        }
        self.prev_span = Some(trait_item.span);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        // If the method is an impl for a trait, don't doc.
        match cx.tcx.associated_item(impl_item.owner_id).container {
            AssocContainer::Trait | AssocContainer::TraitImpl(_) => {
                note_prev_span_then_ret!(self.prev_span, impl_item.span);
            },
            AssocContainer::InherentImpl => {}
        }

        let (article, desc) = cx.tcx.article_and_description(impl_item.owner_id.to_def_id());
        let attrs = cx.tcx.hir_attrs(impl_item.hir_id());
        if !is_from_proc_macro(cx, impl_item) {
            self.check_missing_docs_attrs(cx, impl_item.owner_id.def_id, attrs, impl_item.span, article, desc);
        }
        self.prev_span = Some(impl_item.span);
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, sf: &'tcx hir::FieldDef<'_>) {
        if !(sf.is_positional()
            || is_from_proc_macro(cx, sf)
            || self.allow_unused && sf.ident.as_str().starts_with('_'))
        {
            let attrs = cx.tcx.hir_attrs(sf.hir_id);
            self.check_missing_docs_attrs(cx, sf.def_id, attrs, sf.span, "a", "struct field");
        }
        self.prev_span = Some(sf.span);
    }

    fn check_variant(&mut self, cx: &LateContext<'tcx>, v: &'tcx hir::Variant<'_>) {
        let attrs = cx.tcx.hir_attrs(v.hir_id);
        if !is_from_proc_macro(cx, v) {
            self.check_missing_docs_attrs(cx, v.def_id, attrs, v.span, "a", "variant");
        }
        self.prev_span = Some(v.span);
    }
}

fn span_to_snippet_contains_docs(cx: &LateContext<'_>, search_span: Span) -> bool {
    search_span.check_source_text(cx, |src| src.lines().rev().any(|line| line.trim().starts_with("///")))
}
