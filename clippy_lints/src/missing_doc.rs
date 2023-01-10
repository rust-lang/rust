// Note: More specifically this lint is largely inspired (aka copied) from
// *rustc*'s
// [`missing_doc`].
//
// [`missing_doc`]: https://github.com/rust-lang/rust/blob/cf9cf7c923eb01146971429044f216a3ca905e06/compiler/rustc_lint/src/builtin.rs#L415
//

use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use rustc_ast::ast::{self, MetaItem, MetaItemKind};
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::DefIdTree;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::source_map::Span;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Warns if there is missing doc for any documentable item
    /// (public or private).
    ///
    /// ### Why is this bad?
    /// Doc is good. *rustc* has a `MISSING_DOCS`
    /// allowed-by-default lint for
    /// public members, but has no way to enforce documentation of private items.
    /// This lint fixes that.
    #[clippy::version = "pre 1.29.0"]
    pub MISSING_DOCS_IN_PRIVATE_ITEMS,
    restriction,
    "detects missing documentation for public and private members"
}

pub struct MissingDoc {
    /// Stack of whether #[doc(hidden)] is set
    /// at each level which has lint attributes.
    doc_hidden_stack: Vec<bool>,
}

impl Default for MissingDoc {
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl MissingDoc {
    #[must_use]
    pub fn new() -> Self {
        Self {
            doc_hidden_stack: vec![false],
        }
    }

    fn doc_hidden(&self) -> bool {
        *self.doc_hidden_stack.last().expect("empty doc_hidden_stack")
    }

    fn has_include(meta: Option<MetaItem>) -> bool {
        if_chain! {
            if let Some(meta) = meta;
            if let MetaItemKind::List(list) = meta.kind;
            if let Some(meta) = list.get(0);
            if let Some(name) = meta.ident();
            then {
                name.name == sym::include
            } else {
                false
            }
        }
    }

    fn check_missing_docs_attrs(
        &self,
        cx: &LateContext<'_>,
        attrs: &[ast::Attribute],
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

        let has_doc = attrs
            .iter()
            .any(|a| a.doc_str().is_some() || Self::has_include(a.meta()));
        if !has_doc {
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                sp,
                &format!("missing documentation for {article} {desc}"),
            );
        }
    }
}

impl_lint_pass!(MissingDoc => [MISSING_DOCS_IN_PRIVATE_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for MissingDoc {
    fn enter_lint_attrs(&mut self, _: &LateContext<'tcx>, attrs: &'tcx [ast::Attribute]) {
        let doc_hidden = self.doc_hidden() || is_doc_hidden(attrs);
        self.doc_hidden_stack.push(doc_hidden);
    }

    fn exit_lint_attrs(&mut self, _: &LateContext<'tcx>, _: &'tcx [ast::Attribute]) {
        self.doc_hidden_stack.pop().expect("empty doc_hidden_stack");
    }

    fn check_crate(&mut self, cx: &LateContext<'tcx>) {
        let attrs = cx.tcx.hir().attrs(hir::CRATE_HIR_ID);
        self.check_missing_docs_attrs(cx, attrs, cx.tcx.def_span(CRATE_DEF_ID), "the", "crate");
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, it: &'tcx hir::Item<'_>) {
        match it.kind {
            hir::ItemKind::Fn(..) => {
                // ignore main()
                if it.ident.name == sym::main {
                    let at_root = cx.tcx.local_parent(it.owner_id.def_id) == CRATE_DEF_ID;
                    if at_root {
                        return;
                    }
                }
            },
            hir::ItemKind::Const(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::Static(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::TraitAlias(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::OpaqueTy(..) => {},
            hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::GlobalAsm(..)
            | hir::ItemKind::Impl { .. }
            | hir::ItemKind::Use(..) => return,
        };

        let (article, desc) = cx.tcx.article_and_description(it.owner_id.to_def_id());

        let attrs = cx.tcx.hir().attrs(it.hir_id());
        if !is_from_proc_macro(cx, it) {
            self.check_missing_docs_attrs(cx, attrs, it.span, article, desc);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx hir::TraitItem<'_>) {
        let (article, desc) = cx.tcx.article_and_description(trait_item.owner_id.to_def_id());

        let attrs = cx.tcx.hir().attrs(trait_item.hir_id());
        if !is_from_proc_macro(cx, trait_item) {
            self.check_missing_docs_attrs(cx, attrs, trait_item.span, article, desc);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx hir::ImplItem<'_>) {
        // If the method is an impl for a trait, don't doc.
        if let Some(cid) = cx.tcx.associated_item(impl_item.owner_id).impl_container(cx.tcx) {
            if cx.tcx.impl_trait_ref(cid).is_some() {
                return;
            }
        } else {
            return;
        }

        let (article, desc) = cx.tcx.article_and_description(impl_item.owner_id.to_def_id());
        let attrs = cx.tcx.hir().attrs(impl_item.hir_id());
        if !is_from_proc_macro(cx, impl_item) {
            self.check_missing_docs_attrs(cx, attrs, impl_item.span, article, desc);
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, sf: &'tcx hir::FieldDef<'_>) {
        if !sf.is_positional() {
            let attrs = cx.tcx.hir().attrs(sf.hir_id);
            if !is_from_proc_macro(cx, sf) {
                self.check_missing_docs_attrs(cx, attrs, sf.span, "a", "struct field");
            }
        }
    }

    fn check_variant(&mut self, cx: &LateContext<'tcx>, v: &'tcx hir::Variant<'_>) {
        let attrs = cx.tcx.hir().attrs(v.hir_id);
        if !is_from_proc_macro(cx, v) {
            self.check_missing_docs_attrs(cx, attrs, v.span, "a", "variant");
        }
    }
}
