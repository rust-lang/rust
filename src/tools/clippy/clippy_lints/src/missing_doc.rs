use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{is_doc_hidden, is_from_proc_macro};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{
    AttrArgs, Attribute, Body, BodyId, FieldDef, HirId, ImplItem, Item, ItemKind, Node, TraitItem, Variant,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::middle::privacy::Level;
use rustc_middle::ty::Visibility;
use rustc_session::impl_lint_pass;
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::sym;
use rustc_span::symbol::kw;

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

pub struct MissingDoc {
    /// Whether to **only** check for missing documentation in items visible within the current
    /// crate. For example, `pub(crate)` items.
    crate_items_only: bool,
    /// Whether to allow fields starting with an underscore to skip documentation requirements
    allow_unused: bool,
    /// The current number of modules since the crate root.
    module_depth: u32,
    macro_module_depth: u32,
    /// The current level of the attribute stack.
    attr_depth: u32,
    /// What `attr_depth` level the first `doc(hidden)` attribute was seen. This is zero if the
    /// attribute hasn't been seen.
    doc_hidden_depth: u32,
    /// What `attr_depth` level the first `automatically_derived` attribute was seen. This is zero
    /// if the attribute hasn't been seen.
    automatically_derived_depth: u32,
    /// The id of the first body we've seen.
    in_body: Option<BodyId>,
    /// The module/crate id an item must be visible at to be linted.
    require_visibility_at: Option<LocalDefId>,
}

impl MissingDoc {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            crate_items_only: conf.missing_docs_in_crate_items,
            allow_unused: conf.missing_docs_allow_unused,
            module_depth: 0,
            macro_module_depth: 0,
            attr_depth: 0,
            doc_hidden_depth: 0,
            automatically_derived_depth: 0,
            in_body: None,
            require_visibility_at: None,
        }
    }

    fn is_missing_docs(&self, cx: &LateContext<'_>, def_id: LocalDefId, hir_id: HirId) -> bool {
        if cx.tcx.sess.opts.test {
            return false;
        }

        match cx.effective_visibilities.effective_vis(def_id) {
            None if self.require_visibility_at.is_some() => return false,
            None if self.crate_items_only && self.module_depth != 0 => return false,
            // `missing_docs` lint uses `Reexported` because rustdoc doesn't render documentation
            // for items without a reachable path.
            Some(vis) if vis.is_public_at_level(Level::Reexported) => return false,
            Some(vis) => {
                if self.crate_items_only {
                    // Use the `Reachable` level since rustdoc will be able to render the documentation
                    // when building private docs.
                    let vis = vis.at_level(Level::Reachable);
                    if !(vis.is_public() || matches!(vis, Visibility::Restricted(id) if id.is_top_level_module())) {
                        return false;
                    }
                } else if let Some(id) = self.require_visibility_at
                    && !vis.at_level(Level::Reexported).is_accessible_from(id, cx.tcx)
                {
                    return false;
                }
            },
            None => {},
        }

        !cx.tcx.hir_attrs(hir_id).iter().any(is_doc_attr)
    }
}

impl_lint_pass!(MissingDoc => [MISSING_DOCS_IN_PRIVATE_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for MissingDoc {
    fn check_attributes(&mut self, _: &LateContext<'tcx>, attrs: &'tcx [Attribute]) {
        self.attr_depth += 1;
        if self.doc_hidden_depth == 0 && is_doc_hidden(attrs) {
            self.doc_hidden_depth = self.attr_depth;
        }
    }

    fn check_attributes_post(&mut self, _: &LateContext<'tcx>, _: &'tcx [Attribute]) {
        self.attr_depth -= 1;
        if self.attr_depth < self.doc_hidden_depth {
            self.doc_hidden_depth = 0;
        }
        if self.attr_depth < self.automatically_derived_depth {
            self.automatically_derived_depth = 0;
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if self.doc_hidden_depth != 0 || self.automatically_derived_depth != 0 || self.in_body.is_some() {
            return;
        }

        let span = match item.kind {
            // ignore main()
            ItemKind::Fn { ident, .. }
                if ident.name == sym::main && cx.tcx.local_parent(item.owner_id.def_id) == CRATE_DEF_ID =>
            {
                return;
            },
            ItemKind::Const(ident, ..) if ident.name == kw::Underscore => return,
            ItemKind::Impl { .. } => {
                if cx.tcx.is_automatically_derived(item.owner_id.def_id.to_def_id()) {
                    self.automatically_derived_depth = self.attr_depth;
                }
                return;
            },
            ItemKind::ExternCrate(..)
            | ItemKind::ForeignMod { .. }
            | ItemKind::GlobalAsm { .. }
            | ItemKind::Use(..) => return,

            ItemKind::Mod(ident, ..) => {
                if item.span.from_expansion() && item.span.eq_ctxt(ident.span) {
                    self.module_depth += 1;
                    self.require_visibility_at = cx.tcx.opt_local_parent(item.owner_id.def_id);
                    self.macro_module_depth = self.module_depth;
                    return;
                }
                ident.span
            },

            ItemKind::Const(ident, ..)
            | ItemKind::Enum(ident, ..)
            | ItemKind::Fn { ident, .. }
            | ItemKind::Macro(ident, ..)
            | ItemKind::Static(_, ident, ..)
            | ItemKind::Struct(ident, ..)
            | ItemKind::Trait(_, _, _, ident, ..)
            | ItemKind::TraitAlias(_, ident, ..)
            | ItemKind::TyAlias(ident, ..)
            | ItemKind::Union(ident, ..) => ident.span,
        };

        if !item.span.from_expansion()
            && self.is_missing_docs(cx, item.owner_id.def_id, item.hir_id())
            && !is_from_proc_macro(cx, item)
        {
            let (article, desc) = cx.tcx.article_and_description(item.owner_id.to_def_id());
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                span,
                format!("missing documentation for {article} {desc}"),
            );
        }
        if matches!(item.kind, ItemKind::Mod(..)) {
            self.module_depth += 1;
        }
    }

    fn check_item_post(&mut self, _: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if matches!(item.kind, ItemKind::Mod(..))
            && self.doc_hidden_depth == 0
            && self.automatically_derived_depth == 0
            && self.in_body.is_none()
        {
            self.module_depth -= 1;
            if self.module_depth < self.macro_module_depth {
                self.require_visibility_at = None;
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if self.doc_hidden_depth == 0
            && self.automatically_derived_depth == 0
            && self.in_body.is_none()
            && !item.span.from_expansion()
            && self.is_missing_docs(cx, item.owner_id.def_id, item.hir_id())
            && !is_from_proc_macro(cx, item)
        {
            let (article, desc) = cx.tcx.article_and_description(item.owner_id.to_def_id());
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                item.ident.span,
                format!("missing documentation for {article} {desc}"),
            );
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if self.doc_hidden_depth == 0
            && self.automatically_derived_depth == 0
            && self.in_body.is_none()
            && let Node::Item(parent) = cx.tcx.parent_hir_node(item.hir_id())
            && let ItemKind::Impl(impl_) = parent.kind
            && impl_.of_trait.is_none()
            && !item.span.from_expansion()
            && self.is_missing_docs(cx, item.owner_id.def_id, item.hir_id())
            && !is_from_proc_macro(cx, item)
        {
            let (article, desc) = cx.tcx.article_and_description(item.owner_id.to_def_id());
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                item.ident.span,
                format!("missing documentation for {article} {desc}"),
            );
        }
    }

    fn check_body(&mut self, _: &LateContext<'tcx>, body: &Body<'tcx>) {
        if self.doc_hidden_depth == 0 && self.automatically_derived_depth == 0 && self.in_body.is_none() {
            self.in_body = Some(body.id());
        }
    }

    fn check_body_post(&mut self, _: &LateContext<'tcx>, body: &Body<'tcx>) {
        if self.in_body == Some(body.id()) {
            self.in_body = None;
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx FieldDef<'_>) {
        if self.doc_hidden_depth == 0
            && self.automatically_derived_depth == 0
            && self.in_body.is_none()
            && !field.is_positional()
            && !field.span.from_expansion()
            && !(self.allow_unused && field.ident.name.as_str().starts_with('_'))
            && self.is_missing_docs(cx, field.def_id, field.hir_id)
            && !is_from_proc_macro(cx, field)
        {
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                field.ident.span,
                "missing documentation for a field",
            );
        }
    }

    fn check_variant(&mut self, cx: &LateContext<'tcx>, variant: &'tcx Variant<'_>) {
        if self.doc_hidden_depth == 0
            && self.automatically_derived_depth == 0
            && self.in_body.is_none()
            && !variant.span.from_expansion()
            && self.is_missing_docs(cx, variant.def_id, variant.hir_id)
            && !is_from_proc_macro(cx, variant)
        {
            span_lint(
                cx,
                MISSING_DOCS_IN_PRIVATE_ITEMS,
                variant.ident.span,
                "missing documentation for a variant",
            );
        }
    }
}

fn is_doc_attr(attr: &Attribute) -> bool {
    match attr {
        Attribute::Parsed(AttributeKind::DocComment { .. }) => true,
        Attribute::Unparsed(attr)
            if let [ident] = &*attr.path.segments
                && ident.name == sym::doc =>
        {
            matches!(attr.args, AttrArgs::Eq { .. })
        },
        _ => false,
    }
}
