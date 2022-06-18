//! This module defines an accumulator for completions which are going to be presented to user.

pub(crate) mod attribute;
pub(crate) mod dot;
pub(crate) mod expr;
pub(crate) mod extern_abi;
pub(crate) mod field;
pub(crate) mod flyimport;
pub(crate) mod fn_param;
pub(crate) mod format_string;
pub(crate) mod item_list;
pub(crate) mod keyword;
pub(crate) mod lifetime;
pub(crate) mod mod_;
pub(crate) mod pattern;
pub(crate) mod postfix;
pub(crate) mod record;
pub(crate) mod snippet;
pub(crate) mod r#type;
pub(crate) mod use_;
pub(crate) mod vis;

use std::iter;

use hir::{db::HirDatabase, known, ScopeDef};
use ide_db::SymbolKind;

use crate::{
    context::Visible,
    item::Builder,
    render::{
        const_::render_const,
        function::{render_fn, render_method},
        literal::{render_struct_literal, render_variant_lit},
        macro_::render_macro,
        pattern::{render_struct_pat, render_variant_pat},
        render_field, render_resolution, render_resolution_simple, render_tuple_field,
        type_alias::{render_type_alias, render_type_alias_with_eq},
        union_literal::render_union_literal,
        RenderContext,
    },
    CompletionContext, CompletionItem, CompletionItemKind,
};

fn module_or_attr(db: &dyn HirDatabase, def: ScopeDef) -> Option<ScopeDef> {
    match def {
        ScopeDef::ModuleDef(hir::ModuleDef::Macro(m)) if m.is_attr(db) => Some(def),
        ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) => Some(def),
        _ => None,
    }
}

fn module_or_fn_macro(db: &dyn HirDatabase, def: ScopeDef) -> Option<ScopeDef> {
    match def {
        ScopeDef::ModuleDef(hir::ModuleDef::Macro(m)) if m.is_fn_like(db) => Some(def),
        ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) => Some(def),
        _ => None,
    }
}

/// Represents an in-progress set of completions being built.
#[derive(Debug, Default)]
pub struct Completions {
    buf: Vec<CompletionItem>,
}

impl From<Completions> for Vec<CompletionItem> {
    fn from(val: Completions) -> Self {
        val.buf
    }
}

impl Builder {
    /// Convenience method, which allows to add a freshly created completion into accumulator
    /// without binding it to the variable.
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }
}

impl Completions {
    fn add(&mut self, item: CompletionItem) {
        self.buf.push(item)
    }

    fn add_opt(&mut self, item: Option<CompletionItem>) {
        if let Some(item) = item {
            self.buf.push(item)
        }
    }

    pub(crate) fn add_all<I>(&mut self, items: I)
    where
        I: IntoIterator,
        I::Item: Into<CompletionItem>,
    {
        items.into_iter().for_each(|item| self.add(item.into()))
    }

    pub(crate) fn add_keyword(&mut self, ctx: &CompletionContext, keyword: &'static str) {
        let item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), keyword);
        item.add_to(self);
    }

    pub(crate) fn add_nameref_keywords_with_colon(&mut self, ctx: &CompletionContext) {
        ["self::", "super::", "crate::"].into_iter().for_each(|kw| self.add_keyword(ctx, kw));
    }

    pub(crate) fn add_nameref_keywords(&mut self, ctx: &CompletionContext) {
        ["self", "super", "crate"].into_iter().for_each(|kw| self.add_keyword(ctx, kw));
    }

    pub(crate) fn add_keyword_snippet_expr(
        &mut self,
        ctx: &CompletionContext,
        kw: &str,
        snippet: &str,
        incomplete_let: bool,
    ) {
        let mut item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), kw);

        match ctx.config.snippet_cap {
            Some(cap) => {
                if snippet.ends_with('}') && incomplete_let {
                    // complete block expression snippets with a trailing semicolon, if inside an incomplete let
                    cov_mark::hit!(let_semi);
                    item.insert_snippet(cap, format!("{};", snippet));
                } else {
                    item.insert_snippet(cap, snippet);
                }
            }
            None => {
                item.insert_text(if snippet.contains('$') { kw } else { snippet });
            }
        };
        item.add_to(self);
    }

    pub(crate) fn add_keyword_snippet(&mut self, ctx: &CompletionContext, kw: &str, snippet: &str) {
        let mut item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), kw);

        match ctx.config.snippet_cap {
            Some(cap) => item.insert_snippet(cap, snippet),
            None => item.insert_text(if snippet.contains('$') { kw } else { snippet }),
        };
        item.add_to(self);
    }

    pub(crate) fn add_crate_roots(&mut self, ctx: &CompletionContext) {
        ctx.process_all_names(&mut |name, res| match res {
            ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) if m.is_crate_root(ctx.db) => {
                self.add_resolution(ctx, name, res);
            }
            _ => (),
        });
    }

    pub(crate) fn add_resolution(
        &mut self,
        ctx: &CompletionContext,
        local_name: hir::Name,
        resolution: hir::ScopeDef,
    ) {
        if ctx.is_scope_def_hidden(resolution) {
            cov_mark::hit!(qualified_path_doc_hidden);
            return;
        }
        self.add(render_resolution(RenderContext::new(ctx), local_name, resolution).build());
    }

    pub(crate) fn add_resolution_simple(
        &mut self,
        ctx: &CompletionContext,
        local_name: hir::Name,
        resolution: hir::ScopeDef,
    ) {
        if ctx.is_scope_def_hidden(resolution) {
            return;
        }
        self.add(render_resolution_simple(RenderContext::new(ctx), local_name, resolution).build());
    }

    pub(crate) fn add_macro(
        &mut self,
        ctx: &CompletionContext,
        mac: hir::Macro,
        local_name: hir::Name,
    ) {
        let is_private_editable = match ctx.is_visible(&mac) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add(
            render_macro(
                RenderContext::new(ctx).private_editable(is_private_editable),
                local_name,
                mac,
            )
            .build(),
        );
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: &CompletionContext,
        func: hir::Function,
        local_name: Option<hir::Name>,
    ) {
        let is_private_editable = match ctx.is_visible(&func) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add(
            render_fn(
                RenderContext::new(ctx).private_editable(is_private_editable),
                local_name,
                func,
            )
            .build(),
        );
    }

    pub(crate) fn add_method(
        &mut self,
        ctx: &CompletionContext,
        func: hir::Function,
        receiver: Option<hir::Name>,
        local_name: Option<hir::Name>,
    ) {
        let is_private_editable = match ctx.is_visible(&func) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add(
            render_method(
                RenderContext::new(ctx).private_editable(is_private_editable),
                receiver,
                local_name,
                func,
            )
            .build(),
        );
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext, konst: hir::Const) {
        let is_private_editable = match ctx.is_visible(&konst) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add_opt(render_const(
            RenderContext::new(ctx).private_editable(is_private_editable),
            konst,
        ));
    }

    pub(crate) fn add_type_alias(&mut self, ctx: &CompletionContext, type_alias: hir::TypeAlias) {
        let is_private_editable = match ctx.is_visible(&type_alias) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add_opt(render_type_alias(
            RenderContext::new(ctx).private_editable(is_private_editable),
            type_alias,
        ));
    }

    pub(crate) fn add_type_alias_with_eq(
        &mut self,
        ctx: &CompletionContext,
        type_alias: hir::TypeAlias,
    ) {
        self.add_opt(render_type_alias_with_eq(RenderContext::new(ctx), type_alias));
    }

    pub(crate) fn add_qualified_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        path: hir::ModPath,
    ) {
        if let Some(builder) =
            render_variant_lit(RenderContext::new(ctx), None, variant, Some(path))
        {
            self.add(builder.build());
        }
    }

    pub(crate) fn add_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        local_name: Option<hir::Name>,
    ) {
        if let Some(builder) =
            render_variant_lit(RenderContext::new(ctx), local_name, variant, None)
        {
            self.add(builder.build());
        }
    }

    pub(crate) fn add_field(
        &mut self,
        ctx: &CompletionContext,
        receiver: Option<hir::Name>,
        field: hir::Field,
        ty: &hir::Type,
    ) {
        let is_private_editable = match ctx.is_visible(&field) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        let item = render_field(
            RenderContext::new(ctx).private_editable(is_private_editable),
            receiver,
            field,
            ty,
        );
        self.add(item);
    }

    pub(crate) fn add_struct_literal(
        &mut self,
        ctx: &CompletionContext,
        strukt: hir::Struct,
        path: Option<hir::ModPath>,
        local_name: Option<hir::Name>,
    ) {
        if let Some(builder) =
            render_struct_literal(RenderContext::new(ctx), strukt, path, local_name)
        {
            self.add(builder.build());
        }
    }

    pub(crate) fn add_union_literal(
        &mut self,
        ctx: &CompletionContext,
        un: hir::Union,
        path: Option<hir::ModPath>,
        local_name: Option<hir::Name>,
    ) {
        let item = render_union_literal(RenderContext::new(ctx), un, path, local_name);
        self.add_opt(item);
    }

    pub(crate) fn add_tuple_field(
        &mut self,
        ctx: &CompletionContext,
        receiver: Option<hir::Name>,
        field: usize,
        ty: &hir::Type,
    ) {
        let item = render_tuple_field(RenderContext::new(ctx), receiver, field, ty);
        self.add(item);
    }

    pub(crate) fn add_lifetime(&mut self, ctx: &CompletionContext, name: hir::Name) {
        CompletionItem::new(SymbolKind::LifetimeParam, ctx.source_range(), name.to_smol_str())
            .add_to(self)
    }

    pub(crate) fn add_label(&mut self, ctx: &CompletionContext, name: hir::Name) {
        CompletionItem::new(SymbolKind::Label, ctx.source_range(), name.to_smol_str()).add_to(self)
    }

    pub(crate) fn add_variant_pat(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        local_name: Option<hir::Name>,
    ) {
        self.add_opt(render_variant_pat(
            RenderContext::new(ctx),
            variant,
            local_name.clone(),
            None,
        ));
    }

    pub(crate) fn add_qualified_variant_pat(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        path: hir::ModPath,
    ) {
        let path = Some(&path);
        self.add_opt(render_variant_pat(RenderContext::new(ctx), variant, None, path));
    }

    pub(crate) fn add_struct_pat(
        &mut self,
        ctx: &CompletionContext,
        strukt: hir::Struct,
        local_name: Option<hir::Name>,
    ) {
        self.add_opt(render_struct_pat(RenderContext::new(ctx), strukt, local_name));
    }
}

/// Calls the callback for each variant of the provided enum with the path to the variant.
/// Skips variants that are visible with single segment paths.
fn enum_variants_with_paths(
    acc: &mut Completions,
    ctx: &CompletionContext,
    enum_: hir::Enum,
    cb: impl Fn(&mut Completions, &CompletionContext, hir::Variant, hir::ModPath),
) {
    let variants = enum_.variants(ctx.db);

    if let Some(impl_) = ctx.impl_def.as_ref().and_then(|impl_| ctx.sema.to_def(impl_)) {
        if impl_.self_ty(ctx.db).as_adt() == Some(hir::Adt::Enum(enum_)) {
            for &variant in &variants {
                let self_path = hir::ModPath::from_segments(
                    hir::PathKind::Plain,
                    iter::once(known::SELF_TYPE).chain(iter::once(variant.name(ctx.db))),
                );
                cb(acc, ctx, variant, self_path);
            }
        }
    }

    for variant in variants {
        if let Some(path) = ctx.module.find_use_path(ctx.db, hir::ModuleDef::from(variant)) {
            // Variants with trivial paths are already added by the existing completion logic,
            // so we should avoid adding these twice
            if path.segments().len() > 1 {
                cb(acc, ctx, variant, path);
            }
        }
    }
}
