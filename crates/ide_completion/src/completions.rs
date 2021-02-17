//! This module defines an accumulator for completions which are going to be presented to user.

pub(crate) mod attribute;
pub(crate) mod dot;
pub(crate) mod record;
pub(crate) mod pattern;
pub(crate) mod fn_param;
pub(crate) mod keyword;
pub(crate) mod snippet;
pub(crate) mod qualified_path;
pub(crate) mod unqualified_path;
pub(crate) mod postfix;
pub(crate) mod macro_in_item_position;
pub(crate) mod trait_impl;
pub(crate) mod mod_;
pub(crate) mod flyimport;

use std::iter;

use hir::{known, ModPath, ScopeDef, Type};

use crate::{
    item::Builder,
    render::{
        const_::render_const,
        enum_variant::render_variant,
        function::render_fn,
        macro_::render_macro,
        pattern::{render_struct_pat, render_variant_pat},
        render_field, render_resolution, render_tuple_field,
        type_alias::render_type_alias,
        RenderContext,
    },
    CompletionContext, CompletionItem,
};

/// Represents an in-progress set of completions being built.
#[derive(Debug, Default)]
pub struct Completions {
    buf: Vec<CompletionItem>,
}

impl Into<Vec<CompletionItem>> for Completions {
    fn into(self) -> Vec<CompletionItem> {
        self.buf
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
    pub(crate) fn add(&mut self, item: CompletionItem) {
        self.buf.push(item.into())
    }

    pub(crate) fn add_all<I>(&mut self, items: I)
    where
        I: IntoIterator,
        I::Item: Into<CompletionItem>,
    {
        items.into_iter().for_each(|item| self.add(item.into()))
    }

    pub(crate) fn add_field(&mut self, ctx: &CompletionContext, field: hir::Field, ty: &Type) {
        let item = render_field(RenderContext::new(ctx), field, ty);
        self.add(item);
    }

    pub(crate) fn add_tuple_field(&mut self, ctx: &CompletionContext, field: usize, ty: &Type) {
        let item = render_tuple_field(RenderContext::new(ctx), field, ty);
        self.add(item);
    }

    pub(crate) fn add_resolution(
        &mut self,
        ctx: &CompletionContext,
        local_name: String,
        resolution: &ScopeDef,
    ) {
        if let Some(item) = render_resolution(RenderContext::new(ctx), local_name, resolution) {
            self.add(item);
        }
    }

    pub(crate) fn add_macro(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        macro_: hir::MacroDef,
    ) {
        let name = match name {
            Some(it) => it,
            None => return,
        };
        if let Some(item) = render_macro(RenderContext::new(ctx), None, name, macro_) {
            self.add(item);
        }
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: &CompletionContext,
        func: hir::Function,
        local_name: Option<String>,
    ) {
        if let Some(item) = render_fn(RenderContext::new(ctx), None, local_name, func) {
            self.add(item)
        }
    }

    pub(crate) fn add_variant_pat(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        local_name: Option<hir::Name>,
    ) {
        if let Some(item) = render_variant_pat(RenderContext::new(ctx), variant, local_name, None) {
            self.add(item);
        }
    }

    pub(crate) fn add_qualified_variant_pat(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        path: ModPath,
    ) {
        if let Some(item) = render_variant_pat(RenderContext::new(ctx), variant, None, Some(path)) {
            self.add(item);
        }
    }

    pub(crate) fn add_struct_pat(
        &mut self,
        ctx: &CompletionContext,
        strukt: hir::Struct,
        local_name: Option<hir::Name>,
    ) {
        if let Some(item) = render_struct_pat(RenderContext::new(ctx), strukt, local_name) {
            self.add(item);
        }
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext, constant: hir::Const) {
        if let Some(item) = render_const(RenderContext::new(ctx), constant) {
            self.add(item);
        }
    }

    pub(crate) fn add_type_alias(&mut self, ctx: &CompletionContext, type_alias: hir::TypeAlias) {
        if let Some(item) = render_type_alias(RenderContext::new(ctx), type_alias) {
            self.add(item)
        }
    }

    pub(crate) fn add_qualified_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        path: ModPath,
    ) {
        let item = render_variant(RenderContext::new(ctx), None, None, variant, Some(path));
        self.add(item);
    }

    pub(crate) fn add_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::Variant,
        local_name: Option<String>,
    ) {
        let item = render_variant(RenderContext::new(ctx), None, local_name, variant, None);
        self.add(item);
    }
}

fn complete_enum_variants(
    acc: &mut Completions,
    ctx: &CompletionContext,
    ty: &hir::Type,
    cb: impl Fn(&mut Completions, &CompletionContext, hir::Variant, hir::ModPath),
) {
    if let Some(hir::Adt::Enum(enum_data)) =
        iter::successors(Some(ty.clone()), |ty| ty.remove_ref()).last().and_then(|ty| ty.as_adt())
    {
        let variants = enum_data.variants(ctx.db);

        let module = if let Some(module) = ctx.scope.module() {
            // Compute path from the completion site if available.
            module
        } else {
            // Otherwise fall back to the enum's definition site.
            enum_data.module(ctx.db)
        };

        if let Some(impl_) = ctx.impl_def.as_ref().and_then(|impl_| ctx.sema.to_def(impl_)) {
            if impl_.target_ty(ctx.db) == *ty {
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
            if let Some(path) = module.find_use_path(ctx.db, hir::ModuleDef::from(variant)) {
                // Variants with trivial paths are already added by the existing completion logic,
                // so we should avoid adding these twice
                if path.segments().len() > 1 {
                    cb(acc, ctx, variant, path);
                }
            }
        }
    }
}
