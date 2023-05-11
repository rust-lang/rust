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
pub(crate) mod env_vars;

use std::iter;

use hir::{known, ScopeDef, Variant};
use ide_db::{imports::import_assets::LocatedImport, SymbolKind};
use syntax::ast;

use crate::{
    context::{
        DotAccess, ItemListKind, NameContext, NameKind, NameRefContext, NameRefKind,
        PathCompletionCtx, PathKind, PatternContext, TypeLocation, Visible,
    },
    item::Builder,
    render::{
        const_::render_const,
        function::{render_fn, render_method},
        literal::{render_struct_literal, render_variant_lit},
        macro_::render_macro,
        pattern::{render_struct_pat, render_variant_pat},
        render_field, render_path_resolution, render_pattern_resolution, render_tuple_field,
        type_alias::{render_type_alias, render_type_alias_with_eq},
        union_literal::render_union_literal,
        RenderContext,
    },
    CompletionContext, CompletionItem, CompletionItemKind,
};

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

    pub(crate) fn add_keyword(&mut self, ctx: &CompletionContext<'_>, keyword: &'static str) {
        let item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), keyword);
        item.add_to(self);
    }

    pub(crate) fn add_nameref_keywords_with_colon(&mut self, ctx: &CompletionContext<'_>) {
        ["self::", "crate::"].into_iter().for_each(|kw| self.add_keyword(ctx, kw));

        if ctx.depth_from_crate_root > 0 {
            self.add_keyword(ctx, "super::");
        }
    }

    pub(crate) fn add_nameref_keywords(&mut self, ctx: &CompletionContext<'_>) {
        ["self", "crate"].into_iter().for_each(|kw| self.add_keyword(ctx, kw));

        if ctx.depth_from_crate_root > 0 {
            self.add_keyword(ctx, "super");
        }
    }

    pub(crate) fn add_super_keyword(
        &mut self,
        ctx: &CompletionContext<'_>,
        super_chain_len: Option<usize>,
    ) {
        if let Some(len) = super_chain_len {
            if len > 0 && len < ctx.depth_from_crate_root {
                self.add_keyword(ctx, "super::");
            }
        }
    }

    pub(crate) fn add_keyword_snippet_expr(
        &mut self,
        ctx: &CompletionContext<'_>,
        incomplete_let: bool,
        kw: &str,
        snippet: &str,
    ) {
        let mut item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), kw);

        match ctx.config.snippet_cap {
            Some(cap) => {
                if incomplete_let && snippet.ends_with('}') {
                    // complete block expression snippets with a trailing semicolon, if inside an incomplete let
                    cov_mark::hit!(let_semi);
                    item.insert_snippet(cap, format!("{snippet};"));
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

    pub(crate) fn add_keyword_snippet(
        &mut self,
        ctx: &CompletionContext<'_>,
        kw: &str,
        snippet: &str,
    ) {
        let mut item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), kw);

        match ctx.config.snippet_cap {
            Some(cap) => item.insert_snippet(cap, snippet),
            None => item.insert_text(if snippet.contains('$') { kw } else { snippet }),
        };
        item.add_to(self);
    }

    pub(crate) fn add_crate_roots(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
    ) {
        ctx.process_all_names(&mut |name, res| match res {
            ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) if m.is_crate_root(ctx.db) => {
                self.add_module(ctx, path_ctx, m, name);
            }
            _ => (),
        });
    }

    pub(crate) fn add_path_resolution(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
        local_name: hir::Name,
        resolution: hir::ScopeDef,
    ) {
        let is_private_editable = match ctx.def_is_visible(&resolution) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add(
            render_path_resolution(
                RenderContext::new(ctx).private_editable(is_private_editable),
                path_ctx,
                local_name,
                resolution,
            )
            .build(),
        );
    }

    pub(crate) fn add_pattern_resolution(
        &mut self,
        ctx: &CompletionContext<'_>,
        pattern_ctx: &PatternContext,
        local_name: hir::Name,
        resolution: hir::ScopeDef,
    ) {
        let is_private_editable = match ctx.def_is_visible(&resolution) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add(
            render_pattern_resolution(
                RenderContext::new(ctx).private_editable(is_private_editable),
                pattern_ctx,
                local_name,
                resolution,
            )
            .build(),
        );
    }

    pub(crate) fn add_enum_variants(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
        e: hir::Enum,
    ) {
        e.variants(ctx.db)
            .into_iter()
            .for_each(|variant| self.add_enum_variant(ctx, path_ctx, variant, None));
    }

    pub(crate) fn add_module(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
        module: hir::Module,
        local_name: hir::Name,
    ) {
        self.add_path_resolution(
            ctx,
            path_ctx,
            local_name,
            hir::ScopeDef::ModuleDef(module.into()),
        );
    }

    pub(crate) fn add_macro(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
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
                path_ctx,
                local_name,
                mac,
            )
            .build(),
        );
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
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
                path_ctx,
                local_name,
                func,
            )
            .build(),
        );
    }

    pub(crate) fn add_method(
        &mut self,
        ctx: &CompletionContext<'_>,
        dot_access: &DotAccess,
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
                dot_access,
                receiver,
                local_name,
                func,
            )
            .build(),
        );
    }

    pub(crate) fn add_method_with_import(
        &mut self,
        ctx: &CompletionContext<'_>,
        dot_access: &DotAccess,
        func: hir::Function,
        import: LocatedImport,
    ) {
        let is_private_editable = match ctx.is_visible(&func) {
            Visible::Yes => false,
            Visible::Editable => true,
            Visible::No => return,
        };
        self.add(
            render_method(
                RenderContext::new(ctx)
                    .private_editable(is_private_editable)
                    .import_to_add(Some(import)),
                dot_access,
                None,
                None,
                func,
            )
            .build(),
        );
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext<'_>, konst: hir::Const) {
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

    pub(crate) fn add_type_alias(
        &mut self,
        ctx: &CompletionContext<'_>,
        type_alias: hir::TypeAlias,
    ) {
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
        ctx: &CompletionContext<'_>,
        type_alias: hir::TypeAlias,
    ) {
        self.add_opt(render_type_alias_with_eq(RenderContext::new(ctx), type_alias));
    }

    pub(crate) fn add_qualified_enum_variant(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
        variant: hir::Variant,
        path: hir::ModPath,
    ) {
        if let Some(builder) =
            render_variant_lit(RenderContext::new(ctx), path_ctx, None, variant, Some(path))
        {
            self.add(builder.build());
        }
    }

    pub(crate) fn add_enum_variant(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
        variant: hir::Variant,
        local_name: Option<hir::Name>,
    ) {
        if let PathCompletionCtx { kind: PathKind::Pat { pat_ctx }, .. } = path_ctx {
            cov_mark::hit!(enum_variant_pattern_path);
            self.add_variant_pat(ctx, pat_ctx, Some(path_ctx), variant, local_name);
            return;
        }

        if let Some(builder) =
            render_variant_lit(RenderContext::new(ctx), path_ctx, local_name, variant, None)
        {
            self.add(builder.build());
        }
    }

    pub(crate) fn add_field(
        &mut self,
        ctx: &CompletionContext<'_>,
        dot_access: &DotAccess,
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
            dot_access,
            receiver,
            field,
            ty,
        );
        self.add(item);
    }

    pub(crate) fn add_struct_literal(
        &mut self,
        ctx: &CompletionContext<'_>,
        path_ctx: &PathCompletionCtx,
        strukt: hir::Struct,
        path: Option<hir::ModPath>,
        local_name: Option<hir::Name>,
    ) {
        if let Some(builder) =
            render_struct_literal(RenderContext::new(ctx), path_ctx, strukt, path, local_name)
        {
            self.add(builder.build());
        }
    }

    pub(crate) fn add_union_literal(
        &mut self,
        ctx: &CompletionContext<'_>,
        un: hir::Union,
        path: Option<hir::ModPath>,
        local_name: Option<hir::Name>,
    ) {
        let item = render_union_literal(RenderContext::new(ctx), un, path, local_name);
        self.add_opt(item);
    }

    pub(crate) fn add_tuple_field(
        &mut self,
        ctx: &CompletionContext<'_>,
        receiver: Option<hir::Name>,
        field: usize,
        ty: &hir::Type,
    ) {
        let item = render_tuple_field(RenderContext::new(ctx), receiver, field, ty);
        self.add(item);
    }

    pub(crate) fn add_lifetime(&mut self, ctx: &CompletionContext<'_>, name: hir::Name) {
        CompletionItem::new(SymbolKind::LifetimeParam, ctx.source_range(), name.to_smol_str())
            .add_to(self)
    }

    pub(crate) fn add_label(&mut self, ctx: &CompletionContext<'_>, name: hir::Name) {
        CompletionItem::new(SymbolKind::Label, ctx.source_range(), name.to_smol_str()).add_to(self)
    }

    pub(crate) fn add_variant_pat(
        &mut self,
        ctx: &CompletionContext<'_>,
        pattern_ctx: &PatternContext,
        path_ctx: Option<&PathCompletionCtx>,
        variant: hir::Variant,
        local_name: Option<hir::Name>,
    ) {
        self.add_opt(render_variant_pat(
            RenderContext::new(ctx),
            pattern_ctx,
            path_ctx,
            variant,
            local_name,
            None,
        ));
    }

    pub(crate) fn add_qualified_variant_pat(
        &mut self,
        ctx: &CompletionContext<'_>,
        pattern_ctx: &PatternContext,
        variant: hir::Variant,
        path: hir::ModPath,
    ) {
        let path = Some(&path);
        self.add_opt(render_variant_pat(
            RenderContext::new(ctx),
            pattern_ctx,
            None,
            variant,
            None,
            path,
        ));
    }

    pub(crate) fn add_struct_pat(
        &mut self,
        ctx: &CompletionContext<'_>,
        pattern_ctx: &PatternContext,
        strukt: hir::Struct,
        local_name: Option<hir::Name>,
    ) {
        self.add_opt(render_struct_pat(RenderContext::new(ctx), pattern_ctx, strukt, local_name));
    }
}

/// Calls the callback for each variant of the provided enum with the path to the variant.
/// Skips variants that are visible with single segment paths.
fn enum_variants_with_paths(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    enum_: hir::Enum,
    impl_: &Option<ast::Impl>,
    cb: impl Fn(&mut Completions, &CompletionContext<'_>, hir::Variant, hir::ModPath),
) {
    let mut process_variant = |variant: Variant| {
        let self_path = hir::ModPath::from_segments(
            hir::PathKind::Plain,
            iter::once(known::SELF_TYPE).chain(iter::once(variant.name(ctx.db))),
        );

        cb(acc, ctx, variant, self_path);
    };

    let variants = enum_.variants(ctx.db);

    if let Some(impl_) = impl_.as_ref().and_then(|impl_| ctx.sema.to_def(impl_)) {
        if impl_.self_ty(ctx.db).as_adt() == Some(hir::Adt::Enum(enum_)) {
            variants.iter().for_each(|variant| process_variant(*variant));
        }
    }

    for variant in variants {
        if let Some(path) = ctx.module.find_use_path(
            ctx.db,
            hir::ModuleDef::from(variant),
            ctx.config.prefer_no_std,
        ) {
            // Variants with trivial paths are already added by the existing completion logic,
            // so we should avoid adding these twice
            if path.segments().len() > 1 {
                cb(acc, ctx, variant, path);
            }
        }
    }
}

pub(super) fn complete_name(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    NameContext { name, kind }: &NameContext,
) {
    match kind {
        NameKind::Const => {
            item_list::trait_impl::complete_trait_impl_const(acc, ctx, name);
        }
        NameKind::Function => {
            item_list::trait_impl::complete_trait_impl_fn(acc, ctx, name);
        }
        NameKind::IdentPat(pattern_ctx) => {
            if ctx.token.kind() != syntax::T![_] {
                complete_patterns(acc, ctx, pattern_ctx)
            }
        }
        NameKind::Module(mod_under_caret) => {
            mod_::complete_mod(acc, ctx, mod_under_caret);
        }
        NameKind::TypeAlias => {
            item_list::trait_impl::complete_trait_impl_type_alias(acc, ctx, name);
        }
        NameKind::RecordField => {
            field::complete_field_list_record_variant(acc, ctx);
        }
        NameKind::ConstParam
        | NameKind::Enum
        | NameKind::MacroDef
        | NameKind::MacroRules
        | NameKind::Rename
        | NameKind::SelfParam
        | NameKind::Static
        | NameKind::Struct
        | NameKind::Trait
        | NameKind::TypeParam
        | NameKind::Union
        | NameKind::Variant => (),
    }
}

pub(super) fn complete_name_ref(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    NameRefContext { nameref, kind }: &NameRefContext,
) {
    match kind {
        NameRefKind::Path(path_ctx) => {
            flyimport::import_on_the_fly_path(acc, ctx, path_ctx);

            match &path_ctx.kind {
                PathKind::Expr { expr_ctx } => {
                    expr::complete_expr_path(acc, ctx, path_ctx, expr_ctx);

                    dot::complete_undotted_self(acc, ctx, path_ctx, expr_ctx);
                    item_list::complete_item_list_in_expr(acc, ctx, path_ctx, expr_ctx);
                    snippet::complete_expr_snippet(acc, ctx, path_ctx, expr_ctx);
                }
                PathKind::Type { location } => {
                    r#type::complete_type_path(acc, ctx, path_ctx, location);

                    match location {
                        TypeLocation::TupleField => {
                            field::complete_field_list_tuple_variant(acc, ctx, path_ctx);
                        }
                        TypeLocation::TypeAscription(ascription) => {
                            r#type::complete_ascribed_type(acc, ctx, path_ctx, ascription);
                        }
                        TypeLocation::GenericArgList(_)
                        | TypeLocation::TypeBound
                        | TypeLocation::ImplTarget
                        | TypeLocation::ImplTrait
                        | TypeLocation::Other => (),
                    }
                }
                PathKind::Attr { attr_ctx } => {
                    attribute::complete_attribute_path(acc, ctx, path_ctx, attr_ctx);
                }
                PathKind::Derive { existing_derives } => {
                    attribute::complete_derive_path(acc, ctx, path_ctx, existing_derives);
                }
                PathKind::Item { kind } => {
                    item_list::complete_item_list(acc, ctx, path_ctx, kind);

                    snippet::complete_item_snippet(acc, ctx, path_ctx, kind);
                    if let ItemListKind::TraitImpl(impl_) = kind {
                        item_list::trait_impl::complete_trait_impl_item_by_name(
                            acc, ctx, path_ctx, nameref, impl_,
                        );
                    }
                }
                PathKind::Pat { .. } => {
                    pattern::complete_pattern_path(acc, ctx, path_ctx);
                }
                PathKind::Vis { has_in_token } => {
                    vis::complete_vis_path(acc, ctx, path_ctx, has_in_token);
                }
                PathKind::Use => {
                    use_::complete_use_path(acc, ctx, path_ctx, nameref);
                }
            }
        }
        NameRefKind::DotAccess(dot_access) => {
            flyimport::import_on_the_fly_dot(acc, ctx, dot_access);
            dot::complete_dot(acc, ctx, dot_access);
            postfix::complete_postfix(acc, ctx, dot_access);
        }
        NameRefKind::Keyword(item) => {
            keyword::complete_for_and_where(acc, ctx, item);
        }
        NameRefKind::RecordExpr { dot_prefix, expr } => {
            record::complete_record_expr_fields(acc, ctx, expr, dot_prefix);
        }
        NameRefKind::Pattern(pattern_ctx) => complete_patterns(acc, ctx, pattern_ctx),
    }
}

fn complete_patterns(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    pattern_ctx: &PatternContext,
) {
    flyimport::import_on_the_fly_pat(acc, ctx, pattern_ctx);
    fn_param::complete_fn_param(acc, ctx, pattern_ctx);
    pattern::complete_pattern(acc, ctx, pattern_ctx);
    record::complete_record_pattern_fields(acc, ctx, pattern_ctx);
}
