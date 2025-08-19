//! Completion of paths and keywords at item list position.

use crate::{
    CompletionContext, Completions,
    context::{ItemListKind, PathCompletionCtx, PathExprCtx, Qualified},
};

pub(crate) mod trait_impl;

pub(crate) fn complete_item_list_in_expr(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx<'_>,
    expr_ctx: &PathExprCtx<'_>,
) {
    if !expr_ctx.in_block_expr {
        return;
    }
    if !path_ctx.is_trivial_path() {
        return;
    }
    add_keywords(acc, ctx, None);
}

pub(crate) fn complete_item_list(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx<'_>,
    kind: &ItemListKind,
) {
    let _p = tracing::info_span!("complete_item_list").entered();

    // We handle completions for trait-impls in [`item_list::trait_impl`]
    if path_ctx.is_trivial_path() && !matches!(kind, ItemListKind::TraitImpl(_)) {
        add_keywords(acc, ctx, Some(kind));
    }

    match qualified {
        Qualified::With {
            resolution: Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))),
            super_chain_len,
            ..
        } => {
            for (name, def) in module.scope(ctx.db, Some(ctx.module)) {
                match def {
                    hir::ScopeDef::ModuleDef(hir::ModuleDef::Macro(m)) if m.is_fn_like(ctx.db) => {
                        acc.add_macro(ctx, path_ctx, m, name)
                    }
                    hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                        acc.add_module(ctx, path_ctx, m, name, vec![])
                    }
                    _ => (),
                }
            }

            acc.add_super_keyword(ctx, *super_chain_len);
        }
        Qualified::Absolute => acc.add_crate_roots(ctx, path_ctx),
        Qualified::No if ctx.qualifier_ctx.none() => {
            ctx.process_all_names(&mut |name, def, doc_aliases| match def {
                hir::ScopeDef::ModuleDef(hir::ModuleDef::Macro(m)) if m.is_fn_like(ctx.db) => {
                    acc.add_macro(ctx, path_ctx, m, name)
                }
                hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                    acc.add_module(ctx, path_ctx, m, name, doc_aliases)
                }
                _ => (),
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::TypeAnchor { .. } | Qualified::No | Qualified::With { .. } => {}
    }
}

fn add_keywords(acc: &mut Completions, ctx: &CompletionContext<'_>, kind: Option<&ItemListKind>) {
    let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);

    let in_item_list = matches!(kind, Some(ItemListKind::SourceFile | ItemListKind::Module) | None);
    let in_assoc_non_trait_impl = matches!(kind, Some(ItemListKind::Impl | ItemListKind::Trait));

    let in_extern_block = matches!(kind, Some(ItemListKind::ExternBlock { .. }));
    let in_unsafe_extern_block =
        matches!(kind, Some(ItemListKind::ExternBlock { is_unsafe: true }));

    let in_trait = matches!(kind, Some(ItemListKind::Trait));
    let in_inherent_impl = matches!(kind, Some(ItemListKind::Impl));
    let in_block = kind.is_none();

    let no_vis_qualifiers = ctx.qualifier_ctx.vis_node.is_none();
    let has_unsafe_kw = ctx.qualifier_ctx.unsafe_tok.is_some();
    let has_async_kw = ctx.qualifier_ctx.async_tok.is_some();
    let has_safe_kw = ctx.qualifier_ctx.safe_tok.is_some();

    // Some keywords are invalid after non-vis qualifiers, so we handle them first.
    if (has_unsafe_kw || has_safe_kw) && in_extern_block {
        add_keyword("fn", "fn $1($2);");
        add_keyword("static", "static $1: $2;");
        return;
    }

    if has_unsafe_kw || has_async_kw {
        if !has_unsafe_kw {
            add_keyword("unsafe", "unsafe $0");
        }
        if !has_async_kw {
            add_keyword("async", "async $0");
        }

        if in_item_list || in_assoc_non_trait_impl {
            add_keyword("fn", "fn $1($2) {\n    $0\n}");
        }

        if has_unsafe_kw && in_item_list {
            add_keyword("trait", "trait $1 {\n    $0\n}");
            if no_vis_qualifiers {
                add_keyword("impl", "impl $1 {\n    $0\n}");
                add_keyword("impl for", "impl $1 for $2 {\n    $0\n}");
            }
        }

        if !has_async_kw && no_vis_qualifiers && in_item_list {
            add_keyword("extern", "extern $0");
        }

        return;
    }

    // ...and the rest deals with cases without any non-vis qualifiers.

    // Visibility qualifiers
    if !in_trait && !in_block && no_vis_qualifiers {
        add_keyword("pub(crate)", "pub(crate) $0");
        add_keyword("pub(super)", "pub(super) $0");
        add_keyword("pub", "pub $0");
    }

    // Keywords that are valid in `item_list`
    if in_item_list {
        add_keyword("enum", "enum $1 {\n    $0\n}");
        add_keyword("mod", "mod $0");
        add_keyword("static", "static $0");
        add_keyword("struct", "struct $0");
        add_keyword("trait", "trait $1 {\n    $0\n}");
        add_keyword("union", "union $1 {\n    $0\n}");
        add_keyword("use", "use $0;");
        if no_vis_qualifiers {
            add_keyword("impl", "impl $1 {\n    $0\n}");
            add_keyword("impl for", "impl $1 for $2 {\n    $0\n}");
        }
    }

    if in_extern_block {
        add_keyword("unsafe", "unsafe $0");
        if in_unsafe_extern_block {
            add_keyword("safe", "safe $0");
        }

        add_keyword("fn", "fn $1($2);");
        add_keyword("static", "static $1: $2;");
    } else {
        if !in_inherent_impl {
            if !in_trait {
                add_keyword("extern", "extern $0");
            }
            add_keyword("type", "type $0");
        }

        add_keyword("fn", "fn $1($2) {\n    $0\n}");
        add_keyword("unsafe", "unsafe $0");
        add_keyword("const", "const $0");
        add_keyword("async", "async $0");
    }
}
