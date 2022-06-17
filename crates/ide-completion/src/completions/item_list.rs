//! Completion of paths and keywords at item list position.

use crate::{
    completions::module_or_fn_macro,
    context::{ItemListKind, PathCompletionCtx, PathKind, Qualified},
    CompletionContext, Completions,
};

mod trait_impl;

pub(crate) fn complete_item_list(acc: &mut Completions, ctx: &CompletionContext) {
    let _p = profile::span("complete_item_list");

    if let Some(_) = ctx.name_ctx() {
        trait_impl::complete_trait_impl(acc, ctx);
        return;
    }

    let (qualified, kind, is_trivial_path) = match ctx.path_context() {
        Some(ctx @ PathCompletionCtx { kind: PathKind::Item { kind }, qualified, .. }) => {
            (qualified, Some(kind), ctx.is_trivial_path())
        }
        Some(
            ctx @ PathCompletionCtx {
                kind: PathKind::Expr { in_block_expr: true, .. },
                qualified,
                ..
            },
        ) => (qualified, None, ctx.is_trivial_path()),
        _ => return,
    };

    if matches!(kind, Some(ItemListKind::TraitImpl)) {
        trait_impl::complete_trait_impl(acc, ctx);
    }

    if is_trivial_path {
        add_keywords(acc, ctx, kind);
    }

    if kind.is_none() {
        // this is already handled by expression
        return;
    }

    match qualified {
        Qualified::With {
            resolution: Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))),
            is_super_chain,
            ..
        } => {
            for (name, def) in module.scope(ctx.db, Some(ctx.module)) {
                if let Some(def) = module_or_fn_macro(ctx.db, def) {
                    acc.add_resolution(ctx, name, def);
                }
            }

            if *is_super_chain {
                acc.add_keyword(ctx, "super::");
            }
        }
        Qualified::Absolute => acc.add_crate_roots(ctx),
        Qualified::No if ctx.qualifier_ctx.none() => {
            ctx.process_all_names(&mut |name, def| {
                if let Some(def) = module_or_fn_macro(ctx.db, def) {
                    acc.add_resolution(ctx, name, def);
                }
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::Infer | Qualified::No | Qualified::With { .. } => {}
    }
}

fn add_keywords(acc: &mut Completions, ctx: &CompletionContext, kind: Option<&ItemListKind>) {
    let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);

    let in_item_list = matches!(kind, Some(ItemListKind::SourceFile | ItemListKind::Module) | None);
    let in_assoc_non_trait_impl = matches!(kind, Some(ItemListKind::Impl | ItemListKind::Trait));
    let in_extern_block = matches!(kind, Some(ItemListKind::ExternBlock));
    let in_trait = matches!(kind, Some(ItemListKind::Trait));
    let in_trait_impl = matches!(kind, Some(ItemListKind::TraitImpl));
    let in_inherent_impl = matches!(kind, Some(ItemListKind::Impl));
    let no_qualifiers = ctx.qualifier_ctx.vis_node.is_none();
    let in_block = matches!(kind, None);

    if !in_trait_impl {
        if ctx.qualifier_ctx.unsafe_tok.is_some() {
            if in_item_list || in_assoc_non_trait_impl {
                add_keyword("fn", "fn $1($2) {\n    $0\n}");
            }
            if in_item_list {
                add_keyword("trait", "trait $1 {\n    $0\n}");
                if no_qualifiers {
                    add_keyword("impl", "impl $1 {\n    $0\n}");
                }
            }
            return;
        }

        if in_item_list {
            add_keyword("enum", "enum $1 {\n    $0\n}");
            add_keyword("mod", "mod $0");
            add_keyword("static", "static $0");
            add_keyword("struct", "struct $0");
            add_keyword("trait", "trait $1 {\n    $0\n}");
            add_keyword("union", "union $1 {\n    $0\n}");
            add_keyword("use", "use $0");
            if no_qualifiers {
                add_keyword("impl", "impl $1 {\n    $0\n}");
            }
        }

        if !in_trait && !in_block && no_qualifiers {
            add_keyword("pub(crate)", "pub(crate)");
            add_keyword("pub(super)", "pub(super)");
            add_keyword("pub", "pub");
        }

        if in_extern_block {
            add_keyword("fn", "fn $1($2);");
        } else {
            if !in_inherent_impl {
                if !in_trait {
                    add_keyword("extern", "extern $0");
                }
                add_keyword("type", "type $0");
            }

            add_keyword("fn", "fn $1($2) {\n    $0\n}");
            add_keyword("unsafe", "unsafe");
            add_keyword("const", "const $0");
        }
    }
}
