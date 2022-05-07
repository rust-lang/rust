//! Completion of paths and keywords at item list position.

use crate::{
    completions::module_or_fn_macro,
    context::{PathCompletionCtx, PathKind, PathQualifierCtx},
    CompletionContext, Completions,
};

pub(crate) fn complete_item_list(acc: &mut Completions, ctx: &CompletionContext) {
    let _p = profile::span("complete_item_list");
    if ctx.is_path_disallowed() || ctx.has_unfinished_impl_or_trait_prev_sibling() {
        return;
    }

    let (&is_absolute_path, qualifier) = match ctx.path_context() {
        Some(PathCompletionCtx {
            kind: PathKind::Item { .. },
            is_absolute_path,
            qualifier,
            ..
        }) => (is_absolute_path, qualifier),
        _ => return,
    };

    match qualifier {
        Some(PathQualifierCtx { resolution, is_super_chain, .. }) => {
            if let Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))) = resolution {
                for (name, def) in module.scope(ctx.db, Some(ctx.module)) {
                    if let Some(def) = module_or_fn_macro(ctx.db, def) {
                        acc.add_resolution(ctx, name, def);
                    }
                }
            }

            if *is_super_chain {
                acc.add_keyword(ctx, "super::");
            }
        }
        None if is_absolute_path => {
            acc.add_crate_roots(ctx);
        }
        None => {
            ctx.process_all_names(&mut |name, def| {
                if let Some(def) = module_or_fn_macro(ctx.db, def) {
                    acc.add_resolution(ctx, name, def);
                }
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
    }
}
