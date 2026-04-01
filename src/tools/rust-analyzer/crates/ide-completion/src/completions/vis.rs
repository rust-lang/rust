//! Completion for visibility specifiers.

use crate::{
    Completions,
    context::{CompletionContext, PathCompletionCtx, Qualified},
};

pub(crate) fn complete_vis_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx<'_>,
    &has_in_token: &bool,
) {
    match qualified {
        Qualified::With {
            resolution: Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))),
            super_chain_len,
            ..
        } => {
            // Try completing next child module of the path that is still a parent of the current module
            let next_towards_current =
                ctx.module.path_to_root(ctx.db).into_iter().take_while(|it| it != module).last();
            if let Some(next) = next_towards_current
                && let Some(name) = next.name(ctx.db)
            {
                cov_mark::hit!(visibility_qualified);
                acc.add_module(ctx, path_ctx, next, name, vec![]);
            }

            acc.add_super_keyword(ctx, *super_chain_len);
        }
        Qualified::Absolute | Qualified::TypeAnchor { .. } | Qualified::With { .. } => {}
        Qualified::No => {
            if !has_in_token {
                cov_mark::hit!(kw_completion_in);
                acc.add_keyword_snippet(ctx, "in", "in $0");
            }
            acc.add_nameref_keywords(ctx);
        }
    }
}
