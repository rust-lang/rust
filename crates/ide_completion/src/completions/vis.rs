//! Completion for visibility specifiers.

use hir::ScopeDef;

use crate::{
    context::{CompletionContext, PathCompletionCtx, PathKind, PathQualifierCtx},
    Completions,
};

pub(crate) fn complete_vis(acc: &mut Completions, ctx: &CompletionContext) {
    let (is_absolute_path, qualifier, has_in_token) = match ctx.path_context {
        Some(PathCompletionCtx {
            kind: Some(PathKind::Vis { has_in_token }),
            is_absolute_path,
            ref qualifier,
            ..
        }) => (is_absolute_path, qualifier, has_in_token),
        _ => return,
    };

    match qualifier {
        Some(PathQualifierCtx { resolution, is_super_chain, .. }) => {
            // Try completing next child module of the path that is still a parent of the current module
            if let Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))) = resolution {
                if let Some(current_module) = ctx.module {
                    let next_towards_current = current_module
                        .path_to_root(ctx.db)
                        .into_iter()
                        .take_while(|it| it != module)
                        .last();
                    if let Some(next) = next_towards_current {
                        if let Some(name) = next.name(ctx.db) {
                            cov_mark::hit!(visibility_qualified);
                            acc.add_resolution(ctx, name, ScopeDef::ModuleDef(next.into()));
                        }
                    }
                }
            }

            if *is_super_chain {
                acc.add_keyword(ctx, "super::");
            }
        }
        None if !is_absolute_path => {
            if !has_in_token {
                cov_mark::hit!(kw_completion_in);
                acc.add_keyword(ctx, "in");
            }
            ["self", "super", "crate"].into_iter().for_each(|kw| acc.add_keyword(ctx, kw));
        }
        _ => {}
    }
}
