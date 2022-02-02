//! Completion for visibility specifiers.

use std::iter;

use hir::ScopeDef;

use crate::{
    context::{CompletionContext, PathCompletionContext, PathKind},
    Completions,
};

pub(crate) fn complete_vis(acc: &mut Completions, ctx: &CompletionContext) {
    let (is_trivial_path, qualifier, has_in_token) = match ctx.path_context {
        Some(PathCompletionContext {
            kind: Some(PathKind::Vis { has_in_token }),
            is_trivial_path,
            ref qualifier,
            ..
        }) => (is_trivial_path, qualifier, has_in_token),
        _ => return,
    };

    match qualifier {
        Some((path, qualifier)) => {
            if let Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))) = qualifier {
                if let Some(current_module) = ctx.module {
                    let next_towards_current = current_module
                        .path_to_root(ctx.db)
                        .into_iter()
                        .take_while(|it| it != module)
                        .next();
                    if let Some(next) = next_towards_current {
                        if let Some(name) = next.name(ctx.db) {
                            cov_mark::hit!(visibility_qualified);
                            acc.add_resolution(ctx, name, ScopeDef::ModuleDef(next.into()));
                        }
                    }
                }
            }

            let is_super_chain = iter::successors(Some(path.clone()), |p| p.qualifier())
                .all(|p| p.segment().and_then(|s| s.super_token()).is_some());
            if is_super_chain {
                acc.add_keyword(ctx, "super::");
            }
        }
        None if is_trivial_path => {
            if !has_in_token {
                cov_mark::hit!(kw_completion_in);
                acc.add_keyword(ctx, "in");
            }
            ["self", "super", "crate"].into_iter().for_each(|kw| acc.add_keyword(ctx, kw));
        }
        _ => {}
    }
}
