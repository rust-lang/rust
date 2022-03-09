//! Completion for use trees

use hir::ScopeDef;
use syntax::{ast, AstNode};

use crate::{
    context::{CompletionContext, PathCompletionCtx, PathKind, PathQualifierCtx},
    Completions,
};

pub(crate) fn complete_use_tree(acc: &mut Completions, ctx: &CompletionContext) {
    let (is_absolute_path, qualifier) = match ctx.path_context {
        Some(PathCompletionCtx {
            kind: Some(PathKind::Use),
            is_absolute_path,
            ref qualifier,
            ..
        }) => (is_absolute_path, qualifier),
        _ => return,
    };

    match qualifier {
        Some(PathQualifierCtx { path, resolution, is_super_chain, use_tree_parent }) => {
            if *is_super_chain {
                acc.add_keyword(ctx, "super::");
            }
            // only show `self` in a new use-tree when the qualifier doesn't end in self
            let not_preceded_by_self = *use_tree_parent
                && !matches!(
                    path.segment().and_then(|it| it.kind()),
                    Some(ast::PathSegmentKind::SelfKw)
                );
            if not_preceded_by_self {
                acc.add_keyword(ctx, "self");
            }

            let resolution = match resolution {
                Some(it) => it,
                None => return,
            };

            match resolution {
                hir::PathResolution::Def(hir::ModuleDef::Module(module)) => {
                    let module_scope = module.scope(ctx.db, ctx.module);
                    let unknown_is_current = |name: &hir::Name| {
                        matches!(
                            ctx.name_syntax.as_ref(),
                            Some(ast::NameLike::NameRef(name_ref))
                                if name_ref.syntax().text() == name.to_smol_str().as_str()
                        )
                    };
                    for (name, def) in module_scope {
                        let add_resolution = match def {
                            ScopeDef::Unknown if unknown_is_current(&name) => {
                                // for `use self::foo$0`, don't suggest `foo` as a completion
                                cov_mark::hit!(dont_complete_current_use);
                                continue;
                            }
                            ScopeDef::ModuleDef(_) | ScopeDef::Unknown => true,
                            _ => false,
                        };

                        if add_resolution {
                            acc.add_resolution(ctx, name, def);
                        }
                    }
                }
                hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                    cov_mark::hit!(enum_plain_qualified_use_tree);
                    e.variants(ctx.db)
                        .into_iter()
                        .for_each(|variant| acc.add_enum_variant(ctx, variant, None));
                }
                _ => {}
            }
        }
        // fresh use tree with leading colon2, only show crate roots
        None if is_absolute_path => {
            cov_mark::hit!(use_tree_crate_roots_only);
            acc.add_crate_roots(ctx);
        }
        // only show modules in a fresh UseTree
        None => {
            cov_mark::hit!(unqualified_path_only_modules_in_import);
            ctx.process_all_names(&mut |name, res| {
                if let ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) = res {
                    acc.add_resolution(ctx, name, res);
                }
            });
            acc.add_nameref_keywords(ctx);
        }
    }
}
