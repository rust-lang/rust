//! Completion for use trees

use hir::ScopeDef;
use ide_db::{FxHashSet, SymbolKind};
use syntax::{AstNode, ast, format_smolstr};

use crate::{
    CompletionItem, CompletionItemKind, CompletionRelevance, Completions,
    context::{CompletionContext, PathCompletionCtx, Qualified},
    item::Builder,
};

pub(crate) fn complete_use_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, use_tree_parent, .. }: &PathCompletionCtx<'_>,
    name_ref: &Option<ast::NameRef>,
) {
    match qualified {
        Qualified::With { path, resolution: Some(resolution), super_chain_len } => {
            acc.add_super_keyword(ctx, *super_chain_len);

            // only show `self` in a new use-tree when the qualifier doesn't end in self
            let not_preceded_by_self = *use_tree_parent
                && !matches!(
                    path.segment().and_then(|it| it.kind()),
                    Some(ast::PathSegmentKind::SelfKw)
                );
            if not_preceded_by_self {
                acc.add_keyword(ctx, "self");
            }

            let mut already_imported_names = FxHashSet::default();
            if let Some(list) = ctx.token.parent_ancestors().find_map(ast::UseTreeList::cast) {
                let use_tree = list.parent_use_tree();
                if use_tree.path().as_ref() == Some(path) {
                    for tree in list.use_trees().filter(|tree| tree.is_simple_path()) {
                        if let Some(name) = tree.path().and_then(|path| path.as_single_name_ref()) {
                            already_imported_names.insert(name.to_string());
                        }
                    }
                }
            }

            match resolution {
                hir::PathResolution::Def(hir::ModuleDef::Module(module)) => {
                    let module_scope = module.scope(ctx.db, Some(ctx.module));
                    let unknown_is_current = |name: &hir::Name| {
                        matches!(
                            name_ref,
                            Some(name_ref) if name_ref.syntax().text() == name.as_str()
                        )
                    };
                    for (name, def) in module_scope {
                        if let (Some(attrs), Some(defining_crate)) =
                            (def.attrs(ctx.db), def.krate(ctx.db))
                        {
                            if !ctx.check_stability(Some(&attrs))
                                || ctx.is_doc_hidden(&attrs, defining_crate)
                            {
                                continue;
                            }
                        }
                        let is_name_already_imported =
                            already_imported_names.contains(name.as_str());

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
                            let mut builder = Builder::from_resolution(ctx, path_ctx, name, def);
                            builder.with_relevance(|r| CompletionRelevance {
                                is_name_already_imported,
                                ..r
                            });
                            acc.add(builder.build(ctx.db));
                        }
                    }
                }
                hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                    cov_mark::hit!(enum_plain_qualified_use_tree);
                    acc.add_enum_variants(ctx, path_ctx, *e);
                }
                _ => {}
            }
        }
        // fresh use tree with leading colon2, only show crate roots
        Qualified::Absolute => {
            cov_mark::hit!(use_tree_crate_roots_only);
            acc.add_crate_roots(ctx, path_ctx);
        }
        // only show modules and non-std enum in a fresh UseTree
        Qualified::No => {
            cov_mark::hit!(unqualified_path_selected_only);
            ctx.process_all_names(&mut |name, res, doc_aliases| {
                match res {
                    ScopeDef::ModuleDef(hir::ModuleDef::Module(module)) => {
                        acc.add_module(ctx, path_ctx, module, name, doc_aliases);
                    }
                    ScopeDef::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                        // exclude prelude enum
                        let is_builtin =
                            res.krate(ctx.db).is_some_and(|krate| krate.is_builtin(ctx.db));

                        if !is_builtin {
                            let item = CompletionItem::new(
                                CompletionItemKind::SymbolKind(SymbolKind::Enum),
                                ctx.source_range(),
                                format_smolstr!(
                                    "{}::",
                                    e.name(ctx.db).display(ctx.db, ctx.edition)
                                ),
                                ctx.edition,
                            );
                            acc.add(item.build(ctx.db));
                        }
                    }
                    _ => {}
                };
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::TypeAnchor { .. } | Qualified::With { resolution: None, .. } => {}
    }
}
