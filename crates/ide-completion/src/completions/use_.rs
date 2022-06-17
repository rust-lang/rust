//! Completion for use trees

use hir::ScopeDef;
use ide_db::{FxHashSet, SymbolKind};
use syntax::{ast, AstNode};

use crate::{
    context::{
        CompletionContext, NameRefContext, NameRefKind, PathCompletionCtx, PathKind, Qualified,
    },
    item::Builder,
    CompletionItem, CompletionItemKind, CompletionRelevance, Completions,
};

pub(crate) fn complete_use_tree(acc: &mut Completions, ctx: &CompletionContext) {
    let (qualified, name_ref, use_tree_parent) = match ctx.nameref_ctx() {
        Some(NameRefContext {
            kind:
                Some(NameRefKind::Path(PathCompletionCtx {
                    kind: PathKind::Use,
                    qualified,
                    use_tree_parent,
                    ..
                })),
            nameref,
            ..
        }) => (qualified, nameref, use_tree_parent),
        _ => return,
    };

    match qualified {
        Qualified::With { path, resolution, is_super_chain } => {
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
                            Some(name_ref) if name_ref.syntax().text() == name.to_smol_str().as_str()
                        )
                    };
                    for (name, def) in module_scope {
                        let is_name_already_imported = name
                            .as_text()
                            .map_or(false, |text| already_imported_names.contains(text.as_str()));

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
                            let mut builder = Builder::from_resolution(ctx, name, def);
                            builder.set_relevance(CompletionRelevance {
                                is_name_already_imported,
                                ..Default::default()
                            });
                            acc.add(builder.build());
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
        Qualified::Absolute => {
            cov_mark::hit!(use_tree_crate_roots_only);
            acc.add_crate_roots(ctx);
        }
        // only show modules and non-std enum in a fresh UseTree
        Qualified::No => {
            cov_mark::hit!(unqualified_path_selected_only);
            ctx.process_all_names(&mut |name, res| {
                match res {
                    ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) => {
                        acc.add_resolution(ctx, name, res);
                    }
                    ScopeDef::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                        // exclude prelude enum
                        let is_builtin =
                            res.krate(ctx.db).map_or(false, |krate| krate.is_builtin(ctx.db));

                        if !is_builtin {
                            let item = CompletionItem::new(
                                CompletionItemKind::SymbolKind(SymbolKind::Enum),
                                ctx.source_range(),
                                format!("{}::", e.name(ctx.db)),
                            );
                            acc.add(item.build());
                        }
                    }
                    _ => {}
                };
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::Infer => {}
    }
}
