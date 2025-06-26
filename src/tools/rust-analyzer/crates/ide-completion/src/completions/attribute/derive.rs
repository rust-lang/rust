//! Completion for derives
use hir::ScopeDef;
use ide_db::{SymbolKind, documentation::HasDocs};
use itertools::Itertools;
use syntax::{SmolStr, ToSmolStr};

use crate::{
    Completions,
    context::{CompletionContext, ExistingDerives, PathCompletionCtx, Qualified},
    item::CompletionItem,
};

pub(crate) fn complete_derive_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx<'_>,
    existing_derives: &ExistingDerives,
) {
    let core = ctx.famous_defs().core();

    match qualified {
        Qualified::With {
            resolution: Some(hir::PathResolution::Def(hir::ModuleDef::Module(module))),
            super_chain_len,
            ..
        } => {
            acc.add_super_keyword(ctx, *super_chain_len);

            for (name, def) in module.scope(ctx.db, Some(ctx.module)) {
                match def {
                    ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac))
                        if !existing_derives.contains(&mac) && mac.is_derive(ctx.db) =>
                    {
                        acc.add_macro(ctx, path_ctx, mac, name)
                    }
                    ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                        acc.add_module(ctx, path_ctx, m, name, vec![])
                    }
                    _ => (),
                }
            }
        }
        Qualified::Absolute => acc.add_crate_roots(ctx, path_ctx),
        // only show modules in a fresh UseTree
        Qualified::No => {
            ctx.process_all_names(&mut |name, def, doc_aliases| {
                let mac = match def {
                    ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac))
                        if !existing_derives.contains(&mac) && mac.is_derive(ctx.db) =>
                    {
                        mac
                    }
                    ScopeDef::ModuleDef(hir::ModuleDef::Module(m)) => {
                        return acc.add_module(ctx, path_ctx, m, name, doc_aliases);
                    }
                    _ => return,
                };

                match (core, mac.module(ctx.db).krate()) {
                    // show derive dependencies for `core`/`std` derives
                    (Some(core), mac_krate) if core == mac_krate => {}
                    _ => return acc.add_macro(ctx, path_ctx, mac, name),
                };

                let name_ = name.display_no_db(ctx.edition).to_smolstr();
                let find = DEFAULT_DERIVE_DEPENDENCIES
                    .iter()
                    .find(|derive_completion| derive_completion.label == name_);

                match find {
                    Some(derive_completion) => {
                        let mut components = vec![derive_completion.label];
                        components.extend(derive_completion.dependencies.iter().filter(
                            |&&dependency| {
                                !existing_derives.iter().map(|it| it.name(ctx.db)).any(|it| {
                                    it.display_no_db(ctx.edition).to_smolstr() == dependency
                                })
                            },
                        ));
                        let lookup = components.join(", ");
                        let label = Itertools::intersperse(components.into_iter().rev(), ", ");

                        let mut item = CompletionItem::new(
                            SymbolKind::Derive,
                            ctx.source_range(),
                            SmolStr::from_iter(label),
                            ctx.edition,
                        );
                        if let Some(docs) = mac.docs(ctx.db) {
                            item.documentation(docs);
                        }
                        item.lookup_by(lookup);
                        item.add_to(acc, ctx.db);
                    }
                    None => acc.add_macro(ctx, path_ctx, mac, name),
                }
            });
            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::TypeAnchor { .. } | Qualified::With { .. } => {}
    }
}

struct DeriveDependencies {
    label: &'static str,
    dependencies: &'static [&'static str],
}

/// Standard Rust derives that have dependencies
/// (the dependencies are needed so that the main derive don't break the compilation when added)
const DEFAULT_DERIVE_DEPENDENCIES: &[DeriveDependencies] = &[
    DeriveDependencies { label: "Copy", dependencies: &["Clone"] },
    DeriveDependencies { label: "Eq", dependencies: &["PartialEq"] },
    DeriveDependencies { label: "Ord", dependencies: &["PartialOrd", "Eq", "PartialEq"] },
    DeriveDependencies { label: "PartialOrd", dependencies: &["PartialEq"] },
];
