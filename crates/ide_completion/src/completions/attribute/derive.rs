//! Completion for derives
use hir::{HasAttrs, MacroDef, MacroKind};
use ide_db::{
    helpers::{import_assets::ImportAssets, insert_use::ImportScope},
    SymbolKind,
};
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::{ast, SmolStr, SyntaxKind};

use crate::{
    completions::flyimport::compute_fuzzy_completion_order_key, context::CompletionContext,
    item::CompletionItem, Completions, ImportEdit,
};

pub(super) fn complete_derive(acc: &mut Completions, ctx: &CompletionContext, attr: &ast::Attr) {
    let core = ctx.famous_defs().core();
    let existing_derives: FxHashSet<_> =
        ctx.sema.resolve_derive_macro(attr).into_iter().flatten().flatten().collect();

    for (name, mac) in get_derives_in_scope(ctx) {
        if existing_derives.contains(&mac) {
            continue;
        }

        let name = name.to_smol_str();
        let (label, lookup) = match core.zip(mac.module(ctx.db).map(|it| it.krate())) {
            // show derive dependencies for `core`/`std` derives
            Some((core, mac_krate)) if core == mac_krate => {
                if let Some(derive_completion) = DEFAULT_DERIVE_DEPENDENCIES
                    .iter()
                    .find(|derive_completion| derive_completion.label == name)
                {
                    let mut components = vec![derive_completion.label];
                    components.extend(derive_completion.dependencies.iter().filter(
                        |&&dependency| {
                            !existing_derives
                                .iter()
                                .filter_map(|it| it.name(ctx.db))
                                .any(|it| it.to_smol_str() == dependency)
                        },
                    ));
                    let lookup = components.join(", ");
                    let label = Itertools::intersperse(components.into_iter().rev(), ", ");
                    (SmolStr::from_iter(label), Some(lookup))
                } else {
                    (name, None)
                }
            }
            _ => (name, None),
        };

        let mut item = CompletionItem::new(SymbolKind::Derive, ctx.source_range(), label);
        if let Some(docs) = mac.docs(ctx.db) {
            item.documentation(docs);
        }
        if let Some(lookup) = lookup {
            item.lookup_by(lookup);
        }
        item.add_to(acc);
    }

    flyimport_derive(acc, ctx);
}

fn get_derives_in_scope(ctx: &CompletionContext) -> Vec<(hir::Name, MacroDef)> {
    let mut result = Vec::default();
    ctx.process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::MacroDef(mac) = scope_def {
            if mac.kind() == hir::MacroKind::Derive {
                result.push((name, mac));
            }
        }
    });
    result
}

fn flyimport_derive(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if ctx.token.kind() != SyntaxKind::IDENT {
        return None;
    };
    let potential_import_name = ctx.token.to_string();
    let module = ctx.scope.module()?;
    let parent = ctx.token.parent()?;
    let user_input_lowercased = potential_import_name.to_lowercase();
    let import_assets = ImportAssets::for_fuzzy_path(
        module,
        None,
        potential_import_name,
        &ctx.sema,
        parent.clone(),
    )?;
    let import_scope = ImportScope::find_insert_use_container(&parent, &ctx.sema)?;
    acc.add_all(
        import_assets
            .search_for_imports(&ctx.sema, ctx.config.insert_use.prefix_kind)
            .into_iter()
            .filter_map(|import| match import.original_item {
                hir::ItemInNs::Macros(mac) => Some((import, mac)),
                _ => None,
            })
            .filter(|&(_, mac)| mac.kind() == MacroKind::Derive)
            .filter(|&(_, mac)| !ctx.is_item_hidden(&hir::ItemInNs::Macros(mac)))
            .sorted_by_key(|(import, _)| {
                compute_fuzzy_completion_order_key(&import.import_path, &user_input_lowercased)
            })
            .filter_map(|(import, mac)| {
                let mut item = CompletionItem::new(
                    SymbolKind::Derive,
                    ctx.source_range(),
                    mac.name(ctx.db)?.to_smol_str(),
                );
                item.add_import(ImportEdit { import, scope: import_scope.clone() });
                if let Some(docs) = mac.docs(ctx.db) {
                    item.documentation(docs);
                }
                Some(item.build())
            }),
    );
    Some(())
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
