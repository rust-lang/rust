//! Completion for derives
use hir::{HasAttrs, MacroDef, MacroKind};
use ide_db::helpers::FamousDefs;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use syntax::ast;

use crate::{
    context::CompletionContext,
    item::{CompletionItem, CompletionItemKind},
    Completions,
};

pub(super) fn complete_derive(
    acc: &mut Completions,
    ctx: &CompletionContext,
    existing_derives: &[ast::Path],
) {
    let core = FamousDefs(&ctx.sema, ctx.krate).core();
    let existing_derives: FxHashSet<_> = existing_derives
        .into_iter()
        .filter_map(|path| ctx.scope.speculative_resolve_as_mac(&path))
        .filter(|mac| mac.kind() == MacroKind::Derive)
        .collect();

    for (name, mac) in get_derives_in_scope(ctx) {
        if existing_derives.contains(&mac) {
            continue;
        }

        let name = name.to_smol_str();
        let label;
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
                    label = components.iter().rev().join(", ");
                    (label.as_str(), Some(lookup))
                } else {
                    (&*name, None)
                }
            }
            _ => (&*name, None),
        };

        let mut item =
            CompletionItem::new(CompletionItemKind::Attribute, ctx.source_range(), label);
        if let Some(docs) = mac.docs(ctx.db) {
            item.documentation(docs);
        }
        if let Some(lookup) = lookup {
            item.lookup_by(lookup);
        }
        item.add_to(acc);
    }
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
