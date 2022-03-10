//! Completion for derives
use hir::{HasAttrs, Macro};
use ide_db::SymbolKind;
use itertools::Itertools;
use syntax::SmolStr;

use crate::{
    context::{CompletionContext, PathCompletionCtx, PathKind},
    item::CompletionItem,
    Completions,
};

pub(crate) fn complete_derive(acc: &mut Completions, ctx: &CompletionContext) {
    match ctx.path_context {
        // FIXME: Enable qualified completions
        Some(PathCompletionCtx { kind: Some(PathKind::Derive), qualifier: None, .. }) => (),
        _ => return,
    }

    let core = ctx.famous_defs().core();

    for (name, mac) in get_derives_in_scope(ctx) {
        if ctx.existing_derives.contains(&mac) {
            continue;
        }

        let name = name.to_smol_str();
        let (label, lookup) = match (core, mac.module(ctx.db).krate()) {
            // show derive dependencies for `core`/`std` derives
            (Some(core), mac_krate) if core == mac_krate => {
                if let Some(derive_completion) = DEFAULT_DERIVE_DEPENDENCIES
                    .iter()
                    .find(|derive_completion| derive_completion.label == name)
                {
                    let mut components = vec![derive_completion.label];
                    components.extend(derive_completion.dependencies.iter().filter(
                        |&&dependency| {
                            !ctx.existing_derives
                                .iter()
                                .map(|it| it.name(ctx.db))
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
}

fn get_derives_in_scope(ctx: &CompletionContext) -> Vec<(hir::Name, Macro)> {
    let mut result = Vec::default();
    ctx.process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) = scope_def {
            if mac.kind(ctx.db) == hir::MacroKind::Derive {
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
