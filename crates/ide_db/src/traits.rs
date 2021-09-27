//! Functionality for obtaining data related to traits from the DB.

use crate::RootDatabase;
use hir::Semantics;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, HasName},
    AstNode,
};

/// Given the `impl` block, attempts to find the trait this `impl` corresponds to.
pub fn resolve_target_trait(
    sema: &Semantics<RootDatabase>,
    impl_def: &ast::Impl,
) -> Option<hir::Trait> {
    let ast_path =
        impl_def.trait_().map(|it| it.syntax().clone()).and_then(ast::PathType::cast)?.path()?;

    match sema.resolve_path(&ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => Some(def),
        _ => None,
    }
}

/// Given the `impl` block, returns the list of associated items (e.g. functions or types) that are
/// missing in this `impl` block.
pub fn get_missing_assoc_items(
    sema: &Semantics<RootDatabase>,
    impl_def: &ast::Impl,
) -> Vec<hir::AssocItem> {
    // Names must be unique between constants and functions. However, type aliases
    // may share the same name as a function or constant.
    let mut impl_fns_consts = FxHashSet::default();
    let mut impl_type = FxHashSet::default();

    if let Some(item_list) = impl_def.assoc_item_list() {
        for item in item_list.assoc_items() {
            match item {
                ast::AssocItem::Fn(f) => {
                    if let Some(n) = f.name() {
                        impl_fns_consts.insert(n.syntax().to_string());
                    }
                }

                ast::AssocItem::TypeAlias(t) => {
                    if let Some(n) = t.name() {
                        impl_type.insert(n.syntax().to_string());
                    }
                }

                ast::AssocItem::Const(c) => {
                    if let Some(n) = c.name() {
                        impl_fns_consts.insert(n.syntax().to_string());
                    }
                }
                ast::AssocItem::MacroCall(_) => (),
            }
        }
    }

    resolve_target_trait(sema, impl_def).map_or(vec![], |target_trait| {
        target_trait
            .items(sema.db)
            .into_iter()
            .filter(|i| match i {
                hir::AssocItem::Function(f) => {
                    !impl_fns_consts.contains(&f.name(sema.db).to_string())
                }
                hir::AssocItem::TypeAlias(t) => !impl_type.contains(&t.name(sema.db).to_string()),
                hir::AssocItem::Const(c) => c
                    .name(sema.db)
                    .map(|n| !impl_fns_consts.contains(&n.to_string()))
                    .unwrap_or_default(),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests;
