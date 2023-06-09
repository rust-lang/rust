//! Random assortment of ide helpers for high-level ide features that don't fit in any other module.

use std::collections::VecDeque;

use base_db::{FileId, SourceDatabaseExt};
use hir::{Crate, ItemInNs, ModuleDef, Name, Semantics};
use syntax::{
    ast::{self, make},
    AstToken, SyntaxKind, SyntaxToken, TokenAtOffset,
};

use crate::{defs::Definition, generated, RootDatabase};

pub fn item_name(db: &RootDatabase, item: ItemInNs) -> Option<Name> {
    match item {
        ItemInNs::Types(module_def_id) => module_def_id.name(db),
        ItemInNs::Values(module_def_id) => module_def_id.name(db),
        ItemInNs::Macros(macro_def_id) => Some(macro_def_id.name(db)),
    }
}

/// Picks the token with the highest rank returned by the passed in function.
pub fn pick_best_token(
    tokens: TokenAtOffset<SyntaxToken>,
    f: impl Fn(SyntaxKind) -> usize,
) -> Option<SyntaxToken> {
    tokens.max_by_key(move |t| f(t.kind()))
}
pub fn pick_token<T: AstToken>(mut tokens: TokenAtOffset<SyntaxToken>) -> Option<T> {
    tokens.find_map(T::cast)
}

/// Converts the mod path struct into its ast representation.
pub fn mod_path_to_ast(path: &hir::ModPath) -> ast::Path {
    let _p = profile::span("mod_path_to_ast");

    let mut segments = Vec::new();
    let mut is_abs = false;
    match path.kind {
        hir::PathKind::Plain => {}
        hir::PathKind::Super(0) => segments.push(make::path_segment_self()),
        hir::PathKind::Super(n) => segments.extend((0..n).map(|_| make::path_segment_super())),
        hir::PathKind::DollarCrate(_) | hir::PathKind::Crate => {
            segments.push(make::path_segment_crate())
        }
        hir::PathKind::Abs => is_abs = true,
    }

    segments.extend(
        path.segments()
            .iter()
            .map(|segment| make::path_segment(make::name_ref(&segment.to_smol_str()))),
    );
    make::path_from_segments(segments, is_abs)
}

/// Iterates all `ModuleDef`s and `Impl` blocks of the given file.
pub fn visit_file_defs(
    sema: &Semantics<'_, RootDatabase>,
    file_id: FileId,
    cb: &mut dyn FnMut(Definition),
) {
    let db = sema.db;
    let module = match sema.to_module_def(file_id) {
        Some(it) => it,
        None => return,
    };
    let mut defs: VecDeque<_> = module.declarations(db).into();
    while let Some(def) = defs.pop_front() {
        if let ModuleDef::Module(submodule) = def {
            if let hir::ModuleSource::Module(_) = submodule.definition_source(db).value {
                defs.extend(submodule.declarations(db));
                submodule.impl_defs(db).into_iter().for_each(|impl_| cb(impl_.into()));
            }
        }
        cb(def.into());
    }
    module.impl_defs(db).into_iter().for_each(|impl_| cb(impl_.into()));

    let is_root = module.is_crate_root();
    module
        .legacy_macros(db)
        .into_iter()
        // don't show legacy macros declared in the crate-root that were already covered in declarations earlier
        .filter(|it| !(is_root && it.is_macro_export(db)))
        .for_each(|mac| cb(mac.into()));
}

/// Checks if the given lint is equal or is contained by the other lint which may or may not be a group.
pub fn lint_eq_or_in_group(lint: &str, lint_is: &str) -> bool {
    if lint == lint_is {
        return true;
    }

    if let Some(group) = generated::lints::DEFAULT_LINT_GROUPS
        .iter()
        .chain(generated::lints::CLIPPY_LINT_GROUPS.iter())
        .chain(generated::lints::RUSTDOC_LINT_GROUPS.iter())
        .find(|&check| check.lint.label == lint_is)
    {
        group.children.contains(&lint)
    } else {
        false
    }
}

pub fn is_editable_crate(krate: Crate, db: &RootDatabase) -> bool {
    let root_file = krate.root_file(db);
    let source_root_id = db.file_source_root(root_file);
    !db.source_root(source_root_id).is_library
}
