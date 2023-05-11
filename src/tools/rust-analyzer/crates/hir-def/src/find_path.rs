//! An algorithm to find a path to refer to a certain item.

use std::{cmp::Ordering, iter};

use hir_expand::name::{known, AsName, Name};
use rustc_hash::FxHashSet;

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    nameres::DefMap,
    path::{ModPath, PathKind},
    visibility::Visibility,
    ModuleDefId, ModuleId,
};

/// Find a path that can be used to refer to a certain item. This can depend on
/// *from where* you're referring to the item, hence the `from` parameter.
pub fn find_path(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    prefer_no_std: bool,
) -> Option<ModPath> {
    let _p = profile::span("find_path");
    find_path_inner(db, item, from, None, prefer_no_std)
}

pub fn find_path_prefixed(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    prefix_kind: PrefixKind,
    prefer_no_std: bool,
) -> Option<ModPath> {
    let _p = profile::span("find_path_prefixed");
    find_path_inner(db, item, from, Some(prefix_kind), prefer_no_std)
}

const MAX_PATH_LEN: usize = 15;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrefixKind {
    /// Causes paths to always start with either `self`, `super`, `crate` or a crate-name.
    /// This is the same as plain, just that paths will start with `self` iprepended f the path
    /// starts with an identifier that is not a crate.
    BySelf,
    /// Causes paths to ignore imports in the local module.
    Plain,
    /// Causes paths to start with `crate` where applicable, effectively forcing paths to be absolute.
    ByCrate,
}

impl PrefixKind {
    #[inline]
    fn prefix(self) -> PathKind {
        match self {
            PrefixKind::BySelf => PathKind::Super(0),
            PrefixKind::Plain => PathKind::Plain,
            PrefixKind::ByCrate => PathKind::Crate,
        }
    }

    #[inline]
    fn is_absolute(&self) -> bool {
        self == &PrefixKind::ByCrate
    }
}

/// Attempts to find a path to refer to the given `item` visible from the `from` ModuleId
fn find_path_inner(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    prefixed: Option<PrefixKind>,
    prefer_no_std: bool,
) -> Option<ModPath> {
    // - if the item is a builtin, it's in scope
    if let ItemInNs::Types(ModuleDefId::BuiltinType(builtin)) = item {
        return Some(ModPath::from_segments(PathKind::Plain, Some(builtin.as_name())));
    }

    let def_map = from.def_map(db);
    let crate_root = def_map.crate_root(db);
    // - if the item is a module, jump straight to module search
    if let ItemInNs::Types(ModuleDefId::ModuleId(module_id)) = item {
        let mut visited_modules = FxHashSet::default();
        return find_path_for_module(
            db,
            &def_map,
            &mut visited_modules,
            crate_root,
            from,
            module_id,
            MAX_PATH_LEN,
            prefixed,
            prefer_no_std || db.crate_supports_no_std(crate_root.krate),
        );
    }

    // - if the item is already in scope, return the name under which it is
    let scope_name = find_in_scope(db, &def_map, from, item);
    if prefixed.is_none() {
        if let Some(scope_name) = scope_name {
            return Some(ModPath::from_segments(PathKind::Plain, Some(scope_name)));
        }
    }

    // - if the item is in the prelude, return the name from there
    if let value @ Some(_) = find_in_prelude(db, &crate_root.def_map(db), &def_map, item, from) {
        return value;
    }

    if let Some(ModuleDefId::EnumVariantId(variant)) = item.as_module_def_id() {
        // - if the item is an enum variant, refer to it via the enum
        if let Some(mut path) = find_path_inner(
            db,
            ItemInNs::Types(variant.parent.into()),
            from,
            prefixed,
            prefer_no_std,
        ) {
            let data = db.enum_data(variant.parent);
            path.push_segment(data.variants[variant.local_id].name.clone());
            return Some(path);
        }
        // If this doesn't work, it seems we have no way of referring to the
        // enum; that's very weird, but there might still be a reexport of the
        // variant somewhere
    }

    let mut visited_modules = FxHashSet::default();

    calculate_best_path(
        db,
        &def_map,
        &mut visited_modules,
        crate_root,
        MAX_PATH_LEN,
        item,
        from,
        prefixed,
        prefer_no_std || db.crate_supports_no_std(crate_root.krate),
        scope_name,
    )
}

fn find_path_for_module(
    db: &dyn DefDatabase,
    def_map: &DefMap,
    visited_modules: &mut FxHashSet<ModuleId>,
    crate_root: ModuleId,
    from: ModuleId,
    module_id: ModuleId,
    max_len: usize,
    prefixed: Option<PrefixKind>,
    prefer_no_std: bool,
) -> Option<ModPath> {
    if max_len == 0 {
        return None;
    }

    // Base cases:
    // - if the item is already in scope, return the name under which it is
    let scope_name = find_in_scope(db, def_map, from, ItemInNs::Types(module_id.into()));
    if prefixed.is_none() {
        if let Some(scope_name) = scope_name {
            return Some(ModPath::from_segments(PathKind::Plain, Some(scope_name)));
        }
    }

    // - if the item is the crate root, return `crate`
    if module_id == crate_root {
        return Some(ModPath::from_segments(PathKind::Crate, None));
    }

    // - if relative paths are fine, check if we are searching for a parent
    if prefixed.filter(PrefixKind::is_absolute).is_none() {
        if let modpath @ Some(_) = find_self_super(def_map, module_id, from) {
            return modpath;
        }
    }

    // - if the item is the crate root of a dependency crate, return the name from the extern prelude
    let root_def_map = crate_root.def_map(db);
    for (name, &def_id) in root_def_map.extern_prelude() {
        if module_id == def_id {
            let name = scope_name.unwrap_or_else(|| name.clone());

            let name_already_occupied_in_type_ns = def_map
                .with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
                    def_map[local_id]
                        .scope
                        .type_(&name)
                        .filter(|&(id, _)| id != ModuleDefId::ModuleId(def_id))
                })
                .is_some();
            let kind = if name_already_occupied_in_type_ns {
                cov_mark::hit!(ambiguous_crate_start);
                PathKind::Abs
            } else {
                PathKind::Plain
            };
            return Some(ModPath::from_segments(kind, Some(name)));
        }
    }

    if let value @ Some(_) =
        find_in_prelude(db, &root_def_map, &def_map, ItemInNs::Types(module_id.into()), from)
    {
        return value;
    }
    calculate_best_path(
        db,
        def_map,
        visited_modules,
        crate_root,
        max_len,
        ItemInNs::Types(module_id.into()),
        from,
        prefixed,
        prefer_no_std,
        scope_name,
    )
}

fn find_in_scope(
    db: &dyn DefDatabase,
    def_map: &DefMap,
    from: ModuleId,
    item: ItemInNs,
) -> Option<Name> {
    def_map.with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
        def_map[local_id].scope.name_of(item).map(|(name, _)| name.clone())
    })
}

/// Returns single-segment path (i.e. without any prefix) if `item` is found in prelude and its
/// name doesn't clash in current scope.
fn find_in_prelude(
    db: &dyn DefDatabase,
    root_def_map: &DefMap,
    local_def_map: &DefMap,
    item: ItemInNs,
    from: ModuleId,
) -> Option<ModPath> {
    let prelude_module = root_def_map.prelude()?;
    // Preludes in block DefMaps are ignored, only the crate DefMap is searched
    let prelude_def_map = prelude_module.def_map(db);
    let prelude_scope = &prelude_def_map[prelude_module.local_id].scope;
    let (name, vis) = prelude_scope.name_of(item)?;
    if !vis.is_visible_from(db, from) {
        return None;
    }

    // Check if the name is in current scope and it points to the same def.
    let found_and_same_def =
        local_def_map.with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
            let per_ns = def_map[local_id].scope.get(name);
            let same_def = match item {
                ItemInNs::Types(it) => per_ns.take_types()? == it,
                ItemInNs::Values(it) => per_ns.take_values()? == it,
                ItemInNs::Macros(it) => per_ns.take_macros()? == it,
            };
            Some(same_def)
        });

    if found_and_same_def.unwrap_or(true) {
        Some(ModPath::from_segments(PathKind::Plain, Some(name.clone())))
    } else {
        None
    }
}

fn find_self_super(def_map: &DefMap, item: ModuleId, from: ModuleId) -> Option<ModPath> {
    if item == from {
        // - if the item is the module we're in, use `self`
        Some(ModPath::from_segments(PathKind::Super(0), None))
    } else if let Some(parent_id) = def_map[from.local_id].parent {
        // - if the item is the parent module, use `super` (this is not used recursively, since `super::super` is ugly)
        let parent_id = def_map.module_id(parent_id);
        if item == parent_id {
            Some(ModPath::from_segments(PathKind::Super(1), None))
        } else {
            None
        }
    } else {
        None
    }
}

fn calculate_best_path(
    db: &dyn DefDatabase,
    def_map: &DefMap,
    visited_modules: &mut FxHashSet<ModuleId>,
    crate_root: ModuleId,
    max_len: usize,
    item: ItemInNs,
    from: ModuleId,
    mut prefixed: Option<PrefixKind>,
    prefer_no_std: bool,
    scope_name: Option<Name>,
) -> Option<ModPath> {
    if max_len <= 1 {
        return None;
    }
    let mut best_path = None;
    // Recursive case:
    // - otherwise, look for modules containing (reexporting) it and import it from one of those
    if item.krate(db) == Some(from.krate) {
        let mut best_path_len = max_len;
        // Item was defined in the same crate that wants to import it. It cannot be found in any
        // dependency in this case.
        for (module_id, name) in find_local_import_locations(db, item, from) {
            if !visited_modules.insert(module_id) {
                cov_mark::hit!(recursive_imports);
                continue;
            }
            if let Some(mut path) = find_path_for_module(
                db,
                def_map,
                visited_modules,
                crate_root,
                from,
                module_id,
                best_path_len - 1,
                prefixed,
                prefer_no_std,
            ) {
                path.push_segment(name);

                let new_path = match best_path {
                    Some(best_path) => select_best_path(best_path, path, prefer_no_std),
                    None => path,
                };
                best_path_len = new_path.len();
                best_path = Some(new_path);
            }
        }
    } else {
        // Item was defined in some upstream crate. This means that it must be exported from one,
        // too (unless we can't name it at all). It could *also* be (re)exported by the same crate
        // that wants to import it here, but we always prefer to use the external path here.

        let crate_graph = db.crate_graph();
        let extern_paths = crate_graph[from.krate].dependencies.iter().filter_map(|dep| {
            let import_map = db.import_map(dep.crate_id);
            import_map.import_info_for(item).and_then(|info| {
                // Determine best path for containing module and append last segment from `info`.
                // FIXME: we should guide this to look up the path locally, or from the same crate again?
                let mut path = find_path_for_module(
                    db,
                    def_map,
                    visited_modules,
                    crate_root,
                    from,
                    info.container,
                    max_len - 1,
                    prefixed,
                    prefer_no_std,
                )?;
                cov_mark::hit!(partially_imported);
                path.push_segment(info.path.segments.last()?.clone());
                Some(path)
            })
        });

        for path in extern_paths {
            let new_path = match best_path {
                Some(best_path) => select_best_path(best_path, path, prefer_no_std),
                None => path,
            };
            best_path = Some(new_path);
        }
    }
    if let Some(module) = item.module(db) {
        if module.def_map(db).block_id().is_some() && prefixed.is_some() {
            cov_mark::hit!(prefixed_in_block_expression);
            prefixed = Some(PrefixKind::Plain);
        }
    }
    match prefixed.map(PrefixKind::prefix) {
        Some(prefix) => best_path.or_else(|| {
            scope_name.map(|scope_name| ModPath::from_segments(prefix, Some(scope_name)))
        }),
        None => best_path,
    }
}

fn select_best_path(old_path: ModPath, new_path: ModPath, prefer_no_std: bool) -> ModPath {
    const STD_CRATES: [Name; 3] = [known::std, known::core, known::alloc];
    match (old_path.segments().first(), new_path.segments().first()) {
        (Some(old), Some(new)) if STD_CRATES.contains(old) && STD_CRATES.contains(new) => {
            let rank = match prefer_no_std {
                false => |name: &Name| match name {
                    name if name == &known::core => 0,
                    name if name == &known::alloc => 0,
                    name if name == &known::std => 1,
                    _ => unreachable!(),
                },
                true => |name: &Name| match name {
                    name if name == &known::core => 2,
                    name if name == &known::alloc => 1,
                    name if name == &known::std => 0,
                    _ => unreachable!(),
                },
            };
            let nrank = rank(new);
            let orank = rank(old);
            match nrank.cmp(&orank) {
                Ordering::Less => old_path,
                Ordering::Equal => {
                    if new_path.len() < old_path.len() {
                        new_path
                    } else {
                        old_path
                    }
                }
                Ordering::Greater => new_path,
            }
        }
        _ => {
            if new_path.len() < old_path.len() {
                new_path
            } else {
                old_path
            }
        }
    }
}

// FIXME: Remove allocations
/// Finds locations in `from.krate` from which `item` can be imported by `from`.
fn find_local_import_locations(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
) -> Vec<(ModuleId, Name)> {
    let _p = profile::span("find_local_import_locations");

    // `from` can import anything below `from` with visibility of at least `from`, and anything
    // above `from` with any visibility. That means we do not need to descend into private siblings
    // of `from` (and similar).

    let def_map = from.def_map(db);

    // Compute the initial worklist. We start with all direct child modules of `from` as well as all
    // of its (recursive) parent modules.
    let data = &def_map[from.local_id];
    let mut worklist =
        data.children.values().map(|child| def_map.module_id(*child)).collect::<Vec<_>>();
    // FIXME: do we need to traverse out of block expressions here?
    for ancestor in iter::successors(from.containing_module(db), |m| m.containing_module(db)) {
        worklist.push(ancestor);
    }

    let def_map = def_map.crate_root(db).def_map(db);

    let mut seen: FxHashSet<_> = FxHashSet::default();

    let mut locations = Vec::new();
    while let Some(module) = worklist.pop() {
        if !seen.insert(module) {
            continue; // already processed this module
        }

        let ext_def_map;
        let data = if module.krate == from.krate {
            if module.block.is_some() {
                // Re-query the block's DefMap
                ext_def_map = module.def_map(db);
                &ext_def_map[module.local_id]
            } else {
                // Reuse the root DefMap
                &def_map[module.local_id]
            }
        } else {
            // The crate might reexport a module defined in another crate.
            ext_def_map = module.def_map(db);
            &ext_def_map[module.local_id]
        };

        if let Some((name, vis)) = data.scope.name_of(item) {
            if vis.is_visible_from(db, from) {
                let is_private = match vis {
                    Visibility::Module(private_to) => private_to.local_id == module.local_id,
                    Visibility::Public => false,
                };
                let is_original_def = match item.as_module_def_id() {
                    Some(module_def_id) => data.scope.declarations().any(|it| it == module_def_id),
                    None => false,
                };

                // Ignore private imports. these could be used if we are
                // in a submodule of this module, but that's usually not
                // what the user wants; and if this module can import
                // the item and we're a submodule of it, so can we.
                // Also this keeps the cached data smaller.
                if !is_private || is_original_def {
                    locations.push((module, name.clone()));
                }
            }
        }

        // Descend into all modules visible from `from`.
        for (ty, vis) in data.scope.types() {
            if let ModuleDefId::ModuleId(module) = ty {
                if vis.is_visible_from(db, from) {
                    worklist.push(module);
                }
            }
        }
    }

    locations
}

#[cfg(test)]
mod tests {
    use base_db::fixture::WithFixture;
    use hir_expand::hygiene::Hygiene;
    use syntax::ast::AstNode;

    use crate::test_db::TestDB;

    use super::*;

    /// `code` needs to contain a cursor marker; checks that `find_path` for the
    /// item the `path` refers to returns that same path when called from the
    /// module the cursor is in.
    fn check_found_path_(ra_fixture: &str, path: &str, prefix_kind: Option<PrefixKind>) {
        let (db, pos) = TestDB::with_position(ra_fixture);
        let module = db.module_at_position(pos);
        let parsed_path_file = syntax::SourceFile::parse(&format!("use {path};"));
        let ast_path =
            parsed_path_file.syntax_node().descendants().find_map(syntax::ast::Path::cast).unwrap();
        let mod_path = ModPath::from_src(&db, ast_path, &Hygiene::new_unhygienic()).unwrap();

        let def_map = module.def_map(&db);
        let resolved = def_map
            .resolve_path(
                &db,
                module.local_id,
                &mod_path,
                crate::item_scope::BuiltinShadowMode::Module,
            )
            .0
            .take_types()
            .unwrap();

        let found_path =
            find_path_inner(&db, ItemInNs::Types(resolved), module, prefix_kind, false);
        assert_eq!(found_path, Some(mod_path), "{prefix_kind:?}");
    }

    fn check_found_path(
        ra_fixture: &str,
        unprefixed: &str,
        prefixed: &str,
        absolute: &str,
        self_prefixed: &str,
    ) {
        check_found_path_(ra_fixture, unprefixed, None);
        check_found_path_(ra_fixture, prefixed, Some(PrefixKind::Plain));
        check_found_path_(ra_fixture, absolute, Some(PrefixKind::ByCrate));
        check_found_path_(ra_fixture, self_prefixed, Some(PrefixKind::BySelf));
    }

    #[test]
    fn same_module() {
        check_found_path(
            r#"
struct S;
$0
        "#,
            "S",
            "S",
            "crate::S",
            "self::S",
        );
    }

    #[test]
    fn enum_variant() {
        check_found_path(
            r#"
enum E { A }
$0
        "#,
            "E::A",
            "E::A",
            "crate::E::A",
            "self::E::A",
        );
    }

    #[test]
    fn sub_module() {
        check_found_path(
            r#"
mod foo {
    pub struct S;
}
$0
        "#,
            "foo::S",
            "foo::S",
            "crate::foo::S",
            "self::foo::S",
        );
    }

    #[test]
    fn super_module() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
mod bar;
struct S;
//- /foo/bar.rs
$0
        "#,
            "super::S",
            "super::S",
            "crate::foo::S",
            "super::S",
        );
    }

    #[test]
    fn self_module() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
        "#,
            "self",
            "self",
            "crate::foo",
            "self",
        );
    }

    #[test]
    fn crate_root() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
        "#,
            "crate",
            "crate",
            "crate",
            "crate",
        );
    }

    #[test]
    fn same_crate() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
struct S;
//- /foo.rs
$0
        "#,
            "crate::S",
            "crate::S",
            "crate::S",
            "crate::S",
        );
    }

    #[test]
    fn different_crate() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
$0
//- /std.rs crate:std
pub struct S;
        "#,
            "std::S",
            "std::S",
            "std::S",
            "std::S",
        );
    }

    #[test]
    fn different_crate_renamed() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
extern crate std as std_renamed;
$0
//- /std.rs crate:std
pub struct S;
        "#,
            "std_renamed::S",
            "std_renamed::S",
            "std_renamed::S",
            "std_renamed::S",
        );
    }

    #[test]
    fn partially_imported() {
        cov_mark::check!(partially_imported);
        // Tests that short paths are used even for external items, when parts of the path are
        // already in scope.
        check_found_path(
            r#"
//- /main.rs crate:main deps:syntax

use syntax::ast;
$0

//- /lib.rs crate:syntax
pub mod ast {
    pub enum ModuleItem {
        A, B, C,
    }
}
        "#,
            "ast::ModuleItem",
            "syntax::ast::ModuleItem",
            "syntax::ast::ModuleItem",
            "syntax::ast::ModuleItem",
        );

        check_found_path(
            r#"
//- /main.rs crate:main deps:syntax
$0

//- /lib.rs crate:syntax
pub mod ast {
    pub enum ModuleItem {
        A, B, C,
    }
}
        "#,
            "syntax::ast::ModuleItem",
            "syntax::ast::ModuleItem",
            "syntax::ast::ModuleItem",
            "syntax::ast::ModuleItem",
        );
    }

    #[test]
    fn same_crate_reexport() {
        check_found_path(
            r#"
mod bar {
    mod foo { pub(super) struct S; }
    pub(crate) use foo::*;
}
$0
        "#,
            "bar::S",
            "bar::S",
            "crate::bar::S",
            "self::bar::S",
        );
    }

    #[test]
    fn same_crate_reexport_rename() {
        check_found_path(
            r#"
mod bar {
    mod foo { pub(super) struct S; }
    pub(crate) use foo::S as U;
}
$0
        "#,
            "bar::U",
            "bar::U",
            "crate::bar::U",
            "self::bar::U",
        );
    }

    #[test]
    fn different_crate_reexport() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
$0
//- /std.rs crate:std deps:core
pub use core::S;
//- /core.rs crate:core
pub struct S;
        "#,
            "std::S",
            "std::S",
            "std::S",
            "std::S",
        );
    }

    #[test]
    fn prelude() {
        check_found_path(
            r#"
//- /main.rs edition:2018 crate:main deps:std
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct S;
    }
}
        "#,
            "S",
            "S",
            "S",
            "S",
        );
    }

    #[test]
    fn shadowed_prelude() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std
struct S;
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct S;
    }
}
"#,
            "std::prelude::rust_2018::S",
            "std::prelude::rust_2018::S",
            "std::prelude::rust_2018::S",
            "std::prelude::rust_2018::S",
        );
    }

    #[test]
    fn imported_prelude() {
        check_found_path(
            r#"
//- /main.rs edition:2018 crate:main deps:std
use S;
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct S;
    }
}
"#,
            "S",
            "S",
            "S",
            "S",
        );
    }

    #[test]
    fn enum_variant_from_prelude() {
        let code = r#"
//- /main.rs edition:2018 crate:main deps:std
$0
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub enum Option<T> { Some(T), None }
        pub use Option::*;
    }
}
        "#;
        check_found_path(code, "None", "None", "None", "None");
        check_found_path(code, "Some", "Some", "Some", "Some");
    }

    #[test]
    fn shortest_path() {
        check_found_path(
            r#"
//- /main.rs
pub mod foo;
pub mod baz;
struct S;
$0
//- /foo.rs
pub mod bar { pub struct S; }
//- /baz.rs
pub use crate::foo::bar::S;
        "#,
            "baz::S",
            "baz::S",
            "crate::baz::S",
            "self::baz::S",
        );
    }

    #[test]
    fn discount_private_imports() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
pub mod bar { pub struct S; }
use bar::S;
//- /foo.rs
$0
        "#,
            // crate::S would be shorter, but using private imports seems wrong
            "crate::bar::S",
            "crate::bar::S",
            "crate::bar::S",
            "crate::bar::S",
        );
    }

    #[test]
    fn import_cycle() {
        check_found_path(
            r#"
//- /main.rs
pub mod foo;
pub mod bar;
pub mod baz;
//- /bar.rs
$0
//- /foo.rs
pub use super::baz;
pub struct S;
//- /baz.rs
pub use super::foo;
        "#,
            "crate::foo::S",
            "crate::foo::S",
            "crate::foo::S",
            "crate::foo::S",
        );
    }

    #[test]
    fn prefer_std_paths_over_alloc() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:alloc,std
$0

//- /std.rs crate:std deps:alloc
pub mod sync {
    pub use alloc::sync::Arc;
}

//- /zzz.rs crate:alloc
pub mod sync {
    pub struct Arc;
}
        "#,
            "std::sync::Arc",
            "std::sync::Arc",
            "std::sync::Arc",
            "std::sync::Arc",
        );
    }

    #[test]
    fn prefer_core_paths_over_std() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:core,std
#![no_std]

$0

//- /std.rs crate:std deps:core

pub mod fmt {
    pub use core::fmt::Error;
}

//- /zzz.rs crate:core

pub mod fmt {
    pub struct Error;
}
        "#,
            "core::fmt::Error",
            "core::fmt::Error",
            "core::fmt::Error",
            "core::fmt::Error",
        );

        // Should also work (on a best-effort basis) if `no_std` is conditional.
        check_found_path(
            r#"
//- /main.rs crate:main deps:core,std
#![cfg_attr(not(test), no_std)]

$0

//- /std.rs crate:std deps:core

pub mod fmt {
    pub use core::fmt::Error;
}

//- /zzz.rs crate:core

pub mod fmt {
    pub struct Error;
}
        "#,
            "core::fmt::Error",
            "core::fmt::Error",
            "core::fmt::Error",
            "core::fmt::Error",
        );
    }

    #[test]
    fn prefer_alloc_paths_over_std() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:alloc,std
#![no_std]

$0

//- /std.rs crate:std deps:alloc

pub mod sync {
    pub use alloc::sync::Arc;
}

//- /zzz.rs crate:alloc

pub mod sync {
    pub struct Arc;
}
            "#,
            "alloc::sync::Arc",
            "alloc::sync::Arc",
            "alloc::sync::Arc",
            "alloc::sync::Arc",
        );
    }

    #[test]
    fn prefer_shorter_paths_if_not_alloc() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:megaalloc,std
$0

//- /std.rs crate:std deps:megaalloc
pub mod sync {
    pub use megaalloc::sync::Arc;
}

//- /zzz.rs crate:megaalloc
pub struct Arc;
            "#,
            "megaalloc::Arc",
            "megaalloc::Arc",
            "megaalloc::Arc",
            "megaalloc::Arc",
        );
    }

    #[test]
    fn builtins_are_in_scope() {
        let code = r#"
$0

pub mod primitive {
    pub use u8;
}
        "#;
        check_found_path(code, "u8", "u8", "u8", "u8");
        check_found_path(code, "u16", "u16", "u16", "u16");
    }

    #[test]
    fn inner_items() {
        check_found_path(
            r#"
fn main() {
    struct Inner {}
    $0
}
        "#,
            "Inner",
            "Inner",
            "Inner",
            "Inner",
        );
    }

    #[test]
    fn inner_items_from_outer_scope() {
        check_found_path(
            r#"
fn main() {
    struct Struct {}
    {
        $0
    }
}
        "#,
            "Struct",
            "Struct",
            "Struct",
            "Struct",
        );
    }

    #[test]
    fn inner_items_from_inner_module() {
        cov_mark::check!(prefixed_in_block_expression);
        check_found_path(
            r#"
fn main() {
    mod module {
        struct Struct {}
    }
    {
        $0
    }
}
        "#,
            "module::Struct",
            "module::Struct",
            "module::Struct",
            "module::Struct",
        );
    }

    #[test]
    fn outer_items_with_inner_items_present() {
        check_found_path(
            r#"
mod module {
    pub struct CompleteMe;
}

fn main() {
    fn inner() {}
    $0
}
            "#,
            // FIXME: these could use fewer/better prefixes
            "module::CompleteMe",
            "crate::module::CompleteMe",
            "crate::module::CompleteMe",
            "crate::module::CompleteMe",
        )
    }

    #[test]
    fn from_inside_module() {
        // This worked correctly, but the test suite logic was broken.
        cov_mark::check!(submodule_in_testdb);
        check_found_path(
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    fn bar() {
        $0
    }
}
            "#,
            "crate::baz::Foo",
            "crate::baz::Foo",
            "crate::baz::Foo",
            "crate::baz::Foo",
        )
    }

    #[test]
    fn from_inside_module_with_inner_items() {
        check_found_path(
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    fn bar() {
        fn inner() {}
        $0
    }
}
            "#,
            "crate::baz::Foo",
            "crate::baz::Foo",
            "crate::baz::Foo",
            "crate::baz::Foo",
        )
    }

    #[test]
    fn recursive_pub_mod_reexport() {
        cov_mark::check!(recursive_imports);
        check_found_path(
            r#"
fn main() {
    let _ = 22_i32.as_name$0();
}

pub mod name {
    pub trait AsName {
        fn as_name(&self) -> String;
    }
    impl AsName for i32 {
        fn as_name(&self) -> String {
            format!("Name: {}", self)
        }
    }
    pub use crate::name;
}
"#,
            "name::AsName",
            "name::AsName",
            "crate::name::AsName",
            "self::name::AsName",
        );
    }

    #[test]
    fn extern_crate() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:dep
$0
//- /dep.rs crate:dep
"#,
            "dep",
            "dep",
            "dep",
            "dep",
        );

        check_found_path(
            r#"
//- /main.rs crate:main deps:dep
fn f() {
    fn inner() {}
    $0
}
//- /dep.rs crate:dep
"#,
            "dep",
            "dep",
            "dep",
            "dep",
        );
    }

    #[test]
    fn prelude_with_inner_items() {
        check_found_path(
            r#"
//- /main.rs edition:2018 crate:main deps:std
fn f() {
    fn inner() {}
    $0
}
//- /std.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub enum Option { None }
        pub use Option::*;
    }
}
        "#,
            "None",
            "None",
            "None",
            "None",
        );
    }
}
