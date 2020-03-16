//! An algorithm to find a path to refer to a certain item.

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    path::{ModPath, PathKind},
    visibility::Visibility,
    CrateId, ModuleDefId, ModuleId,
};
use hir_expand::name::{known, Name};
use test_utils::tested_by;

const MAX_PATH_LEN: usize = 15;

impl ModPath {
    fn starts_with_std(&self) -> bool {
        self.segments.first().filter(|&first_segment| first_segment == &known::std).is_some()
    }

    // When std library is present, paths starting with `std::`
    // should be preferred over paths starting with `core::` and `alloc::`
    fn should_start_with_std(&self) -> bool {
        self.segments
            .first()
            .filter(|&first_segment| {
                first_segment == &known::alloc || first_segment == &known::core
            })
            .is_some()
    }

    fn len(&self) -> usize {
        self.segments.len()
            + match self.kind {
                PathKind::Plain => 0,
                PathKind::Super(i) => i as usize,
                PathKind::Crate => 1,
                PathKind::Abs => 0,
                PathKind::DollarCrate(_) => 1,
            }
    }
}

// FIXME: handle local items

/// Find a path that can be used to refer to a certain item. This can depend on
/// *from where* you're referring to the item, hence the `from` parameter.
pub fn find_path(db: &dyn DefDatabase, item: ItemInNs, from: ModuleId) -> Option<ModPath> {
    find_path_inner(db, item, from, MAX_PATH_LEN)
}

fn find_path_inner(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    max_len: usize,
) -> Option<ModPath> {
    if max_len == 0 {
        return None;
    }

    // Base cases:

    // - if the item is already in scope, return the name under which it is
    let def_map = db.crate_def_map(from.krate);
    let from_scope: &crate::item_scope::ItemScope = &def_map.modules[from.local_id].scope;
    if let Some((name, _)) = from_scope.name_of(item) {
        return Some(ModPath::from_segments(PathKind::Plain, vec![name.clone()]));
    }

    // - if the item is the crate root, return `crate`
    if item
        == ItemInNs::Types(ModuleDefId::ModuleId(ModuleId {
            krate: from.krate,
            local_id: def_map.root,
        }))
    {
        return Some(ModPath::from_segments(PathKind::Crate, Vec::new()));
    }

    // - if the item is the module we're in, use `self`
    if item == ItemInNs::Types(from.into()) {
        return Some(ModPath::from_segments(PathKind::Super(0), Vec::new()));
    }

    // - if the item is the parent module, use `super` (this is not used recursively, since `super::super` is ugly)
    if let Some(parent_id) = def_map.modules[from.local_id].parent {
        if item
            == ItemInNs::Types(ModuleDefId::ModuleId(ModuleId {
                krate: from.krate,
                local_id: parent_id,
            }))
        {
            return Some(ModPath::from_segments(PathKind::Super(1), Vec::new()));
        }
    }

    // - if the item is the crate root of a dependency crate, return the name from the extern prelude
    for (name, def_id) in &def_map.extern_prelude {
        if item == ItemInNs::Types(*def_id) {
            return Some(ModPath::from_segments(PathKind::Plain, vec![name.clone()]));
        }
    }

    // - if the item is in the prelude, return the name from there
    if let Some(prelude_module) = def_map.prelude {
        let prelude_def_map = db.crate_def_map(prelude_module.krate);
        let prelude_scope: &crate::item_scope::ItemScope =
            &prelude_def_map.modules[prelude_module.local_id].scope;
        if let Some((name, vis)) = prelude_scope.name_of(item) {
            if vis.is_visible_from(db, from) {
                return Some(ModPath::from_segments(PathKind::Plain, vec![name.clone()]));
            }
        }
    }

    // Recursive case:
    // - if the item is an enum variant, refer to it via the enum
    if let Some(ModuleDefId::EnumVariantId(variant)) = item.as_module_def_id() {
        if let Some(mut path) = find_path(db, ItemInNs::Types(variant.parent.into()), from) {
            let data = db.enum_data(variant.parent);
            path.segments.push(data.variants[variant.local_id].name.clone());
            return Some(path);
        }
        // If this doesn't work, it seems we have no way of referring to the
        // enum; that's very weird, but there might still be a reexport of the
        // variant somewhere
    }

    // - otherwise, look for modules containing (reexporting) it and import it from one of those
    let importable_locations = find_importable_locations(db, item, from);
    let mut best_path = None;
    let mut best_path_len = max_len;
    for (module_id, name) in importable_locations {
        let mut path = match find_path_inner(
            db,
            ItemInNs::Types(ModuleDefId::ModuleId(module_id)),
            from,
            best_path_len - 1,
        ) {
            None => continue,
            Some(path) => path,
        };
        path.segments.push(name);

        let new_path =
            if let Some(best_path) = best_path { select_best_path(best_path, path) } else { path };
        best_path_len = new_path.len();
        best_path = Some(new_path);
    }
    best_path
}

fn select_best_path(old_path: ModPath, new_path: ModPath) -> ModPath {
    if old_path.starts_with_std() && new_path.should_start_with_std() {
        tested_by!(prefer_std_paths);
        old_path
    } else if new_path.starts_with_std() && old_path.should_start_with_std() {
        tested_by!(prefer_std_paths);
        new_path
    } else if new_path.len() < old_path.len() {
        new_path
    } else {
        old_path
    }
}

fn find_importable_locations(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
) -> Vec<(ModuleId, Name)> {
    let crate_graph = db.crate_graph();
    let mut result = Vec::new();
    // We only look in the crate from which we are importing, and the direct
    // dependencies. We cannot refer to names from transitive dependencies
    // directly (only through reexports in direct dependencies).
    for krate in Some(from.krate)
        .into_iter()
        .chain(crate_graph[from.krate].dependencies.iter().map(|dep| dep.crate_id))
    {
        result.extend(
            importable_locations_in_crate(db, item, krate)
                .iter()
                .filter(|(_, _, vis)| vis.is_visible_from(db, from))
                .map(|(m, n, _)| (*m, n.clone())),
        );
    }
    result
}

/// Collects all locations from which we might import the item in a particular
/// crate. These include the original definition of the item, and any
/// non-private `use`s.
///
/// Note that the crate doesn't need to be the one in which the item is defined;
/// it might be re-exported in other crates.
fn importable_locations_in_crate(
    db: &dyn DefDatabase,
    item: ItemInNs,
    krate: CrateId,
) -> Vec<(ModuleId, Name, Visibility)> {
    let def_map = db.crate_def_map(krate);
    let mut result = Vec::new();
    for (local_id, data) in def_map.modules.iter() {
        if let Some((name, vis)) = data.scope.name_of(item) {
            let is_private = if let Visibility::Module(private_to) = vis {
                private_to.local_id == local_id
            } else {
                false
            };
            let is_original_def = if let Some(module_def_id) = item.as_module_def_id() {
                data.scope.declarations().any(|it| it == module_def_id)
            } else {
                false
            };
            if is_private && !is_original_def {
                // Ignore private imports. these could be used if we are
                // in a submodule of this module, but that's usually not
                // what the user wants; and if this module can import
                // the item and we're a submodule of it, so can we.
                // Also this keeps the cached data smaller.
                continue;
            }
            result.push((ModuleId { krate, local_id }, name.clone(), vis));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_db::TestDB;
    use hir_expand::hygiene::Hygiene;
    use ra_db::fixture::WithFixture;
    use ra_syntax::ast::AstNode;
    use test_utils::covers;

    /// `code` needs to contain a cursor marker; checks that `find_path` for the
    /// item the `path` refers to returns that same path when called from the
    /// module the cursor is in.
    fn check_found_path(code: &str, path: &str) {
        let (db, pos) = TestDB::with_position(code);
        let module = db.module_for_file(pos.file_id);
        let parsed_path_file = ra_syntax::SourceFile::parse(&format!("use {};", path));
        let ast_path = parsed_path_file
            .syntax_node()
            .descendants()
            .find_map(ra_syntax::ast::Path::cast)
            .unwrap();
        let mod_path = ModPath::from_src(ast_path, &Hygiene::new_unhygienic()).unwrap();

        let crate_def_map = db.crate_def_map(module.krate);
        let resolved = crate_def_map
            .resolve_path(
                &db,
                module.local_id,
                &mod_path,
                crate::item_scope::BuiltinShadowMode::Module,
            )
            .0
            .take_types()
            .unwrap();

        let found_path = find_path(&db, ItemInNs::Types(resolved), module);

        assert_eq!(found_path, Some(mod_path));
    }

    #[test]
    fn same_module() {
        let code = r#"
            //- /main.rs
            struct S;
            <|>
        "#;
        check_found_path(code, "S");
    }

    #[test]
    fn enum_variant() {
        let code = r#"
            //- /main.rs
            enum E { A }
            <|>
        "#;
        check_found_path(code, "E::A");
    }

    #[test]
    fn sub_module() {
        let code = r#"
            //- /main.rs
            mod foo {
                pub struct S;
            }
            <|>
        "#;
        check_found_path(code, "foo::S");
    }

    #[test]
    fn super_module() {
        let code = r#"
            //- /main.rs
            mod foo;
            //- /foo.rs
            mod bar;
            struct S;
            //- /foo/bar.rs
            <|>
        "#;
        check_found_path(code, "super::S");
    }

    #[test]
    fn self_module() {
        let code = r#"
            //- /main.rs
            mod foo;
            //- /foo.rs
            <|>
        "#;
        check_found_path(code, "self");
    }

    #[test]
    fn crate_root() {
        let code = r#"
            //- /main.rs
            mod foo;
            //- /foo.rs
            <|>
        "#;
        check_found_path(code, "crate");
    }

    #[test]
    fn same_crate() {
        let code = r#"
            //- /main.rs
            mod foo;
            struct S;
            //- /foo.rs
            <|>
        "#;
        check_found_path(code, "crate::S");
    }

    #[test]
    fn different_crate() {
        let code = r#"
            //- /main.rs crate:main deps:std
            <|>
            //- /std.rs crate:std
            pub struct S;
        "#;
        check_found_path(code, "std::S");
    }

    #[test]
    fn different_crate_renamed() {
        let code = r#"
            //- /main.rs crate:main deps:std
            extern crate std as std_renamed;
            <|>
            //- /std.rs crate:std
            pub struct S;
        "#;
        check_found_path(code, "std_renamed::S");
    }

    #[test]
    fn same_crate_reexport() {
        let code = r#"
            //- /main.rs
            mod bar {
                mod foo { pub(super) struct S; }
                pub(crate) use foo::*;
            }
            <|>
        "#;
        check_found_path(code, "bar::S");
    }

    #[test]
    fn same_crate_reexport_rename() {
        let code = r#"
            //- /main.rs
            mod bar {
                mod foo { pub(super) struct S; }
                pub(crate) use foo::S as U;
            }
            <|>
        "#;
        check_found_path(code, "bar::U");
    }

    #[test]
    fn different_crate_reexport() {
        let code = r#"
            //- /main.rs crate:main deps:std
            <|>
            //- /std.rs crate:std deps:core
            pub use core::S;
            //- /core.rs crate:core
            pub struct S;
        "#;
        check_found_path(code, "std::S");
    }

    #[test]
    fn prelude() {
        let code = r#"
            //- /main.rs crate:main deps:std
            <|>
            //- /std.rs crate:std
            pub mod prelude { pub struct S; }
            #[prelude_import]
            pub use prelude::*;
        "#;
        check_found_path(code, "S");
    }

    #[test]
    fn enum_variant_from_prelude() {
        let code = r#"
            //- /main.rs crate:main deps:std
            <|>
            //- /std.rs crate:std
            pub mod prelude {
                pub enum Option<T> { Some(T), None }
                pub use Option::*;
            }
            #[prelude_import]
            pub use prelude::*;
        "#;
        check_found_path(code, "None");
        check_found_path(code, "Some");
    }

    #[test]
    fn shortest_path() {
        let code = r#"
            //- /main.rs
            pub mod foo;
            pub mod baz;
            struct S;
            <|>
            //- /foo.rs
            pub mod bar { pub struct S; }
            //- /baz.rs
            pub use crate::foo::bar::S;
        "#;
        check_found_path(code, "baz::S");
    }

    #[test]
    fn discount_private_imports() {
        let code = r#"
            //- /main.rs
            mod foo;
            pub mod bar { pub struct S; }
            use bar::S;
            //- /foo.rs
            <|>
        "#;
        // crate::S would be shorter, but using private imports seems wrong
        check_found_path(code, "crate::bar::S");
    }

    #[test]
    fn import_cycle() {
        let code = r#"
            //- /main.rs
            pub mod foo;
            pub mod bar;
            pub mod baz;
            //- /bar.rs
            <|>
            //- /foo.rs
            pub use super::baz;
            pub struct S;
            //- /baz.rs
            pub use super::foo;
        "#;
        check_found_path(code, "crate::foo::S");
    }

    #[test]
    fn prefer_std_paths_over_alloc() {
        covers!(prefer_std_paths);
        let code = r#"
        //- /main.rs crate:main deps:alloc,std
        <|>

        //- /std.rs crate:std deps:alloc
        pub mod sync {
            pub use alloc::sync::Arc;
        }

        //- /zzz.rs crate:alloc
        pub mod sync {
            pub struct Arc;
        }
        "#;
        check_found_path(code, "std::sync::Arc");
    }

    #[test]
    fn prefer_shorter_paths_if_not_alloc() {
        let code = r#"
        //- /main.rs crate:main deps:megaalloc,std
        <|>

        //- /std.rs crate:std deps:megaalloc
        pub mod sync {
            pub use megaalloc::sync::Arc;
        }

        //- /zzz.rs crate:megaalloc
        pub struct Arc;
        "#;
        check_found_path(code, "megaalloc::Arc");
    }
}
