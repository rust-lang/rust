//! An algorithm to find a path to refer to a certain item.

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    path::{ModPath, PathKind},
    ModuleId, ModuleDefId,
};
use hir_expand::name::Name;

// TODO don't import from super imports? or at least deprioritize
// TODO use super?
// TODO performance / memoize

pub fn find_path(db: &impl DefDatabase, item: ItemInNs, from: ModuleId) -> Option<ModPath> {
    // Base cases:

    // - if the item is already in scope, return the name under which it is
    let def_map = db.crate_def_map(from.krate);
    let from_scope: &crate::item_scope::ItemScope = &def_map.modules[from.local_id].scope;
    if let Some((name, _)) = from_scope.reverse_get(item) {
        return Some(ModPath::from_simple_segments(PathKind::Plain, vec![name.clone()]));
    }

    // - if the item is the crate root, return `crate`
    if item == ItemInNs::Types(ModuleDefId::ModuleId(ModuleId { krate: from.krate, local_id: def_map.root })) {
        return Some(ModPath::from_simple_segments(PathKind::Crate, Vec::new()));
    }

    // - if the item is the crate root of a dependency crate, return the name from the extern prelude
    for (name, def_id) in &def_map.extern_prelude {
        if item == ItemInNs::Types(*def_id) {
            return Some(ModPath::from_simple_segments(PathKind::Plain, vec![name.clone()]));
        }
    }

    // - if the item is in the prelude, return the name from there
    if let Some(prelude_module) = def_map.prelude {
        let prelude_def_map = db.crate_def_map(prelude_module.krate);
        let prelude_scope: &crate::item_scope::ItemScope = &prelude_def_map.modules[prelude_module.local_id].scope;
        if let Some((name, vis)) = prelude_scope.reverse_get(item) {
            if vis.is_visible_from(db, from) {
                return Some(ModPath::from_simple_segments(PathKind::Plain, vec![name.clone()]));
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
    let mut candidate_paths = Vec::new();
    for (module_id, name) in importable_locations {
        // TODO prevent infinite loops
        let mut path = match find_path(db, ItemInNs::Types(ModuleDefId::ModuleId(module_id)), from) {
            None => continue,
            Some(path) => path,
        };
        path.segments.push(name);
        candidate_paths.push(path);
    }
    candidate_paths.into_iter().min_by_key(|path| path.segments.len())
}

fn find_importable_locations(db: &impl DefDatabase, item: ItemInNs, from: ModuleId) -> Vec<(ModuleId, Name)> {
    let crate_graph = db.crate_graph();
    let mut result = Vec::new();
    for krate in Some(from.krate).into_iter().chain(crate_graph.dependencies(from.krate).map(|dep| dep.crate_id)) {
        let def_map = db.crate_def_map(krate);
        for (local_id, data) in def_map.modules.iter() {
            if let Some((name, vis)) = data.scope.reverse_get(item) {
                if vis.is_visible_from(db, from) {
                    result.push((ModuleId { krate, local_id }, name.clone()));
                }
            }
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
}
