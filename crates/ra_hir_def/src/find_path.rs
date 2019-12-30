//! An algorithm to find a path to refer to a certain item.

use crate::{ModuleDefId, path::ModPath, ModuleId};

pub fn find_path(item: ModuleDefId, from: ModuleId) -> ModPath {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ra_db::{fixture::WithFixture, SourceDatabase};
    use crate::{db::DefDatabase, test_db::TestDB};
    use ra_syntax::ast::AstNode;
    use hir_expand::hygiene::Hygiene;

    /// `code` needs to contain a cursor marker; checks that `find_path` for the
    /// item the `path` refers to returns that same path when called from the
    /// module the cursor is in.
    fn check_found_path(code: &str, path: &str) {
        let (db, pos) = TestDB::with_position(code);
        let module = db.module_for_file(pos.file_id);
        let parsed_path_file = ra_syntax::SourceFile::parse(&format!("use {};", path));
        let ast_path = parsed_path_file.syntax_node().descendants().find_map(ra_syntax::ast::Path::cast).unwrap();
        let mod_path = ModPath::from_src(ast_path, &Hygiene::new_unhygienic()).unwrap();

        let crate_def_map = db.crate_def_map(module.krate);
        let resolved = crate_def_map.resolve_path(&db, module.local_id, &mod_path, crate::item_scope::BuiltinShadowMode::Module).0.take_types().unwrap();

        let found_path = find_path(resolved, module);

        assert_eq!(mod_path, found_path);
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
}
