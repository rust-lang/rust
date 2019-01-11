use ra_db::{FileId, Cancelable, SyntaxDatabase};
use ra_syntax::{
    AstNode, ast,
    algo::find_node_at_offset,
};

use crate::{FilePosition, NavigationTarget, db::RootDatabase, RangeInfo};

pub(crate) fn goto_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<RangeInfo<Vec<NavigationTarget>>>> {
    let file = db.source_file(position.file_id);
    let syntax = file.syntax();
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
        let navs = reference_definition(db, position.file_id, name_ref)?;
        return Ok(Some(RangeInfo::new(name_ref.syntax().range(), navs)));
    }
    if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset) {
        let navs = ctry!(name_definition(db, position.file_id, name)?);
        return Ok(Some(RangeInfo::new(name.syntax().range(), navs)));
    }
    Ok(None)
}

pub(crate) fn reference_definition(
    db: &RootDatabase,
    file_id: FileId,
    name_ref: &ast::NameRef,
) -> Cancelable<Vec<NavigationTarget>> {
    if let Some(fn_descr) =
        hir::source_binder::function_from_child_node(db, file_id, name_ref.syntax())?
    {
        let scope = fn_descr.scopes(db)?;
        // First try to resolve the symbol locally
        if let Some(entry) = scope.resolve_local_name(name_ref) {
            let nav = NavigationTarget::from_scope_entry(file_id, &entry);
            return Ok(vec![nav]);
        };
    }
    // Then try module name resolution
    if let Some(module) =
        hir::source_binder::module_from_child_node(db, file_id, name_ref.syntax())?
    {
        if let Some(path) = name_ref
            .syntax()
            .ancestors()
            .find_map(ast::Path::cast)
            .and_then(hir::Path::from_ast)
        {
            let resolved = module.resolve_path(db, &path)?;
            if let Some(def_id) = resolved.take_types().or(resolved.take_values()) {
                if let Some(target) = NavigationTarget::from_def(db, def_id.resolve(db)?)? {
                    return Ok(vec![target]);
                }
            }
        }
    }
    // If that fails try the index based approach.
    let navs = db
        .index_resolve(name_ref)?
        .into_iter()
        .map(NavigationTarget::from_symbol)
        .collect();
    Ok(navs)
}

fn name_definition(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Cancelable<Option<Vec<NavigationTarget>>> {
    if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
        if module.has_semi() {
            if let Some(child_module) =
                hir::source_binder::module_from_declaration(db, file_id, module)?
            {
                let nav = NavigationTarget::from_module(db, child_module)?;
                return Ok(Some(vec![nav]));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use test_utils::assert_eq_dbg;
    use crate::mock_analysis::analysis_and_position;

    #[test]
    fn goto_definition_works_in_items() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            struct Foo;
            enum E { X(Foo<|>) }
            ",
        );

        let symbols = analysis.goto_definition(pos).unwrap().unwrap();
        assert_eq_dbg(
            r#"[NavigationTarget { file_id: FileId(1), name: "Foo",
                                   kind: STRUCT_DEF, range: [0; 11),
                                   ptr: Some(LocalSyntaxPtr { range: [0; 11), kind: STRUCT_DEF }) }]"#,
            &symbols,
        );
    }

    #[test]
    fn goto_definition_resolves_correct_name() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            use a::Foo;
            mod a;
            mod b;
            enum E { X(Foo<|>) }
            //- /a.rs
            struct Foo;
            //- /b.rs
            struct Foo;
            ",
        );

        let symbols = analysis.goto_definition(pos).unwrap().unwrap();
        assert_eq_dbg(
            r#"[NavigationTarget { file_id: FileId(2), name: "Foo",
                                   kind: STRUCT_DEF, range: [0; 11),
                                   ptr: Some(LocalSyntaxPtr { range: [0; 11), kind: STRUCT_DEF }) }]"#,
            &symbols,
        );
    }

    #[test]
    fn goto_definition_works_for_module_declaration() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            mod <|>foo;
            //- /foo.rs
            // empty
        ",
        );

        let symbols = analysis.goto_definition(pos).unwrap().unwrap();
        assert_eq_dbg(
            r#"[NavigationTarget { file_id: FileId(2), name: "foo", kind: MODULE, range: [0; 0), ptr: None }]"#,
            &symbols,
        );

        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            mod <|>foo;
            //- /foo/mod.rs
            // empty
        ",
        );

        let symbols = analysis.goto_definition(pos).unwrap().unwrap();
        assert_eq_dbg(
            r#"[NavigationTarget { file_id: FileId(2), name: "foo", kind: MODULE, range: [0; 0), ptr: None }]"#,
            &symbols,
        );
    }
}
