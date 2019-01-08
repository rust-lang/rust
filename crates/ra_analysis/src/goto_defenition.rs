use ra_db::{FileId, Cancelable, SyntaxDatabase};
use ra_syntax::{TextRange, AstNode, ast, SyntaxKind::{NAME, MODULE}};

use ra_editor::find_node_at_offset;

use crate::{FilePosition, NavigationTarget, db::RootDatabase};

pub(crate) fn goto_defenition(
    db: &RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<Vec<NavigationTarget>>> {
    let file = db.source_file(position.file_id);
    let syntax = file.syntax();
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
        return Ok(Some(reference_defenition(db, position.file_id, name_ref)?));
    }
    if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset) {
        return name_defenition(db, position.file_id, name);
    }
    Ok(None)
}

pub(crate) fn reference_defenition(
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
            let nav = NavigationTarget {
                file_id,
                name: entry.name().to_string().into(),
                range: entry.ptr().range(),
                kind: NAME,
                ptr: None,
            };
            return Ok(vec![nav]);
        };
    }
    // If that fails try the index based approach.
    let navs = db
        .index_resolve(name_ref)?
        .into_iter()
        .map(NavigationTarget::from_symbol)
        .collect();
    Ok(navs)
}

fn name_defenition(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Cancelable<Option<Vec<NavigationTarget>>> {
    if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
        if module.has_semi() {
            if let Some(child_module) =
                hir::source_binder::module_from_declaration(db, file_id, module)?
            {
                let (file_id, _) = child_module.defenition_source(db)?;
                let name = match child_module.name(db)? {
                    Some(name) => name.to_string().into(),
                    None => "".into(),
                };
                let nav = NavigationTarget {
                    file_id,
                    name,
                    range: TextRange::offset_len(0.into(), 0.into()),
                    kind: MODULE,
                    ptr: None,
                };
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
    fn goto_defenition_works_in_items() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            struct Foo;
            enum E { X(Foo<|>) }
            ",
        );

        let symbols = analysis.goto_defenition(pos).unwrap().unwrap();
        assert_eq_dbg(
            r#"[NavigationTarget { file_id: FileId(1), name: "Foo",
                                   kind: STRUCT_DEF, range: [0; 11),
                                   ptr: Some(LocalSyntaxPtr { range: [0; 11), kind: STRUCT_DEF }) }]"#,
            &symbols,
        );
    }

    #[test]
    fn goto_defenition_works_for_module_declaration() {
        let (analysis, pos) = analysis_and_position(
            "
            //- /lib.rs
            mod <|>foo;
            //- /foo.rs
            // empty
        ",
        );

        let symbols = analysis.goto_defenition(pos).unwrap().unwrap();
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

        let symbols = analysis.goto_defenition(pos).unwrap().unwrap();
        assert_eq_dbg(
            r#"[NavigationTarget { file_id: FileId(2), name: "foo", kind: MODULE, range: [0; 0), ptr: None }]"#,
            &symbols,
        );
    }
}
