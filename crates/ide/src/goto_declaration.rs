use hir::Semantics;
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};
use syntax::{ast, match_ast, AstNode, SyntaxKind::*, T};

use crate::{FilePosition, NavigationTarget, RangeInfo};

// Feature: Go to Declaration
//
// Navigates to the declaration of an identifier.
pub(crate) fn goto_declaration(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let original_token = file
        .token_at_offset(position.offset)
        .find(|it| matches!(it.kind(), IDENT | T![self] | T![super] | T![crate]))?;
    let token = sema.descend_into_macros(original_token.clone());
    let parent = token.parent()?;
    let def = match_ast! {
        match parent {
            ast::NameRef(name_ref) => {
                let name_kind = NameRefClass::classify(&sema, &name_ref)?;
                name_kind.referenced(sema.db)
            },
            ast::Name(name) => {
                NameClass::classify(&sema, &name)?.referenced_or_defined(sema.db)
            },
            _ => return None,
        }
    };
    match def {
        Definition::ModuleDef(hir::ModuleDef::Module(module)) => Some(RangeInfo::new(
            original_token.text_range(),
            vec![NavigationTarget::from_module_to_decl(db, module)],
        )),
        _ => return None,
    }
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::nav_target_annotation(ra_fixture);
        let mut navs = analysis
            .goto_declaration(position)
            .unwrap()
            .expect("no declaration or definition found")
            .info;
        if navs.len() == 0 {
            panic!("unresolved reference")
        }
        assert_eq!(navs.len(), 1);

        let nav = navs.pop().unwrap();
        assert_eq!(expected, FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() });
    }

    #[test]
    fn goto_decl_module_outline() {
        check(
            r#"
//- /main.rs
mod foo;
 // ^^^
//- /foo.rs
use self$0;
"#,
        )
    }

    #[test]
    fn goto_decl_module_inline() {
        check(
            r#"
mod foo {
 // ^^^
    use self$0;
}
"#,
        )
    }
}
