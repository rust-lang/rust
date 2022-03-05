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
        .find(|it| matches!(it.kind(), IDENT | T![self] | T![super] | T![crate] | T![Self]))?;
    let range = original_token.text_range();
    let info: Vec<NavigationTarget> = sema
        .descend_into_macros(original_token)
        .iter()
        .filter_map(|token| {
            let parent = token.parent()?;
            let def = match_ast! {
                match parent {
                    ast::NameRef(name_ref) => match NameRefClass::classify(&sema, &name_ref)? {
                        NameRefClass::Definition(it) => Some(it),
                        _ => None
                    },
                    ast::Name(name) => match NameClass::classify(&sema, &name)? {
                        NameClass::Definition(it) => Some(it),
                        _ => None
                    },
                    _ => None
                }
            };
            match def? {
                Definition::Module(module) => {
                    Some(NavigationTarget::from_module_to_decl(db, module))
                }
                _ => None,
            }
        })
        .collect();

    Some(RangeInfo::new(range, info))
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;
    use itertools::Itertools;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis
            .goto_declaration(position)
            .unwrap()
            .expect("no declaration or definition found")
            .info;
        if navs.is_empty() {
            panic!("unresolved reference")
        }

        let cmp = |&FileRange { file_id, range }: &_| (file_id, range.start());
        let navs = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|(FileRange { file_id, range }, _)| FileRange { file_id, range })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        assert_eq!(expected, navs);
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
