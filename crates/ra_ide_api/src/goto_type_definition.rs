//! FIXME: write short doc here

use ra_db::SourceDatabase;
use ra_syntax::{ast, AstNode};

use crate::{db::RootDatabase, FilePosition, NavigationTarget, RangeInfo};

pub(crate) fn goto_type_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let parse = db.parse(position.file_id);

    let node = parse.tree().syntax().token_at_offset(position.offset).find_map(|token| {
        token
            .parent()
            .ancestors()
            .find(|n| ast::Expr::cast(n.clone()).is_some() || ast::Pat::cast(n.clone()).is_some())
    })?;

    let analyzer = hir::SourceAnalyzer::new(db, position.file_id, &node, None);

    let ty: hir::Ty = if let Some(ty) =
        ast::Expr::cast(node.clone()).and_then(|e| analyzer.type_of(db, &e))
    {
        ty
    } else if let Some(ty) = ast::Pat::cast(node.clone()).and_then(|p| analyzer.type_of_pat(db, &p))
    {
        ty
    } else {
        return None;
    };

    let adt_def = analyzer.autoderef(db, ty).find_map(|ty| ty.as_adt().map(|adt| adt.0))?;

    let nav = NavigationTarget::from_adt_def(db, adt_def);
    Some(RangeInfo::new(node.text_range(), vec![nav]))
}

#[cfg(test)]
mod tests {
    use crate::mock_analysis::analysis_and_position;

    fn check_goto(fixture: &str, expected: &str) {
        let (analysis, pos) = analysis_and_position(fixture);

        let mut navs = analysis.goto_type_definition(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        nav.assert_match(expected);
    }

    #[test]
    fn goto_type_definition_works_simple() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            fn foo() {
                let f: Foo;
                f<|>
            }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
        );
    }

    #[test]
    fn goto_type_definition_works_simple_ref() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            fn foo() {
                let f: &Foo;
                f<|>
            }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
        );
    }
}
