//! FIXME: write short doc here

use hir::db::AstDatabase;
use ra_syntax::{ast, AstNode, SyntaxKind::*, SyntaxToken, TokenAtOffset};

use crate::{
    db::RootDatabase, display::ToNav, expand::descend_into_macros, FilePosition, NavigationTarget,
    RangeInfo,
};

pub(crate) fn goto_type_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let file = db.parse_or_expand(position.file_id.into())?;
    let token = pick_best(file.token_at_offset(position.offset))?;
    let token = descend_into_macros(db, position.file_id, token);

    let node = token.value.ancestors().find_map(|token| {
        token
            .ancestors()
            .find(|n| ast::Expr::cast(n.clone()).is_some() || ast::Pat::cast(n.clone()).is_some())
    })?;

    let analyzer = hir::SourceAnalyzer::new(db, token.with_value(&node), None);

    let ty: hir::Type = if let Some(ty) =
        ast::Expr::cast(node.clone()).and_then(|e| analyzer.type_of(db, &e))
    {
        ty
    } else if let Some(ty) = ast::Pat::cast(node.clone()).and_then(|p| analyzer.type_of_pat(db, &p))
    {
        ty
    } else {
        return None;
    };

    let adt_def = ty.autoderef(db).find_map(|ty| ty.as_adt())?;

    let nav = adt_def.to_nav(db);
    Some(RangeInfo::new(node.text_range(), vec![nav]))
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
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

    #[test]
    fn goto_type_definition_works_through_macro() {
        check_goto(
            "
            //- /lib.rs
            macro_rules! id {
                ($($tt:tt)*) => { $($tt)* }
            }
            struct Foo {}
            id! {
                fn bar() {
                    let f<|> = Foo {};
                }
            }
            ",
            "Foo STRUCT_DEF FileId(1) [52; 65) [59; 62)",
        );
    }

    #[test]
    fn goto_type_definition_for_param() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            fn foo(<|>f: Foo) {}
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
        );
    }

    #[test]
    fn goto_type_definition_for_tuple_field() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            struct Bar(Foo);
            fn foo() {
                let bar = Bar(Foo);
                bar.<|>0;
            }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
        );
    }
}
