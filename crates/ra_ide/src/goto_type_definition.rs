use ra_ide_db::RootDatabase;
use ra_syntax::{ast, match_ast, AstNode, SyntaxKind::*, SyntaxToken, TokenAtOffset};

use crate::{display::ToNav, FilePosition, NavigationTarget, RangeInfo};

// Feature: Go to Type Definition
//
// Navigates to the type of an identifier.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Go to Type Definition*
// |===
pub(crate) fn goto_type_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = hir::Semantics::new(db);

    let file: ast::SourceFile = sema.parse(position.file_id);
    let token: SyntaxToken = pick_best(file.syntax().token_at_offset(position.offset))?;
    let token: SyntaxToken = sema.descend_into_macros(token);

    let (ty, node) = sema.ancestors_with_macros(token.parent()).find_map(|node| {
        let ty = match_ast! {
            match node {
                ast::Expr(expr) => sema.type_of_expr(&expr)?,
                ast::Pat(pat) => sema.type_of_pat(&pat)?,
                _ => return None,
            }
        };

        Some((ty, node))
    })?;

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

    fn check_goto(ra_fixture: &str, expected: &str) {
        let (analysis, pos) = analysis_and_position(ra_fixture);

        let mut navs = analysis.goto_type_definition(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        nav.assert_match(expected);
    }

    #[test]
    fn goto_type_definition_works_simple() {
        check_goto(
            r"
            //- /lib.rs
            struct Foo;
            fn foo() {
                let f: Foo;
                f<|>
            }
            ",
            "Foo STRUCT_DEF FileId(1) 0..11 7..10",
        );
    }

    #[test]
    fn goto_type_definition_works_simple_ref() {
        check_goto(
            r"
            //- /lib.rs
            struct Foo;
            fn foo() {
                let f: &Foo;
                f<|>
            }
            ",
            "Foo STRUCT_DEF FileId(1) 0..11 7..10",
        );
    }

    #[test]
    fn goto_type_definition_works_through_macro() {
        check_goto(
            r"
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
            "Foo STRUCT_DEF FileId(1) 52..65 59..62",
        );
    }

    #[test]
    fn goto_type_definition_for_param() {
        check_goto(
            r"
            //- /lib.rs
            struct Foo;
            fn foo(<|>f: Foo) {}
            ",
            "Foo STRUCT_DEF FileId(1) 0..11 7..10",
        );
    }

    #[test]
    fn goto_type_definition_for_tuple_field() {
        check_goto(
            r"
            //- /lib.rs
            struct Foo;
            struct Bar(Foo);
            fn foo() {
                let bar = Bar(Foo);
                bar.<|>0;
            }
            ",
            "Foo STRUCT_DEF FileId(1) 0..11 7..10",
        );
    }
}
