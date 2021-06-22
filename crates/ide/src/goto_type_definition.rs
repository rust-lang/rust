use ide_db::base_db::Upcast;
use ide_db::helpers::pick_best_token;
use ide_db::RootDatabase;
use syntax::{ast, match_ast, AstNode, SyntaxKind::*, SyntaxToken, T};

use crate::{display::TryToNav, FilePosition, NavigationTarget, RangeInfo};

// Feature: Go to Type Definition
//
// Navigates to the type of an identifier.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Go to Type Definition*
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113020657-b560f500-917a-11eb-9007-0f809733a338.gif[]
pub(crate) fn goto_type_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = hir::Semantics::new(db);

    let file: ast::SourceFile = sema.parse(position.file_id);
    let token: SyntaxToken =
        pick_best_token(file.syntax().token_at_offset(position.offset), |kind| match kind {
            IDENT | INT_NUMBER | T![self] => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        })?;
    let token: SyntaxToken = sema.descend_into_macros(token);

    let (ty, node) = sema.token_ancestors_with_macros(token).find_map(|node| {
        let ty = match_ast! {
            match node {
                ast::Expr(it) => sema.type_of_expr(&it)?,
                ast::Pat(it) => sema.type_of_pat(&it)?,
                ast::SelfParam(it) => sema.type_of_self(&it)?,
                ast::Type(it) => sema.resolve_type(&it)?,
                ast::RecordField(it) => sema.to_def(&it).map(|d| d.ty(db.upcast()))?,
                ast::RecordField(it) => sema.to_def(&it).map(|d| d.ty(db.upcast()))?,
                // can't match on RecordExprField directly as `ast::Expr` will match an iteration too early otherwise
                ast::NameRef(it) => {
                    if let Some(record_field) = ast::RecordExprField::for_name_ref(&it) {
                        let (_, _, ty) = sema.resolve_record_field(&record_field)?;
                        ty
                    } else {
                        let record_field = ast::RecordPatField::for_field_name_ref(&it)?;
                        sema.resolve_record_pat_field(&record_field)?.ty(db)
                    }
                },
                _ => return None,
            }
        };

        Some((ty, node))
    })?;
    let adt_def = ty.autoderef(db).filter_map(|ty| ty.as_adt()).last()?;

    let nav = adt_def.try_to_nav(db)?;
    Some(RangeInfo::new(node.text_range(), vec![nav]))
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, mut annotations) = fixture::annotations(ra_fixture);
        let (expected, data) = annotations.pop().unwrap();
        assert!(data.is_empty());

        let mut navs = analysis.goto_type_definition(position).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        assert_eq!(expected, FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() });
    }

    #[test]
    fn goto_type_definition_works_simple() {
        check(
            r#"
struct Foo;
     //^^^
fn foo() {
    let f: Foo; f$0
}
"#,
        );
    }

    #[test]
    fn goto_type_definition_record_expr_field() {
        check(
            r#"
struct Bar;
    // ^^^
struct Foo { foo: Bar }
fn foo() {
    Foo { foo$0 }
}
"#,
        );
        check(
            r#"
struct Bar;
    // ^^^
struct Foo { foo: Bar }
fn foo() {
    Foo { foo$0: Bar }
}
"#,
        );
    }

    #[test]
    fn goto_type_definition_record_pat_field() {
        check(
            r#"
struct Bar;
    // ^^^
struct Foo { foo: Bar }
fn foo() {
    let Foo { foo$0 };
}
"#,
        );
        check(
            r#"
struct Bar;
    // ^^^
struct Foo { foo: Bar }
fn foo() {
    let Foo { foo$0: bar };
}
"#,
        );
    }

    #[test]
    fn goto_type_definition_works_simple_ref() {
        check(
            r#"
struct Foo;
     //^^^
fn foo() {
    let f: &Foo; f$0
}
"#,
        );
    }

    #[test]
    fn goto_type_definition_works_through_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
struct Foo {}
     //^^^
id! {
    fn bar() { let f$0 = Foo {}; }
}
"#,
        );
    }

    #[test]
    fn goto_type_definition_for_param() {
        check(
            r#"
struct Foo;
     //^^^
fn foo($0f: Foo) {}
"#,
        );
    }

    #[test]
    fn goto_type_definition_for_tuple_field() {
        check(
            r#"
struct Foo;
     //^^^
struct Bar(Foo);
fn foo() {
    let bar = Bar(Foo);
    bar.$00;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_self_param() {
        check(
            r#"
struct Foo;
     //^^^
impl Foo {
    fn f(&self$0) {}
}
"#,
        )
    }

    #[test]
    fn goto_def_for_type_fallback() {
        check(
            r#"
struct Foo;
     //^^^
impl Foo$0 {}
"#,
        )
    }

    #[test]
    fn goto_def_for_struct_field() {
        check(
            r#"
struct Bar;
     //^^^

struct Foo {
    bar$0: Bar,
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_struct_field() {
        check(
            r#"
struct Bar;
     //^^^

enum Foo {
    Bar {
        bar$0: Bar
    },
}
"#,
        );
    }
}
