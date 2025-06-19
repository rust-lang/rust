use hir::GenericParam;
use ide_db::{RootDatabase, defs::Definition, helpers::pick_best_token};
use syntax::{AstNode, SyntaxKind::*, SyntaxToken, T, ast, match_ast};

use crate::{FilePosition, NavigationTarget, RangeInfo, TryToNav};

// Feature: Go to Type Definition
//
// Navigates to the type of an identifier.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **Go to Type Definition** |
//
// ![Go to Type Definition](https://user-images.githubusercontent.com/48062697/113020657-b560f500-917a-11eb-9007-0f809733a338.gif)
pub(crate) fn goto_type_definition(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = hir::Semantics::new(db);

    let file: ast::SourceFile = sema.parse_guess_edition(file_id);
    let token: SyntaxToken =
        pick_best_token(file.syntax().token_at_offset(offset), |kind| match kind {
            IDENT | INT_NUMBER | T![self] => 3,
            kind if kind.is_trivia() => 0,
            T![;] => 1,
            _ => 2,
        })?;

    let mut res = Vec::new();
    let mut push = |def: Definition| {
        if let Some(navs) = def.try_to_nav(db) {
            for nav in navs {
                if !res.contains(&nav) {
                    res.push(nav);
                }
            }
        }
    };
    let mut process_ty = |ty: hir::Type| {
        // collect from each `ty` into the `res` result vec
        let ty = ty.strip_references();
        ty.walk(db, |t| {
            if let Some(adt) = t.as_adt() {
                push(adt.into());
            } else if let Some(trait_) = t.as_dyn_trait() {
                push(trait_.into());
            } else if let Some(traits) = t.as_impl_traits(db) {
                traits.for_each(|it| push(it.into()));
            } else if let Some(trait_) = t.as_associated_type_parent_trait(db) {
                push(trait_.into());
            }
        });
    };
    if let Some((range, _, _, resolution)) =
        sema.check_for_format_args_template(token.clone(), offset)
    {
        if let Some(ty) = resolution.and_then(|res| match Definition::from(res) {
            Definition::Const(it) => Some(it.ty(db)),
            Definition::Static(it) => Some(it.ty(db)),
            Definition::GenericParam(GenericParam::ConstParam(it)) => Some(it.ty(db)),
            Definition::Local(it) => Some(it.ty(db)),
            Definition::Adt(hir::Adt::Struct(it)) => Some(it.ty(db)),
            _ => None,
        }) {
            process_ty(ty);
        }
        return Some(RangeInfo::new(range, res));
    }

    let range = token.text_range();
    sema.descend_into_macros_no_opaque(token,false)
        .into_iter()
        .filter_map(|token| {
            sema
                .token_ancestors_with_macros(token.value)
                // When `token` is within a macro call, we can't determine its type. Don't continue
                // this traversal because otherwise we'll end up returning the type of *that* macro
                // call, which is not what we want in general.
                //
                // Macro calls always wrap `TokenTree`s, so it's sufficient and efficient to test
                // if the current node is a `TokenTree`.
                .take_while(|node| !ast::TokenTree::can_cast(node.kind()))
                .find_map(|node| {
                    let ty = match_ast! {
                        match node {
                            ast::Expr(it) => sema.type_of_expr(&it)?.original,
                            ast::Pat(it) => sema.type_of_pat(&it)?.original,
                            ast::SelfParam(it) => sema.type_of_self(&it)?,
                            ast::Type(it) => sema.resolve_type(&it)?,
                            ast::RecordField(it) => sema.to_def(&it)?.ty(db),
                            // can't match on RecordExprField directly as `ast::Expr` will match an iteration too early otherwise
                            ast::NameRef(it) => {
                                if let Some(record_field) = ast::RecordExprField::for_name_ref(&it) {
                                    let (_, _, ty) = sema.resolve_record_field(&record_field)?;
                                    ty
                                } else {
                                    let record_field = ast::RecordPatField::for_field_name_ref(&it)?;
                                    sema.resolve_record_pat_field(&record_field)?.1
                                }
                            },
                            _ => return None,
                        }
                    };

                    Some(ty)
                })
        })
        .for_each(process_ty);
    Some(RangeInfo::new(range, res))
}

#[cfg(test)]
mod tests {
    use ide_db::FileRange;
    use itertools::Itertools;

    use crate::fixture;

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis.goto_type_definition(position).unwrap().unwrap().info;
        assert!(!navs.is_empty(), "navigation is empty");

        let cmp = |&FileRange { file_id, range }: &_| (file_id, range.start());
        let navs = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|(file_range, _)| file_range)
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        assert_eq!(expected, navs);
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
    fn dont_collect_type_from_token_in_macro_call() {
        check(
            r#"
struct DontCollectMe;
struct S;
     //^

macro_rules! inner {
    ($t:tt) => { DontCollectMe }
}
macro_rules! m {
    ($t:ident) => {
        match $t {
            _ => inner!($t);
        }
    }
}

fn test() {
    m!($0S);
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

    #[test]
    fn goto_def_considers_generics() {
        check(
            r#"
struct Foo;
     //^^^
struct Bar<T, U>(T, U);
     //^^^
struct Baz<T>(T);
     //^^^

fn foo(x$0: Bar<Baz<Foo>, Baz<usize>) {}
"#,
        );
    }

    #[test]
    fn implicit_format_args() {
        check(
            r#"
//- minicore: fmt
struct Bar;
    // ^^^
    fn test() {
    let a = Bar;
    format_args!("hello {a$0}");
}
"#,
        );
        check(
            r#"
//- minicore: fmt
struct Bar;
    // ^^^
    fn test() {
    format_args!("hello {Bar$0}");
}
"#,
        );
        check(
            r#"
//- minicore: fmt
struct Bar;
    // ^^^
const BAR: Bar = Bar;
fn test() {
    format_args!("hello {BAR$0}");
}
"#,
        );
    }
}
