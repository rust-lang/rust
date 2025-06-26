use hir::{AsAssocItem, Impl, Semantics};
use ide_db::{
    RootDatabase,
    defs::{Definition, NameClass, NameRefClass},
    helpers::pick_best_token,
};
use syntax::{AstNode, SyntaxKind::*, T, ast};

use crate::{FilePosition, NavigationTarget, RangeInfo, TryToNav};

// Feature: Go to Implementation
//
// Navigates to the impl items of types.
//
// | Editor  | Shortcut |
// |---------|----------|
// | VS Code | <kbd>Ctrl+F12</kbd>
//
// ![Go to Implementation](https://user-images.githubusercontent.com/48062697/113065566-02f85480-91b1-11eb-9288-aaad8abd8841.gif)
pub(crate) fn goto_implementation(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(file_id);
    let syntax = source_file.syntax().clone();

    let original_token = pick_best_token(syntax.token_at_offset(offset), |kind| match kind {
        IDENT | T![self] | INT_NUMBER => 1,
        _ => 0,
    })?;
    let range = original_token.text_range();
    let navs = sema
        .descend_into_macros_exact(original_token)
        .iter()
        .filter_map(|token| {
            token
                .parent()
                .and_then(ast::NameLike::cast)
                .and_then(|node| match &node {
                    ast::NameLike::Name(name) => {
                        NameClass::classify(&sema, name).and_then(|class| match class {
                            NameClass::Definition(it) | NameClass::ConstReference(it) => Some(it),
                            NameClass::PatFieldShorthand { .. } => None,
                        })
                    }
                    ast::NameLike::NameRef(name_ref) => NameRefClass::classify(&sema, name_ref)
                        .and_then(|class| match class {
                            NameRefClass::Definition(def, _) => Some(def),
                            NameRefClass::FieldShorthand { .. }
                            | NameRefClass::ExternCrateShorthand { .. } => None,
                        }),
                    ast::NameLike::Lifetime(_) => None,
                })
                .and_then(|def| {
                    let navs = match def {
                        Definition::Trait(trait_) => impls_for_trait(&sema, trait_),
                        Definition::Adt(adt) => impls_for_ty(&sema, adt.ty(sema.db)),
                        Definition::TypeAlias(alias) => impls_for_ty(&sema, alias.ty(sema.db)),
                        Definition::BuiltinType(builtin) => {
                            impls_for_ty(&sema, builtin.ty(sema.db))
                        }
                        Definition::Function(f) => {
                            let assoc = f.as_assoc_item(sema.db)?;
                            let name = assoc.name(sema.db)?;
                            let trait_ = assoc.container_or_implemented_trait(sema.db)?;
                            impls_for_trait_item(&sema, trait_, name)
                        }
                        Definition::Const(c) => {
                            let assoc = c.as_assoc_item(sema.db)?;
                            let name = assoc.name(sema.db)?;
                            let trait_ = assoc.container_or_implemented_trait(sema.db)?;
                            impls_for_trait_item(&sema, trait_, name)
                        }
                        _ => return None,
                    };
                    Some(navs)
                })
        })
        .flatten()
        .collect();

    Some(RangeInfo { range, info: navs })
}

fn impls_for_ty(sema: &Semantics<'_, RootDatabase>, ty: hir::Type<'_>) -> Vec<NavigationTarget> {
    Impl::all_for_type(sema.db, ty)
        .into_iter()
        .filter_map(|imp| imp.try_to_nav(sema.db))
        .flatten()
        .collect()
}

fn impls_for_trait(
    sema: &Semantics<'_, RootDatabase>,
    trait_: hir::Trait,
) -> Vec<NavigationTarget> {
    Impl::all_for_trait(sema.db, trait_)
        .into_iter()
        .filter_map(|imp| imp.try_to_nav(sema.db))
        .flatten()
        .collect()
}

fn impls_for_trait_item(
    sema: &Semantics<'_, RootDatabase>,
    trait_: hir::Trait,
    fun_name: hir::Name,
) -> Vec<NavigationTarget> {
    Impl::all_for_trait(sema.db, trait_)
        .into_iter()
        .filter_map(|imp| {
            let item = imp.items(sema.db).iter().find_map(|itm| {
                let itm_name = itm.name(sema.db)?;
                (itm_name == fun_name).then_some(*itm)
            })?;
            item.try_to_nav(sema.db)
        })
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use ide_db::FileRange;
    use itertools::Itertools;

    use crate::fixture;

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);

        let navs = analysis.goto_implementation(position).unwrap().unwrap().info;

        let cmp = |frange: &FileRange| (frange.file_id, frange.range.start());

        let actual = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        let expected =
            expected.into_iter().map(|(range, _)| range).sorted_by_key(cmp).collect::<Vec<_>>();
        assert_eq!(expected, actual);
    }

    #[test]
    fn goto_implementation_works() {
        check(
            r#"
struct Foo$0;
impl Foo {}
   //^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_works_multiple_blocks() {
        check(
            r#"
struct Foo$0;
impl Foo {}
   //^^^
impl Foo {}
   //^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_works_multiple_mods() {
        check(
            r#"
struct Foo$0;
mod a {
    impl super::Foo {}
       //^^^^^^^^^^
}
mod b {
    impl super::Foo {}
       //^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn goto_implementation_works_multiple_files() {
        check(
            r#"
//- /lib.rs
struct Foo$0;
mod a;
mod b;
//- /a.rs
impl crate::Foo {}
   //^^^^^^^^^^
//- /b.rs
impl crate::Foo {}
   //^^^^^^^^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_for_trait() {
        check(
            r#"
trait T$0 {}
struct Foo;
impl T for Foo {}
         //^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_for_trait_multiple_files() {
        check(
            r#"
//- /lib.rs
trait T$0 {};
struct Foo;
mod a;
mod b;
//- /a.rs
impl crate::T for crate::Foo {}
                //^^^^^^^^^^
//- /b.rs
impl crate::T for crate::Foo {}
                //^^^^^^^^^^
            "#,
        );
    }

    #[test]
    fn goto_implementation_all_impls() {
        check(
            r#"
//- /lib.rs
trait T {}
struct Foo$0;
impl Foo {}
   //^^^
impl T for Foo {}
         //^^^
impl T for &Foo {}
         //^^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_to_builtin_derive() {
        check(
            r#"
//- minicore: copy, derive
  #[derive(Copy)]
         //^^^^
struct Foo$0;
"#,
        );
    }

    #[test]
    fn goto_implementation_type_alias() {
        check(
            r#"
struct Foo;

type Bar$0 = Foo;

impl Foo {}
   //^^^
impl Bar {}
   //^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_adt_generic() {
        check(
            r#"
struct Foo$0<T>;

impl<T> Foo<T> {}
      //^^^^^^
impl Foo<str> {}
   //^^^^^^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_builtin() {
        check(
            r#"
//- /lib.rs crate:main deps:core
fn foo(_: bool$0) {{}}
//- /libcore.rs crate:core
#![rustc_coherence_is_core]
#[lang = "bool"]
impl bool {}
   //^^^^
"#,
        );
    }

    #[test]
    fn goto_implementation_trait_functions() {
        check(
            r#"
trait Tr {
    fn f$0();
}

struct S;

impl Tr for S {
    fn f() {
     //^
        println!("Hello, world!");
    }
}
"#,
        );
    }

    #[test]
    fn goto_implementation_trait_assoc_const() {
        check(
            r#"
trait Tr {
    const C$0: usize;
}

struct S;

impl Tr for S {
    const C: usize = 4;
        //^
}
"#,
        );
    }

    #[test]
    fn goto_adt_implementation_inside_block() {
        check(
            r#"
//- minicore: copy, derive
trait Bar {}

fn test() {
    #[derive(Copy)]
  //^^^^^^^^^^^^^^^
    struct Foo$0;

    impl Foo {}
       //^^^

    trait Baz {}

    impl Bar for Foo {}
               //^^^

    impl Baz for Foo {}
               //^^^
}
"#,
        );
    }

    #[test]
    fn goto_trait_implementation_inside_block() {
        check(
            r#"
struct Bar;

fn test() {
    trait Foo$0 {}

    struct Baz;

    impl Foo for Bar {}
               //^^^

    impl Foo for Baz {}
               //^^^
}
"#,
        );
        check(
            r#"
struct Bar;

fn test() {
    trait Foo {
        fn foo$0() {}
    }

    struct Baz;

    impl Foo for Bar {
        fn foo() {}
         //^^^
    }

    impl Foo for Baz {
        fn foo() {}
         //^^^
    }
}
"#,
        );
    }
}
