use hir::{AsAssocItem, Impl, Semantics};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};
use syntax::{ast, AstNode};

use crate::{display::TryToNav, FilePosition, NavigationTarget, RangeInfo};

// Feature: Go to Implementation
//
// Navigates to the impl block of structs, enums or traits. Also implemented as a code lens.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[Ctrl+F12]
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113065566-02f85480-91b1-11eb-9288-aaad8abd8841.gif[]
pub(crate) fn goto_implementation(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);
    let syntax = source_file.syntax().clone();

    let node = sema.find_node_at_offset_with_descend(&syntax, position.offset)?;
    let def = match &node {
        ast::NameLike::Name(name) => {
            NameClass::classify(&sema, name).map(|class| class.referenced_or_defined(sema.db))
        }
        ast::NameLike::NameRef(name_ref) => {
            NameRefClass::classify(&sema, name_ref).map(|class| class.referenced(sema.db))
        }
        ast::NameLike::Lifetime(_) => None,
    }?;

    let def = match def {
        Definition::ModuleDef(def) => def,
        _ => return None,
    };
    let navs = match def {
        hir::ModuleDef::Trait(trait_) => impls_for_trait(&sema, trait_),
        hir::ModuleDef::Adt(adt) => impls_for_ty(&sema, adt.ty(sema.db)),
        hir::ModuleDef::TypeAlias(alias) => impls_for_ty(&sema, alias.ty(sema.db)),
        hir::ModuleDef::BuiltinType(builtin) => {
            let module = sema.to_module_def(position.file_id)?;
            impls_for_ty(&sema, builtin.ty(sema.db, module))
        }
        hir::ModuleDef::Function(f) => {
            let assoc = f.as_assoc_item(sema.db)?;
            let name = assoc.name(sema.db)?;
            let trait_ = assoc.containing_trait(sema.db)?;
            impls_for_trait_item(&sema, trait_, name)
        }
        hir::ModuleDef::Const(c) => {
            let assoc = c.as_assoc_item(sema.db)?;
            let name = assoc.name(sema.db)?;
            let trait_ = assoc.containing_trait(sema.db)?;
            impls_for_trait_item(&sema, trait_, name)
        }
        _ => return None,
    };
    Some(RangeInfo { range: node.syntax().text_range(), info: navs })
}

fn impls_for_ty(sema: &Semantics<RootDatabase>, ty: hir::Type) -> Vec<NavigationTarget> {
    Impl::all_for_type(sema.db, ty).into_iter().filter_map(|imp| imp.try_to_nav(sema.db)).collect()
}

fn impls_for_trait(sema: &Semantics<RootDatabase>, trait_: hir::Trait) -> Vec<NavigationTarget> {
    Impl::all_for_trait(sema.db, trait_)
        .into_iter()
        .filter_map(|imp| imp.try_to_nav(sema.db))
        .collect()
}

fn impls_for_trait_item(
    sema: &Semantics<RootDatabase>,
    trait_: hir::Trait,
    fun_name: hir::Name,
) -> Vec<NavigationTarget> {
    Impl::all_for_trait(sema.db, trait_)
        .into_iter()
        .filter_map(|imp| {
            let item = imp.items(sema.db).iter().find_map(|itm| {
                let itm_name = itm.name(sema.db)?;
                (itm_name == fun_name).then(|| itm.clone())
            })?;
            item.try_to_nav(sema.db)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, annotations) = fixture::annotations(ra_fixture);

        let navs = analysis.goto_implementation(position).unwrap().unwrap().info;

        let key = |frange: &FileRange| (frange.file_id, frange.range.start());

        let mut expected = annotations
            .into_iter()
            .map(|(range, data)| {
                assert!(data.is_empty());
                range
            })
            .collect::<Vec<_>>();
        expected.sort_by_key(key);

        let mut actual = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .collect::<Vec<_>>();
        actual.sort_by_key(key);

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
  #[derive(Copy)]
//^^^^^^^^^^^^^^^
struct Foo$0;

mod marker {
    trait Copy {}
}
#[rustc_builtin_macro]
macro Copy {}
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
}
