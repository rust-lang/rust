use hir::{Crate, Impl, Semantics};
use ide_db::RootDatabase;
use syntax::{algo::find_node_at_offset, ast, AstNode};

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
pub(crate) fn goto_implementation(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let source_file = sema.parse(position.file_id);
    let syntax = source_file.syntax().clone();

    let krate = sema.to_module_def(position.file_id)?.krate();

    if let Some(nominal_def) = find_node_at_offset::<ast::AdtDef>(&syntax, position.offset) {
        return Some(RangeInfo::new(
            nominal_def.syntax().text_range(),
            impls_for_def(&sema, &nominal_def, krate)?,
        ));
    } else if let Some(trait_def) = find_node_at_offset::<ast::Trait>(&syntax, position.offset) {
        return Some(RangeInfo::new(
            trait_def.syntax().text_range(),
            impls_for_trait(&sema, &trait_def, krate)?,
        ));
    }

    None
}

fn impls_for_def(
    sema: &Semantics<RootDatabase>,
    node: &ast::AdtDef,
    krate: Crate,
) -> Option<Vec<NavigationTarget>> {
    let ty = match node {
        ast::AdtDef::Struct(def) => sema.to_def(def)?.ty(sema.db),
        ast::AdtDef::Enum(def) => sema.to_def(def)?.ty(sema.db),
        ast::AdtDef::Union(def) => sema.to_def(def)?.ty(sema.db),
    };

    let impls = Impl::all_in_crate(sema.db, krate);

    Some(
        impls
            .into_iter()
            .filter(|impl_def| ty.is_equal_for_find_impls(&impl_def.target_ty(sema.db)))
            .filter_map(|imp| imp.try_to_nav(sema.db))
            .collect(),
    )
}

fn impls_for_trait(
    sema: &Semantics<RootDatabase>,
    node: &ast::Trait,
    krate: Crate,
) -> Option<Vec<NavigationTarget>> {
    let tr = sema.to_def(node)?;

    let impls = Impl::for_trait(sema.db, krate, tr);

    Some(impls.into_iter().filter_map(|imp| imp.try_to_nav(sema.db)).collect())
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
}
