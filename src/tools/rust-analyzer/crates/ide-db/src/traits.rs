//! Functionality for obtaining data related to traits from the DB.

use crate::{RootDatabase, defs::Definition};
use hir::{AsAssocItem, Semantics, db::HirDatabase};
use rustc_hash::FxHashSet;
use syntax::{AstNode, ast};

/// Given the `impl` block, attempts to find the trait this `impl` corresponds to.
pub fn resolve_target_trait(
    sema: &Semantics<'_, RootDatabase>,
    impl_def: &ast::Impl,
) -> Option<hir::Trait> {
    let ast_path =
        impl_def.trait_().map(|it| it.syntax().clone()).and_then(ast::PathType::cast)?.path()?;

    match sema.resolve_path(&ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => Some(def),
        _ => None,
    }
}

/// Given the `impl` block, returns the list of associated items (e.g. functions or types) that are
/// missing in this `impl` block.
pub fn get_missing_assoc_items(
    sema: &Semantics<'_, RootDatabase>,
    impl_def: &ast::Impl,
) -> Vec<hir::AssocItem> {
    let imp = match sema.to_def(impl_def) {
        Some(it) => it,
        None => return vec![],
    };

    // Names must be unique between constants and functions. However, type aliases
    // may share the same name as a function or constant.
    let mut impl_fns_consts = FxHashSet::default();
    let mut impl_type = FxHashSet::default();
    let edition = imp.module(sema.db).krate().edition(sema.db);

    for item in imp.items(sema.db) {
        match item {
            hir::AssocItem::Function(it) => {
                impl_fns_consts.insert(it.name(sema.db).display(sema.db, edition).to_string());
            }
            hir::AssocItem::Const(it) => {
                if let Some(name) = it.name(sema.db) {
                    impl_fns_consts.insert(name.display(sema.db, edition).to_string());
                }
            }
            hir::AssocItem::TypeAlias(it) => {
                impl_type.insert(it.name(sema.db).display(sema.db, edition).to_string());
            }
        }
    }

    resolve_target_trait(sema, impl_def).map_or(vec![], |target_trait| {
        target_trait
            .items(sema.db)
            .into_iter()
            .filter(|i| match i {
                hir::AssocItem::Function(f) => !impl_fns_consts
                    .contains(&f.name(sema.db).display(sema.db, edition).to_string()),
                hir::AssocItem::TypeAlias(t) => {
                    !impl_type.contains(&t.name(sema.db).display(sema.db, edition).to_string())
                }
                hir::AssocItem::Const(c) => c
                    .name(sema.db)
                    .map(|n| !impl_fns_consts.contains(&n.display(sema.db, edition).to_string()))
                    .unwrap_or_default(),
            })
            .collect()
    })
}

/// Converts associated trait impl items to their trait definition counterpart
pub(crate) fn convert_to_def_in_trait(db: &dyn HirDatabase, def: Definition) -> Definition {
    (|| {
        let assoc = def.as_assoc_item(db)?;
        let trait_ = assoc.implemented_trait(db)?;
        assoc_item_of_trait(db, assoc, trait_)
    })()
    .unwrap_or(def)
}

/// If this is an trait (impl) assoc item, returns the assoc item of the corresponding trait definition.
pub(crate) fn as_trait_assoc_def(db: &dyn HirDatabase, def: Definition) -> Option<Definition> {
    let assoc = def.as_assoc_item(db)?;
    let trait_ = match assoc.container(db) {
        hir::AssocItemContainer::Trait(_) => return Some(def),
        hir::AssocItemContainer::Impl(i) => i.trait_(db),
    }?;
    assoc_item_of_trait(db, assoc, trait_)
}

fn assoc_item_of_trait(
    db: &dyn HirDatabase,
    assoc: hir::AssocItem,
    trait_: hir::Trait,
) -> Option<Definition> {
    use hir::AssocItem::*;
    let name = match assoc {
        Function(it) => it.name(db),
        Const(it) => it.name(db)?,
        TypeAlias(it) => it.name(db),
    };
    let item = trait_.items(db).into_iter().find(|it| match (it, assoc) {
        (Function(trait_func), Function(_)) => trait_func.name(db) == name,
        (Const(trait_konst), Const(_)) => trait_konst.name(db).map_or(false, |it| it == name),
        (TypeAlias(trait_type_alias), TypeAlias(_)) => trait_type_alias.name(db) == name,
        _ => false,
    })?;
    Some(Definition::from(item))
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use hir::FilePosition;
    use hir::Semantics;
    use span::Edition;
    use syntax::ast::{self, AstNode};
    use test_fixture::ChangeFixture;

    use crate::RootDatabase;

    /// Creates analysis from a multi-file fixture, returns positions marked with $0.
    pub(crate) fn position(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> (RootDatabase, FilePosition) {
        let mut database = RootDatabase::default();
        let change_fixture = ChangeFixture::parse(&database, ra_fixture);
        database.apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ($0)");
        let offset = range_or_offset.expect_offset();
        (database, FilePosition { file_id, offset })
    }

    fn check_trait(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
        let (db, position) = position(ra_fixture);
        let sema = Semantics::new(&db);

        let file = sema.parse(position.file_id);
        let impl_block: ast::Impl =
            sema.find_node_at_offset_with_descend(file.syntax(), position.offset).unwrap();
        let trait_ = crate::traits::resolve_target_trait(&sema, &impl_block);
        let actual = match trait_ {
            Some(trait_) => trait_.name(&db).display(&db, Edition::CURRENT).to_string(),
            None => String::new(),
        };
        expect.assert_eq(&actual);
    }

    fn check_missing_assoc(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
        let (db, position) = position(ra_fixture);
        let sema = Semantics::new(&db);

        let file = sema.parse(position.file_id);
        let impl_block: ast::Impl =
            sema.find_node_at_offset_with_descend(file.syntax(), position.offset).unwrap();
        let items = crate::traits::get_missing_assoc_items(&sema, &impl_block);
        let actual = items
            .into_iter()
            .map(|item| item.name(&db).unwrap().display(&db, Edition::CURRENT).to_string())
            .collect::<Vec<_>>()
            .join("\n");
        expect.assert_eq(&actual);
    }

    #[test]
    fn resolve_trait() {
        check_trait(
            r#"
pub trait Foo {
    fn bar();
}
impl Foo for u8 {
    $0
}
            "#,
            expect![["Foo"]],
        );
        check_trait(
            r#"
pub trait Foo {
    fn bar();
}
impl Foo for u8 {
    fn bar() {
        fn baz() {
            $0
        }
        baz();
    }
}
            "#,
            expect![["Foo"]],
        );
        check_trait(
            r#"
pub trait Foo {
    fn bar();
}
pub struct Bar;
impl Bar {
    $0
}
            "#,
            expect![[""]],
        );
    }

    #[test]
    fn missing_assoc_items() {
        check_missing_assoc(
            r#"
pub trait Foo {
    const FOO: u8;
    fn bar();
}
impl Foo for u8 {
    $0
}"#,
            expect![[r#"
                FOO
                bar"#]],
        );

        check_missing_assoc(
            r#"
pub trait Foo {
    const FOO: u8;
    fn bar();
}
impl Foo for u8 {
    const FOO: u8 = 10;
    $0
}"#,
            expect![[r#"
                bar"#]],
        );

        check_missing_assoc(
            r#"
pub trait Foo {
    const FOO: u8;
    fn bar();
}
impl Foo for u8 {
    const FOO: u8 = 10;
    fn bar() {$0}
}"#,
            expect![[r#""#]],
        );

        check_missing_assoc(
            r#"
pub struct Foo;
impl Foo {
    fn bar() {$0}
}"#,
            expect![[r#""#]],
        );

        check_missing_assoc(
            r#"
trait Tr {
    fn required();
}
macro_rules! m {
    () => { fn required() {} };
}
impl Tr for () {
    m!();
    $0
}

            "#,
            expect![[r#""#]],
        );
    }
}
