//! Functionality for obtaining data related to traits from the DB.

use crate::RootDatabase;
use hir::Semantics;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, HasName},
    AstNode,
};

/// Given the `impl` block, attempts to find the trait this `impl` corresponds to.
pub fn resolve_target_trait(
    sema: &Semantics<RootDatabase>,
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
    sema: &Semantics<RootDatabase>,
    impl_def: &ast::Impl,
) -> Vec<hir::AssocItem> {
    // Names must be unique between constants and functions. However, type aliases
    // may share the same name as a function or constant.
    let mut impl_fns_consts = FxHashSet::default();
    let mut impl_type = FxHashSet::default();

    if let Some(item_list) = impl_def.assoc_item_list() {
        for item in item_list.assoc_items() {
            match item {
                ast::AssocItem::Fn(f) => {
                    if let Some(n) = f.name() {
                        impl_fns_consts.insert(n.syntax().to_string());
                    }
                }

                ast::AssocItem::TypeAlias(t) => {
                    if let Some(n) = t.name() {
                        impl_type.insert(n.syntax().to_string());
                    }
                }

                ast::AssocItem::Const(c) => {
                    if let Some(n) = c.name() {
                        impl_fns_consts.insert(n.syntax().to_string());
                    }
                }
                ast::AssocItem::MacroCall(_) => (),
            }
        }
    }

    resolve_target_trait(sema, impl_def).map_or(vec![], |target_trait| {
        target_trait
            .items(sema.db)
            .into_iter()
            .filter(|i| match i {
                hir::AssocItem::Function(f) => {
                    !impl_fns_consts.contains(&f.name(sema.db).to_string())
                }
                hir::AssocItem::TypeAlias(t) => !impl_type.contains(&t.name(sema.db).to_string()),
                hir::AssocItem::Const(c) => c
                    .name(sema.db)
                    .map(|n| !impl_fns_consts.contains(&n.to_string()))
                    .unwrap_or_default(),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::ChangeFixture, FilePosition};
    use expect_test::{expect, Expect};
    use hir::Semantics;
    use syntax::ast::{self, AstNode};

    use crate::RootDatabase;

    /// Creates analysis from a multi-file fixture, returns positions marked with $0.
    pub(crate) fn position(ra_fixture: &str) -> (RootDatabase, FilePosition) {
        let change_fixture = ChangeFixture::parse(ra_fixture);
        let mut database = RootDatabase::default();
        database.apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ($0)");
        let offset = range_or_offset.expect_offset();
        (database, FilePosition { file_id, offset })
    }

    fn check_trait(ra_fixture: &str, expect: Expect) {
        let (db, position) = position(ra_fixture);
        let sema = Semantics::new(&db);
        let file = sema.parse(position.file_id);
        let impl_block: ast::Impl =
            sema.find_node_at_offset_with_descend(file.syntax(), position.offset).unwrap();
        let trait_ = crate::traits::resolve_target_trait(&sema, &impl_block);
        let actual = match trait_ {
            Some(trait_) => trait_.name(&db).to_string(),
            None => String::new(),
        };
        expect.assert_eq(&actual);
    }

    fn check_missing_assoc(ra_fixture: &str, expect: Expect) {
        let (db, position) = position(ra_fixture);
        let sema = Semantics::new(&db);
        let file = sema.parse(position.file_id);
        let impl_block: ast::Impl =
            sema.find_node_at_offset_with_descend(file.syntax(), position.offset).unwrap();
        let items = crate::traits::get_missing_assoc_items(&sema, &impl_block);
        let actual = items
            .into_iter()
            .map(|item| item.name(&db).unwrap().to_string())
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
    }
}
