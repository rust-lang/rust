//! Functionality for obtaining data related to traits from the DB.

use crate::{RootDatabase, defs::Definition};
use base_db::FxIndexMap;
use hir::{AsAssocItem, HasAttrs, HasCrate, Semantics, db::HirDatabase, sym};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsRequiredAssocItem(pub bool);

/// Names must be unique between constants and functions. However, type aliases
/// may share the same name as a function or constant.
#[derive(PartialEq, Eq, Hash)]
enum AssocItemKind {
    FnOrConst,
    Type,
}

pub fn trait_items_with_required(
    db: &RootDatabase,
    trait_: hir::Trait,
) -> Vec<(hir::AssocItem, IsRequiredAssocItem)> {
    diff_assoc_items(db, trait_, Vec::new(), trait_.krate(db))
}

/// Given the `impl` block, returns the list of associated items (e.g. functions or types) that are
/// missing in this `impl` block.
pub fn get_missing_assoc_items(
    sema: &Semantics<'_, RootDatabase>,
    impl_def: &ast::Impl,
) -> Vec<(hir::AssocItem, IsRequiredAssocItem)> {
    let imp = match sema.to_def(impl_def) {
        Some(it) => it,
        None => return vec![],
    };

    let Some(target_trait) = imp.trait_(sema.db) else { return Vec::new() };

    diff_assoc_items(sema.db, target_trait, imp.items(sema.db), imp.krate(sema.db))
}

fn diff_assoc_items(
    db: &RootDatabase,
    target_trait: hir::Trait,
    impl_items: Vec<hir::AssocItem>,
    impl_crate: hir::Crate,
) -> Vec<(hir::AssocItem, IsRequiredAssocItem)> {
    // `Drop` has two methods, `drop()` and `pin_drop()`, and you can only implement one of them, so
    // we consider `pin_drop()` to not exist, unless you already implement it.
    let drop_trait = hir::Trait::lang(db, impl_crate, hir::LangItem::Drop);
    if let Some(drop_trait) = drop_trait
        && target_trait == drop_trait
    {
        return if impl_items.is_empty() {
            // No method implemented, return `drop()`.
            let drop_drop = drop_trait.function(db, sym::drop);
            match drop_drop {
                Some(drop_drop) => {
                    vec![(hir::AssocItem::Function(drop_drop), IsRequiredAssocItem(true))]
                }
                None => Vec::new(),
            }
        } else {
            // Some method is already implemented, leave it.
            Vec::new()
        };
    }

    let must_implement_one_of = target_trait.must_implement_one_of(db).unwrap_or_default();

    // We keep one map because we want to keep the trait's order.
    let mut trait_items = FxIndexMap::default();

    for i in target_trait.items(db) {
        match i {
            hir::AssocItem::Function(f) => {
                let is_required = !f.has_body(db);
                trait_items.insert(
                    (f.name(db), AssocItemKind::FnOrConst),
                    (i, IsRequiredAssocItem(is_required)),
                );
            }
            hir::AssocItem::Const(c) => {
                if let Some(name) = c.name(db) {
                    let is_required = !c.has_body(db);
                    trait_items.insert(
                        (name, AssocItemKind::FnOrConst),
                        (i, IsRequiredAssocItem(is_required)),
                    );
                }
            }
            hir::AssocItem::TypeAlias(t) => {
                let is_required = !t.has_type(db);
                trait_items.insert(
                    (t.name(db), AssocItemKind::Type),
                    (i, IsRequiredAssocItem(is_required)),
                );
            }
        }
    }

    let mut abides_must_implement_one_of = must_implement_one_of.is_empty();
    for item in impl_items {
        match item {
            hir::AssocItem::Function(it) => {
                let name = it.name(db);
                if !abides_must_implement_one_of && must_implement_one_of.contains(&name) {
                    abides_must_implement_one_of = true;
                }
                trait_items.shift_remove(&(name, AssocItemKind::FnOrConst));
            }
            hir::AssocItem::Const(it) => {
                if let Some(name) = it.name(db) {
                    trait_items.shift_remove(&(name, AssocItemKind::FnOrConst));
                }
            }
            hir::AssocItem::TypeAlias(it) => {
                trait_items.shift_remove(&(it.name(db), AssocItemKind::Type));
            }
        }
    }

    if !abides_must_implement_one_of {
        for name in must_implement_one_of {
            let Some((item, is_required)) =
                trait_items.get_mut(&(name.clone(), AssocItemKind::FnOrConst))
            else {
                continue;
            };
            if item
                .attrs(db)
                .unstable_feature(db)
                .is_none_or(|feature| impl_crate.is_unstable_feature_enabled(db, &feature))
            {
                // `#[rustc_must_implement_one_of]` always has all its methods with default body.
                // If it isn't followed, mark one as required.
                // We mark the first, see https://github.com/rust-lang/rust/pull/106643#issuecomment-5187934543.
                is_required.0 = true;
                break;
            }
        }
    }

    trait_items.into_values().collect()
}

/// Converts associated trait impl items to their trait definition counterpart
pub(crate) fn convert_to_def_in_trait<'db>(
    db: &'db dyn HirDatabase,
    def: Definition<'db>,
) -> Definition<'db> {
    (|| {
        let assoc = def.as_assoc_item(db)?;
        let trait_ = assoc.implemented_trait(db)?;
        assoc_item_of_trait(db, assoc, trait_)
    })()
    .unwrap_or(def)
}

/// If this is an trait (impl) assoc item, returns the assoc item of the corresponding trait definition.
pub(crate) fn as_trait_assoc_def<'db>(
    db: &dyn HirDatabase,
    def: Definition<'db>,
) -> Option<Definition<'db>> {
    let assoc = def.as_assoc_item(db)?;
    let trait_ = match assoc.container(db) {
        hir::AssocItemContainer::Trait(_) => return Some(def),
        hir::AssocItemContainer::Impl(i) => i.trait_(db),
    }?;
    assoc_item_of_trait(db, assoc, trait_)
}

fn assoc_item_of_trait<'db>(
    db: &dyn HirDatabase,
    assoc: hir::AssocItem,
    trait_: hir::Trait,
) -> Option<Definition<'db>> {
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
    use hir::{EditionedFileId, FilePosition, Semantics};
    use span::Edition;
    use syntax::ast::{self, AstNode};
    use test_fixture::ChangeFixture;

    use crate::RootDatabase;

    /// Creates analysis from a multi-file fixture, returns positions marked with $0.
    pub(crate) fn position(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> (RootDatabase, FilePosition) {
        let mut database = RootDatabase::default();
        let change_fixture = ChangeFixture::parse(ra_fixture);
        database.apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ($0)");

        let file_id = EditionedFileId::from_span_file_id(&database, file_id);
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
        let items =
            hir::attach_db(&db, || crate::traits::get_missing_assoc_items(&sema, &impl_block));
        let actual = items
            .into_iter()
            .map(|(item, _)| item.name(&db).unwrap().display(&db, Edition::CURRENT).to_string())
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
