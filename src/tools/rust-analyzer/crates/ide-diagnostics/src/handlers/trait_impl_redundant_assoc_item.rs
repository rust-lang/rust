use hir::{HasSource, HirDisplay, db::ExpandDatabase};
use ide_db::text_edit::TextRange;
use ide_db::{
    assists::{Assist, AssistId},
    label::Label,
    source_change::SourceChangeBuilder,
};
use syntax::{
    AstNode, ToSmolStr,
    ast::{HasName, edit::AstNodeEdit},
};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: trait-impl-redundant-assoc_item
//
// Diagnoses redundant trait items in a trait impl.
pub(crate) fn trait_impl_redundant_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplRedundantAssocItems,
) -> Diagnostic {
    let db = ctx.sema.db;
    let name = d.assoc_item.0.clone();
    let redundant_assoc_item_name = name.display(db, ctx.edition);
    let assoc_item = d.assoc_item.1;

    let default_range = d.impl_.syntax_node_ptr().text_range();
    let trait_name = d.trait_.name(db).display_no_db(ctx.edition).to_smolstr();
    let indent_level = d.trait_.source(db).map_or(0, |it| it.value.indent_level().0) + 1;

    let (redundant_item_name, diagnostic_range, redundant_item_def) = match assoc_item {
        hir::AssocItem::Function(id) => {
            let function = id;
            (
                format!("`fn {redundant_assoc_item_name}`"),
                function.source(db).map(|it| it.syntax().text_range()).unwrap_or(default_range),
                format!("\n{};", function.display(db, ctx.display_target)),
            )
        }
        hir::AssocItem::Const(id) => {
            let constant = id;
            (
                format!("`const {redundant_assoc_item_name}`"),
                constant.source(db).map(|it| it.syntax().text_range()).unwrap_or(default_range),
                format!("\n{};", constant.display(db, ctx.display_target)),
            )
        }
        hir::AssocItem::TypeAlias(id) => {
            let type_alias = id;
            (
                format!("`type {redundant_assoc_item_name}`"),
                type_alias.source(db).map(|it| it.syntax().text_range()).unwrap_or(default_range),
                // FIXME cannot generate generic parameter and bounds
                format!("\ntype {};", type_alias.name(ctx.sema.db).display_no_db(ctx.edition)),
            )
        }
    };

    let hir::FileRange { file_id, range } =
        hir::InFile::new(d.file_id, diagnostic_range).original_node_file_range_rooted(db);
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0407"),
        format!("{redundant_item_name} is not a member of trait `{trait_name}`"),
        ide_db::FileRange { file_id: file_id.file_id(ctx.sema.db), range },
    )
    .stable()
    .with_fixes(quickfix_for_redundant_assoc_item(
        ctx,
        d,
        stdx::indent_string(&redundant_item_def, indent_level),
        diagnostic_range,
    ))
}

/// add assoc item into the trait def body
fn quickfix_for_redundant_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplRedundantAssocItems,
    redundant_item_def: String,
    range: TextRange,
) -> Option<Vec<Assist>> {
    let file_id = d.file_id.file_id()?;
    let add_assoc_item_def = |builder: &mut SourceChangeBuilder| -> Option<()> {
        let db = ctx.sema.db;
        let root = db.parse_or_expand(d.file_id);
        // don't modify trait def in outer crate
        let impl_def = d.impl_.to_node(&root);
        let current_crate = ctx.sema.scope(impl_def.syntax())?.krate();
        let trait_def_crate = d.trait_.module(db).krate(db);
        if trait_def_crate != current_crate {
            return None;
        }

        let trait_def = d.trait_.source(db)?.value;
        let insert_after = find_insert_after(range, &impl_def, &trait_def)?;

        let where_to_insert =
            hir::InFile::new(d.file_id, insert_after).original_node_file_range_rooted_opt(db)?;
        if where_to_insert.file_id != file_id {
            return None;
        }

        builder.insert(where_to_insert.range.end(), redundant_item_def);
        Some(())
    };
    let mut source_change_builder = SourceChangeBuilder::new(file_id.file_id(ctx.sema.db));
    add_assoc_item_def(&mut source_change_builder)?;

    Some(vec![Assist {
        id: AssistId::quick_fix("add assoc item def into trait def"),
        label: Label::new("Add assoc item def into trait def".to_owned()),
        group: None,
        target: range,
        source_change: Some(source_change_builder.finish()),
        command: None,
    }])
}

fn find_insert_after(
    redundant_range: TextRange,
    impl_def: &syntax::ast::Impl,
    trait_def: &syntax::ast::Trait,
) -> Option<TextRange> {
    let impl_items_before_redundant = impl_def
        .assoc_item_list()?
        .assoc_items()
        .take_while(|it| it.syntax().text_range().start() < redundant_range.start())
        .filter_map(|it| name_of(&it))
        .collect::<Vec<_>>();

    let after_item = trait_def
        .assoc_item_list()?
        .assoc_items()
        .filter(|it| {
            name_of(it).is_some_and(|name| {
                impl_items_before_redundant.iter().any(|it| it.text() == name.text())
            })
        })
        .last()
        .map(|it| it.syntax().text_range());

    return after_item.or_else(|| Some(trait_def.assoc_item_list()?.l_curly_token()?.text_range()));

    fn name_of(it: &syntax::ast::AssocItem) -> Option<syntax::ast::Name> {
        match it {
            syntax::ast::AssocItem::Const(it) => it.name(),
            syntax::ast::AssocItem::Fn(it) => it.name(),
            syntax::ast::AssocItem::TypeAlias(it) => it.name(),
            syntax::ast::AssocItem::MacroCall(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix, check_no_fix};

    #[test]
    fn quickfix_for_assoc_func() {
        check_fix(
            r#"
trait Marker {
    fn boo();
}
struct Foo;
impl Marker for Foo {
    fn$0 bar(_a: i32, _b: String) -> String {}
    fn boo() {}
}
            "#,
            r#"
trait Marker {
    fn bar(_a: i32, _b: String) -> String;
    fn boo();
}
struct Foo;
impl Marker for Foo {
    fn bar(_a: i32, _b: String) -> String {}
    fn boo() {}
}
            "#,
        )
    }

    #[test]
    fn quickfix_for_assoc_const() {
        check_fix(
            r#"
trait Marker {
    fn foo () {}
}
struct Foo;
impl Marker for Foo {
    const FLAG: bool$0 = false;
}
            "#,
            r#"
trait Marker {
    const FLAG: bool;
    fn foo () {}
}
struct Foo;
impl Marker for Foo {
    const FLAG: bool = false;
}
            "#,
        )
    }

    #[test]
    fn quickfix_for_assoc_type() {
        check_fix(
            r#"
trait Marker {
}
struct Foo;
impl Marker for Foo {
    type T = i32;$0
}
            "#,
            r#"
trait Marker {
    type T;
}
struct Foo;
impl Marker for Foo {
    type T = i32;
}
            "#,
        )
    }

    #[test]
    fn quickfix_indentations() {
        check_fix(
            r#"
mod indent {
    trait Marker {
        fn boo();
    }
    struct Foo;
    impl Marker for Foo {
        fn$0 bar<T: Copy>(_a: i32, _b: T) -> String {}
        fn boo() {}
    }
}
            "#,
            r#"
mod indent {
    trait Marker {
        fn bar<T>(_a: i32, _b: T) -> String
        where
            T: Copy,;
        fn boo();
    }
    struct Foo;
    impl Marker for Foo {
        fn bar<T: Copy>(_a: i32, _b: T) -> String {}
        fn boo() {}
    }
}
            "#,
        );

        check_fix(
            r#"
mod indent {
    trait Marker {
        fn foo () {}
    }
    struct Foo;
    impl Marker for Foo {
        const FLAG: bool$0 = false;
    }
}
            "#,
            r#"
mod indent {
    trait Marker {
        const FLAG: bool;
        fn foo () {}
    }
    struct Foo;
    impl Marker for Foo {
        const FLAG: bool = false;
    }
}
            "#,
        );

        check_fix(
            r#"
mod indent {
    trait Marker {
    }
    struct Foo;
    impl Marker for Foo {
        type T = i32;$0
    }
}
            "#,
            r#"
mod indent {
    trait Marker {
        type T;
    }
    struct Foo;
    impl Marker for Foo {
        type T = i32;
    }
}
            "#,
        );
    }

    #[test]
    fn quickfix_order() {
        check_fix(
            r#"
trait Marker {
    fn foo();
    fn baz();
}
struct Foo;
impl Marker for Foo {
    fn foo() {}
    fn missing() {}$0
    fn baz() {}
}
            "#,
            r#"
trait Marker {
    fn foo();
    fn missing();
    fn baz();
}
struct Foo;
impl Marker for Foo {
    fn foo() {}
    fn missing() {}
    fn baz() {}
}
            "#,
        );

        check_fix(
            r#"
trait Marker {
    type Item;
    fn bar();
    fn baz();
}
struct Foo;
impl Marker for Foo {
    type Item = Foo;
    fn missing() {}$0
    fn bar() {}
    fn baz() {}
}
            "#,
            r#"
trait Marker {
    type Item;
    fn missing();
    fn bar();
    fn baz();
}
struct Foo;
impl Marker for Foo {
    type Item = Foo;
    fn missing() {}
    fn bar() {}
    fn baz() {}
}
            "#,
        );
    }

    #[test]
    fn quickfix_dont_work() {
        check_no_fix(
            r#"
            //- /dep.rs crate:dep
            trait Marker {
            }
            //- /main.rs crate:main deps:dep
            struct Foo;
            impl dep::Marker for Foo {
                type T = i32;$0
            }
            "#,
        )
    }

    #[test]
    fn trait_with_default_value() {
        check_diagnostics(
            r#"
trait Marker {
    const FLAG: bool = false;
    fn boo();
    fn foo () {}
}
struct Foo;
impl Marker for Foo {
    type T = i32;
  //^^^^^^^^^^^^^ 💡 error: `type T` is not a member of trait `Marker`

    const FLAG: bool = true;

    fn bar() {}
  //^^^^^^^^^^^ 💡 error: `fn bar` is not a member of trait `Marker`

    fn boo() {}
}
            "#,
        )
    }

    #[test]
    fn dont_work_for_negative_impl() {
        check_diagnostics(
            r#"
trait Marker {
    const FLAG: bool = false;
    fn boo();
    fn foo () {}
}
struct Foo;
impl !Marker for Foo {
    type T = i32;
    const FLAG: bool = true;
    fn bar() {}
    fn boo() {}
}
            "#,
        )
    }
}
