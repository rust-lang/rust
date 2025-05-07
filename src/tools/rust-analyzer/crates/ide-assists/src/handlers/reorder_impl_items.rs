use hir::{PathResolution, Semantics};
use ide_db::{FxHashMap, RootDatabase};
use itertools::Itertools;
use syntax::{
    AstNode, SyntaxElement,
    ast::{self, HasName},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: reorder_impl_items
//
// Reorder the items of an `impl Trait`. The items will be ordered
// in the same order as in the trait definition.
//
// ```
// trait Foo {
//     type A;
//     const B: u8;
//     fn c();
// }
//
// struct Bar;
// $0impl Foo for Bar$0 {
//     const B: u8 = 17;
//     fn c() {}
//     type A = String;
// }
// ```
// ->
// ```
// trait Foo {
//     type A;
//     const B: u8;
//     fn c();
// }
//
// struct Bar;
// impl Foo for Bar {
//     type A = String;
//     const B: u8 = 17;
//     fn c() {}
// }
// ```
pub(crate) fn reorder_impl_items(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let impl_ast = ctx.find_node_at_offset::<ast::Impl>()?;
    let items = impl_ast.assoc_item_list()?;

    let parent_node = match ctx.covering_element() {
        SyntaxElement::Node(n) => n,
        SyntaxElement::Token(t) => t.parent()?,
    };

    // restrict the range
    // if cursor is in assoc_items, abort
    let assoc_range = items.syntax().text_range();
    let cursor_position = ctx.offset();
    if assoc_range.contains_inclusive(cursor_position) {
        cov_mark::hit!(not_applicable_editing_assoc_items);
        return None;
    }

    let assoc_items = items.assoc_items().collect::<Vec<_>>();

    let path = impl_ast
        .trait_()
        .and_then(|t| match t {
            ast::Type::PathType(path) => Some(path),
            _ => None,
        })?
        .path()?;

    let ranks = compute_item_ranks(&path, ctx)?;
    let sorted: Vec<_> = assoc_items
        .iter()
        .cloned()
        .sorted_by_key(|i| {
            let name = match i {
                ast::AssocItem::Const(c) => c.name(),
                ast::AssocItem::Fn(f) => f.name(),
                ast::AssocItem::TypeAlias(t) => t.name(),
                ast::AssocItem::MacroCall(_) => None,
            };

            name.and_then(|n| ranks.get(n.text().as_str().trim_start_matches("r#")).copied())
                .unwrap_or(usize::MAX)
        })
        .collect();

    // Don't edit already sorted methods:
    if assoc_items == sorted {
        cov_mark::hit!(not_applicable_if_sorted);
        return None;
    }

    let target = items.syntax().text_range();
    acc.add(
        AssistId::refactor_rewrite("reorder_impl_items"),
        "Sort items by trait definition",
        target,
        |builder| {
            let mut editor = builder.make_editor(&parent_node);

            assoc_items
                .into_iter()
                .zip(sorted)
                .for_each(|(old, new)| editor.replace(old.syntax(), new.syntax()));

            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn compute_item_ranks(
    path: &ast::Path,
    ctx: &AssistContext<'_>,
) -> Option<FxHashMap<String, usize>> {
    let td = trait_definition(path, &ctx.sema)?;

    Some(
        td.items(ctx.db())
            .iter()
            .flat_map(|i| i.name(ctx.db()))
            .enumerate()
            .map(|(idx, name)| (name.as_str().to_owned(), idx))
            .collect(),
    )
}

fn trait_definition(path: &ast::Path, sema: &Semantics<'_, RootDatabase>) -> Option<hir::Trait> {
    match sema.resolve_path(path)? {
        PathResolution::Def(hir::ModuleDef::Trait(trait_)) => Some(trait_),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_if_sorted() {
        cov_mark::check!(not_applicable_if_sorted);
        check_assist_not_applicable(
            reorder_impl_items,
            r#"
trait Bar {
    type T;
    const C: ();
    fn a() {}
    fn z() {}
    fn b() {}
}
struct Foo;
$0impl Bar for Foo {
    type T = ();
    const C: () = ();
    fn a() {}
    fn z() {}
    fn b() {}
}
        "#,
        )
    }

    #[test]
    fn reorder_impl_trait_functions() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    fn a() {}
    fn c() {}
    fn b() {}
    fn d() {}
}

struct Foo;
$0impl Bar for Foo {
    fn d() {}
    fn b() {}
    fn c() {}
    fn a() {}
}
"#,
            r#"
trait Bar {
    fn a() {}
    fn c() {}
    fn b() {}
    fn d() {}
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    fn c() {}
    fn b() {}
    fn d() {}
}
"#,
        )
    }

    #[test]
    fn not_applicable_if_empty() {
        check_assist_not_applicable(
            reorder_impl_items,
            r#"
trait Bar {};
struct Foo;
$0impl Bar for Foo {}
        "#,
        )
    }

    #[test]
    fn reorder_impl_trait_items() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
$0impl Bar for Foo {
    type T1 = ();
    fn d() {}
    fn b() {}
    fn c() {}
    const C1: () = ();
    fn a() {}
    type T0 = ();
    const C0: () = ();
}
        "#,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    type T0 = ();
    fn c() {}
    const C1: () = ();
    fn b() {}
    type T1 = ();
    fn d() {}
    const C0: () = ();
}
        "#,
        )
    }

    #[test]
    fn reorder_impl_trait_items_uneven_ident_lengths() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    type Foo;
    type Fooo;
}

struct Foo;
$0impl Bar for Foo {
    type Fooo = ();
    type Foo = ();
}"#,
            r#"
trait Bar {
    type Foo;
    type Fooo;
}

struct Foo;
impl Bar for Foo {
    type Foo = ();
    type Fooo = ();
}"#,
        )
    }

    #[test]
    fn not_applicable_editing_assoc_items() {
        cov_mark::check!(not_applicable_editing_assoc_items);
        check_assist_not_applicable(
            reorder_impl_items,
            r#"
trait Bar {
    type T;
    const C: ();
    fn a() {}
    fn z() {}
    fn b() {}
}
struct Foo;
impl Bar for Foo {
    type T = ();$0
    const C: () = ();
    fn z() {}
    fn a() {}
    fn b() {}
}
        "#,
        )
    }
}
