use std::cmp::Ordering;

use hir::known::Option;
use itertools::Itertools;

use syntax::{
    ast::{self, NameOwner},
    ted, AstNode,
};

use crate::{utils::get_methods, AssistContext, AssistId, AssistKind, Assists};

// Assist: sort_items
//
pub(crate) fn sort_items(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if let Some(trait_ast) = ctx.find_node_at_offset::<ast::Trait>() {
        add_sort_methods_assist(acc, trait_ast.assoc_item_list()?)
    } else if let Some(impl_ast) = ctx.find_node_at_offset::<ast::Impl>() {
        add_sort_methods_assist(acc, impl_ast.assoc_item_list()?)
    } else if let Some(struct_ast) = ctx.find_node_at_offset::<ast::Struct>() {
        add_sort_fields_assist(acc, struct_ast.field_list()?)
    } else {
        None
    }
}

fn add_sort_methods_assist(acc: &mut Assists, item_list: ast::AssocItemList) -> Option<()> {
    let methods = get_methods(&item_list);
    let sorted = sort_by_name(&methods);

    if methods == sorted {
        cov_mark::hit!(not_applicable_if_sorted);
        return None;
    }

    acc.add(
        AssistId("sort_items", AssistKind::RefactorRewrite),
        "Sort methods alphabetically",
        item_list.syntax().text_range(),
        |builder| {
            let methods = methods.into_iter().map(|fn_| builder.make_mut(fn_)).collect::<Vec<_>>();
            methods
                .into_iter()
                .zip(sorted)
                .for_each(|(old, new)| ted::replace(old.syntax(), new.clone_for_update().syntax()));
        },
    )
}

fn add_sort_fields_assist(acc: &mut Assists, field_list: ast::FieldList) -> Option<()> {
    fn record_fields(field_list: &ast::FieldList) -> Option<Vec<ast::RecordField>> {
        match field_list {
            ast::FieldList::RecordFieldList(it) => Some(it.fields().collect()),
            ast::FieldList::TupleFieldList(_) => None,
        }
    }

    let fields = record_fields(&field_list)?;
    let sorted = sort_by_name(&fields);

    if fields == sorted {
        cov_mark::hit!(not_applicable_if_sorted);
        return None;
    }

    acc.add(
        AssistId("sort_items", AssistKind::RefactorRewrite),
        "Sort methods alphabetically",
        field_list.syntax().text_range(),
        |builder| {
            let methods = fields.into_iter().map(|fn_| builder.make_mut(fn_)).collect::<Vec<_>>();
            methods
                .into_iter()
                .zip(sorted)
                .for_each(|(old, new)| ted::replace(old.syntax(), new.clone_for_update().syntax()));
        },
    )
}

fn sort_by_name<T: NameOwner + Clone>(initial: &[T]) -> Vec<T> {
    initial
        .iter()
        .cloned()
        .sorted_by(|a, b| match (a.name(), b.name()) {
            (Some(a), Some(b)) => Ord::cmp(&a.to_string(), &b.to_string()),

            // unexpected, but just in case
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_if_trait_sorted() {
        cov_mark::check!(not_applicable_if_sorted);

        check_assist_not_applicable(
            sort_items,
            r#"
t$0rait Bar {
    fn a() {}
    fn b() {}
    fn c() {}
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_impl_sorted() {
        cov_mark::check!(not_applicable_if_sorted);

        check_assist_not_applicable(
            sort_items,
            r#"
struct Bar;            
$0impl Bar {
    fn a() {}
    fn b() {}
    fn c() {}
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_struct_sorted() {
        cov_mark::check!(not_applicable_if_sorted);

        check_assist_not_applicable(
            sort_items,
            r#"
$0struct Bar {
    a: u32,
    b: u8,
    c: u64,
}
        "#,
        )
    }

    #[test]
    fn sort_trait() {
        check_assist(
            sort_items,
            r#"
$0trait Bar {
    fn a() {}
    fn c() {}
    fn z() {}
    fn b() {}
}
        "#,
            r#"
trait Bar {
    fn a() {}
    fn b() {}
    fn c() {}
    fn z() {}
}
        "#,
        )
    }

    #[test]
    fn sort_impl() {
        check_assist(
            sort_items,
            r#"
struct Bar;
$0impl Bar {
    fn c() {}
    fn a() {}
    fn z() {}
    fn d() {}
}
        "#,
            r#"
struct Bar;
impl Bar {
    fn a() {}
    fn c() {}
    fn d() {}
    fn z() {}
}
        "#,
        )
    }

    #[test]
    fn sort_struct() {
        check_assist(
            sort_items,
            r#"
$0struct Bar {
    b: u8,
    a: u32,
    c: u64,
}
        "#,
            r#"
struct Bar {
    a: u32,
    b: u8,
    c: u64,
}
        "#,
        )
    }

    #[test]
    fn sort_generic_struct_with_lifetime() {
        check_assist(
            sort_items,
            r#"
$0struct Bar<'a, T> {
    d: &'a str,
    b: u8,
    a: T,
    c: u64,
}
        "#,
            r#"
struct Bar<'a, T> {
    a: T,
    b: u8,
    c: u64,
    d: &'a str,
}
        "#,
        )
    }

    #[test]
    fn sort_struct_fields_diff_len() {
        check_assist(
            sort_items,
            r#"
$0struct Bar {
    aaa: u8,
    a: usize,
    b: u8,
}
        "#,
            r#"
struct Bar {
    a: usize,
    aaa: u8,
    b: u8,
}
        "#,
        )
    }
}
