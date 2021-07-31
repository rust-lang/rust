use std::cmp::Ordering;

use itertools::Itertools;

use syntax::{
    ast::{self, NameOwner},
    ted, AstNode, TextRange,
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
        match struct_ast.field_list() {
            Some(ast::FieldList::RecordFieldList(it)) => add_sort_fields_assist(acc, it),
            _ => {
                cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
                None
            }
        }
    } else if let Some(union_ast) = ctx.find_node_at_offset::<ast::Union>() {
        add_sort_fields_assist(acc, union_ast.record_field_list()?)
    } else if let Some(enum_ast) = ctx.find_node_at_offset::<ast::Enum>() {
        add_sort_variants_assist(acc, enum_ast.variant_list()?)
    } else {
        None
    }
}

trait AddRewrite {
    fn add_rewrite<T: AstNode>(
        &mut self,
        label: &str,
        old: Vec<T>,
        new: Vec<T>,
        target: TextRange,
    ) -> Option<()>;
}

impl AddRewrite for Assists {
    fn add_rewrite<T: AstNode>(
        &mut self,
        label: &str,
        old: Vec<T>,
        new: Vec<T>,
        target: TextRange,
    ) -> Option<()> {
        self.add(AssistId("sort_items", AssistKind::RefactorRewrite), label, target, |builder| {
            let mutable: Vec<_> = old.into_iter().map(|it| builder.make_mut(it)).collect();
            mutable
                .into_iter()
                .zip(new)
                .for_each(|(old, new)| ted::replace(old.syntax(), new.clone_for_update().syntax()));
        })
    }
}

fn add_sort_methods_assist(acc: &mut Assists, item_list: ast::AssocItemList) -> Option<()> {
    let methods = get_methods(&item_list);
    let sorted = sort_by_name(&methods);

    if methods == sorted {
        cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
        return None;
    }

    acc.add_rewrite("Sort methods alphabetically", methods, sorted, item_list.syntax().text_range())
}

fn add_sort_fields_assist(
    acc: &mut Assists,
    record_field_list: ast::RecordFieldList,
) -> Option<()> {
    let fields: Vec<_> = record_field_list.fields().collect();
    let sorted = sort_by_name(&fields);

    if fields == sorted {
        cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
        return None;
    }

    acc.add_rewrite(
        "Sort fields alphabetically",
        fields,
        sorted,
        record_field_list.syntax().text_range(),
    )
}

fn add_sort_variants_assist(acc: &mut Assists, variant_list: ast::VariantList) -> Option<()> {
    let variants: Vec<_> = variant_list.variants().collect();
    let sorted = sort_by_name(&variants);

    if variants == sorted {
        cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
        return None;
    }

    acc.add_rewrite(
        "Sort variants alphabetically",
        variants,
        sorted,
        variant_list.syntax().text_range(),
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
    fn not_applicable_if_trait_empty() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
t$0rait Bar {
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_impl_empty() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
struct Bar;            
$0impl Bar {
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_struct_empty() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0struct Bar;
        "#,
        )
    }

    #[test]
    fn not_applicable_if_struct_empty2() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0struct Bar { };
        "#,
        )
    }

    #[test]
    fn not_applicable_if_enum_empty() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0enum ZeroVariants {};
        "#,
        )
    }

    #[test]
    fn not_applicable_if_trait_sorted() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

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
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

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
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

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
    fn not_applicable_if_union_sorted() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0union Bar {
    a: u32,
    b: u8,
    c: u64,
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_enum_sorted() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0enum Bar {
    a,
    b,
    c,
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

    #[test]
    fn sort_union() {
        check_assist(
            sort_items,
            r#"
$0union Bar {
    b: u8,
    a: u32,
    c: u64,
}
        "#,
            r#"
union Bar {
    a: u32,
    b: u8,
    c: u64,
}
        "#,
        )
    }

    #[test]
    fn sort_enum() {
        check_assist(
            sort_items,
            r#"
$0enum Bar {
    d{ first: u32, second: usize},
    b = 14,
    a,
    c(u32, usize),
}
        "#,
            r#"
enum Bar {
    a,
    b = 14,
    c(u32, usize),
    d{ first: u32, second: usize},
}
        "#,
        )
    }
}
