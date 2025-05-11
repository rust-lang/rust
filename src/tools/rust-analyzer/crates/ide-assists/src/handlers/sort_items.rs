use std::cmp::Ordering;

use itertools::Itertools;

use syntax::{
    AstNode, SyntaxNode,
    ast::{self, HasName},
};

use crate::{AssistContext, AssistId, Assists, utils::get_methods};

// Assist: sort_items
//
// Sorts item members alphabetically: fields, enum variants and methods.
//
// ```
// struct $0Foo$0 { second: u32, first: String }
// ```
// ->
// ```
// struct Foo { first: String, second: u32 }
// ```
// ---
// ```
// trait $0Bar$0 {
//     fn second(&self) -> u32;
//     fn first(&self) -> String;
// }
// ```
// ->
// ```
// trait Bar {
//     fn first(&self) -> String;
//     fn second(&self) -> u32;
// }
// ```
// ---
// ```
// struct Baz;
// impl $0Baz$0 {
//     fn second(&self) -> u32;
//     fn first(&self) -> String;
// }
// ```
// ->
// ```
// struct Baz;
// impl Baz {
//     fn first(&self) -> String;
//     fn second(&self) -> u32;
// }
// ```
// ---
// There is a difference between sorting enum variants:
//
// ```
// enum $0Animal$0 {
//   Dog(String, f64),
//   Cat { weight: f64, name: String },
// }
// ```
// ->
// ```
// enum Animal {
//   Cat { weight: f64, name: String },
//   Dog(String, f64),
// }
// ```
// and sorting a single enum struct variant:
//
// ```
// enum Animal {
//   Dog(String, f64),
//   Cat $0{ weight: f64, name: String }$0,
// }
// ```
// ->
// ```
// enum Animal {
//   Dog(String, f64),
//   Cat { name: String, weight: f64 },
// }
// ```
pub(crate) fn sort_items(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if ctx.has_empty_selection() {
        cov_mark::hit!(not_applicable_if_no_selection);
        return None;
    }

    if let Some(struct_ast) = ctx.find_node_at_offset::<ast::Struct>() {
        add_sort_field_list_assist(acc, struct_ast.field_list())
    } else if let Some(union_ast) = ctx.find_node_at_offset::<ast::Union>() {
        add_sort_fields_assist(acc, union_ast.record_field_list()?)
    } else if let Some(variant_ast) = ctx.find_node_at_offset::<ast::Variant>() {
        add_sort_field_list_assist(acc, variant_ast.field_list())
    } else if let Some(enum_struct_variant_ast) = ctx.find_node_at_offset::<ast::RecordFieldList>()
    {
        // should be above enum and below struct
        add_sort_fields_assist(acc, enum_struct_variant_ast)
    } else if let Some(enum_ast) = ctx.find_node_at_offset::<ast::Enum>() {
        add_sort_variants_assist(acc, enum_ast.variant_list()?)
    } else if let Some(trait_ast) = ctx.find_node_at_offset::<ast::Trait>() {
        add_sort_methods_assist(acc, ctx, trait_ast.assoc_item_list()?)
    } else if let Some(impl_ast) = ctx.find_node_at_offset::<ast::Impl>() {
        add_sort_methods_assist(acc, ctx, impl_ast.assoc_item_list()?)
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
        target: &SyntaxNode,
    ) -> Option<()>;
}

impl AddRewrite for Assists {
    fn add_rewrite<T: AstNode>(
        &mut self,
        label: &str,
        old: Vec<T>,
        new: Vec<T>,
        target: &SyntaxNode,
    ) -> Option<()> {
        self.add(AssistId::refactor_rewrite("sort_items"), label, target.text_range(), |builder| {
            let mut editor = builder.make_editor(target);

            old.into_iter()
                .zip(new)
                .for_each(|(old, new)| editor.replace(old.syntax(), new.syntax()));

            builder.add_file_edits(builder.file_id, editor)
        })
    }
}

fn add_sort_field_list_assist(acc: &mut Assists, field_list: Option<ast::FieldList>) -> Option<()> {
    match field_list {
        Some(ast::FieldList::RecordFieldList(it)) => add_sort_fields_assist(acc, it),
        _ => {
            cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
            None
        }
    }
}

fn add_sort_methods_assist(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    item_list: ast::AssocItemList,
) -> Option<()> {
    let selection = ctx.selection_trimmed();

    // ignore assist if the selection intersects with an associated item.
    if item_list.assoc_items().any(|item| item.syntax().text_range().intersect(selection).is_some())
    {
        return None;
    }

    let methods = get_methods(&item_list);
    let sorted = sort_by_name(&methods);

    if methods == sorted {
        cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
        return None;
    }

    acc.add_rewrite("Sort methods alphabetically", methods, sorted, item_list.syntax())
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

    acc.add_rewrite("Sort fields alphabetically", fields, sorted, record_field_list.syntax())
}

fn add_sort_variants_assist(acc: &mut Assists, variant_list: ast::VariantList) -> Option<()> {
    let variants: Vec<_> = variant_list.variants().collect();
    let sorted = sort_by_name(&variants);

    if variants == sorted {
        cov_mark::hit!(not_applicable_if_sorted_or_empty_or_single);
        return None;
    }

    acc.add_rewrite("Sort variants alphabetically", variants, sorted, variant_list.syntax())
}

fn sort_by_name<T: HasName + Clone>(initial: &[T]) -> Vec<T> {
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
    fn not_applicable_if_selection_in_fn_body() {
        check_assist_not_applicable(
            sort_items,
            r#"
struct S;
impl S {
    fn func2() {
        $0 bar $0
    }
    fn func() {}
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_selection_at_associated_const() {
        check_assist_not_applicable(
            sort_items,
            r#"
struct S;
impl S {
    fn func2() {}
    fn func() {}
    const C: () = $0()$0;
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_selection_overlaps_nodes() {
        check_assist_not_applicable(
            sort_items,
            r#"
struct S;
impl $0S {
    fn$0 func2() {}
    fn func() {}
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_no_selection() {
        cov_mark::check!(not_applicable_if_no_selection);

        check_assist_not_applicable(
            sort_items,
            r#"
t$0rait Bar {
    fn b();
    fn a();
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_selection_in_trait_fn_body() {
        check_assist_not_applicable(
            sort_items,
            r#"
trait Bar {
    fn b() {
        $0 hello $0
    }
    fn a();
}
        "#,
        )
    }

    #[test]
    fn not_applicable_if_trait_empty() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
t$0rait Bar$0 {
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
$0impl Bar$0 {
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
$0struct Bar$0 ;
        "#,
        )
    }

    #[test]
    fn not_applicable_if_struct_empty2() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0struct Bar$0 { };
        "#,
        )
    }

    #[test]
    fn not_applicable_if_enum_empty() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0enum ZeroVariants$0 {};
        "#,
        )
    }

    #[test]
    fn not_applicable_if_trait_sorted() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
t$0rait Bar$0 {
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
$0impl Bar$0 {
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
$0struct Bar$0 {
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
$0union Bar$0 {
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
$0enum Bar$0 {
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
$0trait Bar$0 {
    fn a() {

    }

    // comment for c
    fn c() {}
    fn z() {}
    fn b() {}
}
        "#,
            r#"
trait Bar {
    fn a() {

    }

    fn b() {}
    // comment for c
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
$0impl Bar$0 {
    fn c() {}
    fn a() {}
    /// long
    /// doc
    /// comment
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
    /// long
    /// doc
    /// comment
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
$0struct Bar$0 {
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
    fn sort_struct_inside_a_function() {
        check_assist(
            sort_items,
            r#"
fn hello() {
    $0struct Bar$0 {
        b: u8,
        a: u32,
        c: u64,
    }
}
        "#,
            r#"
fn hello() {
    struct Bar {
        a: u32,
        b: u8,
        c: u64,
    }
}
        "#,
        )
    }

    #[test]
    fn sort_generic_struct_with_lifetime() {
        check_assist(
            sort_items,
            r#"
$0struct Bar<'a,$0 T> {
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
$0struct Bar $0{
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
$0union Bar$0 {
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
$0enum Bar $0{
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

    #[test]
    fn sort_struct_enum_variant_fields() {
        check_assist(
            sort_items,
            r#"
enum Bar {
    d$0{ second: usize, first: u32 }$0,
    b = 14,
    a,
    c(u32, usize),
}
        "#,
            r#"
enum Bar {
    d{ first: u32, second: usize },
    b = 14,
    a,
    c(u32, usize),
}
        "#,
        )
    }

    #[test]
    fn sort_struct_enum_variant() {
        check_assist(
            sort_items,
            r#"
enum Bar {
    $0d$0{ second: usize, first: u32 },
}
        "#,
            r#"
enum Bar {
    d{ first: u32, second: usize },
}
        "#,
        )
    }
}
