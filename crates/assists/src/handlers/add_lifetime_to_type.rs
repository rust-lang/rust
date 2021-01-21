use ast::FieldList;
use syntax::ast::{self, AstNode, GenericParamsOwner, NameOwner, RefType, Type};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: add_lifetime_to_type
//
// Adds a new lifetime to a struct, enum or union.
//
// ```
// struct Point {
//     x: &$0u32,
//     y: u32,
// }
// ```
// ->
// ```
// struct Point<'a> {
//     x: &'a u32,
//     y: u32,
// }
// ```
pub(crate) fn add_lifetime_to_type(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let ref_type_focused = ctx.find_node_at_offset::<ast::RefType>()?;
    if ref_type_focused.lifetime().is_some() {
        return None;
    }

    let node = ctx.find_node_at_offset::<ast::AdtDef>()?;
    let has_lifetime = node
        .generic_param_list()
        .map(|gen_list| gen_list.lifetime_params().count() > 0)
        .unwrap_or_default();

    if has_lifetime {
        return None;
    }

    let ref_types = fetch_borrowed_types(&node)?;
    let target = node.syntax().text_range();

    acc.add(
        AssistId("add_lifetime_to_type", AssistKind::Generate),
        "Add lifetime`",
        target,
        |builder| {
            match node.generic_param_list() {
                Some(gen_param) => {
                    if let Some(left_angle) = gen_param.l_angle_token() {
                        builder.insert(left_angle.text_range().end(), "'a, ");
                    }
                }
                None => {
                    if let Some(name) = node.name() {
                        builder.insert(name.syntax().text_range().end(), "<'a>");
                    }
                }
            }

            for ref_type in ref_types {
                if let Some(amp_token) = ref_type.amp_token() {
                    builder.insert(amp_token.text_range().end(), "'a ");
                }
            }
        },
    )
}

fn fetch_borrowed_types(node: &ast::AdtDef) -> Option<Vec<RefType>> {
    let ref_types: Vec<RefType> = match node {
        ast::AdtDef::Enum(enum_) => {
            let variant_list = enum_.variant_list()?;
            variant_list
                .variants()
                .filter_map(|variant| {
                    let field_list = variant.field_list()?;

                    find_ref_types_from_field_list(&field_list)
                })
                .flatten()
                .collect()
        }
        ast::AdtDef::Struct(strukt) => {
            let field_list = strukt.field_list()?;
            find_ref_types_from_field_list(&field_list)?
        }
        ast::AdtDef::Union(un) => {
            let record_field_list = un.record_field_list()?;
            record_field_list
                .fields()
                .filter_map(|r_field| {
                    if let Type::RefType(ref_type) = r_field.ty()? {
                        if ref_type.lifetime().is_none() {
                            return Some(ref_type);
                        }
                    }

                    None
                })
                .collect()
        }
    };

    if ref_types.is_empty() {
        None
    } else {
        Some(ref_types)
    }
}

fn find_ref_types_from_field_list(field_list: &FieldList) -> Option<Vec<RefType>> {
    let ref_types: Vec<RefType> = match field_list {
        ast::FieldList::RecordFieldList(record_list) => record_list
            .fields()
            .filter_map(|f| {
                if let Type::RefType(ref_type) = f.ty()? {
                    if ref_type.lifetime().is_none() {
                        return Some(ref_type);
                    }
                }

                None
            })
            .collect(),
        ast::FieldList::TupleFieldList(tuple_field_list) => tuple_field_list
            .fields()
            .filter_map(|f| {
                if let Type::RefType(ref_type) = f.ty()? {
                    if ref_type.lifetime().is_none() {
                        return Some(ref_type);
                    }
                }

                None
            })
            .collect(),
    };

    if ref_types.is_empty() {
        None
    } else {
        Some(ref_types)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_lifetime_to_struct() {
        check_assist(
            add_lifetime_to_type,
            "struct Foo { a: &$0i32 }",
            "struct Foo<'a> { a: &'a i32 }",
        );

        check_assist(
            add_lifetime_to_type,
            "struct Foo { a: &$0i32, b: &usize }",
            "struct Foo<'a> { a: &'a i32, b: &'a usize }",
        );

        check_assist(
            add_lifetime_to_type,
            "struct Foo { a: &$0i32, b: usize }",
            "struct Foo<'a> { a: &'a i32, b: usize }",
        );

        check_assist(
            add_lifetime_to_type,
            "struct Foo<T> { a: &$0T, b: usize }",
            "struct Foo<'a, T> { a: &'a T, b: usize }",
        );

        check_assist_not_applicable(add_lifetime_to_type, "struct Foo<'a> { a: &$0'a i32 }");
        check_assist_not_applicable(add_lifetime_to_type, "struct Foo { a: &'a$0 i32 }");
    }

    #[test]
    fn add_lifetime_to_enum() {
        check_assist(
            add_lifetime_to_type,
            "enum Foo { Bar { a: i32 }, Other, Tuple(u32, &$0u32)}",
            "enum Foo<'a> { Bar { a: i32 }, Other, Tuple(u32, &'a u32)}",
        );

        check_assist(
            add_lifetime_to_type,
            "enum Foo { Bar { a: &$0i32 }}",
            "enum Foo<'a> { Bar { a: &'a i32 }}",
        );

        check_assist(
            add_lifetime_to_type,
            "enum Foo<T> { Bar { a: &$0i32, b: &T }}",
            "enum Foo<'a, T> { Bar { a: &'a i32, b: &'a T }}",
        );

        check_assist_not_applicable(add_lifetime_to_type, "enum Foo<'a> { Bar { a: &$0'a i32 }}");
        check_assist_not_applicable(add_lifetime_to_type, "enum Foo { Bar, $0Misc }");
    }

    #[test]
    fn add_lifetime_to_union() {
        check_assist(
            add_lifetime_to_type,
            "union Foo { a: &$0i32 }",
            "union Foo<'a> { a: &'a i32 }",
        );

        check_assist(
            add_lifetime_to_type,
            "union Foo { a: &$0i32, b: &usize }",
            "union Foo<'a> { a: &'a i32, b: &'a usize }",
        );

        check_assist(
            add_lifetime_to_type,
            "union Foo<T> { a: &$0T, b: usize }",
            "union Foo<'a, T> { a: &'a T, b: usize }",
        );

        check_assist_not_applicable(add_lifetime_to_type, "struct Foo<'a> { a: &'a $0i32 }");
    }
}
