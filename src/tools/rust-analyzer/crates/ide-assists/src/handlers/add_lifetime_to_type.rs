use syntax::ast::{self, AstNode, HasGenericParams, HasName};

use crate::{AssistContext, AssistId, Assists};

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
pub(crate) fn add_lifetime_to_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let ref_type_focused = ctx.find_node_at_offset::<ast::RefType>()?;
    if ref_type_focused.lifetime().is_some() {
        return None;
    }

    let node = ctx.find_node_at_offset::<ast::Adt>()?;
    let has_lifetime = node
        .generic_param_list()
        .is_some_and(|gen_list| gen_list.lifetime_params().next().is_some());

    if has_lifetime {
        return None;
    }

    let ref_types = fetch_borrowed_types(&node)?;
    let target = node.syntax().text_range();

    acc.add(AssistId::generate("add_lifetime_to_type"), "Add lifetime", target, |builder| {
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
    })
}

fn fetch_borrowed_types(node: &ast::Adt) -> Option<Vec<ast::RefType>> {
    let ref_types: Vec<ast::RefType> = match node {
        ast::Adt::Enum(enum_) => {
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
        ast::Adt::Struct(strukt) => {
            let field_list = strukt.field_list()?;
            find_ref_types_from_field_list(&field_list)?
        }
        ast::Adt::Union(un) => {
            let record_field_list = un.record_field_list()?;
            record_field_list
                .fields()
                .filter_map(|r_field| {
                    if let ast::Type::RefType(ref_type) = r_field.ty()?
                        && ref_type.lifetime().is_none()
                    {
                        return Some(ref_type);
                    }

                    None
                })
                .collect()
        }
    };

    if ref_types.is_empty() { None } else { Some(ref_types) }
}

fn find_ref_types_from_field_list(field_list: &ast::FieldList) -> Option<Vec<ast::RefType>> {
    let ref_types: Vec<ast::RefType> = match field_list {
        ast::FieldList::RecordFieldList(record_list) => record_list
            .fields()
            .filter_map(|f| {
                if let ast::Type::RefType(ref_type) = f.ty()?
                    && ref_type.lifetime().is_none()
                {
                    return Some(ref_type);
                }

                None
            })
            .collect(),
        ast::FieldList::TupleFieldList(tuple_field_list) => tuple_field_list
            .fields()
            .filter_map(|f| {
                if let ast::Type::RefType(ref_type) = f.ty()?
                    && ref_type.lifetime().is_none()
                {
                    return Some(ref_type);
                }

                None
            })
            .collect(),
    };

    if ref_types.is_empty() { None } else { Some(ref_types) }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_lifetime_to_struct() {
        check_assist(
            add_lifetime_to_type,
            r#"struct Foo { a: &$0i32 }"#,
            r#"struct Foo<'a> { a: &'a i32 }"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"struct Foo { a: &$0i32, b: &usize }"#,
            r#"struct Foo<'a> { a: &'a i32, b: &'a usize }"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"struct Foo { a: &$0i32, b: usize }"#,
            r#"struct Foo<'a> { a: &'a i32, b: usize }"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"struct Foo<T> { a: &$0T, b: usize }"#,
            r#"struct Foo<'a, T> { a: &'a T, b: usize }"#,
        );

        check_assist_not_applicable(add_lifetime_to_type, r#"struct Foo<'a> { a: &$0'a i32 }"#);
        check_assist_not_applicable(add_lifetime_to_type, r#"struct Foo { a: &'a$0 i32 }"#);
    }

    #[test]
    fn add_lifetime_to_enum() {
        check_assist(
            add_lifetime_to_type,
            r#"enum Foo { Bar { a: i32 }, Other, Tuple(u32, &$0u32)}"#,
            r#"enum Foo<'a> { Bar { a: i32 }, Other, Tuple(u32, &'a u32)}"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"enum Foo { Bar { a: &$0i32 }}"#,
            r#"enum Foo<'a> { Bar { a: &'a i32 }}"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"enum Foo<T> { Bar { a: &$0i32, b: &T }}"#,
            r#"enum Foo<'a, T> { Bar { a: &'a i32, b: &'a T }}"#,
        );

        check_assist_not_applicable(
            add_lifetime_to_type,
            r#"enum Foo<'a> { Bar { a: &$0'a i32 }}"#,
        );
        check_assist_not_applicable(add_lifetime_to_type, r#"enum Foo { Bar, $0Misc }"#);
    }

    #[test]
    fn add_lifetime_to_union() {
        check_assist(
            add_lifetime_to_type,
            r#"union Foo { a: &$0i32 }"#,
            r#"union Foo<'a> { a: &'a i32 }"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"union Foo { a: &$0i32, b: &usize }"#,
            r#"union Foo<'a> { a: &'a i32, b: &'a usize }"#,
        );

        check_assist(
            add_lifetime_to_type,
            r#"union Foo<T> { a: &$0T, b: usize }"#,
            r#"union Foo<'a, T> { a: &'a T, b: usize }"#,
        );

        check_assist_not_applicable(add_lifetime_to_type, r#"struct Foo<'a> { a: &'a $0i32 }"#);
    }
}
