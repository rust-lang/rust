use ide_db::{RootDatabase, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode, HasName};

use crate::{AssistContext, AssistId, Assists, utils::generate_trait_impl_text_intransitive};

// Assist: generate_from_impl_for_enum
//
// Adds a From impl for this enum variant with one tuple field.
//
// ```
// enum A { $0One(u32) }
// ```
// ->
// ```
// enum A { One(u32) }
//
// impl From<u32> for A {
//     fn from(v: u32) -> Self {
//         Self::One(v)
//     }
// }
// ```
pub(crate) fn generate_from_impl_for_enum(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let enum_ = ast::Adt::Enum(variant.parent_enum());
    let (field_name, field_type) = match variant.kind() {
        ast::StructKind::Tuple(field_list) => {
            if field_list.fields().count() != 1 {
                return None;
            }
            (None, field_list.fields().next()?.ty()?)
        }
        ast::StructKind::Record(field_list) => {
            if field_list.fields().count() != 1 {
                return None;
            }
            let field = field_list.fields().next()?;
            (Some(field.name()?), field.ty()?)
        }
        ast::StructKind::Unit => return None,
    };

    if existing_from_impl(&ctx.sema, &variant).is_some() {
        cov_mark::hit!(test_add_from_impl_already_exists);
        return None;
    }

    let target = variant.syntax().text_range();
    acc.add(
        AssistId::generate("generate_from_impl_for_enum"),
        "Generate `From` impl for this enum variant",
        target,
        |edit| {
            let start_offset = variant.parent_enum().syntax().text_range().end();
            let from_trait = format!("From<{field_type}>");
            let impl_code = if let Some(name) = field_name {
                format!(
                    r#"    fn from({name}: {field_type}) -> Self {{
        Self::{variant_name} {{ {name} }}
    }}"#
                )
            } else {
                format!(
                    r#"    fn from(v: {field_type}) -> Self {{
        Self::{variant_name}(v)
    }}"#
                )
            };
            let from_impl = generate_trait_impl_text_intransitive(&enum_, &from_trait, &impl_code);
            edit.insert(start_offset, from_impl);
        },
    )
}

fn existing_from_impl(
    sema: &'_ hir::Semantics<'_, RootDatabase>,
    variant: &ast::Variant,
) -> Option<()> {
    let variant = sema.to_def(variant)?;
    let enum_ = variant.parent_enum(sema.db);
    let krate = enum_.module(sema.db).krate();

    let from_trait = FamousDefs(sema, krate).core_convert_From()?;

    let enum_type = enum_.ty(sema.db);

    let wrapped_type = variant.fields(sema.db).first()?.ty(sema.db);

    if enum_type.impls_trait(sema.db, from_trait, &[wrapped_type]) { Some(()) } else { None }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_from_impl_for_enum() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32) }
"#,
            r#"
enum A { One(u32) }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_from_impl_for_enum_complicated_path() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(foo::bar::baz::Boo) }
"#,
            r#"
enum A { One(foo::bar::baz::Boo) }

impl From<foo::bar::baz::Boo> for A {
    fn from(v: foo::bar::baz::Boo) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_no_element() {
        check_assist_not_applicable(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One }
"#,
        );
    }

    #[test]
    fn test_add_from_impl_more_than_one_element_in_tuple() {
        check_assist_not_applicable(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32, String) }
"#,
        );
    }

    #[test]
    fn test_add_from_impl_struct_variant() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One { x: u32 } }
"#,
            r#"
enum A { One { x: u32 } }

impl From<u32> for A {
    fn from(x: u32) -> Self {
        Self::One { x }
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_already_exists() {
        cov_mark::check!(test_add_from_impl_already_exists);
        check_assist_not_applicable(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32), }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_different_variant_impl_exists() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(u32), Two(String), }

impl From<String> for A {
    fn from(v: String) -> Self {
        A::Two(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}
"#,
            r#"
enum A { One(u32), Two(String), }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
    }
}

impl From<String> for A {
    fn from(v: String) -> Self {
        A::Two(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_static_str() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum A { $0One(&'static str) }
"#,
            r#"
enum A { One(&'static str) }

impl From<&'static str> for A {
    fn from(v: &'static str) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_generic_enum() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum Generic<T, U: Clone> { $0One(T), Two(U) }
"#,
            r#"
enum Generic<T, U: Clone> { One(T), Two(U) }

impl<T, U: Clone> From<T> for Generic<T, U> {
    fn from(v: T) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_with_lifetime() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum Generic<'a> { $0One(&'a i32) }
"#,
            r#"
enum Generic<'a> { One(&'a i32) }

impl<'a> From<&'a i32> for Generic<'a> {
    fn from(v: &'a i32) -> Self {
        Self::One(v)
    }
}
"#,
        );
    }
}
