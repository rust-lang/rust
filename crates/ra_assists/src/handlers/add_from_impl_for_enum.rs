use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    TextUnit,
};
use stdx::format_to;

use crate::{Assist, AssistCtx, AssistId};
use ra_ide_db::RootDatabase;

// Assist add_from_impl_for_enum
//
// Adds a From impl for an enum variant with one tuple field
//
// ```
// enum A { <|>One(u32) }
// ```
// ->
// ```
// enum A { One(u32) }
//
// impl From<u32> for A {
//     fn from(v: u32) -> Self {
//         A::One(v)
//     }
// }
// ```
pub(crate) fn add_from_impl_for_enum(ctx: AssistCtx) -> Option<Assist> {
    let variant = ctx.find_node_at_offset::<ast::EnumVariant>()?;
    let variant_name = variant.name()?;
    let enum_name = variant.parent_enum().name()?;
    let field_list = match variant.kind() {
        ast::StructKind::Tuple(field_list) => field_list,
        _ => return None,
    };
    if field_list.fields().count() != 1 {
        return None;
    }
    let field_type = field_list.fields().next()?.type_ref()?;
    let path = match field_type {
        ast::TypeRef::PathType(p) => p,
        _ => return None,
    };

    if already_has_from_impl(ctx.sema, &variant) {
        return None;
    }

    ctx.add_assist(
        AssistId("add_from_impl_for_enum"),
        "Add From impl for this enum variant",
        |edit| {
            let start_offset = variant.parent_enum().syntax().text_range().end();
            let mut buf = String::new();
            format_to!(
                buf,
                r#"

impl From<{0}> for {1} {{
    fn from(v: {0}) -> Self {{
        {1}::{2}(v)
    }}
}}"#,
                path.syntax(),
                enum_name,
                variant_name
            );
            edit.insert(start_offset, buf);
            edit.set_cursor(start_offset + TextUnit::of_str("\n\n"));
        },
    )
}

fn already_has_from_impl(
    sema: &'_ hir::Semantics<'_, RootDatabase>,
    variant: &ast::EnumVariant,
) -> bool {
    let scope = sema.scope(&variant.syntax());

    let from_path = ast::make::path_from_text("From");
    let from_hir_path = match hir::Path::from_ast(from_path) {
        Some(p) => p,
        None => return false,
    };
    let from_trait = match scope.resolve_hir_path(&from_hir_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(t))) => t,
        _ => return false,
    };

    let e: hir::Enum = match sema.to_def(&variant.parent_enum()) {
        Some(e) => e,
        None => return false,
    };
    let e_ty = e.ty(sema.db);

    let hir_enum_var: hir::EnumVariant = match sema.to_def(variant) {
        Some(ev) => ev,
        None => return false,
    };
    let var_ty = hir_enum_var.fields(sema.db)[0].signature_ty(sema.db);

    e_ty.impls_trait(sema.db, from_trait, &[var_ty])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_add_from_impl_for_enum() {
        check_assist(
            add_from_impl_for_enum,
            "enum A { <|>One(u32) }",
            r#"enum A { One(u32) }

<|>impl From<u32> for A {
    fn from(v: u32) -> Self {
        A::One(v)
    }
}"#,
        );
    }

    #[test]
    fn test_add_from_impl_for_enum_complicated_path() {
        check_assist(
            add_from_impl_for_enum,
            "enum A { <|>One(foo::bar::baz::Boo) }",
            r#"enum A { One(foo::bar::baz::Boo) }

<|>impl From<foo::bar::baz::Boo> for A {
    fn from(v: foo::bar::baz::Boo) -> Self {
        A::One(v)
    }
}"#,
        );
    }

    #[test]
    fn test_add_from_impl_no_element() {
        check_assist_not_applicable(add_from_impl_for_enum, "enum A { <|>One }");
    }

    #[test]
    fn test_add_from_impl_more_than_one_element_in_tuple() {
        check_assist_not_applicable(add_from_impl_for_enum, "enum A { <|>One(u32, String) }");
    }

    #[test]
    fn test_add_from_impl_struct_variant() {
        check_assist_not_applicable(add_from_impl_for_enum, "enum A { <|>One { x: u32 } }");
    }

    #[test]
    fn test_add_from_impl_already_exists() {
        check_assist_not_applicable(
            add_from_impl_for_enum,
            r#"enum A { <|>One(u32), }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        A::One(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}"#,
        );
    }

    #[test]
    fn test_add_from_impl_different_variant_impl_exists() {
        check_assist(
            add_from_impl_for_enum,
            r#"enum A { <|>One(u32), Two(String), }

impl From<String> for A {
    fn from(v: String) -> Self {
        A::Two(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}"#,
            r#"enum A { One(u32), Two(String), }

<|>impl From<u32> for A {
    fn from(v: u32) -> Self {
        A::One(v)
    }
}

impl From<String> for A {
    fn from(v: String) -> Self {
        A::Two(v)
    }
}

pub trait From<T> {
    fn from(T) -> Self;
}"#,
        );
    }
}
