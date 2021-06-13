use stdx::to_lower_snake_case;
use syntax::ast::VisibilityOwner;
use syntax::ast::{self, AstNode, NameOwner};

use crate::{
    utils::{add_method_to_adt, find_struct_impl},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: generate_enum_is_method
//
// Generate an `is_` method for an enum variant.
//
// ```
// enum Version {
//  Undefined,
//  Minor$0,
//  Major,
// }
// ```
// ->
// ```
// enum Version {
//  Undefined,
//  Minor,
//  Major,
// }
//
// impl Version {
//     /// Returns `true` if the version is [`Minor`].
//     fn is_minor(&self) -> bool {
//         matches!(self, Self::Minor)
//     }
// }
// ```
pub(crate) fn generate_enum_is_method(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let parent_enum = ast::Adt::Enum(variant.parent_enum());
    let pattern_suffix = match variant.kind() {
        ast::StructKind::Record(_) => " { .. }",
        ast::StructKind::Tuple(_) => "(..)",
        ast::StructKind::Unit => "",
    };

    let enum_lowercase_name = to_lower_snake_case(&parent_enum.name()?.to_string());
    let fn_name = format!("is_{}", &to_lower_snake_case(&variant_name.text()));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(ctx, &parent_enum, &fn_name)?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId("generate_enum_is_method", AssistKind::Generate),
        "Generate an `is_` method for an enum variant",
        target,
        |builder| {
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));
            let method = format!(
                "    /// Returns `true` if the {} is [`{}`].
    {}fn {}(&self) -> bool {{
        matches!(self, Self::{}{})
    }}",
                enum_lowercase_name, variant_name, vis, fn_name, variant_name, pattern_suffix,
            );

            add_method_to_adt(builder, &parent_enum, impl_def, &method);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_enum_is_from_variant() {
        check_assist(
            generate_enum_is_method,
            r#"
enum Variant {
    Undefined,
    Minor$0,
    Major,
}"#,
            r#"enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Variant {
    /// Returns `true` if the variant is [`Minor`].
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_is_already_implemented() {
        check_assist_not_applicable(
            generate_enum_is_method,
            r#"
enum Variant {
    Undefined,
    Minor$0,
    Major,
}

impl Variant {
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_is_from_tuple_variant() {
        check_assist(
            generate_enum_is_method,
            r#"
enum Variant {
    Undefined,
    Minor(u32)$0,
    Major,
}"#,
            r#"enum Variant {
    Undefined,
    Minor(u32),
    Major,
}

impl Variant {
    /// Returns `true` if the variant is [`Minor`].
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor(..))
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_is_from_record_variant() {
        check_assist(
            generate_enum_is_method,
            r#"
enum Variant {
    Undefined,
    Minor { foo: i32 }$0,
    Major,
}"#,
            r#"enum Variant {
    Undefined,
    Minor { foo: i32 },
    Major,
}

impl Variant {
    /// Returns `true` if the variant is [`Minor`].
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor { .. })
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_is_from_variant_with_one_variant() {
        check_assist(
            generate_enum_is_method,
            r#"enum Variant { Undefi$0ned }"#,
            r#"
enum Variant { Undefined }

impl Variant {
    /// Returns `true` if the variant is [`Undefined`].
    fn is_undefined(&self) -> bool {
        matches!(self, Self::Undefined)
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_is_from_variant_with_visibility_marker() {
        check_assist(
            generate_enum_is_method,
            r#"
pub(crate) enum Variant {
    Undefined,
    Minor$0,
    Major,
}"#,
            r#"pub(crate) enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Variant {
    /// Returns `true` if the variant is [`Minor`].
    pub(crate) fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
        );
    }

    #[test]
    fn test_multiple_generate_enum_is_from_variant() {
        check_assist(
            generate_enum_is_method,
            r#"
enum Variant {
    Undefined,
    Minor,
    Major$0,
}

impl Variant {
    /// Returns `true` if the variant is [`Minor`].
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}"#,
            r#"enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Variant {
    /// Returns `true` if the variant is [`Minor`].
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }

    /// Returns `true` if the variant is [`Major`].
    fn is_major(&self) -> bool {
        matches!(self, Self::Major)
    }
}"#,
        );
    }
}
