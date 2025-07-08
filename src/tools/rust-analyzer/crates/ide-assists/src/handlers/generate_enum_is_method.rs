use std::slice;

use ide_db::assists::GroupLabel;
use stdx::to_lower_snake_case;
use syntax::ast::HasVisibility;
use syntax::ast::{self, AstNode, HasName};

use crate::{
    AssistContext, AssistId, Assists,
    utils::{add_method_to_adt, find_struct_impl},
};

// Assist: generate_enum_is_method
//
// Generate an `is_` method for this enum variant.
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
//     ///
//     /// [`Minor`]: Version::Minor
//     #[must_use]
//     fn is_minor(&self) -> bool {
//         matches!(self, Self::Minor)
//     }
// }
// ```
pub(crate) fn generate_enum_is_method(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let parent_enum = ast::Adt::Enum(variant.parent_enum());
    let pattern_suffix = match variant.kind() {
        ast::StructKind::Record(_) => " { .. }",
        ast::StructKind::Tuple(_) => "(..)",
        ast::StructKind::Unit => "",
    };

    let enum_name = parent_enum.name()?;
    let enum_lowercase_name = to_lower_snake_case(&enum_name.to_string()).replace('_', " ");
    let fn_name = format!("is_{}", &to_lower_snake_case(&variant_name.text()));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(ctx, &parent_enum, slice::from_ref(&fn_name))?;

    let target = variant.syntax().text_range();
    acc.add_group(
        &GroupLabel("Generate an `is_`,`as_`, or `try_into_` for this enum variant".to_owned()),
        AssistId::generate("generate_enum_is_method"),
        "Generate an `is_` method for this enum variant",
        target,
        |builder| {
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{v} "));
            let method = format!(
                "    /// Returns `true` if the {enum_lowercase_name} is [`{variant_name}`].
    ///
    /// [`{variant_name}`]: {enum_name}::{variant_name}
    #[must_use]
    {vis}fn {fn_name}(&self) -> bool {{
        matches!(self, Self::{variant_name}{pattern_suffix})
    }}",
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
    ///
    /// [`Minor`]: Variant::Minor
    #[must_use]
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
    ///
    /// [`Minor`]: Variant::Minor
    #[must_use]
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
    ///
    /// [`Minor`]: Variant::Minor
    #[must_use]
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
    ///
    /// [`Undefined`]: Variant::Undefined
    #[must_use]
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
    ///
    /// [`Minor`]: Variant::Minor
    #[must_use]
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
    ///
    /// [`Minor`]: Variant::Minor
    #[must_use]
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
    ///
    /// [`Minor`]: Variant::Minor
    #[must_use]
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }

    /// Returns `true` if the variant is [`Major`].
    ///
    /// [`Major`]: Variant::Major
    #[must_use]
    fn is_major(&self) -> bool {
        matches!(self, Self::Major)
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_is_variant_names() {
        check_assist(
            generate_enum_is_method,
            r#"
enum CoroutineState {
    Yielded,
    Complete$0,
    Major,
}"#,
            r#"enum CoroutineState {
    Yielded,
    Complete,
    Major,
}

impl CoroutineState {
    /// Returns `true` if the coroutine state is [`Complete`].
    ///
    /// [`Complete`]: CoroutineState::Complete
    #[must_use]
    fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }
}"#,
        );
    }
}
