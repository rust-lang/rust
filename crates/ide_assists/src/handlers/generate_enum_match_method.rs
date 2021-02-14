use stdx::{format_to, to_lower_snake_case};
use syntax::ast::VisibilityOwner;
use syntax::ast::{self, AstNode, NameOwner};

use crate::{
    utils::{find_impl_block_end, find_struct_impl, generate_impl_text},
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
    let variant_kind = variant_kind(&variant);

    let enum_lowercase_name = to_lower_snake_case(&parent_enum.name()?.to_string());
    let fn_name = format!("is_{}", &to_lower_snake_case(variant_name.text()));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(
        &ctx,
        &parent_enum,
        &fn_name,
    )?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId("generate_enum_is_method", AssistKind::Generate),
        "Generate an `is_` method for an enum variant",
        target,
        |builder| {
            let mut buf = String::with_capacity(512);

            if impl_def.is_some() {
                buf.push('\n');
            }

            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));
            format_to!(
                buf,
                "    /// Returns `true` if the {} is [`{}`].
    {}fn {}(&self) -> bool {{
        matches!(self, Self::{}{})
    }}",
                enum_lowercase_name,
                variant_name,
                vis,
                fn_name,
                variant_name,
                variant_kind.pattern_suffix(),
            );

            let start_offset = impl_def
                .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
                .unwrap_or_else(|| {
                    buf = generate_impl_text(&parent_enum, &buf);
                    parent_enum.syntax().text_range().end()
                });

            builder.insert(start_offset, buf);
        },
    )
}

enum VariantKind {
    Unit,
    /// Tuple with a single field
    NewtypeTuple,
    /// Tuple with 0 or more than 2 fields
    Tuple,
    /// Record with a single field
    NewtypeRecord { field_name: Option<ast::Name> },
    /// Record with 0 or more than 2 fields
    Record,
}

impl VariantKind {
    fn pattern_suffix(&self) -> &'static str {
        match self {
            VariantKind::Unit => "",
            VariantKind::NewtypeTuple |
            VariantKind::Tuple => "(..)",
            VariantKind::NewtypeRecord { .. } |
            VariantKind::Record => " { .. }",
        }
    }
}

fn variant_kind(variant: &ast::Variant) -> VariantKind {
    match variant.kind() {
        ast::StructKind::Record(record) => {
            if record.fields().count() == 1 {
                let field_name = record.fields().nth(0).unwrap().name();
                VariantKind::NewtypeRecord { field_name }
            } else {
                VariantKind::Record
            }
        }
        ast::StructKind::Tuple(tuple) => {
            if tuple.fields().count() == 1 {
                VariantKind::NewtypeTuple
            } else {
                VariantKind::Tuple
            }
        }
        ast::StructKind::Unit => VariantKind::Unit,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    fn check_not_applicable(ra_fixture: &str) {
        check_assist_not_applicable(generate_enum_is_method, ra_fixture)
    }

    #[test]
    fn test_generate_enum_match_from_variant() {
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
    fn test_generate_enum_match_already_implemented() {
        check_not_applicable(
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
    fn test_generate_enum_match_from_tuple_variant() {
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
    fn test_generate_enum_match_from_record_variant() {
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
    fn test_generate_enum_match_from_variant_with_one_variant() {
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
    fn test_generate_enum_match_from_variant_with_visibility_marker() {
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
    fn test_multiple_generate_enum_match_from_variant() {
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
