use itertools::Itertools;
use stdx::to_lower_snake_case;
use syntax::ast::VisibilityOwner;
use syntax::ast::{self, AstNode, NameOwner};

use crate::{AssistContext, AssistId, AssistKind, Assists, assist_context::AssistBuilder, utils::{find_impl_block_end, find_struct_impl, generate_impl_text}};

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
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));
            let method = format!(
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

            add_method_to_adt(builder, &parent_enum, impl_def, &method);
        },
    )
}

// Assist: generate_enum_into_method
//
// Generate an `into_` method for an enum variant.
//
// ```
// enum Value {
//  Number(i32),
//  Text(String)$0,
// }
// ```
// ->
// ```
// enum Value {
//  Number(i32),
//  Text(String),
// }
//
// impl Value {
//     fn into_text(self) -> Option<String> {
//         if let Self::Text(v) = self {
//             Some(v)
//         } else {
//             None
//         }
//     }
// }
// ```
pub(crate) fn generate_enum_into_method(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let parent_enum = ast::Adt::Enum(variant.parent_enum());
    let variant_kind = variant_kind(&variant);

    let fn_name = format!("into_{}", &to_lower_snake_case(variant_name.text()));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(
        &ctx,
        &parent_enum,
        &fn_name,
    )?;

    let field_type = variant_kind.single_field_type()?;
    let (pattern_suffix, bound_name) = variant_kind.binding_pattern()?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId("generate_enum_into_method", AssistKind::Generate),
        "Generate an `into_` method for an enum variant",
        target,
        |builder| {
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));
            let method = format!(
                "    {}fn {}(self) -> Option<{}> {{
        if let Self::{}{} = self {{
            Some({})
        }} else {{
            None
        }}
    }}",
                vis,
                fn_name,
                field_type.syntax(),
                variant_name,
                pattern_suffix,
                bound_name,
            );

            add_method_to_adt(builder, &parent_enum, impl_def, &method);
        },
    )
}

// Assist: generate_enum_as_method
//
// Generate an `as_` method for an enum variant.
//
// ```
// enum Value {
//  Number(i32),
//  Text(String)$0,
// }
// ```
// ->
// ```
// enum Value {
//  Number(i32),
//  Text(String),
// }
//
// impl Value {
//     fn as_text(&self) -> Option<&String> {
//         if let Self::Text(v) = self {
//             Some(v)
//         } else {
//             None
//         }
//     }
// }
// ```
pub(crate) fn generate_enum_as_method(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let parent_enum = ast::Adt::Enum(variant.parent_enum());
    let variant_kind = variant_kind(&variant);

    let fn_name = format!("as_{}", &to_lower_snake_case(variant_name.text()));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(
        &ctx,
        &parent_enum,
        &fn_name,
    )?;

    let field_type = variant_kind.single_field_type()?;
    let (pattern_suffix, bound_name) = variant_kind.binding_pattern()?;

    let target = variant.syntax().text_range();
    acc.add(
        AssistId("generate_enum_as_method", AssistKind::Generate),
        "Generate an `as_` method for an enum variant",
        target,
        |builder| {
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));
            let method = format!(
                "    {}fn {}(&self) -> Option<&{}> {{
        if let Self::{}{} = self {{
            Some({})
        }} else {{
            None
        }}
    }}",
                vis,
                fn_name,
                field_type.syntax(),
                variant_name,
                pattern_suffix,
                bound_name,
            );

            add_method_to_adt(builder, &parent_enum, impl_def, &method);
        },
    )
}

fn add_method_to_adt(
    builder: &mut AssistBuilder,
    adt: &ast::Adt,
    impl_def: Option<ast::Impl>,
    method: &str,
) {
    let mut buf = String::with_capacity(method.len() + 2);
    if impl_def.is_some() {
        buf.push('\n');
    }
    buf.push_str(method);

    let start_offset = impl_def
        .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
        .unwrap_or_else(|| {
            buf = generate_impl_text(&adt, &buf);
            adt.syntax().text_range().end()
        });

    builder.insert(start_offset, buf);
}

enum VariantKind {
    Unit,
    /// Tuple with a single field
    NewtypeTuple { ty: Option<ast::Type> },
    /// Tuple with 0 or more than 2 fields
    Tuple,
    /// Record with a single field
    NewtypeRecord {
        field_name: Option<ast::Name>,
        field_type: Option<ast::Type>,
    },
    /// Record with 0 or more than 2 fields
    Record,
}

impl VariantKind {
    fn pattern_suffix(&self) -> &'static str {
        match self {
            VariantKind::Unit => "",
            VariantKind::NewtypeTuple { .. } |
            VariantKind::Tuple => "(..)",
            VariantKind::NewtypeRecord { .. } |
            VariantKind::Record => " { .. }",
        }
    }

    fn binding_pattern(&self) -> Option<(String, String)> {
        match self {
            VariantKind::Unit |
            VariantKind::Tuple |
            VariantKind::Record |
            VariantKind::NewtypeRecord { field_name: None, .. } => None,
            VariantKind::NewtypeTuple { .. } => {
                Some(("(v)".to_owned(), "v".to_owned()))
            }
            VariantKind::NewtypeRecord { field_name: Some(name), .. } => {
                Some((
                    format!(" {{ {} }}", name.syntax()),
                    name.syntax().to_string(),
                ))
            }
        }
    }

    fn single_field_type(&self) -> Option<&ast::Type> {
        match self {
            VariantKind::Unit |
            VariantKind::Tuple |
            VariantKind::Record => None,
            VariantKind::NewtypeTuple { ty } => ty.as_ref(),
            VariantKind::NewtypeRecord { field_type, .. } => field_type.as_ref(),
        }
    }
}

fn variant_kind(variant: &ast::Variant) -> VariantKind {
    match variant.kind() {
        ast::StructKind::Record(record) => {
            if let Some((single_field,)) = record.fields().collect_tuple() {
                let field_name = single_field.name();
                let field_type = single_field.ty();
                VariantKind::NewtypeRecord { field_name, field_type }
            } else {
                VariantKind::Record
            }
        }
        ast::StructKind::Tuple(tuple) => {
            if let Some((single_field,)) = tuple.fields().collect_tuple() {
                let ty = single_field.ty();
                VariantKind::NewtypeTuple { ty }
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

    #[test]
    fn test_generate_enum_into_tuple_variant() {
        check_assist(
            generate_enum_into_method,
            r#"
enum Value {
    Number(i32),
    Text(String)$0,
}"#,
            r#"enum Value {
    Number(i32),
    Text(String),
}

impl Value {
    fn into_text(self) -> Option<String> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_into_already_implemented() {
        check_assist_not_applicable(
            generate_enum_into_method,
            r#"enum Value {
    Number(i32),
    Text(String)$0,
}

impl Value {
    fn into_text(self) -> Option<String> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_into_unit_variant() {
        check_assist_not_applicable(
            generate_enum_into_method,
            r#"enum Value {
    Number(i32),
    Text(String),
    Unit$0,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_into_record_with_multiple_fields() {
        check_assist_not_applicable(
            generate_enum_into_method,
            r#"enum Value {
    Number(i32),
    Text(String),
    Both { first: i32, second: String }$0,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_into_tuple_with_multiple_fields() {
        check_assist_not_applicable(
            generate_enum_into_method,
            r#"enum Value {
    Number(i32),
    Text(String, String)$0,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_into_record_variant() {
        check_assist(
            generate_enum_into_method,
            r#"enum Value {
    Number(i32),
    Text { text: String }$0,
}"#,
            r#"enum Value {
    Number(i32),
    Text { text: String },
}

impl Value {
    fn into_text(self) -> Option<String> {
        if let Self::Text { text } = self {
            Some(text)
        } else {
            None
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_as_tuple_variant() {
        check_assist(
            generate_enum_as_method,
            r#"
enum Value {
    Number(i32),
    Text(String)$0,
}"#,
            r#"enum Value {
    Number(i32),
    Text(String),
}

impl Value {
    fn as_text(&self) -> Option<&String> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_as_record_variant() {
        check_assist(
            generate_enum_as_method,
            r#"enum Value {
    Number(i32),
    Text { text: String }$0,
}"#,
            r#"enum Value {
    Number(i32),
    Text { text: String },
}

impl Value {
    fn as_text(&self) -> Option<&String> {
        if let Self::Text { text } = self {
            Some(text)
        } else {
            None
        }
    }
}"#,
        );
    }
}
