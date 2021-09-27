use itertools::Itertools;
use stdx::to_lower_snake_case;
use syntax::ast::HasVisibility;
use syntax::ast::{self, AstNode, HasName};

use crate::{
    utils::{add_method_to_adt, find_struct_impl},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: generate_enum_try_into_method
//
// Generate an `try_into_` method for an enum variant.
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
//     fn try_into_text(self) -> Result<String, Self> {
//         if let Self::Text(v) = self {
//             Ok(v)
//         } else {
//             Err(self)
//         }
//     }
// }
// ```
pub(crate) fn generate_enum_try_into_method(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    generate_enum_projection_method(
        acc,
        ctx,
        "generate_enum_try_into_method",
        "Generate an `try_into_` method for an enum variant",
        ProjectionProps {
            fn_name_prefix: "try_into",
            self_param: "self",
            return_prefix: "Result<",
            return_suffix: ", Self>",
            happy_case: "Ok",
            sad_case: "Err(self)",
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
    generate_enum_projection_method(
        acc,
        ctx,
        "generate_enum_as_method",
        "Generate an `as_` method for an enum variant",
        ProjectionProps {
            fn_name_prefix: "as",
            self_param: "&self",
            return_prefix: "Option<&",
            return_suffix: ">",
            happy_case: "Some",
            sad_case: "None",
        },
    )
}

struct ProjectionProps {
    fn_name_prefix: &'static str,
    self_param: &'static str,
    return_prefix: &'static str,
    return_suffix: &'static str,
    happy_case: &'static str,
    sad_case: &'static str,
}

fn generate_enum_projection_method(
    acc: &mut Assists,
    ctx: &AssistContext,
    assist_id: &'static str,
    assist_description: &str,
    props: ProjectionProps,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let parent_enum = ast::Adt::Enum(variant.parent_enum());

    let (pattern_suffix, field_type, bound_name) = match variant.kind() {
        ast::StructKind::Record(record) => {
            let (field,) = record.fields().collect_tuple()?;
            let name = field.name()?.to_string();
            let ty = field.ty()?;
            let pattern_suffix = format!(" {{ {} }}", name);
            (pattern_suffix, ty, name)
        }
        ast::StructKind::Tuple(tuple) => {
            let (field,) = tuple.fields().collect_tuple()?;
            let ty = field.ty()?;
            ("(v)".to_owned(), ty, "v".to_owned())
        }
        ast::StructKind::Unit => return None,
    };

    let fn_name =
        format!("{}_{}", props.fn_name_prefix, &to_lower_snake_case(&variant_name.text()));

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(ctx, &parent_enum, &fn_name)?;

    let target = variant.syntax().text_range();
    acc.add(AssistId(assist_id, AssistKind::Generate), assist_description, target, |builder| {
        let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{} ", v));
        let method = format!(
            "    {0}fn {1}({2}) -> {3}{4}{5} {{
        if let Self::{6}{7} = self {{
            {8}({9})
        }} else {{
            {10}
        }}
    }}",
            vis,
            fn_name,
            props.self_param,
            props.return_prefix,
            field_type.syntax(),
            props.return_suffix,
            variant_name,
            pattern_suffix,
            props.happy_case,
            bound_name,
            props.sad_case,
        );

        add_method_to_adt(builder, &parent_enum, impl_def, &method);
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_enum_try_into_tuple_variant() {
        check_assist(
            generate_enum_try_into_method,
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
    fn try_into_text(self) -> Result<String, Self> {
        if let Self::Text(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_try_into_already_implemented() {
        check_assist_not_applicable(
            generate_enum_try_into_method,
            r#"enum Value {
    Number(i32),
    Text(String)$0,
}

impl Value {
    fn try_into_text(self) -> Result<String, Self> {
        if let Self::Text(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_enum_try_into_unit_variant() {
        check_assist_not_applicable(
            generate_enum_try_into_method,
            r#"enum Value {
    Number(i32),
    Text(String),
    Unit$0,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_try_into_record_with_multiple_fields() {
        check_assist_not_applicable(
            generate_enum_try_into_method,
            r#"enum Value {
    Number(i32),
    Text(String),
    Both { first: i32, second: String }$0,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_try_into_tuple_with_multiple_fields() {
        check_assist_not_applicable(
            generate_enum_try_into_method,
            r#"enum Value {
    Number(i32),
    Text(String, String)$0,
}"#,
        );
    }

    #[test]
    fn test_generate_enum_try_into_record_variant() {
        check_assist(
            generate_enum_try_into_method,
            r#"enum Value {
    Number(i32),
    Text { text: String }$0,
}"#,
            r#"enum Value {
    Number(i32),
    Text { text: String },
}

impl Value {
    fn try_into_text(self) -> Result<String, Self> {
        if let Self::Text { text } = self {
            Ok(text)
        } else {
            Err(self)
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
