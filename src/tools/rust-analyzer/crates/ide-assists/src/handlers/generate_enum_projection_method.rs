use ide_db::assists::GroupLabel;
use stdx::to_lower_snake_case;
use syntax::{
    AstNode, Edition,
    ast::{self, HasName, HasVisibility, edit::AstNodeEdit},
    syntax_editor::Position,
};

use crate::{
    AssistContext, AssistId, Assists,
    utils::{find_struct_impl, generate_impl_with_item, is_selected},
};

// Assist: generate_enum_try_into_method
//
// Generate a `try_into_` method for this enum variant.
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
pub(crate) fn generate_enum_try_into_method(
    acc: &mut Assists,
    ctx: &AssistContext<'_, '_>,
) -> Option<()> {
    generate_enum_projection_method(
        acc,
        ctx,
        "generate_enum_try_into_method",
        "Generate a `try_into_` method for this enum variant",
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
// Generate an `as_` method for this enum variant.
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
pub(crate) fn generate_enum_as_method(
    acc: &mut Assists,
    ctx: &AssistContext<'_, '_>,
) -> Option<()> {
    generate_enum_projection_method(
        acc,
        ctx,
        "generate_enum_as_method",
        "Generate an `as_` method for this enum variant",
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
    ctx: &AssistContext<'_, '_>,
    assist_id: &'static str,
    assist_description: &str,
    props: ProjectionProps,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let parent_enum = ast::Adt::Enum(variant.parent_enum());
    let variants = variant
        .parent_enum()
        .variant_list()?
        .variants()
        .filter(|it| is_selected(it, ctx.selection_trimmed(), true))
        .collect::<Vec<_>>();
    let methods = variants
        .iter()
        .map(|variant| Method::new(variant, props.fn_name_prefix))
        .collect::<Option<Vec<_>>>()?;
    let fn_names = methods.iter().map(|it| it.fn_name.clone()).collect::<Vec<_>>();
    stdx::never!(variants.is_empty());

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(ctx, &parent_enum, &fn_names)?;

    let target = variant.syntax().text_range();
    acc.add_group(
        &GroupLabel("Generate an `is_`,`as_`, or `try_into_` for this enum variant".to_owned()),
        AssistId::generate(assist_id),
        assist_description,
        target,
        |builder| {
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{v} "));
            let must_use = if ctx.config.assist_emit_must_use { "#[must_use]\n" } else { "" };

            let fn_items: Vec<ast::AssocItem> = methods
                .iter()
                .map(|method| build_fn_item(method, &vis, must_use, &props))
                .collect();

            if let Some(impl_def) = &impl_def {
                let editor = builder.make_editor(impl_def.syntax());
                impl_def.assoc_item_list().unwrap().add_items(&editor, fn_items);
                builder.add_file_edits(ctx.vfs_file_id(), editor);
                return;
            }

            let editor = builder.make_editor(parent_enum.syntax());
            let make = editor.make();
            let indent = parent_enum.indent_level();
            let assoc_list = make.assoc_item_list(fn_items);
            let new_impl = generate_impl_with_item(make, &parent_enum, Some(assoc_list));
            editor.insert_all(
                Position::after(parent_enum.syntax()),
                vec![
                    make.whitespace(&format!("\n\n{indent}")).into(),
                    new_impl.syntax().clone().into(),
                ],
            );
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn build_fn_item(
    method: &Method,
    vis: &str,
    must_use: &str,
    props: &ProjectionProps,
) -> ast::AssocItem {
    let Method { pattern_suffix, field_type, bound_name, fn_name, variant_name } = method;
    let ProjectionProps { self_param, return_prefix, return_suffix, happy_case, sad_case, .. } =
        props;
    let fn_text = format!(
        "{must_use}{vis}fn {fn_name}({self_param}) -> {return_prefix}{field_type}{return_suffix} {{
    if let Self::{variant_name}{pattern_suffix} = self {{
        {happy_case}({bound_name})
    }} else {{
        {sad_case}
    }}
}}"
    );
    let wrapped = format!("impl X {{ {fn_text} }}");
    let parse = syntax::SourceFile::parse(&wrapped, Edition::CURRENT);
    let fn_ = parse
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Fn::cast)
        .expect("fn text must produce a valid fn node");
    ast::AssocItem::Fn(fn_.indent(1.into()))
}

struct Method {
    pattern_suffix: String,
    field_type: ast::Type,
    bound_name: String,
    fn_name: String,
    variant_name: ast::Name,
}

impl Method {
    fn new(variant: &ast::Variant, fn_name_prefix: &str) -> Option<Self> {
        use itertools::Itertools as _;
        let variant_name = variant.name()?;
        let fn_name = format!("{fn_name_prefix}_{}", &to_lower_snake_case(&variant_name.text()));

        match variant.kind() {
            ast::StructKind::Record(record) => {
                let (field,) = record.fields().collect_tuple()?;
                let name = field.name()?.to_string();
                let field_type = field.ty()?;
                let pattern_suffix = format!(" {{ {name} }}");
                Some(Method { pattern_suffix, field_type, bound_name: name, fn_name, variant_name })
            }
            ast::StructKind::Tuple(tuple) => {
                let (field,) = tuple.fields().collect_tuple()?;
                let field_type = field.ty()?;
                Some(Method {
                    pattern_suffix: "(v)".to_owned(),
                    field_type,
                    bound_name: "v".to_owned(),
                    variant_name,
                    fn_name,
                })
            }
            ast::StructKind::Unit => None,
        }
    }
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
    fn test_generate_enum_multiple_try_into_tuple_variant() {
        check_assist(
            generate_enum_try_into_method,
            r#"
enum Value {
    Unit(()),
    $0Number(i32),
    Text(String)$0,
}"#,
            r#"enum Value {
    Unit(()),
    Number(i32),
    Text(String),
}

impl Value {
    fn try_into_number(self) -> Result<i32, Self> {
        if let Self::Number(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

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
    fn test_generate_enum_as_multiple_tuple_variant() {
        check_assist(
            generate_enum_as_method,
            r#"
enum Value {
    Unit(()),
    $0Number(i32),
    Text(String)$0,
}"#,
            r#"enum Value {
    Unit(()),
    Number(i32),
    Text(String),
}

impl Value {
    fn as_number(&self) -> Option<&i32> {
        if let Self::Number(v) = self {
            Some(v)
        } else {
            None
        }
    }

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
