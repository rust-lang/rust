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
    let parent_enum = ast::Adt::Enum(variant.parent_enum());
    let variants = variant
        .parent_enum()
        .variant_list()?
        .variants()
        .filter(|it| is_selected(it, ctx.selection_trimmed(), true))
        .collect::<Vec<_>>();
    let methods = variants.iter().map(Method::new).collect::<Option<Vec<_>>>()?;
    let enum_name = parent_enum.name()?;
    let enum_lowercase_name = to_lower_snake_case(&enum_name.to_string()).replace('_', " ");
    let fn_names = methods.iter().map(|it| it.fn_name.clone()).collect::<Vec<_>>();
    stdx::never!(variants.is_empty());

    // Return early if we've found an existing new fn
    let impl_def = find_struct_impl(ctx, &parent_enum, &fn_names)?;

    let target = variant.syntax().text_range();
    acc.add_group(
        &GroupLabel("Generate an `is_`,`as_`, or `try_into_` for this enum variant".to_owned()),
        AssistId::generate("generate_enum_is_method"),
        "Generate an `is_` method for this enum variant",
        target,
        |builder| {
            let vis = parent_enum.visibility().map_or(String::new(), |v| format!("{v} "));

            let fn_items: Vec<ast::AssocItem> = methods
                .iter()
                .map(|method| build_fn_item(method, &enum_lowercase_name, &enum_name, &vis))
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
    enum_lowercase_name: &str,
    enum_name: &ast::Name,
    vis: &str,
) -> ast::AssocItem {
    let Method { pattern_suffix, fn_name, variant_name } = method;
    let fn_text = format!(
        "/// Returns `true` if the {enum_lowercase_name} is [`{variant_name}`].
///
/// [`{variant_name}`]: {enum_name}::{variant_name}
#[must_use]
{vis}fn {fn_name}(&self) -> bool {{
    matches!(self, Self::{variant_name}{pattern_suffix})
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
    pattern_suffix: &'static str,
    fn_name: String,
    variant_name: ast::Name,
}

impl Method {
    fn new(variant: &ast::Variant) -> Option<Self> {
        let pattern_suffix = match variant.kind() {
            ast::StructKind::Record(_) => " { .. }",
            ast::StructKind::Tuple(_) => "(..)",
            ast::StructKind::Unit => "",
        };

        let variant_name = variant.name()?;
        let fn_name = format!("is_{}", &to_lower_snake_case(&variant_name.text()));
        Some(Method { pattern_suffix, fn_name, variant_name })
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
    fn test_generate_enum_is_from_multiple_variant() {
        check_assist(
            generate_enum_is_method,
            r#"
enum Variant {
    Undefined,
    $0Minor,
    M$0ajor,
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
