use ide_db::{RootDatabase, famous_defs::FamousDefs};
use syntax::{
    ast::{self, AstNode, HasName, edit::AstNodeEdit, syntax_factory::SyntaxFactory},
    syntax_editor::{Position, SyntaxEditor},
};

use crate::{AssistContext, AssistId, Assists, utils};

// Assist: generate_default_from_enum_variant
//
// Adds a Default impl for an enum using a variant.
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
// impl Default for Version {
//     fn default() -> Self {
//         Self::Minor
//     }
// }
// ```
pub(crate) fn generate_default_from_enum_variant(
    acc: &mut Assists,
    ctx: &AssistContext<'_, '_>,
) -> Option<()> {
    let variant = ctx.find_node_at_offset::<ast::Variant>()?;
    let variant_name = variant.name()?;
    let adt = ast::Adt::Enum(variant.parent_enum());
    if !matches!(variant.kind(), ast::StructKind::Unit) {
        cov_mark::hit!(test_gen_default_on_non_unit_variant_not_implemented);
        return None;
    }
    if !variant.syntax().text_range().contains_range(ctx.selection_trimmed()) {
        return None;
    }

    if existing_default_impl(&ctx.sema, &variant).is_some() {
        cov_mark::hit!(test_gen_default_impl_already_exists);
        return None;
    }

    let target = variant.syntax().text_range();
    acc.add(
        AssistId::generate("generate_default_from_enum_variant"),
        "Generate `Default` impl from this enum variant",
        target,
        |edit| {
            let editor = edit.make_editor(adt.syntax());
            let make = editor.make();
            let indent = adt.indent_level();
            let impl_ = default_impl(variant_name, &adt, make);

            editor.insert_all(
                Position::after(adt.syntax()),
                vec![
                    make.whitespace(&format!("\n\n{indent}")).into(),
                    impl_.indent(indent).syntax().clone().into(),
                ],
            );
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn default_impl(variant_name: ast::Name, adt: &ast::Adt, make: &SyntaxFactory) -> ast::Impl {
    let impl_ = utils::generate_trait_impl_intransitive(make, adt, make.ty("Default"));
    let fn_ = default_fn(&variant_name.text(), make);
    let (impl_editor, impl_) = SyntaxEditor::with_ast_node(&impl_);

    impl_
        .get_or_create_assoc_item_list_with_editor(&impl_editor)
        .add_items(&impl_editor, vec![fn_.into()]);
    ast::Impl::cast(impl_editor.finish().new_root().clone()).unwrap()
}

fn default_fn(variant_name: &str, make: &SyntaxFactory) -> ast::Fn {
    let path = make.path_from_idents(["Self", variant_name]).unwrap();
    let body = make.block_expr(None, Some(make.expr_path(path)));
    make.fn_(
        None,
        None,
        make.name("default"),
        None,
        None,
        make.param_list(None, None),
        body,
        Some(make.ret_type(make.ty("Self"))),
        false,
        false,
        false,
        false,
    )
    .indent(1.into())
}

fn existing_default_impl(
    sema: &'_ hir::Semantics<'_, RootDatabase>,
    variant: &ast::Variant,
) -> Option<()> {
    let variant = sema.to_def(variant)?;
    let enum_ = variant.parent_enum(sema.db);
    let krate = enum_.module(sema.db).krate(sema.db);

    let default_trait = FamousDefs(sema, krate).core_default_Default()?;
    let enum_type = enum_.ty(sema.db);

    if enum_type.impls_trait(sema.db, default_trait, &[]) { Some(()) } else { None }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_default_from_variant() {
        check_assist(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant {
    Undefined,
    Minor$0,
    Major,
}
"#,
            r#"
enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Default for Variant {
    fn default() -> Self {
        Self::Minor
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_default_selected_variant() {
        check_assist(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant {
    Undefined,
    $0Minor$0,
    Major,
}
"#,
            r#"
enum Variant {
    Undefined,
    Minor,
    Major,
}

impl Default for Variant {
    fn default() -> Self {
        Self::Minor
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_default_not_applicable_with_multiple_variant_selection() {
        check_assist_not_applicable(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant {
    Undefined,
    $0Minor,
    M$0ajor,
}
"#,
        );
    }

    #[test]
    fn test_generate_default_already_implemented() {
        cov_mark::check!(test_gen_default_impl_already_exists);
        check_assist_not_applicable(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant {
    Undefined,
    Minor$0,
    Major,
}

impl Default for Variant {
    fn default() -> Self {
        Self::Minor
    }
}
"#,
        );
    }

    #[test]
    fn test_add_from_impl_no_element() {
        cov_mark::check!(test_gen_default_on_non_unit_variant_not_implemented);
        check_assist_not_applicable(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant {
    Undefined,
    Minor(u32)$0,
    Major,
}
"#,
        );
    }

    #[test]
    fn test_generate_default_from_variant_with_one_variant() {
        check_assist(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant { Undefi$0ned }
"#,
            r#"
enum Variant { Undefined }

impl Default for Variant {
    fn default() -> Self {
        Self::Undefined
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_default_from_variant_with_generics() {
        check_assist(
            generate_default_from_enum_variant,
            r#"
//- minicore: default
enum Variant<T> {
    Defined(T),
    Undefi$0ned,
}
"#,
            r#"
enum Variant<T> {
    Defined(T),
    Undefined,
}

impl<T> Default for Variant<T> {
    fn default() -> Self {
        Self::Undefined
    }
}
"#,
        );
    }
}
