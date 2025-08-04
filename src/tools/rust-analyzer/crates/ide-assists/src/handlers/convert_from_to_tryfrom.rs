use ide_db::{famous_defs::FamousDefs, traits::resolve_target_trait};
use syntax::ast::edit::IndentLevel;
use syntax::ast::{self, AstNode, HasGenericArgs, HasName, make};
use syntax::syntax_editor::{Element, Position};

use crate::{AssistContext, AssistId, Assists};

// Assist: convert_from_to_tryfrom
//
// Converts a From impl to a TryFrom impl, wrapping returns in `Ok`.
//
// ```
// # //- minicore: from
// impl $0From<usize> for Thing {
//     fn from(val: usize) -> Self {
//         Thing {
//             b: val.to_string(),
//             a: val
//         }
//     }
// }
// ```
// ->
// ```
// impl TryFrom<usize> for Thing {
//     type Error = ${0:()};
//
//     fn try_from(val: usize) -> Result<Self, Self::Error> {
//         Ok(Thing {
//             b: val.to_string(),
//             a: val
//         })
//     }
// }
// ```
pub(crate) fn convert_from_to_tryfrom(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let impl_ = ctx.find_node_at_offset::<ast::Impl>()?;
    let trait_ty = impl_.trait_()?;

    let module = ctx.sema.scope(impl_.syntax())?.module();

    let from_type = match &trait_ty {
        ast::Type::PathType(path) => {
            path.path()?.segment()?.generic_arg_list()?.generic_args().next()?
        }
        _ => return None,
    };

    let associated_items = impl_.assoc_item_list()?;
    let associated_l_curly = associated_items.l_curly_token()?;
    let from_fn = associated_items.assoc_items().find_map(|item| {
        if let ast::AssocItem::Fn(f) = item
            && f.name()?.text() == "from"
        {
            return Some(f);
        };
        None
    })?;

    let from_fn_name = from_fn.name()?;
    let from_fn_return_type = from_fn.ret_type()?.ty()?;

    let return_exprs = from_fn.body()?.syntax().descendants().filter_map(ast::ReturnExpr::cast);
    let tail_expr = from_fn.body()?.tail_expr()?;

    if resolve_target_trait(&ctx.sema, &impl_)?
        != FamousDefs(&ctx.sema, module.krate()).core_convert_From()?
    {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("convert_from_to_tryfrom"),
        "Convert From to TryFrom",
        impl_.syntax().text_range(),
        |builder| {
            let mut editor = builder.make_editor(impl_.syntax());
            editor.replace(
                trait_ty.syntax(),
                make::ty(&format!("TryFrom<{from_type}>")).syntax().clone_for_update(),
            );
            editor.replace(
                from_fn_return_type.syntax(),
                make::ty("Result<Self, Self::Error>").syntax().clone_for_update(),
            );
            editor
                .replace(from_fn_name.syntax(), make::name("try_from").syntax().clone_for_update());
            editor.replace(
                tail_expr.syntax(),
                wrap_ok(tail_expr.clone()).syntax().clone_for_update(),
            );

            for r in return_exprs {
                let t = r.expr().unwrap_or_else(make::ext::expr_unit);
                editor.replace(t.syntax(), wrap_ok(t.clone()).syntax().clone_for_update());
            }

            let error_type = ast::AssocItem::TypeAlias(make::ty_alias(
                "Error",
                None,
                None,
                None,
                Some((make::ty_unit(), None)),
            ))
            .clone_for_update();

            if let Some(cap) = ctx.config.snippet_cap
                && let ast::AssocItem::TypeAlias(type_alias) = &error_type
                && let Some(ty) = type_alias.ty()
            {
                let placeholder = builder.make_placeholder_snippet(cap);
                editor.add_annotation(ty.syntax(), placeholder);
            }

            let indent = IndentLevel::from_token(&associated_l_curly) + 1;
            editor.insert_all(
                Position::after(associated_l_curly),
                vec![
                    make::tokens::whitespace(&format!("\n{indent}")).syntax_element(),
                    error_type.syntax().syntax_element(),
                    make::tokens::whitespace("\n").syntax_element(),
                ],
            );
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn wrap_ok(expr: ast::Expr) -> ast::Expr {
    make::expr_call(
        make::expr_path(make::ext::ident_path("Ok")),
        make::arg_list(std::iter::once(expr)),
    )
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn converts_from_to_tryfrom() {
        check_assist(
            convert_from_to_tryfrom,
            r#"
//- minicore: from
struct Foo(String);

impl $0From<String> for Foo {
    fn from(val: String) -> Self {
        if val == "bar" {
            return Foo(val);
        }
        Self(val)
    }
}
            "#,
            r#"
struct Foo(String);

impl TryFrom<String> for Foo {
    type Error = ${0:()};

    fn try_from(val: String) -> Result<Self, Self::Error> {
        if val == "bar" {
            return Ok(Foo(val));
        }
        Ok(Self(val))
    }
}
            "#,
        );
    }

    #[test]
    fn converts_from_to_tryfrom_nested_type() {
        check_assist(
            convert_from_to_tryfrom,
            r#"
//- minicore: from
struct Foo(String);

impl $0From<Option<String>> for Foo {
    fn from(val: Option<String>) -> Self {
        match val {
            Some(val) => Foo(val),
            None => Foo("".to_string())
        }
    }
}
            "#,
            r#"
struct Foo(String);

impl TryFrom<Option<String>> for Foo {
    type Error = ${0:()};

    fn try_from(val: Option<String>) -> Result<Self, Self::Error> {
        Ok(match val {
            Some(val) => Foo(val),
            None => Foo("".to_string())
        })
    }
}
            "#,
        );
    }

    #[test]
    fn converts_from_to_tryfrom_preserves_lifetimes() {
        check_assist(
            convert_from_to_tryfrom,
            r#"
//- minicore: from
struct Foo<'a>(&'a str);

impl<'a> $0From<&'a str> for Foo<'a> {
    fn from(val: &'a str) -> Self {
        Self(val)
    }
}
            "#,
            r#"
struct Foo<'a>(&'a str);

impl<'a> TryFrom<&'a str> for Foo<'a> {
    type Error = ${0:()};

    fn try_from(val: &'a str) -> Result<Self, Self::Error> {
        Ok(Self(val))
    }
}
            "#,
        );
    }

    #[test]
    fn other_trait_not_applicable() {
        check_assist_not_applicable(
            convert_from_to_tryfrom,
            r#"
struct Foo(String);

impl $0TryFrom<String> for Foo {
    fn try_from(val: String) -> Result<Self, Self::Error> {
        Ok(Self(val))
    }
}
            "#,
        );
    }
}
