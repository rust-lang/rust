use hir::{self, HasCrate, HasSource, HirDisplay};
use syntax::ast::{self, make, AstNode, HasName, HasVisibility};

use crate::{
    utils::{find_struct_impl, render_snippet, Cursor},
    AssistContext, AssistId, AssistKind, Assists, GroupLabel,
};
use syntax::ast::edit::AstNodeEdit;

// Assist: generate_setter
//
// Generate a setter method.
//
// ```
// struct Person {
//     nam$0e: String,
// }
// ```
// ->
// ```
// struct Person {
//     name: String,
// }
//
// impl Person {
//     /// Set the person's name.
//     fn set_name(&mut self, name: String) {
//         self.name = name;
//     }
// }
// ```
pub(crate) fn generate_delegate(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let strukt_name = strukt.name()?;

    let field = ctx.find_node_at_offset::<ast::RecordField>()?;
    let field_name = field.name()?;
    let field_ty = field.ty()?;

    let sema_field_ty = ctx.sema.resolve_type(&field_ty)?;
    let krate = sema_field_ty.krate(ctx.db());
    let mut methods = vec![];
    sema_field_ty.iterate_assoc_items(ctx.db(), krate, |item| {
        if let hir::AssocItem::Function(f) = item {
            if f.self_param(ctx.db()).is_some() {
                methods.push(f)
            }
        }
        Some(())
    });

    let target = field_ty.syntax().text_range();
    for method in methods {
        let impl_def = find_struct_impl(
            ctx,
            &ast::Adt::Struct(strukt.clone()),
            &method.name(ctx.db()).to_string(),
        )?;
        acc.add_group(
            &GroupLabel("Generate delegate".to_owned()),
            AssistId("generate_delegate", AssistKind::Generate),
            format!("Generate a delegate method for '{}'", method.name(ctx.db())),
            target,
            |builder| {
                // make function
                let method_source = match method.source(ctx.db()) {
                    Some(source) => source.value,
                    None => return,
                };
                let method_name = method.name(ctx.db());
                let vis = method_source.visibility();
                let name = make::name(&method.name(ctx.db()).to_string());
                let type_params = None;
                let self_ty = method
                    .self_param(ctx.db())
                    .map(|s| s.source(ctx.db()).map(|s| s.value))
                    .flatten();
                let params = make::param_list(self_ty, []);
                let tail_expr = make::expr_method_call(
                    field_from_idents(["self", &field_name.to_string()]).unwrap(),
                    make::name_ref(&method_name.to_string()),
                    make::arg_list([]),
                );
                let body = make::block_expr([], Some(tail_expr));
                let ret_type = &method.ret_type(ctx.db()).display(ctx.db()).to_string();
                let ret_type = Some(make::ret_type(make::ty(ret_type)));
                let is_async = false;
                let f = make::fn_(vis, name, type_params, params, body, ret_type, is_async)
                    .indent(ast::edit::IndentLevel(1))
                    .clone_for_update();

                let cursor = Cursor::Before(f.syntax());
                let cap = ctx.config.snippet_cap.unwrap(); // FIXME.

                // Create or update an impl block, and attach the function to it.
                match impl_def {
                    Some(impl_def) => {
                        // Remember where in our source our `impl` block lives.
                        let impl_def = impl_def.clone_for_update();
                        let old_range = impl_def.syntax().text_range();

                        // Attach the function to the impl block
                        let assoc_items = impl_def.get_or_create_assoc_item_list();
                        assoc_items.add_item(f.clone().into());

                        // Update the impl block.
                        let snippet = render_snippet(cap, impl_def.syntax(), cursor);
                        builder.replace_snippet(cap, old_range, snippet);
                    }
                    None => {
                        // Attach the function to the impl block
                        let name = &strukt_name.to_string();
                        let impl_def = make::impl_(make::ext::ident_path(name)).clone_for_update();
                        let assoc_items = impl_def.get_or_create_assoc_item_list();
                        assoc_items.add_item(f.clone().into());

                        // Insert the impl block.
                        let offset = strukt.syntax().text_range().end();
                        let snippet = render_snippet(cap, impl_def.syntax(), cursor);
                        let snippet = format!("\n\n{}", snippet);
                        builder.insert_snippet(cap, offset, snippet);
                    }
                }
            },
        )?;
    }
    Some(())
}

pub fn field_from_idents<'a>(
    parts: impl std::iter::IntoIterator<Item = &'a str>,
) -> Option<ast::Expr> {
    let mut iter = parts.into_iter();
    let base = make::expr_path(make::ext::ident_path(iter.next()?));
    let expr = iter.fold(base, |base, s| make::expr_field(base, s));
    Some(expr)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn test_generate_delegate_create_impl_block() {
        check_assist(
            generate_delegate,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    ag$0e: Age,
}"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    age: Age,
}

impl Person {
    $0fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }

    #[test]
    fn test_generate_delegate_update_impl_block() {
        check_assist(
            generate_delegate,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    ag$0e: Age,
}

impl Person {}"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    age: Age,
}

impl Person {
    $0fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }
        self.age.age()
    }
}"#,
        );
    }
}
