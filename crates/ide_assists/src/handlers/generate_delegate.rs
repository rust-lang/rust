use hir::{self, HasCrate, HirDisplay};
use syntax::ast::{self, make, AstNode, HasName, HasVisibility};

use crate::{
    utils::{find_struct_impl, render_snippet, Cursor},
    AssistContext, AssistId, AssistKind, Assists, GroupLabel,
};

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
                let vis = strukt.visibility();
                let name = make::name(&method.name(ctx.db()).to_string());
                let type_params = None;
                let params = make::param_list(None, []);
                let body = make::block_expr([], None);
                let ret_type = &method.ret_type(ctx.db()).display(ctx.db()).to_string();
                let ret_type = Some(make::ret_type(make::ty(ret_type)));
                let is_async = false;
                let f = make::fn_(vis, name, type_params, params, body, ret_type, is_async);

                let cursor = Cursor::Before(f.syntax());
                let cap = ctx.config.snippet_cap.unwrap(); // FIXME.

                match impl_def {
                    Some(impl_def) => {
                        let impl_def = impl_def.clone_for_update();
                        let old_range = impl_def.syntax().text_range();
                        let assoc_items = impl_def.get_or_create_assoc_item_list();
                        assoc_items.add_item(f.clone().into());
                        let snippet = render_snippet(cap, impl_def.syntax(), cursor);
                        builder.replace_snippet(cap, old_range, snippet);
                    }
                    None => {
                        let name = &strukt_name.to_string();
                        let impl_def = make::impl_(make::ext::ident_path(name));
                        let assoc_items = impl_def.get_or_create_assoc_item_list();
                        assoc_items.add_item(f.clone().into());
                        let start_offset = strukt.syntax().text_range().end();
                        let snippet = render_snippet(cap, impl_def.syntax(), cursor);
                        builder.insert_snippet(cap, start_offset, snippet);
                    }
                }
            },
        )?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn test_generate_setter_from_field() {
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

impl Person {}
"#,
            r#"
struct Age(u8);
impl Age {
    $0fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    age: Age,
}

impl Person {
    fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }
}
