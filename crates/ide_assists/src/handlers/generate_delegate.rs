use hir::{self, HasCrate, HirDisplay};
use stdx::format_to;
use syntax::ast::{self, AstNode, HasName, HasVisibility};

use crate::{
    utils::{find_impl_block_end, find_struct_impl, generate_impl_text},
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
                let mut buf = String::with_capacity(512);

                let vis = strukt.visibility().map_or(String::new(), |v| format!("{} ", v));
                let return_type = method.ret_type(ctx.db());
                let return_type = if return_type.is_unit() || return_type.is_unknown() {
                    String::new()
                } else {
                    let module = match ctx.sema.scope(strukt.syntax()).module() {
                        Some(m) => m,
                        None => return,
                    };
                    match return_type.display_source_code(ctx.db(), module.into()) {
                        Ok(rt) => format!("-> {}", rt),
                        Err(_) => return,
                    }
                };

                format_to!(
                    buf,
                    "{}fn {}(&self) {} {{
                    self.{}.{}()
                }}",
                    vis,
                    method.name(ctx.db()),
                    return_type,
                    field_name,
                    method.name(ctx.db())
                );

                let start_offset = impl_def
                    .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
                    .unwrap_or_else(|| {
                        buf = generate_impl_text(&ast::Adt::Struct(strukt.clone()), &buf);
                        strukt.syntax().text_range().end()
                    });

                builder.insert(start_offset, buf);
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
"#,
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
    fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }
}
