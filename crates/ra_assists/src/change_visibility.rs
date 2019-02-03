use hir::db::HirDatabase;
use ra_syntax::{
    AstNode, SyntaxNode, TextUnit,
    ast::{self, VisibilityOwner, NameOwner},
    SyntaxKind::{VISIBILITY, FN_KW, MOD_KW, STRUCT_KW, ENUM_KW, TRAIT_KW, FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF, IDENT, WHITESPACE, COMMENT, ATTR},
};

use crate::{AssistCtx, Assist};

pub(crate) fn change_visibility(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    if let Some(vis) = ctx.node_at_offset::<ast::Visibility>() {
        return change_vis(ctx, vis);
    }
    add_vis(ctx)
}

fn add_vis(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let item_keyword = ctx.leaf_at_offset().find(|leaf| match leaf.kind() {
        FN_KW | MOD_KW | STRUCT_KW | ENUM_KW | TRAIT_KW => true,
        _ => false,
    });

    let offset = if let Some(keyword) = item_keyword {
        let parent = keyword.parent()?;
        let def_kws = vec![FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF];
        // Parent is not a definition, can't add visibility
        if !def_kws.iter().any(|&def_kw| def_kw == parent.kind()) {
            return None;
        }
        // Already have visibility, do nothing
        if parent.children().any(|child| child.kind() == VISIBILITY) {
            return None;
        }
        vis_offset(parent)
    } else {
        let ident = ctx.leaf_at_offset().find(|leaf| leaf.kind() == IDENT)?;
        let field = ident.ancestors().find_map(ast::NamedFieldDef::cast)?;
        if field.name()?.syntax().range() != ident.range() && field.visibility().is_some() {
            return None;
        }
        vis_offset(field.syntax())
    };

    ctx.build("make pub(crate)", |edit| {
        edit.insert(offset, "pub(crate) ");
        edit.set_cursor(offset);
    })
}

fn vis_offset(node: &SyntaxNode) -> TextUnit {
    node.children()
        .skip_while(|it| match it.kind() {
            WHITESPACE | COMMENT | ATTR => true,
            _ => false,
        })
        .next()
        .map(|it| it.range().start())
        .unwrap_or(node.range().start())
}

fn change_vis(ctx: AssistCtx<impl HirDatabase>, vis: &ast::Visibility) -> Option<Assist> {
    if vis.syntax().text() == "pub" {
        return ctx.build("chage to pub(crate)", |edit| {
            edit.replace(vis.syntax().range(), "pub(crate)");
            edit.set_cursor(vis.syntax().range().start());
        });
    }
    if vis.syntax().text() == "pub(crate)" {
        return ctx.build("chage to pub", |edit| {
            edit.replace(vis.syntax().range(), "pub");
            edit.set_cursor(vis.syntax().range().start());
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::check_assist;

    #[test]
    fn change_visibility_adds_pub_crate_to_items() {
        check_assist(
            change_visibility,
            "<|>fn foo() {}",
            "<|>pub(crate) fn foo() {}",
        );
        check_assist(
            change_visibility,
            "f<|>n foo() {}",
            "<|>pub(crate) fn foo() {}",
        );
        check_assist(
            change_visibility,
            "<|>struct Foo {}",
            "<|>pub(crate) struct Foo {}",
        );
        check_assist(
            change_visibility,
            "<|>mod foo {}",
            "<|>pub(crate) mod foo {}",
        );
        check_assist(
            change_visibility,
            "<|>trait Foo {}",
            "<|>pub(crate) trait Foo {}",
        );
        check_assist(change_visibility, "m<|>od {}", "<|>pub(crate) mod {}");
        check_assist(
            change_visibility,
            "unsafe f<|>n foo() {}",
            "<|>pub(crate) unsafe fn foo() {}",
        );
    }

    #[test]
    fn change_visibility_works_with_struct_fields() {
        check_assist(
            change_visibility,
            "struct S { <|>field: u32 }",
            "struct S { <|>pub(crate) field: u32 }",
        )
    }

    #[test]
    fn change_visibility_pub_to_pub_crate() {
        check_assist(
            change_visibility,
            "<|>pub fn foo() {}",
            "<|>pub(crate) fn foo() {}",
        )
    }

    #[test]
    fn change_visibility_pub_crate_to_pub() {
        check_assist(
            change_visibility,
            "<|>pub(crate) fn foo() {}",
            "<|>pub fn foo() {}",
        )
    }

    #[test]
    fn change_visibility_handles_comment_attrs() {
        check_assist(
            change_visibility,
            "
            /// docs

            // comments

            #[derive(Debug)]
            <|>struct Foo;
            ",
            "
            /// docs

            // comments

            #[derive(Debug)]
            <|>pub(crate) struct Foo;
            ",
        )
    }
}
