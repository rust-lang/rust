use hir::db::HirDatabase;
use ra_syntax::{
    ast::{self, NameOwner, VisibilityOwner},
    AstNode,
    SyntaxKind::{
        ATTR, COMMENT, ENUM_DEF, FN_DEF, IDENT, MODULE, STRUCT_DEF, TRAIT_DEF, VISIBILITY,
        WHITESPACE,
    },
    SyntaxNode, TextUnit, T,
};

use crate::{Assist, AssistCtx, AssistId};

// Assist: change_visibility
//
// Adds or changes existing visibility specifier.
//
// ```
// <|>fn frobnicate() {}
// ```
// ->
// ```
// pub(crate) fn frobnicate() {}
// ```
pub(crate) fn change_visibility(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    if let Some(vis) = ctx.find_node_at_offset::<ast::Visibility>() {
        return change_vis(ctx, vis);
    }
    add_vis(ctx)
}

fn add_vis(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let item_keyword = ctx.token_at_offset().find(|leaf| match leaf.kind() {
        T![fn] | T![mod] | T![struct] | T![enum] | T![trait] => true,
        _ => false,
    });

    let (offset, target) = if let Some(keyword) = item_keyword {
        let parent = keyword.parent();
        let def_kws = vec![FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF];
        // Parent is not a definition, can't add visibility
        if !def_kws.iter().any(|&def_kw| def_kw == parent.kind()) {
            return None;
        }
        // Already have visibility, do nothing
        if parent.children().any(|child| child.kind() == VISIBILITY) {
            return None;
        }
        (vis_offset(&parent), keyword.text_range())
    } else {
        let ident = ctx.token_at_offset().find(|leaf| leaf.kind() == IDENT)?;
        let field = ident.parent().ancestors().find_map(ast::RecordFieldDef::cast)?;
        if field.name()?.syntax().text_range() != ident.text_range() && field.visibility().is_some()
        {
            return None;
        }
        (vis_offset(field.syntax()), ident.text_range())
    };

    ctx.add_action(AssistId("change_visibility"), "make pub(crate)", |edit| {
        edit.target(target);
        edit.insert(offset, "pub(crate) ");
        edit.set_cursor(offset);
    });

    ctx.build()
}

fn vis_offset(node: &SyntaxNode) -> TextUnit {
    node.children_with_tokens()
        .skip_while(|it| match it.kind() {
            WHITESPACE | COMMENT | ATTR => true,
            _ => false,
        })
        .next()
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

fn change_vis(mut ctx: AssistCtx<impl HirDatabase>, vis: ast::Visibility) -> Option<Assist> {
    if vis.syntax().text() == "pub" {
        ctx.add_action(AssistId("change_visibility"), "change to pub(crate)", |edit| {
            edit.target(vis.syntax().text_range());
            edit.replace(vis.syntax().text_range(), "pub(crate)");
            edit.set_cursor(vis.syntax().text_range().start())
        });

        return ctx.build();
    }
    if vis.syntax().text() == "pub(crate)" {
        ctx.add_action(AssistId("change_visibility"), "change to pub", |edit| {
            edit.target(vis.syntax().text_range());
            edit.replace(vis.syntax().text_range(), "pub");
            edit.set_cursor(vis.syntax().text_range().start());
        });

        return ctx.build();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn change_visibility_adds_pub_crate_to_items() {
        check_assist(change_visibility, "<|>fn foo() {}", "<|>pub(crate) fn foo() {}");
        check_assist(change_visibility, "f<|>n foo() {}", "<|>pub(crate) fn foo() {}");
        check_assist(change_visibility, "<|>struct Foo {}", "<|>pub(crate) struct Foo {}");
        check_assist(change_visibility, "<|>mod foo {}", "<|>pub(crate) mod foo {}");
        check_assist(change_visibility, "<|>trait Foo {}", "<|>pub(crate) trait Foo {}");
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
        check_assist(change_visibility, "<|>pub fn foo() {}", "<|>pub(crate) fn foo() {}")
    }

    #[test]
    fn change_visibility_pub_crate_to_pub() {
        check_assist(change_visibility, "<|>pub(crate) fn foo() {}", "<|>pub fn foo() {}")
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

    #[test]
    fn change_visibility_target() {
        check_assist_target(change_visibility, "<|>fn foo() {}", "fn");
        check_assist_target(change_visibility, "pub(crate)<|> fn foo() {}", "pub(crate)");
        check_assist_target(change_visibility, "struct S { <|>field: u32 }", "field");
    }
}
