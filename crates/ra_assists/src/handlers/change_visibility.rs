use ra_syntax::{
    ast::{self, NameOwner, VisibilityOwner},
    AstNode,
    SyntaxKind::{
        ATTR, COMMENT, CONST_DEF, ENUM_DEF, FN_DEF, MODULE, STRUCT_DEF, TRAIT_DEF, VISIBILITY,
        WHITESPACE,
    },
    SyntaxNode, TextSize, T,
};

use crate::{Assist, AssistCtx, AssistId};
use test_utils::tested_by;

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
pub(crate) fn change_visibility(ctx: AssistCtx) -> Option<Assist> {
    if let Some(vis) = ctx.find_node_at_offset::<ast::Visibility>() {
        return change_vis(ctx, vis);
    }
    add_vis(ctx)
}

fn add_vis(ctx: AssistCtx) -> Option<Assist> {
    let item_keyword = ctx.token_at_offset().find(|leaf| match leaf.kind() {
        T![const] | T![fn] | T![mod] | T![struct] | T![enum] | T![trait] => true,
        _ => false,
    });

    let (offset, target) = if let Some(keyword) = item_keyword {
        let parent = keyword.parent();
        let def_kws = vec![CONST_DEF, FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF];
        // Parent is not a definition, can't add visibility
        if !def_kws.iter().any(|&def_kw| def_kw == parent.kind()) {
            return None;
        }
        // Already have visibility, do nothing
        if parent.children().any(|child| child.kind() == VISIBILITY) {
            return None;
        }
        (vis_offset(&parent), keyword.text_range())
    } else if let Some(field_name) = ctx.find_node_at_offset::<ast::Name>() {
        let field = field_name.syntax().ancestors().find_map(ast::RecordFieldDef::cast)?;
        if field.name()? != field_name {
            tested_by!(change_visibility_field_false_positive);
            return None;
        }
        if field.visibility().is_some() {
            return None;
        }
        (vis_offset(field.syntax()), field_name.syntax().text_range())
    } else if let Some(field) = ctx.find_node_at_offset::<ast::TupleFieldDef>() {
        if field.visibility().is_some() {
            return None;
        }
        (vis_offset(field.syntax()), field.syntax().text_range())
    } else {
        return None;
    };

    ctx.add_assist(AssistId("change_visibility"), "Change visibility to pub(crate)", |edit| {
        edit.target(target);
        edit.insert(offset, "pub(crate) ");
        edit.set_cursor(offset);
    })
}

fn vis_offset(node: &SyntaxNode) -> TextSize {
    node.children_with_tokens()
        .skip_while(|it| match it.kind() {
            WHITESPACE | COMMENT | ATTR => true,
            _ => false,
        })
        .next()
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

fn change_vis(ctx: AssistCtx, vis: ast::Visibility) -> Option<Assist> {
    if vis.syntax().text() == "pub" {
        return ctx.add_assist(
            AssistId("change_visibility"),
            "Change Visibility to pub(crate)",
            |edit| {
                edit.target(vis.syntax().text_range());
                edit.replace(vis.syntax().text_range(), "pub(crate)");
                edit.set_cursor(vis.syntax().text_range().start())
            },
        );
    }
    if vis.syntax().text() == "pub(crate)" {
        return ctx.add_assist(AssistId("change_visibility"), "Change visibility to pub", |edit| {
            edit.target(vis.syntax().text_range());
            edit.replace(vis.syntax().text_range(), "pub");
            edit.set_cursor(vis.syntax().text_range().start());
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

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
            r"struct S { <|>field: u32 }",
            r"struct S { <|>pub(crate) field: u32 }",
        );
        check_assist(change_visibility, r"struct S ( <|>u32 )", r"struct S ( <|>pub(crate) u32 )");
    }

    #[test]
    fn change_visibility_field_false_positive() {
        covers!(change_visibility_field_false_positive);
        check_assist_not_applicable(
            change_visibility,
            r"struct S { field: [(); { let <|>x = ();}] }",
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
    fn change_visibility_const() {
        check_assist(change_visibility, "<|>const FOO = 3u8;", "<|>pub(crate) const FOO = 3u8;");
    }

    #[test]
    fn change_visibility_handles_comment_attrs() {
        check_assist(
            change_visibility,
            r"
            /// docs

            // comments

            #[derive(Debug)]
            <|>struct Foo;
            ",
            r"
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
