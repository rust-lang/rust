use syntax::{
    ast::{self, HasName, HasVisibility},
    AstNode,
    SyntaxKind::{
        CONST, ENUM, FN, MACRO_DEF, MODULE, STATIC, STRUCT, TRAIT, TYPE_ALIAS, USE, VISIBILITY,
    },
    T,
};

use crate::{utils::vis_offset, AssistContext, AssistId, AssistKind, Assists};

// Assist: change_visibility
//
// Adds or changes existing visibility specifier.
//
// ```
// $0fn frobnicate() {}
// ```
// ->
// ```
// pub(crate) fn frobnicate() {}
// ```
pub(crate) fn change_visibility(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if let Some(vis) = ctx.find_node_at_offset::<ast::Visibility>() {
        return change_vis(acc, vis);
    }
    add_vis(acc, ctx)
}

fn add_vis(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let item_keyword = ctx.token_at_offset().find(|leaf| {
        matches!(
            leaf.kind(),
            T![const]
                | T![static]
                | T![fn]
                | T![mod]
                | T![struct]
                | T![enum]
                | T![trait]
                | T![type]
                | T![use]
                | T![macro]
        )
    });

    let (offset, target) = if let Some(keyword) = item_keyword {
        let parent = keyword.parent()?;
        let def_kws =
            vec![CONST, STATIC, TYPE_ALIAS, FN, MODULE, STRUCT, ENUM, TRAIT, USE, MACRO_DEF];
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
        let field = field_name.syntax().ancestors().find_map(ast::RecordField::cast)?;
        if field.name()? != field_name {
            cov_mark::hit!(change_visibility_field_false_positive);
            return None;
        }
        if field.visibility().is_some() {
            return None;
        }
        (vis_offset(field.syntax()), field_name.syntax().text_range())
    } else if let Some(field) = ctx.find_node_at_offset::<ast::TupleField>() {
        if field.visibility().is_some() {
            return None;
        }
        (vis_offset(field.syntax()), field.syntax().text_range())
    } else {
        return None;
    };

    acc.add(
        AssistId("change_visibility", AssistKind::RefactorRewrite),
        "Change visibility to pub(crate)",
        target,
        |edit| {
            edit.insert(offset, "pub(crate) ");
        },
    )
}

fn change_vis(acc: &mut Assists, vis: ast::Visibility) -> Option<()> {
    if vis.syntax().text() == "pub" {
        let target = vis.syntax().text_range();
        return acc.add(
            AssistId("change_visibility", AssistKind::RefactorRewrite),
            "Change Visibility to pub(crate)",
            target,
            |edit| {
                edit.replace(vis.syntax().text_range(), "pub(crate)");
            },
        );
    }
    if vis.syntax().text() == "pub(crate)" {
        let target = vis.syntax().text_range();
        return acc.add(
            AssistId("change_visibility", AssistKind::RefactorRewrite),
            "Change visibility to pub",
            target,
            |edit| {
                edit.replace(vis.syntax().text_range(), "pub");
            },
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn change_visibility_adds_pub_crate_to_items() {
        check_assist(change_visibility, "$0fn foo() {}", "pub(crate) fn foo() {}");
        check_assist(change_visibility, "f$0n foo() {}", "pub(crate) fn foo() {}");
        check_assist(change_visibility, "$0struct Foo {}", "pub(crate) struct Foo {}");
        check_assist(change_visibility, "$0mod foo {}", "pub(crate) mod foo {}");
        check_assist(change_visibility, "$0trait Foo {}", "pub(crate) trait Foo {}");
        check_assist(change_visibility, "m$0od {}", "pub(crate) mod {}");
        check_assist(change_visibility, "unsafe f$0n foo() {}", "pub(crate) unsafe fn foo() {}");
        check_assist(change_visibility, "$0macro foo() {}", "pub(crate) macro foo() {}");
        check_assist(change_visibility, "$0use foo;", "pub(crate) use foo;");
    }

    #[test]
    fn change_visibility_works_with_struct_fields() {
        check_assist(
            change_visibility,
            r"struct S { $0field: u32 }",
            r"struct S { pub(crate) field: u32 }",
        );
        check_assist(change_visibility, r"struct S ( $0u32 )", r"struct S ( pub(crate) u32 )");
    }

    #[test]
    fn change_visibility_field_false_positive() {
        cov_mark::check!(change_visibility_field_false_positive);
        check_assist_not_applicable(
            change_visibility,
            r"struct S { field: [(); { let $0x = ();}] }",
        )
    }

    #[test]
    fn change_visibility_pub_to_pub_crate() {
        check_assist(change_visibility, "$0pub fn foo() {}", "pub(crate) fn foo() {}")
    }

    #[test]
    fn change_visibility_pub_crate_to_pub() {
        check_assist(change_visibility, "$0pub(crate) fn foo() {}", "pub fn foo() {}")
    }

    #[test]
    fn change_visibility_const() {
        check_assist(change_visibility, "$0const FOO = 3u8;", "pub(crate) const FOO = 3u8;");
    }

    #[test]
    fn change_visibility_static() {
        check_assist(change_visibility, "$0static FOO = 3u8;", "pub(crate) static FOO = 3u8;");
    }

    #[test]
    fn change_visibility_type_alias() {
        check_assist(change_visibility, "$0type T = ();", "pub(crate) type T = ();");
    }

    #[test]
    fn change_visibility_handles_comment_attrs() {
        check_assist(
            change_visibility,
            r"
            /// docs

            // comments

            #[derive(Debug)]
            $0struct Foo;
            ",
            r"
            /// docs

            // comments

            #[derive(Debug)]
            pub(crate) struct Foo;
            ",
        )
    }

    #[test]
    fn not_applicable_for_enum_variants() {
        check_assist_not_applicable(
            change_visibility,
            r"mod foo { pub enum Foo {Foo1} }
              fn main() { foo::Foo::Foo1$0 } ",
        );
    }

    #[test]
    fn change_visibility_target() {
        check_assist_target(change_visibility, "$0fn foo() {}", "fn");
        check_assist_target(change_visibility, "pub(crate)$0 fn foo() {}", "pub(crate)");
        check_assist_target(change_visibility, "struct S { $0field: u32 }", "field");
    }
}
