use syntax::{
    ast::{self, AttrsOwner},
    AstNode, AstToken,
};

use crate::{utils::test_related_attribute, AssistContext, AssistId, AssistKind, Assists};

// Assist: ignore_test
//
// Adds `#[ignore]` attribute to the test.
//
// ```
// <|>#[test]
// fn arithmetics {
//     assert_eq!(2 + 2, 5);
// }
// ```
// ->
// ```
// #[test]
// #[ignore]
// fn arithmetics {
//     assert_eq!(2 + 2, 5);
// }
// ```
pub(crate) fn ignore_test(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let attr: ast::Attr = ctx.find_node_at_offset()?;
    let func = attr.syntax().parent().and_then(ast::Fn::cast)?;
    let attr = test_related_attribute(&func)?;

    match has_ignore_attribute(&func) {
        None => acc.add(
            AssistId("ignore_test", AssistKind::None),
            "Ignore this test",
            attr.syntax().text_range(),
            |builder| builder.insert(attr.syntax().text_range().end(), &format!("\n#[ignore]")),
        ),
        Some(ignore_attr) => acc.add(
            AssistId("unignore_test", AssistKind::None),
            "Re-enable this test",
            ignore_attr.syntax().text_range(),
            |builder| {
                builder.delete(ignore_attr.syntax().text_range());
                let whitespace = ignore_attr
                    .syntax()
                    .next_sibling_or_token()
                    .and_then(|x| x.into_token())
                    .and_then(ast::Whitespace::cast);
                if let Some(whitespace) = whitespace {
                    builder.delete(whitespace.syntax().text_range());
                }
            },
        ),
    }
}

fn has_ignore_attribute(fn_def: &ast::Fn) -> Option<ast::Attr> {
    fn_def.attrs().find_map(|attr| {
        if attr.path()?.syntax().text() == "ignore" {
            Some(attr)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::ignore_test;
    use crate::tests::check_assist;

    #[test]
    fn test_base_case() {
        check_assist(
            ignore_test,
            r#"
            #[test<|>]
            fn test() {}
            "#,
            r#"
            #[test]
            #[ignore]
            fn test() {}
            "#,
        )
    }

    #[test]
    fn test_unignore() {
        check_assist(
            ignore_test,
            r#"
            #[test<|>]
            #[ignore]
            fn test() {}
            "#,
            r#"
            #[test]
            fn test() {}
            "#,
        )
    }
}
