use syntax::{
    AstNode, AstToken,
    ast::{self, HasAttrs},
};

use crate::{AssistContext, AssistId, Assists, utils::test_related_attribute_syn};

// Assist: toggle_ignore
//
// Adds `#[ignore]` attribute to the test.
//
// ```
// $0#[test]
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
pub(crate) fn toggle_ignore(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let attr: ast::Attr = ctx.find_node_at_offset()?;
    let func = attr.syntax().parent().and_then(ast::Fn::cast)?;
    let attr = test_related_attribute_syn(&func)?;

    match has_ignore_attribute(&func) {
        None => acc.add(
            AssistId::refactor("toggle_ignore"),
            "Ignore this test",
            attr.syntax().text_range(),
            |builder| builder.insert(attr.syntax().text_range().end(), "\n#[ignore]"),
        ),
        Some(ignore_attr) => acc.add(
            AssistId::refactor("toggle_ignore"),
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
    fn_def.attrs().find(|attr| attr.path().is_some_and(|it| it.syntax().text() == "ignore"))
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn test_base_case() {
        check_assist(
            toggle_ignore,
            r#"
            #[test$0]
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
            toggle_ignore,
            r#"
            #[test$0]
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
