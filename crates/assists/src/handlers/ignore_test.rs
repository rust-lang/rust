use syntax::{ast, AstNode};

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

    acc.add(
        AssistId("ignore_test", AssistKind::None),
        "Ignore this test",
        attr.syntax().text_range(),
        |builder| builder.insert(attr.syntax().text_range().end(), &format!("\n#[ignore]")),
    )
}
