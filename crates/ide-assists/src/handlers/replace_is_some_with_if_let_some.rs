use syntax::ast::{self, AstNode};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_is_some_with_if_let_some
//
// Replace `if x.is_some()` with `if let Some(_tmp) = x`.
//
// ```
// fn main() {
//     let x = Some(1);
//     if x.is_som$0e() {}
// }
// ```
// ->
// ```
// fn main() {
//     let x = Some(1);
//     if let Some(_tmp) = x {}
// }
// ```
pub(crate) fn replace_is_some_with_if_let_some(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let if_expr = ctx.find_node_at_offset::<ast::IfExpr>()?;

    let cond = if_expr.condition()?;
    let call_expr = match cond {
        ast::Expr::MethodCallExpr(call) => call,
        _ => return None,
    };

    let name_ref = call_expr.name_ref()?;
    if name_ref.text() != "is_some" {
        return None;
    }

    let receiver = call_expr.receiver()?;
    let target = call_expr.syntax().text_range();

    acc.add(
        AssistId("replace_is_some_with_if_let_some", AssistKind::RefactorRewrite),
        "Replace `is_some` with `if let Some`",
        target,
        |edit| {
            let replacement = format!("let Some(_tmp) = {}", receiver);
            edit.replace(target, replacement);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::replace_is_some_with_if_let_some;

    #[test]
    fn replace_is_some_with_if_let_some_works() {
        check_assist(
            replace_is_some_with_if_let_some,
            r#"
fn main() {
    let x = Some(1);
    if x.is_som$0e() {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    if let Some(_tmp) = x {}
}
"#,
        );
    }

    #[test]
    fn replace_is_some_with_if_let_some_not_applicable() {
        check_assist_not_applicable(
            replace_is_some_with_if_let_some,
            r#"
fn main() {
    let x = Some(1);
    if x.is_non$0e() {}
}
"#,
        );
    }
}
