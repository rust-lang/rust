use syntax::ast::{self, AstNode};

use crate::{utils::suggest_name, AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_is_some_with_if_let_some
//
// Replace `if x.is_some()` with `if let Some(_tmp) = x` or `if x.is_ok()` with `if let Ok(_tmp) = x`.
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
//     if let Some(${0:x}) = x {}
// }
// ```
pub(crate) fn replace_is_method_with_if_let_method(
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
    match name_ref.text().as_str() {
        "is_some" | "is_ok" => {
            let receiver = call_expr.receiver()?;

            let var_name = if let ast::Expr::PathExpr(path_expr) = receiver.clone() {
                path_expr.path()?.to_string()
            } else {
                suggest_name::for_variable(&receiver, &ctx.sema)
            };

            let target = call_expr.syntax().text_range();

            let (assist_id, message, text) = if name_ref.text() == "is_some" {
                ("replace_is_some_with_if_let_some", "Replace `is_some` with `if let Some`", "Some")
            } else {
                ("replace_is_ok_with_if_let_ok", "Replace `is_ok` with `if let Ok`", "Ok")
            };

            acc.add(AssistId(assist_id, AssistKind::RefactorRewrite), message, target, |edit| {
                let var_name = format!("${{0:{}}}", var_name);
                let replacement = format!("let {}({}) = {}", text, var_name, receiver);
                edit.replace(target, replacement);
            })
        }
        _ => return None,
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::replace_is_method_with_if_let_method;

    #[test]
    fn replace_is_some_with_if_let_some_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    if x.is_som$0e() {}
}
"#,
            r#"
fn main() {
    let x = Some(1);
    if let Some(${0:x}) = x {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn test() -> Option<i32> {
    Some(1)
}
fn main() {
    if test().is_som$0e() {}
}
"#,
            r#"
fn test() -> Option<i32> {
    Some(1)
}
fn main() {
    if let Some(${0:test}) = test() {}
}
"#,
        );
    }

    #[test]
    fn replace_is_some_with_if_let_some_not_applicable() {
        check_assist_not_applicable(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Some(1);
    if x.is_non$0e() {}
}
"#,
        );
    }

    #[test]
    fn replace_is_ok_with_if_let_ok_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Ok(1);
    if x.is_o$0k() {}
}
"#,
            r#"
fn main() {
    let x = Ok(1);
    if let Ok(${0:x}) = x {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if test().is_o$0k() {}
}
"#,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if let Ok(${0:test}) = test() {}
}
"#,
        );
    }

    #[test]
    fn replace_is_ok_with_if_let_ok_not_applicable() {
        check_assist_not_applicable(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let x = Ok(1);
    if x.is_e$0rr() {}
}
"#,
        );
    }
}
