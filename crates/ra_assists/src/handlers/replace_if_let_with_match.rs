use ra_fmt::unwrap_trivial_block;
use ra_syntax::{
    ast::{self, edit::IndentLevel, make},
    AstNode,
};

use crate::{utils::TryEnum, Assist, AssistCtx, AssistId};

// Assist: replace_if_let_with_match
//
// Replaces `if let` with an else branch with a `match` expression.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     <|>if let Action::Move { distance } = action {
//         foo(distance)
//     } else {
//         bar()
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => foo(distance),
//         _ => bar(),
//     }
// }
// ```
pub(crate) fn replace_if_let_with_match(ctx: AssistCtx) -> Option<Assist> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    let cond = if_expr.condition()?;
    let pat = cond.pat()?;
    let expr = cond.expr()?;
    let then_block = if_expr.then_branch()?;
    let else_block = match if_expr.else_branch()? {
        ast::ElseBranch::Block(it) => it,
        ast::ElseBranch::IfExpr(_) => return None,
    };

    let sema = ctx.sema;
    ctx.add_assist(AssistId("replace_if_let_with_match"), "Replace with match", move |edit| {
        let match_expr = {
            let then_arm = {
                let then_expr = unwrap_trivial_block(then_block);
                make::match_arm(vec![pat.clone()], then_expr)
            };
            let else_arm = {
                let pattern = sema
                    .type_of_pat(&pat)
                    .and_then(|ty| TryEnum::from_ty(sema, &ty))
                    .map(|it| it.sad_pattern())
                    .unwrap_or_else(|| make::placeholder_pat().into());
                let else_expr = unwrap_trivial_block(else_block);
                make::match_arm(vec![pattern], else_expr)
            };
            make::expr_match(expr, make::match_arm_list(vec![then_arm, else_arm]))
        };

        let match_expr = IndentLevel::from_node(if_expr.syntax()).increase_indent(match_expr);

        edit.target(if_expr.syntax().text_range());
        edit.set_cursor(if_expr.syntax().text_range().start());
        edit.replace_ast::<ast::Expr>(if_expr.into(), match_expr);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_target};

    #[test]
    fn test_replace_if_let_with_match_unwraps_simple_expressions() {
        check_assist(
            replace_if_let_with_match,
            "
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if <|>let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
}           ",
            "
impl VariantData {
    pub fn is_struct(&self) -> bool {
        <|>match *self {
            VariantData::Struct(..) => true,
            _ => false,
        }
    }
}           ",
        )
    }

    #[test]
    fn test_replace_if_let_with_match_doesnt_unwrap_multiline_expressions() {
        check_assist(
            replace_if_let_with_match,
            "
fn foo() {
    if <|>let VariantData::Struct(..) = a {
        bar(
            123
        )
    } else {
        false
    }
}           ",
            "
fn foo() {
    <|>match a {
        VariantData::Struct(..) => {
            bar(
                123
            )
        }
        _ => false,
    }
}           ",
        )
    }

    #[test]
    fn replace_if_let_with_match_target() {
        check_assist_target(
            replace_if_let_with_match,
            "
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if <|>let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
}           ",
            "if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }",
        );
    }

    #[test]
    fn special_case_option() {
        check_assist(
            replace_if_let_with_match,
            r#"
enum Option<T> { Some(T), None }
use Option::*;

fn foo(x: Option<i32>) {
    <|>if let Some(x) = x {
        println!("{}", x)
    } else {
        println!("none")
    }
}
           "#,
            r#"
enum Option<T> { Some(T), None }
use Option::*;

fn foo(x: Option<i32>) {
    <|>match x {
        Some(x) => println!("{}", x),
        None => println!("none"),
    }
}
           "#,
        );
    }

    #[test]
    fn special_case_result() {
        check_assist(
            replace_if_let_with_match,
            r#"
enum Result<T, E> { Ok(T), Err(E) }
use Result::*;

fn foo(x: Result<i32, ()>) {
    <|>if let Ok(x) = x {
        println!("{}", x)
    } else {
        println!("none")
    }
}
           "#,
            r#"
enum Result<T, E> { Ok(T), Err(E) }
use Result::*;

fn foo(x: Result<i32, ()>) {
    <|>match x {
        Ok(x) => println!("{}", x),
        Err(_) => println!("none"),
    }
}
           "#,
        );
    }
}
