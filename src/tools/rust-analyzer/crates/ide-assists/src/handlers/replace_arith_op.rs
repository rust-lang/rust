use ide_db::assists::{AssistId, AssistKind, GroupLabel};
use syntax::{
    ast::{self, ArithOp, BinaryOp},
    AstNode, TextRange,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: replace_arith_with_checked
//
// Replaces arithmetic on integers with the `checked_*` equivalent.
//
// ```
// fn main() {
//   let x = 1 $0+ 2;
// }
// ```
// ->
// ```
// fn main() {
//   let x = 1.checked_add(2);
// }
// ```
pub(crate) fn replace_arith_with_checked(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    replace_arith(acc, ctx, ArithKind::Checked)
}

// Assist: replace_arith_with_saturating
//
// Replaces arithmetic on integers with the `saturating_*` equivalent.
//
// ```
// fn main() {
//   let x = 1 $0+ 2;
// }
// ```
// ->
// ```
// fn main() {
//   let x = 1.saturating_add(2);
// }
// ```
pub(crate) fn replace_arith_with_saturating(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    replace_arith(acc, ctx, ArithKind::Saturating)
}

// Assist: replace_arith_with_wrapping
//
// Replaces arithmetic on integers with the `wrapping_*` equivalent.
//
// ```
// fn main() {
//   let x = 1 $0+ 2;
// }
// ```
// ->
// ```
// fn main() {
//   let x = 1.wrapping_add(2);
// }
// ```
pub(crate) fn replace_arith_with_wrapping(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    replace_arith(acc, ctx, ArithKind::Wrapping)
}

fn replace_arith(acc: &mut Assists, ctx: &AssistContext<'_>, kind: ArithKind) -> Option<()> {
    let (lhs, op, rhs) = parse_binary_op(ctx)?;

    if !is_primitive_int(ctx, &lhs) || !is_primitive_int(ctx, &rhs) {
        return None;
    }

    let start = lhs.syntax().text_range().start();
    let end = rhs.syntax().text_range().end();
    let range = TextRange::new(start, end);

    acc.add_group(
        &GroupLabel("Replace arithmetic...".into()),
        kind.assist_id(),
        kind.label(),
        range,
        |builder| {
            let method_name = kind.method_name(op);

            builder.replace(range, format!("{lhs}.{method_name}({rhs})"))
        },
    )
}

fn is_primitive_int(ctx: &AssistContext<'_>, expr: &ast::Expr) -> bool {
    match ctx.sema.type_of_expr(expr) {
        Some(ty) => ty.adjusted().is_int_or_uint(),
        _ => false,
    }
}

/// Extract the operands of an arithmetic expression (e.g. `1 + 2` or `1.checked_add(2)`)
fn parse_binary_op(ctx: &AssistContext<'_>) -> Option<(ast::Expr, ArithOp, ast::Expr)> {
    let expr = ctx.find_node_at_offset::<ast::BinExpr>()?;

    let op = match expr.op_kind() {
        Some(BinaryOp::ArithOp(ArithOp::Add)) => ArithOp::Add,
        Some(BinaryOp::ArithOp(ArithOp::Sub)) => ArithOp::Sub,
        Some(BinaryOp::ArithOp(ArithOp::Mul)) => ArithOp::Mul,
        Some(BinaryOp::ArithOp(ArithOp::Div)) => ArithOp::Div,
        _ => return None,
    };

    let lhs = expr.lhs()?;
    let rhs = expr.rhs()?;

    Some((lhs, op, rhs))
}

pub(crate) enum ArithKind {
    Saturating,
    Wrapping,
    Checked,
}

impl ArithKind {
    fn assist_id(&self) -> AssistId {
        let s = match self {
            ArithKind::Saturating => "replace_arith_with_saturating",
            ArithKind::Checked => "replace_arith_with_checked",
            ArithKind::Wrapping => "replace_arith_with_wrapping",
        };

        AssistId(s, AssistKind::RefactorRewrite)
    }

    fn label(&self) -> &'static str {
        match self {
            ArithKind::Saturating => "Replace arithmetic with call to saturating_*",
            ArithKind::Checked => "Replace arithmetic with call to checked_*",
            ArithKind::Wrapping => "Replace arithmetic with call to wrapping_*",
        }
    }

    fn method_name(&self, op: ArithOp) -> String {
        let prefix = match self {
            ArithKind::Checked => "checked_",
            ArithKind::Wrapping => "wrapping_",
            ArithKind::Saturating => "saturating_",
        };

        let suffix = match op {
            ArithOp::Add => "add",
            ArithOp::Sub => "sub",
            ArithOp::Mul => "mul",
            ArithOp::Div => "div",
            _ => unreachable!("this function should only be called with +, -, / or *"),
        };
        format!("{prefix}{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn arith_kind_method_name() {
        assert_eq!(ArithKind::Saturating.method_name(ArithOp::Add), "saturating_add");
        assert_eq!(ArithKind::Checked.method_name(ArithOp::Sub), "checked_sub");
    }

    #[test]
    fn replace_arith_with_checked_add() {
        check_assist(
            replace_arith_with_checked,
            r#"
fn main() {
    let x = 1 $0+ 2;
}
"#,
            r#"
fn main() {
    let x = 1.checked_add(2);
}
"#,
        )
    }

    #[test]
    fn replace_arith_with_saturating_add() {
        check_assist(
            replace_arith_with_saturating,
            r#"
fn main() {
    let x = 1 $0+ 2;
}
"#,
            r#"
fn main() {
    let x = 1.saturating_add(2);
}
"#,
        )
    }

    #[test]
    fn replace_arith_with_wrapping_add() {
        check_assist(
            replace_arith_with_wrapping,
            r#"
fn main() {
    let x = 1 $0+ 2;
}
"#,
            r#"
fn main() {
    let x = 1.wrapping_add(2);
}
"#,
        )
    }
}
