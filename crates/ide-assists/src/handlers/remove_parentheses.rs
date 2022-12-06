use syntax::{ast, AstNode, SyntaxKind, TextRange};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_parentheses
//
// Removes useless parentheses.
//
// ```
// fn main() {
//     _ = $0(2) + 2;
// }
// ```
// ->
// ```
// fn main() {
//     _ = 2 + 2;
// }
// ```
pub(crate) fn remove_parentheses(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let parens = ctx.find_node_at_offset::<ast::ParenExpr>()?;
    let l_paren = parens.l_paren_token()?;
    let r_paren = parens.r_paren_token()?;

    let cursor_in_range = l_paren.text_range().contains_range(ctx.selection_trimmed())
        || r_paren.text_range().contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let expr = parens.expr()?;
    let parent = ast::Expr::cast(parens.syntax().parent()?);
    let is_ok_to_remove =
        parent.map_or(true, |p| ExprPrecedence::of(&expr) >= ExprPrecedence::of(&p));
    if !is_ok_to_remove {
        return None;
    }

    let delete_from_l = l_paren.text_range().start();
    let delete_to_l = match l_paren.next_token() {
        Some(it) if it.kind() == SyntaxKind::WHITESPACE => it.text_range().end(),
        _ => l_paren.text_range().end(),
    };

    let delete_from_r = match r_paren.prev_token() {
        Some(it) if it.kind() == SyntaxKind::WHITESPACE => it.text_range().start(),
        _ => r_paren.text_range().start(),
    };
    let delete_to_r = r_paren.text_range().end();

    let target = parens.syntax().text_range();
    acc.add(
        AssistId("remove_parentheses", AssistKind::Refactor),
        "Remove parentheses",
        target,
        |builder| {
            builder.delete(TextRange::new(delete_from_l, delete_to_l));
            builder.delete(TextRange::new(delete_from_r, delete_to_r));
        },
    )
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum ExprPrecedence {
    // N.B.: Order is important
    /// Precedence is unknown
    Dummy,
    Closure,
    Jump,
    Range,
    Bin(BinOpPresedence),
    Prefix,
    Postfix,
    Paren,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum BinOpPresedence {
    // N.B.: Order is important
    /// `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `|=`, `&=`
    Assign,
    /// `||`
    LOr,
    /// `&&`
    LAnd,
    /// `<`, `<=`, `>`, `>=`, `==` and `!=`
    Cmp,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `&`
    BitAnd,
    /// `<<` and `>>`
    Shift,
    /// `+` and `-`
    Add,
    /// `*`, `/` and `%`
    Mul,
    /// `as`
    As,
}

impl ExprPrecedence {
    pub fn of(expr: &ast::Expr) -> Self {
        // Copied from <https://github.com/rust-lang/rust/blob/b6852428a8ea9728369b64b9964cad8e258403d3/compiler/rustc_ast/src/util/parser.rs#L296>
        use ast::Expr::*;

        match expr {
            ClosureExpr(_) => Self::Closure,

            ContinueExpr(_) | ReturnExpr(_) | YieldExpr(_) | BreakExpr(_) => Self::Jump,

            RangeExpr(_) => Self::Range,

            BinExpr(bin_expr) => bin_expr
                .op_kind()
                .map(|op| match op {
                    ast::BinaryOp::LogicOp(op) => match op {
                        ast::LogicOp::And => BinOpPresedence::LAnd,
                        ast::LogicOp::Or => BinOpPresedence::LOr,
                    },
                    ast::BinaryOp::ArithOp(op) => match op {
                        ast::ArithOp::Add => BinOpPresedence::Add,
                        ast::ArithOp::Mul => BinOpPresedence::Mul,
                        ast::ArithOp::Sub => BinOpPresedence::Add,
                        ast::ArithOp::Div => BinOpPresedence::Mul,
                        ast::ArithOp::Rem => BinOpPresedence::Mul,
                        ast::ArithOp::Shl => BinOpPresedence::Shift,
                        ast::ArithOp::Shr => BinOpPresedence::Shift,
                        ast::ArithOp::BitXor => BinOpPresedence::BitXor,
                        ast::ArithOp::BitOr => BinOpPresedence::BitOr,
                        ast::ArithOp::BitAnd => BinOpPresedence::BitAnd,
                    },
                    ast::BinaryOp::CmpOp(_) => BinOpPresedence::Cmp,
                    ast::BinaryOp::Assignment { .. } => BinOpPresedence::Assign,
                })
                .map(Self::Bin)
                .unwrap_or(Self::Dummy),
            CastExpr(_) => Self::Bin(BinOpPresedence::As),

            BoxExpr(_) | RefExpr(_) | LetExpr(_) | PrefixExpr(_) => Self::Prefix,

            AwaitExpr(_) | CallExpr(_) | MethodCallExpr(_) | FieldExpr(_) | IndexExpr(_)
            | TryExpr(_) | MacroExpr(_) => Self::Postfix,

            ArrayExpr(_) | TupleExpr(_) | Literal(_) | PathExpr(_) | ParenExpr(_) | IfExpr(_)
            | WhileExpr(_) | ForExpr(_) | LoopExpr(_) | MatchExpr(_) | BlockExpr(_)
            | RecordExpr(_) | UnderscoreExpr(_) => Self::Paren,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn remove_parens_simple() {
        check_assist(remove_parentheses, r#"fn f() { $0(2) + 2; }"#, r#"fn f() { 2 + 2; }"#);
        check_assist(remove_parentheses, r#"fn f() { ($02) + 2; }"#, r#"fn f() { 2 + 2; }"#);
        check_assist(remove_parentheses, r#"fn f() { (2)$0 + 2; }"#, r#"fn f() { 2 + 2; }"#);
        check_assist(remove_parentheses, r#"fn f() { (2$0) + 2; }"#, r#"fn f() { 2 + 2; }"#);
    }

    #[test]
    fn remove_parens_precedence() {
        check_assist(
            remove_parentheses,
            r#"fn f() { $0(2 * 3) + 1; }"#,
            r#"fn f() { 2 * 3 + 1; }"#,
        );
        check_assist(remove_parentheses, r#"fn f() { ( $0(2) ); }"#, r#"fn f() { ( 2 ); }"#);
        check_assist(remove_parentheses, r#"fn f() { $0(2?)?; }"#, r#"fn f() { 2??; }"#);
        check_assist(remove_parentheses, r#"fn f() { f(($02 + 2)); }"#, r#"fn f() { f(2 + 2); }"#);
        check_assist(
            remove_parentheses,
            r#"fn f() { (1<2)&&$0(3>4); }"#,
            r#"fn f() { (1<2)&&3>4; }"#,
        );
    }

    #[test]
    fn remove_parens_doesnt_apply_precedence() {
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(2 + 2) * 8; }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(2 + 2).f(); }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0(2 + 2).await; }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() { $0!(2..2); }"#);
    }

    #[test]
    fn remove_parens_doesnt_apply_with_cursor_not_on_paren() {
        check_assist_not_applicable(remove_parentheses, r#"fn f() { (2 +$0 2) }"#);
        check_assist_not_applicable(remove_parentheses, r#"fn f() {$0 (2 + 2) }"#);
    }
}
