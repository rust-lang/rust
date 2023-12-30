//! Routines the parser and pretty-printer use to classify AST nodes.

use crate::{ast, token::Delimiter};

/// Does this expression require a semicolon to be treated as a statement?
///
/// The negation of this: "can this expression be used as a statement without a
/// semicolon" -- is used as an early bail-out in the parser so that, for
/// instance,
///
/// ```ignore (illustrative)
/// if true {...} else {...}
/// |x| 5
/// ```
///
/// isn't parsed as `(if true {...} else {...} | x) | 5`.
///
/// Nearly the same early bail-out also occurs in the right-hand side of match
/// arms:
///
/// ```ignore (illustrative)
/// match i {
///     0 => if true {...} else {...}
///     | x => {}
/// }
/// ```
///
/// Here the `|` is a leading vert in a second match arm. It is not a binary
/// operator with the If as its left operand. If the first arm were some other
/// expression for which `expr_requires_semi_to_be_stmt` returns true, then the
/// `|` on the next line would be a binary operator (leading to a parse error).
///
/// The statement case and the match-arm case are "nearly" the same early
/// bail-out because of 1 edge case. Macro calls with brace delimiter terminate
/// a statement without a semicolon, but do not terminate a match-arm without
/// comma.
///
/// ```ignore (illustrative)
/// m! {} - 1;  // two statements: a macro call followed by -1 literal
///
/// match () {
///     _ => m! {} - 1,  // binary subtraction operator
/// }
/// ```
pub fn expr_requires_semi_to_be_stmt(e: &ast::Expr) -> bool {
    use ast::ExprKind::*;

    match &e.kind {
        If(..)
        | Match(..)
        | Block(..)
        | While(..)
        | Loop(..)
        | ForLoop { .. }
        | TryBlock(..)
        | ConstBlock(..) => false,

        MacCall(mac_call) => mac_call.args.delim != Delimiter::Brace,

        _ => true,
    }
}

/// If an expression ends with `}`, returns the innermost expression ending in the `}`
pub fn expr_trailing_brace(mut expr: &ast::Expr) -> Option<&ast::Expr> {
    use ast::ExprKind::*;

    loop {
        match &expr.kind {
            AddrOf(_, _, e)
            | Assign(_, e, _)
            | AssignOp(_, _, e)
            | Binary(_, _, e)
            | Break(_, Some(e))
            | Let(_, e, _, _)
            | Range(_, Some(e), _)
            | Ret(Some(e))
            | Unary(_, e)
            | Yield(Some(e))
            | Yeet(Some(e))
            | Become(e) => {
                expr = e;
            }
            Closure(closure) => {
                expr = &closure.body;
            }
            Gen(..)
            | Block(..)
            | ForLoop { .. }
            | If(..)
            | Loop(..)
            | Match(..)
            | Struct(..)
            | TryBlock(..)
            | While(..)
            | ConstBlock(_) => break Some(expr),

            MacCall(mac) => {
                break (mac.args.delim == Delimiter::Brace).then_some(expr);
            }

            InlineAsm(_) | OffsetOf(_, _) | IncludedBytes(_) | FormatArgs(_) => {
                // These should have been denied pre-expansion.
                break None;
            }

            Break(_, None)
            | Range(_, None, _)
            | Ret(None)
            | Yield(None)
            | Array(_)
            | Call(_, _)
            | MethodCall(_)
            | Tup(_)
            | Lit(_)
            | Cast(_, _)
            | Type(_, _)
            | Await(_, _)
            | Field(_, _)
            | Index(_, _, _)
            | Underscore
            | Path(_, _)
            | Continue(_)
            | Repeat(_, _)
            | Paren(_)
            | Try(_)
            | Yeet(None)
            | Err(_)
            | Dummy => break None,
        }
    }
}
