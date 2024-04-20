//! Routines the parser and pretty-printer use to classify AST nodes.

use crate::ast::ExprKind::*;
use crate::{ast, token::Delimiter};

/// This classification determines whether various syntactic positions break out
/// of parsing the current expression (true) or continue parsing more of the
/// same expression (false).
///
/// For example, it's relevant in the parsing of match arms:
///
/// ```ignore (illustrative)
/// match ... {
///     // Is this calling $e as a function, or is it the start of a new arm
///     // with a tuple pattern?
///     _ => $e (
///             ^                                                          )
///
///     // Is this an Index operation, or new arm with a slice pattern?
///     _ => $e [
///             ^                                                          ]
///
///     // Is this a binary operator, or leading vert in a new arm? Same for
///     // other punctuation which can either be a binary operator in
///     // expression or unary operator in pattern, such as `&` and `-`.
///     _ => $e |
///             ^
/// }
/// ```
///
/// If $e is something like `{}` or `if … {}`, then terminate the current
/// arm and parse a new arm.
///
/// If $e is something like `path::to` or `(…)`, continue parsing the same
/// arm.
///
/// *Almost* the same classification is used as an early bail-out for parsing
/// statements. See `expr_requires_semi_to_be_stmt`.
pub fn expr_is_complete(e: &ast::Expr) -> bool {
    matches!(
        e.kind,
        If(..)
            | Match(..)
            | Block(..)
            | While(..)
            | Loop(..)
            | ForLoop { .. }
            | TryBlock(..)
            | ConstBlock(..)
    )
}

/// Does this expression require a semicolon to be treated as a statement?
///
/// The negation of this: "can this expression be used as a statement without a
/// semicolon" -- is used as an early bail-out when parsing statements so that,
/// for instance,
///
/// ```ignore (illustrative)
/// if true {...} else {...}
/// |x| 5
/// ```
///
/// isn't parsed as `(if true {...} else {...} | x) | 5`.
///
/// Surprising special case: even though braced macro calls like `m! {}`
/// normally do not introduce a boundary when found at the head of a match arm,
/// they do terminate the parsing of a statement.
///
/// ```ignore (illustrative)
/// match ... {
///     _ => m! {} (),  // macro that expands to a function, which is then called
/// }
///
/// let _ = { m! {} () };  // macro call followed by unit
/// ```
pub fn expr_requires_semi_to_be_stmt(e: &ast::Expr) -> bool {
    match &e.kind {
        MacCall(mac_call) => mac_call.args.delim != Delimiter::Brace,
        _ => !expr_is_complete(e),
    }
}

/// If an expression ends with `}`, returns the innermost expression ending in the `}`
pub fn expr_trailing_brace(mut expr: &ast::Expr) -> Option<&ast::Expr> {
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
