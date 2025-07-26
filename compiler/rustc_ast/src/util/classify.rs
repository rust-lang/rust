//! Routines the parser and pretty-printer use to classify AST nodes.

use crate::ast::ExprKind::*;
use crate::ast::{self, MatchKind};
use crate::token::Delimiter;

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

/// Returns whether the leftmost token of the given expression is the label of a
/// labeled loop or block, such as in `'inner: loop { break 'inner 1 } + 1`.
///
/// Such expressions are not allowed as the value of an unlabeled break.
///
/// ```ignore (illustrative)
/// 'outer: {
///     break 'inner: loop { break 'inner 1 } + 1;  // invalid syntax
///
///     break 'outer 'inner: loop { break 'inner 1 } + 1;  // okay
///
///     break ('inner: loop { break 'inner 1 } + 1);  // okay
///
///     break ('inner: loop { break 'inner 1 }) + 1;  // okay
/// }
/// ```
pub fn leading_labeled_expr(mut expr: &ast::Expr) -> bool {
    loop {
        match &expr.kind {
            Block(_, label) | ForLoop { label, .. } | Loop(_, label, _) | While(_, _, label) => {
                return label.is_some();
            }

            Assign(e, _, _)
            | AssignOp(_, e, _)
            | Await(e, _)
            | Use(e, _)
            | Binary(_, e, _)
            | Call(e, _)
            | Cast(e, _)
            | Field(e, _)
            | Index(e, _, _)
            | Match(e, _, MatchKind::Postfix)
            | Range(Some(e), _, _)
            | Try(e) => {
                expr = e;
            }
            MethodCall(method_call) => {
                expr = &method_call.receiver;
            }

            AddrOf(..)
            | Array(..)
            | Become(..)
            | Break(..)
            | Closure(..)
            | ConstBlock(..)
            | Continue(..)
            | FormatArgs(..)
            | Gen(..)
            | If(..)
            | IncludedBytes(..)
            | InlineAsm(..)
            | Let(..)
            | Lit(..)
            | MacCall(..)
            | Match(_, _, MatchKind::Prefix)
            | OffsetOf(..)
            | Paren(..)
            | Path(..)
            | Range(None, _, _)
            | Repeat(..)
            | Ret(..)
            | Struct(..)
            | TryBlock(..)
            | Tup(..)
            | Type(..)
            | Unary(..)
            | Underscore
            | Yeet(..)
            | Yield(..)
            | UnsafeBinderCast(..)
            | Err(..)
            | Dummy => return false,
        }
    }
}

pub enum TrailingBrace<'a> {
    /// Trailing brace in a macro call, like the one in `x as *const brace! {}`.
    /// We will suggest changing the macro call to a different delimiter.
    MacCall(&'a ast::MacCall),
    /// Trailing brace in any other expression, such as `a + B {}`. We will
    /// suggest wrapping the innermost expression in parentheses: `a + (B {})`.
    Expr(&'a ast::Expr),
}

/// If an expression ends with `}`, returns the innermost expression ending in the `}`
pub fn expr_trailing_brace(mut expr: &ast::Expr) -> Option<TrailingBrace<'_>> {
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
            | Yeet(Some(e))
            | Become(e) => {
                expr = e;
            }
            Yield(kind) => match kind.expr() {
                Some(e) => expr = e,
                None => break None,
            },
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
            | ConstBlock(_) => break Some(TrailingBrace::Expr(expr)),

            Cast(_, ty) => {
                break type_trailing_braced_mac_call(ty).map(TrailingBrace::MacCall);
            }

            MacCall(mac) => {
                break (mac.args.delim == Delimiter::Brace).then_some(TrailingBrace::MacCall(mac));
            }

            InlineAsm(_) | OffsetOf(_, _) | IncludedBytes(_) | FormatArgs(_) => {
                // These should have been denied pre-expansion.
                break None;
            }

            Break(_, None)
            | Range(_, None, _)
            | Ret(None)
            | Array(_)
            | Call(_, _)
            | MethodCall(_)
            | Tup(_)
            | Lit(_)
            | Type(_, _)
            | Await(_, _)
            | Use(_, _)
            | Field(_, _)
            | Index(_, _, _)
            | Underscore
            | Path(_, _)
            | Continue(_)
            | Repeat(_, _)
            | Paren(_)
            | Try(_)
            | Yeet(None)
            | UnsafeBinderCast(..)
            | Err(_)
            | Dummy => {
                break None;
            }
        }
    }
}

/// If the type's last token is `}`, it must be due to a braced macro call, such
/// as in `*const brace! { ... }`. Returns that trailing macro call.
fn type_trailing_braced_mac_call(mut ty: &ast::Ty) -> Option<&ast::MacCall> {
    loop {
        match &ty.kind {
            ast::TyKind::MacCall(mac) => {
                break (mac.args.delim == Delimiter::Brace).then_some(mac);
            }

            ast::TyKind::Ptr(mut_ty)
            | ast::TyKind::Ref(_, mut_ty)
            | ast::TyKind::PinnedRef(_, mut_ty) => {
                ty = &mut_ty.ty;
            }

            ast::TyKind::UnsafeBinder(binder) => {
                ty = &binder.inner_ty;
            }

            ast::TyKind::FnPtr(fn_ty) => match &fn_ty.decl.output {
                ast::FnRetTy::Default(_) => break None,
                ast::FnRetTy::Ty(ret) => ty = ret,
            },

            ast::TyKind::Path(_, path) => match path_return_type(path) {
                Some(trailing_ty) => ty = trailing_ty,
                None => break None,
            },

            ast::TyKind::TraitObject(bounds, _) | ast::TyKind::ImplTrait(_, bounds) => {
                match bounds.last() {
                    Some(ast::GenericBound::Trait(bound)) => {
                        match path_return_type(&bound.trait_ref.path) {
                            Some(trailing_ty) => ty = trailing_ty,
                            None => break None,
                        }
                    }
                    Some(ast::GenericBound::Outlives(_) | ast::GenericBound::Use(..)) | None => {
                        break None;
                    }
                }
            }

            ast::TyKind::Slice(..)
            | ast::TyKind::Array(..)
            | ast::TyKind::Never
            | ast::TyKind::Tup(..)
            | ast::TyKind::Paren(..)
            | ast::TyKind::Typeof(..)
            | ast::TyKind::Infer
            | ast::TyKind::ImplicitSelf
            | ast::TyKind::CVarArgs
            | ast::TyKind::Pat(..)
            | ast::TyKind::Dummy
            | ast::TyKind::Err(..) => break None,
        }
    }
}

/// Returns the trailing return type in the given path, if it has one.
///
/// ```ignore (illustrative)
/// ::std::ops::FnOnce(&str) -> fn() -> *const c_void
///                             ^^^^^^^^^^^^^^^^^^^^^
/// ```
fn path_return_type(path: &ast::Path) -> Option<&ast::Ty> {
    let last_segment = path.segments.last()?;
    let args = last_segment.args.as_ref()?;
    match &**args {
        ast::GenericArgs::Parenthesized(args) => match &args.output {
            ast::FnRetTy::Default(_) => None,
            ast::FnRetTy::Ty(ret) => Some(ret),
        },
        ast::GenericArgs::AngleBracketed(_) | ast::GenericArgs::ParenthesizedElided(_) => None,
    }
}
