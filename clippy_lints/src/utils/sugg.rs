//! Contains utility functions to generate suggestions.
#![deny(missing_docs_in_private_items)]
// currently ignores lifetimes and generics
#![allow(use_self)]

use rustc::hir;
use rustc::lint::{EarlyContext, LateContext, LintContext};
use rustc_errors;
use std::borrow::Cow;
use std::fmt::Display;
use std;
use syntax::codemap::{CharPos, Span};
use syntax::parse::token;
use syntax::print::pprust::token_to_string;
use syntax::util::parser::AssocOp;
use syntax::ast;
use utils::{higher, snippet, snippet_opt};

/// A helper type to build suggestion correctly handling parenthesis.
pub enum Sugg<'a> {
    /// An expression that never needs parenthesis such as `1337` or `[0; 42]`.
    NonParen(Cow<'a, str>),
    /// An expression that does not fit in other variants.
    MaybeParen(Cow<'a, str>),
    /// A binary operator expression, including `as`-casts and explicit type
    /// coercion.
    BinOp(AssocOp, Cow<'a, str>),
}

/// Literal constant `1`, for convenience.
pub const ONE: Sugg<'static> = Sugg::NonParen(Cow::Borrowed("1"));

impl<'a> Display for Sugg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match *self {
            Sugg::NonParen(ref s) |
            Sugg::MaybeParen(ref s) |
            Sugg::BinOp(_, ref s) => s.fmt(f),
        }
    }
}

#[allow(wrong_self_convention)] // ok, because of the function `as_ty` method
impl<'a> Sugg<'a> {
    /// Prepare a suggestion from an expression.
    pub fn hir_opt(cx: &LateContext, expr: &hir::Expr) -> Option<Self> {
        snippet_opt(cx, expr.span).map(|snippet| {
            let snippet = Cow::Owned(snippet);
            match expr.node {
                hir::ExprAddrOf(..) |
                hir::ExprBox(..) |
                hir::ExprClosure(..) |
                hir::ExprIf(..) |
                hir::ExprUnary(..) |
                hir::ExprMatch(..) => Sugg::MaybeParen(snippet),
                hir::ExprAgain(..) |
                hir::ExprArray(..) |
                hir::ExprBlock(..) |
                hir::ExprBreak(..) |
                hir::ExprCall(..) |
                hir::ExprField(..) |
                hir::ExprIndex(..) |
                hir::ExprInlineAsm(..) |
                hir::ExprLit(..) |
                hir::ExprLoop(..) |
                hir::ExprMethodCall(..) |
                hir::ExprPath(..) |
                hir::ExprRepeat(..) |
                hir::ExprRet(..) |
                hir::ExprStruct(..) |
                hir::ExprTup(..) |
                hir::ExprTupField(..) |
                hir::ExprWhile(..) => Sugg::NonParen(snippet),
                hir::ExprAssign(..) => Sugg::BinOp(AssocOp::Assign, snippet),
                hir::ExprAssignOp(op, ..) => Sugg::BinOp(hirbinop2assignop(op), snippet),
                hir::ExprBinary(op, ..) => Sugg::BinOp(AssocOp::from_ast_binop(higher::binop(op.node)), snippet),
                hir::ExprCast(..) => Sugg::BinOp(AssocOp::As, snippet),
                hir::ExprType(..) => Sugg::BinOp(AssocOp::Colon, snippet),
            }
        })
    }

    /// Convenience function around `hir_opt` for suggestions with a default
    /// text.
    pub fn hir(cx: &LateContext, expr: &hir::Expr, default: &'a str) -> Self {
        Self::hir_opt(cx, expr).unwrap_or_else(|| Sugg::NonParen(Cow::Borrowed(default)))
    }

    /// Prepare a suggestion from an expression.
    pub fn ast(cx: &EarlyContext, expr: &ast::Expr, default: &'a str) -> Self {
        use syntax::ast::RangeLimits;

        let snippet = snippet(cx, expr.span, default);

        match expr.node {
            ast::ExprKind::AddrOf(..) |
            ast::ExprKind::Box(..) |
            ast::ExprKind::Closure(..) |
            ast::ExprKind::If(..) |
            ast::ExprKind::IfLet(..) |
            ast::ExprKind::InPlace(..) |
            ast::ExprKind::Unary(..) |
            ast::ExprKind::Match(..) => Sugg::MaybeParen(snippet),
            ast::ExprKind::Block(..) |
            ast::ExprKind::Break(..) |
            ast::ExprKind::Call(..) |
            ast::ExprKind::Catch(..) |
            ast::ExprKind::Continue(..) |
            ast::ExprKind::Field(..) |
            ast::ExprKind::ForLoop(..) |
            ast::ExprKind::Index(..) |
            ast::ExprKind::InlineAsm(..) |
            ast::ExprKind::Lit(..) |
            ast::ExprKind::Loop(..) |
            ast::ExprKind::Mac(..) |
            ast::ExprKind::MethodCall(..) |
            ast::ExprKind::Paren(..) |
            ast::ExprKind::Path(..) |
            ast::ExprKind::Repeat(..) |
            ast::ExprKind::Ret(..) |
            ast::ExprKind::Struct(..) |
            ast::ExprKind::Try(..) |
            ast::ExprKind::Tup(..) |
            ast::ExprKind::TupField(..) |
            ast::ExprKind::Array(..) |
            ast::ExprKind::While(..) |
            ast::ExprKind::WhileLet(..) => Sugg::NonParen(snippet),
            ast::ExprKind::Range(.., RangeLimits::HalfOpen) => Sugg::BinOp(AssocOp::DotDot, snippet),
            ast::ExprKind::Range(.., RangeLimits::Closed) => Sugg::BinOp(AssocOp::DotDotDot, snippet),
            ast::ExprKind::Assign(..) => Sugg::BinOp(AssocOp::Assign, snippet),
            ast::ExprKind::AssignOp(op, ..) => Sugg::BinOp(astbinop2assignop(op), snippet),
            ast::ExprKind::Binary(op, ..) => Sugg::BinOp(AssocOp::from_ast_binop(op.node), snippet),
            ast::ExprKind::Cast(..) => Sugg::BinOp(AssocOp::As, snippet),
            ast::ExprKind::Type(..) => Sugg::BinOp(AssocOp::Colon, snippet),
        }
    }

    /// Convenience method to create the `<lhs> && <rhs>` suggestion.
    pub fn and(self, rhs: Self) -> Sugg<'static> {
        make_binop(ast::BinOpKind::And, &self, &rhs)
    }

    /// Convenience method to create the `<lhs> as <rhs>` suggestion.
    pub fn as_ty<R: Display>(self, rhs: R) -> Sugg<'static> {
        make_assoc(AssocOp::As, &self, &Sugg::NonParen(rhs.to_string().into()))
    }

    /// Convenience method to create the `&<expr>` suggestion.
    pub fn addr(self) -> Sugg<'static> {
        make_unop("&", self)
    }

    /// Convenience method to create the `&mut <expr>` suggestion.
    pub fn mut_addr(self) -> Sugg<'static> {
        make_unop("&mut ", self)
    }

    /// Convenience method to create the `*<expr>` suggestion.
    pub fn deref(self) -> Sugg<'static> {
        make_unop("*", self)
    }

    /// Convenience method to create the `<lhs>..<rhs>` or `<lhs>...<rhs>`
    /// suggestion.
    pub fn range(self, end: Self, limit: ast::RangeLimits) -> Sugg<'static> {
        match limit {
            ast::RangeLimits::HalfOpen => make_assoc(AssocOp::DotDot, &self, &end),
            ast::RangeLimits::Closed => make_assoc(AssocOp::DotDotDot, &self, &end),
        }
    }

    /// Add parenthesis to any expression that might need them. Suitable to the
    /// `self` argument of
    /// a method call (eg. to build `bar.foo()` or `(1 + 2).foo()`).
    pub fn maybe_par(self) -> Self {
        match self {
            Sugg::NonParen(..) => self,
            // (x) and (x).y() both don't need additional parens
            Sugg::MaybeParen(sugg) => {
                if sugg.starts_with('(') && sugg.ends_with(')') {
                    Sugg::MaybeParen(sugg)
                } else {
                    Sugg::NonParen(format!("({})", sugg).into())
                }
            },
            Sugg::BinOp(_, sugg) => Sugg::NonParen(format!("({})", sugg).into()),
        }
    }
}

impl<'a, 'b> std::ops::Add<Sugg<'b>> for Sugg<'a> {
    type Output = Sugg<'static>;
    fn add(self, rhs: Sugg<'b>) -> Sugg<'static> {
        make_binop(ast::BinOpKind::Add, &self, &rhs)
    }
}

impl<'a, 'b> std::ops::Sub<Sugg<'b>> for Sugg<'a> {
    type Output = Sugg<'static>;
    fn sub(self, rhs: Sugg<'b>) -> Sugg<'static> {
        make_binop(ast::BinOpKind::Sub, &self, &rhs)
    }
}

impl<'a> std::ops::Not for Sugg<'a> {
    type Output = Sugg<'static>;
    fn not(self) -> Sugg<'static> {
        make_unop("!", self)
    }
}

/// Helper type to display either `foo` or `(foo)`.
struct ParenHelper<T> {
    /// Whether parenthesis are needed.
    paren: bool,
    /// The main thing to display.
    wrapped: T,
}

impl<T> ParenHelper<T> {
    /// Build a `ParenHelper`.
    fn new(paren: bool, wrapped: T) -> Self {
        Self {
            paren: paren,
            wrapped: wrapped,
        }
    }
}

impl<T: Display> Display for ParenHelper<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        if self.paren {
            write!(f, "({})", self.wrapped)
        } else {
            self.wrapped.fmt(f)
        }
    }
}

/// Build the string for `<op><expr>` adding parenthesis when necessary.
///
/// For convenience, the operator is taken as a string because all unary
/// operators have the same
/// precedence.
pub fn make_unop(op: &str, expr: Sugg) -> Sugg<'static> {
    Sugg::MaybeParen(format!("{}{}", op, expr.maybe_par()).into())
}

/// Build the string for `<lhs> <op> <rhs>` adding parenthesis when necessary.
///
/// Precedence of shift operator relative to other arithmetic operation is
/// often confusing so
/// parenthesis will always be added for a mix of these.
pub fn make_assoc(op: AssocOp, lhs: &Sugg, rhs: &Sugg) -> Sugg<'static> {
    /// Whether the operator is a shift operator `<<` or `>>`.
    fn is_shift(op: &AssocOp) -> bool {
        matches!(*op, AssocOp::ShiftLeft | AssocOp::ShiftRight)
    }

    /// Whether the operator is a arithmetic operator (`+`, `-`, `*`, `/`, `%`).
    fn is_arith(op: &AssocOp) -> bool {
        matches!(
            *op,
            AssocOp::Add | AssocOp::Subtract | AssocOp::Multiply | AssocOp::Divide | AssocOp::Modulus
        )
    }

    /// Whether the operator `op` needs parenthesis with the operator `other`
    /// in the direction
    /// `dir`.
    fn needs_paren(op: &AssocOp, other: &AssocOp, dir: Associativity) -> bool {
        other.precedence() < op.precedence() ||
            (other.precedence() == op.precedence() &&
                 ((op != other && associativity(op) != dir) ||
                      (op == other && associativity(op) != Associativity::Both))) ||
            is_shift(op) && is_arith(other) || is_shift(other) && is_arith(op)
    }

    let lhs_paren = if let Sugg::BinOp(ref lop, _) = *lhs {
        needs_paren(&op, lop, Associativity::Left)
    } else {
        false
    };

    let rhs_paren = if let Sugg::BinOp(ref rop, _) = *rhs {
        needs_paren(&op, rop, Associativity::Right)
    } else {
        false
    };

    let lhs = ParenHelper::new(lhs_paren, lhs);
    let rhs = ParenHelper::new(rhs_paren, rhs);
    let sugg = match op {
        AssocOp::Add | AssocOp::BitAnd | AssocOp::BitOr | AssocOp::BitXor | AssocOp::Divide | AssocOp::Equal |
        AssocOp::Greater | AssocOp::GreaterEqual | AssocOp::LAnd | AssocOp::LOr | AssocOp::Less |
        AssocOp::LessEqual | AssocOp::Modulus | AssocOp::Multiply | AssocOp::NotEqual | AssocOp::ShiftLeft |
        AssocOp::ShiftRight | AssocOp::Subtract => {
            format!("{} {} {}", lhs, op.to_ast_binop().expect("Those are AST ops").to_string(), rhs)
        },
        AssocOp::Inplace => format!("in ({}) {}", lhs, rhs),
        AssocOp::Assign => format!("{} = {}", lhs, rhs),
        AssocOp::AssignOp(op) => format!("{} {}= {}", lhs, token_to_string(&token::BinOp(op)), rhs),
        AssocOp::As => format!("{} as {}", lhs, rhs),
        AssocOp::DotDot => format!("{}..{}", lhs, rhs),
        AssocOp::DotDotDot => format!("{}...{}", lhs, rhs),
        AssocOp::Colon => format!("{}: {}", lhs, rhs),
    };

    Sugg::BinOp(op, sugg.into())
}

/// Convinience wrapper arround `make_assoc` and `AssocOp::from_ast_binop`.
pub fn make_binop(op: ast::BinOpKind, lhs: &Sugg, rhs: &Sugg) -> Sugg<'static> {
    make_assoc(AssocOp::from_ast_binop(op), lhs, rhs)
}

#[derive(PartialEq, Eq, Clone, Copy)]
/// Operator associativity.
enum Associativity {
    /// The operator is both left-associative and right-associative.
    Both,
    /// The operator is left-associative.
    Left,
    /// The operator is not associative.
    None,
    /// The operator is right-associative.
    Right,
}

/// Return the associativity/fixity of an operator. The difference with
/// `AssocOp::fixity` is that
/// an operator can be both left and right associative (such as `+`:
/// `a + b + c == (a + b) + c == a + (b + c)`.
///
/// Chained `as` and explicit `:` type coercion never need inner parenthesis so
/// they are considered
/// associative.
fn associativity(op: &AssocOp) -> Associativity {
    use syntax::util::parser::AssocOp::*;

    match *op {
        Inplace | Assign | AssignOp(_) => Associativity::Right,
        Add | BitAnd | BitOr | BitXor | LAnd | LOr | Multiply | As | Colon => Associativity::Both,
        Divide | Equal | Greater | GreaterEqual | Less | LessEqual | Modulus | NotEqual | ShiftLeft | ShiftRight |
        Subtract => Associativity::Left,
        DotDot | DotDotDot => Associativity::None,
    }
}

/// Convert a `hir::BinOp` to the corresponding assigning binary operator.
fn hirbinop2assignop(op: hir::BinOp) -> AssocOp {
    use rustc::hir::BinOp_::*;
    use syntax::parse::token::BinOpToken::*;

    AssocOp::AssignOp(match op.node {
        BiAdd => Plus,
        BiBitAnd => And,
        BiBitOr => Or,
        BiBitXor => Caret,
        BiDiv => Slash,
        BiMul => Star,
        BiRem => Percent,
        BiShl => Shl,
        BiShr => Shr,
        BiSub => Minus,
        BiAnd | BiEq | BiGe | BiGt | BiLe | BiLt | BiNe | BiOr => panic!("This operator does not exist"),
    })
}

/// Convert an `ast::BinOp` to the corresponding assigning binary operator.
fn astbinop2assignop(op: ast::BinOp) -> AssocOp {
    use syntax::ast::BinOpKind::*;
    use syntax::parse::token::BinOpToken;

    AssocOp::AssignOp(match op.node {
        Add => BinOpToken::Plus,
        BitAnd => BinOpToken::And,
        BitOr => BinOpToken::Or,
        BitXor => BinOpToken::Caret,
        Div => BinOpToken::Slash,
        Mul => BinOpToken::Star,
        Rem => BinOpToken::Percent,
        Shl => BinOpToken::Shl,
        Shr => BinOpToken::Shr,
        Sub => BinOpToken::Minus,
        And | Eq | Ge | Gt | Le | Lt | Ne | Or => panic!("This operator does not exist"),
    })
}

/// Return the indentation before `span` if there are nothing but `[ \t]`
/// before it on its line.
fn indentation<'a, T: LintContext<'a>>(cx: &T, span: Span) -> Option<String> {
    let lo = cx.sess().codemap().lookup_char_pos(span.lo);
    if let Some(line) = lo.file.get_line(
        lo.line - 1, /* line numbers in `Loc` are 1-based */
    )
    {
        if let Some((pos, _)) = line.char_indices().find(|&(_, c)| c != ' ' && c != '\t') {
            // we can mix char and byte positions here because we only consider `[ \t]`
            if lo.col == CharPos(pos) {
                Some(line[..pos].into())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

/// Convenience extension trait for `DiagnosticBuilder`.
pub trait DiagnosticBuilderExt<'a, T: LintContext<'a>> {
    /// Suggests to add an attribute to an item.
    ///
    /// Correctly handles indentation of the attribute and item.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// db.suggest_item_with_attr(cx, item, "#[derive(Default)]");
    /// ```
    fn suggest_item_with_attr<D: Display + ?Sized>(&mut self, cx: &T, item: Span, msg: &str, attr: &D);

    /// Suggest to add an item before another.
    ///
    /// The item should not be indented (expect for inner indentation).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// db.suggest_prepend_item(cx, item,
    /// "fn foo() {
    ///     bar();
    /// }");
    /// ```
    fn suggest_prepend_item(&mut self, cx: &T, item: Span, msg: &str, new_item: &str);
}

impl<'a, 'b, 'c, T: LintContext<'c>> DiagnosticBuilderExt<'c, T> for rustc_errors::DiagnosticBuilder<'b> {
    fn suggest_item_with_attr<D: Display + ?Sized>(&mut self, cx: &T, item: Span, msg: &str, attr: &D) {
        if let Some(indent) = indentation(cx, item) {
            let span = Span {
                hi: item.lo,
                ..item
            };

            self.span_suggestion(span, msg, format!("{}\n{}", attr, indent));
        }
    }

    fn suggest_prepend_item(&mut self, cx: &T, item: Span, msg: &str, new_item: &str) {
        if let Some(indent) = indentation(cx, item) {
            let span = Span {
                hi: item.lo,
                ..item
            };

            let mut first = true;
            let new_item = new_item
                .lines()
                .map(|l| if first {
                    first = false;
                    format!("{}\n", l)
                } else {
                    format!("{}{}\n", indent, l)
                })
                .collect::<String>();

            self.span_suggestion(span, msg, format!("{}\n{}", new_item, indent));
        }
    }
}
