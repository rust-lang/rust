//! This module describes hir-level representation of expressions.
//!
//! This representaion is:
//!
//! 1. Identity-based. Each expression has an `id`, so we can distinguish
//!    between different `1` in `1 + 1`.
//! 2. Independent of syntax. Though syntactic provenance information can be
//!    attached separately via id-based side map.
//! 3. Unresolved. Paths are stored as sequences of names, and not as defs the
//!    names refer to.
//! 4. Desugared. There's no `if let`.
//!
//! See also a neighboring `body` module.

use hir_expand::name::Name;
use ra_arena::{Idx, RawId};
use ra_syntax::ast::RangeOp;

use crate::{
    builtin_type::{BuiltinFloat, BuiltinInt},
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
};

pub type ExprId = Idx<Expr>;
pub(crate) fn dummy_expr_id() -> ExprId {
    ExprId::from_raw(RawId::from(!0))
}

pub type PatId = Idx<Pat>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Literal {
    String(String),
    ByteString(Vec<u8>),
    Char(char),
    Bool(bool),
    Int(u64, Option<BuiltinInt>),
    Float(u64, Option<BuiltinFloat>), // FIXME: f64 is not Eq
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Expr {
    /// This is produced if the syntax tree does not have a required expression piece.
    Missing,
    Path(Path),
    If {
        condition: ExprId,
        then_branch: ExprId,
        else_branch: Option<ExprId>,
    },
    Block {
        statements: Vec<Statement>,
        tail: Option<ExprId>,
        label: Option<Name>,
    },
    Loop {
        body: ExprId,
        label: Option<Name>,
    },
    While {
        condition: ExprId,
        body: ExprId,
        label: Option<Name>,
    },
    For {
        iterable: ExprId,
        pat: PatId,
        body: ExprId,
        label: Option<Name>,
    },
    Call {
        callee: ExprId,
        args: Vec<ExprId>,
    },
    MethodCall {
        receiver: ExprId,
        method_name: Name,
        args: Vec<ExprId>,
        generic_args: Option<GenericArgs>,
    },
    Match {
        expr: ExprId,
        arms: Vec<MatchArm>,
    },
    Continue {
        label: Option<Name>,
    },
    Break {
        expr: Option<ExprId>,
        label: Option<Name>,
    },
    Return {
        expr: Option<ExprId>,
    },
    RecordLit {
        path: Option<Path>,
        fields: Vec<RecordLitField>,
        spread: Option<ExprId>,
    },
    Field {
        expr: ExprId,
        name: Name,
    },
    Await {
        expr: ExprId,
    },
    Try {
        expr: ExprId,
    },
    TryBlock {
        body: ExprId,
    },
    Cast {
        expr: ExprId,
        type_ref: TypeRef,
    },
    Ref {
        expr: ExprId,
        rawness: Rawness,
        mutability: Mutability,
    },
    Box {
        expr: ExprId,
    },
    UnaryOp {
        expr: ExprId,
        op: UnaryOp,
    },
    BinaryOp {
        lhs: ExprId,
        rhs: ExprId,
        op: Option<BinaryOp>,
    },
    Range {
        lhs: Option<ExprId>,
        rhs: Option<ExprId>,
        range_type: RangeOp,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    Lambda {
        args: Vec<PatId>,
        arg_types: Vec<Option<TypeRef>>,
        ret_type: Option<TypeRef>,
        body: ExprId,
    },
    Tuple {
        exprs: Vec<ExprId>,
    },
    Unsafe {
        body: ExprId,
    },
    Array(Array),
    Literal(Literal),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    LogicOp(LogicOp),
    ArithOp(ArithOp),
    CmpOp(CmpOp),
    Assignment { op: Option<ArithOp> },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicOp {
    And,
    Or,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq { negated: bool },
    Ord { ordering: Ordering, strict: bool },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    Less,
    Greater,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArithOp {
    Add,
    Mul,
    Sub,
    Div,
    Rem,
    Shl,
    Shr,
    BitXor,
    BitOr,
    BitAnd,
}

pub use ra_syntax::ast::PrefixOp as UnaryOp;
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Array {
    ElementList(Vec<ExprId>),
    Repeat { initializer: ExprId, repeat: ExprId },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MatchArm {
    pub pat: PatId,
    pub guard: Option<ExprId>,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecordLitField {
    pub name: Name,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Statement {
    Let { pat: PatId, type_ref: Option<TypeRef>, initializer: Option<ExprId> },
    Expr(ExprId),
}

impl Expr {
    pub fn walk_child_exprs(&self, mut f: impl FnMut(ExprId)) {
        match self {
            Expr::Missing => {}
            Expr::Path(_) => {}
            Expr::If { condition, then_branch, else_branch } => {
                f(*condition);
                f(*then_branch);
                if let Some(else_branch) = else_branch {
                    f(*else_branch);
                }
            }
            Expr::Block { statements, tail, .. } => {
                for stmt in statements {
                    match stmt {
                        Statement::Let { initializer, .. } => {
                            if let Some(expr) = initializer {
                                f(*expr);
                            }
                        }
                        Statement::Expr(e) => f(*e),
                    }
                }
                if let Some(expr) = tail {
                    f(*expr);
                }
            }
            Expr::TryBlock { body } | Expr::Unsafe { body } => f(*body),
            Expr::Loop { body, .. } => f(*body),
            Expr::While { condition, body, .. } => {
                f(*condition);
                f(*body);
            }
            Expr::For { iterable, body, .. } => {
                f(*iterable);
                f(*body);
            }
            Expr::Call { callee, args } => {
                f(*callee);
                for arg in args {
                    f(*arg);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                f(*receiver);
                for arg in args {
                    f(*arg);
                }
            }
            Expr::Match { expr, arms } => {
                f(*expr);
                for arm in arms {
                    f(arm.expr);
                }
            }
            Expr::Continue { .. } => {}
            Expr::Break { expr, .. } | Expr::Return { expr } => {
                if let Some(expr) = expr {
                    f(*expr);
                }
            }
            Expr::RecordLit { fields, spread, .. } => {
                for field in fields {
                    f(field.expr);
                }
                if let Some(expr) = spread {
                    f(*expr);
                }
            }
            Expr::Lambda { body, .. } => {
                f(*body);
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Expr::Range { lhs, rhs, .. } => {
                if let Some(lhs) = rhs {
                    f(*lhs);
                }
                if let Some(rhs) = lhs {
                    f(*rhs);
                }
            }
            Expr::Index { base, index } => {
                f(*base);
                f(*index);
            }
            Expr::Field { expr, .. }
            | Expr::Await { expr }
            | Expr::Try { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. }
            | Expr::Box { expr } => {
                f(*expr);
            }
            Expr::Tuple { exprs } => {
                for expr in exprs {
                    f(*expr);
                }
            }
            Expr::Array(a) => match a {
                Array::ElementList(exprs) => {
                    for expr in exprs {
                        f(*expr);
                    }
                }
                Array::Repeat { initializer, repeat } => {
                    f(*initializer);
                    f(*repeat)
                }
            },
            Expr::Literal(_) => {}
        }
    }
}

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Clone, PartialEq, Eq, Debug, Copy)]
pub enum BindingAnnotation {
    /// No binding annotation given: this means that the final binding mode
    /// will depend on whether we have skipped through a `&` reference
    /// when matching. For example, the `x` in `Some(x)` will have binding
    /// mode `None`; if you do `let Some(x) = &Some(22)`, it will
    /// ultimately be inferred to be by-reference.
    Unannotated,

    /// Annotated with `mut x` -- could be either ref or not, similar to `None`.
    Mutable,

    /// Annotated as `ref`, like `ref x`
    Ref,

    /// Annotated as `ref mut x`.
    RefMut,
}

impl BindingAnnotation {
    pub fn new(is_mutable: bool, is_ref: bool) -> Self {
        match (is_mutable, is_ref) {
            (true, true) => BindingAnnotation::RefMut,
            (false, true) => BindingAnnotation::Ref,
            (true, false) => BindingAnnotation::Mutable,
            (false, false) => BindingAnnotation::Unannotated,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecordFieldPat {
    pub name: Name,
    pub pat: PatId,
}

/// Close relative to rustc's hir::PatKind
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Pat {
    Missing,
    Wild,
    Tuple { args: Vec<PatId>, ellipsis: Option<usize> },
    Or(Vec<PatId>),
    Record { path: Option<Path>, args: Vec<RecordFieldPat>, ellipsis: bool },
    Range { start: ExprId, end: ExprId },
    Slice { prefix: Vec<PatId>, slice: Option<PatId>, suffix: Vec<PatId> },
    Path(Path),
    Lit(ExprId),
    Bind { mode: BindingAnnotation, name: Name, subpat: Option<PatId> },
    TupleStruct { path: Option<Path>, args: Vec<PatId>, ellipsis: Option<usize> },
    Ref { pat: PatId, mutability: Mutability },
}

impl Pat {
    pub fn walk_child_pats(&self, mut f: impl FnMut(PatId)) {
        match self {
            Pat::Range { .. } | Pat::Lit(..) | Pat::Path(..) | Pat::Wild | Pat::Missing => {}
            Pat::Bind { subpat, .. } => {
                subpat.iter().copied().for_each(f);
            }
            Pat::Or(args) | Pat::Tuple { args, .. } | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, slice, suffix } => {
                let total_iter = prefix.iter().chain(slice.iter()).chain(suffix.iter());
                total_iter.copied().for_each(f);
            }
            Pat::Record { args, .. } => {
                args.iter().map(|f| f.pat).for_each(f);
            }
        }
    }
}
