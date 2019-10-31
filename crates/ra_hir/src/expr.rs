//! FIXME: write short doc here

pub(crate) mod lower;
pub(crate) mod scope;
pub(crate) mod validation;

use std::{ops::Index, sync::Arc};

use hir_def::{
    path::GenericArgs,
    type_ref::{Mutability, TypeRef},
};
use ra_arena::{impl_arena_id, map::ArenaMap, Arena, RawId};
use ra_syntax::{ast, AstPtr};
use rustc_hash::FxHashMap;

use crate::{
    db::HirDatabase,
    ty::primitive::{UncertainFloatTy, UncertainIntTy},
    DefWithBody, Either, HasSource, Name, Path, Resolver, Source,
};

pub use self::scope::ExprScopes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(RawId);
impl_arena_id!(ExprId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatId(RawId);
impl_arena_id!(PatId);

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    /// The def of the item this body belongs to
    owner: DefWithBody,
    exprs: Arena<ExprId, Expr>,
    pats: Arena<PatId, Pat>,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    params: Vec<PatId>,
    /// The `ExprId` of the actual body expression.
    body_expr: ExprId,
}

type ExprPtr = Either<AstPtr<ast::Expr>, AstPtr<ast::RecordField>>;
type ExprSource = Source<ExprPtr>;

type PatPtr = Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>;
type PatSource = Source<PatPtr>;

/// An item body together with the mapping from syntax nodes to HIR expression
/// IDs. This is needed to go from e.g. a position in a file to the HIR
/// expression containing it; but for type inference etc., we want to operate on
/// a structure that is agnostic to the actual positions of expressions in the
/// file, so that we don't recompute types whenever some whitespace is typed.
///
/// One complication here is that, due to macro expansion, a single `Body` might
/// be spread across several files. So, for each ExprId and PatId, we record
/// both the HirFileId and the position inside the file. However, we only store
/// AST -> ExprId mapping for non-macro files, as it is not clear how to handle
/// this properly for macros.
#[derive(Default, Debug, Eq, PartialEq)]
pub struct BodySourceMap {
    expr_map: FxHashMap<ExprPtr, ExprId>,
    expr_map_back: ArenaMap<ExprId, ExprSource>,
    pat_map: FxHashMap<PatPtr, PatId>,
    pat_map_back: ArenaMap<PatId, PatSource>,
    field_map: FxHashMap<(ExprId, usize), AstPtr<ast::RecordField>>,
}

impl Body {
    pub fn params(&self) -> &[PatId] {
        &self.params
    }

    pub fn body_expr(&self) -> ExprId {
        self.body_expr
    }

    pub fn owner(&self) -> DefWithBody {
        self.owner
    }

    pub fn exprs(&self) -> impl Iterator<Item = (ExprId, &Expr)> {
        self.exprs.iter()
    }

    pub fn pats(&self) -> impl Iterator<Item = (PatId, &Pat)> {
        self.pats.iter()
    }
}

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub(crate) fn resolver_for_expr(
    body: Arc<Body>,
    db: &impl HirDatabase,
    expr_id: ExprId,
) -> Resolver {
    let scopes = db.expr_scopes(body.owner);
    resolver_for_scope(body, db, scopes.scope_for(expr_id))
}

pub(crate) fn resolver_for_scope(
    body: Arc<Body>,
    db: &impl HirDatabase,
    scope_id: Option<scope::ScopeId>,
) -> Resolver {
    let mut r = body.owner.resolver(db);
    let scopes = db.expr_scopes(body.owner);
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    for scope in scope_chain.into_iter().rev() {
        r = r.push_expr_scope(Arc::clone(&scopes), scope);
    }
    r
}

impl Index<ExprId> for Body {
    type Output = Expr;

    fn index(&self, expr: ExprId) -> &Expr {
        &self.exprs[expr]
    }
}

impl Index<PatId> for Body {
    type Output = Pat;

    fn index(&self, pat: PatId) -> &Pat {
        &self.pats[pat]
    }
}

impl BodySourceMap {
    pub(crate) fn expr_syntax(&self, expr: ExprId) -> Option<ExprSource> {
        self.expr_map_back.get(expr).copied()
    }

    pub(crate) fn node_expr(&self, node: &ast::Expr) -> Option<ExprId> {
        self.expr_map.get(&Either::A(AstPtr::new(node))).cloned()
    }

    pub(crate) fn pat_syntax(&self, pat: PatId) -> Option<PatSource> {
        self.pat_map_back.get(pat).copied()
    }

    pub(crate) fn node_pat(&self, node: &ast::Pat) -> Option<PatId> {
        self.pat_map.get(&Either::A(AstPtr::new(node))).cloned()
    }

    pub(crate) fn field_syntax(&self, expr: ExprId, field: usize) -> AstPtr<ast::RecordField> {
        self.field_map[&(expr, field)]
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Literal {
    String(String),
    ByteString(Vec<u8>),
    Char(char),
    Bool(bool),
    Int(u64, UncertainIntTy),
    Float(u64, UncertainFloatTy), // FIXME: f64 is not Eq
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Expr {
    /// This is produced if syntax tree does not have a required expression piece.
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
    },
    Loop {
        body: ExprId,
    },
    While {
        condition: ExprId,
        body: ExprId,
    },
    For {
        iterable: ExprId,
        pat: PatId,
        body: ExprId,
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
    Continue,
    Break {
        expr: Option<ExprId>,
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
    Index {
        base: ExprId,
        index: ExprId,
    },
    Lambda {
        args: Vec<PatId>,
        arg_types: Vec<Option<TypeRef>>,
        body: ExprId,
    },
    Tuple {
        exprs: Vec<ExprId>,
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
    pub pats: Vec<PatId>,
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
            Expr::Block { statements, tail } => {
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
            Expr::TryBlock { body } => f(*body),
            Expr::Loop { body } => f(*body),
            Expr::While { condition, body } => {
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
            Expr::Continue => {}
            Expr::Break { expr } | Expr::Return { expr } => {
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
    fn new(is_mutable: bool, is_ref: bool) -> Self {
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
    pub(crate) name: Name,
    pub(crate) pat: PatId,
}

/// Close relative to rustc's hir::PatKind
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Pat {
    Missing,
    Wild,
    Tuple(Vec<PatId>),
    Record {
        path: Option<Path>,
        args: Vec<RecordFieldPat>,
        // FIXME: 'ellipsis' option
    },
    Range {
        start: ExprId,
        end: ExprId,
    },
    Slice {
        prefix: Vec<PatId>,
        rest: Option<PatId>,
        suffix: Vec<PatId>,
    },
    Path(Path),
    Lit(ExprId),
    Bind {
        mode: BindingAnnotation,
        name: Name,
        subpat: Option<PatId>,
    },
    TupleStruct {
        path: Option<Path>,
        args: Vec<PatId>,
    },
    Ref {
        pat: PatId,
        mutability: Mutability,
    },
}

impl Pat {
    pub fn walk_child_pats(&self, mut f: impl FnMut(PatId)) {
        match self {
            Pat::Range { .. } | Pat::Lit(..) | Pat::Path(..) | Pat::Wild | Pat::Missing => {}
            Pat::Bind { subpat, .. } => {
                subpat.iter().copied().for_each(f);
            }
            Pat::Tuple(args) | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, rest, suffix } => {
                let total_iter = prefix.iter().chain(rest.iter()).chain(suffix.iter());
                total_iter.copied().for_each(f);
            }
            Pat::Record { args, .. } => {
                args.iter().map(|f| f.pat).for_each(f);
            }
        }
    }
}

// Queries
pub(crate) fn body_with_source_map_query(
    db: &impl HirDatabase,
    def: DefWithBody,
) -> (Arc<Body>, Arc<BodySourceMap>) {
    let mut params = None;

    let (file_id, body) = match def {
        DefWithBody::Function(f) => {
            let src = f.source(db);
            params = src.ast.param_list();
            (src.file_id, src.ast.body().map(ast::Expr::from))
        }
        DefWithBody::Const(c) => {
            let src = c.source(db);
            (src.file_id, src.ast.body())
        }
        DefWithBody::Static(s) => {
            let src = s.source(db);
            (src.file_id, src.ast.body())
        }
    };

    let (body, source_map) = lower::lower(db, def.resolver(db), file_id, def, params, body);
    (Arc::new(body), Arc::new(source_map))
}

pub(crate) fn body_hir_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<Body> {
    db.body_with_source_map(def).0
}
