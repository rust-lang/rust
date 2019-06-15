//! The MIR is built from some high-level abstract IR
//! (HAIR). This section defines the HAIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

use rustc::mir::{BinOp, BorrowKind, Field, UnOp};
use rustc::hir::def_id::DefId;
use rustc::infer::canonical::Canonical;
use rustc::middle::region;
use rustc::ty::subst::SubstsRef;
use rustc::ty::{AdtDef, UpvarSubsts, Ty, Const, UserType};
use rustc::ty::adjustment::{PointerCast};
use rustc::ty::layout::VariantIdx;
use rustc::hir;
use syntax_pos::Span;
use self::cx::Cx;

pub mod cx;
mod constant;

pub mod pattern;
pub use self::pattern::{BindingMode, Pattern, PatternKind, PatternRange, FieldPattern};
pub(crate) use self::pattern::PatternTypeProjection;

mod util;

#[derive(Copy, Clone, Debug)]
pub enum LintLevel {
    Inherited,
    Explicit(hir::HirId)
}

#[derive(Clone, Debug)]
pub struct Block<'tcx> {
    pub targeted_by_break: bool,
    pub region_scope: region::Scope,
    pub opt_destruction_scope: Option<region::Scope>,
    pub span: Span,
    pub stmts: Vec<StmtRef<'tcx>>,
    pub expr: Option<ExprRef<'tcx>>,
    pub safety_mode: BlockSafety,
}

#[derive(Copy, Clone, Debug)]
pub enum BlockSafety {
    Safe,
    ExplicitUnsafe(hir::HirId),
    PushUnsafe,
    PopUnsafe
}

#[derive(Clone, Debug)]
pub enum StmtRef<'tcx> {
    Mirror(Box<Stmt<'tcx>>),
}

#[derive(Clone, Debug)]
pub struct Stmt<'tcx> {
    pub kind: StmtKind<'tcx>,
    pub opt_destruction_scope: Option<region::Scope>,
}

#[derive(Clone, Debug)]
pub enum StmtKind<'tcx> {
    Expr {
        /// scope for this statement; may be used as lifetime of temporaries
        scope: region::Scope,

        /// expression being evaluated in this statement
        expr: ExprRef<'tcx>,
    },

    Let {
        /// scope for variables bound in this let; covers this and
        /// remaining statements in block
        remainder_scope: region::Scope,

        /// scope for the initialization itself; might be used as
        /// lifetime of temporaries
        init_scope: region::Scope,

        /// `let <PAT> = ...`
        ///
        /// if a type is included, it is added as an ascription pattern
        pattern: Pattern<'tcx>,

        /// let pat: ty = <INIT> ...
        initializer: Option<ExprRef<'tcx>>,

        /// the lint level for this let-statement
        lint_level: LintLevel,
    },
}

/// The Hair trait implementor lowers their expressions (`&'tcx H::Expr`)
/// into instances of this `Expr` enum. This lowering can be done
/// basically as lazily or as eagerly as desired: every recursive
/// reference to an expression in this enum is an `ExprRef<'tcx>`, which
/// may in turn be another instance of this enum (boxed), or else an
/// unlowered `&'tcx H::Expr`. Note that instances of `Expr` are very
/// short-lived. They are created by `Hair::to_expr`, analyzed and
/// converted into MIR, and then discarded.
///
/// If you compare `Expr` to the full compiler AST, you will see it is
/// a good bit simpler. In fact, a number of the more straight-forward
/// MIR simplifications are already done in the impl of `Hair`. For
/// example, method calls and overloaded operators are absent: they are
/// expected to be converted into `Expr::Call` instances.
#[derive(Clone, Debug)]
pub struct Expr<'tcx> {
    /// type of this expression
    pub ty: Ty<'tcx>,

    /// lifetime of this expression if it should be spilled into a
    /// temporary; should be None only if in a constant context
    pub temp_lifetime: Option<region::Scope>,

    /// span of the expression in the source
    pub span: Span,

    /// kind of expression
    pub kind: ExprKind<'tcx>,
}

#[derive(Clone, Debug)]
pub enum ExprKind<'tcx> {
    Scope {
        region_scope: region::Scope,
        lint_level: LintLevel,
        value: ExprRef<'tcx>,
    },
    Box {
        value: ExprRef<'tcx>,
    },
    Call {
        ty: Ty<'tcx>,
        fun: ExprRef<'tcx>,
        args: Vec<ExprRef<'tcx>>,
        // Whether this is from a call in HIR, rather than from an overloaded
        // operator. True for overloaded function call.
        from_hir_call: bool,
    },
    Deref {
        arg: ExprRef<'tcx>,
    }, // NOT overloaded!
    Binary {
        op: BinOp,
        lhs: ExprRef<'tcx>,
        rhs: ExprRef<'tcx>,
    }, // NOT overloaded!
    LogicalOp {
        op: LogicalOp,
        lhs: ExprRef<'tcx>,
        rhs: ExprRef<'tcx>,
    }, // NOT overloaded!
       // LogicalOp is distinct from BinaryOp because of lazy evaluation of the operands.
    Unary {
        op: UnOp,
        arg: ExprRef<'tcx>,
    }, // NOT overloaded!
    Cast {
        source: ExprRef<'tcx>,
    },
    Use {
        source: ExprRef<'tcx>,
    }, // Use a lexpr to get a vexpr.
    NeverToAny {
        source: ExprRef<'tcx>,
    },
    Pointer {
        cast: PointerCast,
        source: ExprRef<'tcx>,
    },
    Loop {
        condition: Option<ExprRef<'tcx>>,
        body: ExprRef<'tcx>,
    },
    Match {
        scrutinee: ExprRef<'tcx>,
        arms: Vec<Arm<'tcx>>,
    },
    Block {
        body: &'tcx hir::Block,
    },
    Assign {
        lhs: ExprRef<'tcx>,
        rhs: ExprRef<'tcx>,
    },
    AssignOp {
        op: BinOp,
        lhs: ExprRef<'tcx>,
        rhs: ExprRef<'tcx>,
    },
    Field {
        lhs: ExprRef<'tcx>,
        name: Field,
    },
    Index {
        lhs: ExprRef<'tcx>,
        index: ExprRef<'tcx>,
    },
    VarRef {
        id: hir::HirId,
    },
    /// first argument, used for self in a closure
    SelfRef,
    StaticRef {
        id: DefId,
    },
    Borrow {
        borrow_kind: BorrowKind,
        arg: ExprRef<'tcx>,
    },
    Break {
        label: region::Scope,
        value: Option<ExprRef<'tcx>>,
    },
    Continue {
        label: region::Scope,
    },
    Return {
        value: Option<ExprRef<'tcx>>,
    },
    Repeat {
        value: ExprRef<'tcx>,
        count: u64,
    },
    Array {
        fields: Vec<ExprRef<'tcx>>,
    },
    Tuple {
        fields: Vec<ExprRef<'tcx>>,
    },
    Adt {
        adt_def: &'tcx AdtDef,
        variant_index: VariantIdx,
        substs: SubstsRef<'tcx>,

        /// Optional user-given substs: for something like `let x =
        /// Bar::<T> { ... }`.
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,

        fields: Vec<FieldExprRef<'tcx>>,
        base: Option<FruInfo<'tcx>>
    },
    PlaceTypeAscription {
        source: ExprRef<'tcx>,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    ValueTypeAscription {
        source: ExprRef<'tcx>,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    Closure {
        closure_id: DefId,
        substs: UpvarSubsts<'tcx>,
        upvars: Vec<ExprRef<'tcx>>,
        movability: Option<hir::GeneratorMovability>,
    },
    Literal {
        literal: &'tcx Const<'tcx>,
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    InlineAsm {
        asm: &'tcx hir::InlineAsm,
        outputs: Vec<ExprRef<'tcx>>,
        inputs: Vec<ExprRef<'tcx>>
    },
    Yield {
        value: ExprRef<'tcx>,
    },
}

#[derive(Clone, Debug)]
pub enum ExprRef<'tcx> {
    Hair(&'tcx hir::Expr),
    Mirror(Box<Expr<'tcx>>),
}

#[derive(Clone, Debug)]
pub struct FieldExprRef<'tcx> {
    pub name: Field,
    pub expr: ExprRef<'tcx>,
}

#[derive(Clone, Debug)]
pub struct FruInfo<'tcx> {
    pub base: ExprRef<'tcx>,
    pub field_types: Vec<Ty<'tcx>>
}

#[derive(Clone, Debug)]
pub struct Arm<'tcx> {
    pub patterns: Vec<Pattern<'tcx>>,
    pub guard: Option<Guard<'tcx>>,
    pub body: ExprRef<'tcx>,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Guard<'tcx> {
    If(ExprRef<'tcx>),
}

#[derive(Copy, Clone, Debug)]
pub enum LogicalOp {
    And,
    Or,
}

impl<'tcx> ExprRef<'tcx> {
    pub fn span(&self) -> Span {
        match self {
            ExprRef::Hair(expr) => expr.span,
            ExprRef::Mirror(expr) => expr.span,
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// The Mirror trait

/// "Mirroring" is the process of converting from a HIR type into one
/// of the HAIR types defined in this file. This is basically a "on
/// the fly" desugaring step that hides a lot of the messiness in the
/// tcx. For example, the mirror of a `&'tcx hir::Expr` is an
/// `Expr<'tcx>`.
///
/// Mirroring is gradual: when you mirror an outer expression like `e1
/// + e2`, the references to the inner expressions `e1` and `e2` are
/// `ExprRef<'tcx>` instances, and they may or may not be eagerly
/// mirrored. This allows a single AST node from the compiler to
/// expand into one or more Hair nodes, which lets the Hair nodes be
/// simpler.
pub trait Mirror<'tcx> {
    type Output;

    fn make_mirror(self, cx: &mut Cx<'_, 'tcx>) -> Self::Output;
}

impl<'tcx> Mirror<'tcx> for Expr<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror(self, _: &mut Cx<'_, 'tcx>) -> Expr<'tcx> {
        self
    }
}

impl<'tcx> Mirror<'tcx> for ExprRef<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror(self, hir: &mut Cx<'a, 'tcx>) -> Expr<'tcx> {
        match self {
            ExprRef::Hair(h) => h.make_mirror(hir),
            ExprRef::Mirror(m) => *m,
        }
    }
}

impl<'tcx> Mirror<'tcx> for Stmt<'tcx> {
    type Output = Stmt<'tcx>;

    fn make_mirror(self, _: &mut Cx<'_, 'tcx>) -> Stmt<'tcx> {
        self
    }
}

impl<'tcx> Mirror<'tcx> for StmtRef<'tcx> {
    type Output = Stmt<'tcx>;

    fn make_mirror(self, _: &mut Cx<'_, 'tcx>) -> Stmt<'tcx> {
        match self {
            StmtRef::Mirror(m) => *m,
        }
    }
}

impl<'tcx> Mirror<'tcx> for Block<'tcx> {
    type Output = Block<'tcx>;

    fn make_mirror(self, _: &mut Cx<'_, 'tcx>) -> Block<'tcx> {
        self
    }
}
