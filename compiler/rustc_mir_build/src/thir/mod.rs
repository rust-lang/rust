//! The MIR is built from some typed high-level IR
//! (THIR). This section defines the THIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::infer::canonical::Canonical;
use rustc_middle::middle::region;
use rustc_middle::mir::{BinOp, BorrowKind, FakeReadCause, Field, UnOp};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{AdtDef, Const, Ty, UpvarSubsts, UserType};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;
use rustc_target::asm::InlineAsmRegOrRegClass;

crate mod constant;

crate mod cx;
pub use cx::build_thir;

crate mod pattern;
pub use self::pattern::{Ascription, BindingMode, FieldPat, Pat, PatKind, PatRange, PatTyProj};

mod arena;
pub use arena::Arena;

mod util;

#[derive(Copy, Clone, Debug)]
pub enum LintLevel {
    Inherited,
    Explicit(hir::HirId),
}

#[derive(Debug)]
pub struct Block<'thir, 'tcx> {
    pub targeted_by_break: bool,
    pub region_scope: region::Scope,
    pub opt_destruction_scope: Option<region::Scope>,
    pub span: Span,
    pub stmts: &'thir [Stmt<'thir, 'tcx>],
    pub expr: Option<&'thir Expr<'thir, 'tcx>>,
    pub safety_mode: BlockSafety,
}

#[derive(Copy, Clone, Debug)]
pub enum BlockSafety {
    Safe,
    ExplicitUnsafe(hir::HirId),
    PushUnsafe,
    PopUnsafe,
}

#[derive(Debug)]
pub struct Stmt<'thir, 'tcx> {
    pub kind: StmtKind<'thir, 'tcx>,
    pub opt_destruction_scope: Option<region::Scope>,
}

#[derive(Debug)]
pub enum StmtKind<'thir, 'tcx> {
    Expr {
        /// scope for this statement; may be used as lifetime of temporaries
        scope: region::Scope,

        /// expression being evaluated in this statement
        expr: &'thir Expr<'thir, 'tcx>,
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
        pattern: Pat<'tcx>,

        /// let pat: ty = <INIT> ...
        initializer: Option<&'thir Expr<'thir, 'tcx>>,

        /// the lint level for this let-statement
        lint_level: LintLevel,
    },
}

// `Expr` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Expr<'_, '_>, 144);

/// The Thir trait implementor lowers their expressions (`&'tcx H::Expr`)
/// into instances of this `Expr` enum. This lowering can be done
/// basically as lazily or as eagerly as desired: every recursive
/// reference to an expression in this enum is an `&'thir Expr<'thir, 'tcx>`, which
/// may in turn be another instance of this enum (boxed), or else an
/// unlowered `&'tcx H::Expr`. Note that instances of `Expr` are very
/// short-lived. They are created by `Thir::to_expr`, analyzed and
/// converted into MIR, and then discarded.
///
/// If you compare `Expr` to the full compiler AST, you will see it is
/// a good bit simpler. In fact, a number of the more straight-forward
/// MIR simplifications are already done in the impl of `Thir`. For
/// example, method calls and overloaded operators are absent: they are
/// expected to be converted into `Expr::Call` instances.
#[derive(Debug)]
pub struct Expr<'thir, 'tcx> {
    /// type of this expression
    pub ty: Ty<'tcx>,

    /// lifetime of this expression if it should be spilled into a
    /// temporary; should be None only if in a constant context
    pub temp_lifetime: Option<region::Scope>,

    /// span of the expression in the source
    pub span: Span,

    /// kind of expression
    pub kind: ExprKind<'thir, 'tcx>,
}

#[derive(Debug)]
pub enum ExprKind<'thir, 'tcx> {
    Scope {
        region_scope: region::Scope,
        lint_level: LintLevel,
        value: &'thir Expr<'thir, 'tcx>,
    },
    Box {
        value: &'thir Expr<'thir, 'tcx>,
    },
    If {
        cond: &'thir Expr<'thir, 'tcx>,
        then: &'thir Expr<'thir, 'tcx>,
        else_opt: Option<&'thir Expr<'thir, 'tcx>>,
    },
    Call {
        ty: Ty<'tcx>,
        fun: &'thir Expr<'thir, 'tcx>,
        args: &'thir [Expr<'thir, 'tcx>],
        /// Whether this is from a call in HIR, rather than from an overloaded
        /// operator. `true` for overloaded function call.
        from_hir_call: bool,
        /// This `Span` is the span of the function, without the dot and receiver
        /// (e.g. `foo(a, b)` in `x.foo(a, b)`
        fn_span: Span,
    },
    Deref {
        arg: &'thir Expr<'thir, 'tcx>,
    }, // NOT overloaded!
    Binary {
        op: BinOp,
        lhs: &'thir Expr<'thir, 'tcx>,
        rhs: &'thir Expr<'thir, 'tcx>,
    }, // NOT overloaded!
    LogicalOp {
        op: LogicalOp,
        lhs: &'thir Expr<'thir, 'tcx>,
        rhs: &'thir Expr<'thir, 'tcx>,
    }, // NOT overloaded!
    // LogicalOp is distinct from BinaryOp because of lazy evaluation of the operands.
    Unary {
        op: UnOp,
        arg: &'thir Expr<'thir, 'tcx>,
    }, // NOT overloaded!
    Cast {
        source: &'thir Expr<'thir, 'tcx>,
    },
    Use {
        source: &'thir Expr<'thir, 'tcx>,
    }, // Use a lexpr to get a vexpr.
    NeverToAny {
        source: &'thir Expr<'thir, 'tcx>,
    },
    Pointer {
        cast: PointerCast,
        source: &'thir Expr<'thir, 'tcx>,
    },
    Loop {
        body: &'thir Expr<'thir, 'tcx>,
    },
    Match {
        scrutinee: &'thir Expr<'thir, 'tcx>,
        arms: &'thir [Arm<'thir, 'tcx>],
    },
    Block {
        body: Block<'thir, 'tcx>,
    },
    Assign {
        lhs: &'thir Expr<'thir, 'tcx>,
        rhs: &'thir Expr<'thir, 'tcx>,
    },
    AssignOp {
        op: BinOp,
        lhs: &'thir Expr<'thir, 'tcx>,
        rhs: &'thir Expr<'thir, 'tcx>,
    },
    Field {
        lhs: &'thir Expr<'thir, 'tcx>,
        name: Field,
    },
    Index {
        lhs: &'thir Expr<'thir, 'tcx>,
        index: &'thir Expr<'thir, 'tcx>,
    },
    VarRef {
        id: hir::HirId,
    },
    /// Used to represent upvars mentioned in a closure/generator
    UpvarRef {
        /// DefId of the closure/generator
        closure_def_id: DefId,

        /// HirId of the root variable
        var_hir_id: hir::HirId,
    },
    Borrow {
        borrow_kind: BorrowKind,
        arg: &'thir Expr<'thir, 'tcx>,
    },
    /// A `&raw [const|mut] $place_expr` raw borrow resulting in type `*[const|mut] T`.
    AddressOf {
        mutability: hir::Mutability,
        arg: &'thir Expr<'thir, 'tcx>,
    },
    Break {
        label: region::Scope,
        value: Option<&'thir Expr<'thir, 'tcx>>,
    },
    Continue {
        label: region::Scope,
    },
    Return {
        value: Option<&'thir Expr<'thir, 'tcx>>,
    },
    ConstBlock {
        value: &'tcx Const<'tcx>,
    },
    Repeat {
        value: &'thir Expr<'thir, 'tcx>,
        count: &'tcx Const<'tcx>,
    },
    Array {
        fields: &'thir [Expr<'thir, 'tcx>],
    },
    Tuple {
        fields: &'thir [Expr<'thir, 'tcx>],
    },
    Adt {
        adt_def: &'tcx AdtDef,
        variant_index: VariantIdx,
        substs: SubstsRef<'tcx>,

        /// Optional user-given substs: for something like `let x =
        /// Bar::<T> { ... }`.
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,

        fields: &'thir [FieldExpr<'thir, 'tcx>],
        base: Option<FruInfo<'thir, 'tcx>>,
    },
    PlaceTypeAscription {
        source: &'thir Expr<'thir, 'tcx>,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    ValueTypeAscription {
        source: &'thir Expr<'thir, 'tcx>,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    Closure {
        closure_id: DefId,
        substs: UpvarSubsts<'tcx>,
        upvars: &'thir [Expr<'thir, 'tcx>],
        movability: Option<hir::Movability>,
        fake_reads: Vec<(&'thir Expr<'thir, 'tcx>, FakeReadCause, hir::HirId)>,
    },
    Literal {
        literal: &'tcx Const<'tcx>,
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
        /// The `DefId` of the `const` item this literal
        /// was produced from, if this is not a user-written
        /// literal value.
        const_id: Option<DefId>,
    },
    /// A literal containing the address of a `static`.
    ///
    /// This is only distinguished from `Literal` so that we can register some
    /// info for diagnostics.
    StaticRef {
        literal: &'tcx Const<'tcx>,
        def_id: DefId,
    },
    InlineAsm {
        template: &'tcx [InlineAsmTemplatePiece],
        operands: &'thir [InlineAsmOperand<'thir, 'tcx>],
        options: InlineAsmOptions,
        line_spans: &'tcx [Span],
    },
    /// An expression taking a reference to a thread local.
    ThreadLocalRef(DefId),
    LlvmInlineAsm {
        asm: &'tcx hir::LlvmInlineAsmInner,
        outputs: &'thir [Expr<'thir, 'tcx>],
        inputs: &'thir [Expr<'thir, 'tcx>],
    },
    Yield {
        value: &'thir Expr<'thir, 'tcx>,
    },
}

#[derive(Debug)]
pub struct FieldExpr<'thir, 'tcx> {
    pub name: Field,
    pub expr: &'thir Expr<'thir, 'tcx>,
}

#[derive(Debug)]
pub struct FruInfo<'thir, 'tcx> {
    pub base: &'thir Expr<'thir, 'tcx>,
    pub field_types: &'thir [Ty<'tcx>],
}

#[derive(Debug)]
pub struct Arm<'thir, 'tcx> {
    pub pattern: Pat<'tcx>,
    pub guard: Option<Guard<'thir, 'tcx>>,
    pub body: &'thir Expr<'thir, 'tcx>,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
}

#[derive(Debug)]
pub enum Guard<'thir, 'tcx> {
    If(&'thir Expr<'thir, 'tcx>),
    IfLet(Pat<'tcx>, &'thir Expr<'thir, 'tcx>),
}

#[derive(Copy, Clone, Debug)]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Debug)]
pub enum InlineAsmOperand<'thir, 'tcx> {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: &'thir Expr<'thir, 'tcx>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<&'thir Expr<'thir, 'tcx>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: &'thir Expr<'thir, 'tcx>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: &'thir Expr<'thir, 'tcx>,
        out_expr: Option<&'thir Expr<'thir, 'tcx>>,
    },
    Const {
        expr: &'thir Expr<'thir, 'tcx>,
    },
    SymFn {
        expr: &'thir Expr<'thir, 'tcx>,
    },
    SymStatic {
        def_id: DefId,
    },
}
