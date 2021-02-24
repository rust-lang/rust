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
use rustc_middle::mir::{BinOp, BorrowKind, Field, UnOp};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{AdtDef, Const, Ty, UpvarSubsts, UserType};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;
use rustc_target::asm::InlineAsmRegOrRegClass;

crate mod constant;
crate mod cx;

crate mod pattern;
crate use self::pattern::PatTyProj;
crate use self::pattern::{BindingMode, FieldPat, Pat, PatKind, PatRange};

mod util;

#[derive(Copy, Clone, Debug)]
crate enum LintLevel {
    Inherited,
    Explicit(hir::HirId),
}

#[derive(Clone, Debug)]
crate struct Block<'tcx> {
    crate targeted_by_break: bool,
    crate region_scope: region::Scope,
    crate opt_destruction_scope: Option<region::Scope>,
    crate span: Span,
    crate stmts: Vec<Stmt<'tcx>>,
    crate expr: Option<Box<Expr<'tcx>>>,
    crate safety_mode: BlockSafety,
}

#[derive(Copy, Clone, Debug)]
crate enum BlockSafety {
    Safe,
    ExplicitUnsafe(hir::HirId),
    PushUnsafe,
    PopUnsafe,
}

#[derive(Clone, Debug)]
crate struct Stmt<'tcx> {
    crate kind: StmtKind<'tcx>,
    crate opt_destruction_scope: Option<region::Scope>,
}

#[derive(Clone, Debug)]
crate enum StmtKind<'tcx> {
    Expr {
        /// scope for this statement; may be used as lifetime of temporaries
        scope: region::Scope,

        /// expression being evaluated in this statement
        expr: Box<Expr<'tcx>>,
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
        initializer: Option<Box<Expr<'tcx>>>,

        /// the lint level for this let-statement
        lint_level: LintLevel,
    },
}

// `Expr` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Expr<'_>, 160);

/// The Thir trait implementor lowers their expressions (`&'tcx H::Expr`)
/// into instances of this `Expr` enum. This lowering can be done
/// basically as lazily or as eagerly as desired: every recursive
/// reference to an expression in this enum is an `Box<Expr<'tcx>>`, which
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
#[derive(Clone, Debug)]
crate struct Expr<'tcx> {
    /// type of this expression
    crate ty: Ty<'tcx>,

    /// lifetime of this expression if it should be spilled into a
    /// temporary; should be None only if in a constant context
    crate temp_lifetime: Option<region::Scope>,

    /// span of the expression in the source
    crate span: Span,

    /// kind of expression
    crate kind: ExprKind<'tcx>,
}

#[derive(Clone, Debug)]
crate enum ExprKind<'tcx> {
    Scope {
        region_scope: region::Scope,
        lint_level: LintLevel,
        value: Box<Expr<'tcx>>,
    },
    Box {
        value: Box<Expr<'tcx>>,
    },
    If {
        cond: Box<Expr<'tcx>>,
        then: Box<Expr<'tcx>>,
        else_opt: Option<Box<Expr<'tcx>>>,
    },
    Call {
        ty: Ty<'tcx>,
        fun: Box<Expr<'tcx>>,
        args: Vec<Expr<'tcx>>,
        /// Whether this is from a call in HIR, rather than from an overloaded
        /// operator. `true` for overloaded function call.
        from_hir_call: bool,
        /// This `Span` is the span of the function, without the dot and receiver
        /// (e.g. `foo(a, b)` in `x.foo(a, b)`
        fn_span: Span,
    },
    Deref {
        arg: Box<Expr<'tcx>>,
    }, // NOT overloaded!
    Binary {
        op: BinOp,
        lhs: Box<Expr<'tcx>>,
        rhs: Box<Expr<'tcx>>,
    }, // NOT overloaded!
    LogicalOp {
        op: LogicalOp,
        lhs: Box<Expr<'tcx>>,
        rhs: Box<Expr<'tcx>>,
    }, // NOT overloaded!
    // LogicalOp is distinct from BinaryOp because of lazy evaluation of the operands.
    Unary {
        op: UnOp,
        arg: Box<Expr<'tcx>>,
    }, // NOT overloaded!
    Cast {
        source: Box<Expr<'tcx>>,
    },
    Use {
        source: Box<Expr<'tcx>>,
    }, // Use a lexpr to get a vexpr.
    NeverToAny {
        source: Box<Expr<'tcx>>,
    },
    Pointer {
        cast: PointerCast,
        source: Box<Expr<'tcx>>,
    },
    Loop {
        body: Box<Expr<'tcx>>,
    },
    Match {
        scrutinee: Box<Expr<'tcx>>,
        arms: Vec<Arm<'tcx>>,
    },
    Block {
        body: Block<'tcx>,
    },
    Assign {
        lhs: Box<Expr<'tcx>>,
        rhs: Box<Expr<'tcx>>,
    },
    AssignOp {
        op: BinOp,
        lhs: Box<Expr<'tcx>>,
        rhs: Box<Expr<'tcx>>,
    },
    Field {
        lhs: Box<Expr<'tcx>>,
        name: Field,
    },
    Index {
        lhs: Box<Expr<'tcx>>,
        index: Box<Expr<'tcx>>,
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
        arg: Box<Expr<'tcx>>,
    },
    /// A `&raw [const|mut] $place_expr` raw borrow resulting in type `*[const|mut] T`.
    AddressOf {
        mutability: hir::Mutability,
        arg: Box<Expr<'tcx>>,
    },
    Break {
        label: region::Scope,
        value: Option<Box<Expr<'tcx>>>,
    },
    Continue {
        label: region::Scope,
    },
    Return {
        value: Option<Box<Expr<'tcx>>>,
    },
    ConstBlock {
        value: &'tcx Const<'tcx>,
    },
    Repeat {
        value: Box<Expr<'tcx>>,
        count: &'tcx Const<'tcx>,
    },
    Array {
        fields: Vec<Expr<'tcx>>,
    },
    Tuple {
        fields: Vec<Expr<'tcx>>,
    },
    Adt {
        adt_def: &'tcx AdtDef,
        variant_index: VariantIdx,
        substs: SubstsRef<'tcx>,

        /// Optional user-given substs: for something like `let x =
        /// Bar::<T> { ... }`.
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,

        fields: Vec<FieldExpr<'tcx>>,
        base: Option<FruInfo<'tcx>>,
    },
    PlaceTypeAscription {
        source: Box<Expr<'tcx>>,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    ValueTypeAscription {
        source: Box<Expr<'tcx>>,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    Closure {
        closure_id: DefId,
        substs: UpvarSubsts<'tcx>,
        upvars: Vec<Expr<'tcx>>,
        movability: Option<hir::Movability>,
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
        operands: Vec<InlineAsmOperand<'tcx>>,
        options: InlineAsmOptions,
        line_spans: &'tcx [Span],
    },
    /// An expression taking a reference to a thread local.
    ThreadLocalRef(DefId),
    LlvmInlineAsm {
        asm: &'tcx hir::LlvmInlineAsmInner,
        outputs: Vec<Expr<'tcx>>,
        inputs: Vec<Expr<'tcx>>,
    },
    Yield {
        value: Box<Expr<'tcx>>,
    },
}

#[derive(Clone, Debug)]
crate struct FieldExpr<'tcx> {
    crate name: Field,
    crate expr: Expr<'tcx>,
}

#[derive(Clone, Debug)]
crate struct FruInfo<'tcx> {
    crate base: Box<Expr<'tcx>>,
    crate field_types: Vec<Ty<'tcx>>,
}

#[derive(Clone, Debug)]
crate struct Arm<'tcx> {
    crate pattern: Pat<'tcx>,
    crate guard: Option<Guard<'tcx>>,
    crate body: Expr<'tcx>,
    crate lint_level: LintLevel,
    crate scope: region::Scope,
    crate span: Span,
}

#[derive(Clone, Debug)]
crate enum Guard<'tcx> {
    If(Box<Expr<'tcx>>),
    IfLet(Pat<'tcx>, Box<Expr<'tcx>>),
}

#[derive(Copy, Clone, Debug)]
crate enum LogicalOp {
    And,
    Or,
}

#[derive(Clone, Debug)]
crate enum InlineAsmOperand<'tcx> {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: Expr<'tcx>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<Expr<'tcx>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Expr<'tcx>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: Expr<'tcx>,
        out_expr: Option<Expr<'tcx>>,
    },
    Const {
        expr: Expr<'tcx>,
    },
    SymFn {
        expr: Expr<'tcx>,
    },
    SymStatic {
        def_id: DefId,
    },
}
