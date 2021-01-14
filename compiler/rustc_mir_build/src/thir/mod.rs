//! The MIR is built from some typed high-level IR
//! (THIR). This section defines the THIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

use self::cx::Cx;
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
    crate stmts: Vec<StmtRef<'tcx>>,
    crate expr: Option<ExprRef<'tcx>>,
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
crate enum StmtRef<'tcx> {
    Mirror(Box<Stmt<'tcx>>),
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
        pattern: Pat<'tcx>,

        /// let pat: ty = <INIT> ...
        initializer: Option<ExprRef<'tcx>>,

        /// the lint level for this let-statement
        lint_level: LintLevel,
    },
}

// `Expr` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
rustc_data_structures::static_assert_size!(Expr<'_>, 168);

/// The Thir trait implementor lowers their expressions (`&'tcx H::Expr`)
/// into instances of this `Expr` enum. This lowering can be done
/// basically as lazily or as eagerly as desired: every recursive
/// reference to an expression in this enum is an `ExprRef<'tcx>`, which
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
        value: ExprRef<'tcx>,
    },
    Box {
        value: ExprRef<'tcx>,
    },
    If {
        cond: ExprRef<'tcx>,
        then: ExprRef<'tcx>,
        else_opt: Option<ExprRef<'tcx>>,
    },
    Call {
        ty: Ty<'tcx>,
        fun: ExprRef<'tcx>,
        args: Vec<ExprRef<'tcx>>,
        // Whether this is from a call in HIR, rather than from an overloaded
        // operator. True for overloaded function call.
        from_hir_call: bool,
        /// This `Span` is the span of the function, without the dot and receiver
        /// (e.g. `foo(a, b)` in `x.foo(a, b)`
        fn_span: Span,
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
        body: ExprRef<'tcx>,
    },
    Match {
        scrutinee: ExprRef<'tcx>,
        arms: Vec<Arm<'tcx>>,
    },
    Block {
        body: &'tcx hir::Block<'tcx>,
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
    /// Used to represent upvars mentioned in a closure/generator
    UpvarRef {
        /// DefId of the closure/generator
        closure_def_id: DefId,

        /// HirId of the root variable
        var_hir_id: hir::HirId,
    },
    Borrow {
        borrow_kind: BorrowKind,
        arg: ExprRef<'tcx>,
    },
    /// A `&raw [const|mut] $place_expr` raw borrow resulting in type `*[const|mut] T`.
    AddressOf {
        mutability: hir::Mutability,
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
    ConstBlock {
        value: &'tcx Const<'tcx>,
    },
    Repeat {
        value: ExprRef<'tcx>,
        count: &'tcx Const<'tcx>,
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
        base: Option<FruInfo<'tcx>>,
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
        outputs: Vec<ExprRef<'tcx>>,
        inputs: Vec<ExprRef<'tcx>>,
    },
    Yield {
        value: ExprRef<'tcx>,
    },
}

#[derive(Clone, Debug)]
crate enum ExprRef<'tcx> {
    Thir(&'tcx hir::Expr<'tcx>),
    Mirror(Box<Expr<'tcx>>),
}

#[derive(Clone, Debug)]
crate struct FieldExprRef<'tcx> {
    crate name: Field,
    crate expr: ExprRef<'tcx>,
}

#[derive(Clone, Debug)]
crate struct FruInfo<'tcx> {
    crate base: ExprRef<'tcx>,
    crate field_types: Vec<Ty<'tcx>>,
}

#[derive(Clone, Debug)]
crate struct Arm<'tcx> {
    crate pattern: Pat<'tcx>,
    crate guard: Option<Guard<'tcx>>,
    crate body: ExprRef<'tcx>,
    crate lint_level: LintLevel,
    crate scope: region::Scope,
    crate span: Span,
}

#[derive(Clone, Debug)]
crate enum Guard<'tcx> {
    If(ExprRef<'tcx>),
    IfLet(Pat<'tcx>, ExprRef<'tcx>),
}

#[derive(Copy, Clone, Debug)]
crate enum LogicalOp {
    And,
    Or,
}

impl<'tcx> ExprRef<'tcx> {
    crate fn span(&self) -> Span {
        match self {
            ExprRef::Thir(expr) => expr.span,
            ExprRef::Mirror(expr) => expr.span,
        }
    }
}

#[derive(Clone, Debug)]
crate enum InlineAsmOperand<'tcx> {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: ExprRef<'tcx>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<ExprRef<'tcx>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: ExprRef<'tcx>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: ExprRef<'tcx>,
        out_expr: Option<ExprRef<'tcx>>,
    },
    Const {
        expr: ExprRef<'tcx>,
    },
    SymFn {
        expr: ExprRef<'tcx>,
    },
    SymStatic {
        def_id: DefId,
    },
}

///////////////////////////////////////////////////////////////////////////
// The Mirror trait

/// "Mirroring" is the process of converting from a HIR type into one
/// of the THIR types defined in this file. This is basically a "on
/// the fly" desugaring step that hides a lot of the messiness in the
/// tcx. For example, the mirror of a `&'tcx hir::Expr` is an
/// `Expr<'tcx>`.
///
/// Mirroring is gradual: when you mirror an outer expression like `e1
/// + e2`, the references to the inner expressions `e1` and `e2` are
/// `ExprRef<'tcx>` instances, and they may or may not be eagerly
/// mirrored. This allows a single AST node from the compiler to
/// expand into one or more Thir nodes, which lets the Thir nodes be
/// simpler.
crate trait Mirror<'tcx> {
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

    fn make_mirror(self, hir: &mut Cx<'_, 'tcx>) -> Expr<'tcx> {
        match self {
            ExprRef::Thir(h) => h.make_mirror(hir),
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
