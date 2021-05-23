//! The MIR is built from some typed high-level IR
//! (THIR). This section defines the THIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::newtype_index;
use rustc_index::vec::IndexVec;
use rustc_middle::infer::canonical::Canonical;
use rustc_middle::middle::region;
use rustc_middle::mir::{BinOp, BorrowKind, FakeReadCause, Field, UnOp};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{AdtDef, Const, Ty, UpvarSubsts, UserType};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;
use rustc_target::asm::InlineAsmRegOrRegClass;

use std::ops::Index;

crate mod constant;

crate mod cx;
pub use cx::build_thir;

crate mod pattern;
pub use self::pattern::{Ascription, BindingMode, FieldPat, Pat, PatKind, PatRange, PatTyProj};

mod util;
pub mod visit;

newtype_index! {
    pub struct ArmId {
        DEBUG_FORMAT = "a{}"
    }
}

newtype_index! {
    pub struct ExprId {
        DEBUG_FORMAT = "e{}"
    }
}

newtype_index! {
    pub struct StmtId {
        DEBUG_FORMAT = "s{}"
    }
}

macro_rules! thir_with_elements {
    ($($name:ident: $id:ty => $value:ty,)*) => {
        pub struct Thir<'tcx> {
            $(
                $name: IndexVec<$id, $value>,
            )*
        }

        impl<'tcx> Thir<'tcx> {
            fn new() -> Thir<'tcx> {
                Thir {
                    $(
                        $name: IndexVec::new(),
                    )*
                }
            }
        }

        $(
            impl<'tcx> Index<$id> for Thir<'tcx> {
                type Output = $value;
                fn index(&self, index: $id) -> &Self::Output {
                    &self.$name[index]
                }
            }
        )*
    }
}

thir_with_elements! {
    arms: ArmId => Arm<'tcx>,
    exprs: ExprId => Expr<'tcx>,
    stmts: StmtId => Stmt<'tcx>,
}

#[derive(Copy, Clone, Debug)]
pub enum LintLevel {
    Inherited,
    Explicit(hir::HirId),
}

#[derive(Debug)]
pub struct Block {
    pub targeted_by_break: bool,
    pub region_scope: region::Scope,
    pub opt_destruction_scope: Option<region::Scope>,
    pub span: Span,
    pub stmts: Box<[StmtId]>,
    pub expr: Option<ExprId>,
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
pub struct Stmt<'tcx> {
    pub kind: StmtKind<'tcx>,
    pub opt_destruction_scope: Option<region::Scope>,
}

#[derive(Debug)]
pub enum StmtKind<'tcx> {
    Expr {
        /// scope for this statement; may be used as lifetime of temporaries
        scope: region::Scope,

        /// expression being evaluated in this statement
        expr: ExprId,
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
        initializer: Option<ExprId>,

        /// the lint level for this let-statement
        lint_level: LintLevel,
    },
}

// `Expr` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Expr<'_>, 144);

/// The Thir trait implementor lowers their expressions (`&'tcx H::Expr`)
/// into instances of this `Expr` enum. This lowering can be done
/// basically as lazily or as eagerly as desired: every recursive
/// reference to an expression in this enum is an `ExprId`, which
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

#[derive(Debug)]
pub enum ExprKind<'tcx> {
    Scope {
        region_scope: region::Scope,
        lint_level: LintLevel,
        value: ExprId,
    },
    Box {
        value: ExprId,
    },
    If {
        cond: ExprId,
        then: ExprId,
        else_opt: Option<ExprId>,
    },
    Call {
        ty: Ty<'tcx>,
        fun: ExprId,
        args: Box<[ExprId]>,
        /// Whether this is from a call in HIR, rather than from an overloaded
        /// operator. `true` for overloaded function call.
        from_hir_call: bool,
        /// This `Span` is the span of the function, without the dot and receiver
        /// (e.g. `foo(a, b)` in `x.foo(a, b)`
        fn_span: Span,
    },
    Deref {
        arg: ExprId,
    }, // NOT overloaded!
    Binary {
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    }, // NOT overloaded!
    LogicalOp {
        op: LogicalOp,
        lhs: ExprId,
        rhs: ExprId,
    }, // NOT overloaded!
    // LogicalOp is distinct from BinaryOp because of lazy evaluation of the operands.
    Unary {
        op: UnOp,
        arg: ExprId,
    }, // NOT overloaded!
    Cast {
        source: ExprId,
    },
    Use {
        source: ExprId,
    }, // Use a lexpr to get a vexpr.
    NeverToAny {
        source: ExprId,
    },
    Pointer {
        cast: PointerCast,
        source: ExprId,
    },
    Loop {
        body: ExprId,
    },
    Match {
        scrutinee: ExprId,
        arms: Box<[ArmId]>,
    },
    Block {
        body: Block,
    },
    Assign {
        lhs: ExprId,
        rhs: ExprId,
    },
    AssignOp {
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    Field {
        lhs: ExprId,
        name: Field,
    },
    Index {
        lhs: ExprId,
        index: ExprId,
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
        arg: ExprId,
    },
    /// A `&raw [const|mut] $place_expr` raw borrow resulting in type `*[const|mut] T`.
    AddressOf {
        mutability: hir::Mutability,
        arg: ExprId,
    },
    Break {
        label: region::Scope,
        value: Option<ExprId>,
    },
    Continue {
        label: region::Scope,
    },
    Return {
        value: Option<ExprId>,
    },
    ConstBlock {
        value: &'tcx Const<'tcx>,
    },
    Repeat {
        value: ExprId,
        count: &'tcx Const<'tcx>,
    },
    Array {
        fields: Box<[ExprId]>,
    },
    Tuple {
        fields: Box<[ExprId]>,
    },
    Adt {
        adt_def: &'tcx AdtDef,
        variant_index: VariantIdx,
        substs: SubstsRef<'tcx>,

        /// Optional user-given substs: for something like `let x =
        /// Bar::<T> { ... }`.
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,

        fields: Box<[FieldExpr]>,
        base: Option<FruInfo<'tcx>>,
    },
    PlaceTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    ValueTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    Closure {
        closure_id: DefId,
        substs: UpvarSubsts<'tcx>,
        upvars: Box<[ExprId]>,
        movability: Option<hir::Movability>,
        fake_reads: Vec<(ExprId, FakeReadCause, hir::HirId)>,
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
        operands: Box<[InlineAsmOperand<'tcx>]>,
        options: InlineAsmOptions,
        line_spans: &'tcx [Span],
    },
    /// An expression taking a reference to a thread local.
    ThreadLocalRef(DefId),
    LlvmInlineAsm {
        asm: &'tcx hir::LlvmInlineAsmInner,
        outputs: Box<[ExprId]>,
        inputs: Box<[ExprId]>,
    },
    Yield {
        value: ExprId,
    },
}

#[derive(Debug)]
pub struct FieldExpr {
    pub name: Field,
    pub expr: ExprId,
}

#[derive(Debug)]
pub struct FruInfo<'tcx> {
    pub base: ExprId,
    pub field_types: Box<[Ty<'tcx>]>,
}

#[derive(Debug)]
pub struct Arm<'tcx> {
    pub pattern: Pat<'tcx>,
    pub guard: Option<Guard<'tcx>>,
    pub body: ExprId,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
}

#[derive(Debug)]
pub enum Guard<'tcx> {
    If(ExprId),
    IfLet(Pat<'tcx>, ExprId),
}

#[derive(Copy, Clone, Debug)]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Debug)]
pub enum InlineAsmOperand<'tcx> {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: ExprId,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<ExprId>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: ExprId,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: ExprId,
        out_expr: Option<ExprId>,
    },
    Const {
        value: &'tcx Const<'tcx>,
        span: Span,
    },
    SymFn {
        expr: ExprId,
    },
    SymStatic {
        def_id: DefId,
    },
}
