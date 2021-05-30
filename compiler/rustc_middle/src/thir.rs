use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::DefId;
use rustc_hir::RangeEnd;
use rustc_index::newtype_index;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::infer::canonical::Canonical;
use rustc_middle::middle::region;
use rustc_middle::mir::{
    BinOp, BorrowKind, FakeReadCause, Field, Mutability, UnOp, UserTypeProjection,
};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, AdtDef, Const, Ty, UpvarSubsts, UserType};
use rustc_middle::ty::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations,
};
use rustc_span::{Span, Symbol, DUMMY_SP};
use rustc_target::abi::VariantIdx;
use rustc_target::asm::InlineAsmRegOrRegClass;

use std::fmt;
use std::ops::Index;

newtype_index! {
    #[derive(HashStable)]
    pub struct ArmId {
        DEBUG_FORMAT = "a{}"
    }
}

newtype_index! {
    #[derive(HashStable)]
    pub struct ExprId {
        DEBUG_FORMAT = "e{}"
    }
}

newtype_index! {
    #[derive(HashStable)]
    pub struct StmtId {
        DEBUG_FORMAT = "s{}"
    }
}

macro_rules! thir_with_elements {
    ($($name:ident: $id:ty => $value:ty,)*) => {
        #[derive(Debug, HashStable)]
        pub struct Thir<'tcx> {
            $(
                pub $name: IndexVec<$id, $value>,
            )*
        }

        impl<'tcx> Thir<'tcx> {
            pub fn new() -> Thir<'tcx> {
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

#[derive(Copy, Clone, Debug, HashStable)]
pub enum LintLevel {
    Inherited,
    Explicit(hir::HirId),
}

#[derive(Debug, HashStable)]
pub struct Block {
    pub targeted_by_break: bool,
    pub region_scope: region::Scope,
    pub opt_destruction_scope: Option<region::Scope>,
    pub span: Span,
    pub stmts: Box<[StmtId]>,
    pub expr: Option<ExprId>,
    pub safety_mode: BlockSafety,
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum BlockSafety {
    Safe,
    ExplicitUnsafe(hir::HirId),
    PushUnsafe,
    PopUnsafe,
}

#[derive(Debug, HashStable)]
pub struct Stmt<'tcx> {
    pub kind: StmtKind<'tcx>,
    pub opt_destruction_scope: Option<region::Scope>,
}

#[derive(Debug, HashStable)]
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
#[derive(Debug, HashStable)]
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

#[derive(Debug, HashStable)]
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

#[derive(Debug, HashStable)]
pub struct FieldExpr {
    pub name: Field,
    pub expr: ExprId,
}

#[derive(Debug, HashStable)]
pub struct FruInfo<'tcx> {
    pub base: ExprId,
    pub field_types: Box<[Ty<'tcx>]>,
}

#[derive(Debug, HashStable)]
pub struct Arm<'tcx> {
    pub pattern: Pat<'tcx>,
    pub guard: Option<Guard<'tcx>>,
    pub body: ExprId,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
}

#[derive(Debug, HashStable)]
pub enum Guard<'tcx> {
    If(ExprId),
    IfLet(Pat<'tcx>, ExprId),
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Debug, HashStable)]
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

#[derive(Copy, Clone, Debug, PartialEq, HashStable)]
pub enum BindingMode {
    ByValue,
    ByRef(BorrowKind),
}

#[derive(Clone, Debug, PartialEq, HashStable)]
pub struct FieldPat<'tcx> {
    pub field: Field,
    pub pattern: Pat<'tcx>,
}

#[derive(Clone, Debug, PartialEq, HashStable)]
pub struct Pat<'tcx> {
    pub ty: Ty<'tcx>,
    pub span: Span,
    pub kind: Box<PatKind<'tcx>>,
}

impl<'tcx> Pat<'tcx> {
    pub fn wildcard_from_ty(ty: Ty<'tcx>) -> Self {
        Pat { ty, span: DUMMY_SP, kind: Box::new(PatKind::Wild) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, HashStable)]
pub struct PatTyProj<'tcx> {
    pub user_ty: CanonicalUserType<'tcx>,
}

impl<'tcx> PatTyProj<'tcx> {
    pub fn from_user_type(user_annotation: CanonicalUserType<'tcx>) -> Self {
        Self { user_ty: user_annotation }
    }

    pub fn user_ty(
        self,
        annotations: &mut CanonicalUserTypeAnnotations<'tcx>,
        inferred_ty: Ty<'tcx>,
        span: Span,
    ) -> UserTypeProjection {
        UserTypeProjection {
            base: annotations.push(CanonicalUserTypeAnnotation {
                span,
                user_ty: self.user_ty,
                inferred_ty,
            }),
            projs: Vec::new(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, HashStable)]
pub struct Ascription<'tcx> {
    pub user_ty: PatTyProj<'tcx>,
    /// Variance to use when relating the type `user_ty` to the **type of the value being
    /// matched**. Typically, this is `Variance::Covariant`, since the value being matched must
    /// have a type that is some subtype of the ascribed type.
    ///
    /// Note that this variance does not apply for any bindings within subpatterns. The type
    /// assigned to those bindings must be exactly equal to the `user_ty` given here.
    ///
    /// The only place where this field is not `Covariant` is when matching constants, where
    /// we currently use `Contravariant` -- this is because the constant type just needs to
    /// be "comparable" to the type of the input value. So, for example:
    ///
    /// ```text
    /// match x { "foo" => .. }
    /// ```
    ///
    /// requires that `&'static str <: T_x`, where `T_x` is the type of `x`. Really, we should
    /// probably be checking for a `PartialEq` impl instead, but this preserves the behavior
    /// of the old type-check for now. See #57280 for details.
    pub variance: ty::Variance,
    pub user_ty_span: Span,
}

#[derive(Clone, Debug, PartialEq, HashStable)]
pub enum PatKind<'tcx> {
    Wild,

    AscribeUserType {
        ascription: Ascription<'tcx>,
        subpattern: Pat<'tcx>,
    },

    /// `x`, `ref x`, `x @ P`, etc.
    Binding {
        mutability: Mutability,
        name: Symbol,
        mode: BindingMode,
        var: hir::HirId,
        ty: Ty<'tcx>,
        subpattern: Option<Pat<'tcx>>,
        /// Is this the leftmost occurrence of the binding, i.e., is `var` the
        /// `HirId` of this pattern?
        is_primary: bool,
    },

    /// `Foo(...)` or `Foo{...}` or `Foo`, where `Foo` is a variant name from an ADT with
    /// multiple variants.
    Variant {
        adt_def: &'tcx AdtDef,
        substs: SubstsRef<'tcx>,
        variant_index: VariantIdx,
        subpatterns: Vec<FieldPat<'tcx>>,
    },

    /// `(...)`, `Foo(...)`, `Foo{...}`, or `Foo`, where `Foo` is a variant name from an ADT with
    /// a single variant.
    Leaf {
        subpatterns: Vec<FieldPat<'tcx>>,
    },

    /// `box P`, `&P`, `&mut P`, etc.
    Deref {
        subpattern: Pat<'tcx>,
    },

    /// One of the following:
    /// * `&str`, which will be handled as a string pattern and thus exhaustiveness
    ///   checking will detect if you use the same string twice in different patterns.
    /// * integer, bool, char or float, which will be handled by exhaustivenes to cover exactly
    ///   its own value, similar to `&str`, but these values are much simpler.
    /// * Opaque constants, that must not be matched structurally. So anything that does not derive
    ///   `PartialEq` and `Eq`.
    Constant {
        value: &'tcx ty::Const<'tcx>,
    },

    Range(PatRange<'tcx>),

    /// Matches against a slice, checking the length and extracting elements.
    /// irrefutable when there is a slice pattern and both `prefix` and `suffix` are empty.
    /// e.g., `&[ref xs @ ..]`.
    Slice {
        prefix: Vec<Pat<'tcx>>,
        slice: Option<Pat<'tcx>>,
        suffix: Vec<Pat<'tcx>>,
    },

    /// Fixed match against an array; irrefutable.
    Array {
        prefix: Vec<Pat<'tcx>>,
        slice: Option<Pat<'tcx>>,
        suffix: Vec<Pat<'tcx>>,
    },

    /// An or-pattern, e.g. `p | q`.
    /// Invariant: `pats.len() >= 2`.
    Or {
        pats: Vec<Pat<'tcx>>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, HashStable)]
pub struct PatRange<'tcx> {
    pub lo: &'tcx ty::Const<'tcx>,
    pub hi: &'tcx ty::Const<'tcx>,
    pub end: RangeEnd,
}

impl<'tcx> fmt::Display for Pat<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Printing lists is a chore.
        let mut first = true;
        let mut start_or_continue = |s| {
            if first {
                first = false;
                ""
            } else {
                s
            }
        };
        let mut start_or_comma = || start_or_continue(", ");

        match *self.kind {
            PatKind::Wild => write!(f, "_"),
            PatKind::AscribeUserType { ref subpattern, .. } => write!(f, "{}: _", subpattern),
            PatKind::Binding { mutability, name, mode, ref subpattern, .. } => {
                let is_mut = match mode {
                    BindingMode::ByValue => mutability == Mutability::Mut,
                    BindingMode::ByRef(bk) => {
                        write!(f, "ref ")?;
                        matches!(bk, BorrowKind::Mut { .. })
                    }
                };
                if is_mut {
                    write!(f, "mut ")?;
                }
                write!(f, "{}", name)?;
                if let Some(ref subpattern) = *subpattern {
                    write!(f, " @ {}", subpattern)?;
                }
                Ok(())
            }
            PatKind::Variant { ref subpatterns, .. } | PatKind::Leaf { ref subpatterns } => {
                let variant = match *self.kind {
                    PatKind::Variant { adt_def, variant_index, .. } => {
                        Some(&adt_def.variants[variant_index])
                    }
                    _ => {
                        if let ty::Adt(adt, _) = self.ty.kind() {
                            if !adt.is_enum() {
                                Some(&adt.variants[VariantIdx::new(0)])
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                };

                if let Some(variant) = variant {
                    write!(f, "{}", variant.ident)?;

                    // Only for Adt we can have `S {...}`,
                    // which we handle separately here.
                    if variant.ctor_kind == CtorKind::Fictive {
                        write!(f, " {{ ")?;

                        let mut printed = 0;
                        for p in subpatterns {
                            if let PatKind::Wild = *p.pattern.kind {
                                continue;
                            }
                            let name = variant.fields[p.field.index()].ident;
                            write!(f, "{}{}: {}", start_or_comma(), name, p.pattern)?;
                            printed += 1;
                        }

                        if printed < variant.fields.len() {
                            write!(f, "{}..", start_or_comma())?;
                        }

                        return write!(f, " }}");
                    }
                }

                let num_fields = variant.map_or(subpatterns.len(), |v| v.fields.len());
                if num_fields != 0 || variant.is_none() {
                    write!(f, "(")?;
                    for i in 0..num_fields {
                        write!(f, "{}", start_or_comma())?;

                        // Common case: the field is where we expect it.
                        if let Some(p) = subpatterns.get(i) {
                            if p.field.index() == i {
                                write!(f, "{}", p.pattern)?;
                                continue;
                            }
                        }

                        // Otherwise, we have to go looking for it.
                        if let Some(p) = subpatterns.iter().find(|p| p.field.index() == i) {
                            write!(f, "{}", p.pattern)?;
                        } else {
                            write!(f, "_")?;
                        }
                    }
                    write!(f, ")")?;
                }

                Ok(())
            }
            PatKind::Deref { ref subpattern } => {
                match self.ty.kind() {
                    ty::Adt(def, _) if def.is_box() => write!(f, "box ")?,
                    ty::Ref(_, _, mutbl) => {
                        write!(f, "&{}", mutbl.prefix_str())?;
                    }
                    _ => bug!("{} is a bad Deref pattern type", self.ty),
                }
                write!(f, "{}", subpattern)
            }
            PatKind::Constant { value } => write!(f, "{}", value),
            PatKind::Range(PatRange { lo, hi, end }) => {
                write!(f, "{}", lo)?;
                write!(f, "{}", end)?;
                write!(f, "{}", hi)
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix }
            | PatKind::Array { ref prefix, ref slice, ref suffix } => {
                write!(f, "[")?;
                for p in prefix {
                    write!(f, "{}{}", start_or_comma(), p)?;
                }
                if let Some(ref slice) = *slice {
                    write!(f, "{}", start_or_comma())?;
                    match *slice.kind {
                        PatKind::Wild => {}
                        _ => write!(f, "{}", slice)?,
                    }
                    write!(f, "..")?;
                }
                for p in suffix {
                    write!(f, "{}{}", start_or_comma(), p)?;
                }
                write!(f, "]")
            }
            PatKind::Or { ref pats } => {
                for pat in pats {
                    write!(f, "{}{}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
        }
    }
}
