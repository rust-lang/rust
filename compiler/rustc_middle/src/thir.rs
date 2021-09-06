//! THIR datatypes and definitions. See the [rustc dev guide] for more info.
//!
//! If you compare the THIR [`ExprKind`] to [`hir::ExprKind`], you will see it is
//! a good bit simpler. In fact, a number of the more straight-forward
//! MIR simplifications are already done in the lowering to THIR. For
//! example, method calls and overloaded operators are absent: they are
//! expected to be converted into [`ExprKind::Call`] instances.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/thir.html

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::DefId;
use rustc_hir::RangeEnd;
use rustc_index::newtype_index;
use rustc_index::vec::IndexVec;
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
    /// An index to an [`Arm`] stored in [`Thir::arms`]
    #[derive(HashStable)]
    pub struct ArmId {
        DEBUG_FORMAT = "a{}"
    }
}

newtype_index! {
    /// An index to an [`Expr`] stored in [`Thir::exprs`]
    #[derive(HashStable)]
    pub struct ExprId {
        DEBUG_FORMAT = "e{}"
    }
}

newtype_index! {
    #[derive(HashStable)]
    /// An index to a [`Stmt`] stored in [`Thir::stmts`]
    pub struct StmtId {
        DEBUG_FORMAT = "s{}"
    }
}

macro_rules! thir_with_elements {
    ($($name:ident: $id:ty => $value:ty,)*) => {
        /// A container for a THIR body.
        ///
        /// This can be indexed directly by any THIR index (e.g. [`ExprId`]).
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
    /// Whether the block itself has a label. Used by `label: {}`
    /// and `try` blocks.
    ///
    /// This does *not* include labels on loops, e.g. `'label: loop {}`.
    pub targeted_by_break: bool,
    pub region_scope: region::Scope,
    pub opt_destruction_scope: Option<region::Scope>,
    /// The span of the block, including the opening braces,
    /// the label, and the `unsafe` keyword, if present.
    pub span: Span,
    /// The statements in the blocK.
    pub stmts: Box<[StmtId]>,
    /// The trailing expression of the block, if any.
    pub expr: Option<ExprId>,
    pub safety_mode: BlockSafety,
}

#[derive(Debug, HashStable)]
pub struct Adt<'tcx> {
    /// The ADT we're constructing.
    pub adt_def: &'tcx AdtDef,
    /// The variant of the ADT.
    pub variant_index: VariantIdx,
    pub substs: SubstsRef<'tcx>,

    /// Optional user-given substs: for something like `let x =
    /// Bar::<T> { ... }`.
    pub user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,

    pub fields: Box<[FieldExpr]>,
    /// The base, e.g. `Foo {x: 1, .. base}`.
    pub base: Option<FruInfo<'tcx>>,
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum BlockSafety {
    Safe,
    /// A compiler-generated unsafe block
    BuiltinUnsafe,
    /// An `unsafe` block. The `HirId` is the ID of the block.
    ExplicitUnsafe(hir::HirId),
}

#[derive(Debug, HashStable)]
pub struct Stmt<'tcx> {
    pub kind: StmtKind<'tcx>,
    pub opt_destruction_scope: Option<region::Scope>,
}

#[derive(Debug, HashStable)]
pub enum StmtKind<'tcx> {
    /// An expression with a trailing semicolon.
    Expr {
        /// The scope for this statement; may be used as lifetime of temporaries.
        scope: region::Scope,

        /// The expression being evaluated in this statement.
        expr: ExprId,
    },

    /// A `let` binding.
    Let {
        /// The scope for variables bound in this `let`; it covers this and
        /// all the remaining statements in the block.
        remainder_scope: region::Scope,

        /// The scope for the initialization itself; might be used as
        /// lifetime of temporaries.
        init_scope: region::Scope,

        /// `let <PAT> = ...`
        ///
        /// If a type annotation is included, it is added as an ascription pattern.
        pattern: Pat<'tcx>,

        /// `let pat: ty = <INIT>`
        initializer: Option<ExprId>,

        /// The lint level for this `let` statement.
        lint_level: LintLevel,
    },
}

// `Expr` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Expr<'_>, 104);

/// A THIR expression.
#[derive(Debug, HashStable)]
pub struct Expr<'tcx> {
    /// The type of this expression
    pub ty: Ty<'tcx>,

    /// The lifetime of this expression if it should be spilled into a
    /// temporary; should be `None` only if in a constant context
    pub temp_lifetime: Option<region::Scope>,

    /// span of the expression in the source
    pub span: Span,

    /// kind of expression
    pub kind: ExprKind<'tcx>,
}

#[derive(Debug, HashStable)]
pub enum ExprKind<'tcx> {
    /// `Scope`s are used to explicitely mark destruction scopes,
    /// and to track the `HirId` of the expressions within the scope.
    Scope {
        region_scope: region::Scope,
        lint_level: LintLevel,
        value: ExprId,
    },
    /// A `box <value>` expression.
    Box {
        value: ExprId,
    },
    /// An `if` expression.
    If {
        if_then_scope: region::Scope,
        cond: ExprId,
        then: ExprId,
        else_opt: Option<ExprId>,
    },
    /// A function call. Method calls and overloaded operators are converted to plain function calls.
    Call {
        /// The type of the function. This is often a [`FnDef`] or a [`FnPtr`].
        ///
        /// [`FnDef`]: ty::TyKind::FnDef
        /// [`FnPtr`]: ty::TyKind::FnPtr
        ty: Ty<'tcx>,
        /// The function itself.
        fun: ExprId,
        /// The arguments passed to the function.
        ///
        /// Note: in some cases (like calling a closure), the function call `f(...args)` gets
        /// rewritten as a call to a function trait method (e.g. `FnOnce::call_once(f, (...args))`).
        args: Box<[ExprId]>,
        /// Whether this is from an overloaded operator rather than a
        /// function call from HIR. `true` for overloaded function call.
        from_hir_call: bool,
        /// The span of the function, without the dot and receiver
        /// (e.g. `foo(a, b)` in `x.foo(a, b)`).
        fn_span: Span,
    },
    /// A *non-overloaded* dereference.
    Deref {
        arg: ExprId,
    },
    /// A *non-overloaded* binary operation.
    Binary {
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// A logical operation. This is distinct from `BinaryOp` because
    /// the operands need to be lazily evaluated.
    LogicalOp {
        op: LogicalOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// A *non-overloaded* unary operation. Note that here the deref (`*`)
    /// operator is represented by `ExprKind::Deref`.
    Unary {
        op: UnOp,
        arg: ExprId,
    },
    /// A cast: `<source> as <type>`. The type we cast to is the type of
    /// the parent expression.
    Cast {
        source: ExprId,
    },
    Use {
        source: ExprId,
    }, // Use a lexpr to get a vexpr.
    /// A coercion from `!` to any type.
    NeverToAny {
        source: ExprId,
    },
    /// A pointer cast. More information can be found in [`PointerCast`].
    Pointer {
        cast: PointerCast,
        source: ExprId,
    },
    /// A `loop` expression.
    Loop {
        body: ExprId,
    },
    Let {
        expr: ExprId,
        pat: Pat<'tcx>,
    },
    /// A `match` expression.
    Match {
        scrutinee: ExprId,
        arms: Box<[ArmId]>,
    },
    /// A block.
    Block {
        body: Block,
    },
    /// An assignment: `lhs = rhs`.
    Assign {
        lhs: ExprId,
        rhs: ExprId,
    },
    /// A *non-overloaded* operation assignment, e.g. `lhs += rhs`.
    AssignOp {
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// Access to a struct or tuple field.
    Field {
        lhs: ExprId,
        /// This can be a named (`.foo`) or unnamed (`.0`) field.
        name: Field,
    },
    /// A *non-overloaded* indexing operation.
    Index {
        lhs: ExprId,
        index: ExprId,
    },
    /// A local variable.
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
    /// A borrow, e.g. `&arg`.
    Borrow {
        borrow_kind: BorrowKind,
        arg: ExprId,
    },
    /// A `&raw [const|mut] $place_expr` raw borrow resulting in type `*[const|mut] T`.
    AddressOf {
        mutability: hir::Mutability,
        arg: ExprId,
    },
    /// A `break` expression.
    Break {
        label: region::Scope,
        value: Option<ExprId>,
    },
    /// A `continue` expression.
    Continue {
        label: region::Scope,
    },
    /// A `return` expression.
    Return {
        value: Option<ExprId>,
    },
    /// An inline `const` block, e.g. `const {}`.
    ConstBlock {
        value: &'tcx Const<'tcx>,
    },
    /// An array literal constructed from one repeated element, e.g. `[1; 5]`.
    Repeat {
        value: ExprId,
        count: &'tcx Const<'tcx>,
    },
    /// An array, e.g. `[a, b, c, d]`.
    Array {
        fields: Box<[ExprId]>,
    },
    /// A tuple, e.g. `(a, b, c, d)`.
    Tuple {
        fields: Box<[ExprId]>,
    },
    /// An ADT constructor, e.g. `Foo {x: 1, y: 2}`.
    Adt(Box<Adt<'tcx>>),
    /// A type ascription on a place.
    PlaceTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    /// A type ascription on a value, e.g. `42: i32`.
    ValueTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: Option<Canonical<'tcx, UserType<'tcx>>>,
    },
    /// A closure definition.
    Closure {
        closure_id: DefId,
        substs: UpvarSubsts<'tcx>,
        upvars: Box<[ExprId]>,
        movability: Option<hir::Movability>,
        fake_reads: Vec<(ExprId, FakeReadCause, hir::HirId)>,
    },
    /// A literal.
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
    /// Inline assembly, i.e. `asm!()`.
    InlineAsm {
        template: &'tcx [InlineAsmTemplatePiece],
        operands: Box<[InlineAsmOperand<'tcx>]>,
        options: InlineAsmOptions,
        line_spans: &'tcx [Span],
    },
    /// An expression taking a reference to a thread local.
    ThreadLocalRef(DefId),
    /// Inline LLVM assembly, i.e. `llvm_asm!()`.
    LlvmInlineAsm {
        asm: &'tcx hir::LlvmInlineAsmInner,
        outputs: Box<[ExprId]>,
        inputs: Box<[ExprId]>,
    },
    /// A `yield` expression.
    Yield {
        value: ExprId,
    },
}

/// Represents the association of a field identifier and an expression.
///
/// This is used in struct constructors.
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

/// A `match` arm.
#[derive(Debug, HashStable)]
pub struct Arm<'tcx> {
    pub pattern: Pat<'tcx>,
    pub guard: Option<Guard<'tcx>>,
    pub body: ExprId,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
}

/// A `match` guard.
#[derive(Debug, HashStable)]
pub enum Guard<'tcx> {
    If(ExprId),
    IfLet(Pat<'tcx>, ExprId),
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum LogicalOp {
    /// The `&&` operator.
    And,
    /// The `||` operator.
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
    /// A wildward pattern: `_`.
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
                    _ => self.ty.ty_adt_def().and_then(|adt| {
                        if !adt.is_enum() { Some(adt.non_enum_variant()) } else { None }
                    }),
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

pub mod visit {
    use super::*;
    pub trait Visitor<'a, 'tcx: 'a>: Sized {
        fn thir(&self) -> &'a Thir<'tcx>;

        fn visit_expr(&mut self, expr: &Expr<'tcx>) {
            walk_expr(self, expr);
        }

        fn visit_stmt(&mut self, stmt: &Stmt<'tcx>) {
            walk_stmt(self, stmt);
        }

        fn visit_block(&mut self, block: &Block) {
            walk_block(self, block);
        }

        fn visit_arm(&mut self, arm: &Arm<'tcx>) {
            walk_arm(self, arm);
        }

        fn visit_pat(&mut self, pat: &Pat<'tcx>) {
            walk_pat(self, pat);
        }

        fn visit_const(&mut self, _cnst: &'tcx Const<'tcx>) {}
    }

    pub fn walk_expr<'a, 'tcx: 'a, V: Visitor<'a, 'tcx>>(visitor: &mut V, expr: &Expr<'tcx>) {
        use ExprKind::*;
        match expr.kind {
            Scope { value, region_scope: _, lint_level: _ } => {
                visitor.visit_expr(&visitor.thir()[value])
            }
            Box { value } => visitor.visit_expr(&visitor.thir()[value]),
            If { cond, then, else_opt, if_then_scope: _ } => {
                visitor.visit_expr(&visitor.thir()[cond]);
                visitor.visit_expr(&visitor.thir()[then]);
                if let Some(else_expr) = else_opt {
                    visitor.visit_expr(&visitor.thir()[else_expr]);
                }
            }
            Call { fun, ref args, ty: _, from_hir_call: _, fn_span: _ } => {
                visitor.visit_expr(&visitor.thir()[fun]);
                for &arg in &**args {
                    visitor.visit_expr(&visitor.thir()[arg]);
                }
            }
            Deref { arg } => visitor.visit_expr(&visitor.thir()[arg]),
            Binary { lhs, rhs, op: _ } | LogicalOp { lhs, rhs, op: _ } => {
                visitor.visit_expr(&visitor.thir()[lhs]);
                visitor.visit_expr(&visitor.thir()[rhs]);
            }
            Unary { arg, op: _ } => visitor.visit_expr(&visitor.thir()[arg]),
            Cast { source } => visitor.visit_expr(&visitor.thir()[source]),
            Use { source } => visitor.visit_expr(&visitor.thir()[source]),
            NeverToAny { source } => visitor.visit_expr(&visitor.thir()[source]),
            Pointer { source, cast: _ } => visitor.visit_expr(&visitor.thir()[source]),
            Let { expr, .. } => {
                visitor.visit_expr(&visitor.thir()[expr]);
            }
            Loop { body } => visitor.visit_expr(&visitor.thir()[body]),
            Match { scrutinee, ref arms } => {
                visitor.visit_expr(&visitor.thir()[scrutinee]);
                for &arm in &**arms {
                    visitor.visit_arm(&visitor.thir()[arm]);
                }
            }
            Block { ref body } => visitor.visit_block(body),
            Assign { lhs, rhs } | AssignOp { lhs, rhs, op: _ } => {
                visitor.visit_expr(&visitor.thir()[lhs]);
                visitor.visit_expr(&visitor.thir()[rhs]);
            }
            Field { lhs, name: _ } => visitor.visit_expr(&visitor.thir()[lhs]),
            Index { lhs, index } => {
                visitor.visit_expr(&visitor.thir()[lhs]);
                visitor.visit_expr(&visitor.thir()[index]);
            }
            VarRef { id: _ } | UpvarRef { closure_def_id: _, var_hir_id: _ } => {}
            Borrow { arg, borrow_kind: _ } => visitor.visit_expr(&visitor.thir()[arg]),
            AddressOf { arg, mutability: _ } => visitor.visit_expr(&visitor.thir()[arg]),
            Break { value, label: _ } => {
                if let Some(value) = value {
                    visitor.visit_expr(&visitor.thir()[value])
                }
            }
            Continue { label: _ } => {}
            Return { value } => {
                if let Some(value) = value {
                    visitor.visit_expr(&visitor.thir()[value])
                }
            }
            ConstBlock { value } => visitor.visit_const(value),
            Repeat { value, count } => {
                visitor.visit_expr(&visitor.thir()[value]);
                visitor.visit_const(count);
            }
            Array { ref fields } | Tuple { ref fields } => {
                for &field in &**fields {
                    visitor.visit_expr(&visitor.thir()[field]);
                }
            }
            Adt(box crate::thir::Adt {
                ref fields,
                ref base,
                adt_def: _,
                variant_index: _,
                substs: _,
                user_ty: _,
            }) => {
                for field in &**fields {
                    visitor.visit_expr(&visitor.thir()[field.expr]);
                }
                if let Some(base) = base {
                    visitor.visit_expr(&visitor.thir()[base.base]);
                }
            }
            PlaceTypeAscription { source, user_ty: _ }
            | ValueTypeAscription { source, user_ty: _ } => {
                visitor.visit_expr(&visitor.thir()[source])
            }
            Closure { closure_id: _, substs: _, upvars: _, movability: _, fake_reads: _ } => {}
            Literal { literal, user_ty: _, const_id: _ } => visitor.visit_const(literal),
            StaticRef { literal, def_id: _ } => visitor.visit_const(literal),
            InlineAsm { ref operands, template: _, options: _, line_spans: _ } => {
                for op in &**operands {
                    use InlineAsmOperand::*;
                    match op {
                        In { expr, reg: _ }
                        | Out { expr: Some(expr), reg: _, late: _ }
                        | InOut { expr, reg: _, late: _ }
                        | SymFn { expr } => visitor.visit_expr(&visitor.thir()[*expr]),
                        SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                            visitor.visit_expr(&visitor.thir()[*in_expr]);
                            if let Some(out_expr) = out_expr {
                                visitor.visit_expr(&visitor.thir()[*out_expr]);
                            }
                        }
                        Out { expr: None, reg: _, late: _ }
                        | Const { value: _, span: _ }
                        | SymStatic { def_id: _ } => {}
                    }
                }
            }
            ThreadLocalRef(_) => {}
            LlvmInlineAsm { ref outputs, ref inputs, asm: _ } => {
                for &out_expr in &**outputs {
                    visitor.visit_expr(&visitor.thir()[out_expr]);
                }
                for &in_expr in &**inputs {
                    visitor.visit_expr(&visitor.thir()[in_expr]);
                }
            }
            Yield { value } => visitor.visit_expr(&visitor.thir()[value]),
        }
    }

    pub fn walk_stmt<'a, 'tcx: 'a, V: Visitor<'a, 'tcx>>(visitor: &mut V, stmt: &Stmt<'tcx>) {
        match &stmt.kind {
            StmtKind::Expr { expr, scope: _ } => visitor.visit_expr(&visitor.thir()[*expr]),
            StmtKind::Let {
                initializer,
                remainder_scope: _,
                init_scope: _,
                ref pattern,
                lint_level: _,
            } => {
                if let Some(init) = initializer {
                    visitor.visit_expr(&visitor.thir()[*init]);
                }
                visitor.visit_pat(pattern);
            }
        }
    }

    pub fn walk_block<'a, 'tcx: 'a, V: Visitor<'a, 'tcx>>(visitor: &mut V, block: &Block) {
        for &stmt in &*block.stmts {
            visitor.visit_stmt(&visitor.thir()[stmt]);
        }
        if let Some(expr) = block.expr {
            visitor.visit_expr(&visitor.thir()[expr]);
        }
    }

    pub fn walk_arm<'a, 'tcx: 'a, V: Visitor<'a, 'tcx>>(visitor: &mut V, arm: &Arm<'tcx>) {
        match arm.guard {
            Some(Guard::If(expr)) => visitor.visit_expr(&visitor.thir()[expr]),
            Some(Guard::IfLet(ref pat, expr)) => {
                visitor.visit_pat(pat);
                visitor.visit_expr(&visitor.thir()[expr]);
            }
            None => {}
        }
        visitor.visit_pat(&arm.pattern);
        visitor.visit_expr(&visitor.thir()[arm.body]);
    }

    pub fn walk_pat<'a, 'tcx: 'a, V: Visitor<'a, 'tcx>>(visitor: &mut V, pat: &Pat<'tcx>) {
        use PatKind::*;
        match pat.kind.as_ref() {
            AscribeUserType { subpattern, ascription: _ }
            | Deref { subpattern }
            | Binding {
                subpattern: Some(subpattern),
                mutability: _,
                mode: _,
                var: _,
                ty: _,
                is_primary: _,
                name: _,
            } => visitor.visit_pat(&subpattern),
            Binding { .. } | Wild => {}
            Variant { subpatterns, adt_def: _, substs: _, variant_index: _ }
            | Leaf { subpatterns } => {
                for subpattern in subpatterns {
                    visitor.visit_pat(&subpattern.pattern);
                }
            }
            Constant { value } => visitor.visit_const(value),
            Range(range) => {
                visitor.visit_const(range.lo);
                visitor.visit_const(range.hi);
            }
            Slice { prefix, slice, suffix } | Array { prefix, slice, suffix } => {
                for subpattern in prefix {
                    visitor.visit_pat(&subpattern);
                }
                if let Some(pat) = slice {
                    visitor.visit_pat(pat);
                }
                for subpattern in suffix {
                    visitor.visit_pat(&subpattern);
                }
            }
            Or { pats } => {
                for pat in pats {
                    visitor.visit_pat(&pat);
                }
            }
        };
    }
}
