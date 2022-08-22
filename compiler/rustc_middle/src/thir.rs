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
use rustc_middle::middle::region;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::mir::{self, BinOp, BorrowKind, FakeReadCause, Field, Mutability, UnOp};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, AdtDef, Ty, UpvarSubsts};
use rustc_middle::ty::{CanonicalUserType, CanonicalUserTypeAnnotation};
use rustc_span::def_id::LocalDefId;
use rustc_span::{sym, Span, Symbol, DUMMY_SP};
use rustc_target::abi::VariantIdx;
use rustc_target::asm::InlineAsmRegOrRegClass;
use std::fmt;
use std::ops::Index;

pub mod visit;

macro_rules! thir_with_elements {
    ($($name:ident: $id:ty => $value:ty => $format:literal,)*) => {
        $(
            newtype_index! {
                #[derive(HashStable)]
                pub struct $id {
                    DEBUG_FORMAT = $format
                }
            }
        )*

        /// A container for a THIR body.
        ///
        /// This can be indexed directly by any THIR index (e.g. [`ExprId`]).
        #[derive(Debug, HashStable, Clone)]
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

pub const UPVAR_ENV_PARAM: ParamId = ParamId::from_u32(0);

thir_with_elements! {
    arms: ArmId => Arm<'tcx> => "a{}",
    blocks: BlockId => Block => "b{}",
    exprs: ExprId => Expr<'tcx> => "e{}",
    stmts: StmtId => Stmt<'tcx> => "s{}",
    params: ParamId => Param<'tcx> => "p{}",
}

/// Description of a type-checked function parameter.
#[derive(Clone, Debug, HashStable)]
pub struct Param<'tcx> {
    /// The pattern that appears in the parameter list, or None for implicit parameters.
    pub pat: Option<Box<Pat<'tcx>>>,
    /// The possibly inferred type.
    pub ty: Ty<'tcx>,
    /// Span of the explicitly provided type, or None if inferred for closures.
    pub ty_span: Option<Span>,
    /// Whether this param is `self`, and how it is bound.
    pub self_kind: Option<hir::ImplicitSelfKind>,
    /// HirId for lints.
    pub hir_id: Option<hir::HirId>,
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum LintLevel {
    Inherited,
    Explicit(hir::HirId),
}

#[derive(Clone, Debug, HashStable)]
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

type UserTy<'tcx> = Option<Box<CanonicalUserType<'tcx>>>;

#[derive(Clone, Debug, HashStable)]
pub struct AdtExpr<'tcx> {
    /// The ADT we're constructing.
    pub adt_def: AdtDef<'tcx>,
    /// The variant of the ADT.
    pub variant_index: VariantIdx,
    pub substs: SubstsRef<'tcx>,

    /// Optional user-given substs: for something like `let x =
    /// Bar::<T> { ... }`.
    pub user_ty: UserTy<'tcx>,

    pub fields: Box<[FieldExpr]>,
    /// The base, e.g. `Foo {x: 1, .. base}`.
    pub base: Option<FruInfo<'tcx>>,
}

#[derive(Clone, Debug, HashStable)]
pub struct ClosureExpr<'tcx> {
    pub closure_id: LocalDefId,
    pub substs: UpvarSubsts<'tcx>,
    pub upvars: Box<[ExprId]>,
    pub movability: Option<hir::Movability>,
    pub fake_reads: Vec<(ExprId, FakeReadCause, hir::HirId)>,
}

#[derive(Clone, Debug, HashStable)]
pub struct InlineAsmExpr<'tcx> {
    pub template: &'tcx [InlineAsmTemplatePiece],
    pub operands: Box<[InlineAsmOperand<'tcx>]>,
    pub options: InlineAsmOptions,
    pub line_spans: &'tcx [Span],
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum BlockSafety {
    Safe,
    /// A compiler-generated unsafe block
    BuiltinUnsafe,
    /// An `unsafe` block. The `HirId` is the ID of the block.
    ExplicitUnsafe(hir::HirId),
}

#[derive(Clone, Debug, HashStable)]
pub struct Stmt<'tcx> {
    pub kind: StmtKind<'tcx>,
    pub opt_destruction_scope: Option<region::Scope>,
}

#[derive(Clone, Debug, HashStable)]
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
        pattern: Box<Pat<'tcx>>,

        /// `let pat: ty = <INIT>`
        initializer: Option<ExprId>,

        /// `let pat: ty = <INIT> else { <ELSE> }
        else_block: Option<BlockId>,

        /// The lint level for this `let` statement.
        lint_level: LintLevel,
    },
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct LocalVarId(pub hir::HirId);

/// A THIR expression.
#[derive(Clone, Debug, HashStable)]
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

#[derive(Clone, Debug, HashStable)]
pub enum ExprKind<'tcx> {
    /// `Scope`s are used to explicitly mark destruction scopes,
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
        pat: Box<Pat<'tcx>>,
    },
    /// A `match` expression.
    Match {
        scrutinee: ExprId,
        arms: Box<[ArmId]>,
    },
    /// A block.
    Block {
        block: BlockId,
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
    /// Access to a field of a struct, a tuple, an union, or an enum.
    Field {
        lhs: ExprId,
        /// Variant containing the field.
        variant_index: VariantIdx,
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
        id: LocalVarId,
    },
    /// Used to represent upvars mentioned in a closure/generator
    UpvarRef {
        /// DefId of the closure/generator
        closure_def_id: DefId,

        /// HirId of the root variable
        var_hir_id: LocalVarId,
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
        did: DefId,
        substs: SubstsRef<'tcx>,
    },
    /// An array literal constructed from one repeated element, e.g. `[1; 5]`.
    Repeat {
        value: ExprId,
        count: ty::Const<'tcx>,
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
    Adt(Box<AdtExpr<'tcx>>),
    /// A type ascription on a place.
    PlaceTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: UserTy<'tcx>,
    },
    /// A type ascription on a value, e.g. `42: i32`.
    ValueTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: UserTy<'tcx>,
    },
    /// A closure definition.
    Closure(Box<ClosureExpr<'tcx>>),
    /// A literal.
    Literal {
        lit: &'tcx hir::Lit,
        neg: bool,
    },
    /// For literals that don't correspond to anything in the HIR
    NonHirLiteral {
        lit: ty::ScalarInt,
        user_ty: UserTy<'tcx>,
    },
    /// A literal of a ZST type.
    ZstLiteral {
        user_ty: UserTy<'tcx>,
    },
    /// Associated constants and named constants
    NamedConst {
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        user_ty: UserTy<'tcx>,
    },
    ConstParam {
        param: ty::ParamConst,
        def_id: DefId,
    },
    // FIXME improve docs for `StaticRef` by distinguishing it from `NamedConst`
    /// A literal containing the address of a `static`.
    ///
    /// This is only distinguished from `Literal` so that we can register some
    /// info for diagnostics.
    StaticRef {
        alloc_id: AllocId,
        ty: Ty<'tcx>,
        def_id: DefId,
    },
    /// Inline assembly, i.e. `asm!()`.
    InlineAsm(Box<InlineAsmExpr<'tcx>>),
    /// An expression taking a reference to a thread local.
    ThreadLocalRef(DefId),
    /// A `yield` expression.
    Yield {
        value: ExprId,
    },
}

/// Represents the association of a field identifier and an expression.
///
/// This is used in struct constructors.
#[derive(Clone, Debug, HashStable)]
pub struct FieldExpr {
    pub name: Field,
    pub expr: ExprId,
}

#[derive(Clone, Debug, HashStable)]
pub struct FruInfo<'tcx> {
    pub base: ExprId,
    pub field_types: Box<[Ty<'tcx>]>,
}

/// A `match` arm.
#[derive(Clone, Debug, HashStable)]
pub struct Arm<'tcx> {
    pub pattern: Box<Pat<'tcx>>,
    pub guard: Option<Guard<'tcx>>,
    pub body: ExprId,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
}

/// A `match` guard.
#[derive(Clone, Debug, HashStable)]
pub enum Guard<'tcx> {
    If(ExprId),
    IfLet(Box<Pat<'tcx>>, ExprId),
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum LogicalOp {
    /// The `&&` operator.
    And,
    /// The `||` operator.
    Or,
}

#[derive(Clone, Debug, HashStable)]
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
        value: mir::ConstantKind<'tcx>,
        span: Span,
    },
    SymFn {
        value: mir::ConstantKind<'tcx>,
        span: Span,
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

#[derive(Clone, Debug, HashStable)]
pub struct FieldPat<'tcx> {
    pub field: Field,
    pub pattern: Box<Pat<'tcx>>,
}

#[derive(Clone, Debug, HashStable)]
pub struct Pat<'tcx> {
    pub ty: Ty<'tcx>,
    pub span: Span,
    pub kind: PatKind<'tcx>,
}

impl<'tcx> Pat<'tcx> {
    pub fn wildcard_from_ty(ty: Ty<'tcx>) -> Self {
        Pat { ty, span: DUMMY_SP, kind: PatKind::Wild }
    }

    pub fn simple_ident(&self) -> Option<Symbol> {
        match self.kind {
            PatKind::Binding { name, mode: BindingMode::ByValue, subpattern: None, .. } => {
                Some(name)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, HashStable)]
pub struct Ascription<'tcx> {
    pub annotation: CanonicalUserTypeAnnotation<'tcx>,
    /// Variance to use when relating the `user_ty` to the **type of the value being
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
}

#[derive(Clone, Debug, HashStable)]
pub enum PatKind<'tcx> {
    /// A wildcard pattern: `_`.
    Wild,

    AscribeUserType {
        ascription: Ascription<'tcx>,
        subpattern: Box<Pat<'tcx>>,
    },

    /// `x`, `ref x`, `x @ P`, etc.
    Binding {
        mutability: Mutability,
        name: Symbol,
        mode: BindingMode,
        var: LocalVarId,
        ty: Ty<'tcx>,
        subpattern: Option<Box<Pat<'tcx>>>,
        /// Is this the leftmost occurrence of the binding, i.e., is `var` the
        /// `HirId` of this pattern?
        is_primary: bool,
    },

    /// `Foo(...)` or `Foo{...}` or `Foo`, where `Foo` is a variant name from an ADT with
    /// multiple variants.
    Variant {
        adt_def: AdtDef<'tcx>,
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
        subpattern: Box<Pat<'tcx>>,
    },

    /// One of the following:
    /// * `&str`, which will be handled as a string pattern and thus exhaustiveness
    ///   checking will detect if you use the same string twice in different patterns.
    /// * integer, bool, char or float, which will be handled by exhaustiveness to cover exactly
    ///   its own value, similar to `&str`, but these values are much simpler.
    /// * Opaque constants, that must not be matched structurally. So anything that does not derive
    ///   `PartialEq` and `Eq`.
    Constant {
        value: mir::ConstantKind<'tcx>,
    },

    Range(Box<PatRange<'tcx>>),

    /// Matches against a slice, checking the length and extracting elements.
    /// irrefutable when there is a slice pattern and both `prefix` and `suffix` are empty.
    /// e.g., `&[ref xs @ ..]`.
    Slice {
        prefix: Box<[Box<Pat<'tcx>>]>,
        slice: Option<Box<Pat<'tcx>>>,
        suffix: Box<[Box<Pat<'tcx>>]>,
    },

    /// Fixed match against an array; irrefutable.
    Array {
        prefix: Box<[Box<Pat<'tcx>>]>,
        slice: Option<Box<Pat<'tcx>>>,
        suffix: Box<[Box<Pat<'tcx>>]>,
    },

    /// An or-pattern, e.g. `p | q`.
    /// Invariant: `pats.len() >= 2`.
    Or {
        pats: Box<[Box<Pat<'tcx>>]>,
    },
}

#[derive(Clone, Debug, PartialEq, HashStable)]
pub struct PatRange<'tcx> {
    pub lo: mir::ConstantKind<'tcx>,
    pub hi: mir::ConstantKind<'tcx>,
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

        match self.kind {
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
                let variant_and_name = match self.kind {
                    PatKind::Variant { adt_def, variant_index, .. } => ty::tls::with(|tcx| {
                        let variant = adt_def.variant(variant_index);
                        let adt_did = adt_def.did();
                        let name = if tcx.get_diagnostic_item(sym::Option) == Some(adt_did)
                            || tcx.get_diagnostic_item(sym::Result) == Some(adt_did)
                        {
                            variant.name.to_string()
                        } else {
                            format!("{}::{}", tcx.def_path_str(adt_def.did()), variant.name)
                        };
                        Some((variant, name))
                    }),
                    _ => self.ty.ty_adt_def().and_then(|adt_def| {
                        if !adt_def.is_enum() {
                            ty::tls::with(|tcx| {
                                Some((adt_def.non_enum_variant(), tcx.def_path_str(adt_def.did())))
                            })
                        } else {
                            None
                        }
                    }),
                };

                if let Some((variant, name)) = &variant_and_name {
                    write!(f, "{}", name)?;

                    // Only for Adt we can have `S {...}`,
                    // which we handle separately here.
                    if variant.ctor_kind == CtorKind::Fictive {
                        write!(f, " {{ ")?;

                        let mut printed = 0;
                        for p in subpatterns {
                            if let PatKind::Wild = p.pattern.kind {
                                continue;
                            }
                            let name = variant.fields[p.field.index()].name;
                            write!(f, "{}{}: {}", start_or_comma(), name, p.pattern)?;
                            printed += 1;
                        }

                        if printed < variant.fields.len() {
                            write!(f, "{}..", start_or_comma())?;
                        }

                        return write!(f, " }}");
                    }
                }

                let num_fields =
                    variant_and_name.as_ref().map_or(subpatterns.len(), |(v, _)| v.fields.len());
                if num_fields != 0 || variant_and_name.is_none() {
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
            PatKind::Range(box PatRange { lo, hi, end }) => {
                write!(f, "{}", lo)?;
                write!(f, "{}", end)?;
                write!(f, "{}", hi)
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix }
            | PatKind::Array { ref prefix, ref slice, ref suffix } => {
                write!(f, "[")?;
                for p in prefix.iter() {
                    write!(f, "{}{}", start_or_comma(), p)?;
                }
                if let Some(ref slice) = *slice {
                    write!(f, "{}", start_or_comma())?;
                    match slice.kind {
                        PatKind::Wild => {}
                        _ => write!(f, "{}", slice)?,
                    }
                    write!(f, "..")?;
                }
                for p in suffix.iter() {
                    write!(f, "{}{}", start_or_comma(), p)?;
                }
                write!(f, "]")
            }
            PatKind::Or { ref pats } => {
                for pat in pats.iter() {
                    write!(f, "{}{}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
        }
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    // These are in alphabetical order, which is easy to maintain.
    static_assert_size!(Block, 56);
    static_assert_size!(Expr<'_>, 64);
    static_assert_size!(ExprKind<'_>, 40);
    #[cfg(not(bootstrap))]
    static_assert_size!(Pat<'_>, 64);
    #[cfg(not(bootstrap))]
    static_assert_size!(PatKind<'_>, 48);
    #[cfg(not(bootstrap))]
    static_assert_size!(Stmt<'_>, 48);
    #[cfg(not(bootstrap))]
    static_assert_size!(StmtKind<'_>, 40);
}
