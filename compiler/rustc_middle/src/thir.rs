//! THIR datatypes and definitions. See the [rustc dev guide] for more info.
//!
//! If you compare the THIR [`ExprKind`] to [`hir::ExprKind`], you will see it is
//! a good bit simpler. In fact, a number of the more straight-forward
//! MIR simplifications are already done in the lowering to THIR. For
//! example, method calls and overloaded operators are absent: they are
//! expected to be converted into [`ExprKind::Call`] instances.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/thir.html

use std::cmp::Ordering;
use std::fmt;
use std::ops::Index;
use std::sync::Arc;

use rustc_abi::{FieldIdx, Integer, Size, VariantIdx};
use rustc_ast::{AsmMacro, InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{BindingMode, ByRef, HirId, MatchSource, RangeEnd};
use rustc_index::{IndexVec, newtype_index};
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeVisitable};
use rustc_span::def_id::LocalDefId;
use rustc_span::{ErrorGuaranteed, Span, Symbol};
use rustc_target::asm::InlineAsmRegOrRegClass;
use tracing::instrument;

use crate::middle::region;
use crate::mir::interpret::AllocId;
use crate::mir::{self, AssignOp, BinOp, BorrowKind, FakeReadCause, UnOp};
use crate::thir::visit::for_each_immediate_subpat;
use crate::ty::adjustment::PointerCoercion;
use crate::ty::layout::IntegerExt;
use crate::ty::{
    self, AdtDef, CanonicalUserType, CanonicalUserTypeAnnotation, FnSig, GenericArgsRef, List, Ty,
    TyCtxt, UpvarArgs,
};

pub mod visit;

macro_rules! thir_with_elements {
    (
        $($name:ident: $id:ty => $value:ty => $format:literal,)*
    ) => {
        $(
            newtype_index! {
                #[derive(HashStable)]
                #[debug_format = $format]
                pub struct $id {}
            }
        )*

        // Note: Making `Thir` implement `Clone` is useful for external tools that need access to
        // THIR bodies even after the `Steal` query result has been stolen.
        // One such tool is https://github.com/rust-corpus/qrates/.
        /// A container for a THIR body.
        ///
        /// This can be indexed directly by any THIR index (e.g. [`ExprId`]).
        #[derive(Debug, HashStable, Clone)]
        pub struct Thir<'tcx> {
            pub body_type: BodyTy<'tcx>,
            $(
                pub $name: IndexVec<$id, $value>,
            )*
        }

        impl<'tcx> Thir<'tcx> {
            pub fn new(body_type: BodyTy<'tcx>) -> Thir<'tcx> {
                Thir {
                    body_type,
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
    arms: ArmId => Arm<'tcx> => "a{}",
    blocks: BlockId => Block => "b{}",
    exprs: ExprId => Expr<'tcx> => "e{}",
    stmts: StmtId => Stmt<'tcx> => "s{}",
    params: ParamId => Param<'tcx> => "p{}",
}

#[derive(Debug, HashStable, Clone)]
pub enum BodyTy<'tcx> {
    Const(Ty<'tcx>),
    Fn(FnSig<'tcx>),
    GlobalAsm(Ty<'tcx>),
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
    pub hir_id: Option<HirId>,
}

#[derive(Copy, Clone, Debug, HashStable)]
pub enum LintLevel {
    Inherited,
    Explicit(HirId),
}

#[derive(Clone, Debug, HashStable)]
pub struct Block {
    /// Whether the block itself has a label. Used by `label: {}`
    /// and `try` blocks.
    ///
    /// This does *not* include labels on loops, e.g. `'label: loop {}`.
    pub targeted_by_break: bool,
    pub region_scope: region::Scope,
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
    pub args: GenericArgsRef<'tcx>,

    /// Optional user-given args: for something like `let x =
    /// Bar::<T> { ... }`.
    pub user_ty: UserTy<'tcx>,

    pub fields: Box<[FieldExpr]>,
    /// The base, e.g. `Foo {x: 1, ..base}`.
    pub base: AdtExprBase<'tcx>,
}

#[derive(Clone, Debug, HashStable)]
pub enum AdtExprBase<'tcx> {
    /// A struct expression where all the fields are explicitly enumerated: `Foo { a, b }`.
    None,
    /// A struct expression with a "base", an expression of the same type as the outer struct that
    /// will be used to populate any fields not explicitly mentioned: `Foo { ..base }`
    Base(FruInfo<'tcx>),
    /// A struct expression with a `..` tail but no "base" expression. The values from the struct
    /// fields' default values will be used to populate any fields not explicitly mentioned:
    /// `Foo { .. }`.
    DefaultFields(Box<[Ty<'tcx>]>),
}

#[derive(Clone, Debug, HashStable)]
pub struct ClosureExpr<'tcx> {
    pub closure_id: LocalDefId,
    pub args: UpvarArgs<'tcx>,
    pub upvars: Box<[ExprId]>,
    pub movability: Option<hir::Movability>,
    pub fake_reads: Vec<(ExprId, FakeReadCause, HirId)>,
}

#[derive(Clone, Debug, HashStable)]
pub struct InlineAsmExpr<'tcx> {
    pub asm_macro: AsmMacro,
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
    ExplicitUnsafe(HirId),
}

#[derive(Clone, Debug, HashStable)]
pub struct Stmt<'tcx> {
    pub kind: StmtKind<'tcx>,
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

        /// `let pat: ty = <INIT> else { <ELSE> }`
        else_block: Option<BlockId>,

        /// The lint level for this `let` statement.
        lint_level: LintLevel,

        /// Span of the `let <PAT> = <INIT>` part.
        span: Span,
    },
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable)]
pub struct LocalVarId(pub HirId);

/// A THIR expression.
#[derive(Clone, Debug, HashStable)]
pub struct Expr<'tcx> {
    /// kind of expression
    pub kind: ExprKind<'tcx>,

    /// The type of this expression
    pub ty: Ty<'tcx>,

    /// The lifetime of this expression if it should be spilled into a
    /// temporary
    pub temp_lifetime: TempLifetime,

    /// span of the expression in the source
    pub span: Span,
}

/// Temporary lifetime information for THIR expressions
#[derive(Clone, Copy, Debug, HashStable)]
pub struct TempLifetime {
    /// Lifetime for temporaries as expected.
    /// This should be `None` in a constant context.
    pub temp_lifetime: Option<region::Scope>,
    /// If `Some(lt)`, indicates that the lifetime of this temporary will change to `lt` in a future edition.
    /// If `None`, then no changes are expected, or lints are disabled.
    pub backwards_incompatible: Option<region::Scope>,
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
    /// A use expression `x.use`.
    ByUse {
        /// The expression on which use is applied.
        expr: ExprId,
        /// The span of use, without the dot and receiver
        /// (e.g. `use` in `x.use`).
        span: Span,
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
    /// Forces its contents to be treated as a value expression, not a place
    /// expression. This is inserted in some places where an operation would
    /// otherwise be erased completely (e.g. some no-op casts), but we still
    /// need to ensure that its operand is treated as a value and not a place.
    Use {
        source: ExprId,
    },
    /// A coercion from `!` to any type.
    NeverToAny {
        source: ExprId,
    },
    /// A pointer coercion. More information can be found in [`PointerCoercion`].
    /// Pointer casts that cannot be done by coercions are represented by [`ExprKind::Cast`].
    PointerCoercion {
        cast: PointerCoercion,
        source: ExprId,
        /// Whether this coercion is written with an `as` cast in the source code.
        is_from_as_cast: bool,
    },
    /// A `loop` expression.
    Loop {
        body: ExprId,
    },
    /// Special expression representing the `let` part of an `if let` or similar construct
    /// (including `if let` guards in match arms, and let-chains formed by `&&`).
    ///
    /// This isn't considered a real expression in surface Rust syntax, so it can
    /// only appear in specific situations, such as within the condition of an `if`.
    ///
    /// (Not to be confused with [`StmtKind::Let`], which is a normal `let` statement.)
    Let {
        expr: ExprId,
        pat: Box<Pat<'tcx>>,
    },
    /// A `match` expression.
    Match {
        scrutinee: ExprId,
        arms: Box<[ArmId]>,
        match_source: MatchSource,
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
        op: AssignOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// Access to a field of a struct, a tuple, an union, or an enum.
    Field {
        lhs: ExprId,
        /// Variant containing the field.
        variant_index: VariantIdx,
        /// This can be a named (`.foo`) or unnamed (`.0`) field.
        name: FieldIdx,
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
    /// Used to represent upvars mentioned in a closure/coroutine
    UpvarRef {
        /// DefId of the closure/coroutine
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
    RawBorrow {
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
    /// A `become` expression.
    Become {
        value: ExprId,
    },
    /// An inline `const` block, e.g. `const {}`.
    ConstBlock {
        did: DefId,
        args: GenericArgsRef<'tcx>,
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
        user_ty_span: Span,
    },
    /// A type ascription on a value, e.g. `type_ascribe!(42, i32)` or `42 as i32`.
    ValueTypeAscription {
        source: ExprId,
        /// Type that the user gave to this expression
        user_ty: UserTy<'tcx>,
        user_ty_span: Span,
    },
    /// An unsafe binder cast on a place, e.g. `unwrap_binder!(*ptr)`.
    PlaceUnwrapUnsafeBinder {
        source: ExprId,
    },
    /// An unsafe binder cast on a value, e.g. `unwrap_binder!(rvalue())`,
    /// which makes a temporary.
    ValueUnwrapUnsafeBinder {
        source: ExprId,
    },
    /// Construct an unsafe binder, e.g. `wrap_binder(&ref)`.
    WrapUnsafeBinder {
        source: ExprId,
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
        args: GenericArgsRef<'tcx>,
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
    /// Field offset (`offset_of!`)
    OffsetOf {
        container: Ty<'tcx>,
        fields: &'tcx List<(VariantIdx, FieldIdx)>,
    },
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
    pub name: FieldIdx,
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
    pub guard: Option<ExprId>,
    pub body: ExprId,
    pub lint_level: LintLevel,
    pub scope: region::Scope,
    pub span: Span,
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
        value: mir::Const<'tcx>,
        span: Span,
    },
    SymFn {
        value: ExprId,
    },
    SymStatic {
        def_id: DefId,
    },
    Label {
        block: BlockId,
    },
}

#[derive(Clone, Debug, HashStable, TypeVisitable)]
pub struct FieldPat<'tcx> {
    pub field: FieldIdx,
    pub pattern: Pat<'tcx>,
}

#[derive(Clone, Debug, HashStable, TypeVisitable)]
pub struct Pat<'tcx> {
    pub ty: Ty<'tcx>,
    pub span: Span,
    pub kind: PatKind<'tcx>,
}

impl<'tcx> Pat<'tcx> {
    pub fn simple_ident(&self) -> Option<Symbol> {
        match self.kind {
            PatKind::Binding {
                name, mode: BindingMode(ByRef::No, _), subpattern: None, ..
            } => Some(name),
            _ => None,
        }
    }

    /// Call `f` on every "binding" in a pattern, e.g., on `a` in
    /// `match foo() { Some(a) => (), None => () }`
    pub fn each_binding(&self, mut f: impl FnMut(Symbol, ByRef, Ty<'tcx>, Span)) {
        self.walk_always(|p| {
            if let PatKind::Binding { name, mode, ty, .. } = p.kind {
                f(name, mode.0, ty, p.span);
            }
        });
    }

    /// Walk the pattern in left-to-right order.
    ///
    /// If `it(pat)` returns `false`, the children are not visited.
    pub fn walk(&self, mut it: impl FnMut(&Pat<'tcx>) -> bool) {
        self.walk_(&mut it)
    }

    fn walk_(&self, it: &mut impl FnMut(&Pat<'tcx>) -> bool) {
        if !it(self) {
            return;
        }

        for_each_immediate_subpat(self, |p| p.walk_(it));
    }

    /// Whether the pattern has a `PatKind::Error` nested within.
    pub fn pat_error_reported(&self) -> Result<(), ErrorGuaranteed> {
        let mut error = None;
        self.walk(|pat| {
            if let PatKind::Error(e) = pat.kind
                && error.is_none()
            {
                error = Some(e);
            }
            error.is_none()
        });
        match error {
            None => Ok(()),
            Some(e) => Err(e),
        }
    }

    /// Walk the pattern in left-to-right order.
    ///
    /// If you always want to recurse, prefer this method over `walk`.
    pub fn walk_always(&self, mut it: impl FnMut(&Pat<'tcx>)) {
        self.walk(|p| {
            it(p);
            true
        })
    }

    /// Whether this a never pattern.
    pub fn is_never_pattern(&self) -> bool {
        let mut is_never_pattern = false;
        self.walk(|pat| match &pat.kind {
            PatKind::Never => {
                is_never_pattern = true;
                false
            }
            PatKind::Or { pats } => {
                is_never_pattern = pats.iter().all(|p| p.is_never_pattern());
                false
            }
            _ => true,
        });
        is_never_pattern
    }
}

#[derive(Clone, Debug, HashStable, TypeVisitable)]
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

#[derive(Clone, Debug, HashStable, TypeVisitable)]
pub enum PatKind<'tcx> {
    /// A missing pattern, e.g. for an anonymous param in a bare fn like `fn f(u32)`.
    Missing,

    /// A wildcard pattern: `_`.
    Wild,

    AscribeUserType {
        ascription: Ascription<'tcx>,
        subpattern: Box<Pat<'tcx>>,
    },

    /// `x`, `ref x`, `x @ P`, etc.
    Binding {
        name: Symbol,
        #[type_visitable(ignore)]
        mode: BindingMode,
        #[type_visitable(ignore)]
        var: LocalVarId,
        ty: Ty<'tcx>,
        subpattern: Option<Box<Pat<'tcx>>>,

        /// Is this the leftmost occurrence of the binding, i.e., is `var` the
        /// `HirId` of this pattern?
        ///
        /// (The same binding can occur multiple times in different branches of
        /// an or-pattern, but only one of them will be primary.)
        is_primary: bool,
    },

    /// `Foo(...)` or `Foo{...}` or `Foo`, where `Foo` is a variant name from an ADT with
    /// multiple variants.
    Variant {
        adt_def: AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
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

    /// Deref pattern, written `box P` for now.
    DerefPattern {
        subpattern: Box<Pat<'tcx>>,
        mutability: hir::Mutability,
    },

    /// One of the following:
    /// * `&str` (represented as a valtree), which will be handled as a string pattern and thus
    ///   exhaustiveness checking will detect if you use the same string twice in different
    ///   patterns.
    /// * integer, bool, char or float (represented as a valtree), which will be handled by
    ///   exhaustiveness to cover exactly its own value, similar to `&str`, but these values are
    ///   much simpler.
    /// * `String`, if `string_deref_patterns` is enabled.
    Constant {
        value: mir::Const<'tcx>,
    },

    /// Pattern obtained by converting a constant (inline or named) to its pattern
    /// representation using `const_to_pat`. This is used for unsafety checking.
    ExpandedConstant {
        /// [DefId] of the constant item.
        def_id: DefId,
        /// The pattern that the constant lowered to.
        ///
        /// HACK: we need to keep the `DefId` of inline constants around for unsafety checking;
        /// therefore when a range pattern contains inline constants, we re-wrap the range pattern
        /// with the `ExpandedConstant` nodes that correspond to the range endpoints. Hence
        /// `subpattern` may actually be a range pattern, and `def_id` be the constant for one of
        /// its endpoints.
        subpattern: Box<Pat<'tcx>>,
    },

    Range(Arc<PatRange<'tcx>>),

    /// Matches against a slice, checking the length and extracting elements.
    /// irrefutable when there is a slice pattern and both `prefix` and `suffix` are empty.
    /// e.g., `&[ref xs @ ..]`.
    Slice {
        prefix: Box<[Pat<'tcx>]>,
        slice: Option<Box<Pat<'tcx>>>,
        suffix: Box<[Pat<'tcx>]>,
    },

    /// Fixed match against an array; irrefutable.
    Array {
        prefix: Box<[Pat<'tcx>]>,
        slice: Option<Box<Pat<'tcx>>>,
        suffix: Box<[Pat<'tcx>]>,
    },

    /// An or-pattern, e.g. `p | q`.
    /// Invariant: `pats.len() >= 2`.
    Or {
        pats: Box<[Pat<'tcx>]>,
    },

    /// A never pattern `!`.
    Never,

    /// An error has been encountered during lowering. We probably shouldn't report more lints
    /// related to this pattern.
    Error(ErrorGuaranteed),
}

/// A range pattern.
/// The boundaries must be of the same type and that type must be numeric.
#[derive(Clone, Debug, PartialEq, HashStable, TypeVisitable)]
pub struct PatRange<'tcx> {
    /// Must not be `PosInfinity`.
    pub lo: PatRangeBoundary<'tcx>,
    /// Must not be `NegInfinity`.
    pub hi: PatRangeBoundary<'tcx>,
    #[type_visitable(ignore)]
    pub end: RangeEnd,
    pub ty: Ty<'tcx>,
}

impl<'tcx> PatRange<'tcx> {
    /// Whether this range covers the full extent of possible values (best-effort, we ignore floats).
    #[inline]
    pub fn is_full_range(&self, tcx: TyCtxt<'tcx>) -> Option<bool> {
        let (min, max, size, bias) = match *self.ty.kind() {
            ty::Char => (0, std::char::MAX as u128, Size::from_bits(32), 0),
            ty::Int(ity) => {
                let size = Integer::from_int_ty(&tcx, ity).size();
                let max = size.truncate(u128::MAX);
                let bias = 1u128 << (size.bits() - 1);
                (0, max, size, bias)
            }
            ty::Uint(uty) => {
                let size = Integer::from_uint_ty(&tcx, uty).size();
                let max = size.unsigned_int_max();
                (0, max, size, 0)
            }
            _ => return None,
        };

        // We want to compare ranges numerically, but the order of the bitwise representation of
        // signed integers does not match their numeric order. Thus, to correct the ordering, we
        // need to shift the range of signed integers to correct the comparison. This is achieved by
        // XORing with a bias (see pattern/deconstruct_pat.rs for another pertinent example of this
        // pattern).
        //
        // Also, for performance, it's important to only do the second `try_to_bits` if necessary.
        let lo_is_min = match self.lo {
            PatRangeBoundary::NegInfinity => true,
            PatRangeBoundary::Finite(value) => {
                let lo = value.try_to_bits(size).unwrap() ^ bias;
                lo <= min
            }
            PatRangeBoundary::PosInfinity => false,
        };
        if lo_is_min {
            let hi_is_max = match self.hi {
                PatRangeBoundary::NegInfinity => false,
                PatRangeBoundary::Finite(value) => {
                    let hi = value.try_to_bits(size).unwrap() ^ bias;
                    hi > max || hi == max && self.end == RangeEnd::Included
                }
                PatRangeBoundary::PosInfinity => true,
            };
            if hi_is_max {
                return Some(true);
            }
        }
        Some(false)
    }

    #[inline]
    pub fn contains(
        &self,
        value: mir::Const<'tcx>,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> Option<bool> {
        use Ordering::*;
        debug_assert_eq!(self.ty, value.ty());
        let ty = self.ty;
        let value = PatRangeBoundary::Finite(value);
        // For performance, it's important to only do the second comparison if necessary.
        Some(
            match self.lo.compare_with(value, ty, tcx, typing_env)? {
                Less | Equal => true,
                Greater => false,
            } && match value.compare_with(self.hi, ty, tcx, typing_env)? {
                Less => true,
                Equal => self.end == RangeEnd::Included,
                Greater => false,
            },
        )
    }

    #[inline]
    pub fn overlaps(
        &self,
        other: &Self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> Option<bool> {
        use Ordering::*;
        debug_assert_eq!(self.ty, other.ty);
        // For performance, it's important to only do the second comparison if necessary.
        Some(
            match other.lo.compare_with(self.hi, self.ty, tcx, typing_env)? {
                Less => true,
                Equal => self.end == RangeEnd::Included,
                Greater => false,
            } && match self.lo.compare_with(other.hi, self.ty, tcx, typing_env)? {
                Less => true,
                Equal => other.end == RangeEnd::Included,
                Greater => false,
            },
        )
    }
}

impl<'tcx> fmt::Display for PatRange<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let PatRangeBoundary::Finite(value) = &self.lo {
            write!(f, "{value}")?;
        }
        if let PatRangeBoundary::Finite(value) = &self.hi {
            write!(f, "{}", self.end)?;
            write!(f, "{value}")?;
        } else {
            // `0..` is parsed as an inclusive range, we must display it correctly.
            write!(f, "..")?;
        }
        Ok(())
    }
}

/// A (possibly open) boundary of a range pattern.
/// If present, the const must be of a numeric type.
#[derive(Copy, Clone, Debug, PartialEq, HashStable, TypeVisitable)]
pub enum PatRangeBoundary<'tcx> {
    Finite(mir::Const<'tcx>),
    NegInfinity,
    PosInfinity,
}

impl<'tcx> PatRangeBoundary<'tcx> {
    #[inline]
    pub fn is_finite(self) -> bool {
        matches!(self, Self::Finite(..))
    }
    #[inline]
    pub fn as_finite(self) -> Option<mir::Const<'tcx>> {
        match self {
            Self::Finite(value) => Some(value),
            Self::NegInfinity | Self::PosInfinity => None,
        }
    }
    pub fn eval_bits(
        self,
        ty: Ty<'tcx>,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> u128 {
        match self {
            Self::Finite(value) => value.eval_bits(tcx, typing_env),
            Self::NegInfinity => {
                // Unwrap is ok because the type is known to be numeric.
                ty.numeric_min_and_max_as_bits(tcx).unwrap().0
            }
            Self::PosInfinity => {
                // Unwrap is ok because the type is known to be numeric.
                ty.numeric_min_and_max_as_bits(tcx).unwrap().1
            }
        }
    }

    #[instrument(skip(tcx, typing_env), level = "debug", ret)]
    pub fn compare_with(
        self,
        other: Self,
        ty: Ty<'tcx>,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> Option<Ordering> {
        use PatRangeBoundary::*;
        match (self, other) {
            // When comparing with infinities, we must remember that `0u8..` and `0u8..=255`
            // describe the same range. These two shortcuts are ok, but for the rest we must check
            // bit values.
            (PosInfinity, PosInfinity) => return Some(Ordering::Equal),
            (NegInfinity, NegInfinity) => return Some(Ordering::Equal),

            // This code is hot when compiling matches with many ranges. So we
            // special-case extraction of evaluated scalars for speed, for types where
            // we can do scalar comparisons. E.g. `unicode-normalization` has
            // many ranges such as '\u{037A}'..='\u{037F}', and chars can be compared
            // in this way.
            (Finite(a), Finite(b)) if matches!(ty.kind(), ty::Int(_) | ty::Uint(_) | ty::Char) => {
                if let (Some(a), Some(b)) = (a.try_to_scalar_int(), b.try_to_scalar_int()) {
                    let sz = ty.primitive_size(tcx);
                    let cmp = match ty.kind() {
                        ty::Uint(_) | ty::Char => a.to_uint(sz).cmp(&b.to_uint(sz)),
                        ty::Int(_) => a.to_int(sz).cmp(&b.to_int(sz)),
                        _ => unreachable!(),
                    };
                    return Some(cmp);
                }
            }
            _ => {}
        }

        let a = self.eval_bits(ty, tcx, typing_env);
        let b = other.eval_bits(ty, tcx, typing_env);

        match ty.kind() {
            ty::Float(ty::FloatTy::F16) => {
                use rustc_apfloat::Float;
                let a = rustc_apfloat::ieee::Half::from_bits(a);
                let b = rustc_apfloat::ieee::Half::from_bits(b);
                a.partial_cmp(&b)
            }
            ty::Float(ty::FloatTy::F32) => {
                use rustc_apfloat::Float;
                let a = rustc_apfloat::ieee::Single::from_bits(a);
                let b = rustc_apfloat::ieee::Single::from_bits(b);
                a.partial_cmp(&b)
            }
            ty::Float(ty::FloatTy::F64) => {
                use rustc_apfloat::Float;
                let a = rustc_apfloat::ieee::Double::from_bits(a);
                let b = rustc_apfloat::ieee::Double::from_bits(b);
                a.partial_cmp(&b)
            }
            ty::Float(ty::FloatTy::F128) => {
                use rustc_apfloat::Float;
                let a = rustc_apfloat::ieee::Quad::from_bits(a);
                let b = rustc_apfloat::ieee::Quad::from_bits(b);
                a.partial_cmp(&b)
            }
            ty::Int(ity) => {
                let size = rustc_abi::Integer::from_int_ty(&tcx, *ity).size();
                let a = size.sign_extend(a) as i128;
                let b = size.sign_extend(b) as i128;
                Some(a.cmp(&b))
            }
            ty::Uint(_) | ty::Char => Some(a.cmp(&b)),
            _ => bug!(),
        }
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(Block, 48);
    static_assert_size!(Expr<'_>, 72);
    static_assert_size!(ExprKind<'_>, 40);
    static_assert_size!(Pat<'_>, 64);
    static_assert_size!(PatKind<'_>, 48);
    static_assert_size!(Stmt<'_>, 48);
    static_assert_size!(StmtKind<'_>, 48);
    // tidy-alphabetical-end
}
