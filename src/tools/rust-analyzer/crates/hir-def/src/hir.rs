//! This module describes hir-level representation of expressions.
//!
//! This representation is:
//!
//! 1. Identity-based. Each expression has an `id`, so we can distinguish
//!    between different `1` in `1 + 1`.
//! 2. Independent of syntax. Though syntactic provenance information can be
//!    attached separately via id-based side map.
//! 3. Unresolved. Paths are stored as sequences of names, and not as defs the
//!    names refer to.
//! 4. Desugared. There's no `if let`.
//!
//! See also a neighboring `body` module.

pub mod format_args;
pub mod generics;
pub mod type_ref;

use std::fmt;

use hir_expand::{MacroDefId, name::Name};
use intern::Symbol;
use la_arena::Idx;
use rustc_apfloat::ieee::{Half as f16, Quad as f128};
use syntax::ast;
use type_ref::TypeRefId;

use crate::{
    BlockId,
    builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint},
    expr_store::{
        HygieneId,
        path::{GenericArgs, Path},
    },
    type_ref::{Mutability, Rawness},
};

pub use syntax::ast::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp};

pub type BindingId = Idx<Binding>;

pub type ExprId = Idx<Expr>;

pub type PatId = Idx<Pat>;

// FIXME: Encode this as a single u32, we won't ever reach all 32 bits especially given these counts
// are local to the body.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, salsa::Update)]
pub enum ExprOrPatId {
    ExprId(ExprId),
    PatId(PatId),
}

impl ExprOrPatId {
    pub fn as_expr(self) -> Option<ExprId> {
        match self {
            Self::ExprId(v) => Some(v),
            _ => None,
        }
    }

    pub fn is_expr(&self) -> bool {
        matches!(self, Self::ExprId(_))
    }

    pub fn as_pat(self) -> Option<PatId> {
        match self {
            Self::PatId(v) => Some(v),
            _ => None,
        }
    }

    pub fn is_pat(&self) -> bool {
        matches!(self, Self::PatId(_))
    }
}
stdx::impl_from!(ExprId, PatId for ExprOrPatId);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Label {
    pub name: Name,
}
pub type LabelId = Idx<Label>;

// We leave float values as a string to avoid double rounding.
// For PartialEq, string comparison should work, as ordering is not important
// https://github.com/rust-lang/rust-analyzer/issues/12380#issuecomment-1137284360
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FloatTypeWrapper(Symbol);

// FIXME(#17451): Use builtin types once stabilised.
impl FloatTypeWrapper {
    pub fn new(sym: Symbol) -> Self {
        Self(sym)
    }

    pub fn to_f128(&self) -> f128 {
        self.0.as_str().parse().unwrap_or_default()
    }

    pub fn to_f64(&self) -> f64 {
        self.0.as_str().parse().unwrap_or_default()
    }

    pub fn to_f32(&self) -> f32 {
        self.0.as_str().parse().unwrap_or_default()
    }

    pub fn to_f16(&self) -> f16 {
        self.0.as_str().parse().unwrap_or_default()
    }
}

impl fmt::Display for FloatTypeWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.as_str())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Literal {
    String(Symbol),
    ByteString(Box<[u8]>),
    CString(Box<[u8]>),
    Char(char),
    Bool(bool),
    Int(i128, Option<BuiltinInt>),
    Uint(u128, Option<BuiltinUint>),
    // Here we are using a wrapper around float because float primitives do not implement Eq, so they
    // could not be used directly here, to understand how the wrapper works go to definition of
    // FloatTypeWrapper
    Float(FloatTypeWrapper, Option<BuiltinFloat>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
/// Used in range patterns.
pub enum LiteralOrConst {
    Literal(Literal),
    Const(PatId),
}

impl Literal {
    pub fn negate(self) -> Option<Self> {
        if let Literal::Int(i, k) = self { Some(Literal::Int(-i, k)) } else { None }
    }
}

impl From<ast::LiteralKind> for Literal {
    fn from(ast_lit_kind: ast::LiteralKind) -> Self {
        use ast::LiteralKind;
        match ast_lit_kind {
            LiteralKind::IntNumber(lit) => {
                if let builtin @ Some(_) = lit.suffix().and_then(BuiltinFloat::from_suffix) {
                    Literal::Float(
                        FloatTypeWrapper::new(Symbol::intern(&lit.value_string())),
                        builtin,
                    )
                } else if let builtin @ Some(_) = lit.suffix().and_then(BuiltinUint::from_suffix) {
                    Literal::Uint(lit.value().unwrap_or(0), builtin)
                } else {
                    let builtin = lit.suffix().and_then(BuiltinInt::from_suffix);
                    Literal::Int(lit.value().unwrap_or(0) as i128, builtin)
                }
            }
            LiteralKind::FloatNumber(lit) => {
                let ty = lit.suffix().and_then(BuiltinFloat::from_suffix);
                Literal::Float(FloatTypeWrapper::new(Symbol::intern(&lit.value_string())), ty)
            }
            LiteralKind::ByteString(bs) => {
                let text = bs.value().map_or_else(|_| Default::default(), Box::from);
                Literal::ByteString(text)
            }
            LiteralKind::String(s) => {
                let text = s.value().map_or_else(|_| Symbol::empty(), |it| Symbol::intern(&it));
                Literal::String(text)
            }
            LiteralKind::CString(s) => {
                let text = s.value().map_or_else(|_| Default::default(), Box::from);
                Literal::CString(text)
            }
            LiteralKind::Byte(b) => {
                Literal::Uint(b.value().unwrap_or_default() as u128, Some(BuiltinUint::U8))
            }
            LiteralKind::Char(c) => Literal::Char(c.value().unwrap_or_default()),
            LiteralKind::Bool(val) => Literal::Bool(val),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum RecordSpread {
    None,
    FieldDefaults,
    Expr(ExprId),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Expr {
    /// This is produced if the syntax tree does not have a required expression piece.
    Missing,
    Path(Path),
    If {
        condition: ExprId,
        then_branch: ExprId,
        else_branch: Option<ExprId>,
    },
    Let {
        pat: PatId,
        expr: ExprId,
    },
    Block {
        id: Option<BlockId>,
        statements: Box<[Statement]>,
        tail: Option<ExprId>,
        label: Option<LabelId>,
    },
    Async {
        id: Option<BlockId>,
        statements: Box<[Statement]>,
        tail: Option<ExprId>,
    },
    Const(ExprId),
    // FIXME: Fold this into Block with an unsafe flag?
    Unsafe {
        id: Option<BlockId>,
        statements: Box<[Statement]>,
        tail: Option<ExprId>,
    },
    Loop {
        body: ExprId,
        label: Option<LabelId>,
    },
    Call {
        callee: ExprId,
        args: Box<[ExprId]>,
    },
    MethodCall {
        receiver: ExprId,
        method_name: Name,
        args: Box<[ExprId]>,
        generic_args: Option<Box<GenericArgs>>,
    },
    Match {
        expr: ExprId,
        arms: Box<[MatchArm]>,
    },
    Continue {
        label: Option<LabelId>,
    },
    Break {
        expr: Option<ExprId>,
        label: Option<LabelId>,
    },
    Return {
        expr: Option<ExprId>,
    },
    Become {
        expr: ExprId,
    },
    Yield {
        expr: Option<ExprId>,
    },
    Yeet {
        expr: Option<ExprId>,
    },
    RecordLit {
        path: Option<Box<Path>>,
        fields: Box<[RecordLitField]>,
        spread: RecordSpread,
    },
    Field {
        expr: ExprId,
        name: Name,
    },
    Await {
        expr: ExprId,
    },
    Cast {
        expr: ExprId,
        type_ref: TypeRefId,
    },
    Ref {
        expr: ExprId,
        rawness: Rawness,
        mutability: Mutability,
    },
    Box {
        expr: ExprId,
    },
    UnaryOp {
        expr: ExprId,
        op: UnaryOp,
    },
    /// `op` cannot be bare `=` (but can be `op=`), these are lowered to `Assignment` instead.
    BinaryOp {
        lhs: ExprId,
        rhs: ExprId,
        op: Option<BinaryOp>,
    },
    // Assignments need a special treatment because of destructuring assignment.
    Assignment {
        target: PatId,
        value: ExprId,
    },
    Range {
        lhs: Option<ExprId>,
        rhs: Option<ExprId>,
        range_type: RangeOp,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    Closure {
        args: Box<[PatId]>,
        arg_types: Box<[Option<TypeRefId>]>,
        ret_type: Option<TypeRefId>,
        body: ExprId,
        closure_kind: ClosureKind,
        capture_by: CaptureBy,
    },
    Tuple {
        exprs: Box<[ExprId]>,
    },
    Array(Array),
    Literal(Literal),
    Underscore,
    OffsetOf(OffsetOf),
    InlineAsm(InlineAsm),
}

impl Expr {
    pub fn precedence(&self) -> ast::prec::ExprPrecedence {
        use ast::prec::ExprPrecedence;

        match self {
            Expr::Array(_)
            | Expr::InlineAsm(_)
            | Expr::Block { .. }
            | Expr::Unsafe { .. }
            | Expr::Const(_)
            | Expr::Async { .. }
            | Expr::If { .. }
            | Expr::Literal(_)
            | Expr::Loop { .. }
            | Expr::Match { .. }
            | Expr::Missing
            | Expr::Path(_)
            | Expr::RecordLit { .. }
            | Expr::Tuple { .. }
            | Expr::OffsetOf(_)
            | Expr::Underscore => ExprPrecedence::Unambiguous,

            Expr::Await { .. }
            | Expr::Call { .. }
            | Expr::Field { .. }
            | Expr::Index { .. }
            | Expr::MethodCall { .. } => ExprPrecedence::Postfix,

            Expr::Box { .. } | Expr::Let { .. } | Expr::UnaryOp { .. } | Expr::Ref { .. } => {
                ExprPrecedence::Prefix
            }

            Expr::Cast { .. } => ExprPrecedence::Cast,

            Expr::BinaryOp { op, .. } => match op {
                None => ExprPrecedence::Unambiguous,
                Some(BinaryOp::LogicOp(LogicOp::Or)) => ExprPrecedence::LOr,
                Some(BinaryOp::LogicOp(LogicOp::And)) => ExprPrecedence::LAnd,
                Some(BinaryOp::CmpOp(_)) => ExprPrecedence::Compare,
                Some(BinaryOp::Assignment { .. }) => ExprPrecedence::Assign,
                Some(BinaryOp::ArithOp(arith_op)) => match arith_op {
                    ArithOp::Add | ArithOp::Sub => ExprPrecedence::Sum,
                    ArithOp::Mul | ArithOp::Div | ArithOp::Rem => ExprPrecedence::Product,
                    ArithOp::Shl | ArithOp::Shr => ExprPrecedence::Shift,
                    ArithOp::BitXor => ExprPrecedence::BitXor,
                    ArithOp::BitOr => ExprPrecedence::BitOr,
                    ArithOp::BitAnd => ExprPrecedence::BitAnd,
                },
            },

            Expr::Assignment { .. } => ExprPrecedence::Assign,

            Expr::Become { .. }
            | Expr::Break { .. }
            | Expr::Closure { .. }
            | Expr::Return { .. }
            | Expr::Yeet { .. }
            | Expr::Yield { .. } => ExprPrecedence::Jump,

            Expr::Continue { .. } => ExprPrecedence::Unambiguous,

            Expr::Range { .. } => ExprPrecedence::Range,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffsetOf {
    pub container: TypeRefId,
    pub fields: Box<[Name]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineAsm {
    pub operands: Box<[(Option<Name>, AsmOperand)]>,
    pub options: AsmOptions,
    pub kind: InlineAsmKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InlineAsmKind {
    /// `asm!()`.
    Asm,
    /// `global_asm!()`.
    GlobalAsm,
    /// `naked_asm!()`.
    NakedAsm,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsmOptions(u16);
bitflags::bitflags! {
    impl AsmOptions: u16 {
        const PURE            = 1 << 0;
        const NOMEM           = 1 << 1;
        const READONLY        = 1 << 2;
        const PRESERVES_FLAGS = 1 << 3;
        const NORETURN        = 1 << 4;
        const NOSTACK         = 1 << 5;
        const ATT_SYNTAX      = 1 << 6;
        const RAW             = 1 << 7;
        const MAY_UNWIND      = 1 << 8;
    }
}

impl AsmOptions {
    pub const COUNT: usize = Self::all().bits().count_ones() as usize;

    pub const GLOBAL_OPTIONS: Self = Self::ATT_SYNTAX.union(Self::RAW);
    pub const NAKED_OPTIONS: Self = Self::ATT_SYNTAX.union(Self::RAW).union(Self::NORETURN);

    pub fn human_readable_names(&self) -> Vec<&'static str> {
        let mut options = vec![];

        if self.contains(AsmOptions::PURE) {
            options.push("pure");
        }
        if self.contains(AsmOptions::NOMEM) {
            options.push("nomem");
        }
        if self.contains(AsmOptions::READONLY) {
            options.push("readonly");
        }
        if self.contains(AsmOptions::PRESERVES_FLAGS) {
            options.push("preserves_flags");
        }
        if self.contains(AsmOptions::NORETURN) {
            options.push("noreturn");
        }
        if self.contains(AsmOptions::NOSTACK) {
            options.push("nostack");
        }
        if self.contains(AsmOptions::ATT_SYNTAX) {
            options.push("att_syntax");
        }
        if self.contains(AsmOptions::RAW) {
            options.push("raw");
        }
        if self.contains(AsmOptions::MAY_UNWIND) {
            options.push("may_unwind");
        }

        options
    }
}

impl std::fmt::Debug for AsmOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum AsmOperand {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: ExprId,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        expr: Option<ExprId>,
        late: bool,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        expr: ExprId,
        late: bool,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        in_expr: ExprId,
        out_expr: Option<ExprId>,
        late: bool,
    },
    Label(ExprId),
    Const(ExprId),
    Sym(Path),
}

impl AsmOperand {
    pub fn reg(&self) -> Option<&InlineAsmRegOrRegClass> {
        match self {
            Self::In { reg, .. }
            | Self::Out { reg, .. }
            | Self::InOut { reg, .. }
            | Self::SplitInOut { reg, .. } => Some(reg),
            Self::Const { .. } | Self::Sym { .. } | Self::Label { .. } => None,
        }
    }

    pub fn is_clobber(&self) -> bool {
        matches!(self, AsmOperand::Out { reg: InlineAsmRegOrRegClass::Reg(_), late: _, expr: None })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum InlineAsmRegOrRegClass {
    Reg(Symbol),
    RegClass(Symbol),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureKind {
    Closure,
    Coroutine(Movability),
    Async,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureBy {
    /// `move |x| y + x`.
    Value,
    /// `move` keyword was not specified.
    Ref,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Movability {
    Static,
    Movable,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Array {
    ElementList { elements: Box<[ExprId]> },
    Repeat { initializer: ExprId, repeat: ExprId },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MatchArm {
    pub pat: PatId,
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
    Let {
        pat: PatId,
        type_ref: Option<TypeRefId>,
        initializer: Option<ExprId>,
        else_branch: Option<ExprId>,
    },
    Expr {
        expr: ExprId,
        has_semi: bool,
    },
    Item(Item),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    MacroDef(Box<MacroDefId>),
    Other,
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
    pub fn new(is_mutable: bool, is_ref: bool) -> Self {
        match (is_mutable, is_ref) {
            (true, true) => BindingAnnotation::RefMut,
            (false, true) => BindingAnnotation::Ref,
            (true, false) => BindingAnnotation::Mutable,
            (false, false) => BindingAnnotation::Unannotated,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BindingProblems {
    /// <https://doc.rust-lang.org/stable/error_codes/E0416.html>
    BoundMoreThanOnce,
    /// <https://doc.rust-lang.org/stable/error_codes/E0409.html>
    BoundInconsistently,
    /// <https://doc.rust-lang.org/stable/error_codes/E0408.html>
    NotBoundAcrossAll,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Binding {
    pub name: Name,
    pub mode: BindingAnnotation,
    pub problems: Option<BindingProblems>,
    /// Note that this may not be the direct `SyntaxContextId` of the binding's expansion, because transparent
    /// expansions are attributed to their parent expansion (recursively).
    pub hygiene: HygieneId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecordFieldPat {
    pub name: Name,
    pub pat: PatId,
}

/// Close relative to rustc's hir::PatKind
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Pat {
    Missing,
    Wild,
    Tuple {
        args: Box<[PatId]>,
        ellipsis: Option<u32>,
    },
    Or(Box<[PatId]>),
    Record {
        path: Option<Box<Path>>,
        args: Box<[RecordFieldPat]>,
        ellipsis: bool,
    },
    Range {
        start: Option<ExprId>,
        end: Option<ExprId>,
        range_type: RangeOp,
    },
    Slice {
        prefix: Box<[PatId]>,
        slice: Option<PatId>,
        suffix: Box<[PatId]>,
    },
    /// This might refer to a variable if a single segment path (specifically, on destructuring assignment).
    Path(Path),
    Lit(ExprId),
    Bind {
        id: BindingId,
        subpat: Option<PatId>,
    },
    TupleStruct {
        path: Option<Box<Path>>,
        args: Box<[PatId]>,
        ellipsis: Option<u32>,
    },
    Ref {
        pat: PatId,
        mutability: Mutability,
    },
    Box {
        inner: PatId,
    },
    ConstBlock(ExprId),
    /// An expression inside a pattern. That can only occur inside assignments.
    ///
    /// E.g. in `(a, *b) = (1, &mut 2)`, `*b` is an expression.
    Expr(ExprId),
}

impl Pat {
    pub fn walk_child_pats(&self, mut f: impl FnMut(PatId)) {
        match self {
            Pat::Range { .. }
            | Pat::Lit(..)
            | Pat::Path(..)
            | Pat::ConstBlock(..)
            | Pat::Wild
            | Pat::Missing
            | Pat::Expr(_) => {}
            Pat::Bind { subpat, .. } => {
                subpat.iter().copied().for_each(f);
            }
            Pat::Or(args) | Pat::Tuple { args, .. } | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, slice, suffix } => {
                let total_iter = prefix.iter().chain(slice.iter()).chain(suffix.iter());
                total_iter.copied().for_each(f);
            }
            Pat::Record { args, .. } => {
                args.iter().map(|f| f.pat).for_each(f);
            }
            Pat::Box { inner } => f(*inner),
        }
    }
}
