// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The MIR is translated from some high-level abstract IR
//! (HAIR). This section defines the HAIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

use repr::{BinOp, BorrowKind, Field, Literal, Mutability, UnOp};
use std::fmt::Debug;
use std::hash::Hash;

pub trait Hair: Sized+Debug+Clone+Eq+Hash { // (*)

    // (*) the `Sized` and Debug` bounds are the only ones that really
    // make sense.  The rest are just there so that we can
    // `#[derive(Clone)]` on things that are parameterized over
    // `H:HAIR`. It's kind of lame.

    type VarId: Copy+Debug+Eq+Hash;                              // e.g., NodeId for a variable
    type DefId: Copy+Debug+Eq+Hash;                              // e.g., DefId
    type AdtDef: Copy+Debug+Eq+Hash;                             // e.g., AdtDef<'tcx>
    type Name: Copy+Debug+Eq+Hash;                               // e.g., ast::Name
    type Ident: Copy+Debug+Eq+Hash;                              // e.g., ast::Ident
    type InternedString: Clone+Debug+Eq+Hash;                    // e.g., InternedString
    type Bytes: Clone+Debug+Eq+Hash;                             // e.g., Rc<Vec<u8>>
    type Span: Copy+Debug+Eq;                                    // e.g., syntax::codemap::Span
    type Projection: Clone+Debug+Eq;                             // e.g., ty::ProjectionTy<'tcx>
    type Substs: Clone+Debug+Eq;                                 // e.g., substs::Substs<'tcx>
    type ClosureSubsts: Clone+Debug+Eq;                          // e.g., ty::ClosureSubsts<'tcx>
    type Ty: Clone+Debug+Eq;                                     // e.g., ty::Ty<'tcx>
    type Region: Copy+Debug;                                     // e.g., ty::Region
    type CodeExtent: Copy+Debug+Hash+Eq;                         // e.g., region::CodeExtent
    type Pattern: Clone+Debug+Mirror<Self,Output=Pattern<Self>>; // e.g., &P<ast::Pat>
    type Expr: Clone+Debug+Mirror<Self,Output=Expr<Self>>;       // e.g., &P<ast::Expr>
    type Stmt: Clone+Debug+Mirror<Self,Output=Stmt<Self>>;       // e.g., &P<ast::Stmt>
    type Block: Clone+Debug+Mirror<Self,Output=Block<Self>>;     // e.g., &P<ast::Block>
    type InlineAsm: Clone+Debug+Eq+Hash;                         // e.g., ast::InlineAsm

    /// Normalizes `ast` into the appropriate `mirror` type.
    fn mirror<M:Mirror<Self>>(&mut self, ast: M) -> M::Output {
        ast.make_mirror(self)
    }

    /// Returns the unit type `()`
    fn unit_ty(&mut self) -> Self::Ty;

    /// Returns the type `usize`.
    fn usize_ty(&mut self) -> Self::Ty;

    /// Returns the type `bool`.
    fn bool_ty(&mut self) -> Self::Ty;

    /// Returns a reference to `PartialEq::<T,T>::eq`
    fn partial_eq(&mut self, ty: Self::Ty) -> ItemRef<Self>;

    /// Returns a reference to `PartialOrd::<T,T>::le`
    fn partial_le(&mut self, ty: Self::Ty) -> ItemRef<Self>;

    /// Returns the number of variants for the given enum
    fn num_variants(&mut self, adt: Self::AdtDef) -> usize;

    fn fields(&mut self, adt: Self::AdtDef, variant_index: usize) -> Vec<Field<Self>>;

    /// true if a value of type `ty` (may) need to be dropped; this
    /// may return false even for non-Copy types if there is no
    /// destructor to execute. If correct result is not known, may be
    /// approximated by returning `true`; this will result in more
    /// drops but not incorrect code.
    fn needs_drop(&mut self, ty: Self::Ty, span: Self::Span) -> bool;

    /// Report an internal inconsistency.
    fn span_bug(&mut self, span: Self::Span, message: &str) -> !;
}

#[derive(Clone, Debug)]
pub struct ItemRef<H:Hair> {
    pub ty: H::Ty,
    pub def_id: H::DefId,
    pub substs: H::Substs,
}

#[derive(Clone, Debug)]
pub struct Block<H:Hair> {
    pub extent: H::CodeExtent,
    pub span: H::Span,
    pub stmts: Vec<StmtRef<H>>,
    pub expr: Option<ExprRef<H>>,
}

#[derive(Clone, Debug)]
pub enum StmtRef<H:Hair> {
    Hair(H::Stmt),
    Mirror(Box<Stmt<H>>),
}

#[derive(Clone, Debug)]
pub struct Stmt<H:Hair> {
    pub span: H::Span,
    pub kind: StmtKind<H>,
}

#[derive(Clone, Debug)]
pub enum StmtKind<H:Hair> {
    Expr {
        /// scope for this statement; may be used as lifetime of temporaries
        scope: H::CodeExtent,

        /// expression being evaluated in this statement
        expr: ExprRef<H>
    },

    Let {
        /// scope for variables bound in this let; covers this and
        /// remaining statements in block
        remainder_scope: H::CodeExtent,

        /// scope for the initialization itself; might be used as
        /// lifetime of temporaries
        init_scope: H::CodeExtent,

        /// let <PAT> = ...
        pattern: PatternRef<H>,

        /// let pat = <INIT> ...
        initializer: Option<ExprRef<H>>,

        /// let pat = init; <STMTS>
        stmts: Vec<StmtRef<H>>
    },
}

// The Hair trait implementor translates their expressions (`H::Expr`)
// into instances of this `Expr` enum. This translation can be done
// basically as lazilly or as eagerly as desired: every recursive
// reference to an expression in this enum is an `ExprRef<H>`, which
// may in turn be another instance of this enum (boxed), or else an
// untranslated `H::Expr`. Note that instances of `Expr` are very
// shortlived. They are created by `Hair::to_expr`, analyzed and
// converted into MIR, and then discarded.
//
// If you compare `Expr` to the full compiler AST, you will see it is
// a good bit simpler. In fact, a number of the more straight-forward
// MIR simplifications are already done in the impl of `Hair`. For
// example, method calls and overloaded operators are absent: they are
// expected to be converted into `Expr::Call` instances.
#[derive(Clone, Debug)]
pub struct Expr<H:Hair> {
    // type of this expression
    pub ty: H::Ty,

    // lifetime of this expression if it should be spilled into a
    // temporary; should be None only if in a constant context
    pub temp_lifetime: Option<H::CodeExtent>,

    // span of the expression in the source
    pub span: H::Span,

    // kind of expression
    pub kind: ExprKind<H>,
}

#[derive(Clone, Debug)]
pub enum ExprKind<H:Hair> {
    Scope { extent: H::CodeExtent, value: ExprRef<H> },
    Paren { arg: ExprRef<H> }, // ugh. should be able to remove this!
    Box { place: Option<ExprRef<H>>, value: ExprRef<H> },
    Call { fun: ExprRef<H>, args: Vec<ExprRef<H>> },
    Deref { arg: ExprRef<H> }, // NOT overloaded!
    Binary { op: BinOp, lhs: ExprRef<H>, rhs: ExprRef<H> }, // NOT overloaded!
    LogicalOp { op: LogicalOp, lhs: ExprRef<H>, rhs: ExprRef<H> },
    Unary { op: UnOp, arg: ExprRef<H> }, // NOT overloaded!
    Cast { source: ExprRef<H> },
    ReifyFnPointer { source: ExprRef<H> },
    UnsafeFnPointer { source: ExprRef<H> },
    Unsize { source: ExprRef<H> },
    If { condition: ExprRef<H>, then: ExprRef<H>, otherwise: Option<ExprRef<H>> },
    Loop { condition: Option<ExprRef<H>>, body: ExprRef<H>, },
    Match { discriminant: ExprRef<H>, arms: Vec<Arm<H>> },
    Block { body: H::Block },
    Assign { lhs: ExprRef<H>, rhs: ExprRef<H> },
    AssignOp { op: BinOp, lhs: ExprRef<H>, rhs: ExprRef<H> },
    Field { lhs: ExprRef<H>, name: Field<H> },
    Index { lhs: ExprRef<H>, index: ExprRef<H> },
    VarRef { id: H::VarId },
    SelfRef, // first argument, used for self in a closure
    StaticRef { id: H::DefId },
    Borrow { region: H::Region, borrow_kind: BorrowKind, arg: ExprRef<H> },
    Break { label: Option<H::CodeExtent> },
    Continue { label: Option<H::CodeExtent> },
    Return { value: Option<ExprRef<H>> },
    Repeat { value: ExprRef<H>, count: ExprRef<H> },
    Vec { fields: Vec<ExprRef<H>> },
    Tuple { fields: Vec<ExprRef<H>> },
    Adt { adt_def: H::AdtDef,
          variant_index: usize,
          substs: H::Substs,
          fields: Vec<FieldExprRef<H>>,
          base: Option<ExprRef<H>> },
    Closure { closure_id: H::DefId, substs: H::ClosureSubsts,
              upvars: Vec<ExprRef<H>> },
    Literal { literal: Literal<H> },
    InlineAsm { asm: H::InlineAsm },
}

#[derive(Clone, Debug)]
pub enum ExprRef<H:Hair> {
    Hair(H::Expr),
    Mirror(Box<Expr<H>>),
}

#[derive(Clone, Debug)]
pub struct FieldExprRef<H:Hair> {
    pub name: Field<H>,
    pub expr: ExprRef<H>,
}

#[derive(Clone, Debug)]
pub struct Arm<H:Hair> {
    pub patterns: Vec<PatternRef<H>>,
    pub guard: Option<ExprRef<H>>,
    pub body: ExprRef<H>,
}

#[derive(Clone, Debug)]
pub struct Pattern<H:Hair> {
    pub ty: H::Ty,
    pub span: H::Span,
    pub kind: PatternKind<H>,
}

#[derive(Copy, Clone, Debug)]
pub enum LogicalOp {
    And,
    Or
}

#[derive(Clone, Debug)]
pub enum PatternKind<H:Hair> {
    Wild,

    // x, ref x, x @ P, etc
    Binding { mutability: Mutability,
              name: H::Ident,
              mode: BindingMode<H>,
              var: H::VarId,
              ty: H::Ty,
              subpattern: Option<PatternRef<H>> },

    // Foo(...) or Foo{...} or Foo, where `Foo` is a variant name from an adt with >1 variants
    Variant { adt_def: H::AdtDef, variant_index: usize, subpatterns: Vec<FieldPatternRef<H>> },

    // (...), Foo(...), Foo{...}, or Foo, where `Foo` is a variant name from an adt with 1 variant
    Leaf { subpatterns: Vec<FieldPatternRef<H>> },

    Deref { subpattern: PatternRef<H> }, // box P, &P, &mut P, etc

    Constant { expr: ExprRef<H> },

    Range { lo: ExprRef<H>, hi: ExprRef<H> },

    // matches against a slice, checking the length and extracting elements
    Slice { prefix: Vec<PatternRef<H>>,
            slice: Option<PatternRef<H>>,
            suffix: Vec<PatternRef<H>> },

    // fixed match against an array, irrefutable
    Array { prefix: Vec<PatternRef<H>>,
            slice: Option<PatternRef<H>>,
            suffix: Vec<PatternRef<H>> },
}

#[derive(Copy, Clone, Debug)]
pub enum BindingMode<H:Hair> {
    ByValue,
    ByRef(H::Region, BorrowKind),
}

#[derive(Clone, Debug)]
pub enum PatternRef<H:Hair> {
    Hair(H::Pattern),
    Mirror(Box<Pattern<H>>),
}

#[derive(Clone, Debug)]
pub struct FieldPatternRef<H:Hair> {
    pub field: Field<H>,
    pub pattern: PatternRef<H>,
}

///////////////////////////////////////////////////////////////////////////
// The Mirror trait

/// "Mirroring" is the process of converting from a Hair type into one
/// of the types in this file. For example, the mirror of a `H::Expr`
/// is an `Expr<H>`. Mirroring is the point at which the actual IR is
/// converting into the more idealized representation described in
/// this file. Mirroring is gradual: when you mirror an outer
/// expression like `e1 + e2`, the references to the inner expressions
/// `e1` and `e2` are `ExprRef<H>` instances, and they may or may not
/// be eagerly mirrored.  This allows a single AST node from the
/// compiler to expand into one or more Hair nodes, which lets the Hair
/// nodes be simpler.
pub trait Mirror<H:Hair> {
    type Output;

    fn make_mirror(self, hir: &mut H) -> Self::Output;
}

impl<H:Hair> Mirror<H> for Expr<H> {
    type Output = Expr<H>;

    fn make_mirror(self, _: &mut H) -> Expr<H> {
        self
    }
}

impl<H:Hair> Mirror<H> for ExprRef<H> {
    type Output = Expr<H>;

    fn make_mirror(self, hir: &mut H) -> Expr<H> {
        match self {
            ExprRef::Hair(h) => h.make_mirror(hir),
            ExprRef::Mirror(m) => *m,
        }
    }
}

impl<H:Hair> Mirror<H> for Stmt<H> {
    type Output = Stmt<H>;

    fn make_mirror(self, _: &mut H) -> Stmt<H> {
        self
    }
}

impl<H:Hair> Mirror<H> for StmtRef<H> {
    type Output = Stmt<H>;

    fn make_mirror(self, hir: &mut H) -> Stmt<H> {
        match self {
            StmtRef::Hair(h) => h.make_mirror(hir),
            StmtRef::Mirror(m) => *m,
        }
    }
}

impl<H:Hair> Mirror<H> for Pattern<H> {
    type Output = Pattern<H>;

    fn make_mirror(self, _: &mut H) -> Pattern<H> {
        self
    }
}

impl<H:Hair> Mirror<H> for PatternRef<H> {
    type Output = Pattern<H>;

    fn make_mirror(self, hir: &mut H) -> Pattern<H> {
        match self {
            PatternRef::Hair(h) => h.make_mirror(hir),
            PatternRef::Mirror(m) => *m,
        }
    }
}

impl<H:Hair> Mirror<H> for Block<H> {
    type Output = Block<H>;

    fn make_mirror(self, _: &mut H) -> Block<H> {
        self
    }
}

