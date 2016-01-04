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

use rustc::mir::repr::{BinOp, BorrowKind, Field, Literal, Mutability, UnOp, ItemKind};
use rustc::middle::const_eval::ConstVal;
use rustc::middle::def_id::DefId;
use rustc::middle::region::CodeExtent;
use rustc::middle::subst::Substs;
use rustc::middle::ty::{AdtDef, ClosureSubsts, Region, Ty};
use rustc_front::hir;
use syntax::ast;
use syntax::codemap::Span;
use self::cx::Cx;

pub mod cx;

#[derive(Clone, Debug)]
pub struct ItemRef<'tcx> {
    pub ty: Ty<'tcx>,
    pub kind: ItemKind,
    pub def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

#[derive(Clone, Debug)]
pub struct Block<'tcx> {
    pub extent: CodeExtent,
    pub span: Span,
    pub stmts: Vec<StmtRef<'tcx>>,
    pub expr: Option<ExprRef<'tcx>>,
}

#[derive(Clone, Debug)]
pub enum StmtRef<'tcx> {
    Mirror(Box<Stmt<'tcx>>),
}

#[derive(Clone, Debug)]
pub struct Stmt<'tcx> {
    pub span: Span,
    pub kind: StmtKind<'tcx>,
}

#[derive(Clone, Debug)]
pub enum StmtKind<'tcx> {
    Expr {
        /// scope for this statement; may be used as lifetime of temporaries
        scope: CodeExtent,

        /// expression being evaluated in this statement
        expr: ExprRef<'tcx>,
    },

    Let {
        /// scope for variables bound in this let; covers this and
        /// remaining statements in block
        remainder_scope: CodeExtent,

        /// scope for the initialization itself; might be used as
        /// lifetime of temporaries
        init_scope: CodeExtent,

        /// let <PAT> = ...
        pattern: Pattern<'tcx>,

        /// let pat = <INIT> ...
        initializer: Option<ExprRef<'tcx>>,

        /// let pat = init; <STMTS>
        stmts: Vec<StmtRef<'tcx>>,
    },
}

// The Hair trait implementor translates their expressions (`&'tcx H::Expr`)
// into instances of this `Expr` enum. This translation can be done
// basically as lazilly or as eagerly as desired: every recursive
// reference to an expression in this enum is an `ExprRef<'tcx>`, which
// may in turn be another instance of this enum (boxed), or else an
// untranslated `&'tcx H::Expr`. Note that instances of `Expr` are very
// shortlived. They are created by `Hair::to_expr`, analyzed and
// converted into MIR, and then discarded.
//
// If you compare `Expr` to the full compiler AST, you will see it is
// a good bit simpler. In fact, a number of the more straight-forward
// MIR simplifications are already done in the impl of `Hair`. For
// example, method calls and overloaded operators are absent: they are
// expected to be converted into `Expr::Call` instances.
#[derive(Clone, Debug)]
pub struct Expr<'tcx> {
    // type of this expression
    pub ty: Ty<'tcx>,

    // lifetime of this expression if it should be spilled into a
    // temporary; should be None only if in a constant context
    pub temp_lifetime: Option<CodeExtent>,

    // span of the expression in the source
    pub span: Span,

    // kind of expression
    pub kind: ExprKind<'tcx>,
}

#[derive(Clone, Debug)]
pub enum ExprKind<'tcx> {
    Scope {
        extent: CodeExtent,
        value: ExprRef<'tcx>,
    },
    Box {
        value: ExprRef<'tcx>,
    },
    Call {
        fun: ExprRef<'tcx>,
        args: Vec<ExprRef<'tcx>>,
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
    },
    Unary {
        op: UnOp,
        arg: ExprRef<'tcx>,
    }, // NOT overloaded!
    Cast {
        source: ExprRef<'tcx>,
    },
    ReifyFnPointer {
        source: ExprRef<'tcx>,
    },
    UnsafeFnPointer {
        source: ExprRef<'tcx>,
    },
    Unsize {
        source: ExprRef<'tcx>,
    },
    If {
        condition: ExprRef<'tcx>,
        then: ExprRef<'tcx>,
        otherwise: Option<ExprRef<'tcx>>,
    },
    Loop {
        condition: Option<ExprRef<'tcx>>,
        body: ExprRef<'tcx>,
    },
    Match {
        discriminant: ExprRef<'tcx>,
        arms: Vec<Arm<'tcx>>,
    },
    Block {
        body: &'tcx hir::Block,
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
        id: ast::NodeId,
    },
    SelfRef, // first argument, used for self in a closure
    StaticRef {
        id: DefId,
    },
    Borrow {
        region: Region,
        borrow_kind: BorrowKind,
        arg: ExprRef<'tcx>,
    },
    Break {
        label: Option<CodeExtent>,
    },
    Continue {
        label: Option<CodeExtent>,
    },
    Return {
        value: Option<ExprRef<'tcx>>,
    },
    Repeat {
        value: ExprRef<'tcx>,
        // FIXME(#29789): Add a separate hair::Constant<'tcx> so this could be more explicit about
        // its contained data. Currently this should only contain expression of ExprKind::Literal
        // kind.
        count: ExprRef<'tcx>,
    },
    Vec {
        fields: Vec<ExprRef<'tcx>>,
    },
    Tuple {
        fields: Vec<ExprRef<'tcx>>,
    },
    Adt {
        adt_def: AdtDef<'tcx>,
        variant_index: usize,
        substs: &'tcx Substs<'tcx>,
        fields: Vec<FieldExprRef<'tcx>>,
        base: Option<ExprRef<'tcx>>,
    },
    Closure {
        closure_id: DefId,
        substs: &'tcx ClosureSubsts<'tcx>,
        upvars: Vec<ExprRef<'tcx>>,
    },
    Literal {
        literal: Literal<'tcx>,
    },
    InlineAsm {
        asm: &'tcx hir::InlineAsm,
    },
}

#[derive(Clone, Debug)]
pub enum ExprRef<'tcx> {
    Hair(&'tcx hir::Expr),
    Mirror(Box<Expr<'tcx>>),
}

#[derive(Clone, Debug)]
pub struct FieldExprRef<'tcx> {
    pub name: Field,
    pub expr: ExprRef<'tcx>,
}

#[derive(Clone, Debug)]
pub struct Arm<'tcx> {
    pub patterns: Vec<Pattern<'tcx>>,
    pub guard: Option<ExprRef<'tcx>>,
    pub body: ExprRef<'tcx>,
}

#[derive(Clone, Debug)]
pub struct Pattern<'tcx> {
    pub ty: Ty<'tcx>,
    pub span: Span,
    pub kind: Box<PatternKind<'tcx>>,
}

#[derive(Copy, Clone, Debug)]
pub enum LogicalOp {
    And,
    Or,
}

#[derive(Clone, Debug)]
pub enum PatternKind<'tcx> {
    Wild,

    // x, ref x, x @ P, etc
    Binding {
        mutability: Mutability,
        name: ast::Name,
        mode: BindingMode,
        var: ast::NodeId,
        ty: Ty<'tcx>,
        subpattern: Option<Pattern<'tcx>>,
    },

    // Foo(...) or Foo{...} or Foo, where `Foo` is a variant name from an adt with >1 variants
    Variant {
        adt_def: AdtDef<'tcx>,
        variant_index: usize,
        subpatterns: Vec<FieldPattern<'tcx>>,
    },

    // (...), Foo(...), Foo{...}, or Foo, where `Foo` is a variant name from an adt with 1 variant
    Leaf {
        subpatterns: Vec<FieldPattern<'tcx>>,
    },

    Deref {
        subpattern: Pattern<'tcx>,
    }, // box P, &P, &mut P, etc

    Constant {
        value: ConstVal,
    },

    Range {
        lo: Literal<'tcx>,
        hi: Literal<'tcx>,
    },

    // matches against a slice, checking the length and extracting elements
    Slice {
        prefix: Vec<Pattern<'tcx>>,
        slice: Option<Pattern<'tcx>>,
        suffix: Vec<Pattern<'tcx>>,
    },

    // fixed match against an array, irrefutable
    Array {
        prefix: Vec<Pattern<'tcx>>,
        slice: Option<Pattern<'tcx>>,
        suffix: Vec<Pattern<'tcx>>,
    },
}

#[derive(Copy, Clone, Debug)]
pub enum BindingMode {
    ByValue,
    ByRef(Region, BorrowKind),
}

#[derive(Clone, Debug)]
pub struct FieldPattern<'tcx> {
    pub field: Field,
    pub pattern: Pattern<'tcx>,
}

///////////////////////////////////////////////////////////////////////////
// The Mirror trait

/// "Mirroring" is the process of converting from a HIR type into one
/// of the HAIR types defined in this file. This is basically a "on
/// the fly" desugaring step that hides a lot of the messiness in the
/// tcx. For example, the mirror of a `&'tcx hir::Expr` is an
/// `Expr<'tcx>`.
///
/// Mirroring is gradual: when you mirror an outer expression like `e1
/// + e2`, the references to the inner expressions `e1` and `e2` are
/// `ExprRef<'tcx>` instances, and they may or may not be eagerly
/// mirrored.  This allows a single AST node from the compiler to
/// expand into one or more Hair nodes, which lets the Hair nodes be
/// simpler.
pub trait Mirror<'tcx> {
    type Output;

    fn make_mirror<'a>(self, cx: &mut Cx<'a, 'tcx>) -> Self::Output;
}

impl<'tcx> Mirror<'tcx> for Expr<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror<'a>(self, _: &mut Cx<'a, 'tcx>) -> Expr<'tcx> {
        self
    }
}

impl<'tcx> Mirror<'tcx> for ExprRef<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror<'a>(self, hir: &mut Cx<'a, 'tcx>) -> Expr<'tcx> {
        match self {
            ExprRef::Hair(h) => h.make_mirror(hir),
            ExprRef::Mirror(m) => *m,
        }
    }
}

impl<'tcx> Mirror<'tcx> for Stmt<'tcx> {
    type Output = Stmt<'tcx>;

    fn make_mirror<'a>(self, _: &mut Cx<'a, 'tcx>) -> Stmt<'tcx> {
        self
    }
}

impl<'tcx> Mirror<'tcx> for StmtRef<'tcx> {
    type Output = Stmt<'tcx>;

    fn make_mirror<'a>(self, _: &mut Cx<'a,'tcx>) -> Stmt<'tcx> {
        match self {
            StmtRef::Mirror(m) => *m,
        }
    }
}

impl<'tcx> Mirror<'tcx> for Block<'tcx> {
    type Output = Block<'tcx>;

    fn make_mirror<'a>(self, _: &mut Cx<'a, 'tcx>) -> Block<'tcx> {
        self
    }
}
