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

use rustc::mir::{BinOp, BorrowKind, Field, Literal, UnOp, TypedConstVal};
use rustc::hir::def_id::DefId;
use rustc::middle::region::CodeExtent;
use rustc::ty::subst::Substs;
use rustc::ty::{self, AdtDef, ClosureSubsts, Region, Ty};
use rustc::hir;
use syntax::ast;
use syntax_pos::Span;
use self::cx::Cx;

pub mod cx;

pub use rustc_const_eval::pattern::{BindingMode, Pattern, PatternKind, FieldPattern};

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
        initializer: Option<ExprRef<'tcx>>
    },
}

/// The Hair trait implementor translates their expressions (`&'tcx H::Expr`)
/// into instances of this `Expr` enum. This translation can be done
/// basically as lazilly or as eagerly as desired: every recursive
/// reference to an expression in this enum is an `ExprRef<'tcx>`, which
/// may in turn be another instance of this enum (boxed), or else an
/// untranslated `&'tcx H::Expr`. Note that instances of `Expr` are very
/// shortlived. They are created by `Hair::to_expr`, analyzed and
/// converted into MIR, and then discarded.
///
/// If you compare `Expr` to the full compiler AST, you will see it is
/// a good bit simpler. In fact, a number of the more straight-forward
/// MIR simplifications are already done in the impl of `Hair`. For
/// example, method calls and overloaded operators are absent: they are
/// expected to be converted into `Expr::Call` instances.
#[derive(Clone, Debug)]
pub struct Expr<'tcx> {
    /// type of this expression
    pub ty: Ty<'tcx>,

    /// lifetime of this expression if it should be spilled into a
    /// temporary; should be None only if in a constant context
    pub temp_lifetime: Option<CodeExtent>,

    /// span of the expression in the source
    pub span: Span,

    /// kind of expression
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
        value_extents: CodeExtent,
    },
    Call {
        ty: ty::Ty<'tcx>,
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
    Use {
        source: ExprRef<'tcx>,
    }, // Use a lexpr to get a vexpr.
    NeverToAny {
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
    /// first argument, used for self in a closure
    SelfRef,
    StaticRef {
        id: DefId,
    },
    Borrow {
        region: &'tcx Region,
        borrow_kind: BorrowKind,
        arg: ExprRef<'tcx>,
    },
    Break {
        label: Option<CodeExtent>,
        value: Option<ExprRef<'tcx>>,
    },
    Continue {
        label: Option<CodeExtent>,
    },
    Return {
        value: Option<ExprRef<'tcx>>,
    },
    Repeat {
        value: ExprRef<'tcx>,
        count: TypedConstVal<'tcx>,
    },
    Array {
        fields: Vec<ExprRef<'tcx>>,
    },
    Tuple {
        fields: Vec<ExprRef<'tcx>>,
    },
    Adt {
        adt_def: &'tcx AdtDef,
        variant_index: usize,
        substs: &'tcx Substs<'tcx>,
        fields: Vec<FieldExprRef<'tcx>>,
        base: Option<FruInfo<'tcx>>
    },
    Closure {
        closure_id: DefId,
        substs: ClosureSubsts<'tcx>,
        upvars: Vec<ExprRef<'tcx>>,
    },
    Literal {
        literal: Literal<'tcx>,
    },
    InlineAsm {
        asm: &'tcx hir::InlineAsm,
        outputs: Vec<ExprRef<'tcx>>,
        inputs: Vec<ExprRef<'tcx>>
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
pub struct FruInfo<'tcx> {
    pub base: ExprRef<'tcx>,
    pub field_types: Vec<Ty<'tcx>>
}

#[derive(Clone, Debug)]
pub struct Arm<'tcx> {
    pub patterns: Vec<Pattern<'tcx>>,
    pub guard: Option<ExprRef<'tcx>>,
    pub body: ExprRef<'tcx>,
}

#[derive(Copy, Clone, Debug)]
pub enum LogicalOp {
    And,
    Or,
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

    fn make_mirror<'a, 'gcx>(self, cx: &mut Cx<'a, 'gcx, 'tcx>) -> Self::Output;
}

impl<'tcx> Mirror<'tcx> for Expr<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror<'a, 'gcx>(self, _: &mut Cx<'a, 'gcx, 'tcx>) -> Expr<'tcx> {
        self
    }
}

impl<'tcx> Mirror<'tcx> for ExprRef<'tcx> {
    type Output = Expr<'tcx>;

    fn make_mirror<'a, 'gcx>(self, hir: &mut Cx<'a, 'gcx, 'tcx>) -> Expr<'tcx> {
        match self {
            ExprRef::Hair(h) => h.make_mirror(hir),
            ExprRef::Mirror(m) => *m,
        }
    }
}

impl<'tcx> Mirror<'tcx> for Stmt<'tcx> {
    type Output = Stmt<'tcx>;

    fn make_mirror<'a, 'gcx>(self, _: &mut Cx<'a, 'gcx, 'tcx>) -> Stmt<'tcx> {
        self
    }
}

impl<'tcx> Mirror<'tcx> for StmtRef<'tcx> {
    type Output = Stmt<'tcx>;

    fn make_mirror<'a, 'gcx>(self, _: &mut Cx<'a, 'gcx, 'tcx>) -> Stmt<'tcx> {
        match self {
            StmtRef::Mirror(m) => *m,
        }
    }
}

impl<'tcx> Mirror<'tcx> for Block<'tcx> {
    type Output = Block<'tcx>;

    fn make_mirror<'a, 'gcx>(self, _: &mut Cx<'a, 'gcx, 'tcx>) -> Block<'tcx> {
        self
    }
}
