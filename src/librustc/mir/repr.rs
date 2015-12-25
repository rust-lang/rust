// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::const_eval::ConstVal;
use middle::def_id::DefId;
use middle::subst::Substs;
use middle::ty::{AdtDef, ClosureSubsts, FnOutput, Region, Ty};
use rustc_back::slice;
use rustc_data_structures::tuple_slice::TupleSlice;
use rustc_front::hir::InlineAsm;
use syntax::ast::Name;
use syntax::codemap::Span;
use std::fmt::{Debug, Formatter, Error};
use std::u32;

/// Lowered representation of a single function.
#[derive(RustcEncodable, RustcDecodable)]
pub struct Mir<'tcx> {
    /// List of basic blocks. References to basic block use a newtyped index type `BasicBlock`
    /// that indexes into this vector.
    pub basic_blocks: Vec<BasicBlockData<'tcx>>,

    /// Return type of the function.
    pub return_ty: FnOutput<'tcx>,

    /// Variables: these are stack slots corresponding to user variables. They may be
    /// assigned many times.
    pub var_decls: Vec<VarDecl<'tcx>>,

    /// Args: these are stack slots corresponding to the input arguments.
    pub arg_decls: Vec<ArgDecl<'tcx>>,

    /// Temp declarations: stack slots that for temporaries created by
    /// the compiler. These are assigned once, but they are not SSA
    /// values in that it is possible to borrow them and mutate them
    /// through the resulting reference.
    pub temp_decls: Vec<TempDecl<'tcx>>,
}

/// where execution begins
pub const START_BLOCK: BasicBlock = BasicBlock(0);

/// where execution ends, on normal return
pub const END_BLOCK: BasicBlock = BasicBlock(1);

/// where execution ends, on panic
pub const DIVERGE_BLOCK: BasicBlock = BasicBlock(2);

impl<'tcx> Mir<'tcx> {
    pub fn all_basic_blocks(&self) -> Vec<BasicBlock> {
        (0..self.basic_blocks.len())
            .map(|i| BasicBlock::new(i))
            .collect()
    }

    pub fn basic_block_data(&self, bb: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks[bb.index()]
    }

    pub fn basic_block_data_mut(&mut self, bb: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks[bb.index()]
    }
}

///////////////////////////////////////////////////////////////////////////
// Mutability and borrow kinds

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum Mutability {
    Mut,
    Not,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Shared,

    /// Data must be immutable but not aliasable.  This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when you the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    ///    let x: &mut isize = ...;
    ///    let y = || *x += 5;
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    ///    struct Env { x: & &mut isize }
    ///    let x: &mut isize = ...;
    ///    let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    ///    struct Env { x: & &mut isize }
    ///    let x: &mut isize = ...;
    ///    let y = (&mut Env { &mut x }, fn_ptr); // changed from &x to &mut x
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// Now the assignment to `**env.x` is legal, but creating a
    /// mutable pointer to `x` is not because `x` is not mutable. We
    /// could fix this by declaring `x` as `let mut x`. This is ok in
    /// user code, if awkward, but extra weird for closures, since the
    /// borrow is hidden.
    ///
    /// So we introduce a "unique imm" borrow -- the referent is
    /// immutable, but not aliasable. This solves the problem. For
    /// simplicity, we don't give users the way to express this
    /// borrow, it's just used when translating closures.
    Unique,

    /// Data is mutable and not aliasable.
    Mut,
}

///////////////////////////////////////////////////////////////////////////
// Variables and temps

// A "variable" is a binding declared by the user as part of the fn
// decl, a let, etc.
#[derive(RustcEncodable, RustcDecodable)]
pub struct VarDecl<'tcx> {
    pub mutability: Mutability,
    pub name: Name,
    pub ty: Ty<'tcx>,
}

// A "temp" is a temporary that we place on the stack. They are
// anonymous, always mutable, and have only a type.
#[derive(RustcEncodable, RustcDecodable)]
pub struct TempDecl<'tcx> {
    pub ty: Ty<'tcx>,
}

// A "arg" is one of the function's formal arguments. These are
// anonymous and distinct from the bindings that the user declares.
//
// For example, in this function:
//
// ```
// fn foo((x, y): (i32, u32)) { ... }
// ```
//
// there is only one argument, of type `(i32, u32)`, but two bindings
// (`x` and `y`).
#[derive(RustcEncodable, RustcDecodable)]
pub struct ArgDecl<'tcx> {
    pub ty: Ty<'tcx>,
}

///////////////////////////////////////////////////////////////////////////
// BasicBlock

/// The index of a particular basic block. The index is into the `basic_blocks`
/// list of the `Mir`.
///
/// (We use a `u32` internally just to save memory.)
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct BasicBlock(u32);

impl BasicBlock {
    pub fn new(index: usize) -> BasicBlock {
        assert!(index < (u32::MAX as usize));
        BasicBlock(index as u32)
    }

    /// Extract the index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl Debug for BasicBlock {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        write!(fmt, "BB({})", self.0)
    }
}

///////////////////////////////////////////////////////////////////////////
// BasicBlock and Terminator

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct BasicBlockData<'tcx> {
    pub statements: Vec<Statement<'tcx>>,
    pub terminator: Terminator<'tcx>,
}

#[derive(RustcEncodable, RustcDecodable)]
pub enum Terminator<'tcx> {
    /// block should have one successor in the graph; we jump there
    Goto {
        target: BasicBlock,
    },

    /// block should initiate unwinding; should be one successor
    /// that does cleanup and branches to DIVERGE_BLOCK
    Panic {
        target: BasicBlock,
    },

    /// jump to branch 0 if this lvalue evaluates to true
    If {
        cond: Operand<'tcx>,
        targets: (BasicBlock, BasicBlock),
    },

    /// lvalue evaluates to some enum; jump depending on the branch
    Switch {
        discr: Lvalue<'tcx>,
        adt_def: AdtDef<'tcx>,
        targets: Vec<BasicBlock>,
    },

    /// operand evaluates to an integer; jump depending on its value
    /// to one of the targets, and otherwise fallback to `otherwise`
    SwitchInt {
        /// discriminant value being tested
        discr: Lvalue<'tcx>,

        /// type of value being tested
        switch_ty: Ty<'tcx>,

        /// Possible values. The locations to branch to in each case
        /// are found in the corresponding indices from the `targets` vector.
        values: Vec<ConstVal>,

        /// Possible branch sites. The length of this vector should be
        /// equal to the length of the `values` vector plus 1 -- the
        /// extra item is the block to branch to if none of the values
        /// fit.
        targets: Vec<BasicBlock>,
    },

    /// Indicates that the last statement in the block panics, aborts,
    /// etc. No successors. This terminator appears on exactly one
    /// basic block which we create in advance. However, during
    /// construction, we use this value as a sentinel for "terminator
    /// not yet assigned", and assert at the end that only the
    /// well-known diverging block actually diverges.
    Diverge,

    /// Indicates a normal return. The ReturnPointer lvalue should
    /// have been filled in by now. This should only occur in the
    /// `END_BLOCK`.
    Return,

    /// block ends with a call; it should have two successors. The
    /// first successor indicates normal return. The second indicates
    /// unwinding.
    Call {
        data: CallData<'tcx>,
        targets: (BasicBlock, BasicBlock),
    },
}

impl<'tcx> Terminator<'tcx> {
    pub fn successors(&self) -> &[BasicBlock] {
        use self::Terminator::*;
        match *self {
            Goto { target: ref b } => slice::ref_slice(b),
            Panic { target: ref b } => slice::ref_slice(b),
            If { cond: _, targets: ref b } => b.as_slice(),
            Switch { targets: ref b, .. } => b,
            SwitchInt { targets: ref b, .. } => b,
            Diverge => &[],
            Return => &[],
            Call { data: _, targets: ref b } => b.as_slice(),
        }
    }

    pub fn successors_mut(&mut self) -> &mut [BasicBlock] {
        use self::Terminator::*;
        match *self {
            Goto { target: ref mut b } => slice::mut_ref_slice(b),
            Panic { target: ref mut b } => slice::mut_ref_slice(b),
            If { cond: _, targets: ref mut b } => b.as_mut_slice(),
            Switch { targets: ref mut b, .. } => b,
            SwitchInt { targets: ref mut b, .. } => b,
            Diverge => &mut [],
            Return => &mut [],
            Call { data: _, targets: ref mut b } => b.as_mut_slice(),
        }
    }
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct CallData<'tcx> {
    /// where the return value is written to
    pub destination: Lvalue<'tcx>,

    /// the fn being called
    pub func: Operand<'tcx>,

    /// the arguments
    pub args: Vec<Operand<'tcx>>,
}

impl<'tcx> BasicBlockData<'tcx> {
    pub fn new(terminator: Terminator<'tcx>) -> BasicBlockData<'tcx> {
        BasicBlockData {
            statements: vec![],
            terminator: terminator,
        }
    }
}

impl<'tcx> Debug for Terminator<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Terminator::*;
        match *self {
            Goto { target } =>
                write!(fmt, "goto -> {:?}", target),
            Panic { target } =>
                write!(fmt, "panic -> {:?}", target),
            If { cond: ref lv, ref targets } =>
                write!(fmt, "if({:?}) -> {:?}", lv, targets),
            Switch { discr: ref lv, adt_def: _, ref targets } =>
                write!(fmt, "switch({:?}) -> {:?}", lv, targets),
            SwitchInt { discr: ref lv, switch_ty: _, ref values, ref targets } =>
                write!(fmt, "switchInt({:?}, {:?}) -> {:?}", lv, values, targets),
            Diverge =>
                write!(fmt, "diverge"),
            Return =>
                write!(fmt, "return"),
            Call { data: ref c, targets } => {
                try!(write!(fmt, "{:?} = {:?}(", c.destination, c.func));
                for (index, arg) in c.args.iter().enumerate() {
                    if index > 0 {
                        try!(write!(fmt, ", "));
                    }
                    try!(write!(fmt, "{:?}", arg));
                }
                write!(fmt, ") -> {:?}", targets)
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////
// Statements

#[derive(RustcEncodable, RustcDecodable)]
pub struct Statement<'tcx> {
    pub span: Span,
    pub kind: StatementKind<'tcx>,
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub enum StatementKind<'tcx> {
    Assign(Lvalue<'tcx>, Rvalue<'tcx>),
    Drop(DropKind, Lvalue<'tcx>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum DropKind {
    Free, // free a partially constructed box, should go away eventually
    Deep
}

impl<'tcx> Debug for Statement<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::StatementKind::*;
        match self.kind {
            Assign(ref lv, ref rv) => write!(fmt, "{:?} = {:?}", lv, rv),
            Drop(DropKind::Free, ref lv) => write!(fmt, "free {:?}", lv),
            Drop(DropKind::Deep, ref lv) => write!(fmt, "drop {:?}", lv),
        }
    }
}
///////////////////////////////////////////////////////////////////////////
// Lvalues

/// A path to a value; something that can be evaluated without
/// changing or disturbing program state.
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Lvalue<'tcx> {
    /// local variable declared by the user
    Var(u32),

    /// temporary introduced during lowering into MIR
    Temp(u32),

    /// formal parameter of the function; note that these are NOT the
    /// bindings that the user declares, which are vars
    Arg(u32),

    /// static or static mut variable
    Static(DefId),

    /// the return pointer of the fn
    ReturnPointer,

    /// projection out of an lvalue (access a field, deref a pointer, etc)
    Projection(Box<LvalueProjection<'tcx>>),
}

/// The `Projection` data structure defines things of the form `B.x`
/// or `*B` or `B[index]`. Note that it is parameterized because it is
/// shared between `Constant` and `Lvalue`. See the aliases
/// `LvalueProjection` etc below.
#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub struct Projection<'tcx, B, V> {
    pub base: B,
    pub elem: ProjectionElem<'tcx, V>,
}

#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub enum ProjectionElem<'tcx, V> {
    Deref,
    Field(Field),
    Index(V),

    // These indices are generated by slice patterns. Easiest to explain
    // by example:
    //
    // ```
    // [X, _, .._, _, _] => { offset: 0, min_length: 4, from_end: false },
    // [_, X, .._, _, _] => { offset: 1, min_length: 4, from_end: false },
    // [_, _, .._, X, _] => { offset: 2, min_length: 4, from_end: true },
    // [_, _, .._, _, X] => { offset: 1, min_length: 4, from_end: true },
    // ```
    ConstantIndex {
        offset: u32,      // index or -index (in Python terms), depending on from_end
        min_length: u32,  // thing being indexed must be at least this long
        from_end: bool,   // counting backwards from end?
    },

    // "Downcast" to a variant of an ADT. Currently, we only introduce
    // this for ADTs with more than one variant. It may be better to
    // just introduce it always, or always for enums.
    Downcast(AdtDef<'tcx>, usize),
}

/// Alias for projections as they appear in lvalues, where the base is an lvalue
/// and the index is an operand.
pub type LvalueProjection<'tcx> =
    Projection<'tcx,Lvalue<'tcx>,Operand<'tcx>>;

/// Alias for projections as they appear in lvalues, where the base is an lvalue
/// and the index is an operand.
pub type LvalueElem<'tcx> =
    ProjectionElem<'tcx,Operand<'tcx>>;

/// Index into the list of fields found in a `VariantDef`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct Field(u32);

impl Field {
    pub fn new(value: usize) -> Field {
        assert!(value < (u32::MAX) as usize);
        Field(value as u32)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl<'tcx> Lvalue<'tcx> {
    pub fn field(self, f: Field) -> Lvalue<'tcx> {
        self.elem(ProjectionElem::Field(f))
    }

    pub fn deref(self) -> Lvalue<'tcx> {
        self.elem(ProjectionElem::Deref)
    }

    pub fn index(self, index: Operand<'tcx>) -> Lvalue<'tcx> {
        self.elem(ProjectionElem::Index(index))
    }

    pub fn elem(self, elem: LvalueElem<'tcx>) -> Lvalue<'tcx> {
        Lvalue::Projection(Box::new(LvalueProjection {
            base: self,
            elem: elem,
        }))
    }
}

impl<'tcx> Debug for Lvalue<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Lvalue::*;

        match *self {
            Var(id) =>
                write!(fmt,"Var({:?})", id),
            Arg(id) =>
                write!(fmt,"Arg({:?})", id),
            Temp(id) =>
                write!(fmt,"Temp({:?})", id),
            Static(id) =>
                write!(fmt,"Static({:?})", id),
            ReturnPointer =>
                write!(fmt,"ReturnPointer"),
            Projection(ref data) =>
                match data.elem {
                    ProjectionElem::Downcast(_, variant_index) =>
                        write!(fmt,"({:?} as {:?})", data.base, variant_index),
                    ProjectionElem::Deref =>
                        write!(fmt,"(*{:?})", data.base),
                    ProjectionElem::Field(field) =>
                        write!(fmt,"{:?}.{:?}", data.base, field.index()),
                    ProjectionElem::Index(ref index) =>
                        write!(fmt,"{:?}[{:?}]", data.base, index),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: false } =>
                        write!(fmt,"{:?}[{:?} of {:?}]", data.base, offset, min_length),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: true } =>
                        write!(fmt,"{:?}[-{:?} of {:?}]", data.base, offset, min_length),
                },
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Operands
//
// These are values that can appear inside an rvalue (or an index
// lvalue). They are intentionally limited to prevent rvalues from
// being nested in one another.

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Operand<'tcx> {
    Consume(Lvalue<'tcx>),
    Constant(Constant<'tcx>),
}

impl<'tcx> Debug for Operand<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Operand::*;
        match *self {
            Constant(ref a) => write!(fmt, "{:?}", a),
            Consume(ref lv) => write!(fmt, "{:?}", lv),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Rvalues

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum Rvalue<'tcx> {
    // x (either a move or copy, depending on type of x)
    Use(Operand<'tcx>),

    // [x; 32]
    Repeat(Operand<'tcx>, Constant<'tcx>),

    // &x or &mut x
    Ref(Region, BorrowKind, Lvalue<'tcx>),

    // length of a [X] or [X;n] value
    Len(Lvalue<'tcx>),

    Cast(CastKind, Operand<'tcx>, Ty<'tcx>),

    BinaryOp(BinOp, Operand<'tcx>, Operand<'tcx>),

    UnaryOp(UnOp, Operand<'tcx>),

    // Creates an *uninitialized* Box
    Box(Ty<'tcx>),

    // Create an aggregate value, like a tuple or struct.  This is
    // only needed because we want to distinguish `dest = Foo { x:
    // ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case
    // that `Foo` has a destructor. These rvalues can be optimized
    // away after type-checking and before lowering.
    Aggregate(AggregateKind<'tcx>, Vec<Operand<'tcx>>),

    // Generates a slice of the form `&input[from_start..L-from_end]`
    // where `L` is the length of the slice. This is only created by
    // slice pattern matching, so e.g. a pattern of the form `[x, y,
    // .., z]` might create a slice with `from_start=2` and
    // `from_end=1`.
    Slice {
        input: Lvalue<'tcx>,
        from_start: usize,
        from_end: usize,
    },

    InlineAsm(InlineAsm),
}

#[derive(Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum CastKind {
    Misc,

    /// Convert unique, zero-sized type for a fn to fn()
    ReifyFnPointer,

    /// Convert safe fn() to unsafe fn()
    UnsafeFnPointer,

    /// "Unsize" -- convert a thin-or-fat pointer to a fat pointer.
    /// trans must figure out the details once full monomorphization
    /// is known. For example, this could be used to cast from a
    /// `&[i32;N]` to a `&[i32]`, or a `Box<T>` to a `Box<Trait>`
    /// (presuming `T: Trait`).
    Unsize,
}

#[derive(Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum AggregateKind<'tcx> {
    Vec,
    Tuple,
    Adt(AdtDef<'tcx>, usize, &'tcx Substs<'tcx>),
    Closure(DefId, &'tcx ClosureSubsts<'tcx>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum BinOp {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

impl<'tcx> Debug for Rvalue<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Rvalue::*;

        match *self {
            Use(ref lvalue) => write!(fmt, "{:?}", lvalue),
            Repeat(ref a, ref b) => write!(fmt, "[{:?}; {:?}]", a, b),
            Ref(ref a, bk, ref b) => write!(fmt, "&{:?} {:?} {:?}", a, bk, b),
            Len(ref a) => write!(fmt, "LEN({:?})", a),
            Cast(ref kind, ref lv, ref ty) => write!(fmt, "{:?} as {:?} ({:?}", lv, ty, kind),
            BinaryOp(ref op, ref a, ref b) => write!(fmt, "{:?}({:?},{:?})", op, a, b),
            UnaryOp(ref op, ref a) => write!(fmt, "{:?}({:?})", op, a),
            Box(ref t) => write!(fmt, "Box {:?}", t),
            Aggregate(ref kind, ref lvs) => write!(fmt, "Aggregate<{:?}>({:?})", kind, lvs),
            InlineAsm(ref asm) => write!(fmt, "InlineAsm({:?})", asm),
            Slice { ref input, from_start, from_end } =>
                write!(fmt, "{:?}[{:?}..-{:?}]", input, from_start, from_end),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Constants
//
// Two constants are equal if they are the same constant. Note that
// this does not necessarily mean that they are "==" in Rust -- in
// particular one must be wary of `NaN`!

#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub struct Constant<'tcx> {
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub literal: Literal<'tcx>,
}

#[derive(Clone, Copy, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub enum ItemKind {
    Constant,
    /// This is any sort of callable (usually those that have a type of `fn(…) -> …`). This
    /// includes functions, constructors, but not methods which have their own ItemKind.
    Function,
    Method,
}

#[derive(Clone, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Literal<'tcx> {
    Item {
        def_id: DefId,
        kind: ItemKind,
        substs: &'tcx Substs<'tcx>,
    },
    Value {
        value: ConstVal,
    },
}
