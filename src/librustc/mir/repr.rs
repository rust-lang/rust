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
use middle::ty::{self, AdtDef, ClosureSubsts, FnOutput, Region, Ty};
use rustc_back::slice;
use rustc_data_structures::tuple_slice::TupleSlice;
use rustc_front::hir::InlineAsm;
use syntax::ast::{self, Name};
use syntax::codemap::Span;
use std::ascii;
use std::borrow::{Cow, IntoCow};
use std::fmt::{self, Debug, Formatter, Write};
use std::{iter, u32};

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
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "bb{}", self.0)
    }
}

///////////////////////////////////////////////////////////////////////////
// BasicBlock and Terminator

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct BasicBlockData<'tcx> {
    pub statements: Vec<Statement<'tcx>>,
    pub terminator: Option<Terminator<'tcx>>,
    pub is_cleanup: bool,
}

#[derive(RustcEncodable, RustcDecodable)]
pub enum Terminator<'tcx> {
    /// block should have one successor in the graph; we jump there
    Goto {
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

    /// Indicates that the landing pad is finished and unwinding should
    /// continue. Emitted by build::scope::diverge_cleanup.
    Resume,

    /// Indicates a normal return. The ReturnPointer lvalue should
    /// have been filled in by now. This should only occur in the
    /// `END_BLOCK`.
    Return,

    /// Block ends with a call of a converging function
    Call {
        /// The function that’s being called
        func: Operand<'tcx>,
        /// Arguments the function is called with
        args: Vec<Operand<'tcx>>,
        /// The kind of call with associated information
        kind: CallKind<'tcx>,
    },
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum CallKind<'tcx> {
    /// Diverging function without associated cleanup
    Diverging,
    /// Diverging function with associated cleanup
    DivergingCleanup(BasicBlock),
    /// Converging function without associated cleanup
    Converging {
        /// Destination where the call result is written
        destination: Lvalue<'tcx>,
        /// Block to branch into on successful return
        target: BasicBlock,
    },
    ConvergingCleanup {
        /// Destination where the call result is written
        destination: Lvalue<'tcx>,
        /// First target is branched to on successful return.
        /// Second block contains the cleanups to do on unwind.
        targets: (BasicBlock, BasicBlock)
    }
}

impl<'tcx> CallKind<'tcx> {
    pub fn successors(&self) -> &[BasicBlock] {
        match *self {
            CallKind::Diverging => &[],
            CallKind::DivergingCleanup(ref b) |
            CallKind::Converging { target: ref b, .. } => slice::ref_slice(b),
            CallKind::ConvergingCleanup { ref targets, .. } => targets.as_slice(),
        }
    }

    pub fn successors_mut(&mut self) -> &mut [BasicBlock] {
        match *self {
            CallKind::Diverging => &mut [],
            CallKind::DivergingCleanup(ref mut b) |
            CallKind::Converging { target: ref mut b, .. } => slice::mut_ref_slice(b),
            CallKind::ConvergingCleanup { ref mut targets, .. } => targets.as_mut_slice(),
        }
    }

    pub fn destination(&self) -> Option<Lvalue<'tcx>> {
        match *self {
            CallKind::Converging { ref destination, .. } |
            CallKind::ConvergingCleanup { ref destination, .. } => Some(destination.clone()),
            CallKind::Diverging |
            CallKind::DivergingCleanup(_) => None
        }
    }
}

impl<'tcx> Terminator<'tcx> {
    pub fn successors(&self) -> &[BasicBlock] {
        use self::Terminator::*;
        match *self {
            Goto { target: ref b } => slice::ref_slice(b),
            If { targets: ref b, .. } => b.as_slice(),
            Switch { targets: ref b, .. } => b,
            SwitchInt { targets: ref b, .. } => b,
            Resume => &[],
            Return => &[],
            Call { ref kind, .. } => kind.successors(),
        }
    }

    pub fn successors_mut(&mut self) -> &mut [BasicBlock] {
        use self::Terminator::*;
        match *self {
            Goto { target: ref mut b } => slice::mut_ref_slice(b),
            If { targets: ref mut b, .. } => b.as_mut_slice(),
            Switch { targets: ref mut b, .. } => b,
            SwitchInt { targets: ref mut b, .. } => b,
            Resume => &mut [],
            Return => &mut [],
            Call { ref mut kind, .. } => kind.successors_mut(),
        }
    }
}

impl<'tcx> BasicBlockData<'tcx> {
    pub fn new(terminator: Option<Terminator<'tcx>>) -> BasicBlockData<'tcx> {
        BasicBlockData {
            statements: vec![],
            terminator: terminator,
            is_cleanup: false,
        }
    }

    /// Accessor for terminator.
    ///
    /// Terminator may not be None after construction of the basic block is complete. This accessor
    /// provides a convenience way to reach the terminator.
    pub fn terminator(&self) -> &Terminator<'tcx> {
        self.terminator.as_ref().expect("invalid terminator state")
    }

    pub fn terminator_mut(&mut self) -> &mut Terminator<'tcx> {
        self.terminator.as_mut().expect("invalid terminator state")
    }
}

impl<'tcx> Debug for Terminator<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        try!(self.fmt_head(fmt));
        let successors = self.successors();
        let labels = self.fmt_successor_labels();
        assert_eq!(successors.len(), labels.len());

        match successors.len() {
            0 => Ok(()),

            1 => write!(fmt, " -> {:?}", successors[0]),

            _ => {
                try!(write!(fmt, " -> ["));
                for (i, target) in successors.iter().enumerate() {
                    if i > 0 {
                        try!(write!(fmt, ", "));
                    }
                    try!(write!(fmt, "{}: {:?}", labels[i], target));
                }
                write!(fmt, "]")
            }

        }
    }
}

impl<'tcx> Terminator<'tcx> {
    /// Write the "head" part of the terminator; that is, its name and the data it uses to pick the
    /// successor basic block, if any. The only information not inlcuded is the list of possible
    /// successors, which may be rendered differently between the text and the graphviz format.
    pub fn fmt_head<W: Write>(&self, fmt: &mut W) -> fmt::Result {
        use self::Terminator::*;
        match *self {
            Goto { .. } => write!(fmt, "goto"),
            If { cond: ref lv, .. } => write!(fmt, "if({:?})", lv),
            Switch { discr: ref lv, .. } => write!(fmt, "switch({:?})", lv),
            SwitchInt { discr: ref lv, .. } => write!(fmt, "switchInt({:?})", lv),
            Return => write!(fmt, "return"),
            Resume => write!(fmt, "resume"),
            Call { ref kind, ref func, ref args } => {
                if let Some(destination) = kind.destination() {
                    try!(write!(fmt, "{:?} = ", destination));
                }
                try!(write!(fmt, "{:?}(", func));
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        try!(write!(fmt, ", "));
                    }
                    try!(write!(fmt, "{:?}", arg));
                }
                write!(fmt, ")")
            }
        }
    }

    /// Return the list of labels for the edges to the successor basic blocks.
    pub fn fmt_successor_labels(&self) -> Vec<Cow<'static, str>> {
        use self::Terminator::*;
        match *self {
            Return | Resume => vec![],
            Goto { .. } => vec!["".into_cow()],
            If { .. } => vec!["true".into_cow(), "false".into_cow()],
            Switch { ref adt_def, .. } => {
                adt_def.variants
                       .iter()
                       .map(|variant| variant.name.to_string().into_cow())
                       .collect()
            }
            SwitchInt { ref values, .. } => {
                values.iter()
                      .map(|const_val| {
                          let mut buf = String::new();
                          fmt_const_val(&mut buf, const_val).unwrap();
                          buf.into_cow()
                      })
                      .chain(iter::once(String::from("otherwise").into_cow()))
                      .collect()
            }
            Call { ref kind, .. } => match *kind {
                CallKind::Diverging =>
                    vec![],
                CallKind::DivergingCleanup(..) =>
                    vec!["unwind".into_cow()],
                CallKind::Converging { .. } =>
                    vec!["return".into_cow()],
                CallKind::ConvergingCleanup { .. } =>
                    vec!["return".into_cow(), "unwind".into_cow()],
            },
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
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
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
pub type LvalueProjection<'tcx> = Projection<'tcx, Lvalue<'tcx>, Operand<'tcx>>;

/// Alias for projections as they appear in lvalues, where the base is an lvalue
/// and the index is an operand.
pub type LvalueElem<'tcx> = ProjectionElem<'tcx, Operand<'tcx>>;

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
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        use self::Lvalue::*;

        match *self {
            Var(id) =>
                write!(fmt, "var{:?}", id),
            Arg(id) =>
                write!(fmt, "arg{:?}", id),
            Temp(id) =>
                write!(fmt, "tmp{:?}", id),
            Static(def_id) =>
                write!(fmt, "{}", ty::tls::with(|tcx| tcx.item_path_str(def_id))),
            ReturnPointer =>
                write!(fmt, "return"),
            Projection(ref data) =>
                match data.elem {
                    ProjectionElem::Downcast(ref adt_def, index) =>
                        write!(fmt, "({:?} as {})", data.base, adt_def.variants[index].name),
                    ProjectionElem::Deref =>
                        write!(fmt, "(*{:?})", data.base),
                    ProjectionElem::Field(field) =>
                        write!(fmt, "{:?}.{:?}", data.base, field.index()),
                    ProjectionElem::Index(ref index) =>
                        write!(fmt, "{:?}[{:?}]", data.base, index),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: false } =>
                        write!(fmt, "{:?}[{:?} of {:?}]", data.base, offset, min_length),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: true } =>
                        write!(fmt, "{:?}[-{:?} of {:?}]", data.base, offset, min_length),
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
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
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
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        use self::Rvalue::*;

        match *self {
            Use(ref lvalue) => write!(fmt, "{:?}", lvalue),
            Repeat(ref a, ref b) => write!(fmt, "[{:?}; {:?}]", a, b),
            Len(ref a) => write!(fmt, "Len({:?})", a),
            Cast(ref kind, ref lv, ref ty) => write!(fmt, "{:?} as {:?} ({:?})", lv, ty, kind),
            BinaryOp(ref op, ref a, ref b) => write!(fmt, "{:?}({:?}, {:?})", op, a, b),
            UnaryOp(ref op, ref a) => write!(fmt, "{:?}({:?})", op, a),
            Box(ref t) => write!(fmt, "Box({:?})", t),
            InlineAsm(ref asm) => write!(fmt, "InlineAsm({:?})", asm),
            Slice { ref input, from_start, from_end } =>
                write!(fmt, "{:?}[{:?}..-{:?}]", input, from_start, from_end),

            Ref(_, borrow_kind, ref lv) => {
                let kind_str = match borrow_kind {
                    BorrowKind::Shared => "",
                    BorrowKind::Mut | BorrowKind::Unique => "mut ",
                };
                write!(fmt, "&{}{:?}", kind_str, lv)
            }

            Aggregate(ref kind, ref lvs) => {
                use self::AggregateKind::*;

                fn fmt_tuple(fmt: &mut Formatter, name: &str, lvs: &[Operand]) -> fmt::Result {
                    let mut tuple_fmt = fmt.debug_tuple(name);
                    for lv in lvs {
                        tuple_fmt.field(lv);
                    }
                    tuple_fmt.finish()
                }

                match *kind {
                    Vec => write!(fmt, "{:?}", lvs),

                    Tuple => {
                        match lvs.len() {
                            0 => write!(fmt, "()"),
                            1 => write!(fmt, "({:?},)", lvs[0]),
                            _ => fmt_tuple(fmt, "", lvs),
                        }
                    }

                    Adt(adt_def, variant, _) => {
                        let variant_def = &adt_def.variants[variant];
                        let name = ty::tls::with(|tcx| tcx.item_path_str(variant_def.did));

                        match variant_def.kind() {
                            ty::VariantKind::Unit => write!(fmt, "{}", name),
                            ty::VariantKind::Tuple => fmt_tuple(fmt, &name, lvs),
                            ty::VariantKind::Struct => {
                                let mut struct_fmt = fmt.debug_struct(&name);
                                for (field, lv) in variant_def.fields.iter().zip(lvs) {
                                    struct_fmt.field(&field.name.as_str(), lv);
                                }
                                struct_fmt.finish()
                            }
                        }
                    }

                    Closure(def_id, _) => ty::tls::with(|tcx| {
                        if let Some(node_id) = tcx.map.as_local_node_id(def_id) {
                            let name = format!("[closure@{:?}]", tcx.map.span(node_id));
                            let mut struct_fmt = fmt.debug_struct(&name);

                            tcx.with_freevars(node_id, |freevars| {
                                for (freevar, lv) in freevars.iter().zip(lvs) {
                                    let var_name = tcx.local_var_name_str(freevar.def.var_id());
                                    struct_fmt.field(&var_name, lv);
                                }
                            });

                            struct_fmt.finish()
                        } else {
                            write!(fmt, "[closure]")
                        }
                    }),
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Constants
//
// Two constants are equal if they are the same constant. Note that
// this does not necessarily mean that they are "==" in Rust -- in
// particular one must be wary of `NaN`!

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable)]
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

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable)]
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

impl<'tcx> Debug for Constant<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "{:?}", self.literal)
    }
}

impl<'tcx> Debug for Literal<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        use self::Literal::*;
        match *self {
            Item { def_id, .. } =>
                write!(fmt, "{}", item_path_str(def_id)),
            Value { ref value } => {
                try!(write!(fmt, "const "));
                fmt_const_val(fmt, value)
            }
        }
    }
}

/// Write a `ConstVal` in a way closer to the original source code than the `Debug` output.
fn fmt_const_val<W: Write>(fmt: &mut W, const_val: &ConstVal) -> fmt::Result {
    use middle::const_eval::ConstVal::*;
    match *const_val {
        Float(f) => write!(fmt, "{:?}", f),
        Int(n) => write!(fmt, "{:?}", n),
        Uint(n) => write!(fmt, "{:?}", n),
        Str(ref s) => write!(fmt, "{:?}", s),
        ByteStr(ref bytes) => {
            let escaped: String = bytes
                .iter()
                .flat_map(|&ch| ascii::escape_default(ch).map(|c| c as char))
                .collect();
            write!(fmt, "b\"{}\"", escaped)
        }
        Bool(b) => write!(fmt, "{:?}", b),
        Function(def_id) => write!(fmt, "{}", item_path_str(def_id)),
        Struct(node_id) | Tuple(node_id) | Array(node_id, _) | Repeat(node_id, _) =>
            write!(fmt, "{}", node_to_string(node_id)),
    }
}

fn node_to_string(node_id: ast::NodeId) -> String {
    ty::tls::with(|tcx| tcx.map.node_to_user_string(node_id))
}

fn item_path_str(def_id: DefId) -> String {
    ty::tls::with(|tcx| tcx.item_path_str(def_id))
}
