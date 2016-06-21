// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use graphviz::IntoCow;
use middle::const_val::ConstVal;
use rustc_const_math::{ConstUsize, ConstInt, ConstMathErr};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use hir::def_id::DefId;
use ty::subst::Substs;
use ty::{self, AdtDef, ClosureSubsts, FnOutput, Region, Ty};
use util::ppaux;
use rustc_back::slice;
use hir::InlineAsm;
use std::ascii;
use std::borrow::{Cow};
use std::cell::Ref;
use std::fmt::{self, Debug, Formatter, Write};
use std::{iter, u32};
use std::ops::{Index, IndexMut};
use syntax::ast::{self, Name};
use syntax_pos::Span;

use super::cache::Cache;

macro_rules! newtype_index {
    ($name:ident, $debug_name:expr) => (
        #[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord,
         RustcEncodable, RustcDecodable)]
        pub struct $name(u32);

        impl Idx for $name {
            fn new(value: usize) -> Self {
                assert!(value < (u32::MAX) as usize);
                $name(value as u32)
            }
            fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl Debug for $name {
            fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
                write!(fmt, "{}{}", $debug_name, self.0)
            }
        }
    )
}

/// Lowered representation of a single function.
#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Mir<'tcx> {
    /// List of basic blocks. References to basic block use a newtyped index type `BasicBlock`
    /// that indexes into this vector.
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,

    /// List of visibility (lexical) scopes; these are referenced by statements
    /// and used (eventually) for debuginfo. Indexed by a `VisibilityScope`.
    pub visibility_scopes: IndexVec<VisibilityScope, VisibilityScopeData>,

    /// Rvalues promoted from this function, such as borrows of constants.
    /// Each of them is the Mir of a constant with the fn's type parameters
    /// in scope, but no vars or args and a separate set of temps.
    pub promoted: IndexVec<Promoted, Mir<'tcx>>,

    /// Return type of the function.
    pub return_ty: FnOutput<'tcx>,

    /// Variables: these are stack slots corresponding to user variables. They may be
    /// assigned many times.
    pub var_decls: IndexVec<Var, VarDecl<'tcx>>,

    /// Args: these are stack slots corresponding to the input arguments.
    pub arg_decls: IndexVec<Arg, ArgDecl<'tcx>>,

    /// Temp declarations: stack slots that for temporaries created by
    /// the compiler. These are assigned once, but they are not SSA
    /// values in that it is possible to borrow them and mutate them
    /// through the resulting reference.
    pub temp_decls: IndexVec<Temp, TempDecl<'tcx>>,

    /// Names and capture modes of all the closure upvars, assuming
    /// the first argument is either the closure or a reference to it.
    pub upvar_decls: Vec<UpvarDecl>,

    /// A span representing this MIR, for error reporting
    pub span: Span,

    /// A cache for various calculations
    cache: Cache
}

/// where execution begins
pub const START_BLOCK: BasicBlock = BasicBlock(0);

impl<'tcx> Mir<'tcx> {
    pub fn new(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
               visibility_scopes: IndexVec<VisibilityScope, VisibilityScopeData>,
               promoted: IndexVec<Promoted, Mir<'tcx>>,
               return_ty: FnOutput<'tcx>,
               var_decls: IndexVec<Var, VarDecl<'tcx>>,
               arg_decls: IndexVec<Arg, ArgDecl<'tcx>>,
               temp_decls: IndexVec<Temp, TempDecl<'tcx>>,
               upvar_decls: Vec<UpvarDecl>,
               span: Span) -> Self
    {
        Mir {
            basic_blocks: basic_blocks,
            visibility_scopes: visibility_scopes,
            promoted: promoted,
            return_ty: return_ty,
            var_decls: var_decls,
            arg_decls: arg_decls,
            temp_decls: temp_decls,
            upvar_decls: upvar_decls,
            span: span,
            cache: Cache::new()
        }
    }

    #[inline]
    pub fn basic_blocks(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &self.basic_blocks
    }

    #[inline]
    pub fn basic_blocks_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.cache.invalidate();
        &mut self.basic_blocks
    }

    #[inline]
    pub fn predecessors(&self) -> Ref<IndexVec<BasicBlock, Vec<BasicBlock>>> {
        self.cache.predecessors(self)
    }

    #[inline]
    pub fn predecessors_for(&self, bb: BasicBlock) -> Ref<Vec<BasicBlock>> {
        Ref::map(self.predecessors(), |p| &p[bb])
    }

    /// Maps locals (Arg's, Var's, Temp's and ReturnPointer, in that order)
    /// to their index in the whole list of locals. This is useful if you
    /// want to treat all locals the same instead of repeating yourself.
    pub fn local_index(&self, lvalue: &Lvalue<'tcx>) -> Option<Local> {
        let idx = match *lvalue {
            Lvalue::Arg(arg) => arg.index(),
            Lvalue::Var(var) => {
                self.arg_decls.len() +
                var.index()
            }
            Lvalue::Temp(temp) => {
                self.arg_decls.len() +
                self.var_decls.len() +
                temp.index()
            }
            Lvalue::ReturnPointer => {
                self.arg_decls.len() +
                self.var_decls.len() +
                self.temp_decls.len()
            }
            Lvalue::Static(_) |
            Lvalue::Projection(_) => return None
        };
        Some(Local::new(idx))
    }

    /// Counts the number of locals, such that that local_index
    /// will always return an index smaller than this count.
    pub fn count_locals(&self) -> usize {
        self.arg_decls.len() +
        self.var_decls.len() +
        self.temp_decls.len() + 1
    }
}

impl<'tcx> Index<BasicBlock> for Mir<'tcx> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks()[index]
    }
}

impl<'tcx> IndexMut<BasicBlock> for Mir<'tcx> {
    #[inline]
    fn index_mut(&mut self, index: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks_mut()[index]
    }
}

/// Grouped information about the source code origin of a MIR entity.
/// Intended to be inspected by diagnostics and debuginfo.
/// Most passes can work with it as a whole, within a single function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub struct SourceInfo {
    /// Source span for the AST pertaining to this MIR entity.
    pub span: Span,

    /// The lexical visibility scope, i.e. which bindings can be seen.
    pub scope: VisibilityScope
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

/// A "variable" is a binding declared by the user as part of the fn
/// decl, a let, etc.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct VarDecl<'tcx> {
    /// `let mut x` vs `let x`
    pub mutability: Mutability,

    /// name that user gave the variable; not that, internally,
    /// mir references variables by index
    pub name: Name,

    /// type inferred for this variable (`let x: ty = ...`)
    pub ty: Ty<'tcx>,

    /// source information (span, scope, etc.) for the declaration
    pub source_info: SourceInfo,
}

/// A "temp" is a temporary that we place on the stack. They are
/// anonymous, always mutable, and have only a type.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct TempDecl<'tcx> {
    pub ty: Ty<'tcx>,
}

/// A "arg" is one of the function's formal arguments. These are
/// anonymous and distinct from the bindings that the user declares.
///
/// For example, in this function:
///
/// ```
/// fn foo((x, y): (i32, u32)) { ... }
/// ```
///
/// there is only one argument, of type `(i32, u32)`, but two bindings
/// (`x` and `y`).
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct ArgDecl<'tcx> {
    pub ty: Ty<'tcx>,

    /// If true, this argument is a tuple after monomorphization,
    /// and has to be collected from multiple actual arguments.
    pub spread: bool,

    /// Either keywords::Invalid or the name of a single-binding
    /// pattern associated with this argument. Useful for debuginfo.
    pub debug_name: Name
}

/// A closure capture, with its name and mode.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct UpvarDecl {
    pub debug_name: Name,

    /// If true, the capture is behind a reference.
    pub by_ref: bool
}

///////////////////////////////////////////////////////////////////////////
// BasicBlock

newtype_index!(BasicBlock, "bb");

///////////////////////////////////////////////////////////////////////////
// BasicBlockData and Terminator

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct BasicBlockData<'tcx> {
    /// List of statements in this block.
    pub statements: Vec<Statement<'tcx>>,

    /// Terminator for this block.
    ///
    /// NB. This should generally ONLY be `None` during construction.
    /// Therefore, you should generally access it via the
    /// `terminator()` or `terminator_mut()` methods. The only
    /// exception is that certain passes, such as `simplify_cfg`, swap
    /// out the terminator temporarily with `None` while they continue
    /// to recurse over the set of basic blocks.
    pub terminator: Option<Terminator<'tcx>>,

    /// If true, this block lies on an unwind path. This is used
    /// during trans where distinct kinds of basic blocks may be
    /// generated (particularly for MSVC cleanup). Unwind blocks must
    /// only branch to other unwind blocks.
    pub is_cleanup: bool,
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct Terminator<'tcx> {
    pub source_info: SourceInfo,
    pub kind: TerminatorKind<'tcx>
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum TerminatorKind<'tcx> {
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
    /// have been filled in by now. This should occur at most once.
    Return,

    /// Indicates a terminator that can never be reached.
    Unreachable,

    /// Drop the Lvalue
    Drop {
        location: Lvalue<'tcx>,
        target: BasicBlock,
        unwind: Option<BasicBlock>
    },

    /// Drop the Lvalue and assign the new value over it
    DropAndReplace {
        location: Lvalue<'tcx>,
        value: Operand<'tcx>,
        target: BasicBlock,
        unwind: Option<BasicBlock>,
    },

    /// Block ends with a call of a converging function
    Call {
        /// The function that’s being called
        func: Operand<'tcx>,
        /// Arguments the function is called with
        args: Vec<Operand<'tcx>>,
        /// Destination for the return value. If some, the call is converging.
        destination: Option<(Lvalue<'tcx>, BasicBlock)>,
        /// Cleanups to be done if the call unwinds.
        cleanup: Option<BasicBlock>
    },

    /// Jump to the target if the condition has the expected value,
    /// otherwise panic with a message and a cleanup target.
    Assert {
        cond: Operand<'tcx>,
        expected: bool,
        msg: AssertMessage<'tcx>,
        target: BasicBlock,
        cleanup: Option<BasicBlock>
    }
}

impl<'tcx> Terminator<'tcx> {
    pub fn successors(&self) -> Cow<[BasicBlock]> {
        self.kind.successors()
    }

    pub fn successors_mut(&mut self) -> Vec<&mut BasicBlock> {
        self.kind.successors_mut()
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    pub fn successors(&self) -> Cow<[BasicBlock]> {
        use self::TerminatorKind::*;
        match *self {
            Goto { target: ref b } => slice::ref_slice(b).into_cow(),
            If { targets: (b1, b2), .. } => vec![b1, b2].into_cow(),
            Switch { targets: ref b, .. } => b[..].into_cow(),
            SwitchInt { targets: ref b, .. } => b[..].into_cow(),
            Resume => (&[]).into_cow(),
            Return => (&[]).into_cow(),
            Unreachable => (&[]).into_cow(),
            Call { destination: Some((_, t)), cleanup: Some(c), .. } => vec![t, c].into_cow(),
            Call { destination: Some((_, ref t)), cleanup: None, .. } =>
                slice::ref_slice(t).into_cow(),
            Call { destination: None, cleanup: Some(ref c), .. } => slice::ref_slice(c).into_cow(),
            Call { destination: None, cleanup: None, .. } => (&[]).into_cow(),
            DropAndReplace { target, unwind: Some(unwind), .. } |
            Drop { target, unwind: Some(unwind), .. } => {
                vec![target, unwind].into_cow()
            }
            DropAndReplace { ref target, unwind: None, .. } |
            Drop { ref target, unwind: None, .. } => {
                slice::ref_slice(target).into_cow()
            }
            Assert { target, cleanup: Some(unwind), .. } => vec![target, unwind].into_cow(),
            Assert { ref target, .. } => slice::ref_slice(target).into_cow(),
        }
    }

    // FIXME: no mootable cow. I’m honestly not sure what a “cow” between `&mut [BasicBlock]` and
    // `Vec<&mut BasicBlock>` would look like in the first place.
    pub fn successors_mut(&mut self) -> Vec<&mut BasicBlock> {
        use self::TerminatorKind::*;
        match *self {
            Goto { target: ref mut b } => vec![b],
            If { targets: (ref mut b1, ref mut b2), .. } => vec![b1, b2],
            Switch { targets: ref mut b, .. } => b.iter_mut().collect(),
            SwitchInt { targets: ref mut b, .. } => b.iter_mut().collect(),
            Resume => Vec::new(),
            Return => Vec::new(),
            Unreachable => Vec::new(),
            Call { destination: Some((_, ref mut t)), cleanup: Some(ref mut c), .. } => vec![t, c],
            Call { destination: Some((_, ref mut t)), cleanup: None, .. } => vec![t],
            Call { destination: None, cleanup: Some(ref mut c), .. } => vec![c],
            Call { destination: None, cleanup: None, .. } => vec![],
            DropAndReplace { ref mut target, unwind: Some(ref mut unwind), .. } |
            Drop { ref mut target, unwind: Some(ref mut unwind), .. } => vec![target, unwind],
            DropAndReplace { ref mut target, unwind: None, .. } |
            Drop { ref mut target, unwind: None, .. } => {
                vec![target]
            }
            Assert { ref mut target, cleanup: Some(ref mut unwind), .. } => vec![target, unwind],
            Assert { ref mut target, .. } => vec![target]
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

impl<'tcx> Debug for TerminatorKind<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        self.fmt_head(fmt)?;
        let successors = self.successors();
        let labels = self.fmt_successor_labels();
        assert_eq!(successors.len(), labels.len());

        match successors.len() {
            0 => Ok(()),

            1 => write!(fmt, " -> {:?}", successors[0]),

            _ => {
                write!(fmt, " -> [")?;
                for (i, target) in successors.iter().enumerate() {
                    if i > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{}: {:?}", labels[i], target)?;
                }
                write!(fmt, "]")
            }

        }
    }
}

impl<'tcx> TerminatorKind<'tcx> {
    /// Write the "head" part of the terminator; that is, its name and the data it uses to pick the
    /// successor basic block, if any. The only information not inlcuded is the list of possible
    /// successors, which may be rendered differently between the text and the graphviz format.
    pub fn fmt_head<W: Write>(&self, fmt: &mut W) -> fmt::Result {
        use self::TerminatorKind::*;
        match *self {
            Goto { .. } => write!(fmt, "goto"),
            If { cond: ref lv, .. } => write!(fmt, "if({:?})", lv),
            Switch { discr: ref lv, .. } => write!(fmt, "switch({:?})", lv),
            SwitchInt { discr: ref lv, .. } => write!(fmt, "switchInt({:?})", lv),
            Return => write!(fmt, "return"),
            Resume => write!(fmt, "resume"),
            Unreachable => write!(fmt, "unreachable"),
            Drop { ref location, .. } => write!(fmt, "drop({:?})", location),
            DropAndReplace { ref location, ref value, .. } =>
                write!(fmt, "replace({:?} <- {:?})", location, value),
            Call { ref func, ref args, ref destination, .. } => {
                if let Some((ref destination, _)) = *destination {
                    write!(fmt, "{:?} = ", destination)?;
                }
                write!(fmt, "{:?}(", func)?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{:?}", arg)?;
                }
                write!(fmt, ")")
            }
            Assert { ref cond, expected, ref msg, .. } => {
                write!(fmt, "assert(")?;
                if !expected {
                    write!(fmt, "!")?;
                }
                write!(fmt, "{:?}, ", cond)?;

                match *msg {
                    AssertMessage::BoundsCheck { ref len, ref index } => {
                        write!(fmt, "{:?}, {:?}, {:?}",
                               "index out of bounds: the len is {} but the index is {}",
                               len, index)?;
                    }
                    AssertMessage::Math(ref err) => {
                        write!(fmt, "{:?}", err.description())?;
                    }
                }

                write!(fmt, ")")
            }
        }
    }

    /// Return the list of labels for the edges to the successor basic blocks.
    pub fn fmt_successor_labels(&self) -> Vec<Cow<'static, str>> {
        use self::TerminatorKind::*;
        match *self {
            Return | Resume | Unreachable => vec![],
            Goto { .. } => vec!["".into()],
            If { .. } => vec!["true".into(), "false".into()],
            Switch { ref adt_def, .. } => {
                adt_def.variants
                       .iter()
                       .map(|variant| variant.name.to_string().into())
                       .collect()
            }
            SwitchInt { ref values, .. } => {
                values.iter()
                      .map(|const_val| {
                          let mut buf = String::new();
                          fmt_const_val(&mut buf, const_val).unwrap();
                          buf.into()
                      })
                      .chain(iter::once(String::from("otherwise").into()))
                      .collect()
            }
            Call { destination: Some(_), cleanup: Some(_), .. } =>
                vec!["return".into_cow(), "unwind".into_cow()],
            Call { destination: Some(_), cleanup: None, .. } => vec!["return".into_cow()],
            Call { destination: None, cleanup: Some(_), .. } => vec!["unwind".into_cow()],
            Call { destination: None, cleanup: None, .. } => vec![],
            DropAndReplace { unwind: None, .. } |
            Drop { unwind: None, .. } => vec!["return".into_cow()],
            DropAndReplace { unwind: Some(_), .. } |
            Drop { unwind: Some(_), .. } => {
                vec!["return".into_cow(), "unwind".into_cow()]
            }
            Assert { cleanup: None, .. } => vec!["".into()],
            Assert { .. } =>
                vec!["success".into_cow(), "unwind".into_cow()]
        }
    }
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum AssertMessage<'tcx> {
    BoundsCheck {
        len: Operand<'tcx>,
        index: Operand<'tcx>
    },
    Math(ConstMathErr)
}

///////////////////////////////////////////////////////////////////////////
// Statements

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Statement<'tcx> {
    pub source_info: SourceInfo,
    pub kind: StatementKind<'tcx>,
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum StatementKind<'tcx> {
    Assign(Lvalue<'tcx>, Rvalue<'tcx>),
}

impl<'tcx> Debug for Statement<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        use self::StatementKind::*;
        match self.kind {
            Assign(ref lv, ref rv) => write!(fmt, "{:?} = {:?}", lv, rv)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Lvalues

newtype_index!(Var, "var");
newtype_index!(Temp, "tmp");
newtype_index!(Arg, "arg");
newtype_index!(Local, "local");

/// A path to a value; something that can be evaluated without
/// changing or disturbing program state.
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Lvalue<'tcx> {
    /// local variable declared by the user
    Var(Var),

    /// temporary introduced during lowering into MIR
    Temp(Temp),

    /// formal parameter of the function; note that these are NOT the
    /// bindings that the user declares, which are vars
    Arg(Arg),

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
#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct Projection<'tcx, B, V> {
    pub base: B,
    pub elem: ProjectionElem<'tcx, V>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum ProjectionElem<'tcx, V> {
    Deref,
    Field(Field, Ty<'tcx>),
    Index(V),

    /// These indices are generated by slice patterns. Easiest to explain
    /// by example:
    ///
    /// ```
    /// [X, _, .._, _, _] => { offset: 0, min_length: 4, from_end: false },
    /// [_, X, .._, _, _] => { offset: 1, min_length: 4, from_end: false },
    /// [_, _, .._, X, _] => { offset: 2, min_length: 4, from_end: true },
    /// [_, _, .._, _, X] => { offset: 1, min_length: 4, from_end: true },
    /// ```
    ConstantIndex {
        /// index or -index (in Python terms), depending on from_end
        offset: u32,
        /// thing being indexed must be at least this long
        min_length: u32,
        /// counting backwards from end?
        from_end: bool,
    },

    /// These indices are generated by slice patterns.
    ///
    /// slice[from:-to] in Python terms.
    Subslice {
        from: u32,
        to: u32,
    },

    /// "Downcast" to a variant of an ADT. Currently, we only introduce
    /// this for ADTs with more than one variant. It may be better to
    /// just introduce it always, or always for enums.
    Downcast(AdtDef<'tcx>, usize),
}

/// Alias for projections as they appear in lvalues, where the base is an lvalue
/// and the index is an operand.
pub type LvalueProjection<'tcx> = Projection<'tcx, Lvalue<'tcx>, Operand<'tcx>>;

/// Alias for projections as they appear in lvalues, where the base is an lvalue
/// and the index is an operand.
pub type LvalueElem<'tcx> = ProjectionElem<'tcx, Operand<'tcx>>;

newtype_index!(Field, "field");

impl<'tcx> Lvalue<'tcx> {
    pub fn field(self, f: Field, ty: Ty<'tcx>) -> Lvalue<'tcx> {
        self.elem(ProjectionElem::Field(f, ty))
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
            Var(id) => write!(fmt, "{:?}", id),
            Arg(id) => write!(fmt, "{:?}", id),
            Temp(id) => write!(fmt, "{:?}", id),
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
                    ProjectionElem::Field(field, ty) =>
                        write!(fmt, "({:?}.{:?}: {:?})", data.base, field.index(), ty),
                    ProjectionElem::Index(ref index) =>
                        write!(fmt, "{:?}[{:?}]", data.base, index),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: false } =>
                        write!(fmt, "{:?}[{:?} of {:?}]", data.base, offset, min_length),
                    ProjectionElem::ConstantIndex { offset, min_length, from_end: true } =>
                        write!(fmt, "{:?}[-{:?} of {:?}]", data.base, offset, min_length),
                    ProjectionElem::Subslice { from, to } if to == 0 =>
                        write!(fmt, "{:?}[{:?}:", data.base, from),
                    ProjectionElem::Subslice { from, to } if from == 0 =>
                        write!(fmt, "{:?}[:-{:?}]", data.base, to),
                    ProjectionElem::Subslice { from, to } =>
                        write!(fmt, "{:?}[{:?}:-{:?}]", data.base,
                               from, to),

                },
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Scopes

newtype_index!(VisibilityScope, "scope");
pub const ARGUMENT_VISIBILITY_SCOPE : VisibilityScope = VisibilityScope(0);

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct VisibilityScopeData {
    pub span: Span,
    pub parent_scope: Option<VisibilityScope>,
}

///////////////////////////////////////////////////////////////////////////
// Operands

/// These are values that can appear inside an rvalue (or an index
/// lvalue). They are intentionally limited to prevent rvalues from
/// being nested in one another.
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
/// Rvalues

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum Rvalue<'tcx> {
    /// x (either a move or copy, depending on type of x)
    Use(Operand<'tcx>),

    /// [x; 32]
    Repeat(Operand<'tcx>, TypedConstVal<'tcx>),

    /// &x or &mut x
    Ref(Region, BorrowKind, Lvalue<'tcx>),

    /// length of a [X] or [X;n] value
    Len(Lvalue<'tcx>),

    Cast(CastKind, Operand<'tcx>, Ty<'tcx>),

    BinaryOp(BinOp, Operand<'tcx>, Operand<'tcx>),
    CheckedBinaryOp(BinOp, Operand<'tcx>, Operand<'tcx>),

    UnaryOp(UnOp, Operand<'tcx>),

    /// Creates an *uninitialized* Box
    Box(Ty<'tcx>),

    /// Create an aggregate value, like a tuple or struct.  This is
    /// only needed because we want to distinguish `dest = Foo { x:
    /// ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case
    /// that `Foo` has a destructor. These rvalues can be optimized
    /// away after type-checking and before lowering.
    Aggregate(AggregateKind<'tcx>, Vec<Operand<'tcx>>),

    InlineAsm {
        asm: InlineAsm,
        outputs: Vec<Lvalue<'tcx>>,
        inputs: Vec<Operand<'tcx>>
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
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
    Closure(DefId, ClosureSubsts<'tcx>),
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

impl BinOp {
    pub fn is_checkable(self) -> bool {
        use self::BinOp::*;
        match self {
            Add | Sub | Mul | Shl | Shr => true,
            _ => false
        }
    }
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
            CheckedBinaryOp(ref op, ref a, ref b) => {
                write!(fmt, "Checked{:?}({:?}, {:?})", op, a, b)
            }
            UnaryOp(ref op, ref a) => write!(fmt, "{:?}({:?})", op, a),
            Box(ref t) => write!(fmt, "Box({:?})", t),
            InlineAsm { ref asm, ref outputs, ref inputs } => {
                write!(fmt, "asm!({:?} : {:?} : {:?})", asm, outputs, inputs)
            }

            Ref(_, borrow_kind, ref lv) => {
                let kind_str = match borrow_kind {
                    BorrowKind::Shared => "",
                    BorrowKind::Mut | BorrowKind::Unique => "mut ",
                };
                write!(fmt, "&{}{:?}", kind_str, lv)
            }

            Aggregate(ref kind, ref lvs) => {
                use self::AggregateKind::*;

                fn fmt_tuple(fmt: &mut Formatter, lvs: &[Operand]) -> fmt::Result {
                    let mut tuple_fmt = fmt.debug_tuple("");
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
                            _ => fmt_tuple(fmt, lvs),
                        }
                    }

                    Adt(adt_def, variant, substs) => {
                        let variant_def = &adt_def.variants[variant];

                        ppaux::parameterized(fmt, substs, variant_def.did,
                                             ppaux::Ns::Value, &[],
                                             |tcx| {
                            Some(tcx.lookup_item_type(variant_def.did).generics)
                        })?;

                        match variant_def.kind() {
                            ty::VariantKind::Unit => Ok(()),
                            ty::VariantKind::Tuple => fmt_tuple(fmt, lvs),
                            ty::VariantKind::Struct => {
                                let mut struct_fmt = fmt.debug_struct("");
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
/// Constants
///
/// Two constants are equal if they are the same constant. Note that
/// this does not necessarily mean that they are "==" in Rust -- in
/// particular one must be wary of `NaN`!

#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct Constant<'tcx> {
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub literal: Literal<'tcx>,
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct TypedConstVal<'tcx> {
    pub ty: Ty<'tcx>,
    pub span: Span,
    pub value: ConstUsize,
}

impl<'tcx> Debug for TypedConstVal<'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "const {}", ConstInt::Usize(self.value))
    }
}

newtype_index!(Promoted, "promoted");

#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum Literal<'tcx> {
    Item {
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
    },
    Value {
        value: ConstVal,
    },
    Promoted {
        // Index into the `promoted` vector of `Mir`.
        index: Promoted
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
            Item { def_id, substs } => {
                ppaux::parameterized(
                    fmt, substs, def_id, ppaux::Ns::Value, &[],
                    |tcx| Some(tcx.lookup_item_type(def_id).generics))
            }
            Value { ref value } => {
                write!(fmt, "const ")?;
                fmt_const_val(fmt, value)
            }
            Promoted { index } => {
                write!(fmt, "{:?}", index)
            }
        }
    }
}

/// Write a `ConstVal` in a way closer to the original source code than the `Debug` output.
fn fmt_const_val<W: Write>(fmt: &mut W, const_val: &ConstVal) -> fmt::Result {
    use middle::const_val::ConstVal::*;
    match *const_val {
        Float(f) => write!(fmt, "{:?}", f),
        Integral(n) => write!(fmt, "{}", n),
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
        Char(c) => write!(fmt, "{:?}", c),
        Dummy => bug!(),
    }
}

fn node_to_string(node_id: ast::NodeId) -> String {
    ty::tls::with(|tcx| tcx.map.node_to_user_string(node_id))
}

fn item_path_str(def_id: DefId) -> String {
    ty::tls::with(|tcx| tcx.item_path_str(def_id))
}
