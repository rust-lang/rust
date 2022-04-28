//! MIR datatypes and passes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/mir/index.html

use crate::mir::coverage::{CodeRegion, CoverageKind};
use crate::mir::interpret::{ConstAllocation, ConstValue, GlobalAlloc, Scalar};
use crate::mir::visit::MirVisitable;
use crate::ty::adjustment::PointerCast;
use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::fold::{FallibleTypeFolder, TypeFoldable, TypeVisitor};
use crate::ty::print::{FmtPrinter, Printer};
use crate::ty::subst::{GenericArg, InternalSubsts, Subst, SubstsRef};
use crate::ty::{self, List, Ty, TyCtxt};
use crate::ty::{AdtDef, InstanceDef, Region, ScalarInt, UserTypeAnnotationIndex};

use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::{CtorKind, Namespace};
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_hir::{self, GeneratorKind};
use rustc_hir::{self as hir, HirId};
use rustc_session::Session;
use rustc_target::abi::{Size, VariantIdx};

use polonius_engine::Atom;
pub use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::dominators::{dominators, Dominators};
use rustc_data_structures::graph::{self, GraphSuccessors};
use rustc_index::bit_set::BitMatrix;
use rustc_index::vec::{Idx, IndexVec};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::symbol::Symbol;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::asm::InlineAsmRegOrRegClass;

use either::Either;

use std::borrow::Cow;
use std::convert::TryInto;
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::ops::{ControlFlow, Index, IndexMut};
use std::slice;
use std::{iter, mem, option};

use self::graph_cyclic_cache::GraphIsCyclicCache;
use self::predecessors::{PredecessorCache, Predecessors};
pub use self::query::*;
use self::switch_sources::{SwitchSourceCache, SwitchSources};

pub mod coverage;
mod generic_graph;
pub mod generic_graphviz;
mod graph_cyclic_cache;
pub mod graphviz;
pub mod interpret;
pub mod mono;
pub mod patch;
mod predecessors;
pub mod pretty;
mod query;
pub mod spanview;
mod switch_sources;
pub mod tcx;
pub mod terminator;
use crate::mir::traversal::PostorderCache;
pub use terminator::*;

pub mod traversal;
mod type_foldable;
pub mod visit;

pub use self::generic_graph::graphviz_safe_def_name;
pub use self::graphviz::write_mir_graphviz;
pub use self::pretty::{
    create_dump_file, display_allocation, dump_enabled, dump_mir, write_mir_pretty, PassWhere,
};

/// Types for locals
pub type LocalDecls<'tcx> = IndexVec<Local, LocalDecl<'tcx>>;

pub trait HasLocalDecls<'tcx> {
    fn local_decls(&self) -> &LocalDecls<'tcx>;
}

impl<'tcx> HasLocalDecls<'tcx> for LocalDecls<'tcx> {
    #[inline]
    fn local_decls(&self) -> &LocalDecls<'tcx> {
        self
    }
}

impl<'tcx> HasLocalDecls<'tcx> for Body<'tcx> {
    #[inline]
    fn local_decls(&self) -> &LocalDecls<'tcx> {
        &self.local_decls
    }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass<'tcx> {
    fn name(&self) -> Cow<'_, str> {
        let name = std::any::type_name::<Self>();
        if let Some(tail) = name.rfind(':') {
            Cow::from(&name[tail + 1..])
        } else {
            Cow::from(name)
        }
    }

    /// Returns `true` if this pass is enabled with the current combination of compiler flags.
    fn is_enabled(&self, _sess: &Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);

    /// If this pass causes the MIR to enter a new phase, return that phase.
    fn phase_change(&self) -> Option<MirPhase> {
        None
    }

    fn is_mir_dump_enabled(&self) -> bool {
        true
    }
}

/// The various "big phases" that MIR goes through.
///
/// These phases all describe dialects of MIR. Since all MIR uses the same datastructures, the
/// dialects forbid certain variants or values in certain phases. The sections below summarize the
/// changes, but do not document them thoroughly. The full documentation is found in the appropriate
/// documentation for the thing the change is affecting.
///
/// Warning: ordering of variants is significant.
#[derive(Copy, Clone, TyEncodable, TyDecodable, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[derive(HashStable)]
pub enum MirPhase {
    /// The dialect of MIR used during all phases before `DropsLowered` is the same. This is also
    /// the MIR that analysis such as borrowck uses.
    ///
    /// One important thing to remember about the behavior of this section of MIR is that drop terminators
    /// (including drop and replace) are *conditional*. The elaborate drops pass will then replace each
    /// instance of a drop terminator with a nop, an unconditional drop, or a drop conditioned on a drop
    /// flag. Of course, this means that it is important that the drop elaboration can accurately recognize
    /// when things are initialized and when things are de-initialized. That means any code running on this
    /// version of MIR must be sure to produce output that drop elaboration can reason about. See the
    /// section on the drop terminatorss for more details.
    Built = 0,
    // FIXME(oli-obk): it's unclear whether we still need this phase (and its corresponding query).
    // We used to have this for pre-miri MIR based const eval.
    Const = 1,
    /// This phase checks the MIR for promotable elements and takes them out of the main MIR body
    /// by creating a new MIR body per promoted element. After this phase (and thus the termination
    /// of the `mir_promoted` query), these promoted elements are available in the `promoted_mir`
    /// query.
    ConstsPromoted = 2,
    /// Beginning with this phase, the following variants are disallowed:
    /// * [`TerminatorKind::DropAndReplace`](terminator::TerminatorKind::DropAndReplace)
    /// * [`TerminatorKind::FalseUnwind`](terminator::TerminatorKind::FalseUnwind)
    /// * [`TerminatorKind::FalseEdge`](terminator::TerminatorKind::FalseEdge)
    /// * [`StatementKind::FakeRead`]
    /// * [`StatementKind::AscribeUserType`]
    /// * [`Rvalue::Ref`] with `BorrowKind::Shallow`
    ///
    /// And the following variant is allowed:
    /// * [`StatementKind::Retag`]
    ///
    /// Furthermore, `Drop` now uses explicit drop flags visible in the MIR and reaching a `Drop`
    /// terminator means that the auto-generated drop glue will be invoked. Also, `Copy` operands
    /// are allowed for non-`Copy` types.
    DropsLowered = 3,
    /// Beginning with this phase, the following variant is disallowed:
    /// * [`Rvalue::Aggregate`] for any `AggregateKind` except `Array`
    ///
    /// And the following variant is allowed:
    /// * [`StatementKind::SetDiscriminant`]
    Deaggregated = 4,
    /// Before this phase, generators are in the "source code" form, featuring `yield` statements
    /// and such. With this phase change, they are transformed into a proper state machine. Running
    /// optimizations before this change can be potentially dangerous because the source code is to
    /// some extent a "lie." In particular, `yield` terminators effectively make the value of all
    /// locals visible to the caller. This means that dead store elimination before them, or code
    /// motion across them, is not correct in general. This is also exasperated by type checking
    /// having pre-computed a list of the types that it thinks are ok to be live across a yield
    /// point - this is necessary to decide eg whether autotraits are implemented. Introducing new
    /// types across a yield point will lead to ICEs becaues of this.
    ///
    /// Beginning with this phase, the following variants are disallowed:
    /// * [`TerminatorKind::Yield`](terminator::TerminatorKind::Yield)
    /// * [`TerminatorKind::GeneratorDrop](terminator::TerminatorKind::GeneratorDrop)
    GeneratorsLowered = 5,
    Optimized = 6,
}

impl MirPhase {
    /// Gets the index of the current MirPhase within the set of all `MirPhase`s.
    pub fn phase_index(&self) -> usize {
        *self as usize
    }
}

/// Where a specific `mir::Body` comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeFoldable)]
pub struct MirSource<'tcx> {
    pub instance: InstanceDef<'tcx>,

    /// If `Some`, this is a promoted rvalue within the parent function.
    pub promoted: Option<Promoted>,
}

impl<'tcx> MirSource<'tcx> {
    pub fn item(def_id: DefId) -> Self {
        MirSource {
            instance: InstanceDef::Item(ty::WithOptConstParam::unknown(def_id)),
            promoted: None,
        }
    }

    pub fn from_instance(instance: InstanceDef<'tcx>) -> Self {
        MirSource { instance, promoted: None }
    }

    pub fn with_opt_param(self) -> ty::WithOptConstParam<DefId> {
        self.instance.with_opt_param()
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.instance.def_id()
    }
}

#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable)]
pub struct GeneratorInfo<'tcx> {
    /// The yield type of the function, if it is a generator.
    pub yield_ty: Option<Ty<'tcx>>,

    /// Generator drop glue.
    pub generator_drop: Option<Body<'tcx>>,

    /// The layout of a generator. Produced by the state transformation.
    pub generator_layout: Option<GeneratorLayout<'tcx>>,

    /// If this is a generator then record the type of source expression that caused this generator
    /// to be created.
    pub generator_kind: GeneratorKind,
}

/// The lowered representation of a single function.
#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable)]
pub struct Body<'tcx> {
    /// A list of basic blocks. References to basic block use a newtyped index type [`BasicBlock`]
    /// that indexes into this vector.
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,

    /// Records how far through the "desugaring and optimization" process this particular
    /// MIR has traversed. This is particularly useful when inlining, since in that context
    /// we instantiate the promoted constants and add them to our promoted vector -- but those
    /// promoted items have already been optimized, whereas ours have not. This field allows
    /// us to see the difference and forego optimization on the inlined promoted items.
    pub phase: MirPhase,

    pub source: MirSource<'tcx>,

    /// A list of source scopes; these are referenced by statements
    /// and used for debuginfo. Indexed by a `SourceScope`.
    pub source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,

    pub generator: Option<Box<GeneratorInfo<'tcx>>>,

    /// Declarations of locals.
    ///
    /// The first local is the return value pointer, followed by `arg_count`
    /// locals for the function arguments, followed by any user-declared
    /// variables and temporaries.
    pub local_decls: LocalDecls<'tcx>,

    /// User type annotations.
    pub user_type_annotations: ty::CanonicalUserTypeAnnotations<'tcx>,

    /// The number of arguments this function takes.
    ///
    /// Starting at local 1, `arg_count` locals will be provided by the caller
    /// and can be assumed to be initialized.
    ///
    /// If this MIR was built for a constant, this will be 0.
    pub arg_count: usize,

    /// Mark an argument local (which must be a tuple) as getting passed as
    /// its individual components at the LLVM level.
    ///
    /// This is used for the "rust-call" ABI.
    pub spread_arg: Option<Local>,

    /// Debug information pertaining to user variables, including captures.
    pub var_debug_info: Vec<VarDebugInfo<'tcx>>,

    /// A span representing this MIR, for error reporting.
    pub span: Span,

    /// Constants that are required to evaluate successfully for this MIR to be well-formed.
    /// We hold in this field all the constants we are not able to evaluate yet.
    pub required_consts: Vec<Constant<'tcx>>,

    /// Does this body use generic parameters. This is used for the `ConstEvaluatable` check.
    ///
    /// Note that this does not actually mean that this body is not computable right now.
    /// The repeat count in the following example is polymorphic, but can still be evaluated
    /// without knowing anything about the type parameter `T`.
    ///
    /// ```rust
    /// fn test<T>() {
    ///     let _ = [0; std::mem::size_of::<*mut T>()];
    /// }
    /// ```
    ///
    /// **WARNING**: Do not change this flags after the MIR was originally created, even if an optimization
    /// removed the last mention of all generic params. We do not want to rely on optimizations and
    /// potentially allow things like `[u8; std::mem::size_of::<T>() * 0]` due to this.
    pub is_polymorphic: bool,

    predecessor_cache: PredecessorCache,
    switch_source_cache: SwitchSourceCache,
    is_cyclic: GraphIsCyclicCache,
    postorder_cache: PostorderCache,

    pub tainted_by_errors: Option<ErrorGuaranteed>,
}

impl<'tcx> Body<'tcx> {
    pub fn new(
        source: MirSource<'tcx>,
        basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
        source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,
        local_decls: LocalDecls<'tcx>,
        user_type_annotations: ty::CanonicalUserTypeAnnotations<'tcx>,
        arg_count: usize,
        var_debug_info: Vec<VarDebugInfo<'tcx>>,
        span: Span,
        generator_kind: Option<GeneratorKind>,
        tainted_by_errors: Option<ErrorGuaranteed>,
    ) -> Self {
        // We need `arg_count` locals, and one for the return place.
        assert!(
            local_decls.len() > arg_count,
            "expected at least {} locals, got {}",
            arg_count + 1,
            local_decls.len()
        );

        let mut body = Body {
            phase: MirPhase::Built,
            source,
            basic_blocks,
            source_scopes,
            generator: generator_kind.map(|generator_kind| {
                Box::new(GeneratorInfo {
                    yield_ty: None,
                    generator_drop: None,
                    generator_layout: None,
                    generator_kind,
                })
            }),
            local_decls,
            user_type_annotations,
            arg_count,
            spread_arg: None,
            var_debug_info,
            span,
            required_consts: Vec::new(),
            is_polymorphic: false,
            predecessor_cache: PredecessorCache::new(),
            switch_source_cache: SwitchSourceCache::new(),
            is_cyclic: GraphIsCyclicCache::new(),
            postorder_cache: PostorderCache::new(),
            tainted_by_errors,
        };
        body.is_polymorphic = body.has_param_types_or_consts();
        body
    }

    /// Returns a partially initialized MIR body containing only a list of basic blocks.
    ///
    /// The returned MIR contains no `LocalDecl`s (even for the return place) or source scopes. It
    /// is only useful for testing but cannot be `#[cfg(test)]` because it is used in a different
    /// crate.
    pub fn new_cfg_only(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>) -> Self {
        let mut body = Body {
            phase: MirPhase::Built,
            source: MirSource::item(CRATE_DEF_ID.to_def_id()),
            basic_blocks,
            source_scopes: IndexVec::new(),
            generator: None,
            local_decls: IndexVec::new(),
            user_type_annotations: IndexVec::new(),
            arg_count: 0,
            spread_arg: None,
            span: DUMMY_SP,
            required_consts: Vec::new(),
            var_debug_info: Vec::new(),
            is_polymorphic: false,
            predecessor_cache: PredecessorCache::new(),
            switch_source_cache: SwitchSourceCache::new(),
            is_cyclic: GraphIsCyclicCache::new(),
            postorder_cache: PostorderCache::new(),
            tainted_by_errors: None,
        };
        body.is_polymorphic = body.has_param_types_or_consts();
        body
    }

    #[inline]
    pub fn basic_blocks(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &self.basic_blocks
    }

    #[inline]
    pub fn basic_blocks_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        // Because the user could mutate basic block terminators via this reference, we need to
        // invalidate the caches.
        //
        // FIXME: Use a finer-grained API for this, so only transformations that alter terminators
        // invalidate the caches.
        self.predecessor_cache.invalidate();
        self.switch_source_cache.invalidate();
        self.is_cyclic.invalidate();
        self.postorder_cache.invalidate();
        &mut self.basic_blocks
    }

    #[inline]
    pub fn basic_blocks_and_local_decls_mut(
        &mut self,
    ) -> (&mut IndexVec<BasicBlock, BasicBlockData<'tcx>>, &mut LocalDecls<'tcx>) {
        self.predecessor_cache.invalidate();
        self.switch_source_cache.invalidate();
        self.is_cyclic.invalidate();
        self.postorder_cache.invalidate();
        (&mut self.basic_blocks, &mut self.local_decls)
    }

    #[inline]
    pub fn basic_blocks_local_decls_mut_and_var_debug_info(
        &mut self,
    ) -> (
        &mut IndexVec<BasicBlock, BasicBlockData<'tcx>>,
        &mut LocalDecls<'tcx>,
        &mut Vec<VarDebugInfo<'tcx>>,
    ) {
        self.predecessor_cache.invalidate();
        self.switch_source_cache.invalidate();
        self.is_cyclic.invalidate();
        self.postorder_cache.invalidate();
        (&mut self.basic_blocks, &mut self.local_decls, &mut self.var_debug_info)
    }

    /// Returns `true` if a cycle exists in the control-flow graph that is reachable from the
    /// `START_BLOCK`.
    pub fn is_cfg_cyclic(&self) -> bool {
        self.is_cyclic.is_cyclic(self)
    }

    #[inline]
    pub fn local_kind(&self, local: Local) -> LocalKind {
        let index = local.as_usize();
        if index == 0 {
            debug_assert!(
                self.local_decls[local].mutability == Mutability::Mut,
                "return place should be mutable"
            );

            LocalKind::ReturnPointer
        } else if index < self.arg_count + 1 {
            LocalKind::Arg
        } else if self.local_decls[local].is_user_variable() {
            LocalKind::Var
        } else {
            LocalKind::Temp
        }
    }

    /// Returns an iterator over all user-declared mutable locals.
    #[inline]
    pub fn mut_vars_iter<'a>(&'a self) -> impl Iterator<Item = Local> + 'a {
        (self.arg_count + 1..self.local_decls.len()).filter_map(move |index| {
            let local = Local::new(index);
            let decl = &self.local_decls[local];
            if decl.is_user_variable() && decl.mutability == Mutability::Mut {
                Some(local)
            } else {
                None
            }
        })
    }

    /// Returns an iterator over all user-declared mutable arguments and locals.
    #[inline]
    pub fn mut_vars_and_args_iter<'a>(&'a self) -> impl Iterator<Item = Local> + 'a {
        (1..self.local_decls.len()).filter_map(move |index| {
            let local = Local::new(index);
            let decl = &self.local_decls[local];
            if (decl.is_user_variable() || index < self.arg_count + 1)
                && decl.mutability == Mutability::Mut
            {
                Some(local)
            } else {
                None
            }
        })
    }

    /// Returns an iterator over all function arguments.
    #[inline]
    pub fn args_iter(&self) -> impl Iterator<Item = Local> + ExactSizeIterator {
        (1..self.arg_count + 1).map(Local::new)
    }

    /// Returns an iterator over all user-defined variables and compiler-generated temporaries (all
    /// locals that are neither arguments nor the return place).
    #[inline]
    pub fn vars_and_temps_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = Local> + ExactSizeIterator {
        (self.arg_count + 1..self.local_decls.len()).map(Local::new)
    }

    #[inline]
    pub fn drain_vars_and_temps<'a>(&'a mut self) -> impl Iterator<Item = LocalDecl<'tcx>> + 'a {
        self.local_decls.drain(self.arg_count + 1..)
    }

    /// Changes a statement to a nop. This is both faster than deleting instructions and avoids
    /// invalidating statement indices in `Location`s.
    pub fn make_statement_nop(&mut self, location: Location) {
        let block = &mut self.basic_blocks[location.block];
        debug_assert!(location.statement_index < block.statements.len());
        block.statements[location.statement_index].make_nop()
    }

    /// Returns the source info associated with `location`.
    pub fn source_info(&self, location: Location) -> &SourceInfo {
        let block = &self[location.block];
        let stmts = &block.statements;
        let idx = location.statement_index;
        if idx < stmts.len() {
            &stmts[idx].source_info
        } else {
            assert_eq!(idx, stmts.len());
            &block.terminator().source_info
        }
    }

    /// Returns the return type; it always return first element from `local_decls` array.
    #[inline]
    pub fn return_ty(&self) -> Ty<'tcx> {
        self.local_decls[RETURN_PLACE].ty
    }

    /// Gets the location of the terminator for the given block.
    #[inline]
    pub fn terminator_loc(&self, bb: BasicBlock) -> Location {
        Location { block: bb, statement_index: self[bb].statements.len() }
    }

    pub fn stmt_at(&self, location: Location) -> Either<&Statement<'tcx>, &Terminator<'tcx>> {
        let Location { block, statement_index } = location;
        let block_data = &self.basic_blocks[block];
        block_data
            .statements
            .get(statement_index)
            .map(Either::Left)
            .unwrap_or_else(|| Either::Right(block_data.terminator()))
    }

    #[inline]
    pub fn predecessors(&self) -> &Predecessors {
        self.predecessor_cache.compute(&self.basic_blocks)
    }

    #[inline]
    pub fn switch_sources(&self) -> &SwitchSources {
        self.switch_source_cache.compute(&self.basic_blocks)
    }

    #[inline]
    pub fn dominators(&self) -> Dominators<BasicBlock> {
        dominators(self)
    }

    #[inline]
    pub fn yield_ty(&self) -> Option<Ty<'tcx>> {
        self.generator.as_ref().and_then(|generator| generator.yield_ty)
    }

    #[inline]
    pub fn generator_layout(&self) -> Option<&GeneratorLayout<'tcx>> {
        self.generator.as_ref().and_then(|generator| generator.generator_layout.as_ref())
    }

    #[inline]
    pub fn generator_drop(&self) -> Option<&Body<'tcx>> {
        self.generator.as_ref().and_then(|generator| generator.generator_drop.as_ref())
    }

    #[inline]
    pub fn generator_kind(&self) -> Option<GeneratorKind> {
        self.generator.as_ref().map(|generator| generator.generator_kind)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum Safety {
    Safe,
    /// Unsafe because of compiler-generated unsafe code, like `await` desugaring
    BuiltinUnsafe,
    /// Unsafe because of an unsafe fn
    FnUnsafe,
    /// Unsafe because of an `unsafe` block
    ExplicitUnsafe(hir::HirId),
}

impl<'tcx> Index<BasicBlock> for Body<'tcx> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks()[index]
    }
}

impl<'tcx> IndexMut<BasicBlock> for Body<'tcx> {
    #[inline]
    fn index_mut(&mut self, index: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks_mut()[index]
    }
}

#[derive(Copy, Clone, Debug, HashStable, TypeFoldable)]
pub enum ClearCrossCrate<T> {
    Clear,
    Set(T),
}

impl<T> ClearCrossCrate<T> {
    pub fn as_ref(&self) -> ClearCrossCrate<&T> {
        match self {
            ClearCrossCrate::Clear => ClearCrossCrate::Clear,
            ClearCrossCrate::Set(v) => ClearCrossCrate::Set(v),
        }
    }

    pub fn assert_crate_local(self) -> T {
        match self {
            ClearCrossCrate::Clear => bug!("unwrapping cross-crate data"),
            ClearCrossCrate::Set(v) => v,
        }
    }
}

const TAG_CLEAR_CROSS_CRATE_CLEAR: u8 = 0;
const TAG_CLEAR_CROSS_CRATE_SET: u8 = 1;

impl<'tcx, E: TyEncoder<'tcx>, T: Encodable<E>> Encodable<E> for ClearCrossCrate<T> {
    #[inline]
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        if E::CLEAR_CROSS_CRATE {
            return Ok(());
        }

        match *self {
            ClearCrossCrate::Clear => TAG_CLEAR_CROSS_CRATE_CLEAR.encode(e),
            ClearCrossCrate::Set(ref val) => {
                TAG_CLEAR_CROSS_CRATE_SET.encode(e)?;
                val.encode(e)
            }
        }
    }
}
impl<'tcx, D: TyDecoder<'tcx>, T: Decodable<D>> Decodable<D> for ClearCrossCrate<T> {
    #[inline]
    fn decode(d: &mut D) -> ClearCrossCrate<T> {
        if D::CLEAR_CROSS_CRATE {
            return ClearCrossCrate::Clear;
        }

        let discr = u8::decode(d);

        match discr {
            TAG_CLEAR_CROSS_CRATE_CLEAR => ClearCrossCrate::Clear,
            TAG_CLEAR_CROSS_CRATE_SET => {
                let val = T::decode(d);
                ClearCrossCrate::Set(val)
            }
            tag => panic!("Invalid tag for ClearCrossCrate: {:?}", tag),
        }
    }
}

/// Grouped information about the source code origin of a MIR entity.
/// Intended to be inspected by diagnostics and debuginfo.
/// Most passes can work with it as a whole, within a single function.
// The unofficial Cranelift backend, at least as of #65828, needs `SourceInfo` to implement `Eq` and
// `Hash`. Please ping @bjorn3 if removing them.
#[derive(Copy, Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
pub struct SourceInfo {
    /// The source span for the AST pertaining to this MIR entity.
    pub span: Span,

    /// The source scope, keeping track of which bindings can be
    /// seen by debuginfo, active lint levels, `unsafe {...}`, etc.
    pub scope: SourceScope,
}

impl SourceInfo {
    #[inline]
    pub fn outermost(span: Span) -> Self {
        SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE }
    }
}

///////////////////////////////////////////////////////////////////////////
// Borrow kinds

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(Hash, HashStable)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Shared,

    /// The immediately borrowed place must be immutable, but projections from
    /// it don't need to be. For example, a shallow borrow of `a.b` doesn't
    /// conflict with a mutable borrow of `a.b.c`.
    ///
    /// This is used when lowering matches: when matching on a place we want to
    /// ensure that place have the same value from the start of the match until
    /// an arm is selected. This prevents this code from compiling:
    ///
    ///     let mut x = &Some(0);
    ///     match *x {
    ///         None => (),
    ///         Some(_) if { x = &None; false } => (),
    ///         Some(_) => (),
    ///     }
    ///
    /// This can't be a shared borrow because mutably borrowing (*x as Some).0
    /// should not prevent `if let None = x { ... }`, for example, because the
    /// mutating `(*x as Some).0` can't affect the discriminant of `x`.
    /// We can also report errors with this kind of borrow differently.
    Shallow,

    /// Data must be immutable but not aliasable. This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when the closure is
    /// borrowing or mutating a mutable referent, e.g.:
    ///
    ///     let x: &mut isize = ...;
    ///     let y = || *x += 5;
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    ///     struct Env { x: & &mut isize }
    ///     let x: &mut isize = ...;
    ///     let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    ///     fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// This is then illegal because you cannot mutate an `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    ///     struct Env { x: &mut &mut isize }
    ///     let x: &mut isize = ...;
    ///     let y = (&mut Env { &mut x }, fn_ptr); // changed from &x to &mut x
    ///     fn fn_ptr(env: &mut Env) { **env.x += 5; }
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
    Mut {
        /// `true` if this borrow arose from method-call auto-ref
        /// (i.e., `adjustment::Adjust::Borrow`).
        allow_two_phase_borrow: bool,
    },
}

impl BorrowKind {
    pub fn allows_two_phase_borrow(&self) -> bool {
        match *self {
            BorrowKind::Shared | BorrowKind::Shallow | BorrowKind::Unique => false,
            BorrowKind::Mut { allow_two_phase_borrow } => allow_two_phase_borrow,
        }
    }

    pub fn describe_mutability(&self) -> String {
        match *self {
            BorrowKind::Shared | BorrowKind::Shallow | BorrowKind::Unique => {
                "immutable".to_string()
            }
            BorrowKind::Mut { .. } => "mutable".to_string(),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Variables and temps

rustc_index::newtype_index! {
    pub struct Local {
        derive [HashStable]
        DEBUG_FORMAT = "_{}",
        const RETURN_PLACE = 0,
    }
}

impl Atom for Local {
    fn index(self) -> usize {
        Idx::index(self)
    }
}

/// Classifies locals into categories. See `Body::local_kind`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable)]
pub enum LocalKind {
    /// User-declared variable binding.
    Var,
    /// Compiler-introduced temporary.
    Temp,
    /// Function argument.
    Arg,
    /// Location of function's return value.
    ReturnPointer,
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct VarBindingForm<'tcx> {
    /// Is variable bound via `x`, `mut x`, `ref x`, or `ref mut x`?
    pub binding_mode: ty::BindingMode,
    /// If an explicit type was provided for this variable binding,
    /// this holds the source Span of that type.
    ///
    /// NOTE: if you want to change this to a `HirId`, be wary that
    /// doing so breaks incremental compilation (as of this writing),
    /// while a `Span` does not cause our tests to fail.
    pub opt_ty_info: Option<Span>,
    /// Place of the RHS of the =, or the subject of the `match` where this
    /// variable is initialized. None in the case of `let PATTERN;`.
    /// Some((None, ..)) in the case of and `let [mut] x = ...` because
    /// (a) the right-hand side isn't evaluated as a place expression.
    /// (b) it gives a way to separate this case from the remaining cases
    ///     for diagnostics.
    pub opt_match_place: Option<(Option<Place<'tcx>>, Span)>,
    /// The span of the pattern in which this variable was bound.
    pub pat_span: Span,
}

#[derive(Clone, Debug, TyEncodable, TyDecodable)]
pub enum BindingForm<'tcx> {
    /// This is a binding for a non-`self` binding, or a `self` that has an explicit type.
    Var(VarBindingForm<'tcx>),
    /// Binding for a `self`/`&self`/`&mut self` binding where the type is implicit.
    ImplicitSelf(ImplicitSelfKind),
    /// Reference used in a guard expression to ensure immutability.
    RefForGuard,
}

/// Represents what type of implicit self a function has, if any.
#[derive(Clone, Copy, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum ImplicitSelfKind {
    /// Represents a `fn x(self);`.
    Imm,
    /// Represents a `fn x(mut self);`.
    Mut,
    /// Represents a `fn x(&self);`.
    ImmRef,
    /// Represents a `fn x(&mut self);`.
    MutRef,
    /// Represents when a function does not have a self argument or
    /// when a function has a `self: X` argument.
    None,
}

TrivialTypeFoldableAndLiftImpls! { BindingForm<'tcx>, }

mod binding_form_impl {
    use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
    use rustc_query_system::ich::StableHashingContext;

    impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for super::BindingForm<'tcx> {
        fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
            use super::BindingForm::*;
            std::mem::discriminant(self).hash_stable(hcx, hasher);

            match self {
                Var(binding) => binding.hash_stable(hcx, hasher),
                ImplicitSelf(kind) => kind.hash_stable(hcx, hasher),
                RefForGuard => (),
            }
        }
    }
}

/// `BlockTailInfo` is attached to the `LocalDecl` for temporaries
/// created during evaluation of expressions in a block tail
/// expression; that is, a block like `{ STMT_1; STMT_2; EXPR }`.
///
/// It is used to improve diagnostics when such temporaries are
/// involved in borrow_check errors, e.g., explanations of where the
/// temporaries come from, when their destructors are run, and/or how
/// one might revise the code to satisfy the borrow checker's rules.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct BlockTailInfo {
    /// If `true`, then the value resulting from evaluating this tail
    /// expression is ignored by the block's expression context.
    ///
    /// Examples include `{ ...; tail };` and `let _ = { ...; tail };`
    /// but not e.g., `let _x = { ...; tail };`
    pub tail_result_is_ignored: bool,

    /// `Span` of the tail expression.
    pub span: Span,
}

/// A MIR local.
///
/// This can be a binding declared by the user, a temporary inserted by the compiler, a function
/// argument, or the return place.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub struct LocalDecl<'tcx> {
    /// Whether this is a mutable binding (i.e., `let x` or `let mut x`).
    ///
    /// Temporaries and the return place are always mutable.
    pub mutability: Mutability,

    // FIXME(matthewjasper) Don't store in this in `Body`
    pub local_info: Option<Box<LocalInfo<'tcx>>>,

    /// `true` if this is an internal local.
    ///
    /// These locals are not based on types in the source code and are only used
    /// for a few desugarings at the moment.
    ///
    /// The generator transformation will sanity check the locals which are live
    /// across a suspension point against the type components of the generator
    /// which type checking knows are live across a suspension point. We need to
    /// flag drop flags to avoid triggering this check as they are introduced
    /// outside of type inference.
    ///
    /// This should be sound because the drop flags are fully algebraic, and
    /// therefore don't affect the auto-trait or outlives properties of the
    /// generator.
    pub internal: bool,

    /// If this local is a temporary and `is_block_tail` is `Some`,
    /// then it is a temporary created for evaluation of some
    /// subexpression of some block's tail expression (with no
    /// intervening statement context).
    // FIXME(matthewjasper) Don't store in this in `Body`
    pub is_block_tail: Option<BlockTailInfo>,

    /// The type of this local.
    pub ty: Ty<'tcx>,

    /// If the user manually ascribed a type to this variable,
    /// e.g., via `let x: T`, then we carry that type here. The MIR
    /// borrow checker needs this information since it can affect
    /// region inference.
    // FIXME(matthewjasper) Don't store in this in `Body`
    pub user_ty: Option<Box<UserTypeProjections>>,

    /// The *syntactic* (i.e., not visibility) source scope the local is defined
    /// in. If the local was defined in a let-statement, this
    /// is *within* the let-statement, rather than outside
    /// of it.
    ///
    /// This is needed because the visibility source scope of locals within
    /// a let-statement is weird.
    ///
    /// The reason is that we want the local to be *within* the let-statement
    /// for lint purposes, but we want the local to be *after* the let-statement
    /// for names-in-scope purposes.
    ///
    /// That's it, if we have a let-statement like the one in this
    /// function:
    ///
    /// ```
    /// fn foo(x: &str) {
    ///     #[allow(unused_mut)]
    ///     let mut x: u32 = { // <- one unused mut
    ///         let mut y: u32 = x.parse().unwrap();
    ///         y + 2
    ///     };
    ///     drop(x);
    /// }
    /// ```
    ///
    /// Then, from a lint point of view, the declaration of `x: u32`
    /// (and `y: u32`) are within the `#[allow(unused_mut)]` scope - the
    /// lint scopes are the same as the AST/HIR nesting.
    ///
    /// However, from a name lookup point of view, the scopes look more like
    /// as if the let-statements were `match` expressions:
    ///
    /// ```
    /// fn foo(x: &str) {
    ///     match {
    ///         match x.parse().unwrap() {
    ///             y => y + 2
    ///         }
    ///     } {
    ///         x => drop(x)
    ///     };
    /// }
    /// ```
    ///
    /// We care about the name-lookup scopes for debuginfo - if the
    /// debuginfo instruction pointer is at the call to `x.parse()`, we
    /// want `x` to refer to `x: &str`, but if it is at the call to
    /// `drop(x)`, we want it to refer to `x: u32`.
    ///
    /// To allow both uses to work, we need to have more than a single scope
    /// for a local. We have the `source_info.scope` represent the "syntactic"
    /// lint scope (with a variable being under its let block) while the
    /// `var_debug_info.source_info.scope` represents the "local variable"
    /// scope (where the "rest" of a block is under all prior let-statements).
    ///
    /// The end result looks like this:
    ///
    /// ```text
    /// ROOT SCOPE
    ///  │{ argument x: &str }
    ///  │
    ///  │ │{ #[allow(unused_mut)] } // This is actually split into 2 scopes
    ///  │ │                         // in practice because I'm lazy.
    ///  │ │
    ///  │ │← x.source_info.scope
    ///  │ │← `x.parse().unwrap()`
    ///  │ │
    ///  │ │ │← y.source_info.scope
    ///  │ │
    ///  │ │ │{ let y: u32 }
    ///  │ │ │
    ///  │ │ │← y.var_debug_info.source_info.scope
    ///  │ │ │← `y + 2`
    ///  │
    ///  │ │{ let x: u32 }
    ///  │ │← x.var_debug_info.source_info.scope
    ///  │ │← `drop(x)` // This accesses `x: u32`.
    /// ```
    pub source_info: SourceInfo,
}

// `LocalDecl` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(LocalDecl<'_>, 56);

/// Extra information about a some locals that's used for diagnostics and for
/// classifying variables into local variables, statics, etc, which is needed e.g.
/// for unsafety checking.
///
/// Not used for non-StaticRef temporaries, the return place, or anonymous
/// function parameters.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub enum LocalInfo<'tcx> {
    /// A user-defined local variable or function parameter
    ///
    /// The `BindingForm` is solely used for local diagnostics when generating
    /// warnings/errors when compiling the current crate, and therefore it need
    /// not be visible across crates.
    User(ClearCrossCrate<BindingForm<'tcx>>),
    /// A temporary created that references the static with the given `DefId`.
    StaticRef { def_id: DefId, is_thread_local: bool },
    /// A temporary created that references the const with the given `DefId`
    ConstRef { def_id: DefId },
    /// A temporary created during the creation of an aggregate
    /// (e.g. a temporary for `foo` in `MyStruct { my_field: foo }`)
    AggregateTemp,
}

impl<'tcx> LocalDecl<'tcx> {
    /// Returns `true` only if local is a binding that can itself be
    /// made mutable via the addition of the `mut` keyword, namely
    /// something like the occurrences of `x` in:
    /// - `fn foo(x: Type) { ... }`,
    /// - `let x = ...`,
    /// - or `match ... { C(x) => ... }`
    pub fn can_be_made_mutable(&self) -> bool {
        matches!(
            self.local_info,
            Some(box LocalInfo::User(ClearCrossCrate::Set(
                BindingForm::Var(VarBindingForm {
                    binding_mode: ty::BindingMode::BindByValue(_),
                    opt_ty_info: _,
                    opt_match_place: _,
                    pat_span: _,
                }) | BindingForm::ImplicitSelf(ImplicitSelfKind::Imm),
            )))
        )
    }

    /// Returns `true` if local is definitely not a `ref ident` or
    /// `ref mut ident` binding. (Such bindings cannot be made into
    /// mutable bindings, but the inverse does not necessarily hold).
    pub fn is_nonref_binding(&self) -> bool {
        matches!(
            self.local_info,
            Some(box LocalInfo::User(ClearCrossCrate::Set(
                BindingForm::Var(VarBindingForm {
                    binding_mode: ty::BindingMode::BindByValue(_),
                    opt_ty_info: _,
                    opt_match_place: _,
                    pat_span: _,
                }) | BindingForm::ImplicitSelf(_),
            )))
        )
    }

    /// Returns `true` if this variable is a named variable or function
    /// parameter declared by the user.
    #[inline]
    pub fn is_user_variable(&self) -> bool {
        matches!(self.local_info, Some(box LocalInfo::User(_)))
    }

    /// Returns `true` if this is a reference to a variable bound in a `match`
    /// expression that is used to access said variable for the guard of the
    /// match arm.
    pub fn is_ref_for_guard(&self) -> bool {
        matches!(
            self.local_info,
            Some(box LocalInfo::User(ClearCrossCrate::Set(BindingForm::RefForGuard)))
        )
    }

    /// Returns `Some` if this is a reference to a static item that is used to
    /// access that static.
    pub fn is_ref_to_static(&self) -> bool {
        matches!(self.local_info, Some(box LocalInfo::StaticRef { .. }))
    }

    /// Returns `Some` if this is a reference to a thread-local static item that is used to
    /// access that static.
    pub fn is_ref_to_thread_local(&self) -> bool {
        match self.local_info {
            Some(box LocalInfo::StaticRef { is_thread_local, .. }) => is_thread_local,
            _ => false,
        }
    }

    /// Returns `true` is the local is from a compiler desugaring, e.g.,
    /// `__next` from a `for` loop.
    #[inline]
    pub fn from_compiler_desugaring(&self) -> bool {
        self.source_info.span.desugaring_kind().is_some()
    }

    /// Creates a new `LocalDecl` for a temporary: mutable, non-internal.
    #[inline]
    pub fn new(ty: Ty<'tcx>, span: Span) -> Self {
        Self::with_source_info(ty, SourceInfo::outermost(span))
    }

    /// Like `LocalDecl::new`, but takes a `SourceInfo` instead of a `Span`.
    #[inline]
    pub fn with_source_info(ty: Ty<'tcx>, source_info: SourceInfo) -> Self {
        LocalDecl {
            mutability: Mutability::Mut,
            local_info: None,
            internal: false,
            is_block_tail: None,
            ty,
            user_ty: None,
            source_info,
        }
    }

    /// Converts `self` into same `LocalDecl` except tagged as internal.
    #[inline]
    pub fn internal(mut self) -> Self {
        self.internal = true;
        self
    }

    /// Converts `self` into same `LocalDecl` except tagged as immutable.
    #[inline]
    pub fn immutable(mut self) -> Self {
        self.mutability = Mutability::Not;
        self
    }

    /// Converts `self` into same `LocalDecl` except tagged as internal temporary.
    #[inline]
    pub fn block_tail(mut self, info: BlockTailInfo) -> Self {
        assert!(self.is_block_tail.is_none());
        self.is_block_tail = Some(info);
        self
    }
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub enum VarDebugInfoContents<'tcx> {
    /// NOTE(eddyb) There's an unenforced invariant that this `Place` is
    /// based on a `Local`, not a `Static`, and contains no indexing.
    Place(Place<'tcx>),
    Const(Constant<'tcx>),
}

impl<'tcx> Debug for VarDebugInfoContents<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VarDebugInfoContents::Const(c) => write!(fmt, "{}", c),
            VarDebugInfoContents::Place(p) => write!(fmt, "{:?}", p),
        }
    }
}

/// Debug information pertaining to a user variable.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub struct VarDebugInfo<'tcx> {
    pub name: Symbol,

    /// Source info of the user variable, including the scope
    /// within which the variable is visible (to debuginfo)
    /// (see `LocalDecl`'s `source_info` field for more details).
    pub source_info: SourceInfo,

    /// Where the data for this user variable is to be found.
    pub value: VarDebugInfoContents<'tcx>,
}

///////////////////////////////////////////////////////////////////////////
// BasicBlock

rustc_index::newtype_index! {
    /// A node in the MIR [control-flow graph][CFG].
    ///
    /// There are no branches (e.g., `if`s, function calls, etc.) within a basic block, which makes
    /// it easier to do [data-flow analyses] and optimizations. Instead, branches are represented
    /// as an edge in a graph between basic blocks.
    ///
    /// Basic blocks consist of a series of [statements][Statement], ending with a
    /// [terminator][Terminator]. Basic blocks can have multiple predecessors and successors,
    /// however there is a MIR pass ([`CriticalCallEdges`]) that removes *critical edges*, which
    /// are edges that go from a multi-successor node to a multi-predecessor node. This pass is
    /// needed because some analyses require that there are no critical edges in the CFG.
    ///
    /// Note that this type is just an index into [`Body.basic_blocks`](Body::basic_blocks);
    /// the actual data that a basic block holds is in [`BasicBlockData`].
    ///
    /// Read more about basic blocks in the [rustc-dev-guide][guide-mir].
    ///
    /// [CFG]: https://rustc-dev-guide.rust-lang.org/appendix/background.html#cfg
    /// [data-flow analyses]:
    ///     https://rustc-dev-guide.rust-lang.org/appendix/background.html#what-is-a-dataflow-analysis
    /// [`CriticalCallEdges`]: ../../rustc_const_eval/transform/add_call_guards/enum.AddCallGuards.html#variant.CriticalCallEdges
    /// [guide-mir]: https://rustc-dev-guide.rust-lang.org/mir/
    pub struct BasicBlock {
        derive [HashStable]
        DEBUG_FORMAT = "bb{}",
        const START_BLOCK = 0,
    }
}

impl BasicBlock {
    pub fn start_location(self) -> Location {
        Location { block: self, statement_index: 0 }
    }
}

///////////////////////////////////////////////////////////////////////////
// BasicBlockData and Terminator

/// See [`BasicBlock`] for documentation on what basic blocks are at a high level.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub struct BasicBlockData<'tcx> {
    /// List of statements in this block.
    pub statements: Vec<Statement<'tcx>>,

    /// Terminator for this block.
    ///
    /// N.B., this should generally ONLY be `None` during construction.
    /// Therefore, you should generally access it via the
    /// `terminator()` or `terminator_mut()` methods. The only
    /// exception is that certain passes, such as `simplify_cfg`, swap
    /// out the terminator temporarily with `None` while they continue
    /// to recurse over the set of basic blocks.
    pub terminator: Option<Terminator<'tcx>>,

    /// If true, this block lies on an unwind path. This is used
    /// during codegen where distinct kinds of basic blocks may be
    /// generated (particularly for MSVC cleanup). Unwind blocks must
    /// only branch to other unwind blocks.
    pub is_cleanup: bool,
}

/// Information about an assertion failure.
#[derive(Clone, TyEncodable, TyDecodable, Hash, HashStable, PartialEq, PartialOrd)]
pub enum AssertKind<O> {
    BoundsCheck { len: O, index: O },
    Overflow(BinOp, O, O),
    OverflowNeg(O),
    DivisionByZero(O),
    RemainderByZero(O),
    ResumedAfterReturn(GeneratorKind),
    ResumedAfterPanic(GeneratorKind),
}

#[derive(Clone, Debug, PartialEq, TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable)]
pub enum InlineAsmOperand<'tcx> {
    In {
        reg: InlineAsmRegOrRegClass,
        value: Operand<'tcx>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        place: Option<Place<'tcx>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_value: Operand<'tcx>,
        out_place: Option<Place<'tcx>>,
    },
    Const {
        value: Box<Constant<'tcx>>,
    },
    SymFn {
        value: Box<Constant<'tcx>>,
    },
    SymStatic {
        def_id: DefId,
    },
}

/// Type for MIR `Assert` terminator error messages.
pub type AssertMessage<'tcx> = AssertKind<Operand<'tcx>>;

// FIXME: Change `Successors` to `impl Iterator<Item = BasicBlock>`.
#[allow(rustc::pass_by_value)]
pub type Successors<'a> =
    iter::Chain<option::IntoIter<&'a BasicBlock>, slice::Iter<'a, BasicBlock>>;
pub type SuccessorsMut<'a> =
    iter::Chain<option::IntoIter<&'a mut BasicBlock>, slice::IterMut<'a, BasicBlock>>;

impl<'tcx> BasicBlockData<'tcx> {
    pub fn new(terminator: Option<Terminator<'tcx>>) -> BasicBlockData<'tcx> {
        BasicBlockData { statements: vec![], terminator, is_cleanup: false }
    }

    /// Accessor for terminator.
    ///
    /// Terminator may not be None after construction of the basic block is complete. This accessor
    /// provides a convenience way to reach the terminator.
    #[inline]
    pub fn terminator(&self) -> &Terminator<'tcx> {
        self.terminator.as_ref().expect("invalid terminator state")
    }

    #[inline]
    pub fn terminator_mut(&mut self) -> &mut Terminator<'tcx> {
        self.terminator.as_mut().expect("invalid terminator state")
    }

    pub fn retain_statements<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Statement<'_>) -> bool,
    {
        for s in &mut self.statements {
            if !f(s) {
                s.make_nop();
            }
        }
    }

    pub fn expand_statements<F, I>(&mut self, mut f: F)
    where
        F: FnMut(&mut Statement<'tcx>) -> Option<I>,
        I: iter::TrustedLen<Item = Statement<'tcx>>,
    {
        // Gather all the iterators we'll need to splice in, and their positions.
        let mut splices: Vec<(usize, I)> = vec![];
        let mut extra_stmts = 0;
        for (i, s) in self.statements.iter_mut().enumerate() {
            if let Some(mut new_stmts) = f(s) {
                if let Some(first) = new_stmts.next() {
                    // We can already store the first new statement.
                    *s = first;

                    // Save the other statements for optimized splicing.
                    let remaining = new_stmts.size_hint().0;
                    if remaining > 0 {
                        splices.push((i + 1 + extra_stmts, new_stmts));
                        extra_stmts += remaining;
                    }
                } else {
                    s.make_nop();
                }
            }
        }

        // Splice in the new statements, from the end of the block.
        // FIXME(eddyb) This could be more efficient with a "gap buffer"
        // where a range of elements ("gap") is left uninitialized, with
        // splicing adding new elements to the end of that gap and moving
        // existing elements from before the gap to the end of the gap.
        // For now, this is safe code, emulating a gap but initializing it.
        let mut gap = self.statements.len()..self.statements.len() + extra_stmts;
        self.statements.resize(
            gap.end,
            Statement { source_info: SourceInfo::outermost(DUMMY_SP), kind: StatementKind::Nop },
        );
        for (splice_start, new_stmts) in splices.into_iter().rev() {
            let splice_end = splice_start + new_stmts.size_hint().0;
            while gap.end > splice_end {
                gap.start -= 1;
                gap.end -= 1;
                self.statements.swap(gap.start, gap.end);
            }
            self.statements.splice(splice_start..splice_end, new_stmts);
            gap.end = splice_start;
        }
    }

    pub fn visitable(&self, index: usize) -> &dyn MirVisitable<'tcx> {
        if index < self.statements.len() { &self.statements[index] } else { &self.terminator }
    }
}

impl<O> AssertKind<O> {
    /// Getting a description does not require `O` to be printable, and does not
    /// require allocation.
    /// The caller is expected to handle `BoundsCheck` separately.
    pub fn description(&self) -> &'static str {
        use AssertKind::*;
        match self {
            Overflow(BinOp::Add, _, _) => "attempt to add with overflow",
            Overflow(BinOp::Sub, _, _) => "attempt to subtract with overflow",
            Overflow(BinOp::Mul, _, _) => "attempt to multiply with overflow",
            Overflow(BinOp::Div, _, _) => "attempt to divide with overflow",
            Overflow(BinOp::Rem, _, _) => "attempt to calculate the remainder with overflow",
            OverflowNeg(_) => "attempt to negate with overflow",
            Overflow(BinOp::Shr, _, _) => "attempt to shift right with overflow",
            Overflow(BinOp::Shl, _, _) => "attempt to shift left with overflow",
            Overflow(op, _, _) => bug!("{:?} cannot overflow", op),
            DivisionByZero(_) => "attempt to divide by zero",
            RemainderByZero(_) => "attempt to calculate the remainder with a divisor of zero",
            ResumedAfterReturn(GeneratorKind::Gen) => "generator resumed after completion",
            ResumedAfterReturn(GeneratorKind::Async(_)) => "`async fn` resumed after completion",
            ResumedAfterPanic(GeneratorKind::Gen) => "generator resumed after panicking",
            ResumedAfterPanic(GeneratorKind::Async(_)) => "`async fn` resumed after panicking",
            BoundsCheck { .. } => bug!("Unexpected AssertKind"),
        }
    }

    /// Format the message arguments for the `assert(cond, msg..)` terminator in MIR printing.
    pub fn fmt_assert_args<W: Write>(&self, f: &mut W) -> fmt::Result
    where
        O: Debug,
    {
        use AssertKind::*;
        match self {
            BoundsCheck { ref len, ref index } => write!(
                f,
                "\"index out of bounds: the length is {{}} but the index is {{}}\", {:?}, {:?}",
                len, index
            ),

            OverflowNeg(op) => {
                write!(f, "\"attempt to negate `{{}}`, which would overflow\", {:?}", op)
            }
            DivisionByZero(op) => write!(f, "\"attempt to divide `{{}}` by zero\", {:?}", op),
            RemainderByZero(op) => write!(
                f,
                "\"attempt to calculate the remainder of `{{}}` with a divisor of zero\", {:?}",
                op
            ),
            Overflow(BinOp::Add, l, r) => write!(
                f,
                "\"attempt to compute `{{}} + {{}}`, which would overflow\", {:?}, {:?}",
                l, r
            ),
            Overflow(BinOp::Sub, l, r) => write!(
                f,
                "\"attempt to compute `{{}} - {{}}`, which would overflow\", {:?}, {:?}",
                l, r
            ),
            Overflow(BinOp::Mul, l, r) => write!(
                f,
                "\"attempt to compute `{{}} * {{}}`, which would overflow\", {:?}, {:?}",
                l, r
            ),
            Overflow(BinOp::Div, l, r) => write!(
                f,
                "\"attempt to compute `{{}} / {{}}`, which would overflow\", {:?}, {:?}",
                l, r
            ),
            Overflow(BinOp::Rem, l, r) => write!(
                f,
                "\"attempt to compute the remainder of `{{}} % {{}}`, which would overflow\", {:?}, {:?}",
                l, r
            ),
            Overflow(BinOp::Shr, _, r) => {
                write!(f, "\"attempt to shift right by `{{}}`, which would overflow\", {:?}", r)
            }
            Overflow(BinOp::Shl, _, r) => {
                write!(f, "\"attempt to shift left by `{{}}`, which would overflow\", {:?}", r)
            }
            _ => write!(f, "\"{}\"", self.description()),
        }
    }
}

impl<O: fmt::Debug> fmt::Debug for AssertKind<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use AssertKind::*;
        match self {
            BoundsCheck { ref len, ref index } => write!(
                f,
                "index out of bounds: the length is {:?} but the index is {:?}",
                len, index
            ),
            OverflowNeg(op) => write!(f, "attempt to negate `{:#?}`, which would overflow", op),
            DivisionByZero(op) => write!(f, "attempt to divide `{:#?}` by zero", op),
            RemainderByZero(op) => write!(
                f,
                "attempt to calculate the remainder of `{:#?}` with a divisor of zero",
                op
            ),
            Overflow(BinOp::Add, l, r) => {
                write!(f, "attempt to compute `{:#?} + {:#?}`, which would overflow", l, r)
            }
            Overflow(BinOp::Sub, l, r) => {
                write!(f, "attempt to compute `{:#?} - {:#?}`, which would overflow", l, r)
            }
            Overflow(BinOp::Mul, l, r) => {
                write!(f, "attempt to compute `{:#?} * {:#?}`, which would overflow", l, r)
            }
            Overflow(BinOp::Div, l, r) => {
                write!(f, "attempt to compute `{:#?} / {:#?}`, which would overflow", l, r)
            }
            Overflow(BinOp::Rem, l, r) => write!(
                f,
                "attempt to compute the remainder of `{:#?} % {:#?}`, which would overflow",
                l, r
            ),
            Overflow(BinOp::Shr, _, r) => {
                write!(f, "attempt to shift right by `{:#?}`, which would overflow", r)
            }
            Overflow(BinOp::Shl, _, r) => {
                write!(f, "attempt to shift left by `{:#?}`, which would overflow", r)
            }
            _ => write!(f, "{}", self.description()),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Statements

#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub struct Statement<'tcx> {
    pub source_info: SourceInfo,
    pub kind: StatementKind<'tcx>,
}

// `Statement` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Statement<'_>, 32);

impl Statement<'_> {
    /// Changes a statement to a nop. This is both faster than deleting instructions and avoids
    /// invalidating statement indices in `Location`s.
    pub fn make_nop(&mut self) {
        self.kind = StatementKind::Nop
    }

    /// Changes a statement to a nop and returns the original statement.
    #[must_use = "If you don't need the statement, use `make_nop` instead"]
    pub fn replace_nop(&mut self) -> Self {
        Statement {
            source_info: self.source_info,
            kind: mem::replace(&mut self.kind, StatementKind::Nop),
        }
    }
}

/// The various kinds of statements that can appear in MIR.
///
/// Not all of these are allowed at every [`MirPhase`]. Check the documentation there to see which
/// ones you do not have to worry about. The MIR validator will generally enforce such restrictions,
/// causing an ICE if they are violated.
#[derive(Clone, Debug, PartialEq, TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable)]
pub enum StatementKind<'tcx> {
    /// Assign statements roughly correspond to an assignment in Rust proper (`x = ...`) except
    /// without the possibility of dropping the previous value (that must be done separately, if at
    /// all). The *exact* way this works is undecided. It probably does something like evaluating
    /// the LHS to a place and the RHS to a value, and then storing the value to the place. Various
    /// parts of this may do type specific things that are more complicated than simply copying
    /// bytes.
    ///
    /// **Needs clarification**: The implication of the above idea would be that assignment implies
    /// that the resulting value is initialized. I believe we could commit to this separately from
    /// committing to whatever part of the memory model we would need to decide on to make the above
    /// paragragh precise. Do we want to?
    ///
    /// Assignments in which the types of the place and rvalue differ are not well-formed.
    ///
    /// **Needs clarification**: Do we ever want to worry about non-free (in the body) lifetimes for
    /// the typing requirement in post drop-elaboration MIR? I think probably not - I'm not sure we
    /// could meaningfully require this anyway. How about free lifetimes? Is ignoring this
    /// interesting for optimizations? Do we want to allow such optimizations?
    ///
    /// **Needs clarification**: We currently require that the LHS place not overlap with any place
    /// read as part of computation of the RHS for some rvalues (generally those not producing
    /// primitives). This requirement is under discussion in [#68364]. As a part of this discussion,
    /// it is also unclear in what order the components are evaluated.
    ///
    /// [#68364]: https://github.com/rust-lang/rust/issues/68364
    ///
    /// See [`Rvalue`] documentation for details on each of those.
    Assign(Box<(Place<'tcx>, Rvalue<'tcx>)>),

    /// This represents all the reading that a pattern match may do (e.g., inspecting constants and
    /// discriminant values), and the kind of pattern it comes from. This is in order to adapt
    /// potential error messages to these specific patterns.
    ///
    /// Note that this also is emitted for regular `let` bindings to ensure that locals that are
    /// never accessed still get some sanity checks for, e.g., `let x: ! = ..;`
    ///
    /// When executed at runtime this is a nop.
    ///
    /// Disallowed after drop elaboration.
    FakeRead(Box<(FakeReadCause, Place<'tcx>)>),

    /// Write the discriminant for a variant to the enum Place.
    ///
    /// This is permitted for both generators and ADTs. This does not necessarily write to the
    /// entire place; instead, it writes to the minimum set of bytes as required by the layout for
    /// the type.
    SetDiscriminant { place: Box<Place<'tcx>>, variant_index: VariantIdx },

    /// Deinitializes the place.
    ///
    /// This writes `uninit` bytes to the entire place.
    Deinit(Box<Place<'tcx>>),

    /// `StorageLive` and `StorageDead` statements mark the live range of a local.
    ///
    /// Using a local before a `StorageLive` or after a `StorageDead` is not well-formed. These
    /// statements are not required. If the entire MIR body contains no `StorageLive`/`StorageDead`
    /// statements for a particular local, the local is always considered live.
    ///
    /// More precisely, the MIR validator currently does a `MaybeStorageLiveLocals` analysis to
    /// check validity of each use of a local. I believe this is equivalent to requiring for every
    /// use of a local, there exist at least one path from the root to that use that contains a
    /// `StorageLive` more recently than a `StorageDead`.
    ///
    /// **Needs clarification**: Is it permitted to have two `StorageLive`s without an intervening
    /// `StorageDead`? Two `StorageDead`s without an intervening `StorageLive`? LLVM says poison,
    /// yes. If the answer to any of these is "no," is breaking that rule UB or is it an error to
    /// have a path in the CFG that might do this?
    StorageLive(Local),

    /// See `StorageLive` above.
    StorageDead(Local),

    /// Retag references in the given place, ensuring they got fresh tags.
    ///
    /// This is part of the Stacked Borrows model. These statements are currently only interpreted
    /// by miri and only generated when `-Z mir-emit-retag` is passed. See
    /// <https://internals.rust-lang.org/t/stacked-borrows-an-aliasing-model-for-rust/8153/> for
    /// more details.
    ///
    /// For code that is not specific to stacked borrows, you should consider retags to read
    /// and modify the place in an opaque way.
    Retag(RetagKind, Box<Place<'tcx>>),

    /// Encodes a user's type ascription. These need to be preserved
    /// intact so that NLL can respect them. For example:
    ///
    ///     let a: T = y;
    ///
    /// The effect of this annotation is to relate the type `T_y` of the place `y`
    /// to the user-given type `T`. The effect depends on the specified variance:
    ///
    /// - `Covariant` -- requires that `T_y <: T`
    /// - `Contravariant` -- requires that `T_y :> T`
    /// - `Invariant` -- requires that `T_y == T`
    /// - `Bivariant` -- no effect
    ///
    /// When executed at runtime this is a nop.
    ///
    /// Disallowed after drop elaboration.
    AscribeUserType(Box<(Place<'tcx>, UserTypeProjection)>, ty::Variance),

    /// Marks the start of a "coverage region", injected with '-Cinstrument-coverage'. A
    /// `Coverage` statement carries metadata about the coverage region, used to inject a coverage
    /// map into the binary. If `Coverage::kind` is a `Counter`, the statement also generates
    /// executable code, to increment a counter variable at runtime, each time the code region is
    /// executed.
    Coverage(Box<Coverage>),

    /// Denotes a call to the intrinsic function `copy_nonoverlapping`.
    ///
    /// First, all three operands are evaluated. `src` and `dest` must each be a reference, pointer,
    /// or `Box` pointing to the same type `T`. `count` must evaluate to a `usize`. Then, `src` and
    /// `dest` are dereferenced, and `count * size_of::<T>()` bytes beginning with the first byte of
    /// the `src` place are copied to the continguous range of bytes beginning with the first byte
    /// of `dest`.
    ///
    /// **Needs clarification**: In what order are operands computed and dereferenced? It should
    /// probably match the order for assignment, but that is also undecided.
    ///
    /// **Needs clarification**: Is this typed or not, ie is there a typed load and store involved?
    /// I vaguely remember Ralf saying somewhere that he thought it should not be.
    CopyNonOverlapping(Box<CopyNonOverlapping<'tcx>>),

    /// No-op. Useful for deleting instructions without affecting statement indices.
    Nop,
}

impl<'tcx> StatementKind<'tcx> {
    pub fn as_assign_mut(&mut self) -> Option<&mut (Place<'tcx>, Rvalue<'tcx>)> {
        match self {
            StatementKind::Assign(x) => Some(x),
            _ => None,
        }
    }

    pub fn as_assign(&self) -> Option<&(Place<'tcx>, Rvalue<'tcx>)> {
        match self {
            StatementKind::Assign(x) => Some(x),
            _ => None,
        }
    }
}

/// Describes what kind of retag is to be performed.
#[derive(Copy, Clone, TyEncodable, TyDecodable, Debug, PartialEq, Eq, Hash, HashStable)]
pub enum RetagKind {
    /// The initial retag when entering a function.
    FnEntry,
    /// Retag preparing for a two-phase borrow.
    TwoPhase,
    /// Retagging raw pointers.
    Raw,
    /// A "normal" retag.
    Default,
}

/// The `FakeReadCause` describes the type of pattern why a FakeRead statement exists.
#[derive(Copy, Clone, TyEncodable, TyDecodable, Debug, Hash, HashStable, PartialEq)]
pub enum FakeReadCause {
    /// Inject a fake read of the borrowed input at the end of each guards
    /// code.
    ///
    /// This should ensure that you cannot change the variant for an enum while
    /// you are in the midst of matching on it.
    ForMatchGuard,

    /// `let x: !; match x {}` doesn't generate any read of x so we need to
    /// generate a read of x to check that it is initialized and safe.
    ///
    /// If a closure pattern matches a Place starting with an Upvar, then we introduce a
    /// FakeRead for that Place outside the closure, in such a case this option would be
    /// Some(closure_def_id).
    /// Otherwise, the value of the optional DefId will be None.
    ForMatchedPlace(Option<DefId>),

    /// A fake read of the RefWithinGuard version of a bind-by-value variable
    /// in a match guard to ensure that its value hasn't change by the time
    /// we create the OutsideGuard version.
    ForGuardBinding,

    /// Officially, the semantics of
    ///
    /// `let pattern = <expr>;`
    ///
    /// is that `<expr>` is evaluated into a temporary and then this temporary is
    /// into the pattern.
    ///
    /// However, if we see the simple pattern `let var = <expr>`, we optimize this to
    /// evaluate `<expr>` directly into the variable `var`. This is mostly unobservable,
    /// but in some cases it can affect the borrow checker, as in #53695.
    /// Therefore, we insert a "fake read" here to ensure that we get
    /// appropriate errors.
    ///
    /// If a closure pattern matches a Place starting with an Upvar, then we introduce a
    /// FakeRead for that Place outside the closure, in such a case this option would be
    /// Some(closure_def_id).
    /// Otherwise, the value of the optional DefId will be None.
    ForLet(Option<DefId>),

    /// If we have an index expression like
    ///
    /// (*x)[1][{ x = y; 4}]
    ///
    /// then the first bounds check is invalidated when we evaluate the second
    /// index expression. Thus we create a fake borrow of `x` across the second
    /// indexer, which will cause a borrow check error.
    ForIndex,
}

impl Debug for Statement<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use self::StatementKind::*;
        match self.kind {
            Assign(box (ref place, ref rv)) => write!(fmt, "{:?} = {:?}", place, rv),
            FakeRead(box (ref cause, ref place)) => {
                write!(fmt, "FakeRead({:?}, {:?})", cause, place)
            }
            Retag(ref kind, ref place) => write!(
                fmt,
                "Retag({}{:?})",
                match kind {
                    RetagKind::FnEntry => "[fn entry] ",
                    RetagKind::TwoPhase => "[2phase] ",
                    RetagKind::Raw => "[raw] ",
                    RetagKind::Default => "",
                },
                place,
            ),
            StorageLive(ref place) => write!(fmt, "StorageLive({:?})", place),
            StorageDead(ref place) => write!(fmt, "StorageDead({:?})", place),
            SetDiscriminant { ref place, variant_index } => {
                write!(fmt, "discriminant({:?}) = {:?}", place, variant_index)
            }
            Deinit(ref place) => write!(fmt, "Deinit({:?})", place),
            AscribeUserType(box (ref place, ref c_ty), ref variance) => {
                write!(fmt, "AscribeUserType({:?}, {:?}, {:?})", place, variance, c_ty)
            }
            Coverage(box self::Coverage { ref kind, code_region: Some(ref rgn) }) => {
                write!(fmt, "Coverage::{:?} for {:?}", kind, rgn)
            }
            Coverage(box ref coverage) => write!(fmt, "Coverage::{:?}", coverage.kind),
            CopyNonOverlapping(box crate::mir::CopyNonOverlapping {
                ref src,
                ref dst,
                ref count,
            }) => {
                write!(fmt, "copy_nonoverlapping(src={:?}, dst={:?}, count={:?})", src, dst, count)
            }
            Nop => write!(fmt, "nop"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable)]
pub struct Coverage {
    pub kind: CoverageKind,
    pub code_region: Option<CodeRegion>,
}

#[derive(Clone, Debug, PartialEq, TyEncodable, TyDecodable, Hash, HashStable, TypeFoldable)]
pub struct CopyNonOverlapping<'tcx> {
    pub src: Operand<'tcx>,
    pub dst: Operand<'tcx>,
    /// Number of elements to copy from src to dest, not bytes.
    pub count: Operand<'tcx>,
}

///////////////////////////////////////////////////////////////////////////
// Places

/// Places roughly correspond to a "location in memory." Places in MIR are the same mathematical
/// object as places in Rust. This of course means that what exactly they are is undecided and part
/// of the Rust memory model. However, they will likely contain at least the following pieces of
/// information in some form:
///
///  1. The address in memory that the place refers to.
///  2. The provenance with which the place is being accessed.
///  3. The type of the place and an optional variant index. See [`PlaceTy`][tcx::PlaceTy].
///  4. Optionally, some metadata. This exists if and only if the type of the place is not `Sized`.
///
/// We'll give a description below of how all pieces of the place except for the provenance are
/// calculated. We cannot give a description of the provenance, because that is part of the
/// undecided aliasing model - we only include it here at all to acknowledge its existence.
///
/// Each local naturally corresponds to the place `Place { local, projection: [] }`. This place has
/// the address of the local's allocation and the type of the local.
///
/// **Needs clarification:** Unsized locals seem to present a bit of an issue. Their allocation
/// can't actually be created on `StorageLive`, because it's unclear how big to make the allocation.
/// Furthermore, MIR produces assignments to unsized locals, although that is not permitted under
/// `#![feature(unsized_locals)]` in Rust. Besides just putting "unsized locals are special and
/// different" in a bunch of places, I (JakobDegen) don't know how to incorporate this behavior into
/// the current MIR semantics in a clean way - possibly this needs some design work first.
///
/// For places that are not locals, ie they have a non-empty list of projections, we define the
/// values as a function of the parent place, that is the place with its last [`ProjectionElem`]
/// stripped. The way this is computed of course depends on the kind of that last projection
/// element:
///
///  - [`Downcast`](ProjectionElem::Downcast): This projection sets the place's variant index to the
///    given one, and makes no other changes. A `Downcast` projection on a place with its variant
///    index already set is not well-formed.
///  - [`Field`](ProjectionElem::Field): `Field` projections take their parent place and create a
///    place referring to one of the fields of the type. The resulting address is the parent
///    address, plus the offset of the field. The type becomes the type of the field. If the parent
///    was unsized and so had metadata associated with it, then the metadata is retained if the
///    field is unsized and thrown out if it is sized.
///
///    These projections are only legal for tuples, ADTs, closures, and generators. If the ADT or
///    generator has more than one variant, the parent place's variant index must be set, indicating
///    which variant is being used. If it has just one variant, the variant index may or may not be
///    included - the single possible variant is inferred if it is not included.
///  - [`ConstantIndex`](ProjectionElem::ConstantIndex): Computes an offset in units of `T` into the
///    place as described in the documentation for the `ProjectionElem`. The resulting address is
///    the parent's address plus that offset, and the type is `T`. This is only legal if the parent
///    place has type `[T;  N]` or `[T]` (*not* `&[T]`). Since such a `T` is always sized, any
///    resulting metadata is thrown out.
///  - [`Subslice`](ProjectionElem::Subslice): This projection calculates an offset and a new
///    address in a similar manner as `ConstantIndex`. It is also only legal on `[T; N]` and `[T]`.
///    However, this yields a `Place` of type `[T]`, and additionally sets the metadata to be the
///    length of the subslice.
///  - [`Index`](ProjectionElem::Index): Like `ConstantIndex`, only legal on `[T; N]` or `[T]`.
///    However, `Index` additionally takes a local from which the value of the index is computed at
///    runtime. Computing the value of the index involves interpreting the `Local` as a
///    `Place { local, projection: [] }`, and then computing its value as if done via
///    [`Operand::Copy`]. The array/slice is then indexed with the resulting value. The local must
///    have type `usize`.
///  - [`Deref`](ProjectionElem::Deref): Derefs are the last type of projection, and the most
///    complicated. They are only legal on parent places that are references, pointers, or `Box`. A
///    `Deref` projection begins by loading a value from the parent place, as if by
///    [`Operand::Copy`]. It then dereferences the resulting pointer, creating a place of the
///    pointee's type. The resulting address is the address that was stored in the pointer. If the
///    pointee type is unsized, the pointer additionally stored the value of the metadata.
///
/// Computing a place may cause UB. One possibility is that the pointer used for a `Deref` may not
/// be suitably aligned. Another possibility is that the place is not in bounds, meaning it does not
/// point to an actual allocation.
///
/// However, if this is actually UB and when the UB kicks in is undecided. This is being discussed
/// in [UCG#319]. The options include that every place must obey those rules, that only some places
/// must obey them, or that places impose no rules of their own.
///
/// [UCG#319]: https://github.com/rust-lang/unsafe-code-guidelines/issues/319
///
/// Rust currently requires that every place obey those two rules. This is checked by MIRI and taken
/// advantage of by codegen (via `gep inbounds`). That is possibly subject to change.
#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, HashStable)]
pub struct Place<'tcx> {
    pub local: Local,

    /// projection out of a place (access a field, deref a pointer, etc)
    pub projection: &'tcx List<PlaceElem<'tcx>>,
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Place<'_>, 16);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(TyEncodable, TyDecodable, HashStable)]
pub enum ProjectionElem<V, T> {
    Deref,
    Field(Field, T),
    /// Index into a slice/array.
    ///
    /// Note that this does not also dereference, and so it does not exactly correspond to slice
    /// indexing in Rust. In other words, in the below Rust code:
    ///
    /// ```rust
    /// let x = &[1, 2, 3, 4];
    /// let i = 2;
    /// x[i];
    /// ```
    ///
    /// The `x[i]` is turned into a `Deref` followed by an `Index`, not just an `Index`. The same
    /// thing is true of the `ConstantIndex` and `Subslice` projections below.
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
        offset: u64,
        /// The thing being indexed must be at least this long. For arrays this
        /// is always the exact length.
        min_length: u64,
        /// Counting backwards from end? This is always false when indexing an
        /// array.
        from_end: bool,
    },

    /// These indices are generated by slice patterns.
    ///
    /// If `from_end` is true `slice[from..slice.len() - to]`.
    /// Otherwise `array[from..to]`.
    Subslice {
        from: u64,
        to: u64,
        /// Whether `to` counts from the start or end of the array/slice.
        /// For `PlaceElem`s this is `true` if and only if the base is a slice.
        /// For `ProjectionKind`, this can also be `true` for arrays.
        from_end: bool,
    },

    /// "Downcast" to a variant of an ADT. Currently, we only introduce
    /// this for ADTs with more than one variant. It may be better to
    /// just introduce it always, or always for enums.
    ///
    /// The included Symbol is the name of the variant, used for printing MIR.
    Downcast(Option<Symbol>, VariantIdx),
}

impl<V, T> ProjectionElem<V, T> {
    /// Returns `true` if the target of this projection may refer to a different region of memory
    /// than the base.
    fn is_indirect(&self) -> bool {
        match self {
            Self::Deref => true,

            Self::Field(_, _)
            | Self::Index(_)
            | Self::ConstantIndex { .. }
            | Self::Subslice { .. }
            | Self::Downcast(_, _) => false,
        }
    }

    /// Returns `true` if this is a `Downcast` projection with the given `VariantIdx`.
    pub fn is_downcast_to(&self, v: VariantIdx) -> bool {
        matches!(*self, Self::Downcast(_, x) if x == v)
    }

    /// Returns `true` if this is a `Field` projection with the given index.
    pub fn is_field_to(&self, f: Field) -> bool {
        matches!(*self, Self::Field(x, _) if x == f)
    }
}

/// Alias for projections as they appear in places, where the base is a place
/// and the index is a local.
pub type PlaceElem<'tcx> = ProjectionElem<Local, Ty<'tcx>>;

// This type is fairly frequently used, so we shouldn't unintentionally increase
// its size.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(PlaceElem<'_>, 24);

/// Alias for projections as they appear in `UserTypeProjection`, where we
/// need neither the `V` parameter for `Index` nor the `T` for `Field`.
pub type ProjectionKind = ProjectionElem<(), ()>;

rustc_index::newtype_index! {
    /// A [newtype'd][wrapper] index type in the MIR [control-flow graph][CFG]
    ///
    /// A field (e.g., `f` in `_1.f`) is one variant of [`ProjectionElem`]. Conceptually,
    /// rustc can identify that a field projection refers to either two different regions of memory
    /// or the same one between the base and the 'projection element'.
    /// Read more about projections in the [rustc-dev-guide][mir-datatypes]
    ///
    /// [wrapper]: https://rustc-dev-guide.rust-lang.org/appendix/glossary.html#newtype
    /// [CFG]: https://rustc-dev-guide.rust-lang.org/appendix/background.html#cfg
    /// [mir-datatypes]: https://rustc-dev-guide.rust-lang.org/mir/index.html#mir-data-types
    pub struct Field {
        derive [HashStable]
        DEBUG_FORMAT = "field[{}]"
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlaceRef<'tcx> {
    pub local: Local,
    pub projection: &'tcx [PlaceElem<'tcx>],
}

impl<'tcx> Place<'tcx> {
    // FIXME change this to a const fn by also making List::empty a const fn.
    pub fn return_place() -> Place<'tcx> {
        Place { local: RETURN_PLACE, projection: List::empty() }
    }

    /// Returns `true` if this `Place` contains a `Deref` projection.
    ///
    /// If `Place::is_indirect` returns false, the caller knows that the `Place` refers to the
    /// same region of memory as its base.
    pub fn is_indirect(&self) -> bool {
        self.projection.iter().any(|elem| elem.is_indirect())
    }

    /// Finds the innermost `Local` from this `Place`, *if* it is either a local itself or
    /// a single deref of a local.
    #[inline(always)]
    pub fn local_or_deref_local(&self) -> Option<Local> {
        self.as_ref().local_or_deref_local()
    }

    /// If this place represents a local variable like `_X` with no
    /// projections, return `Some(_X)`.
    #[inline(always)]
    pub fn as_local(&self) -> Option<Local> {
        self.as_ref().as_local()
    }

    #[inline]
    pub fn as_ref(&self) -> PlaceRef<'tcx> {
        PlaceRef { local: self.local, projection: &self.projection }
    }

    /// Iterate over the projections in evaluation order, i.e., the first element is the base with
    /// its projection and then subsequently more projections are added.
    /// As a concrete example, given the place a.b.c, this would yield:
    /// - (a, .b)
    /// - (a.b, .c)
    ///
    /// Given a place without projections, the iterator is empty.
    #[inline]
    pub fn iter_projections(
        self,
    ) -> impl Iterator<Item = (PlaceRef<'tcx>, PlaceElem<'tcx>)> + DoubleEndedIterator {
        self.projection.iter().enumerate().map(move |(i, proj)| {
            let base = PlaceRef { local: self.local, projection: &self.projection[..i] };
            (base, proj)
        })
    }

    /// Generates a new place by appending `more_projections` to the existing ones
    /// and interning the result.
    pub fn project_deeper(self, more_projections: &[PlaceElem<'tcx>], tcx: TyCtxt<'tcx>) -> Self {
        if more_projections.is_empty() {
            return self;
        }

        let mut v: Vec<PlaceElem<'tcx>>;

        let new_projections = if self.projection.is_empty() {
            more_projections
        } else {
            v = Vec::with_capacity(self.projection.len() + more_projections.len());
            v.extend(self.projection);
            v.extend(more_projections);
            &v
        };

        Place { local: self.local, projection: tcx.intern_place_elems(new_projections) }
    }
}

impl From<Local> for Place<'_> {
    fn from(local: Local) -> Self {
        Place { local, projection: List::empty() }
    }
}

impl<'tcx> PlaceRef<'tcx> {
    /// Finds the innermost `Local` from this `Place`, *if* it is either a local itself or
    /// a single deref of a local.
    pub fn local_or_deref_local(&self) -> Option<Local> {
        match *self {
            PlaceRef { local, projection: [] }
            | PlaceRef { local, projection: [ProjectionElem::Deref] } => Some(local),
            _ => None,
        }
    }

    /// If this place represents a local variable like `_X` with no
    /// projections, return `Some(_X)`.
    #[inline]
    pub fn as_local(&self) -> Option<Local> {
        match *self {
            PlaceRef { local, projection: [] } => Some(local),
            _ => None,
        }
    }

    #[inline]
    pub fn last_projection(&self) -> Option<(PlaceRef<'tcx>, PlaceElem<'tcx>)> {
        if let &[ref proj_base @ .., elem] = self.projection {
            Some((PlaceRef { local: self.local, projection: proj_base }, elem))
        } else {
            None
        }
    }
}

impl Debug for Place<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        for elem in self.projection.iter().rev() {
            match elem {
                ProjectionElem::Downcast(_, _) | ProjectionElem::Field(_, _) => {
                    write!(fmt, "(").unwrap();
                }
                ProjectionElem::Deref => {
                    write!(fmt, "(*").unwrap();
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {}
            }
        }

        write!(fmt, "{:?}", self.local)?;

        for elem in self.projection.iter() {
            match elem {
                ProjectionElem::Downcast(Some(name), _index) => {
                    write!(fmt, " as {})", name)?;
                }
                ProjectionElem::Downcast(None, index) => {
                    write!(fmt, " as variant#{:?})", index)?;
                }
                ProjectionElem::Deref => {
                    write!(fmt, ")")?;
                }
                ProjectionElem::Field(field, ty) => {
                    write!(fmt, ".{:?}: {:?})", field.index(), ty)?;
                }
                ProjectionElem::Index(ref index) => {
                    write!(fmt, "[{:?}]", index)?;
                }
                ProjectionElem::ConstantIndex { offset, min_length, from_end: false } => {
                    write!(fmt, "[{:?} of {:?}]", offset, min_length)?;
                }
                ProjectionElem::ConstantIndex { offset, min_length, from_end: true } => {
                    write!(fmt, "[-{:?} of {:?}]", offset, min_length)?;
                }
                ProjectionElem::Subslice { from, to, from_end: true } if to == 0 => {
                    write!(fmt, "[{:?}:]", from)?;
                }
                ProjectionElem::Subslice { from, to, from_end: true } if from == 0 => {
                    write!(fmt, "[:-{:?}]", to)?;
                }
                ProjectionElem::Subslice { from, to, from_end: true } => {
                    write!(fmt, "[{:?}:-{:?}]", from, to)?;
                }
                ProjectionElem::Subslice { from, to, from_end: false } => {
                    write!(fmt, "[{:?}..{:?}]", from, to)?;
                }
            }
        }

        Ok(())
    }
}

///////////////////////////////////////////////////////////////////////////
// Scopes

rustc_index::newtype_index! {
    pub struct SourceScope {
        derive [HashStable]
        DEBUG_FORMAT = "scope[{}]",
        const OUTERMOST_SOURCE_SCOPE = 0,
    }
}

impl SourceScope {
    /// Finds the original HirId this MIR item came from.
    /// This is necessary after MIR optimizations, as otherwise we get a HirId
    /// from the function that was inlined instead of the function call site.
    pub fn lint_root<'tcx>(
        self,
        source_scopes: &IndexVec<SourceScope, SourceScopeData<'tcx>>,
    ) -> Option<HirId> {
        let mut data = &source_scopes[self];
        // FIXME(oli-obk): we should be able to just walk the `inlined_parent_scope`, but it
        // does not work as I thought it would. Needs more investigation and documentation.
        while data.inlined.is_some() {
            trace!(?data);
            data = &source_scopes[data.parent_scope.unwrap()];
        }
        trace!(?data);
        match &data.local_data {
            ClearCrossCrate::Set(data) => Some(data.lint_root),
            ClearCrossCrate::Clear => None,
        }
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub struct SourceScopeData<'tcx> {
    pub span: Span,
    pub parent_scope: Option<SourceScope>,

    /// Whether this scope is the root of a scope tree of another body,
    /// inlined into this body by the MIR inliner.
    /// `ty::Instance` is the callee, and the `Span` is the call site.
    pub inlined: Option<(ty::Instance<'tcx>, Span)>,

    /// Nearest (transitive) parent scope (if any) which is inlined.
    /// This is an optimization over walking up `parent_scope`
    /// until a scope with `inlined: Some(...)` is found.
    pub inlined_parent_scope: Option<SourceScope>,

    /// Crate-local information for this source scope, that can't (and
    /// needn't) be tracked across crates.
    pub local_data: ClearCrossCrate<SourceScopeLocalData>,
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct SourceScopeLocalData {
    /// An `HirId` with lint levels equivalent to this scope's lint levels.
    pub lint_root: hir::HirId,
    /// The unsafe block that contains this node.
    pub safety: Safety,
}

///////////////////////////////////////////////////////////////////////////
// Operands

/// An operand in MIR represents a "value" in Rust, the definition of which is undecided and part of
/// the memory model. One proposal for a definition of values can be found [on UCG][value-def].
///
/// [value-def]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/value-domain.md
///
/// The most common way to create values is via loading a place. Loading a place is an operation
/// which reads the memory of the place and converts it to a value. This is a fundamentally *typed*
/// operation. The nature of the value produced depends on the type of the conversion. Furthermore,
/// there may be other effects: if the type has a validity constraint loading the place might be UB
/// if the validity constraint is not met.
///
/// **Needs clarification:** Ralf proposes that loading a place not have side-effects.
/// This is what is implemented in miri today. Are these the semantics we want for MIR? Is this
/// something we can even decide without knowing more about Rust's memory model?
///
/// **Needs clarifiation:** Is loading a place that has its variant index set well-formed? Miri
/// currently implements it, but it seems like this may be something to check against in the
/// validator.
#[derive(Clone, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum Operand<'tcx> {
    /// Creates a value by loading the given place.
    ///
    /// Before drop elaboration, the type of the place must be `Copy`. After drop elaboration there
    /// is no such requirement.
    Copy(Place<'tcx>),

    /// Creates a value by performing loading the place, just like the `Copy` operand.
    ///
    /// This *may* additionally overwrite the place with `uninit` bytes, depending on how we decide
    /// in [UCG#188]. You should not emit MIR that may attempt a subsequent second load of this
    /// place without first re-initializing it.
    ///
    /// [UCG#188]: https://github.com/rust-lang/unsafe-code-guidelines/issues/188
    Move(Place<'tcx>),

    /// Constants are already semantically values, and remain unchanged.
    Constant(Box<Constant<'tcx>>),
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Operand<'_>, 24);

impl<'tcx> Debug for Operand<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use self::Operand::*;
        match *self {
            Constant(ref a) => write!(fmt, "{:?}", a),
            Copy(ref place) => write!(fmt, "{:?}", place),
            Move(ref place) => write!(fmt, "move {:?}", place),
        }
    }
}

impl<'tcx> Operand<'tcx> {
    /// Convenience helper to make a constant that refers to the fn
    /// with given `DefId` and substs. Since this is used to synthesize
    /// MIR, assumes `user_ty` is None.
    pub fn function_handle(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        span: Span,
    ) -> Self {
        let ty = tcx.type_of(def_id).subst(tcx, substs);
        Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: ConstantKind::Ty(ty::Const::zero_sized(tcx, ty)),
        }))
    }

    pub fn is_move(&self) -> bool {
        matches!(self, Operand::Move(..))
    }

    /// Convenience helper to make a literal-like constant from a given scalar value.
    /// Since this is used to synthesize MIR, assumes `user_ty` is None.
    pub fn const_from_scalar(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        val: Scalar,
        span: Span,
    ) -> Operand<'tcx> {
        debug_assert!({
            let param_env_and_ty = ty::ParamEnv::empty().and(ty);
            let type_size = tcx
                .layout_of(param_env_and_ty)
                .unwrap_or_else(|e| panic!("could not compute layout for {:?}: {:?}", ty, e))
                .size;
            let scalar_size = match val {
                Scalar::Int(int) => int.size(),
                _ => panic!("Invalid scalar type {:?}", val),
            };
            scalar_size == type_size
        });
        Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: ConstantKind::Val(ConstValue::Scalar(val), ty),
        }))
    }

    pub fn to_copy(&self) -> Self {
        match *self {
            Operand::Copy(_) | Operand::Constant(_) => self.clone(),
            Operand::Move(place) => Operand::Copy(place),
        }
    }

    /// Returns the `Place` that is the target of this `Operand`, or `None` if this `Operand` is a
    /// constant.
    pub fn place(&self) -> Option<Place<'tcx>> {
        match self {
            Operand::Copy(place) | Operand::Move(place) => Some(*place),
            Operand::Constant(_) => None,
        }
    }

    /// Returns the `Constant` that is the target of this `Operand`, or `None` if this `Operand` is a
    /// place.
    pub fn constant(&self) -> Option<&Constant<'tcx>> {
        match self {
            Operand::Constant(x) => Some(&**x),
            Operand::Copy(_) | Operand::Move(_) => None,
        }
    }

    /// Gets the `ty::FnDef` from an operand if it's a constant function item.
    ///
    /// While this is unlikely in general, it's the normal case of what you'll
    /// find as the `func` in a [`TerminatorKind::Call`].
    pub fn const_fn_def(&self) -> Option<(DefId, SubstsRef<'tcx>)> {
        let const_ty = self.constant()?.literal.ty();
        if let ty::FnDef(def_id, substs) = *const_ty.kind() { Some((def_id, substs)) } else { None }
    }
}

///////////////////////////////////////////////////////////////////////////
/// Rvalues

#[derive(Clone, TyEncodable, TyDecodable, Hash, HashStable, PartialEq)]
/// The various kinds of rvalues that can appear in MIR.
///
/// Not all of these are allowed at every [`MirPhase`] - when this is the case, it's stated below.
///
/// Computing any rvalue begins by evaluating the places and operands in some order (**Needs
/// clarification**: Which order?). These are then used to produce a "value" - the same kind of
/// value that an [`Operand`] produces.
pub enum Rvalue<'tcx> {
    /// Yields the operand unchanged
    Use(Operand<'tcx>),

    /// Creates an array where each element is the value of the operand.
    ///
    /// This is the cause of a bug in the case where the repetition count is zero because the value
    /// is not dropped, see [#74836].
    ///
    /// Corresponds to source code like `[x; 32]`.
    ///
    /// [#74836]: https://github.com/rust-lang/rust/issues/74836
    Repeat(Operand<'tcx>, ty::Const<'tcx>),

    /// Creates a reference of the indicated kind to the place.
    ///
    /// There is not much to document here, because besides the obvious parts the semantics of this
    /// are essentially entirely a part of the aliasing model. There are many UCG issues discussing
    /// exactly what the behavior of this operation should be.
    ///
    /// `Shallow` borrows are disallowed after drop lowering.
    Ref(Region<'tcx>, BorrowKind, Place<'tcx>),

    /// Creates a pointer/reference to the given thread local.
    ///
    /// The yielded type is a `*mut T` if the static is mutable, otherwise if the static is extern a
    /// `*const T`, and if neither of those apply a `&T`.
    ///
    /// **Note:** This is a runtime operation that actually executes code and is in this sense more
    /// like a function call. Also, eliminating dead stores of this rvalue causes `fn main() {}` to
    /// SIGILL for some reason that I (JakobDegen) never got a chance to look into.
    ///
    /// **Needs clarification**: Are there weird additional semantics here related to the runtime
    /// nature of this operation?
    ThreadLocalRef(DefId),

    /// Creates a pointer with the indicated mutability to the place.
    ///
    /// This is generated by pointer casts like `&v as *const _` or raw address of expressions like
    /// `&raw v` or `addr_of!(v)`.
    ///
    /// Like with references, the semantics of this operation are heavily dependent on the aliasing
    /// model.
    AddressOf(Mutability, Place<'tcx>),

    /// Yields the length of the place, as a `usize`.
    ///
    /// If the type of the place is an array, this is the array length. For slices (`[T]`, not
    /// `&[T]`) this accesses the place's metadata to determine the length. This rvalue is
    /// ill-formed for places of other types.
    Len(Place<'tcx>),

    /// Performs essentially all of the casts that can be performed via `as`.
    ///
    /// This allows for casts from/to a variety of types.
    ///
    /// **FIXME**: Document exactly which `CastKind`s allow which types of casts. Figure out why
    /// `ArrayToPointer` and `MutToConstPointer` are special.
    Cast(CastKind, Operand<'tcx>, Ty<'tcx>),

    /// * `Offset` has the same semantics as [`offset`](pointer::offset), except that the second
    ///   parameter may be a `usize` as well.
    /// * The comparison operations accept `bool`s, `char`s, signed or unsigned integers, floats,
    ///   raw pointers, or function pointers and return a `bool`. The types of the operands must be
    ///   matching, up to the usual caveat of the lifetimes in function pointers.
    /// * Left and right shift operations accept signed or unsigned integers not necessarily of the
    ///   same type and return a value of the same type as their LHS. Like in Rust, the RHS is
    ///   truncated as needed.
    /// * The `Bit*` operations accept signed integers, unsigned integers, or bools with matching
    ///   types and return a value of that type.
    /// * The remaining operations accept signed integers, unsigned integers, or floats with
    ///   matching types and return a value of that type.
    BinaryOp(BinOp, Box<(Operand<'tcx>, Operand<'tcx>)>),

    /// Same as `BinaryOp`, but yields `(T, bool)` instead of `T`. In addition to performing the
    /// same computation as the matching `BinaryOp`, checks if the infinite precison result would be
    /// unequal to the actual result and sets the `bool` if this is the case.
    ///
    /// This only supports addition, subtraction, multiplication, and shift operations on integers.
    CheckedBinaryOp(BinOp, Box<(Operand<'tcx>, Operand<'tcx>)>),

    /// Computes a value as described by the operation.
    NullaryOp(NullOp, Ty<'tcx>),

    /// Exactly like `BinaryOp`, but less operands.
    ///
    /// Also does two's-complement arithmetic. Negation requires a signed integer or a float;
    /// bitwise not requires a signed integer, unsigned integer, or bool. Both operation kinds
    /// return a value with the same type as their operand.
    UnaryOp(UnOp, Operand<'tcx>),

    /// Computes the discriminant of the place, returning it as an integer of type
    /// [`discriminant_ty`].
    ///
    /// The validity requirements for the underlying value are undecided for this rvalue, see
    /// [#91095]. Note too that the value of the discriminant is not the same thing as the
    /// variant index; use [`discriminant_for_variant`] to convert.
    ///
    /// For types defined in the source code as enums, this is well behaved. This is also well
    /// formed for other types, but yields no particular value - there is no reason it couldn't be
    /// defined to yield eg zero though.
    ///
    /// [`discriminant_ty`]: crate::ty::Ty::discriminant_ty
    /// [#91095]: https://github.com/rust-lang/rust/issues/91095
    /// [`discriminant_for_variant`]: crate::ty::Ty::discriminant_for_variant
    Discriminant(Place<'tcx>),

    /// Creates an aggregate value, like a tuple or struct.
    ///
    /// This is needed because dataflow analysis needs to distinguish
    /// `dest = Foo { x: ..., y: ... }` from `dest.x = ...; dest.y = ...;` in the case that `Foo`
    /// has a destructor.
    ///
    /// Disallowed after deaggregation for all aggregate kinds except `Array` and `Generator`. After
    /// generator lowering, `Generator` aggregate kinds are disallowed too.
    Aggregate(Box<AggregateKind<'tcx>>, Vec<Operand<'tcx>>),

    /// Transmutes a `*mut u8` into shallow-initialized `Box<T>`.
    ///
    /// This is different from a normal transmute because dataflow analysis will treat the box as
    /// initialized but its content as uninitialized. Like other pointer casts, this in general
    /// affects alias analysis.
    ShallowInitBox(Operand<'tcx>, Ty<'tcx>),
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Rvalue<'_>, 40);

#[derive(Clone, Copy, Debug, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum CastKind {
    Misc,
    Pointer(PointerCast),
}

#[derive(Clone, Debug, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum AggregateKind<'tcx> {
    /// The type is of the element
    Array(Ty<'tcx>),
    Tuple,

    /// The second field is the variant index. It's equal to 0 for struct
    /// and union expressions. The fourth field is
    /// active field number and is present only for union expressions
    /// -- e.g., for a union expression `SomeUnion { c: .. }`, the
    /// active field index would identity the field `c`
    Adt(DefId, VariantIdx, SubstsRef<'tcx>, Option<UserTypeAnnotationIndex>, Option<usize>),

    Closure(DefId, SubstsRef<'tcx>),
    Generator(DefId, SubstsRef<'tcx>, hir::Movability),
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(AggregateKind<'_>, 48);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum BinOp {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    ///
    /// Division by zero is UB, because the compiler should have inserted checks
    /// prior to this.
    Div,
    /// The `%` operator (modulus)
    ///
    /// Using zero as the modulus (second operand) is UB, because the compiler
    /// should have inserted checks prior to this.
    Rem,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    ///
    /// The offset is truncated to the size of the first operand before shifting.
    Shl,
    /// The `>>` operator (shift right)
    ///
    /// The offset is truncated to the size of the first operand before shifting.
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
    /// The `ptr.offset` operator
    Offset,
}

impl BinOp {
    pub fn is_checkable(self) -> bool {
        use self::BinOp::*;
        matches!(self, Add | Sub | Mul | Shl | Shr)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum NullOp {
    /// Returns the size of a value of that type
    SizeOf,
    /// Returns the minimum alignment of a type
    AlignOf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum UnOp {
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

impl<'tcx> Debug for Rvalue<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        use self::Rvalue::*;

        match *self {
            Use(ref place) => write!(fmt, "{:?}", place),
            Repeat(ref a, b) => {
                write!(fmt, "[{:?}; ", a)?;
                pretty_print_const(b, fmt, false)?;
                write!(fmt, "]")
            }
            Len(ref a) => write!(fmt, "Len({:?})", a),
            Cast(ref kind, ref place, ref ty) => {
                write!(fmt, "{:?} as {:?} ({:?})", place, ty, kind)
            }
            BinaryOp(ref op, box (ref a, ref b)) => write!(fmt, "{:?}({:?}, {:?})", op, a, b),
            CheckedBinaryOp(ref op, box (ref a, ref b)) => {
                write!(fmt, "Checked{:?}({:?}, {:?})", op, a, b)
            }
            UnaryOp(ref op, ref a) => write!(fmt, "{:?}({:?})", op, a),
            Discriminant(ref place) => write!(fmt, "discriminant({:?})", place),
            NullaryOp(ref op, ref t) => write!(fmt, "{:?}({:?})", op, t),
            ThreadLocalRef(did) => ty::tls::with(|tcx| {
                let muta = tcx.static_mutability(did).unwrap().prefix_str();
                write!(fmt, "&/*tls*/ {}{}", muta, tcx.def_path_str(did))
            }),
            Ref(region, borrow_kind, ref place) => {
                let kind_str = match borrow_kind {
                    BorrowKind::Shared => "",
                    BorrowKind::Shallow => "shallow ",
                    BorrowKind::Mut { .. } | BorrowKind::Unique => "mut ",
                };

                // When printing regions, add trailing space if necessary.
                let print_region = ty::tls::with(|tcx| {
                    tcx.sess.verbose() || tcx.sess.opts.debugging_opts.identify_regions
                });
                let region = if print_region {
                    let mut region = region.to_string();
                    if !region.is_empty() {
                        region.push(' ');
                    }
                    region
                } else {
                    // Do not even print 'static
                    String::new()
                };
                write!(fmt, "&{}{}{:?}", region, kind_str, place)
            }

            AddressOf(mutability, ref place) => {
                let kind_str = match mutability {
                    Mutability::Mut => "mut",
                    Mutability::Not => "const",
                };

                write!(fmt, "&raw {} {:?}", kind_str, place)
            }

            Aggregate(ref kind, ref places) => {
                let fmt_tuple = |fmt: &mut Formatter<'_>, name: &str| {
                    let mut tuple_fmt = fmt.debug_tuple(name);
                    for place in places {
                        tuple_fmt.field(place);
                    }
                    tuple_fmt.finish()
                };

                match **kind {
                    AggregateKind::Array(_) => write!(fmt, "{:?}", places),

                    AggregateKind::Tuple => {
                        if places.is_empty() {
                            write!(fmt, "()")
                        } else {
                            fmt_tuple(fmt, "")
                        }
                    }

                    AggregateKind::Adt(adt_did, variant, substs, _user_ty, _) => {
                        ty::tls::with(|tcx| {
                            let variant_def = &tcx.adt_def(adt_did).variant(variant);
                            let substs = tcx.lift(substs).expect("could not lift for printing");
                            let name = FmtPrinter::new(tcx, Namespace::ValueNS)
                                .print_def_path(variant_def.def_id, substs)?
                                .into_buffer();

                            match variant_def.ctor_kind {
                                CtorKind::Const => fmt.write_str(&name),
                                CtorKind::Fn => fmt_tuple(fmt, &name),
                                CtorKind::Fictive => {
                                    let mut struct_fmt = fmt.debug_struct(&name);
                                    for (field, place) in iter::zip(&variant_def.fields, places) {
                                        struct_fmt.field(field.name.as_str(), place);
                                    }
                                    struct_fmt.finish()
                                }
                            }
                        })
                    }

                    AggregateKind::Closure(def_id, substs) => ty::tls::with(|tcx| {
                        if let Some(def_id) = def_id.as_local() {
                            let name = if tcx.sess.opts.debugging_opts.span_free_formats {
                                let substs = tcx.lift(substs).unwrap();
                                format!(
                                    "[closure@{}]",
                                    tcx.def_path_str_with_substs(def_id.to_def_id(), substs),
                                )
                            } else {
                                let span = tcx.def_span(def_id);
                                format!(
                                    "[closure@{}]",
                                    tcx.sess.source_map().span_to_diagnostic_string(span)
                                )
                            };
                            let mut struct_fmt = fmt.debug_struct(&name);

                            // FIXME(project-rfc-2229#48): This should be a list of capture names/places
                            if let Some(upvars) = tcx.upvars_mentioned(def_id) {
                                for (&var_id, place) in iter::zip(upvars.keys(), places) {
                                    let var_name = tcx.hir().name(var_id);
                                    struct_fmt.field(var_name.as_str(), place);
                                }
                            }

                            struct_fmt.finish()
                        } else {
                            write!(fmt, "[closure]")
                        }
                    }),

                    AggregateKind::Generator(def_id, _, _) => ty::tls::with(|tcx| {
                        if let Some(def_id) = def_id.as_local() {
                            let name = format!("[generator@{:?}]", tcx.def_span(def_id));
                            let mut struct_fmt = fmt.debug_struct(&name);

                            // FIXME(project-rfc-2229#48): This should be a list of capture names/places
                            if let Some(upvars) = tcx.upvars_mentioned(def_id) {
                                for (&var_id, place) in iter::zip(upvars.keys(), places) {
                                    let var_name = tcx.hir().name(var_id);
                                    struct_fmt.field(var_name.as_str(), place);
                                }
                            }

                            struct_fmt.finish()
                        } else {
                            write!(fmt, "[generator]")
                        }
                    }),
                }
            }

            ShallowInitBox(ref place, ref ty) => {
                write!(fmt, "ShallowInitBox({:?}, {:?})", place, ty)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
/// Constants
///
/// Two constants are equal if they are the same constant. Note that
/// this does not necessarily mean that they are `==` in Rust. In
/// particular, one must be wary of `NaN`!

#[derive(Clone, Copy, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
pub struct Constant<'tcx> {
    pub span: Span,

    /// Optional user-given type: for something like
    /// `collect::<Vec<_>>`, this would be present and would
    /// indicate that `Vec<_>` was explicitly specified.
    ///
    /// Needed for NLL to impose user-given type constraints.
    pub user_ty: Option<UserTypeAnnotationIndex>,

    pub literal: ConstantKind<'tcx>,
}

#[derive(Clone, Copy, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable, Debug)]
#[derive(Lift)]
pub enum ConstantKind<'tcx> {
    /// This constant came from the type system
    Ty(ty::Const<'tcx>),
    /// This constant cannot go back into the type system, as it represents
    /// something the type system cannot handle (e.g. pointers).
    Val(interpret::ConstValue<'tcx>, Ty<'tcx>),
}

impl<'tcx> Constant<'tcx> {
    pub fn check_static_ptr(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.literal.try_to_scalar() {
            Some(Scalar::Ptr(ptr, _size)) => match tcx.global_alloc(ptr.provenance) {
                GlobalAlloc::Static(def_id) => {
                    assert!(!tcx.is_thread_local_static(def_id));
                    Some(def_id)
                }
                _ => None,
            },
            _ => None,
        }
    }
    #[inline]
    pub fn ty(&self) -> Ty<'tcx> {
        self.literal.ty()
    }
}

impl<'tcx> From<ty::Const<'tcx>> for ConstantKind<'tcx> {
    #[inline]
    fn from(ct: ty::Const<'tcx>) -> Self {
        match ct.val() {
            ty::ConstKind::Value(cv) => {
                // FIXME Once valtrees are introduced we need to convert those
                // into `ConstValue` instances here
                Self::Val(cv, ct.ty())
            }
            _ => Self::Ty(ct),
        }
    }
}

impl<'tcx> ConstantKind<'tcx> {
    /// Returns `None` if the constant is not trivially safe for use in the type system.
    pub fn const_for_ty(&self) -> Option<ty::Const<'tcx>> {
        match self {
            ConstantKind::Ty(c) => Some(*c),
            ConstantKind::Val(..) => None,
        }
    }

    pub fn ty(&self) -> Ty<'tcx> {
        match self {
            ConstantKind::Ty(c) => c.ty(),
            ConstantKind::Val(_, ty) => *ty,
        }
    }

    pub fn try_val(&self) -> Option<ConstValue<'tcx>> {
        match self {
            ConstantKind::Ty(c) => match c.val() {
                ty::ConstKind::Value(v) => Some(v),
                _ => None,
            },
            ConstantKind::Val(v, _) => Some(*v),
        }
    }

    #[inline]
    pub fn try_to_value(self) -> Option<interpret::ConstValue<'tcx>> {
        match self {
            ConstantKind::Ty(c) => c.val().try_to_value(),
            ConstantKind::Val(val, _) => Some(val),
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar> {
        self.try_to_value()?.try_to_scalar()
    }

    #[inline]
    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        Some(self.try_to_value()?.try_to_scalar()?.assert_int())
    }

    #[inline]
    pub fn try_to_bits(self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    #[inline]
    pub fn eval(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Self {
        match self {
            Self::Ty(c) => {
                // FIXME Need to use a different evaluation function that directly returns a `ConstValue`
                // if evaluation succeeds and does not create a ValTree first
                if let Some(val) = c.val().try_eval(tcx, param_env) {
                    match val {
                        Ok(val) => Self::Val(val, c.ty()),
                        Err(_) => Self::Ty(tcx.const_error(self.ty())),
                    }
                } else {
                    self
                }
            }
            Self::Val(_, _) => self,
        }
    }

    /// Panics if the value cannot be evaluated or doesn't contain a valid integer of the given type.
    #[inline]
    pub fn eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>, ty: Ty<'tcx>) -> u128 {
        self.try_eval_bits(tcx, param_env, ty)
            .unwrap_or_else(|| bug!("expected bits of {:#?}, got {:#?}", ty, self))
    }

    #[inline]
    pub fn try_eval_bits(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        match self {
            Self::Ty(ct) => ct.try_eval_bits(tcx, param_env, ty),
            Self::Val(val, t) => {
                assert_eq!(*t, ty);
                let size =
                    tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
                val.try_to_bits(size)
            }
        }
    }

    #[inline]
    pub fn try_eval_bool(&self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Option<bool> {
        match self {
            Self::Ty(ct) => ct.try_eval_bool(tcx, param_env),
            Self::Val(val, _) => val.try_to_bool(),
        }
    }

    #[inline]
    pub fn try_eval_usize(&self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Option<u64> {
        match self {
            Self::Ty(ct) => ct.try_eval_usize(tcx, param_env),
            Self::Val(val, _) => val.try_to_machine_usize(tcx),
        }
    }

    pub fn from_bits(
        tcx: TyCtxt<'tcx>,
        bits: u128,
        param_env_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Self {
        let size = tcx
            .layout_of(param_env_ty)
            .unwrap_or_else(|e| {
                bug!("could not compute layout for {:?}: {:?}", param_env_ty.value, e)
            })
            .size;
        let cv = ConstValue::Scalar(Scalar::from_uint(bits, size));

        Self::Val(cv, param_env_ty.value)
    }

    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        let cv = ConstValue::from_bool(v);
        Self::Val(cv, tcx.types.bool)
    }

    pub fn zero_sized(ty: Ty<'tcx>) -> Self {
        let cv = ConstValue::Scalar(Scalar::ZST);
        Self::Val(cv, ty)
    }

    pub fn from_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        let ty = tcx.types.usize;
        Self::from_bits(tcx, n as u128, ty::ParamEnv::empty().and(ty))
    }

    /// Literals are converted to `ConstantKindVal`, const generic parameters are eagerly
    /// converted to a constant, everything else becomes `Unevaluated`.
    pub fn from_anon_const(
        tcx: TyCtxt<'tcx>,
        def_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        Self::from_opt_const_arg_anon_const(tcx, ty::WithOptConstParam::unknown(def_id), param_env)
    }

    #[instrument(skip(tcx), level = "debug")]
    fn from_opt_const_arg_anon_const(
        tcx: TyCtxt<'tcx>,
        def: ty::WithOptConstParam<LocalDefId>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let body_id = match tcx.hir().get_by_def_id(def.did) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => span_bug!(
                tcx.def_span(def.did.to_def_id()),
                "from_anon_const can only process anonymous constants"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        // Unwrap a block, so that e.g. `{ P }` is recognised as a parameter. Const arguments
        // currently have to be wrapped in curly brackets, so it's necessary to special-case.
        let expr = match &expr.kind {
            hir::ExprKind::Block(block, _) if block.stmts.is_empty() && block.expr.is_some() => {
                block.expr.as_ref().unwrap()
            }
            _ => expr,
        };

        let ty = tcx.type_of(def.def_id_for_type_of());

        // FIXME(const_generics): We currently have to special case parameters because `min_const_generics`
        // does not provide the parents generics to anonymous constants. We still allow generic const
        // parameters by themselves however, e.g. `N`.  These constants would cause an ICE if we were to
        // ever try to substitute the generic parameters in their bodies.
        //
        // While this doesn't happen as these constants are always used as `ty::ConstKind::Param`, it does
        // cause issues if we were to remove that special-case and try to evaluate the constant instead.
        use hir::{def::DefKind::ConstParam, def::Res, ExprKind, Path, QPath};
        match expr.kind {
            ExprKind::Path(QPath::Resolved(_, &Path { res: Res::Def(ConstParam, def_id), .. })) => {
                // Find the name and index of the const parameter by indexing the generics of
                // the parent item and construct a `ParamConst`.
                let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
                let item_id = tcx.hir().get_parent_node(hir_id);
                let item_def_id = tcx.hir().local_def_id(item_id);
                let generics = tcx.generics_of(item_def_id.to_def_id());
                let index = generics.param_def_id_to_index[&def_id];
                let name = tcx.hir().name(hir_id);
                let ty_const = tcx.mk_const(ty::ConstS {
                    val: ty::ConstKind::Param(ty::ParamConst::new(index, name)),
                    ty,
                });

                return Self::Ty(ty_const);
            }
            _ => {}
        }

        let hir_id = tcx.hir().local_def_id_to_hir_id(def.did);
        let parent_substs = if let Some(parent_hir_id) = tcx.hir().find_parent_node(hir_id) {
            if let Some(parent_did) = tcx.hir().opt_local_def_id(parent_hir_id) {
                InternalSubsts::identity_for_item(tcx, parent_did.to_def_id())
            } else {
                tcx.mk_substs(Vec::<GenericArg<'tcx>>::new().into_iter())
            }
        } else {
            tcx.mk_substs(Vec::<GenericArg<'tcx>>::new().into_iter())
        };
        debug!(?parent_substs);

        let did = def.did.to_def_id();
        let child_substs = InternalSubsts::identity_for_item(tcx, did);
        let substs = tcx.mk_substs(parent_substs.into_iter().chain(child_substs.into_iter()));
        debug!(?substs);

        let hir_id = tcx.hir().local_def_id_to_hir_id(def.did);
        let span = tcx.hir().span(hir_id);
        let uneval = ty::Unevaluated::new(def.to_global(), substs);
        debug!(?span, ?param_env);

        match tcx.const_eval_resolve(param_env, uneval, Some(span)) {
            Ok(val) => Self::Val(val, ty),
            Err(_) => {
                // Error was handled in `const_eval_resolve`. Here we just create a
                // new unevaluated const and error hard later in codegen
                let ty_const = tcx.mk_const(ty::ConstS {
                    val: ty::ConstKind::Unevaluated(ty::Unevaluated {
                        def: def.to_global(),
                        substs: InternalSubsts::identity_for_item(tcx, def.did.to_def_id()),
                        promoted: None,
                    }),
                    ty,
                });

                Self::Ty(ty_const)
            }
        }
    }
}

/// A collection of projections into user types.
///
/// They are projections because a binding can occur a part of a
/// parent pattern that has been ascribed a type.
///
/// Its a collection because there can be multiple type ascriptions on
/// the path from the root of the pattern down to the binding itself.
///
/// An example:
///
/// ```rust
/// struct S<'a>((i32, &'a str), String);
/// let S((_, w): (i32, &'static str), _): S = ...;
/// //    ------  ^^^^^^^^^^^^^^^^^^^ (1)
/// //  ---------------------------------  ^ (2)
/// ```
///
/// The highlights labelled `(1)` show the subpattern `(_, w)` being
/// ascribed the type `(i32, &'static str)`.
///
/// The highlights labelled `(2)` show the whole pattern being
/// ascribed the type `S`.
///
/// In this example, when we descend to `w`, we will have built up the
/// following two projected types:
///
///   * base: `S`,                   projection: `(base.0).1`
///   * base: `(i32, &'static str)`, projection: `base.1`
///
/// The first will lead to the constraint `w: &'1 str` (for some
/// inferred region `'1`). The second will lead to the constraint `w:
/// &'static str`.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable)]
pub struct UserTypeProjections {
    pub contents: Vec<(UserTypeProjection, Span)>,
}

impl<'tcx> UserTypeProjections {
    pub fn none() -> Self {
        UserTypeProjections { contents: vec![] }
    }

    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }

    pub fn projections_and_spans(
        &self,
    ) -> impl Iterator<Item = &(UserTypeProjection, Span)> + ExactSizeIterator {
        self.contents.iter()
    }

    pub fn projections(&self) -> impl Iterator<Item = &UserTypeProjection> + ExactSizeIterator {
        self.contents.iter().map(|&(ref user_type, _span)| user_type)
    }

    pub fn push_projection(mut self, user_ty: &UserTypeProjection, span: Span) -> Self {
        self.contents.push((user_ty.clone(), span));
        self
    }

    fn map_projections(
        mut self,
        mut f: impl FnMut(UserTypeProjection) -> UserTypeProjection,
    ) -> Self {
        self.contents = self.contents.into_iter().map(|(proj, span)| (f(proj), span)).collect();
        self
    }

    pub fn index(self) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.index())
    }

    pub fn subslice(self, from: u64, to: u64) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.subslice(from, to))
    }

    pub fn deref(self) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.deref())
    }

    pub fn leaf(self, field: Field) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.leaf(field))
    }

    pub fn variant(self, adt_def: AdtDef<'tcx>, variant_index: VariantIdx, field: Field) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.variant(adt_def, variant_index, field))
    }
}

/// Encodes the effect of a user-supplied type annotation on the
/// subcomponents of a pattern. The effect is determined by applying the
/// given list of projections to some underlying base type. Often,
/// the projection element list `projs` is empty, in which case this
/// directly encodes a type in `base`. But in the case of complex patterns with
/// subpatterns and bindings, we want to apply only a *part* of the type to a variable,
/// in which case the `projs` vector is used.
///
/// Examples:
///
/// * `let x: T = ...` -- here, the `projs` vector is empty.
///
/// * `let (x, _): T = ...` -- here, the `projs` vector would contain
///   `field[0]` (aka `.0`), indicating that the type of `s` is
///   determined by finding the type of the `.0` field from `T`.
#[derive(Clone, Debug, TyEncodable, TyDecodable, Hash, HashStable, PartialEq)]
pub struct UserTypeProjection {
    pub base: UserTypeAnnotationIndex,
    pub projs: Vec<ProjectionKind>,
}

impl Copy for ProjectionKind {}

impl UserTypeProjection {
    pub(crate) fn index(mut self) -> Self {
        self.projs.push(ProjectionElem::Index(()));
        self
    }

    pub(crate) fn subslice(mut self, from: u64, to: u64) -> Self {
        self.projs.push(ProjectionElem::Subslice { from, to, from_end: true });
        self
    }

    pub(crate) fn deref(mut self) -> Self {
        self.projs.push(ProjectionElem::Deref);
        self
    }

    pub(crate) fn leaf(mut self, field: Field) -> Self {
        self.projs.push(ProjectionElem::Field(field, ()));
        self
    }

    pub(crate) fn variant(
        mut self,
        adt_def: AdtDef<'_>,
        variant_index: VariantIdx,
        field: Field,
    ) -> Self {
        self.projs.push(ProjectionElem::Downcast(
            Some(adt_def.variant(variant_index).name),
            variant_index,
        ));
        self.projs.push(ProjectionElem::Field(field, ()));
        self
    }
}

TrivialTypeFoldableAndLiftImpls! { ProjectionKind, }

impl<'tcx> TypeFoldable<'tcx> for UserTypeProjection {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(UserTypeProjection {
            base: self.base.try_fold_with(folder)?,
            projs: self.projs.try_fold_with(folder)?,
        })
    }

    fn super_visit_with<Vs: TypeVisitor<'tcx>>(
        &self,
        visitor: &mut Vs,
    ) -> ControlFlow<Vs::BreakTy> {
        self.base.visit_with(visitor)
        // Note: there's nothing in `self.proj` to visit.
    }
}

rustc_index::newtype_index! {
    pub struct Promoted {
        derive [HashStable]
        DEBUG_FORMAT = "promoted[{}]"
    }
}

impl<'tcx> Debug for Constant<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}", self)
    }
}

impl<'tcx> Display for Constant<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match self.ty().kind() {
            ty::FnDef(..) => {}
            _ => write!(fmt, "const ")?,
        }
        Display::fmt(&self.literal, fmt)
    }
}

impl<'tcx> Display for ConstantKind<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            ConstantKind::Ty(c) => pretty_print_const(c, fmt, true),
            ConstantKind::Val(val, ty) => pretty_print_const_value(val, ty, fmt, true),
        }
    }
}

fn pretty_print_const<'tcx>(
    c: ty::Const<'tcx>,
    fmt: &mut Formatter<'_>,
    print_types: bool,
) -> fmt::Result {
    use crate::ty::print::PrettyPrinter;
    ty::tls::with(|tcx| {
        let literal = tcx.lift(c).unwrap();
        let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
        cx.print_alloc_ids = true;
        let cx = cx.pretty_print_const(literal, print_types)?;
        fmt.write_str(&cx.into_buffer())?;
        Ok(())
    })
}

fn pretty_print_const_value<'tcx>(
    val: interpret::ConstValue<'tcx>,
    ty: Ty<'tcx>,
    fmt: &mut Formatter<'_>,
    print_types: bool,
) -> fmt::Result {
    use crate::ty::print::PrettyPrinter;
    ty::tls::with(|tcx| {
        let val = tcx.lift(val).unwrap();
        let ty = tcx.lift(ty).unwrap();
        let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
        cx.print_alloc_ids = true;
        let cx = cx.pretty_print_const_value(val, ty, print_types)?;
        fmt.write_str(&cx.into_buffer())?;
        Ok(())
    })
}

impl<'tcx> graph::DirectedGraph for Body<'tcx> {
    type Node = BasicBlock;
}

impl<'tcx> graph::WithNumNodes for Body<'tcx> {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.basic_blocks.len()
    }
}

impl<'tcx> graph::WithStartNode for Body<'tcx> {
    #[inline]
    fn start_node(&self) -> Self::Node {
        START_BLOCK
    }
}

impl<'tcx> graph::WithSuccessors for Body<'tcx> {
    #[inline]
    fn successors(&self, node: Self::Node) -> <Self as GraphSuccessors<'_>>::Iter {
        self.basic_blocks[node].terminator().successors().cloned()
    }
}

impl<'a, 'b> graph::GraphSuccessors<'b> for Body<'a> {
    type Item = BasicBlock;
    type Iter = iter::Cloned<Successors<'b>>;
}

impl<'tcx, 'graph> graph::GraphPredecessors<'graph> for Body<'tcx> {
    type Item = BasicBlock;
    type Iter = std::iter::Copied<std::slice::Iter<'graph, BasicBlock>>;
}

impl<'tcx> graph::WithPredecessors for Body<'tcx> {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> <Self as graph::GraphPredecessors<'_>>::Iter {
        self.predecessors()[node].iter().copied()
    }
}

/// `Location` represents the position of the start of the statement; or, if
/// `statement_index` equals the number of statements, then the start of the
/// terminator.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, HashStable)]
pub struct Location {
    /// The block that the location is within.
    pub block: BasicBlock,

    pub statement_index: usize,
}

impl fmt::Debug for Location {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{:?}[{}]", self.block, self.statement_index)
    }
}

impl Location {
    pub const START: Location = Location { block: START_BLOCK, statement_index: 0 };

    /// Returns the location immediately after this one within the enclosing block.
    ///
    /// Note that if this location represents a terminator, then the
    /// resulting location would be out of bounds and invalid.
    pub fn successor_within_block(&self) -> Location {
        Location { block: self.block, statement_index: self.statement_index + 1 }
    }

    /// Returns `true` if `other` is earlier in the control flow graph than `self`.
    pub fn is_predecessor_of<'tcx>(&self, other: Location, body: &Body<'tcx>) -> bool {
        // If we are in the same block as the other location and are an earlier statement
        // then we are a predecessor of `other`.
        if self.block == other.block && self.statement_index < other.statement_index {
            return true;
        }

        let predecessors = body.predecessors();

        // If we're in another block, then we want to check that block is a predecessor of `other`.
        let mut queue: Vec<BasicBlock> = predecessors[other.block].to_vec();
        let mut visited = FxHashSet::default();

        while let Some(block) = queue.pop() {
            // If we haven't visited this block before, then make sure we visit its predecessors.
            if visited.insert(block) {
                queue.extend(predecessors[block].iter().cloned());
            } else {
                continue;
            }

            // If we found the block that `self` is in, then we are a predecessor of `other` (since
            // we found that block by looking at the predecessors of `other`).
            if self.block == block {
                return true;
            }
        }

        false
    }

    pub fn dominates(&self, other: Location, dominators: &Dominators<BasicBlock>) -> bool {
        if self.block == other.block {
            self.statement_index <= other.statement_index
        } else {
            dominators.is_dominated_by(other.block, self.block)
        }
    }
}
