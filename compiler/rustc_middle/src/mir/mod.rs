//! MIR datatypes and passes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/mir/index.html

use std::borrow::Cow;
use std::fmt::{self, Debug, Formatter};
use std::iter;
use std::ops::{Index, IndexMut};

pub use basic_blocks::{BasicBlocks, SwitchTargetValue};
use either::Either;
use polonius_engine::Atom;
use rustc_abi::{FieldIdx, VariantIdx};
pub use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::dominators::Dominators;
use rustc_errors::{DiagArgName, DiagArgValue, DiagMessage, ErrorGuaranteed, IntoDiagArg};
use rustc_hir::def::{CtorKind, Namespace};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_hir::{
    self as hir, BindingMode, ByRef, CoroutineDesugaring, CoroutineKind, HirId, ImplicitSelfKind,
};
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Span, Symbol};
use tracing::{debug, trace};

pub use self::query::*;
use crate::mir::interpret::{AllocRange, Scalar};
use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::print::{FmtPrinter, Printer, pretty_print_const, with_no_trimmed_paths};
use crate::ty::{
    self, GenericArg, GenericArgsRef, Instance, InstanceKind, List, Ty, TyCtxt, TypeVisitableExt,
    TypingEnv, UserTypeAnnotationIndex,
};

mod basic_blocks;
mod consts;
pub mod coverage;
mod generic_graph;
pub mod generic_graphviz;
pub mod graphviz;
pub mod interpret;
pub mod mono;
pub mod pretty;
mod query;
mod statement;
mod syntax;
mod terminator;

pub mod traversal;
pub mod visit;

pub use consts::*;
use pretty::pretty_print_const_value;
pub use statement::*;
pub use syntax::*;
pub use terminator::*;

pub use self::generic_graph::graphviz_safe_def_name;
pub use self::graphviz::write_mir_graphviz;
pub use self::pretty::{
    PassWhere, create_dump_file, display_allocation, dump_enabled, dump_mir, write_mir_pretty,
};

/// Types for locals
pub type LocalDecls<'tcx> = IndexSlice<Local, LocalDecl<'tcx>>;

pub trait HasLocalDecls<'tcx> {
    fn local_decls(&self) -> &LocalDecls<'tcx>;
}

impl<'tcx> HasLocalDecls<'tcx> for IndexVec<Local, LocalDecl<'tcx>> {
    #[inline]
    fn local_decls(&self) -> &LocalDecls<'tcx> {
        self
    }
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

impl MirPhase {
    pub fn name(&self) -> &'static str {
        match *self {
            MirPhase::Built => "built",
            MirPhase::Analysis(AnalysisPhase::Initial) => "analysis",
            MirPhase::Analysis(AnalysisPhase::PostCleanup) => "analysis-post-cleanup",
            MirPhase::Runtime(RuntimePhase::Initial) => "runtime",
            MirPhase::Runtime(RuntimePhase::PostCleanup) => "runtime-post-cleanup",
            MirPhase::Runtime(RuntimePhase::Optimized) => "runtime-optimized",
        }
    }

    /// Gets the (dialect, phase) index of the current `MirPhase`. Both numbers
    /// are 1-indexed.
    pub fn index(&self) -> (usize, usize) {
        match *self {
            MirPhase::Built => (1, 1),
            MirPhase::Analysis(analysis_phase) => (2, 1 + analysis_phase as usize),
            MirPhase::Runtime(runtime_phase) => (3, 1 + runtime_phase as usize),
        }
    }

    /// Parses a `MirPhase` from a pair of strings. Panics if this isn't possible for any reason.
    pub fn parse(dialect: String, phase: Option<String>) -> Self {
        match &*dialect.to_ascii_lowercase() {
            "built" => {
                assert!(phase.is_none(), "Cannot specify a phase for `Built` MIR");
                MirPhase::Built
            }
            "analysis" => Self::Analysis(AnalysisPhase::parse(phase)),
            "runtime" => Self::Runtime(RuntimePhase::parse(phase)),
            _ => bug!("Unknown MIR dialect: '{}'", dialect),
        }
    }
}

impl AnalysisPhase {
    pub fn parse(phase: Option<String>) -> Self {
        let Some(phase) = phase else {
            return Self::Initial;
        };

        match &*phase.to_ascii_lowercase() {
            "initial" => Self::Initial,
            "post_cleanup" | "post-cleanup" | "postcleanup" => Self::PostCleanup,
            _ => bug!("Unknown analysis phase: '{}'", phase),
        }
    }
}

impl RuntimePhase {
    pub fn parse(phase: Option<String>) -> Self {
        let Some(phase) = phase else {
            return Self::Initial;
        };

        match &*phase.to_ascii_lowercase() {
            "initial" => Self::Initial,
            "post_cleanup" | "post-cleanup" | "postcleanup" => Self::PostCleanup,
            "optimized" => Self::Optimized,
            _ => bug!("Unknown runtime phase: '{}'", phase),
        }
    }
}

/// Where a specific `mir::Body` comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeFoldable, TypeVisitable)]
pub struct MirSource<'tcx> {
    pub instance: InstanceKind<'tcx>,

    /// If `Some`, this is a promoted rvalue within the parent function.
    pub promoted: Option<Promoted>,
}

impl<'tcx> MirSource<'tcx> {
    pub fn item(def_id: DefId) -> Self {
        MirSource { instance: InstanceKind::Item(def_id), promoted: None }
    }

    pub fn from_instance(instance: InstanceKind<'tcx>) -> Self {
        MirSource { instance, promoted: None }
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.instance.def_id()
    }
}

/// Additional information carried by a MIR body when it is lowered from a coroutine.
/// This information is modified as it is lowered during the `StateTransform` MIR pass,
/// so not all fields will be active at a given time. For example, the `yield_ty` is
/// taken out of the field after yields are turned into returns, and the `coroutine_drop`
/// body is only populated after the state transform pass.
#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
pub struct CoroutineInfo<'tcx> {
    /// The yield type of the function. This field is removed after the state transform pass.
    pub yield_ty: Option<Ty<'tcx>>,

    /// The resume type of the function. This field is removed after the state transform pass.
    pub resume_ty: Option<Ty<'tcx>>,

    /// Coroutine drop glue. This field is populated after the state transform pass.
    pub coroutine_drop: Option<Body<'tcx>>,

    /// The layout of a coroutine. This field is populated after the state transform pass.
    pub coroutine_layout: Option<CoroutineLayout<'tcx>>,

    /// If this is a coroutine then record the type of source expression that caused this coroutine
    /// to be created.
    pub coroutine_kind: CoroutineKind,
}

impl<'tcx> CoroutineInfo<'tcx> {
    // Sets up `CoroutineInfo` for a pre-coroutine-transform MIR body.
    pub fn initial(
        coroutine_kind: CoroutineKind,
        yield_ty: Ty<'tcx>,
        resume_ty: Ty<'tcx>,
    ) -> CoroutineInfo<'tcx> {
        CoroutineInfo {
            coroutine_kind,
            yield_ty: Some(yield_ty),
            resume_ty: Some(resume_ty),
            coroutine_drop: None,
            coroutine_layout: None,
        }
    }
}

/// Some item that needs to monomorphize successfully for a MIR body to be considered well-formed.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum MentionedItem<'tcx> {
    /// A function that gets called. We don't necessarily know its precise type yet, since it can be
    /// hidden behind a generic.
    Fn(Ty<'tcx>),
    /// A type that has its drop shim called.
    Drop(Ty<'tcx>),
    /// Unsizing casts might require vtables, so we have to record them.
    UnsizeCast { source_ty: Ty<'tcx>, target_ty: Ty<'tcx> },
    /// A closure that is coerced to a function pointer.
    Closure(Ty<'tcx>),
}

/// The lowered representation of a single function.
#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
pub struct Body<'tcx> {
    /// A list of basic blocks. References to basic block use a newtyped index type [`BasicBlock`]
    /// that indexes into this vector.
    pub basic_blocks: BasicBlocks<'tcx>,

    /// Records how far through the "desugaring and optimization" process this particular
    /// MIR has traversed. This is particularly useful when inlining, since in that context
    /// we instantiate the promoted constants and add them to our promoted vector -- but those
    /// promoted items have already been optimized, whereas ours have not. This field allows
    /// us to see the difference and forego optimization on the inlined promoted items.
    pub phase: MirPhase,

    /// How many passses we have executed since starting the current phase. Used for debug output.
    pub pass_count: usize,

    pub source: MirSource<'tcx>,

    /// A list of source scopes; these are referenced by statements
    /// and used for debuginfo. Indexed by a `SourceScope`.
    pub source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,

    /// Additional information carried by a MIR body when it is lowered from a coroutine.
    ///
    /// Note that the coroutine drop shim, any promoted consts, and other synthetic MIR
    /// bodies that come from processing a coroutine body are not typically coroutines
    /// themselves, and should probably set this to `None` to avoid carrying redundant
    /// information.
    pub coroutine: Option<Box<CoroutineInfo<'tcx>>>,

    /// Declarations of locals.
    ///
    /// The first local is the return value pointer, followed by `arg_count`
    /// locals for the function arguments, followed by any user-declared
    /// variables and temporaries.
    pub local_decls: IndexVec<Local, LocalDecl<'tcx>>,

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
    /// `None` indicates that the list has not been computed yet.
    ///
    /// This is soundness-critical, we make a guarantee that all consts syntactically mentioned in a
    /// function have successfully evaluated if the function ever gets executed at runtime.
    pub required_consts: Option<Vec<ConstOperand<'tcx>>>,

    /// Further items that were mentioned in this function and hence *may* become monomorphized,
    /// depending on optimizations. We use this to avoid optimization-dependent compile errors: the
    /// collector recursively traverses all "mentioned" items and evaluates all their
    /// `required_consts`.
    /// `None` indicates that the list has not been computed yet.
    ///
    /// This is *not* soundness-critical and the contents of this list are *not* a stable guarantee.
    /// All that's relevant is that this set is optimization-level-independent, and that it includes
    /// everything that the collector would consider "used". (For example, we currently compute this
    /// set after drop elaboration, so some drop calls that can never be reached are not considered
    /// "mentioned".) See the documentation of `CollectionMode` in
    /// `compiler/rustc_monomorphize/src/collector.rs` for more context.
    pub mentioned_items: Option<Vec<Spanned<MentionedItem<'tcx>>>>,

    /// Does this body use generic parameters. This is used for the `ConstEvaluatable` check.
    ///
    /// Note that this does not actually mean that this body is not computable right now.
    /// The repeat count in the following example is polymorphic, but can still be evaluated
    /// without knowing anything about the type parameter `T`.
    ///
    /// ```rust
    /// fn test<T>() {
    ///     let _ = [0; size_of::<*mut T>()];
    /// }
    /// ```
    ///
    /// **WARNING**: Do not change this flags after the MIR was originally created, even if an optimization
    /// removed the last mention of all generic params. We do not want to rely on optimizations and
    /// potentially allow things like `[u8; size_of::<T>() * 0]` due to this.
    pub is_polymorphic: bool,

    /// The phase at which this MIR should be "injected" into the compilation process.
    ///
    /// Everything that comes before this `MirPhase` should be skipped.
    ///
    /// This is only `Some` if the function that this body comes from was annotated with `rustc_custom_mir`.
    pub injection_phase: Option<MirPhase>,

    pub tainted_by_errors: Option<ErrorGuaranteed>,

    /// Coverage information collected from THIR/MIR during MIR building,
    /// to be used by the `InstrumentCoverage` pass.
    ///
    /// Only present if coverage is enabled and this function is eligible.
    /// Boxed to limit space overhead in non-coverage builds.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub coverage_info_hi: Option<Box<coverage::CoverageInfoHi>>,

    /// Per-function coverage information added by the `InstrumentCoverage`
    /// pass, to be used in conjunction with the coverage statements injected
    /// into this body's blocks.
    ///
    /// If `-Cinstrument-coverage` is not active, or if an individual function
    /// is not eligible for coverage, then this should always be `None`.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub function_coverage_info: Option<Box<coverage::FunctionCoverageInfo>>,
}

impl<'tcx> Body<'tcx> {
    pub fn new(
        source: MirSource<'tcx>,
        basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
        source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,
        local_decls: IndexVec<Local, LocalDecl<'tcx>>,
        user_type_annotations: ty::CanonicalUserTypeAnnotations<'tcx>,
        arg_count: usize,
        var_debug_info: Vec<VarDebugInfo<'tcx>>,
        span: Span,
        coroutine: Option<Box<CoroutineInfo<'tcx>>>,
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
            pass_count: 0,
            source,
            basic_blocks: BasicBlocks::new(basic_blocks),
            source_scopes,
            coroutine,
            local_decls,
            user_type_annotations,
            arg_count,
            spread_arg: None,
            var_debug_info,
            span,
            required_consts: None,
            mentioned_items: None,
            is_polymorphic: false,
            injection_phase: None,
            tainted_by_errors,
            coverage_info_hi: None,
            function_coverage_info: None,
        };
        body.is_polymorphic = body.has_non_region_param();
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
            pass_count: 0,
            source: MirSource::item(CRATE_DEF_ID.to_def_id()),
            basic_blocks: BasicBlocks::new(basic_blocks),
            source_scopes: IndexVec::new(),
            coroutine: None,
            local_decls: IndexVec::new(),
            user_type_annotations: IndexVec::new(),
            arg_count: 0,
            spread_arg: None,
            span: DUMMY_SP,
            required_consts: None,
            mentioned_items: None,
            var_debug_info: Vec::new(),
            is_polymorphic: false,
            injection_phase: None,
            tainted_by_errors: None,
            coverage_info_hi: None,
            function_coverage_info: None,
        };
        body.is_polymorphic = body.has_non_region_param();
        body
    }

    #[inline]
    pub fn basic_blocks_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.basic_blocks.as_mut()
    }

    pub fn typing_env(&self, tcx: TyCtxt<'tcx>) -> TypingEnv<'tcx> {
        match self.phase {
            // FIXME(#132279): we should reveal the opaques defined in the body during analysis.
            MirPhase::Built | MirPhase::Analysis(_) => TypingEnv {
                typing_mode: ty::TypingMode::non_body_analysis(),
                param_env: tcx.param_env(self.source.def_id()),
            },
            MirPhase::Runtime(_) => TypingEnv::post_analysis(tcx, self.source.def_id()),
        }
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
        } else {
            LocalKind::Temp
        }
    }

    /// Returns an iterator over all user-declared mutable locals.
    #[inline]
    pub fn mut_vars_iter(&self) -> impl Iterator<Item = Local> {
        (self.arg_count + 1..self.local_decls.len()).filter_map(move |index| {
            let local = Local::new(index);
            let decl = &self.local_decls[local];
            (decl.is_user_variable() && decl.mutability.is_mut()).then_some(local)
        })
    }

    /// Returns an iterator over all user-declared mutable arguments and locals.
    #[inline]
    pub fn mut_vars_and_args_iter(&self) -> impl Iterator<Item = Local> {
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
    pub fn drain_vars_and_temps(&mut self) -> impl Iterator<Item = LocalDecl<'tcx>> {
        self.local_decls.drain(self.arg_count + 1..)
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

    /// Returns the return type; it always return first element from `local_decls` array.
    #[inline]
    pub fn bound_return_ty(&self) -> ty::EarlyBinder<'tcx, Ty<'tcx>> {
        ty::EarlyBinder::bind(self.local_decls[RETURN_PLACE].ty)
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
    pub fn yield_ty(&self) -> Option<Ty<'tcx>> {
        self.coroutine.as_ref().and_then(|coroutine| coroutine.yield_ty)
    }

    #[inline]
    pub fn resume_ty(&self) -> Option<Ty<'tcx>> {
        self.coroutine.as_ref().and_then(|coroutine| coroutine.resume_ty)
    }

    /// Prefer going through [`TyCtxt::coroutine_layout`] rather than using this directly.
    #[inline]
    pub fn coroutine_layout_raw(&self) -> Option<&CoroutineLayout<'tcx>> {
        self.coroutine.as_ref().and_then(|coroutine| coroutine.coroutine_layout.as_ref())
    }

    #[inline]
    pub fn coroutine_drop(&self) -> Option<&Body<'tcx>> {
        self.coroutine.as_ref().and_then(|coroutine| coroutine.coroutine_drop.as_ref())
    }

    #[inline]
    pub fn coroutine_kind(&self) -> Option<CoroutineKind> {
        self.coroutine.as_ref().map(|coroutine| coroutine.coroutine_kind)
    }

    #[inline]
    pub fn should_skip(&self) -> bool {
        let Some(injection_phase) = self.injection_phase else {
            return false;
        };
        injection_phase > self.phase
    }

    #[inline]
    pub fn is_custom_mir(&self) -> bool {
        self.injection_phase.is_some()
    }

    /// If this basic block ends with a [`TerminatorKind::SwitchInt`] for which we can evaluate the
    /// discriminant in monomorphization, we return the discriminant bits and the
    /// [`SwitchTargets`], just so the caller doesn't also have to match on the terminator.
    fn try_const_mono_switchint<'a>(
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        block: &'a BasicBlockData<'tcx>,
    ) -> Option<(u128, &'a SwitchTargets)> {
        // There are two places here we need to evaluate a constant.
        let eval_mono_const = |constant: &ConstOperand<'tcx>| {
            // FIXME(#132279): what is this, why are we using an empty environment here.
            let typing_env = ty::TypingEnv::fully_monomorphized();
            let mono_literal = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                typing_env,
                crate::ty::EarlyBinder::bind(constant.const_),
            );
            mono_literal.try_eval_bits(tcx, typing_env)
        };

        let TerminatorKind::SwitchInt { discr, targets } = &block.terminator().kind else {
            return None;
        };

        // If this is a SwitchInt(const _), then we can just evaluate the constant and return.
        let discr = match discr {
            Operand::Constant(constant) => {
                let bits = eval_mono_const(constant)?;
                return Some((bits, targets));
            }
            Operand::Move(place) | Operand::Copy(place) => place,
        };

        // MIR for `if false` actually looks like this:
        // _1 = const _
        // SwitchInt(_1)
        //
        // And MIR for if intrinsics::ub_checks() looks like this:
        // _1 = UbChecks()
        // SwitchInt(_1)
        //
        // So we're going to try to recognize this pattern.
        //
        // If we have a SwitchInt on a non-const place, we find the most recent statement that
        // isn't a storage marker. If that statement is an assignment of a const to our
        // discriminant place, we evaluate and return the const, as if we've const-propagated it
        // into the SwitchInt.

        let last_stmt = block.statements.iter().rev().find(|stmt| {
            !matches!(stmt.kind, StatementKind::StorageDead(_) | StatementKind::StorageLive(_))
        })?;

        let (place, rvalue) = last_stmt.kind.as_assign()?;

        if discr != place {
            return None;
        }

        match rvalue {
            Rvalue::NullaryOp(NullOp::UbChecks, _) => Some((tcx.sess.ub_checks() as u128, targets)),
            Rvalue::Use(Operand::Constant(constant)) => {
                let bits = eval_mono_const(constant)?;
                Some((bits, targets))
            }
            _ => None,
        }
    }

    /// For a `Location` in this scope, determine what the "caller location" at that point is. This
    /// is interesting because of inlining: the `#[track_caller]` attribute of inlined functions
    /// must be honored. Falls back to the `tracked_caller` value for `#[track_caller]` functions,
    /// or the function's scope.
    pub fn caller_location_span<T>(
        &self,
        mut source_info: SourceInfo,
        caller_location: Option<T>,
        tcx: TyCtxt<'tcx>,
        from_span: impl FnOnce(Span) -> T,
    ) -> T {
        loop {
            let scope_data = &self.source_scopes[source_info.scope];

            if let Some((callee, callsite_span)) = scope_data.inlined {
                // Stop inside the most nested non-`#[track_caller]` function,
                // before ever reaching its caller (which is irrelevant).
                if !callee.def.requires_caller_location(tcx) {
                    return from_span(source_info.span);
                }
                source_info.span = callsite_span;
            }

            // Skip past all of the parents with `inlined: None`.
            match scope_data.inlined_parent_scope {
                Some(parent) => source_info.scope = parent,
                None => break,
            }
        }

        // No inlined `SourceScope`s, or all of them were `#[track_caller]`.
        caller_location.unwrap_or_else(|| from_span(source_info.span))
    }

    #[track_caller]
    pub fn set_required_consts(&mut self, required_consts: Vec<ConstOperand<'tcx>>) {
        assert!(
            self.required_consts.is_none(),
            "required_consts for {:?} have already been set",
            self.source.def_id()
        );
        self.required_consts = Some(required_consts);
    }
    #[track_caller]
    pub fn required_consts(&self) -> &[ConstOperand<'tcx>] {
        match &self.required_consts {
            Some(l) => l,
            None => panic!("required_consts for {:?} have not yet been set", self.source.def_id()),
        }
    }

    #[track_caller]
    pub fn set_mentioned_items(&mut self, mentioned_items: Vec<Spanned<MentionedItem<'tcx>>>) {
        assert!(
            self.mentioned_items.is_none(),
            "mentioned_items for {:?} have already been set",
            self.source.def_id()
        );
        self.mentioned_items = Some(mentioned_items);
    }
    #[track_caller]
    pub fn mentioned_items(&self) -> &[Spanned<MentionedItem<'tcx>>] {
        match &self.mentioned_items {
            Some(l) => l,
            None => panic!("mentioned_items for {:?} have not yet been set", self.source.def_id()),
        }
    }
}

impl<'tcx> Index<BasicBlock> for Body<'tcx> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.basic_blocks[index]
    }
}

impl<'tcx> IndexMut<BasicBlock> for Body<'tcx> {
    #[inline]
    fn index_mut(&mut self, index: BasicBlock) -> &mut BasicBlockData<'tcx> {
        &mut self.basic_blocks.as_mut()[index]
    }
}

#[derive(Copy, Clone, Debug, HashStable, TypeFoldable, TypeVisitable)]
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

    pub fn as_mut(&mut self) -> ClearCrossCrate<&mut T> {
        match self {
            ClearCrossCrate::Clear => ClearCrossCrate::Clear,
            ClearCrossCrate::Set(v) => ClearCrossCrate::Set(v),
        }
    }

    pub fn unwrap_crate_local(self) -> T {
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
    fn encode(&self, e: &mut E) {
        if E::CLEAR_CROSS_CRATE {
            return;
        }

        match *self {
            ClearCrossCrate::Clear => TAG_CLEAR_CROSS_CRATE_CLEAR.encode(e),
            ClearCrossCrate::Set(ref val) => {
                TAG_CLEAR_CROSS_CRATE_SET.encode(e);
                val.encode(e);
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
            tag => panic!("Invalid tag for ClearCrossCrate: {tag:?}"),
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
    /// seen by debuginfo, active lint levels, etc.
    pub scope: SourceScope,
}

impl SourceInfo {
    #[inline]
    pub fn outermost(span: Span) -> Self {
        SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE }
    }
}

///////////////////////////////////////////////////////////////////////////
// Variables and temps

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "_{}"]
    pub struct Local {
        const RETURN_PLACE = 0;
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
    /// User-declared variable binding or compiler-introduced temporary.
    Temp,
    /// Function argument.
    Arg,
    /// Location of function's return value.
    ReturnPointer,
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct VarBindingForm<'tcx> {
    /// Is variable bound via `x`, `mut x`, `ref x`, `ref mut x`, `mut ref x`, or `mut ref mut x`?
    pub binding_mode: BindingMode,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
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
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct LocalDecl<'tcx> {
    /// Whether this is a mutable binding (i.e., `let x` or `let mut x`).
    ///
    /// Temporaries and the return place are always mutable.
    pub mutability: Mutability,

    pub local_info: ClearCrossCrate<Box<LocalInfo<'tcx>>>,

    /// The type of this local.
    pub ty: Ty<'tcx>,

    /// If the user manually ascribed a type to this variable,
    /// e.g., via `let x: T`, then we carry that type here. The MIR
    /// borrow checker needs this information since it can affect
    /// region inference.
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
    ///         match x.parse::<u32>().unwrap() {
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

/// Extra information about a some locals that's used for diagnostics and for
/// classifying variables into local variables, statics, etc, which is needed e.g.
/// for borrow checking.
///
/// Not used for non-StaticRef temporaries, the return place, or anonymous
/// function parameters.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub enum LocalInfo<'tcx> {
    /// A user-defined local variable or function parameter
    ///
    /// The `BindingForm` is solely used for local diagnostics when generating
    /// warnings/errors when compiling the current crate, and therefore it need
    /// not be visible across crates.
    User(BindingForm<'tcx>),
    /// A temporary created that references the static with the given `DefId`.
    StaticRef { def_id: DefId, is_thread_local: bool },
    /// A temporary created that references the const with the given `DefId`
    ConstRef { def_id: DefId },
    /// A temporary created during the creation of an aggregate
    /// (e.g. a temporary for `foo` in `MyStruct { my_field: foo }`)
    AggregateTemp,
    /// A temporary created for evaluation of some subexpression of some block's tail expression
    /// (with no intervening statement context).
    BlockTailTemp(BlockTailInfo),
    /// A temporary created during evaluating `if` predicate, possibly for pattern matching for `let`s,
    /// and subject to Edition 2024 temporary lifetime rules
    IfThenRescopeTemp { if_then: HirId },
    /// A temporary created during the pass `Derefer` to avoid it's retagging
    DerefTemp,
    /// A temporary created for borrow checking.
    FakeBorrow,
    /// A local without anything interesting about it.
    Boring,
}

impl<'tcx> LocalDecl<'tcx> {
    pub fn local_info(&self) -> &LocalInfo<'tcx> {
        self.local_info.as_ref().unwrap_crate_local()
    }

    /// Returns `true` only if local is a binding that can itself be
    /// made mutable via the addition of the `mut` keyword, namely
    /// something like the occurrences of `x` in:
    /// - `fn foo(x: Type) { ... }`,
    /// - `let x = ...`,
    /// - or `match ... { C(x) => ... }`
    pub fn can_be_made_mutable(&self) -> bool {
        matches!(
            self.local_info(),
            LocalInfo::User(
                BindingForm::Var(VarBindingForm {
                    binding_mode: BindingMode(ByRef::No, _),
                    opt_ty_info: _,
                    opt_match_place: _,
                    pat_span: _,
                }) | BindingForm::ImplicitSelf(ImplicitSelfKind::Imm),
            )
        )
    }

    /// Returns `true` if local is definitely not a `ref ident` or
    /// `ref mut ident` binding. (Such bindings cannot be made into
    /// mutable bindings, but the inverse does not necessarily hold).
    pub fn is_nonref_binding(&self) -> bool {
        matches!(
            self.local_info(),
            LocalInfo::User(
                BindingForm::Var(VarBindingForm {
                    binding_mode: BindingMode(ByRef::No, _),
                    opt_ty_info: _,
                    opt_match_place: _,
                    pat_span: _,
                }) | BindingForm::ImplicitSelf(_),
            )
        )
    }

    /// Returns `true` if this variable is a named variable or function
    /// parameter declared by the user.
    #[inline]
    pub fn is_user_variable(&self) -> bool {
        matches!(self.local_info(), LocalInfo::User(_))
    }

    /// Returns `true` if this is a reference to a variable bound in a `match`
    /// expression that is used to access said variable for the guard of the
    /// match arm.
    pub fn is_ref_for_guard(&self) -> bool {
        matches!(self.local_info(), LocalInfo::User(BindingForm::RefForGuard))
    }

    /// Returns `Some` if this is a reference to a static item that is used to
    /// access that static.
    pub fn is_ref_to_static(&self) -> bool {
        matches!(self.local_info(), LocalInfo::StaticRef { .. })
    }

    /// Returns `Some` if this is a reference to a thread-local static item that is used to
    /// access that static.
    pub fn is_ref_to_thread_local(&self) -> bool {
        match self.local_info() {
            LocalInfo::StaticRef { is_thread_local, .. } => *is_thread_local,
            _ => false,
        }
    }

    /// Returns `true` if this is a DerefTemp
    pub fn is_deref_temp(&self) -> bool {
        match self.local_info() {
            LocalInfo::DerefTemp => true,
            _ => false,
        }
    }

    /// Returns `true` is the local is from a compiler desugaring, e.g.,
    /// `__next` from a `for` loop.
    #[inline]
    pub fn from_compiler_desugaring(&self) -> bool {
        self.source_info.span.desugaring_kind().is_some()
    }

    /// Creates a new `LocalDecl` for a temporary, mutable.
    #[inline]
    pub fn new(ty: Ty<'tcx>, span: Span) -> Self {
        Self::with_source_info(ty, SourceInfo::outermost(span))
    }

    /// Like `LocalDecl::new`, but takes a `SourceInfo` instead of a `Span`.
    #[inline]
    pub fn with_source_info(ty: Ty<'tcx>, source_info: SourceInfo) -> Self {
        LocalDecl {
            mutability: Mutability::Mut,
            local_info: ClearCrossCrate::Set(Box::new(LocalInfo::Boring)),
            ty,
            user_ty: None,
            source_info,
        }
    }

    /// Converts `self` into same `LocalDecl` except tagged as immutable.
    #[inline]
    pub fn immutable(mut self) -> Self {
        self.mutability = Mutability::Not;
        self
    }
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub enum VarDebugInfoContents<'tcx> {
    /// This `Place` only contains projection which satisfy `can_use_in_debuginfo`.
    Place(Place<'tcx>),
    Const(ConstOperand<'tcx>),
}

impl<'tcx> Debug for VarDebugInfoContents<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VarDebugInfoContents::Const(c) => write!(fmt, "{c}"),
            VarDebugInfoContents::Place(p) => write!(fmt, "{p:?}"),
        }
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct VarDebugInfoFragment<'tcx> {
    /// Type of the original user variable.
    /// This cannot contain a union or an enum.
    pub ty: Ty<'tcx>,

    /// Where in the composite user variable this fragment is,
    /// represented as a "projection" into the composite variable.
    /// At lower levels, this corresponds to a byte/bit range.
    ///
    /// This can only contain `PlaceElem::Field`.
    // FIXME support this for `enum`s by either using DWARF's
    // more advanced control-flow features (unsupported by LLVM?)
    // to match on the discriminant, or by using custom type debuginfo
    // with non-overlapping variants for the composite variable.
    pub projection: Vec<PlaceElem<'tcx>>,
}

/// Debug information pertaining to a user variable.
#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct VarDebugInfo<'tcx> {
    pub name: Symbol,

    /// Source info of the user variable, including the scope
    /// within which the variable is visible (to debuginfo)
    /// (see `LocalDecl`'s `source_info` field for more details).
    pub source_info: SourceInfo,

    /// The user variable's data is split across several fragments,
    /// each described by a `VarDebugInfoFragment`.
    /// See DWARF 5's "2.6.1.2 Composite Location Descriptions"
    /// and LLVM's `DW_OP_LLVM_fragment` for more details on
    /// the underlying debuginfo feature this relies on.
    pub composite: Option<Box<VarDebugInfoFragment<'tcx>>>,

    /// Where the data for this user variable is to be found.
    pub value: VarDebugInfoContents<'tcx>,

    /// When present, indicates what argument number this variable is in the function that it
    /// originated from (starting from 1). Note, if MIR inlining is enabled, then this is the
    /// argument number in the original function before it was inlined.
    pub argument_index: Option<u16>,
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
    /// [`CriticalCallEdges`]: ../../rustc_mir_transform/add_call_guards/enum.AddCallGuards.html#variant.CriticalCallEdges
    /// [guide-mir]: https://rustc-dev-guide.rust-lang.org/mir/
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "bb{}"]
    pub struct BasicBlock {
        const START_BLOCK = 0;
    }
}

impl BasicBlock {
    pub fn start_location(self) -> Location {
        Location { block: self, statement_index: 0 }
    }
}

///////////////////////////////////////////////////////////////////////////
// BasicBlockData

/// Data for a basic block, including a list of its statements.
///
/// See [`BasicBlock`] for documentation on what basic blocks are at a high level.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
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

impl<'tcx> BasicBlockData<'tcx> {
    pub fn new(terminator: Option<Terminator<'tcx>>, is_cleanup: bool) -> BasicBlockData<'tcx> {
        BasicBlockData { statements: vec![], terminator, is_cleanup }
    }

    /// Accessor for terminator.
    ///
    /// Terminator may not be None after construction of the basic block is complete. This accessor
    /// provides a convenient way to reach the terminator.
    #[inline]
    pub fn terminator(&self) -> &Terminator<'tcx> {
        self.terminator.as_ref().expect("invalid terminator state")
    }

    #[inline]
    pub fn terminator_mut(&mut self) -> &mut Terminator<'tcx> {
        self.terminator.as_mut().expect("invalid terminator state")
    }

    /// Does the block have no statements and an unreachable terminator?
    #[inline]
    pub fn is_empty_unreachable(&self) -> bool {
        self.statements.is_empty() && matches!(self.terminator().kind, TerminatorKind::Unreachable)
    }

    /// Like [`Terminator::successors`] but tries to use information available from the [`Instance`]
    /// to skip successors like the `false` side of an `if const {`.
    ///
    /// This is used to implement [`traversal::mono_reachable`] and
    /// [`traversal::mono_reachable_reverse_postorder`].
    pub fn mono_successors(&self, tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Successors<'_> {
        if let Some((bits, targets)) = Body::try_const_mono_switchint(tcx, instance, self) {
            targets.successors_for_value(bits)
        } else {
            self.terminator().successors()
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Scopes

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[debug_format = "scope[{}]"]
    pub struct SourceScope {
        const OUTERMOST_SOURCE_SCOPE = 0;
    }
}

impl SourceScope {
    /// Finds the original HirId this MIR item came from.
    /// This is necessary after MIR optimizations, as otherwise we get a HirId
    /// from the function that was inlined instead of the function call site.
    pub fn lint_root(
        self,
        source_scopes: &IndexSlice<SourceScope, SourceScopeData<'_>>,
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

    /// The instance this source scope was inlined from, if any.
    #[inline]
    pub fn inlined_instance<'tcx>(
        self,
        source_scopes: &IndexSlice<SourceScope, SourceScopeData<'tcx>>,
    ) -> Option<ty::Instance<'tcx>> {
        let scope_data = &source_scopes[self];
        if let Some((inlined_instance, _)) = scope_data.inlined {
            Some(inlined_instance)
        } else if let Some(inlined_scope) = scope_data.inlined_parent_scope {
            Some(source_scopes[inlined_scope].inlined.unwrap().0)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
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
    pub lint_root: HirId,
}

/// A collection of projections into user types.
///
/// They are projections because a binding can occur a part of a
/// parent pattern that has been ascribed a type.
///
/// It's a collection because there can be multiple type ascriptions on
/// the path from the root of the pattern down to the binding itself.
///
/// An example:
///
/// ```ignore (illustrative)
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
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct UserTypeProjections {
    pub contents: Vec<UserTypeProjection>,
}

impl UserTypeProjections {
    pub fn projections(&self) -> impl Iterator<Item = &UserTypeProjection> + ExactSizeIterator {
        self.contents.iter()
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
#[derive(TypeFoldable, TypeVisitable)]
pub struct UserTypeProjection {
    pub base: UserTypeAnnotationIndex,
    pub projs: Vec<ProjectionKind>,
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    #[debug_format = "promoted[{}]"]
    pub struct Promoted {}
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
    #[inline]
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

        let predecessors = body.basic_blocks.predecessors();

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

    #[inline]
    pub fn dominates(&self, other: Location, dominators: &Dominators<BasicBlock>) -> bool {
        if self.block == other.block {
            self.statement_index <= other.statement_index
        } else {
            dominators.dominates(self.block, other.block)
        }
    }
}

/// `DefLocation` represents the location of a definition - either an argument or an assignment
/// within MIR body.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DefLocation {
    Argument,
    Assignment(Location),
    CallReturn { call: BasicBlock, target: Option<BasicBlock> },
}

impl DefLocation {
    #[inline]
    pub fn dominates(self, location: Location, dominators: &Dominators<BasicBlock>) -> bool {
        match self {
            DefLocation::Argument => true,
            DefLocation::Assignment(def) => {
                def.successor_within_block().dominates(location, dominators)
            }
            DefLocation::CallReturn { target: None, .. } => false,
            DefLocation::CallReturn { call, target: Some(target) } => {
                // The definition occurs on the call -> target edge. The definition dominates a use
                // if and only if the edge is on all paths from the entry to the use.
                //
                // Note that a call terminator has only one edge that can reach the target, so when
                // the call strongly dominates the target, all paths from the entry to the target
                // go through the call -> target edge.
                call != target
                    && dominators.dominates(call, target)
                    && dominators.dominates(target, location.block)
            }
        }
    }
}

/// Checks if the specified `local` is used as the `self` parameter of a method call
/// in the provided `BasicBlock`. If it is, then the `DefId` of the called method is
/// returned.
pub fn find_self_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    local: Local,
    block: BasicBlock,
) -> Option<(DefId, GenericArgsRef<'tcx>)> {
    debug!("find_self_call(local={:?}): terminator={:?}", local, body[block].terminator);
    if let Some(Terminator { kind: TerminatorKind::Call { func, args, .. }, .. }) =
        &body[block].terminator
        && let Operand::Constant(box ConstOperand { const_, .. }) = func
        && let ty::FnDef(def_id, fn_args) = *const_.ty().kind()
        && let Some(item) = tcx.opt_associated_item(def_id)
        && item.is_method()
        && let [Spanned { node: Operand::Move(self_place) | Operand::Copy(self_place), .. }, ..] =
            **args
    {
        if self_place.as_local() == Some(local) {
            return Some((def_id, fn_args));
        }

        // Handle the case where `self_place` gets reborrowed.
        // This happens when the receiver is `&T`.
        for stmt in &body[block].statements {
            if let StatementKind::Assign(box (place, rvalue)) = &stmt.kind
                && let Some(reborrow_local) = place.as_local()
                && self_place.as_local() == Some(reborrow_local)
                && let Rvalue::Ref(_, _, deref_place) = rvalue
                && let PlaceRef { local: deref_local, projection: [ProjectionElem::Deref] } =
                    deref_place.as_ref()
                && deref_local == local
            {
                return Some((def_id, fn_args));
            }
        }
    }
    None
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(BasicBlockData<'_>, 128);
    static_assert_size!(LocalDecl<'_>, 40);
    static_assert_size!(SourceScopeData<'_>, 64);
    static_assert_size!(Statement<'_>, 32);
    static_assert_size!(Terminator<'_>, 96);
    static_assert_size!(VarDebugInfo<'_>, 88);
    // tidy-alphabetical-end
}
