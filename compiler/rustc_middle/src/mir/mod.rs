//! MIR datatypes and passes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/mir/index.html

use crate::mir::interpret::{
    AllocRange, ConstAllocation, ConstValue, ErrorHandled, GlobalAlloc, Scalar,
};
use crate::mir::visit::MirVisitable;
use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::fold::{FallibleTypeFolder, TypeFoldable};
use crate::ty::print::{FmtPrinter, Printer};
use crate::ty::visit::TypeVisitableExt;
use crate::ty::{self, List, Ty, TyCtxt};
use crate::ty::{AdtDef, InstanceDef, ScalarInt, UserTypeAnnotationIndex};
use crate::ty::{GenericArg, GenericArgs, GenericArgsRef};

use rustc_data_structures::captures::Captures;
use rustc_errors::{DiagnosticArgValue, DiagnosticMessage, ErrorGuaranteed, IntoDiagnosticArg};
use rustc_hir::def::{CtorKind, Namespace};
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_hir::{self, GeneratorKind, ImplicitSelfKind};
use rustc_hir::{self as hir, HirId};
use rustc_session::Session;
use rustc_target::abi::{FieldIdx, Size, VariantIdx};

use polonius_engine::Atom;
pub use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::symbol::Symbol;
use rustc_span::{Span, DUMMY_SP};

use either::Either;

use std::borrow::Cow;
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::ops::{Index, IndexMut};
use std::{iter, mem};

pub use self::query::*;
pub use basic_blocks::BasicBlocks;

mod basic_blocks;
pub mod coverage;
mod generic_graph;
pub mod generic_graphviz;
pub mod graphviz;
pub mod interpret;
pub mod mono;
pub mod patch;
pub mod pretty;
mod query;
pub mod spanview;
mod syntax;
pub use syntax::*;
pub mod tcx;
pub mod terminator;
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

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass<'tcx> {
    fn name(&self) -> &'static str {
        let name = std::any::type_name::<Self>();
        if let Some((_, tail)) = name.rsplit_once(':') { tail } else { name }
    }

    /// Returns `true` if this pass is enabled with the current combination of compiler flags.
    fn is_enabled(&self, _sess: &Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>);

    fn is_mir_dump_enabled(&self) -> bool {
        true
    }
}

impl MirPhase {
    /// Gets the index of the current MirPhase within the set of all `MirPhase`s.
    ///
    /// FIXME(JakobDegen): Return a `(usize, usize)` instead.
    pub fn phase_index(&self) -> usize {
        const BUILT_PHASE_COUNT: usize = 1;
        const ANALYSIS_PHASE_COUNT: usize = 2;
        match self {
            MirPhase::Built => 1,
            MirPhase::Analysis(analysis_phase) => {
                1 + BUILT_PHASE_COUNT + (*analysis_phase as usize)
            }
            MirPhase::Runtime(runtime_phase) => {
                1 + BUILT_PHASE_COUNT + ANALYSIS_PHASE_COUNT + (*runtime_phase as usize)
            }
        }
    }

    /// Parses an `MirPhase` from a pair of strings. Panics if this isn't possible for any reason.
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
    pub instance: InstanceDef<'tcx>,

    /// If `Some`, this is a promoted rvalue within the parent function.
    pub promoted: Option<Promoted>,
}

impl<'tcx> MirSource<'tcx> {
    pub fn item(def_id: DefId) -> Self {
        MirSource { instance: InstanceDef::Item(def_id), promoted: None }
    }

    pub fn from_instance(instance: InstanceDef<'tcx>) -> Self {
        MirSource { instance, promoted: None }
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.instance.def_id()
    }
}

#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
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

    pub generator: Option<Box<GeneratorInfo<'tcx>>>,

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

    /// The phase at which this MIR should be "injected" into the compilation process.
    ///
    /// Everything that comes before this `MirPhase` should be skipped.
    ///
    /// This is only `Some` if the function that this body comes from was annotated with `rustc_custom_mir`.
    pub injection_phase: Option<MirPhase>,

    pub tainted_by_errors: Option<ErrorGuaranteed>,
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
            pass_count: 0,
            source,
            basic_blocks: BasicBlocks::new(basic_blocks),
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
            injection_phase: None,
            tainted_by_errors,
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
            generator: None,
            local_decls: IndexVec::new(),
            user_type_annotations: IndexVec::new(),
            arg_count: 0,
            spread_arg: None,
            span: DUMMY_SP,
            required_consts: Vec::new(),
            var_debug_info: Vec::new(),
            is_polymorphic: false,
            injection_phase: None,
            tainted_by_errors: None,
        };
        body.is_polymorphic = body.has_non_region_param();
        body
    }

    #[inline]
    pub fn basic_blocks_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.basic_blocks.as_mut()
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
    pub fn mut_vars_iter<'a>(&'a self) -> impl Iterator<Item = Local> + Captures<'tcx> + 'a {
        (self.arg_count + 1..self.local_decls.len()).filter_map(move |index| {
            let local = Local::new(index);
            let decl = &self.local_decls[local];
            (decl.is_user_variable() && decl.mutability.is_mut()).then_some(local)
        })
    }

    /// Returns an iterator over all user-declared mutable arguments and locals.
    #[inline]
    pub fn mut_vars_and_args_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = Local> + Captures<'tcx> + 'a {
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
    pub fn bound_return_ty(&self) -> ty::EarlyBinder<Ty<'tcx>> {
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

    pub fn assert_crate_local(self) -> T {
        match self {
            ClearCrossCrate::Clear => bug!("unwrapping cross-crate data"),
            ClearCrossCrate::Set(v) => v,
        }
    }
}

const TAG_CLEAR_CROSS_CRATE_CLEAR: u8 = 0;
const TAG_CLEAR_CROSS_CRATE_SET: u8 = 1;

impl<E: TyEncoder, T: Encodable<E>> Encodable<E> for ClearCrossCrate<T> {
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
impl<D: TyDecoder, T: Decodable<D>> Decodable<D> for ClearCrossCrate<T> {
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
// Variables and temps

rustc_index::newtype_index! {
    #[derive(HashStable)]
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

TrivialTypeTraversalAndLiftImpls! { BindingForm<'tcx> }

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
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct LocalDecl<'tcx> {
    /// Whether this is a mutable binding (i.e., `let x` or `let mut x`).
    ///
    /// Temporaries and the return place are always mutable.
    pub mutability: Mutability,

    // FIXME(matthewjasper) Don't store in this in `Body`
    pub local_info: ClearCrossCrate<Box<LocalInfo<'tcx>>>,

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
/// for unsafety checking.
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
    // FIXME(matthewjasper) Don't store in this in `Body`
    BlockTailTemp(BlockTailInfo),
    /// A temporary created during the pass `Derefer` to avoid it's retagging
    DerefTemp,
    /// A temporary created for borrow checking.
    FakeBorrow,
    /// A local without anything interesting about it.
    Boring,
}

impl<'tcx> LocalDecl<'tcx> {
    pub fn local_info(&self) -> &LocalInfo<'tcx> {
        &self.local_info.as_ref().assert_crate_local()
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
                    binding_mode: ty::BindingMode::BindByValue(_),
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
                    binding_mode: ty::BindingMode::BindByValue(_),
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
            LocalInfo::DerefTemp => return true,
            _ => (),
        }
        return false;
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
            local_info: ClearCrossCrate::Set(Box::new(LocalInfo::Boring)),
            internal: false,
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
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub enum VarDebugInfoContents<'tcx> {
    /// This `Place` only contains projection which satisfy `can_use_in_debuginfo`.
    Place(Place<'tcx>),
    Const(Constant<'tcx>),
    /// The user variable's data is split across several fragments,
    /// each described by a `VarDebugInfoFragment`.
    /// See DWARF 5's "2.6.1.2 Composite Location Descriptions"
    /// and LLVM's `DW_OP_LLVM_fragment` for more details on
    /// the underlying debuginfo feature this relies on.
    Composite {
        /// Type of the original user variable.
        /// This cannot contain a union or an enum.
        ty: Ty<'tcx>,
        /// All the parts of the original user variable, which ended
        /// up in disjoint places, due to optimizations.
        fragments: Vec<VarDebugInfoFragment<'tcx>>,
    },
}

impl<'tcx> Debug for VarDebugInfoContents<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match self {
            VarDebugInfoContents::Const(c) => write!(fmt, "{}", c),
            VarDebugInfoContents::Place(p) => write!(fmt, "{:?}", p),
            VarDebugInfoContents::Composite { ty, fragments } => {
                write!(fmt, "{:?}{{ ", ty)?;
                for f in fragments.iter() {
                    write!(fmt, "{:?}, ", f)?;
                }
                write!(fmt, "}}")
            }
        }
    }
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct VarDebugInfoFragment<'tcx> {
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

    /// Where the data for this fragment can be found.
    /// This `Place` only contains projection which satisfy `can_use_in_debuginfo`.
    pub contents: Place<'tcx>,
}

impl Debug for VarDebugInfoFragment<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        for elem in self.projection.iter() {
            match elem {
                ProjectionElem::Field(field, _) => {
                    write!(fmt, ".{:?}", field.index())?;
                }
                _ => bug!("unsupported fragment projection `{:?}`", elem),
            }
        }

        write!(fmt, " => {:?}", self.contents)
    }
}

/// Debug information pertaining to a user variable.
#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct VarDebugInfo<'tcx> {
    pub name: Symbol,

    /// Source info of the user variable, including the scope
    /// within which the variable is visible (to debuginfo)
    /// (see `LocalDecl`'s `source_info` field for more details).
    pub source_info: SourceInfo,

    /// Where the data for this user variable is to be found.
    pub value: VarDebugInfoContents<'tcx>,

    /// When present, indicates what argument number this variable is in the function that it
    /// originated from (starting from 1). Note, if MIR inlining is enabled, then this is the
    /// argument number in the original function before it was inlined.
    pub argument_index: Option<u16>,

    /// The data represents `name` dereferenced `references` times,
    /// and not the direct value.
    pub references: u8,
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
    #[derive(HashStable)]
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
    pub fn new(terminator: Option<Terminator<'tcx>>) -> BasicBlockData<'tcx> {
        BasicBlockData { statements: vec![], terminator, is_cleanup: false }
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

    /// Does the block have no statements and an unreachable terminator?
    pub fn is_empty_unreachable(&self) -> bool {
        self.statements.is_empty() && matches!(self.terminator().kind, TerminatorKind::Unreachable)
    }
}

impl<O> AssertKind<O> {
    /// Returns true if this an overflow checking assertion controlled by -C overflow-checks.
    pub fn is_optional_overflow_check(&self) -> bool {
        use AssertKind::*;
        use BinOp::*;
        matches!(self, OverflowNeg(..) | Overflow(Add | Sub | Mul | Shl | Shr, ..))
    }

    /// Getting a description does not require `O` to be printable, and does not
    /// require allocation.
    /// The caller is expected to handle `BoundsCheck` and `MisalignedPointerDereference` separately.
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
            BoundsCheck { .. } | MisalignedPointerDereference { .. } => {
                bug!("Unexpected AssertKind")
            }
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
            MisalignedPointerDereference { required, found } => {
                write!(
                    f,
                    "\"misaligned pointer dereference: address must be a multiple of {{}} but is {{}}\", {:?}, {:?}",
                    required, found
                )
            }
            _ => write!(f, "\"{}\"", self.description()),
        }
    }

    pub fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use AssertKind::*;

        match self {
            BoundsCheck { .. } => middle_bounds_check,
            Overflow(BinOp::Shl, _, _) => middle_assert_shl_overflow,
            Overflow(BinOp::Shr, _, _) => middle_assert_shr_overflow,
            Overflow(_, _, _) => middle_assert_op_overflow,
            OverflowNeg(_) => middle_assert_overflow_neg,
            DivisionByZero(_) => middle_assert_divide_by_zero,
            RemainderByZero(_) => middle_assert_remainder_by_zero,
            ResumedAfterReturn(GeneratorKind::Async(_)) => middle_assert_async_resume_after_return,
            ResumedAfterReturn(GeneratorKind::Gen) => middle_assert_generator_resume_after_return,
            ResumedAfterPanic(GeneratorKind::Async(_)) => middle_assert_async_resume_after_panic,
            ResumedAfterPanic(GeneratorKind::Gen) => middle_assert_generator_resume_after_panic,

            MisalignedPointerDereference { .. } => middle_assert_misaligned_ptr_deref,
        }
    }

    pub fn add_args(self, adder: &mut dyn FnMut(Cow<'static, str>, DiagnosticArgValue<'static>))
    where
        O: fmt::Debug,
    {
        use AssertKind::*;

        macro_rules! add {
            ($name: expr, $value: expr) => {
                adder($name.into(), $value.into_diagnostic_arg());
            };
        }

        match self {
            BoundsCheck { len, index } => {
                add!("len", format!("{len:?}"));
                add!("index", format!("{index:?}"));
            }
            Overflow(BinOp::Shl | BinOp::Shr, _, val)
            | DivisionByZero(val)
            | RemainderByZero(val)
            | OverflowNeg(val) => {
                add!("val", format!("{val:#?}"));
            }
            Overflow(binop, left, right) => {
                add!("op", binop.to_hir_binop().as_str());
                add!("left", format!("{left:#?}"));
                add!("right", format!("{right:#?}"));
            }
            ResumedAfterReturn(_) | ResumedAfterPanic(_) => {}
            MisalignedPointerDereference { required, found } => {
                add!("required", format!("{required:#?}"));
                add!("found", format!("{found:#?}"));
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Statements

/// A statement in a basic block, including information about its source code.
#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct Statement<'tcx> {
    pub source_info: SourceInfo,
    pub kind: StatementKind<'tcx>,
}

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
            PlaceMention(ref place) => {
                write!(fmt, "PlaceMention({:?})", place)
            }
            AscribeUserType(box (ref place, ref c_ty), ref variance) => {
                write!(fmt, "AscribeUserType({:?}, {:?}, {:?})", place, variance, c_ty)
            }
            Coverage(box self::Coverage { ref kind, code_region: Some(ref rgn) }) => {
                write!(fmt, "Coverage::{:?} for {:?}", kind, rgn)
            }
            Coverage(box ref coverage) => write!(fmt, "Coverage::{:?}", coverage.kind),
            Intrinsic(box ref intrinsic) => write!(fmt, "{intrinsic}"),
            ConstEvalCounter => write!(fmt, "ConstEvalCounter"),
            Nop => write!(fmt, "nop"),
        }
    }
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

///////////////////////////////////////////////////////////////////////////
// Places

impl<V, T> ProjectionElem<V, T> {
    /// Returns `true` if the target of this projection may refer to a different region of memory
    /// than the base.
    fn is_indirect(&self) -> bool {
        match self {
            Self::Deref => true,

            Self::Field(_, _)
            | Self::Index(_)
            | Self::OpaqueCast(_)
            | Self::ConstantIndex { .. }
            | Self::Subslice { .. }
            | Self::Downcast(_, _) => false,
        }
    }

    /// Returns `true` if the target of this projection always refers to the same memory region
    /// whatever the state of the program.
    pub fn is_stable_offset(&self) -> bool {
        match self {
            Self::Deref | Self::Index(_) => false,
            Self::Field(_, _)
            | Self::OpaqueCast(_)
            | Self::ConstantIndex { .. }
            | Self::Subslice { .. }
            | Self::Downcast(_, _) => true,
        }
    }

    /// Returns `true` if this is a `Downcast` projection with the given `VariantIdx`.
    pub fn is_downcast_to(&self, v: VariantIdx) -> bool {
        matches!(*self, Self::Downcast(_, x) if x == v)
    }

    /// Returns `true` if this is a `Field` projection with the given index.
    pub fn is_field_to(&self, f: FieldIdx) -> bool {
        matches!(*self, Self::Field(x, _) if x == f)
    }

    /// Returns `true` if this is accepted inside `VarDebugInfoContents::Place`.
    pub fn can_use_in_debuginfo(&self) -> bool {
        match self {
            Self::ConstantIndex { from_end: false, .. }
            | Self::Deref
            | Self::Downcast(_, _)
            | Self::Field(_, _) => true,
            Self::ConstantIndex { from_end: true, .. }
            | Self::Index(_)
            | Self::OpaqueCast(_)
            | Self::Subslice { .. } => false,
        }
    }
}

/// Alias for projections as they appear in `UserTypeProjection`, where we
/// need neither the `V` parameter for `Index` nor the `T` for `Field`.
pub type ProjectionKind = ProjectionElem<(), ()>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaceRef<'tcx> {
    pub local: Local,
    pub projection: &'tcx [PlaceElem<'tcx>],
}

// Once we stop implementing `Ord` for `DefId`,
// this impl will be unnecessary. Until then, we'll
// leave this impl in place to prevent re-adding a
// dependency on the `Ord` impl for `DefId`
impl<'tcx> !PartialOrd for PlaceRef<'tcx> {}

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

    /// If MirPhase >= Derefered and if projection contains Deref,
    /// It's guaranteed to be in the first place
    pub fn has_deref(&self) -> bool {
        // To make sure this is not accidentally used in wrong mir phase
        debug_assert!(
            self.projection.is_empty() || !self.projection[1..].contains(&PlaceElem::Deref)
        );
        self.projection.first() == Some(&PlaceElem::Deref)
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
        self.as_ref().iter_projections()
    }

    /// Generates a new place by appending `more_projections` to the existing ones
    /// and interning the result.
    pub fn project_deeper(self, more_projections: &[PlaceElem<'tcx>], tcx: TyCtxt<'tcx>) -> Self {
        if more_projections.is_empty() {
            return self;
        }

        self.as_ref().project_deeper(more_projections, tcx)
    }
}

impl From<Local> for Place<'_> {
    #[inline]
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

    /// Returns `true` if this `Place` contains a `Deref` projection.
    ///
    /// If `Place::is_indirect` returns false, the caller knows that the `Place` refers to the
    /// same region of memory as its base.
    pub fn is_indirect(&self) -> bool {
        self.projection.iter().any(|elem| elem.is_indirect())
    }

    /// If MirPhase >= Derefered and if projection contains Deref,
    /// It's guaranteed to be in the first place
    pub fn has_deref(&self) -> bool {
        self.projection.first() == Some(&PlaceElem::Deref)
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
            (base, *proj)
        })
    }

    /// Generates a new place by appending `more_projections` to the existing ones
    /// and interning the result.
    pub fn project_deeper(
        self,
        more_projections: &[PlaceElem<'tcx>],
        tcx: TyCtxt<'tcx>,
    ) -> Place<'tcx> {
        let mut v: Vec<PlaceElem<'tcx>>;

        let new_projections = if self.projection.is_empty() {
            more_projections
        } else {
            v = Vec::with_capacity(self.projection.len() + more_projections.len());
            v.extend(self.projection);
            v.extend(more_projections);
            &v
        };

        Place { local: self.local, projection: tcx.mk_place_elems(new_projections) }
    }
}

impl Debug for Place<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        for elem in self.projection.iter().rev() {
            match elem {
                ProjectionElem::OpaqueCast(_)
                | ProjectionElem::Downcast(_, _)
                | ProjectionElem::Field(_, _) => {
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
                ProjectionElem::OpaqueCast(ty) => {
                    write!(fmt, " as {})", ty)?;
                }
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
    #[derive(HashStable)]
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
    pub lint_root: hir::HirId,
    /// The unsafe block that contains this node.
    pub safety: Safety,
}

///////////////////////////////////////////////////////////////////////////
// Operands

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
    /// with given `DefId` and args. Since this is used to synthesize
    /// MIR, assumes `user_ty` is None.
    pub fn function_handle(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        args: impl IntoIterator<Item = GenericArg<'tcx>>,
        span: Span,
    ) -> Self {
        let ty = Ty::new_fn_def(tcx, def_id, args);
        Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: ConstantKind::Val(ConstValue::ZeroSized, ty),
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
    pub fn const_fn_def(&self) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        let const_ty = self.constant()?.literal.ty();
        if let ty::FnDef(def_id, args) = *const_ty.kind() { Some((def_id, args)) } else { None }
    }
}

///////////////////////////////////////////////////////////////////////////
/// Rvalues

impl<'tcx> Rvalue<'tcx> {
    /// Returns true if rvalue can be safely removed when the result is unused.
    #[inline]
    pub fn is_safe_to_remove(&self) -> bool {
        match self {
            // Pointer to int casts may be side-effects due to exposing the provenance.
            // While the model is undecided, we should be conservative. See
            // <https://www.ralfj.de/blog/2022/04/11/provenance-exposed.html>
            Rvalue::Cast(CastKind::PointerExposeAddress, _, _) => false,

            Rvalue::Use(_)
            | Rvalue::CopyForDeref(_)
            | Rvalue::Repeat(_, _)
            | Rvalue::Ref(_, _, _)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::AddressOf(_, _)
            | Rvalue::Len(_)
            | Rvalue::Cast(
                CastKind::IntToInt
                | CastKind::FloatToInt
                | CastKind::FloatToFloat
                | CastKind::IntToFloat
                | CastKind::FnPtrToPtr
                | CastKind::PtrToPtr
                | CastKind::PointerCoercion(_)
                | CastKind::PointerFromExposedAddress
                | CastKind::DynStar
                | CastKind::Transmute,
                _,
                _,
            )
            | Rvalue::BinaryOp(_, _)
            | Rvalue::CheckedBinaryOp(_, _)
            | Rvalue::NullaryOp(_, _)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::Discriminant(_)
            | Rvalue::Aggregate(_, _)
            | Rvalue::ShallowInitBox(_, _) => true,
        }
    }
}

impl BorrowKind {
    pub fn mutability(&self) -> Mutability {
        match *self {
            BorrowKind::Shared | BorrowKind::Shallow => Mutability::Not,
            BorrowKind::Mut { .. } => Mutability::Mut,
        }
    }

    pub fn allows_two_phase_borrow(&self) -> bool {
        match *self {
            BorrowKind::Shared
            | BorrowKind::Shallow
            | BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::ClosureCapture } => {
                false
            }
            BorrowKind::Mut { kind: MutBorrowKind::TwoPhaseBorrow } => true,
        }
    }
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
            NullaryOp(ref op, ref t) => match op {
                NullOp::SizeOf => write!(fmt, "SizeOf({:?})", t),
                NullOp::AlignOf => write!(fmt, "AlignOf({:?})", t),
                NullOp::OffsetOf(fields) => write!(fmt, "OffsetOf({:?}, {:?})", t, fields),
            },
            ThreadLocalRef(did) => ty::tls::with(|tcx| {
                let muta = tcx.static_mutability(did).unwrap().prefix_str();
                write!(fmt, "&/*tls*/ {}{}", muta, tcx.def_path_str(did))
            }),
            Ref(region, borrow_kind, ref place) => {
                let kind_str = match borrow_kind {
                    BorrowKind::Shared => "",
                    BorrowKind::Shallow => "shallow ",
                    BorrowKind::Mut { .. } => "mut ",
                };

                // When printing regions, add trailing space if necessary.
                let print_region = ty::tls::with(|tcx| {
                    tcx.sess.verbose() || tcx.sess.opts.unstable_opts.identify_regions
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

            CopyForDeref(ref place) => write!(fmt, "deref_copy {:#?}", place),

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

                    AggregateKind::Adt(adt_did, variant, args, _user_ty, _) => {
                        ty::tls::with(|tcx| {
                            let variant_def = &tcx.adt_def(adt_did).variant(variant);
                            let args = tcx.lift(args).expect("could not lift for printing");
                            let name = FmtPrinter::new(tcx, Namespace::ValueNS)
                                .print_def_path(variant_def.def_id, args)?
                                .into_buffer();

                            match variant_def.ctor_kind() {
                                Some(CtorKind::Const) => fmt.write_str(&name),
                                Some(CtorKind::Fn) => fmt_tuple(fmt, &name),
                                None => {
                                    let mut struct_fmt = fmt.debug_struct(&name);
                                    for (field, place) in iter::zip(&variant_def.fields, places) {
                                        struct_fmt.field(field.name.as_str(), place);
                                    }
                                    struct_fmt.finish()
                                }
                            }
                        })
                    }

                    AggregateKind::Closure(def_id, args) => ty::tls::with(|tcx| {
                        let name = if tcx.sess.opts.unstable_opts.span_free_formats {
                            let args = tcx.lift(args).unwrap();
                            format!("[closure@{}]", tcx.def_path_str_with_args(def_id, args),)
                        } else {
                            let span = tcx.def_span(def_id);
                            format!(
                                "[closure@{}]",
                                tcx.sess.source_map().span_to_diagnostic_string(span)
                            )
                        };
                        let mut struct_fmt = fmt.debug_struct(&name);

                        // FIXME(project-rfc-2229#48): This should be a list of capture names/places
                        if let Some(def_id) = def_id.as_local()
                            && let Some(upvars) = tcx.upvars_mentioned(def_id)
                        {
                            for (&var_id, place) in iter::zip(upvars.keys(), places) {
                                let var_name = tcx.hir().name(var_id);
                                struct_fmt.field(var_name.as_str(), place);
                            }
                        } else {
                            for (index, place) in places.iter().enumerate() {
                                struct_fmt.field(&format!("{index}"), place);
                            }
                        }

                        struct_fmt.finish()
                    }),

                    AggregateKind::Generator(def_id, _, _) => ty::tls::with(|tcx| {
                        let name = format!("[generator@{:?}]", tcx.def_span(def_id));
                        let mut struct_fmt = fmt.debug_struct(&name);

                        // FIXME(project-rfc-2229#48): This should be a list of capture names/places
                        if let Some(def_id) = def_id.as_local()
                            && let Some(upvars) = tcx.upvars_mentioned(def_id)
                        {
                            for (&var_id, place) in iter::zip(upvars.keys(), places) {
                                let var_name = tcx.hir().name(var_id);
                                struct_fmt.field(var_name.as_str(), place);
                            }
                        } else {
                            for (index, place) in places.iter().enumerate() {
                                struct_fmt.field(&format!("{index}"), place);
                            }
                        }

                        struct_fmt.finish()
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
#[derive(TypeFoldable, TypeVisitable)]
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
#[derive(Lift, TypeFoldable, TypeVisitable)]
pub enum ConstantKind<'tcx> {
    /// This constant came from the type system
    Ty(ty::Const<'tcx>),

    /// An unevaluated mir constant which is not part of the type system.
    Unevaluated(UnevaluatedConst<'tcx>, Ty<'tcx>),

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

impl<'tcx> ConstantKind<'tcx> {
    #[inline(always)]
    pub fn ty(&self) -> Ty<'tcx> {
        match self {
            ConstantKind::Ty(c) => c.ty(),
            ConstantKind::Val(_, ty) | ConstantKind::Unevaluated(_, ty) => *ty,
        }
    }

    #[inline]
    pub fn try_to_value(self, tcx: TyCtxt<'tcx>) -> Option<interpret::ConstValue<'tcx>> {
        match self {
            ConstantKind::Ty(c) => match c.kind() {
                ty::ConstKind::Value(valtree) => Some(tcx.valtree_to_const_val((c.ty(), valtree))),
                _ => None,
            },
            ConstantKind::Val(val, _) => Some(val),
            ConstantKind::Unevaluated(..) => None,
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar> {
        match self {
            ConstantKind::Ty(c) => match c.kind() {
                ty::ConstKind::Value(valtree) => match valtree {
                    ty::ValTree::Leaf(scalar_int) => Some(Scalar::Int(scalar_int)),
                    ty::ValTree::Branch(_) => None,
                },
                _ => None,
            },
            ConstantKind::Val(val, _) => val.try_to_scalar(),
            ConstantKind::Unevaluated(..) => None,
        }
    }

    #[inline]
    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        Some(self.try_to_scalar()?.assert_int())
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
                if let Some(val) = c.try_eval_for_mir(tcx, param_env) {
                    match val {
                        Ok(val) => Self::Val(val, c.ty()),
                        Err(guar) => Self::Ty(ty::Const::new_error(tcx, guar, self.ty())),
                    }
                } else {
                    self
                }
            }
            Self::Val(_, _) => self,
            Self::Unevaluated(uneval, ty) => {
                // FIXME: We might want to have a `try_eval`-like function on `Unevaluated`
                match tcx.const_eval_resolve(param_env, uneval, None) {
                    Ok(val) => Self::Val(val, ty),
                    Err(ErrorHandled::TooGeneric) => self,
                    Err(ErrorHandled::Reported(guar)) => {
                        Self::Ty(ty::Const::new_error(tcx, guar.into(), ty))
                    }
                }
            }
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
            Self::Unevaluated(uneval, ty) => {
                match tcx.const_eval_resolve(param_env, *uneval, None) {
                    Ok(val) => {
                        let size = tcx
                            .layout_of(param_env.with_reveal_all_normalized(tcx).and(*ty))
                            .ok()?
                            .size;
                        val.try_to_bits(size)
                    }
                    Err(_) => None,
                }
            }
        }
    }

    #[inline]
    pub fn try_eval_bool(&self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Option<bool> {
        match self {
            Self::Ty(ct) => ct.try_eval_bool(tcx, param_env),
            Self::Val(val, _) => val.try_to_bool(),
            Self::Unevaluated(uneval, _) => {
                match tcx.const_eval_resolve(param_env, *uneval, None) {
                    Ok(val) => val.try_to_bool(),
                    Err(_) => None,
                }
            }
        }
    }

    #[inline]
    pub fn try_eval_target_usize(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<u64> {
        match self {
            Self::Ty(ct) => ct.try_eval_target_usize(tcx, param_env),
            Self::Val(val, _) => val.try_to_target_usize(tcx),
            Self::Unevaluated(uneval, _) => {
                match tcx.const_eval_resolve(param_env, *uneval, None) {
                    Ok(val) => val.try_to_target_usize(tcx),
                    Err(_) => None,
                }
            }
        }
    }

    #[inline]
    pub fn from_value(val: ConstValue<'tcx>, ty: Ty<'tcx>) -> Self {
        Self::Val(val, ty)
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

    #[inline]
    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        let cv = ConstValue::from_bool(v);
        Self::Val(cv, tcx.types.bool)
    }

    #[inline]
    pub fn zero_sized(ty: Ty<'tcx>) -> Self {
        let cv = ConstValue::ZeroSized;
        Self::Val(cv, ty)
    }

    pub fn from_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        let ty = tcx.types.usize;
        Self::from_bits(tcx, n as u128, ty::ParamEnv::empty().and(ty))
    }

    #[inline]
    pub fn from_scalar(_tcx: TyCtxt<'tcx>, s: Scalar, ty: Ty<'tcx>) -> Self {
        let val = ConstValue::Scalar(s);
        Self::Val(val, ty)
    }

    /// Literals are converted to `ConstantKindVal`, const generic parameters are eagerly
    /// converted to a constant, everything else becomes `Unevaluated`.
    #[instrument(skip(tcx), level = "debug", ret)]
    pub fn from_anon_const(
        tcx: TyCtxt<'tcx>,
        def: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let body_id = match tcx.hir().get_by_def_id(def) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => {
                span_bug!(tcx.def_span(def), "from_anon_const can only process anonymous constants")
            }
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
        debug!("expr.kind: {:?}", expr.kind);

        let ty = tcx.type_of(def).instantiate_identity();
        debug!(?ty);

        // FIXME(const_generics): We currently have to special case parameters because `min_const_generics`
        // does not provide the parents generics to anonymous constants. We still allow generic const
        // parameters by themselves however, e.g. `N`. These constants would cause an ICE if we were to
        // ever try to substitute the generic parameters in their bodies.
        //
        // While this doesn't happen as these constants are always used as `ty::ConstKind::Param`, it does
        // cause issues if we were to remove that special-case and try to evaluate the constant instead.
        use hir::{def::DefKind::ConstParam, def::Res, ExprKind, Path, QPath};
        match expr.kind {
            ExprKind::Path(QPath::Resolved(_, &Path { res: Res::Def(ConstParam, def_id), .. })) => {
                // Find the name and index of the const parameter by indexing the generics of
                // the parent item and construct a `ParamConst`.
                let item_def_id = tcx.parent(def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id];
                let name = tcx.item_name(def_id);
                let ty_const = ty::Const::new_param(tcx, ty::ParamConst::new(index, name), ty);
                debug!(?ty_const);

                return Self::Ty(ty_const);
            }
            _ => {}
        }

        let hir_id = tcx.hir().local_def_id_to_hir_id(def);
        let parent_args = if let Some(parent_hir_id) = tcx.hir().opt_parent_id(hir_id)
            && let Some(parent_did) = parent_hir_id.as_owner()
        {
            GenericArgs::identity_for_item(tcx, parent_did)
        } else {
            List::empty()
        };
        debug!(?parent_args);

        let did = def.to_def_id();
        let child_args = GenericArgs::identity_for_item(tcx, did);
        let args = tcx.mk_args_from_iter(parent_args.into_iter().chain(child_args.into_iter()));
        debug!(?args);

        let span = tcx.def_span(def);
        let uneval = UnevaluatedConst::new(did, args);
        debug!(?span, ?param_env);

        match tcx.const_eval_resolve(param_env, uneval, Some(span)) {
            Ok(val) => {
                debug!("evaluated const value");
                Self::Val(val, ty)
            }
            Err(_) => {
                debug!("error encountered during evaluation");
                // Error was handled in `const_eval_resolve`. Here we just create a
                // new unevaluated const and error hard later in codegen
                Self::Unevaluated(
                    UnevaluatedConst {
                        def: did,
                        args: GenericArgs::identity_for_item(tcx, did),
                        promoted: None,
                    },
                    ty,
                )
            }
        }
    }

    pub fn from_const(c: ty::Const<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        match c.kind() {
            ty::ConstKind::Value(valtree) => {
                let const_val = tcx.valtree_to_const_val((c.ty(), valtree));
                Self::Val(const_val, c.ty())
            }
            ty::ConstKind::Unevaluated(uv) => Self::Unevaluated(uv.expand(), c.ty()),
            _ => Self::Ty(c),
        }
    }
}

/// An unevaluated (potentially generic) constant used in MIR.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Lift)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct UnevaluatedConst<'tcx> {
    pub def: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub promoted: Option<Promoted>,
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn shrink(self) -> ty::UnevaluatedConst<'tcx> {
        assert_eq!(self.promoted, None);
        ty::UnevaluatedConst { def: self.def, args: self.args }
    }
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn new(def: DefId, args: GenericArgsRef<'tcx>) -> UnevaluatedConst<'tcx> {
        UnevaluatedConst { def, args, promoted: Default::default() }
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

    pub fn leaf(self, field: FieldIdx) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.leaf(field))
    }

    pub fn variant(
        self,
        adt_def: AdtDef<'tcx>,
        variant_index: VariantIdx,
        field_index: FieldIdx,
    ) -> Self {
        self.map_projections(|pat_ty_proj| pat_ty_proj.variant(adt_def, variant_index, field_index))
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

    pub(crate) fn leaf(mut self, field: FieldIdx) -> Self {
        self.projs.push(ProjectionElem::Field(field, ()));
        self
    }

    pub(crate) fn variant(
        mut self,
        adt_def: AdtDef<'_>,
        variant_index: VariantIdx,
        field_index: FieldIdx,
    ) -> Self {
        self.projs.push(ProjectionElem::Downcast(
            Some(adt_def.variant(variant_index).name),
            variant_index,
        ));
        self.projs.push(ProjectionElem::Field(field_index, ()));
        self
    }
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    #[debug_format = "promoted[{}]"]
    pub struct Promoted {}
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
            ConstantKind::Val(val, ty) => pretty_print_const_value(val, ty, fmt),
            // FIXME(valtrees): Correctly print mir constants.
            ConstantKind::Unevaluated(..) => {
                fmt.write_str("_")?;
                Ok(())
            }
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

fn pretty_print_byte_str(fmt: &mut Formatter<'_>, byte_str: &[u8]) -> fmt::Result {
    write!(fmt, "b\"{}\"", byte_str.escape_ascii())
}

fn comma_sep<'tcx>(
    fmt: &mut Formatter<'_>,
    elems: Vec<(ConstValue<'tcx>, Ty<'tcx>)>,
) -> fmt::Result {
    let mut first = true;
    for (ct, ty) in elems {
        if !first {
            fmt.write_str(", ")?;
        }
        pretty_print_const_value(ct, ty, fmt)?;
        first = false;
    }
    Ok(())
}

// FIXME: Move that into `mir/pretty.rs`.
fn pretty_print_const_value<'tcx>(
    ct: ConstValue<'tcx>,
    ty: Ty<'tcx>,
    fmt: &mut Formatter<'_>,
) -> fmt::Result {
    use crate::ty::print::PrettyPrinter;

    ty::tls::with(|tcx| {
        let ct = tcx.lift(ct).unwrap();
        let ty = tcx.lift(ty).unwrap();

        if tcx.sess.verbose() {
            fmt.write_str(&format!("ConstValue({:?}: {})", ct, ty))?;
            return Ok(());
        }

        let u8_type = tcx.types.u8;
        match (ct, ty.kind()) {
            // Byte/string slices, printed as (byte) string literals.
            (ConstValue::Slice { data, start, end }, ty::Ref(_, inner, _)) => {
                match inner.kind() {
                    ty::Slice(t) => {
                        if *t == u8_type {
                            // The `inspect` here is okay since we checked the bounds, and `u8` carries
                            // no provenance (we have an active slice reference here). We don't use
                            // this result to affect interpreter execution.
                            let byte_str = data
                                .inner()
                                .inspect_with_uninit_and_ptr_outside_interpreter(start..end);
                            pretty_print_byte_str(fmt, byte_str)?;
                            return Ok(());
                        }
                    }
                    ty::Str => {
                        // The `inspect` here is okay since we checked the bounds, and `str` carries
                        // no provenance (we have an active `str` reference here). We don't use this
                        // result to affect interpreter execution.
                        let slice = data
                            .inner()
                            .inspect_with_uninit_and_ptr_outside_interpreter(start..end);
                        fmt.write_str(&format!("{:?}", String::from_utf8_lossy(slice)))?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
            (ConstValue::ByRef { alloc, offset }, ty::Array(t, n)) if *t == u8_type => {
                let n = n.try_to_bits(tcx.data_layout.pointer_size).unwrap();
                // cast is ok because we already checked for pointer size (32 or 64 bit) above
                let range = AllocRange { start: offset, size: Size::from_bytes(n) };
                let byte_str = alloc.inner().get_bytes_strip_provenance(&tcx, range).unwrap();
                fmt.write_str("*")?;
                pretty_print_byte_str(fmt, byte_str)?;
                return Ok(());
            }
            // Aggregates, printed as array/tuple/struct/variant construction syntax.
            //
            // NB: the `has_non_region_param` check ensures that we can use
            // the `destructure_const` query with an empty `ty::ParamEnv` without
            // introducing ICEs (e.g. via `layout_of`) from missing bounds.
            // E.g. `transmute([0usize; 2]): (u8, *mut T)` needs to know `T: Sized`
            // to be able to destructure the tuple into `(0u8, *mut T)`
            (_, ty::Array(..) | ty::Tuple(..) | ty::Adt(..)) if !ty.has_non_region_param() => {
                let ct = tcx.lift(ct).unwrap();
                let ty = tcx.lift(ty).unwrap();
                if let Some(contents) = tcx.try_destructure_mir_constant_for_diagnostics((ct, ty)) {
                    let fields: Vec<(ConstValue<'_>, Ty<'_>)> = contents.fields.to_vec();
                    match *ty.kind() {
                        ty::Array(..) => {
                            fmt.write_str("[")?;
                            comma_sep(fmt, fields)?;
                            fmt.write_str("]")?;
                        }
                        ty::Tuple(..) => {
                            fmt.write_str("(")?;
                            comma_sep(fmt, fields)?;
                            if contents.fields.len() == 1 {
                                fmt.write_str(",")?;
                            }
                            fmt.write_str(")")?;
                        }
                        ty::Adt(def, _) if def.variants().is_empty() => {
                            fmt.write_str(&format!("{{unreachable(): {}}}", ty))?;
                        }
                        ty::Adt(def, args) => {
                            let variant_idx = contents
                                .variant
                                .expect("destructed mir constant of adt without variant idx");
                            let variant_def = &def.variant(variant_idx);
                            let args = tcx.lift(args).unwrap();
                            let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                            cx.print_alloc_ids = true;
                            let cx = cx.print_value_path(variant_def.def_id, args)?;
                            fmt.write_str(&cx.into_buffer())?;

                            match variant_def.ctor_kind() {
                                Some(CtorKind::Const) => {}
                                Some(CtorKind::Fn) => {
                                    fmt.write_str("(")?;
                                    comma_sep(fmt, fields)?;
                                    fmt.write_str(")")?;
                                }
                                None => {
                                    fmt.write_str(" {{ ")?;
                                    let mut first = true;
                                    for (field_def, (ct, ty)) in
                                        iter::zip(&variant_def.fields, fields)
                                    {
                                        if !first {
                                            fmt.write_str(", ")?;
                                        }
                                        write!(fmt, "{}: ", field_def.name)?;
                                        pretty_print_const_value(ct, ty, fmt)?;
                                        first = false;
                                    }
                                    fmt.write_str(" }}")?;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                    return Ok(());
                }
            }
            (ConstValue::Scalar(scalar), _) => {
                let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                cx.print_alloc_ids = true;
                let ty = tcx.lift(ty).unwrap();
                cx = cx.pretty_print_const_scalar(scalar, ty)?;
                fmt.write_str(&cx.into_buffer())?;
                return Ok(());
            }
            (ConstValue::ZeroSized, ty::FnDef(d, s)) => {
                let mut cx = FmtPrinter::new(tcx, Namespace::ValueNS);
                cx.print_alloc_ids = true;
                let cx = cx.print_value_path(*d, s)?;
                fmt.write_str(&cx.into_buffer())?;
                return Ok(());
            }
            // FIXME(oli-obk): also pretty print arrays and other aggregate constants by reading
            // their fields instead of just dumping the memory.
            _ => {}
        }
        // Fall back to debug pretty printing for invalid constants.
        write!(fmt, "{ct:?}: {ty}")
    })
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

    pub fn dominates(&self, other: Location, dominators: &Dominators<BasicBlock>) -> bool {
        if self.block == other.block {
            self.statement_index <= other.statement_index
        } else {
            dominators.dominates(self.block, other.block)
        }
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(BasicBlockData<'_>, 136);
    static_assert_size!(LocalDecl<'_>, 40);
    static_assert_size!(SourceScopeData<'_>, 72);
    static_assert_size!(Statement<'_>, 32);
    static_assert_size!(StatementKind<'_>, 16);
    static_assert_size!(Terminator<'_>, 104);
    static_assert_size!(TerminatorKind<'_>, 88);
    static_assert_size!(VarDebugInfo<'_>, 80);
    // tidy-alphabetical-end
}
