//! This module used to be named `build`, but that was causing GitHub's
//! "Go to file" feature to silently ignore all files in the module, probably
//! because it assumes that "build" is a build-output directory.
//! See <https://github.com/rust-lang/rust/pull/134365>.

use itertools::Itertools;
use rustc_abi::{ExternAbi, FieldIdx};
use rustc_apfloat::Float;
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_ast::attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sorted_map::SortedIndexMultiMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, BindingMode, ByRef, HirId, ItemLocalId, Node};
use rustc_index::bit_set::GrowableBitSet;
use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_middle::hir::place::PlaceBase as HirPlaceBase;
use rustc_middle::middle::region;
use rustc_middle::mir::*;
use rustc_middle::thir::{self, ExprId, LintLevel, LocalVarId, Param, ParamId, PatKind, Thir};
use rustc_middle::ty::{self, ScalarInt, Ty, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_middle::{bug, span_bug};
use rustc_span::{Span, Symbol, sym};

use crate::builder::expr::as_place::PlaceBuilder;
use crate::builder::scope::DropKind;

pub(crate) fn closure_saved_names_of_captured_variables<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> IndexVec<FieldIdx, Symbol> {
    tcx.closure_captures(def_id)
        .iter()
        .map(|captured_place| {
            let name = captured_place.to_symbol();
            match captured_place.info.capture_kind {
                ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => name,
                ty::UpvarCapture::ByRef(..) => Symbol::intern(&format!("_ref__{name}")),
            }
        })
        .collect()
}

/// Create the MIR for a given `DefId`, including unreachable code. Do not call
/// this directly; instead use the cached version via `mir_built`.
pub fn build_mir<'tcx>(tcx: TyCtxt<'tcx>, def: LocalDefId) -> Body<'tcx> {
    tcx.ensure_done().thir_abstract_const(def);
    if let Err(e) = tcx.ensure_ok().check_match(def) {
        return construct_error(tcx, def, e);
    }

    if let Err(err) = tcx.ensure_ok().check_tail_calls(def) {
        return construct_error(tcx, def, err);
    }

    let body = match tcx.thir_body(def) {
        Err(error_reported) => construct_error(tcx, def, error_reported),
        Ok((thir, expr)) => {
            let build_mir = |thir: &Thir<'tcx>| match thir.body_type {
                thir::BodyTy::Fn(fn_sig) => construct_fn(tcx, def, thir, expr, fn_sig),
                thir::BodyTy::Const(ty) | thir::BodyTy::GlobalAsm(ty) => {
                    construct_const(tcx, def, thir, expr, ty)
                }
            };

            // this must run before MIR dump, because
            // "not all control paths return a value" is reported here.
            //
            // maybe move the check to a MIR pass?
            tcx.ensure_ok().check_liveness(def);

            // Don't steal here, instead steal in unsafeck. This is so that
            // pattern inline constants can be evaluated as part of building the
            // THIR of the parent function without a cycle.
            build_mir(&thir.borrow())
        }
    };

    // The borrow checker will replace all the regions here with its own
    // inference variables. There's no point having non-erased regions here.
    // The exception is `body.user_type_annotations`, which is used unmodified
    // by borrow checking.
    debug_assert!(
        !(body.local_decls.has_free_regions()
            || body.basic_blocks.has_free_regions()
            || body.var_debug_info.has_free_regions()
            || body.yield_ty().has_free_regions()),
        "Unexpected free regions in MIR: {body:?}",
    );

    body
}

///////////////////////////////////////////////////////////////////////////
// BuildMir -- walks a crate, looking for fn items and methods to build MIR from

#[derive(Debug, PartialEq, Eq)]
enum BlockFrame {
    /// Evaluation is currently within a statement.
    ///
    /// Examples include:
    /// 1. `EXPR;`
    /// 2. `let _ = EXPR;`
    /// 3. `let x = EXPR;`
    Statement {
        /// If true, then statement discards result from evaluating
        /// the expression (such as examples 1 and 2 above).
        ignores_expr_result: bool,
    },

    /// Evaluation is currently within the tail expression of a block.
    ///
    /// Example: `{ STMT_1; STMT_2; EXPR }`
    TailExpr { info: BlockTailInfo },

    /// Generic mark meaning that the block occurred as a subexpression
    /// where the result might be used.
    ///
    /// Examples: `foo(EXPR)`, `match EXPR { ... }`
    SubExpr,
}

impl BlockFrame {
    fn is_tail_expr(&self) -> bool {
        match *self {
            BlockFrame::TailExpr { .. } => true,

            BlockFrame::Statement { .. } | BlockFrame::SubExpr => false,
        }
    }
    fn is_statement(&self) -> bool {
        match *self {
            BlockFrame::Statement { .. } => true,

            BlockFrame::TailExpr { .. } | BlockFrame::SubExpr => false,
        }
    }
}

#[derive(Debug)]
struct BlockContext(Vec<BlockFrame>);

struct Builder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    // FIXME(@lcnr): Why does this use an `infcx`, there should be
    // no shared type inference going on here. I feel like it would
    // clearer to manually construct one where necessary or to provide
    // a nice API for non-type inference trait system checks.
    infcx: InferCtxt<'tcx>,
    region_scope_tree: &'tcx region::ScopeTree,
    param_env: ty::ParamEnv<'tcx>,

    thir: &'a Thir<'tcx>,
    cfg: CFG<'tcx>,

    def_id: LocalDefId,
    hir_id: HirId,
    parent_module: DefId,
    check_overflow: bool,
    fn_span: Span,
    arg_count: usize,
    coroutine: Option<Box<CoroutineInfo<'tcx>>>,

    /// The current set of scopes, updated as we traverse;
    /// see the `scope` module for more details.
    scopes: scope::Scopes<'tcx>,

    /// The block-context: each time we build the code within an thir::Block,
    /// we push a frame here tracking whether we are building a statement or
    /// if we are pushing the tail expression of the block. This is used to
    /// embed information in generated temps about whether they were created
    /// for a block tail expression or not.
    ///
    /// It would be great if we could fold this into `self.scopes`
    /// somehow, but right now I think that is very tightly tied to
    /// the code generation in ways that we cannot (or should not)
    /// start just throwing new entries onto that vector in order to
    /// distinguish the context of EXPR1 from the context of EXPR2 in
    /// `{ STMTS; EXPR1 } + EXPR2`.
    block_context: BlockContext,

    /// The vector of all scopes that we have created thus far;
    /// we track this for debuginfo later.
    source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,
    source_scope: SourceScope,

    /// The guard-context: each time we build the guard expression for
    /// a match arm, we push onto this stack, and then pop when we
    /// finish building it.
    guard_context: Vec<GuardFrame>,

    /// Temporaries with fixed indexes. Used so that if-let guards on arms
    /// with an or-pattern are only created once.
    fixed_temps: FxHashMap<ExprId, Local>,
    /// Scope of temporaries that should be deduplicated using [Self::fixed_temps].
    fixed_temps_scope: Option<region::Scope>,

    /// Maps `HirId`s of variable bindings to the `Local`s created for them.
    /// (A match binding can have two locals; the 2nd is for the arm's guard.)
    var_indices: FxHashMap<LocalVarId, LocalsForNode>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    canonical_user_type_annotations: ty::CanonicalUserTypeAnnotations<'tcx>,
    upvars: CaptureMap<'tcx>,
    unit_temp: Option<Place<'tcx>>,

    var_debug_info: Vec<VarDebugInfo<'tcx>>,

    // A cache for `maybe_lint_level_roots_bounded`. That function is called
    // repeatedly, and each time it effectively traces a path through a tree
    // structure from a node towards the root, doing an attribute check on each
    // node along the way. This cache records which nodes trace all the way to
    // the root (most of them do) and saves us from retracing many sub-paths
    // many times, and rechecking many nodes.
    lint_level_roots_cache: GrowableBitSet<hir::ItemLocalId>,

    /// Collects additional coverage information during MIR building.
    /// Only present if coverage is enabled and this function is eligible.
    coverage_info: Option<coverageinfo::CoverageInfoBuilder>,
}

type CaptureMap<'tcx> = SortedIndexMultiMap<usize, ItemLocalId, Capture<'tcx>>;

#[derive(Debug)]
struct Capture<'tcx> {
    captured_place: &'tcx ty::CapturedPlace<'tcx>,
    use_place: Place<'tcx>,
    mutability: Mutability,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.infcx.typing_env(self.param_env)
    }

    fn is_bound_var_in_guard(&self, id: LocalVarId) -> bool {
        self.guard_context.iter().any(|frame| frame.locals.iter().any(|local| local.id == id))
    }

    fn var_local_id(&self, id: LocalVarId, for_guard: ForGuard) -> Local {
        self.var_indices[&id].local_id(for_guard)
    }
}

impl BlockContext {
    fn new() -> Self {
        BlockContext(vec![])
    }
    fn push(&mut self, bf: BlockFrame) {
        self.0.push(bf);
    }
    fn pop(&mut self) -> Option<BlockFrame> {
        self.0.pop()
    }

    /// Traverses the frames on the `BlockContext`, searching for either
    /// the first block-tail expression frame with no intervening
    /// statement frame.
    ///
    /// Notably, this skips over `SubExpr` frames; this method is
    /// meant to be used in the context of understanding the
    /// relationship of a temp (created within some complicated
    /// expression) with its containing expression, and whether the
    /// value of that *containing expression* (not the temp!) is
    /// ignored.
    fn currently_in_block_tail(&self) -> Option<BlockTailInfo> {
        for bf in self.0.iter().rev() {
            match bf {
                BlockFrame::SubExpr => continue,
                BlockFrame::Statement { .. } => break,
                &BlockFrame::TailExpr { info } => return Some(info),
            }
        }

        None
    }

    /// Looks at the topmost frame on the BlockContext and reports
    /// whether its one that would discard a block tail result.
    ///
    /// Unlike `currently_within_ignored_tail_expression`, this does
    /// *not* skip over `SubExpr` frames: here, we want to know
    /// whether the block result itself is discarded.
    fn currently_ignores_tail_results(&self) -> bool {
        match self.0.last() {
            // no context: conservatively assume result is read
            None => false,

            // sub-expression: block result feeds into some computation
            Some(BlockFrame::SubExpr) => false,

            // otherwise: use accumulated is_ignored state.
            Some(
                BlockFrame::TailExpr { info: BlockTailInfo { tail_result_is_ignored: ign, .. } }
                | BlockFrame::Statement { ignores_expr_result: ign },
            ) => *ign,
        }
    }
}

#[derive(Debug)]
enum LocalsForNode {
    /// In the usual case, a `HirId` for an identifier maps to at most
    /// one `Local` declaration.
    One(Local),

    /// The exceptional case is identifiers in a match arm's pattern
    /// that are referenced in a guard of that match arm. For these,
    /// we have `2` Locals.
    ///
    /// * `for_arm_body` is the Local used in the arm body (which is
    ///   just like the `One` case above),
    ///
    /// * `ref_for_guard` is the Local used in the arm's guard (which
    ///   is a reference to a temp that is an alias of
    ///   `for_arm_body`).
    ForGuard { ref_for_guard: Local, for_arm_body: Local },
}

#[derive(Debug)]
struct GuardFrameLocal {
    id: LocalVarId,
}

impl GuardFrameLocal {
    fn new(id: LocalVarId) -> Self {
        GuardFrameLocal { id }
    }
}

#[derive(Debug)]
struct GuardFrame {
    /// These are the id's of names that are bound by patterns of the
    /// arm of *this* guard.
    ///
    /// (Frames higher up the stack will have the id's bound in arms
    /// further out, such as in a case like:
    ///
    /// match E1 {
    ///      P1(id1) if (... (match E2 { P2(id2) if ... => B2 })) => B1,
    /// }
    ///
    /// here, when building for FIXME.
    locals: Vec<GuardFrameLocal>,
}

/// `ForGuard` indicates whether we are talking about:
///   1. The variable for use outside of guard expressions, or
///   2. The temp that holds reference to (1.), which is actually what the
///      guard expressions see.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ForGuard {
    RefWithinGuard,
    OutsideGuard,
}

impl LocalsForNode {
    fn local_id(&self, for_guard: ForGuard) -> Local {
        match (self, for_guard) {
            (&LocalsForNode::One(local_id), ForGuard::OutsideGuard)
            | (
                &LocalsForNode::ForGuard { ref_for_guard: local_id, .. },
                ForGuard::RefWithinGuard,
            )
            | (&LocalsForNode::ForGuard { for_arm_body: local_id, .. }, ForGuard::OutsideGuard) => {
                local_id
            }

            (&LocalsForNode::One(_), ForGuard::RefWithinGuard) => {
                bug!("anything with one local should never be within a guard.")
            }
        }
    }
}

struct CFG<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
}

rustc_index::newtype_index! {
    struct ScopeId {}
}

#[derive(Debug)]
enum NeedsTemporary {
    /// Use this variant when whatever you are converting with `as_operand`
    /// is the last thing you are converting. This means that if we introduced
    /// an intermediate temporary, we'd only read it immediately after, so we can
    /// also avoid it.
    No,
    /// For all cases where you aren't sure or that are too expensive to compute
    /// for now. It is always safe to fall back to this.
    Maybe,
}

///////////////////////////////////////////////////////////////////////////
/// The `BlockAnd` "monad" packages up the new basic block along with a
/// produced value (sometimes just unit, of course). The `unpack!`
/// macro (and methods below) makes working with `BlockAnd` much more
/// convenient.

#[must_use = "if you don't use one of these results, you're leaving a dangling edge"]
struct BlockAnd<T>(BasicBlock, T);

impl BlockAnd<()> {
    /// Unpacks `BlockAnd<()>` into a [`BasicBlock`].
    #[must_use]
    fn into_block(self) -> BasicBlock {
        let Self(block, ()) = self;
        block
    }
}

trait BlockAndExtension {
    fn and<T>(self, v: T) -> BlockAnd<T>;
    fn unit(self) -> BlockAnd<()>;
}

impl BlockAndExtension for BasicBlock {
    fn and<T>(self, v: T) -> BlockAnd<T> {
        BlockAnd(self, v)
    }

    fn unit(self) -> BlockAnd<()> {
        BlockAnd(self, ())
    }
}

/// Update a block pointer and return the value.
/// Use it like `let x = unpack!(block = self.foo(block, foo))`.
macro_rules! unpack {
    ($x:ident = $c:expr) => {{
        let BlockAnd(b, v) = $c;
        $x = b;
        v
    }};
}

///////////////////////////////////////////////////////////////////////////
/// the main entry point for building MIR for a function

fn construct_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_def: LocalDefId,
    thir: &Thir<'tcx>,
    expr: ExprId,
    fn_sig: ty::FnSig<'tcx>,
) -> Body<'tcx> {
    let span = tcx.def_span(fn_def);
    let fn_id = tcx.local_def_id_to_hir_id(fn_def);

    // The representation of thir for `-Zunpretty=thir-tree` relies on
    // the entry expression being the last element of `thir.exprs`.
    assert_eq!(expr.as_usize(), thir.exprs.len() - 1);

    // Figure out what primary body this item has.
    let body = tcx.hir_body_owned_by(fn_def);
    let span_with_body = tcx.hir_span_with_body(fn_id);
    let return_ty_span = tcx
        .hir_fn_decl_by_hir_id(fn_id)
        .unwrap_or_else(|| span_bug!(span, "can't build MIR for {:?}", fn_def))
        .output
        .span();

    let mut abi = fn_sig.abi;
    if let DefKind::Closure = tcx.def_kind(fn_def) {
        // HACK(eddyb) Avoid having RustCall on closures,
        // as it adds unnecessary (and wrong) auto-tupling.
        abi = ExternAbi::Rust;
    }

    let arguments = &thir.params;

    let return_ty = fn_sig.output();
    let coroutine = match tcx.type_of(fn_def).instantiate_identity().kind() {
        ty::Coroutine(_, args) => Some(Box::new(CoroutineInfo::initial(
            tcx.coroutine_kind(fn_def).unwrap(),
            args.as_coroutine().yield_ty(),
            args.as_coroutine().resume_ty(),
        ))),
        ty::Closure(..) | ty::CoroutineClosure(..) | ty::FnDef(..) => None,
        ty => span_bug!(span_with_body, "unexpected type of body: {ty:?}"),
    };

    if let Some(custom_mir_attr) =
        tcx.hir_attrs(fn_id).iter().find(|attr| attr.has_name(sym::custom_mir))
    {
        return custom::build_custom_mir(
            tcx,
            fn_def.to_def_id(),
            fn_id,
            thir,
            expr,
            arguments,
            return_ty,
            return_ty_span,
            span_with_body,
            custom_mir_attr,
        );
    }

    // FIXME(#132279): This should be able to reveal opaque
    // types defined during HIR typeck.
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let mut builder = Builder::new(
        thir,
        infcx,
        fn_def,
        fn_id,
        span_with_body,
        arguments.len(),
        return_ty,
        return_ty_span,
        coroutine,
    );

    let call_site_scope =
        region::Scope { local_id: body.id().hir_id.local_id, data: region::ScopeData::CallSite };
    let arg_scope =
        region::Scope { local_id: body.id().hir_id.local_id, data: region::ScopeData::Arguments };
    let source_info = builder.source_info(span);
    let call_site_s = (call_site_scope, source_info);
    let _: BlockAnd<()> = builder.in_scope(call_site_s, LintLevel::Inherited, |builder| {
        let arg_scope_s = (arg_scope, source_info);
        // Attribute epilogue to function's closing brace
        let fn_end = span_with_body.shrink_to_hi();
        let return_block = builder
            .in_breakable_scope(None, Place::return_place(), fn_end, |builder| {
                Some(builder.in_scope(arg_scope_s, LintLevel::Inherited, |builder| {
                    builder.args_and_body(START_BLOCK, arguments, arg_scope, expr)
                }))
            })
            .into_block();
        let source_info = builder.source_info(fn_end);
        builder.cfg.terminate(return_block, source_info, TerminatorKind::Return);
        builder.build_drop_trees();
        return_block.unit()
    });

    let mut body = builder.finish();

    body.spread_arg = if abi == ExternAbi::RustCall {
        // RustCall pseudo-ABI untuples the last argument.
        Some(Local::new(arguments.len()))
    } else {
        None
    };

    body
}

fn construct_const<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    thir: &'a Thir<'tcx>,
    expr: ExprId,
    const_ty: Ty<'tcx>,
) -> Body<'tcx> {
    let hir_id = tcx.local_def_id_to_hir_id(def);

    // Figure out what primary body this item has.
    let (span, const_ty_span) = match tcx.hir_node(hir_id) {
        Node::Item(hir::Item {
            kind: hir::ItemKind::Static(_, ty, _, _) | hir::ItemKind::Const(_, ty, _, _),
            span,
            ..
        })
        | Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Const(ty, _), span, .. })
        | Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Const(ty, Some(_)),
            span,
            ..
        }) => (*span, ty.span),
        Node::AnonConst(ct) => (ct.span, ct.span),
        Node::ConstBlock(_) => {
            let span = tcx.def_span(def);
            (span, span)
        }
        Node::Item(hir::Item { kind: hir::ItemKind::GlobalAsm { .. }, span, .. }) => (*span, *span),
        _ => span_bug!(tcx.def_span(def), "can't build MIR for {:?}", def),
    };

    // FIXME(#132279): We likely want to be able to use the hidden types of
    // opaques used by this function here.
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let mut builder =
        Builder::new(thir, infcx, def, hir_id, span, 0, const_ty, const_ty_span, None);

    let mut block = START_BLOCK;
    block = builder.expr_into_dest(Place::return_place(), block, expr).into_block();

    let source_info = builder.source_info(span);
    builder.cfg.terminate(block, source_info, TerminatorKind::Return);

    builder.build_drop_trees();

    builder.finish()
}

/// Construct MIR for an item that has had errors in type checking.
///
/// This is required because we may still want to run MIR passes on an item
/// with type errors, but normal MIR construction can't handle that in general.
fn construct_error(tcx: TyCtxt<'_>, def_id: LocalDefId, guar: ErrorGuaranteed) -> Body<'_> {
    let span = tcx.def_span(def_id);
    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    let (inputs, output, coroutine) = match tcx.def_kind(def_id) {
        DefKind::Const
        | DefKind::AssocConst
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::Static { .. }
        | DefKind::GlobalAsm => (vec![], tcx.type_of(def_id).instantiate_identity(), None),
        DefKind::Ctor(..) | DefKind::Fn | DefKind::AssocFn => {
            let sig = tcx.liberate_late_bound_regions(
                def_id.to_def_id(),
                tcx.fn_sig(def_id).instantiate_identity(),
            );
            (sig.inputs().to_vec(), sig.output(), None)
        }
        DefKind::Closure => {
            let closure_ty = tcx.type_of(def_id).instantiate_identity();
            match closure_ty.kind() {
                ty::Closure(_, args) => {
                    let args = args.as_closure();
                    let sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), args.sig());
                    let self_ty = match args.kind() {
                        ty::ClosureKind::Fn => {
                            Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, closure_ty)
                        }
                        ty::ClosureKind::FnMut => {
                            Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, closure_ty)
                        }
                        ty::ClosureKind::FnOnce => closure_ty,
                    };
                    (
                        [self_ty].into_iter().chain(sig.inputs()[0].tuple_fields()).collect(),
                        sig.output(),
                        None,
                    )
                }
                ty::Coroutine(_, args) => {
                    let args = args.as_coroutine();
                    let resume_ty = args.resume_ty();
                    let yield_ty = args.yield_ty();
                    let return_ty = args.return_ty();
                    (
                        vec![closure_ty, resume_ty],
                        return_ty,
                        Some(Box::new(CoroutineInfo::initial(
                            tcx.coroutine_kind(def_id).unwrap(),
                            yield_ty,
                            resume_ty,
                        ))),
                    )
                }
                ty::CoroutineClosure(did, args) => {
                    let args = args.as_coroutine_closure();
                    let sig = tcx.liberate_late_bound_regions(
                        def_id.to_def_id(),
                        args.coroutine_closure_sig(),
                    );
                    let self_ty = match args.kind() {
                        ty::ClosureKind::Fn => {
                            Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, closure_ty)
                        }
                        ty::ClosureKind::FnMut => {
                            Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, closure_ty)
                        }
                        ty::ClosureKind::FnOnce => closure_ty,
                    };
                    (
                        [self_ty].into_iter().chain(sig.tupled_inputs_ty.tuple_fields()).collect(),
                        sig.to_coroutine(
                            tcx,
                            args.parent_args(),
                            args.kind_ty(),
                            tcx.coroutine_for_closure(*did),
                            Ty::new_error(tcx, guar),
                        ),
                        None,
                    )
                }
                ty::Error(_) => (vec![closure_ty, closure_ty], closure_ty, None),
                kind => {
                    span_bug!(
                        span,
                        "expected type of closure body to be a closure or coroutine, got {kind:?}"
                    );
                }
            }
        }
        dk => span_bug!(span, "{:?} is not a body: {:?}", def_id, dk),
    };

    let source_info = SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE };
    let local_decls = IndexVec::from_iter(
        [output].iter().chain(&inputs).map(|ty| LocalDecl::with_source_info(*ty, source_info)),
    );
    let mut cfg = CFG { basic_blocks: IndexVec::new() };
    let mut source_scopes = IndexVec::new();

    cfg.start_new_block();
    source_scopes.push(SourceScopeData {
        span,
        parent_scope: None,
        inlined: None,
        inlined_parent_scope: None,
        local_data: ClearCrossCrate::Set(SourceScopeLocalData { lint_root: hir_id }),
    });

    cfg.terminate(START_BLOCK, source_info, TerminatorKind::Unreachable);

    Body::new(
        MirSource::item(def_id.to_def_id()),
        cfg.basic_blocks,
        source_scopes,
        local_decls,
        IndexVec::new(),
        inputs.len(),
        vec![],
        span,
        coroutine,
        Some(guar),
    )
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    fn new(
        thir: &'a Thir<'tcx>,
        infcx: InferCtxt<'tcx>,
        def: LocalDefId,
        hir_id: HirId,
        span: Span,
        arg_count: usize,
        return_ty: Ty<'tcx>,
        return_span: Span,
        coroutine: Option<Box<CoroutineInfo<'tcx>>>,
    ) -> Builder<'a, 'tcx> {
        let tcx = infcx.tcx;
        let attrs = tcx.hir_attrs(hir_id);
        // Some functions always have overflow checks enabled,
        // however, they may not get codegen'd, depending on
        // the settings for the crate they are codegened in.
        let mut check_overflow = attr::contains_name(attrs, sym::rustc_inherit_overflow_checks);
        // Respect -C overflow-checks.
        check_overflow |= tcx.sess.overflow_checks();
        // Constants always need overflow checks.
        check_overflow |= matches!(
            tcx.hir_body_owner_kind(def),
            hir::BodyOwnerKind::Const { .. } | hir::BodyOwnerKind::Static(_)
        );

        let lint_level = LintLevel::Explicit(hir_id);
        let param_env = tcx.param_env(def);
        let mut builder = Builder {
            thir,
            tcx,
            infcx,
            region_scope_tree: tcx.region_scope_tree(def),
            param_env,
            def_id: def,
            hir_id,
            parent_module: tcx.parent_module(hir_id).to_def_id(),
            check_overflow,
            cfg: CFG { basic_blocks: IndexVec::new() },
            fn_span: span,
            arg_count,
            coroutine,
            scopes: scope::Scopes::new(),
            block_context: BlockContext::new(),
            source_scopes: IndexVec::new(),
            source_scope: OUTERMOST_SOURCE_SCOPE,
            guard_context: vec![],
            fixed_temps: Default::default(),
            fixed_temps_scope: None,
            local_decls: IndexVec::from_elem_n(LocalDecl::new(return_ty, return_span), 1),
            canonical_user_type_annotations: IndexVec::new(),
            upvars: CaptureMap::new(),
            var_indices: Default::default(),
            unit_temp: None,
            var_debug_info: vec![],
            lint_level_roots_cache: GrowableBitSet::new_empty(),
            coverage_info: coverageinfo::CoverageInfoBuilder::new_if_enabled(tcx, def),
        };

        assert_eq!(builder.cfg.start_new_block(), START_BLOCK);
        assert_eq!(builder.new_source_scope(span, lint_level), OUTERMOST_SOURCE_SCOPE);
        builder.source_scopes[OUTERMOST_SOURCE_SCOPE].parent_scope = None;

        builder
    }

    fn finish(self) -> Body<'tcx> {
        let mut body = Body::new(
            MirSource::item(self.def_id.to_def_id()),
            self.cfg.basic_blocks,
            self.source_scopes,
            self.local_decls,
            self.canonical_user_type_annotations,
            self.arg_count,
            self.var_debug_info,
            self.fn_span,
            self.coroutine,
            None,
        );
        body.coverage_info_hi = self.coverage_info.map(|b| b.into_done());

        for (index, block) in body.basic_blocks.iter().enumerate() {
            if block.terminator.is_none() {
                use rustc_middle::mir::pretty;
                let options = pretty::PrettyPrintMirOptions::from_cli(self.tcx);
                pretty::write_mir_fn(
                    self.tcx,
                    &body,
                    &mut |_, _| Ok(()),
                    &mut std::io::stdout(),
                    options,
                )
                .unwrap();
                span_bug!(self.fn_span, "no terminator on block {:?}", index);
            }
        }

        body
    }

    fn insert_upvar_arg(&mut self) {
        let Some(closure_arg) = self.local_decls.get(ty::CAPTURE_STRUCT_LOCAL) else { return };

        let mut closure_ty = closure_arg.ty;
        let mut closure_env_projs = vec![];
        if let ty::Ref(_, ty, _) = closure_ty.kind() {
            closure_env_projs.push(ProjectionElem::Deref);
            closure_ty = *ty;
        }

        let upvar_args = match closure_ty.kind() {
            ty::Closure(_, args) => ty::UpvarArgs::Closure(args),
            ty::Coroutine(_, args) => ty::UpvarArgs::Coroutine(args),
            ty::CoroutineClosure(_, args) => ty::UpvarArgs::CoroutineClosure(args),
            _ => return,
        };

        // In analyze_closure() in upvar.rs we gathered a list of upvars used by an
        // indexed closure and we stored in a map called closure_min_captures in TypeckResults
        // with the closure's DefId. Here, we run through that vec of UpvarIds for
        // the given closure and use the necessary information to create upvar
        // debuginfo and to fill `self.upvars`.
        let capture_tys = upvar_args.upvar_tys();

        let tcx = self.tcx;
        let mut upvar_owner = None;
        self.upvars = tcx
            .closure_captures(self.def_id)
            .iter()
            .zip_eq(capture_tys)
            .enumerate()
            .map(|(i, (captured_place, ty))| {
                let name = captured_place.to_symbol();

                let capture = captured_place.info.capture_kind;
                let var_id = match captured_place.place.base {
                    HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
                    _ => bug!("Expected an upvar"),
                };
                let upvar_base = upvar_owner.get_or_insert(var_id.owner);
                assert_eq!(*upvar_base, var_id.owner);
                let var_id = var_id.local_id;

                let mutability = captured_place.mutability;

                let mut projs = closure_env_projs.clone();
                projs.push(ProjectionElem::Field(FieldIdx::new(i), ty));
                match capture {
                    ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => {}
                    ty::UpvarCapture::ByRef(..) => {
                        projs.push(ProjectionElem::Deref);
                    }
                };

                let use_place = Place {
                    local: ty::CAPTURE_STRUCT_LOCAL,
                    projection: tcx.mk_place_elems(&projs),
                };
                self.var_debug_info.push(VarDebugInfo {
                    name,
                    source_info: SourceInfo::outermost(captured_place.var_ident.span),
                    value: VarDebugInfoContents::Place(use_place),
                    composite: None,
                    argument_index: None,
                });

                let capture = Capture { captured_place, use_place, mutability };
                (var_id, capture)
            })
            .collect();
    }

    fn args_and_body(
        &mut self,
        mut block: BasicBlock,
        arguments: &IndexSlice<ParamId, Param<'tcx>>,
        argument_scope: region::Scope,
        expr_id: ExprId,
    ) -> BlockAnd<()> {
        let expr_span = self.thir[expr_id].span;
        // Allocate locals for the function arguments
        for (argument_index, param) in arguments.iter().enumerate() {
            let source_info =
                SourceInfo::outermost(param.pat.as_ref().map_or(self.fn_span, |pat| pat.span));
            let arg_local =
                self.local_decls.push(LocalDecl::with_source_info(param.ty, source_info));

            // If this is a simple binding pattern, give debuginfo a nice name.
            if let Some(ref pat) = param.pat
                && let Some(name) = pat.simple_ident()
            {
                self.var_debug_info.push(VarDebugInfo {
                    name,
                    source_info,
                    value: VarDebugInfoContents::Place(arg_local.into()),
                    composite: None,
                    argument_index: Some(argument_index as u16 + 1),
                });
            }
        }

        self.insert_upvar_arg();

        let mut scope = None;
        // Bind the argument patterns
        for (index, param) in arguments.iter().enumerate() {
            // Function arguments always get the first Local indices after the return place
            let local = Local::new(index + 1);
            let place = Place::from(local);

            // Make sure we drop (parts of) the argument even when not matched on.
            self.schedule_drop(
                param.pat.as_ref().map_or(expr_span, |pat| pat.span),
                argument_scope,
                local,
                DropKind::Value,
            );

            let Some(ref pat) = param.pat else {
                continue;
            };
            let original_source_scope = self.source_scope;
            let span = pat.span;
            if let Some(arg_hir_id) = param.hir_id {
                self.set_correct_source_scope_for_arg(arg_hir_id, original_source_scope, span);
            }
            match pat.kind {
                // Don't introduce extra copies for simple bindings
                PatKind::Binding {
                    var,
                    mode: BindingMode(ByRef::No, mutability),
                    subpattern: None,
                    ..
                } => {
                    self.local_decls[local].mutability = mutability;
                    self.local_decls[local].source_info.scope = self.source_scope;
                    **self.local_decls[local].local_info.as_mut().unwrap_crate_local() =
                        if let Some(kind) = param.self_kind {
                            LocalInfo::User(BindingForm::ImplicitSelf(kind))
                        } else {
                            let binding_mode = BindingMode(ByRef::No, mutability);
                            LocalInfo::User(BindingForm::Var(VarBindingForm {
                                binding_mode,
                                opt_ty_info: param.ty_span,
                                opt_match_place: Some((None, span)),
                                pat_span: span,
                            }))
                        };
                    self.var_indices.insert(var, LocalsForNode::One(local));
                }
                _ => {
                    scope = self.declare_bindings(
                        scope,
                        expr_span,
                        &pat,
                        None,
                        Some((Some(&place), span)),
                    );
                    let place_builder = PlaceBuilder::from(local);
                    block = self.place_into_pattern(block, pat, place_builder, false).into_block();
                }
            }
            self.source_scope = original_source_scope;
        }

        // Enter the argument pattern bindings source scope, if it exists.
        if let Some(source_scope) = scope {
            self.source_scope = source_scope;
        }

        if self.tcx.intrinsic(self.def_id).is_some_and(|i| i.must_be_overridden) {
            let source_info = self.source_info(rustc_span::DUMMY_SP);
            self.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
            self.cfg.start_new_block().unit()
        } else {
            // Ensure we don't silently codegen functions with fake bodies.
            match self.tcx.hir_node(self.hir_id) {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn { has_body: false, .. },
                    ..
                }) => {
                    self.tcx.dcx().span_delayed_bug(
                        expr_span,
                        format!("fn item without body has reached MIR building: {:?}", self.def_id),
                    );
                }
                _ => {}
            }
            self.expr_into_dest(Place::return_place(), block, expr_id)
        }
    }

    fn set_correct_source_scope_for_arg(
        &mut self,
        arg_hir_id: HirId,
        original_source_scope: SourceScope,
        pattern_span: Span,
    ) {
        let parent_id = self.source_scopes[original_source_scope]
            .local_data
            .as_ref()
            .unwrap_crate_local()
            .lint_root;
        self.maybe_new_source_scope(pattern_span, arg_hir_id, parent_id);
    }

    fn get_unit_temp(&mut self) -> Place<'tcx> {
        match self.unit_temp {
            Some(tmp) => tmp,
            None => {
                let ty = self.tcx.types.unit;
                let fn_span = self.fn_span;
                let tmp = self.temp(ty, fn_span);
                self.unit_temp = Some(tmp);
                tmp
            }
        }
    }
}

fn parse_float_into_constval<'tcx>(
    num: Symbol,
    float_ty: ty::FloatTy,
    neg: bool,
) -> Option<ConstValue<'tcx>> {
    parse_float_into_scalar(num, float_ty, neg).map(|s| ConstValue::Scalar(s.into()))
}

pub(crate) fn parse_float_into_scalar(
    num: Symbol,
    float_ty: ty::FloatTy,
    neg: bool,
) -> Option<ScalarInt> {
    let num = num.as_str();
    match float_ty {
        // FIXME(f16_f128): When available, compare to the library parser as with `f32` and `f64`
        ty::FloatTy::F16 => {
            let mut f = num.parse::<Half>().ok()?;
            if neg {
                f = -f;
            }
            Some(ScalarInt::from(f))
        }
        ty::FloatTy::F32 => {
            let Ok(rust_f) = num.parse::<f32>() else { return None };
            let mut f = num
                .parse::<Single>()
                .unwrap_or_else(|e| panic!("apfloat::ieee::Single failed to parse `{num}`: {e:?}"));

            assert!(
                u128::from(rust_f.to_bits()) == f.to_bits(),
                "apfloat::ieee::Single gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})",
                rust_f,
                f,
                f.to_bits(),
                Single::from_bits(rust_f.to_bits().into()),
                rust_f.to_bits()
            );

            if neg {
                f = -f;
            }

            Some(ScalarInt::from(f))
        }
        ty::FloatTy::F64 => {
            let Ok(rust_f) = num.parse::<f64>() else { return None };
            let mut f = num
                .parse::<Double>()
                .unwrap_or_else(|e| panic!("apfloat::ieee::Double failed to parse `{num}`: {e:?}"));

            assert!(
                u128::from(rust_f.to_bits()) == f.to_bits(),
                "apfloat::ieee::Double gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})",
                rust_f,
                f,
                f.to_bits(),
                Double::from_bits(rust_f.to_bits().into()),
                rust_f.to_bits()
            );

            if neg {
                f = -f;
            }

            Some(ScalarInt::from(f))
        }
        // FIXME(f16_f128): When available, compare to the library parser as with `f32` and `f64`
        ty::FloatTy::F128 => {
            let mut f = num.parse::<Quad>().ok()?;
            if neg {
                f = -f;
            }
            Some(ScalarInt::from(f))
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Builder methods are broken up into modules, depending on what kind
// of thing is being lowered. Note that they use the `unpack` macro
// above extensively.

mod block;
mod cfg;
mod coverageinfo;
mod custom;
mod expr;
mod matches;
mod misc;
mod scope;

pub(crate) use expr::category::Category as ExprCategory;
