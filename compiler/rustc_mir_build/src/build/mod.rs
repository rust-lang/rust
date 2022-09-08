use crate::build;
pub(crate) use crate::build::expr::as_constant::lit_to_mir_constant;
use crate::build::expr::as_place::PlaceBuilder;
use crate::build::scope::DropKind;
use crate::thir::pattern::pat_from_hir;
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{GeneratorKind, Node};
use rustc_index::vec::{Idx, IndexVec};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_middle::hir::place::PlaceBase as HirPlaceBase;
use rustc_middle::middle::region;
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::thir::{BindingMode, Expr, ExprId, LintLevel, LocalVarId, PatKind, Thir};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitable, TypeckResults};
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use super::lints;

pub(crate) fn mir_built<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx rustc_data_structures::steal::Steal<Body<'tcx>> {
    if let Some(def) = def.try_upgrade(tcx) {
        return tcx.mir_built(def);
    }

    let mut body = mir_build(tcx, def);
    if def.const_param_did.is_some() {
        assert!(matches!(body.source.instance, ty::InstanceDef::Item(_)));
        body.source = MirSource::from_instance(ty::InstanceDef::Item(def.to_global()));
    }

    tcx.alloc_steal_mir(body)
}

/// Construct the MIR for a given `DefId`.
fn mir_build(tcx: TyCtxt<'_>, def: ty::WithOptConstParam<LocalDefId>) -> Body<'_> {
    let id = tcx.hir().local_def_id_to_hir_id(def.did);
    let body_owner_kind = tcx.hir().body_owner_kind(def.did);
    let typeck_results = tcx.typeck_opt_const_arg(def);

    // Ensure unsafeck and abstract const building is ran before we steal the THIR.
    // We can't use `ensure()` for `thir_abstract_const` as it doesn't compute the query
    // if inputs are green. This can cause ICEs when calling `thir_abstract_const` after
    // THIR has been stolen if we haven't computed this query yet.
    match def {
        ty::WithOptConstParam { did, const_param_did: Some(const_param_did) } => {
            tcx.ensure().thir_check_unsafety_for_const_arg((did, const_param_did));
            drop(tcx.thir_abstract_const_of_const_arg((did, const_param_did)));
        }
        ty::WithOptConstParam { did, const_param_did: None } => {
            tcx.ensure().thir_check_unsafety(did);
            drop(tcx.thir_abstract_const(did));
        }
    }

    // Figure out what primary body this item has.
    let (body_id, return_ty_span, span_with_body) = match tcx.hir().get(id) {
        Node::Expr(hir::Expr {
            kind: hir::ExprKind::Closure(hir::Closure { fn_decl, body, .. }),
            ..
        }) => (*body, fn_decl.output.span(), None),
        Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(hir::FnSig { decl, .. }, _, body_id),
            span,
            ..
        })
        | Node::ImplItem(hir::ImplItem {
            kind: hir::ImplItemKind::Fn(hir::FnSig { decl, .. }, body_id),
            span,
            ..
        })
        | Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Fn(hir::FnSig { decl, .. }, hir::TraitFn::Provided(body_id)),
            span,
            ..
        }) => {
            // Use the `Span` of the `Item/ImplItem/TraitItem` as the body span,
            // since the def span of a function does not include the body
            (*body_id, decl.output.span(), Some(*span))
        }
        Node::Item(hir::Item {
            kind: hir::ItemKind::Static(ty, _, body_id) | hir::ItemKind::Const(ty, body_id),
            ..
        })
        | Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Const(ty, body_id), .. })
        | Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Const(ty, Some(body_id)),
            ..
        }) => (*body_id, ty.span, None),
        Node::AnonConst(hir::AnonConst { body, hir_id, .. }) => {
            (*body, tcx.hir().span(*hir_id), None)
        }

        _ => span_bug!(tcx.hir().span(id), "can't build MIR for {:?}", def.did),
    };

    // If we don't have a specialized span for the body, just use the
    // normal def span.
    let span_with_body = span_with_body.unwrap_or_else(|| tcx.hir().span(id));

    tcx.infer_ctxt().enter(|infcx| {
        let body = if let Some(error_reported) = typeck_results.tainted_by_errors {
            build::construct_error(&infcx, def, id, body_id, body_owner_kind, error_reported)
        } else if body_owner_kind.is_fn_or_closure() {
            // fetch the fully liberated fn signature (that is, all bound
            // types/lifetimes replaced)
            let fn_sig = typeck_results.liberated_fn_sigs()[id];
            let fn_def_id = tcx.hir().local_def_id(id);

            let safety = match fn_sig.unsafety {
                hir::Unsafety::Normal => Safety::Safe,
                hir::Unsafety::Unsafe => Safety::FnUnsafe,
            };

            let body = tcx.hir().body(body_id);
            let (thir, expr) = tcx
                .thir_body(def)
                .unwrap_or_else(|_| (tcx.alloc_steal_thir(Thir::new()), ExprId::from_u32(0)));
            // We ran all queries that depended on THIR at the beginning
            // of `mir_build`, so now we can steal it
            let thir = thir.steal();
            let ty = tcx.type_of(fn_def_id);
            let mut abi = fn_sig.abi;
            let implicit_argument = match ty.kind() {
                ty::Closure(..) => {
                    // HACK(eddyb) Avoid having RustCall on closures,
                    // as it adds unnecessary (and wrong) auto-tupling.
                    abi = Abi::Rust;
                    vec![ArgInfo(liberated_closure_env_ty(tcx, id, body_id), None, None, None)]
                }
                ty::Generator(..) => {
                    let gen_ty = tcx.typeck_body(body_id).node_type(id);

                    // The resume argument may be missing, in that case we need to provide it here.
                    // It will always be `()` in this case.
                    if body.params.is_empty() {
                        vec![
                            ArgInfo(gen_ty, None, None, None),
                            ArgInfo(tcx.mk_unit(), None, None, None),
                        ]
                    } else {
                        vec![ArgInfo(gen_ty, None, None, None)]
                    }
                }
                _ => vec![],
            };

            let explicit_arguments = body.params.iter().enumerate().map(|(index, arg)| {
                let owner_id = tcx.hir().body_owner(body_id);
                let opt_ty_info;
                let self_arg;
                if let Some(ref fn_decl) = tcx.hir().fn_decl_by_hir_id(owner_id) {
                    opt_ty_info = fn_decl
                        .inputs
                        .get(index)
                        // Make sure that inferred closure args have no type span
                        .and_then(|ty| if arg.pat.span != ty.span { Some(ty.span) } else { None });
                    self_arg = if index == 0 && fn_decl.implicit_self.has_implicit_self() {
                        match fn_decl.implicit_self {
                            hir::ImplicitSelfKind::Imm => Some(ImplicitSelfKind::Imm),
                            hir::ImplicitSelfKind::Mut => Some(ImplicitSelfKind::Mut),
                            hir::ImplicitSelfKind::ImmRef => Some(ImplicitSelfKind::ImmRef),
                            hir::ImplicitSelfKind::MutRef => Some(ImplicitSelfKind::MutRef),
                            _ => None,
                        }
                    } else {
                        None
                    };
                } else {
                    opt_ty_info = None;
                    self_arg = None;
                }

                // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
                // (as it's created inside the body itself, not passed in from outside).
                let ty = if fn_sig.c_variadic && index == fn_sig.inputs().len() {
                    let va_list_did = tcx.require_lang_item(LangItem::VaList, Some(arg.span));

                    tcx.bound_type_of(va_list_did).subst(tcx, &[tcx.lifetimes.re_erased.into()])
                } else {
                    fn_sig.inputs()[index]
                };

                ArgInfo(ty, opt_ty_info, Some(&arg), self_arg)
            });

            let arguments = implicit_argument.into_iter().chain(explicit_arguments);

            let (yield_ty, return_ty) = if body.generator_kind.is_some() {
                let gen_ty = tcx.typeck_body(body_id).node_type(id);
                let gen_sig = match gen_ty.kind() {
                    ty::Generator(_, gen_substs, ..) => gen_substs.as_generator().sig(),
                    _ => span_bug!(tcx.hir().span(id), "generator w/o generator type: {:?}", ty),
                };
                (Some(gen_sig.yield_ty), gen_sig.return_ty)
            } else {
                (None, fn_sig.output())
            };

            let mut mir = build::construct_fn(
                &thir,
                &infcx,
                def,
                id,
                arguments,
                safety,
                abi,
                return_ty,
                return_ty_span,
                body,
                expr,
                span_with_body,
            );
            if yield_ty.is_some() {
                mir.generator.as_mut().unwrap().yield_ty = yield_ty;
            }
            mir
        } else {
            // Get the revealed type of this const. This is *not* the adjusted
            // type of its body, which may be a subtype of this type. For
            // example:
            //
            // fn foo(_: &()) {}
            // static X: fn(&'static ()) = foo;
            //
            // The adjusted type of the body of X is `for<'a> fn(&'a ())` which
            // is not the same as the type of X. We need the type of the return
            // place to be the type of the constant because NLL typeck will
            // equate them.

            let return_ty = typeck_results.node_type(id);

            let (thir, expr) = tcx
                .thir_body(def)
                .unwrap_or_else(|_| (tcx.alloc_steal_thir(Thir::new()), ExprId::from_u32(0)));
            // We ran all queries that depended on THIR at the beginning
            // of `mir_build`, so now we can steal it
            let thir = thir.steal();

            let span_with_body = span_with_body.to(tcx.hir().span(body_id.hir_id));

            build::construct_const(
                &thir,
                &infcx,
                expr,
                def,
                id,
                return_ty,
                return_ty_span,
                span_with_body,
            )
        };

        lints::check(tcx, &body);

        // The borrow checker will replace all the regions here with its own
        // inference variables. There's no point having non-erased regions here.
        // The exception is `body.user_type_annotations`, which is used unmodified
        // by borrow checking.
        debug_assert!(
            !(body.local_decls.has_free_regions()
                || body.basic_blocks.has_free_regions()
                || body.var_debug_info.has_free_regions()
                || body.yield_ty().has_free_regions()),
            "Unexpected free regions in MIR: {:?}",
            body,
        );

        body
    })
}

///////////////////////////////////////////////////////////////////////////
// BuildMir -- walks a crate, looking for fn items and methods to build MIR from

fn liberated_closure_env_ty(
    tcx: TyCtxt<'_>,
    closure_expr_id: hir::HirId,
    body_id: hir::BodyId,
) -> Ty<'_> {
    let closure_ty = tcx.typeck_body(body_id).node_type(closure_expr_id);

    let ty::Closure(closure_def_id, closure_substs) = *closure_ty.kind() else {
        bug!("closure expr does not have closure type: {:?}", closure_ty);
    };

    let bound_vars =
        tcx.mk_bound_variable_kinds(std::iter::once(ty::BoundVariableKind::Region(ty::BrEnv)));
    let br =
        ty::BoundRegion { var: ty::BoundVar::from_usize(bound_vars.len() - 1), kind: ty::BrEnv };
    let env_region = ty::ReLateBound(ty::INNERMOST, br);
    let closure_env_ty = tcx.closure_env_ty(closure_def_id, closure_substs, env_region).unwrap();
    tcx.erase_late_bound_regions(ty::Binder::bind_with_vars(closure_env_ty, bound_vars))
}

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
    TailExpr {
        /// If true, then the surrounding context of the block ignores
        /// the result of evaluating the block's tail expression.
        ///
        /// Example: `let _ = { STMT_1; EXPR };`
        tail_result_is_ignored: bool,

        /// `Span` of the tail expression.
        span: Span,
    },

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
    infcx: &'a InferCtxt<'a, 'tcx>,
    typeck_results: &'tcx TypeckResults<'tcx>,
    region_scope_tree: &'tcx region::ScopeTree,
    param_env: ty::ParamEnv<'tcx>,

    thir: &'a Thir<'tcx>,
    cfg: CFG<'tcx>,

    def_id: DefId,
    hir_id: hir::HirId,
    parent_module: DefId,
    check_overflow: bool,
    fn_span: Span,
    arg_count: usize,
    generator_kind: Option<GeneratorKind>,

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

    /// The current unsafe block in scope
    in_scope_unsafe: Safety,

    /// The vector of all scopes that we have created thus far;
    /// we track this for debuginfo later.
    source_scopes: IndexVec<SourceScope, SourceScopeData<'tcx>>,
    source_scope: SourceScope,

    /// The guard-context: each time we build the guard expression for
    /// a match arm, we push onto this stack, and then pop when we
    /// finish building it.
    guard_context: Vec<GuardFrame>,

    /// Maps `HirId`s of variable bindings to the `Local`s created for them.
    /// (A match binding can have two locals; the 2nd is for the arm's guard.)
    var_indices: FxHashMap<LocalVarId, LocalsForNode>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    canonical_user_type_annotations: ty::CanonicalUserTypeAnnotations<'tcx>,
    upvar_mutbls: Vec<Mutability>,
    unit_temp: Option<Place<'tcx>>,

    var_debug_info: Vec<VarDebugInfo<'tcx>>,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
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
                &BlockFrame::TailExpr { tail_result_is_ignored, span } => {
                    return Some(BlockTailInfo { tail_result_is_ignored, span });
                }
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
                BlockFrame::TailExpr { tail_result_is_ignored: ignored, .. }
                | BlockFrame::Statement { ignores_expr_result: ignored },
            ) => *ignored,
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
    fn new(id: LocalVarId, _binding_mode: BindingMode) -> Self {
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
    struct ScopeId { .. }
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

    ($c:expr) => {{
        let BlockAnd(b, ()) = $c;
        b
    }};
}

///////////////////////////////////////////////////////////////////////////
/// the main entry point for building MIR for a function

struct ArgInfo<'tcx>(
    Ty<'tcx>,
    Option<Span>,
    Option<&'tcx hir::Param<'tcx>>,
    Option<ImplicitSelfKind>,
);

fn construct_fn<'tcx, A>(
    thir: &Thir<'tcx>,
    infcx: &InferCtxt<'_, 'tcx>,
    fn_def: ty::WithOptConstParam<LocalDefId>,
    fn_id: hir::HirId,
    arguments: A,
    safety: Safety,
    abi: Abi,
    return_ty: Ty<'tcx>,
    return_ty_span: Span,
    body: &'tcx hir::Body<'tcx>,
    expr: ExprId,
    span_with_body: Span,
) -> Body<'tcx>
where
    A: Iterator<Item = ArgInfo<'tcx>>,
{
    let arguments: Vec<_> = arguments.collect();

    let tcx = infcx.tcx;
    let span = tcx.hir().span(fn_id);

    let mut builder = Builder::new(
        thir,
        infcx,
        fn_def,
        fn_id,
        span_with_body,
        arguments.len(),
        safety,
        return_ty,
        return_ty_span,
        body.generator_kind,
    );

    let call_site_scope =
        region::Scope { id: body.value.hir_id.local_id, data: region::ScopeData::CallSite };
    let arg_scope =
        region::Scope { id: body.value.hir_id.local_id, data: region::ScopeData::Arguments };
    let source_info = builder.source_info(span);
    let call_site_s = (call_site_scope, source_info);
    unpack!(builder.in_scope(call_site_s, LintLevel::Inherited, |builder| {
        let arg_scope_s = (arg_scope, source_info);
        // Attribute epilogue to function's closing brace
        let fn_end = span_with_body.shrink_to_hi();
        let return_block =
            unpack!(builder.in_breakable_scope(None, Place::return_place(), fn_end, |builder| {
                Some(builder.in_scope(arg_scope_s, LintLevel::Inherited, |builder| {
                    builder.args_and_body(
                        START_BLOCK,
                        fn_def.did,
                        &arguments,
                        arg_scope,
                        &thir[expr],
                    )
                }))
            }));
        let source_info = builder.source_info(fn_end);
        builder.cfg.terminate(return_block, source_info, TerminatorKind::Return);
        builder.build_drop_trees();
        return_block.unit()
    }));

    let spread_arg = if abi == Abi::RustCall {
        // RustCall pseudo-ABI untuples the last argument.
        Some(Local::new(arguments.len()))
    } else {
        None
    };

    let mut body = builder.finish();
    body.spread_arg = spread_arg;
    body
}

fn construct_const<'a, 'tcx>(
    thir: &'a Thir<'tcx>,
    infcx: &'a InferCtxt<'a, 'tcx>,
    expr: ExprId,
    def: ty::WithOptConstParam<LocalDefId>,
    hir_id: hir::HirId,
    const_ty: Ty<'tcx>,
    const_ty_span: Span,
    span: Span,
) -> Body<'tcx> {
    let mut builder = Builder::new(
        thir,
        infcx,
        def,
        hir_id,
        span,
        0,
        Safety::Safe,
        const_ty,
        const_ty_span,
        None,
    );

    let mut block = START_BLOCK;
    unpack!(block = builder.expr_into_dest(Place::return_place(), block, &thir[expr]));

    let source_info = builder.source_info(span);
    builder.cfg.terminate(block, source_info, TerminatorKind::Return);

    builder.build_drop_trees();

    builder.finish()
}

/// Construct MIR for an item that has had errors in type checking.
///
/// This is required because we may still want to run MIR passes on an item
/// with type errors, but normal MIR construction can't handle that in general.
fn construct_error<'a, 'tcx>(
    infcx: &'a InferCtxt<'a, 'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
    hir_id: hir::HirId,
    body_id: hir::BodyId,
    body_owner_kind: hir::BodyOwnerKind,
    err: ErrorGuaranteed,
) -> Body<'tcx> {
    let tcx = infcx.tcx;
    let span = tcx.hir().span(hir_id);
    let ty = tcx.ty_error();
    let generator_kind = tcx.hir().body(body_id).generator_kind;
    let num_params = match body_owner_kind {
        hir::BodyOwnerKind::Fn => tcx.hir().fn_decl_by_hir_id(hir_id).unwrap().inputs.len(),
        hir::BodyOwnerKind::Closure => {
            if generator_kind.is_some() {
                // Generators have an implicit `self` parameter *and* a possibly
                // implicit resume parameter.
                2
            } else {
                // The implicit self parameter adds another local in MIR.
                1 + tcx.hir().fn_decl_by_hir_id(hir_id).unwrap().inputs.len()
            }
        }
        hir::BodyOwnerKind::Const => 0,
        hir::BodyOwnerKind::Static(_) => 0,
    };
    let mut cfg = CFG { basic_blocks: IndexVec::new() };
    let mut source_scopes = IndexVec::new();
    let mut local_decls = IndexVec::from_elem_n(LocalDecl::new(ty, span), 1);

    cfg.start_new_block();
    source_scopes.push(SourceScopeData {
        span,
        parent_scope: None,
        inlined: None,
        inlined_parent_scope: None,
        local_data: ClearCrossCrate::Set(SourceScopeLocalData {
            lint_root: hir_id,
            safety: Safety::Safe,
        }),
    });
    let source_info = SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE };

    // Some MIR passes will expect the number of parameters to match the
    // function declaration.
    for _ in 0..num_params {
        local_decls.push(LocalDecl::with_source_info(ty, source_info));
    }
    cfg.terminate(START_BLOCK, source_info, TerminatorKind::Unreachable);

    let mut body = Body::new(
        MirSource::item(def.did.to_def_id()),
        cfg.basic_blocks,
        source_scopes,
        local_decls,
        IndexVec::new(),
        num_params,
        vec![],
        span,
        generator_kind,
        Some(err),
    );
    body.generator.as_mut().map(|gen| gen.yield_ty = Some(ty));
    body
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    fn new(
        thir: &'a Thir<'tcx>,
        infcx: &'a InferCtxt<'a, 'tcx>,
        def: ty::WithOptConstParam<LocalDefId>,
        hir_id: hir::HirId,
        span: Span,
        arg_count: usize,
        safety: Safety,
        return_ty: Ty<'tcx>,
        return_span: Span,
        generator_kind: Option<GeneratorKind>,
    ) -> Builder<'a, 'tcx> {
        let tcx = infcx.tcx;
        let attrs = tcx.hir().attrs(hir_id);
        // Some functions always have overflow checks enabled,
        // however, they may not get codegen'd, depending on
        // the settings for the crate they are codegened in.
        let mut check_overflow = tcx.sess.contains_name(attrs, sym::rustc_inherit_overflow_checks);
        // Respect -C overflow-checks.
        check_overflow |= tcx.sess.overflow_checks();
        // Constants always need overflow checks.
        check_overflow |= matches!(
            tcx.hir().body_owner_kind(def.did),
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_)
        );

        let lint_level = LintLevel::Explicit(hir_id);
        let param_env = tcx.param_env(def.did);
        let mut builder = Builder {
            thir,
            tcx,
            infcx,
            typeck_results: tcx.typeck_opt_const_arg(def),
            region_scope_tree: tcx.region_scope_tree(def.did),
            param_env,
            def_id: def.did.to_def_id(),
            hir_id,
            parent_module: tcx.parent_module(hir_id).to_def_id(),
            check_overflow,
            cfg: CFG { basic_blocks: IndexVec::new() },
            fn_span: span,
            arg_count,
            generator_kind,
            scopes: scope::Scopes::new(),
            block_context: BlockContext::new(),
            source_scopes: IndexVec::new(),
            source_scope: OUTERMOST_SOURCE_SCOPE,
            guard_context: vec![],
            in_scope_unsafe: safety,
            local_decls: IndexVec::from_elem_n(LocalDecl::new(return_ty, return_span), 1),
            canonical_user_type_annotations: IndexVec::new(),
            upvar_mutbls: vec![],
            var_indices: Default::default(),
            unit_temp: None,
            var_debug_info: vec![],
        };

        assert_eq!(builder.cfg.start_new_block(), START_BLOCK);
        assert_eq!(
            builder.new_source_scope(span, lint_level, Some(safety)),
            OUTERMOST_SOURCE_SCOPE
        );
        builder.source_scopes[OUTERMOST_SOURCE_SCOPE].parent_scope = None;

        builder
    }

    fn finish(self) -> Body<'tcx> {
        for (index, block) in self.cfg.basic_blocks.iter().enumerate() {
            if block.terminator.is_none() {
                span_bug!(self.fn_span, "no terminator on block {:?}", index);
            }
        }

        Body::new(
            MirSource::item(self.def_id),
            self.cfg.basic_blocks,
            self.source_scopes,
            self.local_decls,
            self.canonical_user_type_annotations,
            self.arg_count,
            self.var_debug_info,
            self.fn_span,
            self.generator_kind,
            self.typeck_results.tainted_by_errors,
        )
    }

    fn args_and_body(
        &mut self,
        mut block: BasicBlock,
        fn_def_id: LocalDefId,
        arguments: &[ArgInfo<'tcx>],
        argument_scope: region::Scope,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<()> {
        // Allocate locals for the function arguments
        for &ArgInfo(ty, _, arg_opt, _) in arguments.iter() {
            let source_info =
                SourceInfo::outermost(arg_opt.map_or(self.fn_span, |arg| arg.pat.span));
            let arg_local = self.local_decls.push(LocalDecl::with_source_info(ty, source_info));

            // If this is a simple binding pattern, give debuginfo a nice name.
            if let Some(arg) = arg_opt && let Some(ident) = arg.pat.simple_ident() {
                self.var_debug_info.push(VarDebugInfo {
                    name: ident.name,
                    source_info,
                    value: VarDebugInfoContents::Place(arg_local.into()),
                });
            }
        }

        let tcx = self.tcx;
        let tcx_hir = tcx.hir();
        let hir_typeck_results = self.typeck_results;

        // In analyze_closure() in upvar.rs we gathered a list of upvars used by an
        // indexed closure and we stored in a map called closure_min_captures in TypeckResults
        // with the closure's DefId. Here, we run through that vec of UpvarIds for
        // the given closure and use the necessary information to create upvar
        // debuginfo and to fill `self.upvar_mutbls`.
        if hir_typeck_results.closure_min_captures.get(&fn_def_id).is_some() {
            let mut closure_env_projs = vec![];
            let mut closure_ty = self.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;
            if let ty::Ref(_, ty, _) = closure_ty.kind() {
                closure_env_projs.push(ProjectionElem::Deref);
                closure_ty = *ty;
            }
            let upvar_substs = match closure_ty.kind() {
                ty::Closure(_, substs) => ty::UpvarSubsts::Closure(substs),
                ty::Generator(_, substs, _) => ty::UpvarSubsts::Generator(substs),
                _ => span_bug!(self.fn_span, "upvars with non-closure env ty {:?}", closure_ty),
            };
            let def_id = self.def_id.as_local().unwrap();
            let capture_syms = tcx.symbols_for_closure_captures((def_id, fn_def_id));
            let capture_tys = upvar_substs.upvar_tys();
            let captures_with_tys = hir_typeck_results
                .closure_min_captures_flattened(fn_def_id)
                .zip(capture_tys.zip(capture_syms));

            self.upvar_mutbls = captures_with_tys
                .enumerate()
                .map(|(i, (captured_place, (ty, sym)))| {
                    let capture = captured_place.info.capture_kind;
                    let var_id = match captured_place.place.base {
                        HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
                        _ => bug!("Expected an upvar"),
                    };

                    let mutability = captured_place.mutability;

                    let mut projs = closure_env_projs.clone();
                    projs.push(ProjectionElem::Field(Field::new(i), ty));
                    match capture {
                        ty::UpvarCapture::ByValue => {}
                        ty::UpvarCapture::ByRef(..) => {
                            projs.push(ProjectionElem::Deref);
                        }
                    };

                    self.var_debug_info.push(VarDebugInfo {
                        name: *sym,
                        source_info: SourceInfo::outermost(tcx_hir.span(var_id)),
                        value: VarDebugInfoContents::Place(Place {
                            local: ty::CAPTURE_STRUCT_LOCAL,
                            projection: tcx.intern_place_elems(&projs),
                        }),
                    });

                    mutability
                })
                .collect();
        }

        let mut scope = None;
        // Bind the argument patterns
        for (index, arg_info) in arguments.iter().enumerate() {
            // Function arguments always get the first Local indices after the return place
            let local = Local::new(index + 1);
            let place = Place::from(local);
            let &ArgInfo(_, opt_ty_info, arg_opt, ref self_binding) = arg_info;

            // Make sure we drop (parts of) the argument even when not matched on.
            self.schedule_drop(
                arg_opt.as_ref().map_or(expr.span, |arg| arg.pat.span),
                argument_scope,
                local,
                DropKind::Value,
            );

            let Some(arg) = arg_opt else {
                continue;
            };
            let pat = match tcx.hir().get(arg.pat.hir_id) {
                Node::Pat(pat) => pat,
                node => bug!("pattern became {:?}", node),
            };
            let pattern = pat_from_hir(tcx, self.param_env, self.typeck_results, pat);
            let original_source_scope = self.source_scope;
            let span = pattern.span;
            self.set_correct_source_scope_for_arg(arg.hir_id, original_source_scope, span);
            match pattern.kind {
                // Don't introduce extra copies for simple bindings
                PatKind::Binding {
                    mutability,
                    var,
                    mode: BindingMode::ByValue,
                    subpattern: None,
                    ..
                } => {
                    self.local_decls[local].mutability = mutability;
                    self.local_decls[local].source_info.scope = self.source_scope;
                    self.local_decls[local].local_info = if let Some(kind) = self_binding {
                        Some(Box::new(LocalInfo::User(ClearCrossCrate::Set(
                            BindingForm::ImplicitSelf(*kind),
                        ))))
                    } else {
                        let binding_mode = ty::BindingMode::BindByValue(mutability);
                        Some(Box::new(LocalInfo::User(ClearCrossCrate::Set(BindingForm::Var(
                            VarBindingForm {
                                binding_mode,
                                opt_ty_info,
                                opt_match_place: Some((None, span)),
                                pat_span: span,
                            },
                        )))))
                    };
                    self.var_indices.insert(var, LocalsForNode::One(local));
                }
                _ => {
                    scope = self.declare_bindings(
                        scope,
                        expr.span,
                        &pattern,
                        matches::ArmHasGuard(false),
                        Some((Some(&place), span)),
                    );
                    let place_builder = PlaceBuilder::from(local);
                    unpack!(
                        block =
                            self.place_into_pattern(block, pattern.as_ref(), place_builder, false)
                    );
                }
            }
            self.source_scope = original_source_scope;
        }

        // Enter the argument pattern bindings source scope, if it exists.
        if let Some(source_scope) = scope {
            self.source_scope = source_scope;
        }

        self.expr_into_dest(Place::return_place(), block, &expr)
    }

    fn set_correct_source_scope_for_arg(
        &mut self,
        arg_hir_id: hir::HirId,
        original_source_scope: SourceScope,
        pattern_span: Span,
    ) {
        let tcx = self.tcx;
        let current_root = tcx.maybe_lint_level_root_bounded(arg_hir_id, self.hir_id);
        let parent_root = tcx.maybe_lint_level_root_bounded(
            self.source_scopes[original_source_scope]
                .local_data
                .as_ref()
                .assert_crate_local()
                .lint_root,
            self.hir_id,
        );
        if current_root != parent_root {
            self.source_scope =
                self.new_source_scope(pattern_span, LintLevel::Explicit(current_root), None);
        }
    }

    fn get_unit_temp(&mut self) -> Place<'tcx> {
        match self.unit_temp {
            Some(tmp) => tmp,
            None => {
                let ty = self.tcx.mk_unit();
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
    parse_float_into_scalar(num, float_ty, neg).map(ConstValue::Scalar)
}

pub(crate) fn parse_float_into_scalar(
    num: Symbol,
    float_ty: ty::FloatTy,
    neg: bool,
) -> Option<Scalar> {
    let num = num.as_str();
    match float_ty {
        ty::FloatTy::F32 => {
            let Ok(rust_f) = num.parse::<f32>() else { return None };
            let mut f = num.parse::<Single>().unwrap_or_else(|e| {
                panic!("apfloat::ieee::Single failed to parse `{}`: {:?}", num, e)
            });

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

            Some(Scalar::from_f32(f))
        }
        ty::FloatTy::F64 => {
            let Ok(rust_f) = num.parse::<f64>() else { return None };
            let mut f = num.parse::<Double>().unwrap_or_else(|e| {
                panic!("apfloat::ieee::Double failed to parse `{}`: {:?}", num, e)
            });

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

            Some(Scalar::from_f64(f))
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Builder methods are broken up into modules, depending on what kind
// of thing is being lowered. Note that they use the `unpack` macro
// above extensively.

mod block;
mod cfg;
mod expr;
mod matches;
mod misc;
mod scope;

pub(crate) use expr::category::Category as ExprCategory;
