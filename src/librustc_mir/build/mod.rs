use build;
use build::scope::{CachedBlock, DropKind};
use hair::cx::Cx;
use hair::{LintLevel, BindingMode, PatternKind};
use rustc::hir;
use rustc::hir::Node;
use rustc::hir::def_id::DefId;
use rustc::middle::region;
use rustc::mir::*;
use rustc::mir::visit::{MutVisitor, TyContext};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;
use rustc::util::nodemap::NodeMap;
use rustc_target::spec::PanicStrategy;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use shim;
use std::mem;
use std::u32;
use rustc_target::spec::abi::Abi;
use syntax::ast;
use syntax::attr::{self, UnwindAttr};
use syntax::symbol::keywords;
use syntax_pos::Span;
use transform::MirSource;
use util as mir_util;

use super::lints;

/// Construct the MIR for a given def-id.
pub fn mir_build<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> Mir<'tcx> {
    let id = tcx.hir().as_local_node_id(def_id).unwrap();

    // Figure out what primary body this item has.
    let (body_id, return_ty_span) = match tcx.hir().get(id) {
        Node::Variant(variant) =>
            return create_constructor_shim(tcx, id, &variant.node.data),
        Node::StructCtor(ctor) =>
            return create_constructor_shim(tcx, id, ctor),

        Node::Expr(hir::Expr { node: hir::ExprKind::Closure(_, decl, body_id, _, _), .. })
        | Node::Item(hir::Item { node: hir::ItemKind::Fn(decl, _, _, body_id), .. })
        | Node::ImplItem(
            hir::ImplItem {
                node: hir::ImplItemKind::Method(hir::MethodSig { decl, .. }, body_id),
                ..
            }
        )
        | Node::TraitItem(
            hir::TraitItem {
                node: hir::TraitItemKind::Method(
                    hir::MethodSig { decl, .. },
                    hir::TraitMethod::Provided(body_id),
                ),
                ..
            }
        ) => {
            (*body_id, decl.output.span())
        }
        Node::Item(hir::Item { node: hir::ItemKind::Static(ty, _, body_id), .. })
        | Node::Item(hir::Item { node: hir::ItemKind::Const(ty, body_id), .. })
        | Node::ImplItem(hir::ImplItem { node: hir::ImplItemKind::Const(ty, body_id), .. })
        | Node::TraitItem(
            hir::TraitItem { node: hir::TraitItemKind::Const(ty, Some(body_id)), .. }
        ) => {
            (*body_id, ty.span)
        }
        Node::AnonConst(hir::AnonConst { body, id, .. }) => {
            (*body, tcx.hir().span(*id))
        }

        _ => span_bug!(tcx.hir().span(id), "can't build MIR for {:?}", def_id),
    };

    tcx.infer_ctxt().enter(|infcx| {
        let cx = Cx::new(&infcx, id);
        let mut mir = if cx.tables().tainted_by_errors {
            build::construct_error(cx, body_id)
        } else if cx.body_owner_kind.is_fn_or_closure() {
            // fetch the fully liberated fn signature (that is, all bound
            // types/lifetimes replaced)
            let fn_hir_id = tcx.hir().node_to_hir_id(id);
            let fn_sig = cx.tables().liberated_fn_sigs()[fn_hir_id].clone();
            let fn_def_id = tcx.hir().local_def_id(id);

            let ty = tcx.type_of(fn_def_id);
            let mut abi = fn_sig.abi;
            let implicit_argument = match ty.sty {
                ty::Closure(..) => {
                    // HACK(eddyb) Avoid having RustCall on closures,
                    // as it adds unnecessary (and wrong) auto-tupling.
                    abi = Abi::Rust;
                    Some(ArgInfo(liberated_closure_env_ty(tcx, id, body_id), None, None, None))
                }
                ty::Generator(..) => {
                    let gen_ty = tcx.body_tables(body_id).node_id_to_type(fn_hir_id);
                    Some(ArgInfo(gen_ty, None, None, None))
                }
                _ => None,
            };

            let safety = match fn_sig.unsafety {
                hir::Unsafety::Normal => Safety::Safe,
                hir::Unsafety::Unsafe => Safety::FnUnsafe,
            };

            let body = tcx.hir().body(body_id);
            let explicit_arguments =
                body.arguments
                    .iter()
                    .enumerate()
                    .map(|(index, arg)| {
                        let owner_id = tcx.hir().body_owner(body_id);
                        let opt_ty_info;
                        let self_arg;
                        if let Some(ref fn_decl) = tcx.hir().fn_decl(owner_id) {
                            let ty_hir_id = fn_decl.inputs[index].hir_id;
                            let ty_span = tcx.hir().span(tcx.hir().hir_to_node_id(ty_hir_id));
                            opt_ty_info = Some(ty_span);
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
                        ArgInfo(fn_sig.inputs()[index], opt_ty_info, Some(&*arg.pat), self_arg)
                    });

            let arguments = implicit_argument.into_iter().chain(explicit_arguments);

            let (yield_ty, return_ty) = if body.is_generator {
                let gen_sig = match ty.sty {
                    ty::Generator(gen_def_id, gen_substs, ..) =>
                        gen_substs.sig(gen_def_id, tcx),
                    _ =>
                        span_bug!(tcx.hir().span(id), "generator w/o generator type: {:?}", ty),
                };
                (Some(gen_sig.yield_ty), gen_sig.return_ty)
            } else {
                (None, fn_sig.output())
            };

            build::construct_fn(cx, id, arguments, safety, abi,
                                return_ty, yield_ty, return_ty_span, body)
        } else {
            build::construct_const(cx, body_id, return_ty_span)
        };

        // Convert the Mir to global types.
        let mut globalizer = GlobalizeMir {
            tcx,
            span: mir.span
        };
        globalizer.visit_mir(&mut mir);
        let mir = unsafe {
            mem::transmute::<Mir, Mir<'tcx>>(mir)
        };

        mir_util::dump_mir(tcx, None, "mir_map", &0,
                           MirSource::item(def_id), &mir, |_, _| Ok(()) );

        lints::check(tcx, &mir, def_id);

        mir
    })
}

/// A pass to lift all the types and substitutions in a Mir
/// to the global tcx. Sadly, we don't have a "folder" that
/// can change 'tcx so we have to transmute afterwards.
struct GlobalizeMir<'a, 'gcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
    span: Span
}

impl<'a, 'gcx: 'tcx, 'tcx> MutVisitor<'tcx> for GlobalizeMir<'a, 'gcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: TyContext) {
        if let Some(lifted) = self.tcx.lift(ty) {
            *ty = lifted;
        } else {
            span_bug!(self.span,
                      "found type `{:?}` with inference types/regions in MIR",
                      ty);
        }
    }

    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, _: Location) {
        if let Some(lifted) = self.tcx.lift(region) {
            *region = lifted;
        } else {
            span_bug!(self.span,
                      "found region `{:?}` with inference types/regions in MIR",
                      region);
        }
    }

    fn visit_const(&mut self, constant: &mut &'tcx ty::LazyConst<'tcx>, _: Location) {
        if let Some(lifted) = self.tcx.lift(constant) {
            *constant = lifted;
        } else {
            span_bug!(self.span,
                      "found constant `{:?}` with inference types/regions in MIR",
                      constant);
        }
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>, _: Location) {
        if let Some(lifted) = self.tcx.lift(substs) {
            *substs = lifted;
        } else {
            span_bug!(self.span,
                      "found substs `{:?}` with inference types/regions in MIR",
                      substs);
        }
    }
}

fn create_constructor_shim<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     ctor_id: ast::NodeId,
                                     v: &'tcx hir::VariantData)
                                     -> Mir<'tcx>
{
    let span = tcx.hir().span(ctor_id);
    if let hir::VariantData::Tuple(ref fields, ctor_id) = *v {
        tcx.infer_ctxt().enter(|infcx| {
            let mut mir = shim::build_adt_ctor(&infcx, ctor_id, fields, span);

            // Convert the Mir to global types.
            let tcx = infcx.tcx.global_tcx();
            let mut globalizer = GlobalizeMir {
                tcx,
                span: mir.span
            };
            globalizer.visit_mir(&mut mir);
            let mir = unsafe {
                mem::transmute::<Mir, Mir<'tcx>>(mir)
            };

            mir_util::dump_mir(tcx, None, "mir_map", &0,
                               MirSource::item(tcx.hir().local_def_id(ctor_id)),
                               &mir, |_, _| Ok(()) );

            mir
        })
    } else {
        span_bug!(span, "attempting to create MIR for non-tuple variant {:?}", v);
    }
}

///////////////////////////////////////////////////////////////////////////
// BuildMir -- walks a crate, looking for fn items and methods to build MIR from

fn liberated_closure_env_ty<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                            closure_expr_id: ast::NodeId,
                                            body_id: hir::BodyId)
                                            -> Ty<'tcx> {
    let closure_expr_hir_id = tcx.hir().node_to_hir_id(closure_expr_id);
    let closure_ty = tcx.body_tables(body_id).node_id_to_type(closure_expr_hir_id);

    let (closure_def_id, closure_substs) = match closure_ty.sty {
        ty::Closure(closure_def_id, closure_substs) => (closure_def_id, closure_substs),
        _ => bug!("closure expr does not have closure type: {:?}", closure_ty)
    };

    let closure_env_ty = tcx.closure_env_ty(closure_def_id, closure_substs).unwrap();
    tcx.liberate_late_bound_regions(closure_def_id, &closure_env_ty)
}

#[derive(Debug, PartialEq, Eq)]
pub enum BlockFrame {
    /// Evaluation is currently within a statement.
    ///
    /// Examples include:
    ///  1. `EXPR;`
    ///  2. `let _ = EXPR;`
    ///  3. `let x = EXPR;`
    Statement {
        /// If true, then statement discards result from evaluating
        /// the expression (such as examples 1 and 2 above).
        ignores_expr_result: bool
    },

    /// Evaluation is currently within the tail expression of a block.
    ///
    /// Example: `{ STMT_1; STMT_2; EXPR }`
    TailExpr {
        /// If true, then the surrounding context of the block ignores
        /// the result of evaluating the block's tail expression.
        ///
        /// Example: `let _ = { STMT_1; EXPR };`
        tail_result_is_ignored: bool
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

            BlockFrame::Statement { .. } |
            BlockFrame::SubExpr => false,
        }
    }
    fn is_statement(&self) -> bool {
        match *self {
            BlockFrame::Statement { .. } => true,

            BlockFrame::TailExpr { .. } |
            BlockFrame::SubExpr => false,
        }
    }
 }

#[derive(Debug)]
struct BlockContext(Vec<BlockFrame>);

struct Builder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    hir: Cx<'a, 'gcx, 'tcx>,
    cfg: CFG<'tcx>,

    fn_span: Span,
    arg_count: usize,

    /// the current set of scopes, updated as we traverse;
    /// see the `scope` module for more details
    scopes: Vec<scope::Scope<'tcx>>,

    /// the block-context: each time we build the code within an hair::Block,
    /// we push a frame here tracking whether we are building a statement or
    /// if we are pushing the tail expression of the block. This is used to
    /// embed information in generated temps about whether they were created
    /// for a block tail expression or not.
    ///
    /// It would be great if we could fold this into `self.scopes`
    /// somehow; but right now I think that is very tightly tied to
    /// the code generation in ways that we cannot (or should not)
    /// start just throwing new entries onto that vector in order to
    /// distinguish the context of EXPR1 from the context of EXPR2 in
    /// `{ STMTS; EXPR1 } + EXPR2`
    block_context: BlockContext,

    /// The current unsafe block in scope, even if it is hidden by
    /// a PushUnsafeBlock
    unpushed_unsafe: Safety,

    /// The number of `push_unsafe_block` levels in scope
    push_unsafe_count: usize,

    /// the current set of breakables; see the `scope` module for more
    /// details
    breakable_scopes: Vec<scope::BreakableScope<'tcx>>,

    /// the vector of all scopes that we have created thus far;
    /// we track this for debuginfo later
    source_scopes: IndexVec<SourceScope, SourceScopeData>,
    source_scope_local_data: IndexVec<SourceScope, SourceScopeLocalData>,
    source_scope: SourceScope,

    /// the guard-context: each time we build the guard expression for
    /// a match arm, we push onto this stack, and then pop when we
    /// finish building it.
    guard_context: Vec<GuardFrame>,

    /// Maps node ids of variable bindings to the `Local`s created for them.
    /// (A match binding can have two locals; the 2nd is for the arm's guard.)
    var_indices: NodeMap<LocalsForNode>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    canonical_user_type_annotations: ty::CanonicalUserTypeAnnotations<'tcx>,
    upvar_decls: Vec<UpvarDecl>,
    unit_temp: Option<Place<'tcx>>,

    /// cached block with the RESUME terminator; this is created
    /// when first set of cleanups are built.
    cached_resume_block: Option<BasicBlock>,
    /// cached block with the RETURN terminator
    cached_return_block: Option<BasicBlock>,
    /// cached block with the UNREACHABLE terminator
    cached_unreachable_block: Option<BasicBlock>,
}

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    fn is_bound_var_in_guard(&self, id: ast::NodeId) -> bool {
        self.guard_context.iter().any(|frame| frame.locals.iter().any(|local| local.id == id))
    }

    fn var_local_id(&self, id: ast::NodeId, for_guard: ForGuard) -> Local {
        self.var_indices[&id].local_id(for_guard)
    }
}

impl BlockContext {
    fn new() -> Self { BlockContext(vec![]) }
    fn push(&mut self, bf: BlockFrame) { self.0.push(bf); }
    fn pop(&mut self) -> Option<BlockFrame> { self.0.pop() }

    /// Traverses the frames on the BlockContext, searching for either
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
                &BlockFrame::TailExpr { tail_result_is_ignored } =>
                    return Some(BlockTailInfo { tail_result_is_ignored })
            }
        }

        return None;
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
            Some(BlockFrame::TailExpr { tail_result_is_ignored: ignored }) |
            Some(BlockFrame::Statement { ignores_expr_result: ignored }) => *ignored,
        }
    }
}

#[derive(Debug)]
enum LocalsForNode {
    /// In the usual case, a node-id for an identifier maps to at most
    /// one Local declaration.
    One(Local),

    /// The exceptional case is identifiers in a match arm's pattern
    /// that are referenced in a guard of that match arm. For these,
    /// we can have `2+k` Locals, where `k` is the number of candidate
    /// patterns (separated by `|`) in the arm.
    ///
    /// * `for_arm_body` is the Local used in the arm body (which is
    ///   just like the `One` case above),
    ///
    /// * `ref_for_guard` is the Local used in the arm's guard (which
    ///   is a reference to a temp that is an alias of
    ///   `for_arm_body`).
    ///
    /// * `vals_for_guard` is the `k` Locals; at most one of them will
    ///   get initialized by the arm's execution, and after it is
    ///   initialized, `ref_for_guard` will be assigned a reference to
    ///   it.
    ///
    /// There reason we have `k` Locals rather than just 1 is to
    /// accommodate some restrictions imposed by two-phase borrows,
    /// which apply when we have a `ref mut` pattern.
    ForGuard { vals_for_guard: Vec<Local>, ref_for_guard: Local, for_arm_body: Local },
}

#[derive(Debug)]
struct GuardFrameLocal {
    id: ast::NodeId,
}

impl GuardFrameLocal {
    fn new(id: ast::NodeId, _binding_mode: BindingMode) -> Self {
        GuardFrameLocal {
            id: id,
        }
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
    /// here, when building for FIXME
    locals: Vec<GuardFrameLocal>,
}

/// ForGuard indicates whether we are talking about:
///   1. the temp for a local binding used solely within guard expressions,
///   2. the temp that holds reference to (1.), which is actually what the
///      guard expressions see, or
///   3. the temp for use outside of guard expressions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ForGuard {
    /// The `usize` identifies for which candidate pattern we want the
    /// local binding. We keep a temp per-candidate to accommodate
    /// two-phase borrows (see `LocalsForNode` documentation).
    ValWithinGuard(usize),
    RefWithinGuard,
    OutsideGuard,
}

impl LocalsForNode {
    fn local_id(&self, for_guard: ForGuard) -> Local {
        match (self, for_guard) {
            (&LocalsForNode::One(local_id), ForGuard::OutsideGuard) |
            (&LocalsForNode::ForGuard { ref_for_guard: local_id, .. }, ForGuard::RefWithinGuard) |
            (&LocalsForNode::ForGuard { for_arm_body: local_id, .. }, ForGuard::OutsideGuard) =>
                local_id,

            (&LocalsForNode::ForGuard { ref vals_for_guard, .. },
             ForGuard::ValWithinGuard(pat_idx)) =>
                vals_for_guard[pat_idx],

            (&LocalsForNode::One(_), ForGuard::ValWithinGuard(_)) |
            (&LocalsForNode::One(_), ForGuard::RefWithinGuard) =>
                bug!("anything with one local should never be within a guard."),
        }
    }
}

struct CFG<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
}

newtype_index! {
    pub struct ScopeId { .. }
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
    ($x:ident = $c:expr) => {
        {
            let BlockAnd(b, v) = $c;
            $x = b;
            v
        }
    };

    ($c:expr) => {
        {
            let BlockAnd(b, ()) = $c;
            b
        }
    };
}

fn should_abort_on_panic<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                         fn_def_id: DefId,
                                         abi: Abi)
                                         -> bool {
    // Not callable from C, so we can safely unwind through these
    if abi == Abi::Rust || abi == Abi::RustCall { return false; }

    // We never unwind, so it's not relevant to stop an unwind
    if tcx.sess.panic_strategy() != PanicStrategy::Unwind { return false; }

    // We cannot add landing pads, so don't add one
    if tcx.sess.no_landing_pads() { return false; }

    // This is a special case: some functions have a C abi but are meant to
    // unwind anyway. Don't stop them.
    let attrs = &tcx.get_attrs(fn_def_id);
    match attr::find_unwind_attr(Some(tcx.sess.diagnostic()), attrs) {
        None => true,
        Some(UnwindAttr::Allowed) => false,
        Some(UnwindAttr::Aborts) => true,
    }
}

///////////////////////////////////////////////////////////////////////////
/// the main entry point for building MIR for a function

struct ArgInfo<'gcx>(Ty<'gcx>,
                     Option<Span>,
                     Option<&'gcx hir::Pat>,
                     Option<ImplicitSelfKind>);

fn construct_fn<'a, 'gcx, 'tcx, A>(hir: Cx<'a, 'gcx, 'tcx>,
                                   fn_id: ast::NodeId,
                                   arguments: A,
                                   safety: Safety,
                                   abi: Abi,
                                   return_ty: Ty<'gcx>,
                                   yield_ty: Option<Ty<'gcx>>,
                                   return_ty_span: Span,
                                   body: &'gcx hir::Body)
                                   -> Mir<'tcx>
    where A: Iterator<Item=ArgInfo<'gcx>>
{
    let arguments: Vec<_> = arguments.collect();

    let tcx = hir.tcx();
    let tcx_hir = tcx.hir();
    let span = tcx_hir.span(fn_id);

    let hir_tables = hir.tables();
    let fn_def_id = tcx_hir.local_def_id(fn_id);

    // Gather the upvars of a closure, if any.
    // In analyze_closure() in upvar.rs we gathered a list of upvars used by a
    // closure and we stored in a map called upvar_list in TypeckTables indexed
    // with the closure's DefId. Here, we run through that vec of UpvarIds for
    // the given closure and use the necessary information to create UpvarDecl.
    let upvar_decls: Vec<_> = hir_tables
        .upvar_list
        .get(&fn_def_id)
        .into_iter()
        .flatten()
        .map(|upvar_id| {
            let var_hir_id = upvar_id.var_path.hir_id;
            let var_node_id = tcx_hir.hir_to_node_id(var_hir_id);
            let capture = hir_tables.upvar_capture(*upvar_id);
            let by_ref = match capture {
                ty::UpvarCapture::ByValue => false,
                ty::UpvarCapture::ByRef(..) => true,
            };
            let mut decl = UpvarDecl {
                debug_name: keywords::Invalid.name(),
                var_hir_id: ClearCrossCrate::Set(var_hir_id),
                by_ref,
                mutability: Mutability::Not,
            };
            if let Some(Node::Binding(pat)) = tcx_hir.find(var_node_id) {
                if let hir::PatKind::Binding(_, _, ident, _) = pat.node {
                    decl.debug_name = ident.name;
                    if let Some(&bm) = hir.tables.pat_binding_modes().get(pat.hir_id) {
                        if bm == ty::BindByValue(hir::MutMutable) {
                            decl.mutability = Mutability::Mut;
                        } else {
                            decl.mutability = Mutability::Not;
                        }
                    } else {
                        tcx.sess.delay_span_bug(pat.span, "missing binding mode");
                    }
                }
            }
            decl
        })
        .collect();

    let mut builder = Builder::new(hir,
        span,
        arguments.len(),
        safety,
        return_ty,
        return_ty_span,
        upvar_decls);

    let call_site_scope = region::Scope {
        id: body.value.hir_id.local_id,
        data: region::ScopeData::CallSite
    };
    let arg_scope = region::Scope {
        id: body.value.hir_id.local_id,
        data: region::ScopeData::Arguments
    };
    let mut block = START_BLOCK;
    let source_info = builder.source_info(span);
    let call_site_s = (call_site_scope, source_info);
    unpack!(block = builder.in_scope(call_site_s, LintLevel::Inherited, block, |builder| {
        if should_abort_on_panic(tcx, fn_def_id, abi) {
            builder.schedule_abort();
        }

        let arg_scope_s = (arg_scope, source_info);
        unpack!(block = builder.in_scope(arg_scope_s, LintLevel::Inherited, block, |builder| {
            builder.args_and_body(block, &arguments, arg_scope, &body.value)
        }));
        // Attribute epilogue to function's closing brace
        let fn_end = span.shrink_to_hi();
        let source_info = builder.source_info(fn_end);
        let return_block = builder.return_block();
        builder.cfg.terminate(block, source_info,
                              TerminatorKind::Goto { target: return_block });
        builder.cfg.terminate(return_block, source_info,
                              TerminatorKind::Return);
        // Attribute any unreachable codepaths to the function's closing brace
        if let Some(unreachable_block) = builder.cached_unreachable_block {
            builder.cfg.terminate(unreachable_block, source_info,
                                  TerminatorKind::Unreachable);
        }
        return_block.unit()
    }));
    assert_eq!(block, builder.return_block());

    let mut spread_arg = None;
    if abi == Abi::RustCall {
        // RustCall pseudo-ABI untuples the last argument.
        spread_arg = Some(Local::new(arguments.len()));
    }
    let closure_expr_id = tcx_hir.local_def_id(fn_id);
    info!("fn_id {:?} has attrs {:?}", closure_expr_id,
          tcx.get_attrs(closure_expr_id));

    let mut mir = builder.finish(yield_ty);
    mir.spread_arg = spread_arg;
    mir
}

fn construct_const<'a, 'gcx, 'tcx>(
    hir: Cx<'a, 'gcx, 'tcx>,
    body_id: hir::BodyId,
    ty_span: Span,
) -> Mir<'tcx> {
    let tcx = hir.tcx();
    let ast_expr = &tcx.hir().body(body_id).value;
    let ty = hir.tables().expr_ty_adjusted(ast_expr);
    let owner_id = tcx.hir().body_owner(body_id);
    let span = tcx.hir().span(owner_id);
    let mut builder = Builder::new(hir, span, 0, Safety::Safe, ty, ty_span,vec![]);

    let mut block = START_BLOCK;
    let expr = builder.hir.mirror(ast_expr);
    unpack!(block = builder.into_expr(&Place::Local(RETURN_PLACE), block, expr));

    let source_info = builder.source_info(span);
    builder.cfg.terminate(block, source_info, TerminatorKind::Return);

    // Constants can't `return` so a return block should not be created.
    assert_eq!(builder.cached_return_block, None);

    // Constants may be match expressions in which case an unreachable block may
    // be created, so terminate it properly.
    if let Some(unreachable_block) = builder.cached_unreachable_block {
        builder.cfg.terminate(unreachable_block, source_info,
                              TerminatorKind::Unreachable);
    }

    builder.finish(None)
}

fn construct_error<'a, 'gcx, 'tcx>(hir: Cx<'a, 'gcx, 'tcx>,
                                   body_id: hir::BodyId)
                                   -> Mir<'tcx> {
    let owner_id = hir.tcx().hir().body_owner(body_id);
    let span = hir.tcx().hir().span(owner_id);
    let ty = hir.tcx().types.err;
    let mut builder = Builder::new(hir, span, 0, Safety::Safe, ty, span, vec![]);
    let source_info = builder.source_info(span);
    builder.cfg.terminate(START_BLOCK, source_info, TerminatorKind::Unreachable);
    builder.finish(None)
}

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    fn new(hir: Cx<'a, 'gcx, 'tcx>,
           span: Span,
           arg_count: usize,
           safety: Safety,
           return_ty: Ty<'tcx>,
           return_span: Span,
           upvar_decls: Vec<UpvarDecl>)
           -> Builder<'a, 'gcx, 'tcx> {
        let lint_level = LintLevel::Explicit(hir.root_lint_level);
        let mut builder = Builder {
            hir,
            cfg: CFG { basic_blocks: IndexVec::new() },
            fn_span: span,
            arg_count,
            scopes: vec![],
            block_context: BlockContext::new(),
            source_scopes: IndexVec::new(),
            source_scope: OUTERMOST_SOURCE_SCOPE,
            source_scope_local_data: IndexVec::new(),
            guard_context: vec![],
            push_unsafe_count: 0,
            unpushed_unsafe: safety,
            breakable_scopes: vec![],
            local_decls: IndexVec::from_elem_n(
                LocalDecl::new_return_place(return_ty, return_span),
                1,
            ),
            canonical_user_type_annotations: IndexVec::new(),
            upvar_decls,
            var_indices: Default::default(),
            unit_temp: None,
            cached_resume_block: None,
            cached_return_block: None,
            cached_unreachable_block: None,
        };

        assert_eq!(builder.cfg.start_new_block(), START_BLOCK);
        assert_eq!(
            builder.new_source_scope(span, lint_level, Some(safety)),
            OUTERMOST_SOURCE_SCOPE);
        builder.source_scopes[OUTERMOST_SOURCE_SCOPE].parent_scope = None;

        builder
    }

    fn finish(self,
              yield_ty: Option<Ty<'tcx>>)
              -> Mir<'tcx> {
        for (index, block) in self.cfg.basic_blocks.iter().enumerate() {
            if block.terminator.is_none() {
                span_bug!(self.fn_span, "no terminator on block {:?}", index);
            }
        }

        Mir::new(
            self.cfg.basic_blocks,
            self.source_scopes,
            ClearCrossCrate::Set(self.source_scope_local_data),
            IndexVec::new(),
            yield_ty,
            self.local_decls,
            self.canonical_user_type_annotations,
            self.arg_count,
            self.upvar_decls,
            self.fn_span,
            self.hir.control_flow_destroyed(),
        )
    }

    fn args_and_body(&mut self,
                     mut block: BasicBlock,
                     arguments: &[ArgInfo<'gcx>],
                     argument_scope: region::Scope,
                     ast_body: &'gcx hir::Expr)
                     -> BlockAnd<()>
    {
        // Allocate locals for the function arguments
        for &ArgInfo(ty, _, pattern, _) in arguments.iter() {
            // If this is a simple binding pattern, give the local a name for
            // debuginfo and so that error reporting knows that this is a user
            // variable. For any other pattern the pattern introduces new
            // variables which will be named instead.
            let mut name = None;
            if let Some(pat) = pattern {
                match pat.node {
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, ident, _)
                    | hir::PatKind::Binding(hir::BindingAnnotation::Mutable, _, ident, _) => {
                        name = Some(ident.name);
                    }
                    _ => (),
                }
            }

            let source_info = SourceInfo {
                scope: OUTERMOST_SOURCE_SCOPE,
                span: pattern.map_or(self.fn_span, |pat| pat.span)
            };
            self.local_decls.push(LocalDecl {
                mutability: Mutability::Mut,
                ty,
                user_ty: UserTypeProjections::none(),
                source_info,
                visibility_scope: source_info.scope,
                name,
                internal: false,
                is_user_variable: None,
                is_block_tail: None,
            });
        }

        let mut scope = None;
        // Bind the argument patterns
        for (index, arg_info) in arguments.iter().enumerate() {
            // Function arguments always get the first Local indices after the return place
            let local = Local::new(index + 1);
            let place = Place::Local(local);
            let &ArgInfo(ty, opt_ty_info, pattern, ref self_binding) = arg_info;

            // Make sure we drop (parts of) the argument even when not matched on.
            self.schedule_drop(
                pattern.as_ref().map_or(ast_body.span, |pat| pat.span),
                argument_scope, &place, ty,
                DropKind::Value { cached_block: CachedBlock::default() },
            );

            if let Some(pattern) = pattern {
                let pattern = self.hir.pattern_from_hir(pattern);
                let span = pattern.span;

                match *pattern.kind {
                    // Don't introduce extra copies for simple bindings
                    PatternKind::Binding { mutability, var, mode: BindingMode::ByValue, .. } => {
                        self.local_decls[local].mutability = mutability;
                        self.local_decls[local].is_user_variable =
                            if let Some(kind) = self_binding {
                                Some(ClearCrossCrate::Set(BindingForm::ImplicitSelf(*kind)))
                            } else {
                                let binding_mode = ty::BindingMode::BindByValue(mutability.into());
                                Some(ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                                    binding_mode,
                                    opt_ty_info,
                                    opt_match_place: Some((Some(place.clone()), span)),
                                    pat_span: span,
                                })))
                            };
                        self.var_indices.insert(var, LocalsForNode::One(local));
                    }
                    _ => {
                        scope = self.declare_bindings(scope, ast_body.span,
                                                      LintLevel::Inherited, &[pattern.clone()],
                                                      matches::ArmHasGuard(false),
                                                      Some((Some(&place), span)));
                        unpack!(block = self.place_into_pattern(block, pattern, &place, false));
                    }
                }
            }
        }

        // Enter the argument pattern bindings source scope, if it exists.
        if let Some(source_scope) = scope {
            self.source_scope = source_scope;
        }

        let body = self.hir.mirror(ast_body);
        self.into(&Place::Local(RETURN_PLACE), block, body)
    }

    fn get_unit_temp(&mut self) -> Place<'tcx> {
        match self.unit_temp {
            Some(ref tmp) => tmp.clone(),
            None => {
                let ty = self.hir.unit_ty();
                let fn_span = self.fn_span;
                let tmp = self.temp(ty, fn_span);
                self.unit_temp = Some(tmp.clone());
                tmp
            }
        }
    }

    fn return_block(&mut self) -> BasicBlock {
        match self.cached_return_block {
            Some(rb) => rb,
            None => {
                let rb = self.cfg.start_new_block();
                self.cached_return_block = Some(rb);
                rb
            }
        }
    }

    fn unreachable_block(&mut self) -> BasicBlock {
        match self.cached_unreachable_block {
            Some(ub) => ub,
            None => {
                let ub = self.cfg.start_new_block();
                self.cached_unreachable_block = Some(ub);
                ub
            }
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
mod into;
mod matches;
mod misc;
mod scope;
