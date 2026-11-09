//! Infers from a function's MIR body whether an unwind can leave its frame.
//!
//! `ty::layout::fn_can_unwind` only consults the ABI and the attributes, so it treats every plain
//! Rust function as able to unwind. This walks the body instead: a call to a callee proven not to
//! unwind cannot unwind, nor can a drop of a type whose glue is proven not to. The answer is a
//! fixpoint over the call graph; a cycle yields no proof, except a call back into the body being
//! analyzed (see `Subject`). `false` means "no proof was found", not "this function unwinds".
//!
//! The answer comes from `optimized_mir`, so anything changing a body afterwards invalidates it,
//! as for `deduce_param_attrs`.

use rustc_abi::ExternAbi;
use rustc_ast::InlineAsmOptions;
use rustc_attr_ir::lang_items::LangItem;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::traversal::reachable_as_bitset;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, layout};
use rustc_session::config::OptLevel;
use rustc_span::DUMMY_SP;

pub(super) fn provide(providers: &mut Providers) {
    providers.cannot_unwind = cannot_unwind;
    providers.cannot_unwind_shim = cannot_unwind_shim;
    providers.cannot_unwind_instance = cannot_unwind_instance;
}

/// Whether the answer can reach codegen at all: only optimized builds read it, and under
/// `panic=abort` `fn_can_unwind` already says "cannot unwind" for every Rust-ABI item.
fn should_analyze(tcx: TyCtxt<'_>) -> bool {
    tcx.sess.opts.optimize != OptLevel::No && tcx.sess.panic_strategy().unwinds()
}

/// The ABI a definition is called with, if it is function-like.
fn body_abi(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ExternAbi> {
    match tcx.def_kind(def_id) {
        DefKind::Fn | DefKind::AssocFn => Some(tcx.fn_sig(def_id).skip_binder().abi()),
        // A closure or coroutine has no `fn_sig`; both are called as `extern "Rust"`, as is a
        // tuple constructor.
        DefKind::Closure | DefKind::SyntheticCoroutineBody | DefKind::Ctor(_, CtorKind::Fn) => {
            Some(ExternAbi::Rust)
        }
        // Nothing else is called.
        _ => None,
    }
}

/// Whether the MIR body is what codegen emits for `def_id`: a comptime function never reaches
/// codegen, an intrinsic and the drop-glue lang items are generated rather than read from the
/// placeholder body, and a naked function's body is a single `asm!` block that can jump anywhere.
fn body_is_emitted(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if let hir::Constness::Const { always: true } = tcx.constness(def_id) {
        return false;
    }
    tcx.intrinsic(def_id).is_none()
        && !tcx.is_lang_item(def_id, LangItem::DropGlue)
        && !tcx.is_lang_item(def_id, LangItem::AsyncDropInPlace)
        && !tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::NAKED)
}

/// The ABI of the declaration a shim is built from, if it is one that can be called.
fn shim_abi<'tcx>(tcx: TyCtxt<'tcx>, shim: ty::ShimKind<'tcx>) -> Option<ExternAbi> {
    match tcx.def_kind(shim.def_id()) {
        DefKind::Fn | DefKind::AssocFn => Some(tcx.fn_sig(shim.def_id()).skip_binder().abi()),
        // A shim built from something other than a function has no signature to ask about.
        _ => None,
    }
}

/// The same question about one monomorphization of the body.
fn cannot_unwind_instance<'tcx>(tcx: TyCtxt<'tcx>, instance: ty::Instance<'tcx>) -> bool {
    if !should_analyze(tcx) {
        return false;
    }

    match instance.def {
        ty::InstanceKind::Item(def_id) => {
            if !body_abi(tcx, def_id).is_some_and(ExternAbi::is_rustic_abi) {
                return false;
            }
            // No MIR: a foreign item was answered when its own crate was compiled.
            if !tcx.is_mir_available(def_id) {
                return tcx.cannot_unwind(def_id);
            }
            if !body_is_emitted(tcx, def_id) {
                return false;
            }
        }
        ty::InstanceKind::Shim(shim) => {
            if !shim_abi(tcx, shim).is_some_and(ExternAbi::is_rustic_abi) {
                return false;
            }
        }
        // No body to read, or more than one.
        ty::InstanceKind::Intrinsic(_)
        | ty::InstanceKind::LlvmIntrinsic(_)
        | ty::InstanceKind::Virtual(..) => return false,
    }

    let body = tcx.instance_mir(instance.def);
    if body.tainted_by_errors.is_some() {
        return false;
    }

    let mut analyzer = Analyzer {
        tcx,
        subject: Subject::Instance(instance),
        typing_env: ty::TypingEnv::fully_monomorphized(),
        drop_glue_cache: FxHashMap::default(),
    };
    analyzer.body_cannot_unwind(body)
}

/// The same analysis, on a body the compiler writes for itself.
///
/// Drop glue is the case that matters, and every crate instantiates a shim with the same body, so
/// unlike `cannot_unwind` the answer needs no metadata.
fn cannot_unwind_shim<'tcx>(tcx: TyCtxt<'tcx>, shim: ty::ShimKind<'tcx>) -> bool {
    if !should_analyze(tcx) {
        return false;
    }

    // As for items, only Rust-like ABIs leave the claim to us.
    if !shim_abi(tcx, shim).is_some_and(ExternAbi::is_rustic_abi) {
        return false;
    }

    let body = tcx.mir_shims(shim);
    if body.tainted_by_errors.is_some() {
        return false;
    }

    let mut analyzer = Analyzer {
        tcx,
        subject: Subject::Shim(shim),
        typing_env: body.typing_env(tcx),
        drop_glue_cache: FxHashMap::default(),
    };
    analyzer.body_cannot_unwind(body)
}

fn cannot_unwind<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> bool {
    if !should_analyze(tcx) {
        return false;
    }

    // Functions with no body (foreign declarations, compiler-builtins, ...).
    if !tcx.is_mir_available(def_id) {
        return false;
    }

    // Only Rust-like ABIs are in scope. An ABI such as `C-unwind` describes the interface, not
    // the body: callers through that boundary are entitled to unwind out of it whatever this
    // body does.
    if !body_abi(tcx, def_id.to_def_id()).is_some_and(ExternAbi::is_rustic_abi) {
        return false;
    }

    if !body_is_emitted(tcx, def_id.to_def_id()) {
        return false;
    }

    let body = tcx.optimized_mir(def_id);
    if body.tainted_by_errors.is_some() {
        return false;
    }

    let mut analyzer = Analyzer {
        tcx,
        subject: Subject::Item(def_id.to_def_id()),
        typing_env: body.typing_env(tcx),
        drop_glue_cache: FxHashMap::default(),
    };
    analyzer.body_cannot_unwind(body)
}

struct Analyzer<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// The body this analysis started from.
    subject: Subject<'tcx>,
    /// The environment the body's types are read in.
    typing_env: ty::TypingEnv<'tcx>,
    /// Answers already computed for a type's drop glue. Without this the work is exponential in
    /// the nesting depth.
    drop_glue_cache: FxHashMap<Ty<'tcx>, bool>,
}

/// The one body whose frame the question is about.
///
/// A call back into this body counts as unable to unwind: the walk sees every other operation, so
/// an unwind leaving the frame was let out by one of those. The assumption is not memoized, and a
/// cycle through any other body yields no proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Subject<'tcx> {
    /// A function's own definition, before monomorphization.
    Item(DefId),
    /// One monomorphization.
    Instance(ty::Instance<'tcx>),
    /// A body the compiler writes for itself.
    Shim(ty::ShimKind<'tcx>),
}

impl<'tcx> Subject<'tcx> {
    /// The monomorphization being analyzed, if this is one.
    fn instance(self) -> Option<ty::Instance<'tcx>> {
        match self {
            Subject::Instance(instance) => Some(instance),
            Subject::Item(_) | Subject::Shim(_) => None,
        }
    }
}

/// The one body a `FnDef` in a call resolves to.
#[derive(Clone, Copy, Debug)]
enum Callee<'tcx> {
    Item(DefId),
    Shim(ty::ShimKind<'tcx>),
}

impl<'tcx> Analyzer<'tcx> {
    /// Returns `true` if nothing in this body can let an unwind leave the frame.
    fn body_cannot_unwind(&mut self, body: &Body<'tcx>) -> bool {
        let reachable = reachable_as_bitset(body);
        let escapes = self.compute_escape(body);

        for (bb, block) in body.basic_blocks.iter_enumerated() {
            if !reachable.contains(bb) {
                continue;
            }
            let terminator = block.terminator();

            // A `resume` in normal control flow is an unwind being resumed deliberately.
            if matches!(terminator.kind, TerminatorKind::UnwindResume) && !block.is_cleanup {
                return false;
            }

            // A tail call has no unwind target of its own: it runs in this frame, so whatever the
            // callee does to unwind leaves this frame as well.
            if let TerminatorKind::TailCall { func, .. } = &terminator.kind {
                if self.callee_can_unwind(body, func) {
                    return false;
                }
                continue;
            }

            if !self.terminator_can_unwind(body, terminator) {
                continue;
            }
            let Some(unwind) = terminator.unwind() else {
                // Claimed to be able to unwind but has nowhere to send one; nothing leaves here.
                continue;
            };
            match unwind {
                // The unwind keeps going, so it leaves this frame.
                UnwindAction::Continue => return false,
                // Terminating aborts the process, so nothing unwinds past this point.
                UnwindAction::Terminate(_) => {}
                UnwindAction::Cleanup(target) => {
                    if escapes.contains(*target) {
                        return false;
                    }
                }
                // Written from the declaration's answer, not the body's, so it is no proof.
                UnwindAction::Unreachable => return false,
            }
        }

        true
    }

    /// For every block, whether an unwind that arrives at it would leave the frame.
    ///
    /// The fixpoint is over both normal and unwind edges because a cleanup chain can do real
    /// work (a `drop`, say) between the block that started the unwind and the `resume`.
    fn compute_escape(&mut self, body: &Body<'tcx>) -> DenseBitSet<BasicBlock> {
        // The answer only grows, so a block needs re-examining exactly when a successor starts
        // escaping.
        let mut escapes = DenseBitSet::new_empty(body.basic_blocks.len());
        let mut worklist: Vec<BasicBlock> = body.basic_blocks.indices().collect();
        while let Some(bb) = worklist.pop() {
            if escapes.contains(bb) || !self.block_escapes(body, &body.basic_blocks[bb], &escapes) {
                continue;
            }
            escapes.insert(bb);
            worklist.extend(body.basic_blocks.predecessors()[bb].iter().copied());
        }
        escapes
    }

    fn block_escapes(
        &mut self,
        body: &Body<'tcx>,
        block: &BasicBlockData<'tcx>,
        escapes: &DenseBitSet<BasicBlock>,
    ) -> bool {
        use TerminatorKind::*;

        let terminator = block.terminator();
        // Normal control flow: if it leads somewhere that escapes, so does this.
        let normal_escapes = match &terminator.kind {
            UnwindResume => return true,
            // `Terminate` aborts, and no unwind can arrive at the rest.
            UnwindTerminate(_) | CoroutineDrop | Return | Unreachable => return false,
            // A tail call has no unwind edge of its own (see `body_cannot_unwind`), so the block
            // escapes exactly when the callee can unwind.
            TailCall { func, .. } => return self.callee_can_unwind(body, func),
            Goto { target, .. } => escapes.contains(*target),
            SwitchInt { targets, .. } => targets.all_targets().iter().any(|t| escapes.contains(*t)),
            Call { target, .. } => target.is_some_and(|t| escapes.contains(t)),
            Drop { target, drop, .. } => {
                escapes.contains(*target) || drop.is_some_and(|d| escapes.contains(d))
            }
            Assert { target, .. } => escapes.contains(*target),
            Yield { resume, drop, .. } => {
                escapes.contains(*resume) || drop.is_some_and(|d| escapes.contains(d))
            }
            FalseEdge { real_target, imaginary_target } => {
                escapes.contains(*real_target) || escapes.contains(*imaginary_target)
            }
            FalseUnwind { real_target, .. } => escapes.contains(*real_target),
            InlineAsm { targets, .. } => targets.iter().any(|t| escapes.contains(*t)),
        };
        if normal_escapes {
            return true;
        }

        if !self.terminator_can_unwind(body, terminator) {
            return false;
        }
        let Some(unwind) = terminator.unwind() else {
            // No unwind edge to send it to, so there is no way for an unwind to leave here.
            return false;
        };
        match unwind {
            UnwindAction::Continue => true,
            // As in `body_cannot_unwind`, `Unreachable` is no proof, so the unwind keeps going.
            UnwindAction::Unreachable => true,
            UnwindAction::Terminate(_) => false,
            UnwindAction::Cleanup(target) => escapes.contains(*target),
        }
    }

    /// Whether this terminator can unwind at all, ignoring where the unwind goes.
    fn terminator_can_unwind(&mut self, body: &Body<'tcx>, terminator: &Terminator<'tcx>) -> bool {
        match &terminator.kind {
            TerminatorKind::Call { func, .. } => self.callee_can_unwind(body, func),
            TerminatorKind::TailCall { func, .. } => self.callee_can_unwind(body, func),
            TerminatorKind::Drop { place, .. } => {
                let panic_in_drop = self.tcx.sess.opts.unstable_opts.panic_in_drop;
                panic_in_drop.unwinds() && self.drop_can_unwind(body, place)
            }
            // These panic through the usual machinery, whose unwind-ness is a property of the
            // panic strategy rather than of anything in this body.
            TerminatorKind::Assert { .. } | TerminatorKind::FalseUnwind { .. } => {
                layout::fn_can_unwind(self.tcx, None, ExternAbi::Rust)
            }
            TerminatorKind::InlineAsm { options, .. } => {
                options.contains(InlineAsmOptions::MAY_UNWIND)
            }
            _ => terminator.unwind().is_some(),
        }
    }

    fn callee_can_unwind(&mut self, body: &Body<'tcx>, func: &Operand<'tcx>) -> bool {
        let ty = func.ty(&body.local_decls, self.tcx);
        let ty = self.concretize(ty);
        // An indirect call could go anywhere, and nothing else is called.
        let ty::FnDef(def_id, args) = *ty.kind() else { return true };
        let abi = ty.fn_sig(self.tcx).abi();

        // The trait solver assumes monomorphization, so a callee that still mentions a parameter
        // of this body is not named.
        let args =
            args.no_bound_vars().filter(|args| !args.has_non_region_param() && !args.has_aliases());

        let resolved = args.and_then(|args| self.resolved_instance(def_id, args));
        let callee = self.resolved_callee(def_id, resolved);

        // A call back into this body asks the question already being answered; see `Subject`.
        if self.is_self_call(resolved, callee) {
            return false;
        }

        // Inside a monomorphization this names the one body codegen emits.
        if matches!(self.subject, Subject::Instance(_))
            && let Some(instance) = resolved
            && !matches!(instance.def, ty::InstanceKind::Virtual(..))
        {
            return layout::fn_can_unwind(self.tcx, Some(instance.def.def_id()), abi)
                && !self.tcx.cannot_unwind_instance(instance);
        }

        match callee {
            // One body is known: ask the same question codegen asks about it.
            Some(Callee::Item(callee)) => {
                layout::fn_can_unwind(self.tcx, Some(callee), abi)
                    && !self.tcx.cannot_unwind(callee)
            }
            Some(Callee::Shim(shim)) => {
                layout::fn_can_unwind(self.tcx, Some(shim.def_id()), abi)
                    && !self.tcx.cannot_unwind_shim(shim)
            }
            // No single body to name, as for a virtual call. The declaration's `#[rustc_nounwind]`
            // is deliberately not relayed: it is a promise about one body.
            None => layout::fn_can_unwind(self.tcx, None, abi),
        }
    }

    /// Whether this call names the body the analysis started from. See `Subject`.
    fn is_self_call(
        &self,
        resolved: Option<ty::Instance<'tcx>>,
        callee: Option<Callee<'tcx>>,
    ) -> bool {
        match self.subject {
            Subject::Instance(subject) => resolved == Some(subject),
            Subject::Item(subject) => {
                matches!(callee, Some(Callee::Item(callee)) if callee == subject)
            }
            Subject::Shim(subject) => {
                matches!(callee, Some(Callee::Shim(shim)) if shim == subject)
            }
        }
    }

    /// The instance `try_resolve` names for a direct call through this `FnDef`.
    fn resolved_instance(
        &self,
        def_id: DefId,
        args: ty::GenericArgs<'tcx>,
    ) -> Option<ty::Instance<'tcx>> {
        // `try_resolve` asserts on the kinds it accepts, so keep the set to the ones a call can
        // name.
        if !matches!(
            self.tcx.def_kind(def_id),
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Ctor(_, CtorKind::Fn)
        ) {
            return None;
        }
        ty::Instance::try_resolve(self.tcx, self.typing_env, def_id, args).ok().flatten()
    }

    /// The body codegen will emit for a direct call through this `FnDef`, if exactly one can be
    /// named.
    fn resolved_callee(
        &self,
        def_id: DefId,
        resolved: Option<ty::Instance<'tcx>>,
    ) -> Option<Callee<'tcx>> {
        match resolved.map(|instance| instance.def) {
            Some(ty::InstanceKind::Item(callee)) => Some(Callee::Item(callee)),
            Some(ty::InstanceKind::Shim(shim)) => Some(Callee::Shim(shim)),
            // A trait method's impl is still open. For anything else the `DefId` is the body
            // for every instantiation.
            _ if self.tcx.trait_of_assoc(def_id).is_some() => None,
            _ => Some(Callee::Item(def_id)),
        }
    }

    fn drop_can_unwind(&mut self, body: &Body<'tcx>, place: &Place<'tcx>) -> bool {
        let ty = place.ty(&body.local_decls, self.tcx).ty;
        let ty = self.concretize(ty);
        self.ty_drop_can_unwind(ty)
    }

    /// Whether dropping a value of this type can unwind.
    fn ty_drop_can_unwind(&mut self, ty: Ty<'tcx>) -> bool {
        if !ty.needs_drop(self.tcx, self.typing_env) {
            return false;
        }

        // Resolving drop glue needs a concrete type. An alias hides the type that will actually
        // be dropped, and a generic parameter names none.
        if ty.has_non_region_param() || ty.has_aliases() {
            return true;
        }

        // A recursive type is cut by the query's cycle handler, which answers "no proof", so only
        // completed answers are cached.
        if let Some(&cannot_unwind) = self.drop_glue_cache.get(&ty) {
            return !cannot_unwind;
        }

        let Some(glue) = self.resolve_drop_glue(ty) else {
            self.drop_glue_cache.insert(ty, false);
            return true;
        };
        let cannot_unwind = match (self.subject.instance().is_some(), glue.def) {
            // Within a monomorphization the glue is one instance, whose own body may drop
            // further types.
            (true, _) => self.tcx.cannot_unwind_instance(glue),
            // The cycle handler ends the recursion on shim identity.
            (false, ty::InstanceKind::Shim(shim)) => self.tcx.cannot_unwind_shim(shim),
            (false, _) => false,
        };
        self.drop_glue_cache.insert(ty, cannot_unwind);

        !cannot_unwind
    }

    /// The concrete form of `ty` inside the monomorphization, or `ty` itself outside one.
    fn concretize(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let Some(instance) = self.subject.instance() else { return ty };
        // The round trip would only erase regions, which nothing below reads.
        if !ty.has_non_region_param() && !ty.has_aliases() {
            return ty;
        }
        instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            self.typing_env,
            ty::EarlyBinder::bind(self.tcx, ty),
        )
    }

    /// The drop glue instance for `ty`, or `None` if it does not resolve.
    ///
    /// `Instance::resolve_drop_glue` panics instead, as it asserts a monomorphic type.
    fn resolve_drop_glue(&self, ty: Ty<'tcx>) -> Option<ty::Instance<'tcx>> {
        let def_id = self.tcx.require_lang_item(LangItem::DropGlue, DUMMY_SP);
        let args = self.tcx.mk_args(&[ty.into()]);
        match ty::Instance::try_resolve(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            def_id,
            args,
        ) {
            Ok(Some(instance)) => Some(instance),
            Ok(None) | Err(_) => None,
        }
    }
}
