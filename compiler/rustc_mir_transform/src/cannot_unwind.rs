//! Infers from a function's MIR body whether an unwind can leave its frame.
//!
//! `ty::layout::fn_can_unwind` only consults the ABI and the attributes, so it treats every
//! plain Rust function as able to unwind. This walks the body instead: a call to a callee
//! proven not to unwind cannot unwind, nor can a drop of a type whose glue is proven not to.
//! The answer is a fixpoint over the call graph, and a cycle yields no proof.
//!
//! `false` means "no proof was found", not "this function unwinds".
//!
//! The answer comes from `optimized_mir`, so it must agree with what codegen emits; anything
//! changing a body afterwards invalidates it, as for `deduce_param_attrs`.

use rustc_abi::ExternAbi;
use rustc_ast::InlineAsmOptions;
use rustc_attr_ir::lang_items::LangItem;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::traversal::reachable_as_bitset;
use rustc_middle::mir::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, layout};
use rustc_session::config::OptLevel;

pub(super) fn provide(providers: &mut Providers) {
    providers.cannot_unwind = cannot_unwind;
    providers.cannot_unwind_shim = cannot_unwind_shim;
}

/// The same analysis, on a body the compiler writes for itself.
///
/// Drop glue is the case that matters: the most common call target with no `DefId` behind it.
/// Every crate that instantiates a shim generates the same body, so the answer needs no
/// metadata, unlike `cannot_unwind`.
fn cannot_unwind_shim<'tcx>(tcx: TyCtxt<'tcx>, shim: ty::ShimKind<'tcx>) -> bool {
    if tcx.sess.opts.optimize == OptLevel::No {
        return false;
    }

    // Under abort `fn_can_unwind` already answers "cannot unwind" for every Rust-ABI function
    // and shim without looking at any body, and the unwind edges are gone anyway.
    if !tcx.sess.panic_strategy().unwinds() {
        return false;
    }

    // As for items, only Rust-like ABIs leave the claim to us. A shim built from something
    // other than a function has no signature to ask about.
    let abi = match tcx.def_kind(shim.def_id()) {
        DefKind::Fn | DefKind::AssocFn => tcx.fn_sig(shim.def_id()).skip_binder().abi(),
        _ => return false,
    };
    if !abi.is_rustic_abi() {
        return false;
    }

    let body = tcx.mir_shims(shim);
    if body.tainted_by_errors.is_some() {
        return false;
    }

    let mut analyzer = Analyzer {
        tcx,
        drop_glue_in_progress: FxHashSet::default(),
        drop_glue_cache: FxHashMap::default(),
    };
    analyzer.body_cannot_unwind(body)
}

fn cannot_unwind<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> bool {
    // The answer only reaches the backend in optimized builds, and only there can it remove
    // anything.
    if tcx.sess.opts.optimize == OptLevel::No {
        return false;
    }

    // Under `panic=abort` `fn_can_unwind` already answers "cannot unwind" for every non-foreign
    // Rust function without looking at a body, and every unwind edge is already gone.
    if !tcx.sess.panic_strategy().unwinds() {
        return false;
    }

    // Functions with no body (foreign declarations, compiler-builtins, ...).
    if !tcx.is_mir_available(def_id) {
        return false;
    }

    // Comptime functions never reach codegen.
    if let hir::Constness::Const { always: true } = tcx.constness(def_id) {
        return false;
    }

    // The emitted code for an intrinsic need not be its MIR body: a backend-implemented one
    // gets a placeholder body of `unreachable`, which would prove anything.
    if tcx.intrinsic(def_id.to_def_id()).is_some() {
        return false;
    }

    // `drop_glue` and `async_drop_in_place` carry empty placeholder bodies in `core`; the
    // compiler substitutes generated glue for them. The placeholder would prove anything.
    if tcx.is_lang_item(def_id.to_def_id(), LangItem::DropGlue)
        || tcx.is_lang_item(def_id.to_def_id(), LangItem::AsyncDropInPlace)
    {
        return false;
    }

    // Only Rust-like ABIs are in scope. An ABI such as `C-unwind` describes the interface, not
    // the body: callers through that boundary are entitled to unwind out of it whatever this
    // body does.
    let abi = match tcx.def_kind(def_id) {
        // A closure or coroutine has no `fn_sig` of its own, but is always `extern "Rust"`.
        DefKind::Closure => ExternAbi::Rust,
        DefKind::Fn | DefKind::AssocFn => tcx.fn_sig(def_id).skip_binder().abi(),
        _ => return false,
    };
    if !abi.is_rustic_abi() {
        return false;
    }

    // A naked function's body is a single `asm!` block that can jump anywhere.
    if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::NAKED) {
        return false;
    }

    let body = tcx.optimized_mir(def_id);
    if body.tainted_by_errors.is_some() {
        return false;
    }

    let mut analyzer = Analyzer {
        tcx,
        drop_glue_in_progress: FxHashSet::default(),
        drop_glue_cache: FxHashMap::default(),
    };
    analyzer.body_cannot_unwind(body)
}

struct Analyzer<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// Types whose drop glue is currently being analysed, to break recursive types. Meeting one
    /// again means "no proof", which is the safe answer.
    drop_glue_in_progress: FxHashSet<Ty<'tcx>>,
    /// Answers already computed for a type's drop glue. Without this the work is exponential in
    /// the nesting depth.
    drop_glue_cache: FxHashMap<Ty<'tcx>, bool>,
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
                // `Unreachable` is written by passes that consult `fn_can_unwind` of the
                // declaration rather than of the body that will run, so it is not a proof. A
                // call the ABI or an attribute does rule out never gets this far, so reaching
                // here means the callee's own answer disagrees and must not be taken on trust.
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
        let mut escapes = DenseBitSet::new_empty(body.basic_blocks.len());
        loop {
            let mut changed = false;
            for (bb, block) in body.basic_blocks.iter_enumerated() {
                if !escapes.contains(bb) && self.block_escapes(body, block, &escapes) {
                    escapes.insert(bb);
                    changed = true;
                }
            }
            if !changed {
                return escapes;
            }
        }
    }

    fn block_escapes(
        &mut self,
        body: &Body<'tcx>,
        block: &BasicBlockData<'tcx>,
        escapes: &DenseBitSet<BasicBlock>,
    ) -> bool {
        use TerminatorKind::*;

        let terminator = block.terminator();
        match &terminator.kind {
            UnwindResume => return true,
            // These stop the unwind: `Terminate` aborts, and the rest cannot be reached by an
            // unwind at all.
            UnwindTerminate(_) | CoroutineDrop | Return | Unreachable => return false,
            // A tail call has no unwind edge of its own (see `body_cannot_unwind`), so the block
            // escapes exactly when the callee can unwind.
            TailCall { func, .. } => return self.callee_can_unwind(body, func),
            _ => {}
        }

        // Normal control flow inside the block: if it leads somewhere that escapes, so does this.
        let normal_escapes = match &terminator.kind {
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
            UnwindResume
            | UnwindTerminate(_)
            | CoroutineDrop
            | Return
            | Unreachable
            | TailCall { .. } => unreachable!("handled above"),
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
            // See `body_cannot_unwind`: `Unreachable` is not a proof, so an unwind arriving here
            // keeps going.
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
        let (def_id, args) = match *ty.kind() {
            // An indirect call could go anywhere.
            ty::FnPtr(..) => return true,
            ty::FnDef(def_id, args) => (def_id, args),
            _ => return true,
        };

        // A callee whose arguments still mention late-bound generics or generic parameters of
        // this body cannot be named without instantiating it, and resolution would then go
        // through the trait solver with a non-concrete type, which assumes monomorphization.
        let callee = args
            .no_bound_vars()
            .filter(|args| !args.has_non_region_param() && !args.has_aliases())
            .and_then(|args| self.resolved_callee(body.typing_env(self.tcx), def_id, args));

        let abi = ty.fn_sig(self.tcx).abi();
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
            // No single body to name, which is the position codegen is in for a virtual call.
            // The declaration's `#[rustc_nounwind]` is deliberately not relayed: it is a promise
            // about one body, and the body that runs belongs to an impl this crate may not see.
            None => layout::fn_can_unwind(self.tcx, None, abi),
        }
    }

    /// The body codegen will emit for a direct call through this `FnDef`, if there is exactly
    /// one and it can be named.
    ///
    /// The `DefId` behind a `FnDef` is the declaration, and for two kinds of declaration that is
    /// not the body that runs: a trait method, where `Self` picks the impl, and a lang item the
    /// compiler substitutes wholesale (`drop_glue`, `async_drop_in_place`, whose empty
    /// placeholder bodies must never be read).
    ///
    /// When resolution fails, a trait method has no body to name, while anything else is its own
    /// body for every instantiation.
    fn resolved_callee(
        &self,
        typing_env: ty::TypingEnv<'tcx>,
        def_id: DefId,
        args: ty::GenericArgs<'tcx>,
    ) -> Option<Callee<'tcx>> {
        // `Instance::try_resolve` asserts on the kinds it will answer for; anything else is not
        // a `FnDef` callee in the first place.
        let resolved = if matches!(
            self.tcx.def_kind(def_id),
            DefKind::Fn
                | DefKind::AssocFn
                | DefKind::Closure
                | DefKind::Ctor(_, rustc_hir::def::CtorKind::Fn)
        ) {
            ty::Instance::try_resolve(self.tcx, typing_env, def_id, args).ok().flatten()
        } else {
            None
        };
        match resolved.map(|instance| instance.def) {
            Some(ty::InstanceKind::Item(callee)) => Some(Callee::Item(callee)),
            Some(ty::InstanceKind::Shim(shim)) => Some(Callee::Shim(shim)),
            // Nothing one body can be named for: a trait method's impl is still open. For
            // anything else the `DefId` is the body for every instantiation.
            _ if self.tcx.trait_of_assoc(def_id).is_some() => None,
            _ => Some(Callee::Item(def_id)),
        }
    }

    fn drop_can_unwind(&mut self, body: &Body<'tcx>, place: &Place<'tcx>) -> bool {
        let ty = place.ty(&body.local_decls, self.tcx).ty;
        self.ty_drop_can_unwind(body.typing_env(self.tcx), ty)
    }

    /// Whether dropping a value of this type can unwind.
    fn ty_drop_can_unwind(&mut self, typing_env: ty::TypingEnv<'tcx>, ty: Ty<'tcx>) -> bool {
        if !ty.needs_drop(self.tcx, typing_env) {
            return false;
        }

        // Resolving drop glue needs a concrete type, and a generic body is analysed before
        // monomorphization. An alias hides the type that will actually be dropped.
        if ty.has_non_region_param() || ty.has_aliases() {
            return true;
        }

        // A type whose glue is already being analysed is recursive; give up on proving it.
        if self.drop_glue_in_progress.contains(&ty) {
            return true;
        }
        if let Some(&cannot_unwind) = self.drop_glue_cache.get(&ty) {
            return !cannot_unwind;
        }

        self.drop_glue_in_progress.insert(ty);
        // `Instance::resolve_drop_glue` panics on failure, which is fine at monomorphization
        // time but not here; "no instance" means "no proof".
        let Some(glue) = self.resolve_drop_glue(ty) else {
            self.drop_glue_in_progress.remove(&ty);
            self.drop_glue_cache.insert(ty, false);
            return true;
        };
        let cannot_unwind = match glue.def {
            // Going through the query lets the cycle handler stop the recursion, rather than
            // this set having to reason about shim identities as well as types.
            ty::InstanceKind::Shim(shim) => self.tcx.cannot_unwind_shim(shim),
            _ => false,
        };
        self.drop_glue_in_progress.remove(&ty);
        self.drop_glue_cache.insert(ty, cannot_unwind);

        !cannot_unwind
    }

    /// The drop glue instance for `ty`, or `None` if it does not resolve.
    ///
    /// This is `Instance::resolve_drop_glue` without the panic, which asserts it is being called
    /// on a monomorphic type.
    fn resolve_drop_glue(&self, ty: Ty<'tcx>) -> Option<ty::Instance<'tcx>> {
        let def_id = self
            .tcx
            .require_lang_item(rustc_attr_ir::lang_items::LangItem::DropGlue, rustc_span::DUMMY_SP);
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
