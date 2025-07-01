use std::iter;

use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::{Body, Local, UnwindTerminateReason, traversal};
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_middle::{bug, mir, span_bug};
use rustc_target::callconv::{FnAbi, PassMode};
use tracing::{debug, instrument};

use crate::base;
use crate::traits::*;

mod analyze;
mod block;
mod constant;
mod coverageinfo;
pub mod debuginfo;
mod intrinsic;
mod locals;
pub mod naked_asm;
pub mod operand;
pub mod place;
mod rvalue;
mod statement;

pub use self::block::store_cast;
use self::debuginfo::{FunctionDebugContext, PerLocalVarDebugInfo};
use self::operand::{OperandRef, OperandValue};
use self::place::PlaceRef;

// Used for tracking the state of generated basic blocks.
enum CachedLlbb<T> {
    /// Nothing created yet.
    None,

    /// Has been created.
    Some(T),

    /// Nothing created yet, and nothing should be.
    Skip,
}

type PerLocalVarDebugInfoIndexVec<'tcx, V> =
    IndexVec<mir::Local, Vec<PerLocalVarDebugInfo<'tcx, V>>>;

/// Master context for codegenning from MIR.
pub struct FunctionCx<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    instance: Instance<'tcx>,

    mir: &'tcx mir::Body<'tcx>,

    debug_context: Option<FunctionDebugContext<'tcx, Bx::DIScope, Bx::DILocation>>,

    llfn: Bx::Function,

    cx: &'a Bx::CodegenCx,

    fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,

    /// When unwinding is initiated, we have to store this personality
    /// value somewhere so that we can load it and re-use it in the
    /// resume instruction. The personality is (afaik) some kind of
    /// value used for C++ unwinding, which must filter by type: we
    /// don't really care about it very much. Anyway, this value
    /// contains an alloca into which the personality is stored and
    /// then later loaded when generating the DIVERGE_BLOCK.
    personality_slot: Option<PlaceRef<'tcx, Bx::Value>>,

    /// A backend `BasicBlock` for each MIR `BasicBlock`, created lazily
    /// as-needed (e.g. RPO reaching it or another block branching to it).
    // FIXME(eddyb) rename `llbbs` and other `ll`-prefixed things to use a
    // more backend-agnostic prefix such as `cg` (i.e. this would be `cgbbs`).
    cached_llbbs: IndexVec<mir::BasicBlock, CachedLlbb<Bx::BasicBlock>>,

    /// The funclet status of each basic block
    cleanup_kinds: Option<IndexVec<mir::BasicBlock, analyze::CleanupKind>>,

    /// When targeting MSVC, this stores the cleanup info for each funclet BB.
    /// This is initialized at the same time as the `landing_pads` entry for the
    /// funclets' head block, i.e. when needed by an unwind / `cleanup_ret` edge.
    funclets: IndexVec<mir::BasicBlock, Option<Bx::Funclet>>,

    /// This stores the cached landing/cleanup pad block for a given BB.
    // FIXME(eddyb) rename this to `eh_pads`.
    landing_pads: IndexVec<mir::BasicBlock, Option<Bx::BasicBlock>>,

    /// Cached unreachable block
    unreachable_block: Option<Bx::BasicBlock>,

    /// Cached terminate upon unwinding block and its reason
    terminate_block: Option<(Bx::BasicBlock, UnwindTerminateReason)>,

    /// A bool flag for each basic block indicating whether it is a cold block.
    /// A cold block is a block that is unlikely to be executed at runtime.
    cold_blocks: IndexVec<mir::BasicBlock, bool>,

    /// The location where each MIR arg/var/tmp/ret is stored. This is
    /// usually an `PlaceRef` representing an alloca, but not always:
    /// sometimes we can skip the alloca and just store the value
    /// directly using an `OperandRef`, which makes for tighter LLVM
    /// IR. The conditions for using an `OperandRef` are as follows:
    ///
    /// - the type of the local must be judged "immediate" by `is_llvm_immediate`
    /// - the operand must never be referenced indirectly
    ///     - we should not take its address using the `&` operator
    ///     - nor should it appear in a place path like `tmp.a`
    /// - the operand must be defined by an rvalue that can generate immediate
    ///   values
    ///
    /// Avoiding allocs can also be important for certain intrinsics,
    /// notably `expect`.
    locals: locals::Locals<'tcx, Bx::Value>,

    /// All `VarDebugInfo` from the MIR body, partitioned by `Local`.
    /// This is `None` if no variable debuginfo/names are needed.
    per_local_var_debug_info: Option<PerLocalVarDebugInfoIndexVec<'tcx, Bx::DIVariable>>,

    /// Caller location propagated if this function has `#[track_caller]`.
    caller_location: Option<OperandRef<'tcx, Bx::Value>>,
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn monomorphize<T>(&self, value: T) -> T
    where
        T: Copy + TypeFoldable<TyCtxt<'tcx>>,
    {
        debug!("monomorphize: self.instance={:?}", self.instance);
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.cx.tcx(),
            self.cx.typing_env(),
            ty::EarlyBinder::bind(value),
        )
    }
}

enum LocalRef<'tcx, V> {
    Place(PlaceRef<'tcx, V>),
    /// `UnsizedPlace(p)`: `p` itself is a thin pointer (indirect place).
    /// `*p` is the wide pointer that references the actual unsized place.
    /// Every time it is initialized, we have to reallocate the place
    /// and update the wide pointer. That's the reason why it is indirect.
    UnsizedPlace(PlaceRef<'tcx, V>),
    /// The backend [`OperandValue`] has already been generated.
    Operand(OperandRef<'tcx, V>),
    /// Will be a `Self::Operand` once we get to its definition.
    PendingOperand,
}

impl<'tcx, V: CodegenObject> LocalRef<'tcx, V> {
    fn new_operand(layout: TyAndLayout<'tcx>) -> LocalRef<'tcx, V> {
        if layout.is_zst() {
            // Zero-size temporaries aren't always initialized, which
            // doesn't matter because they don't contain data, but
            // we need something sufficiently aligned in the operand.
            LocalRef::Operand(OperandRef::zero_sized(layout))
        } else {
            LocalRef::PendingOperand
        }
    }
}

///////////////////////////////////////////////////////////////////////////

#[instrument(level = "debug", skip(cx))]
pub fn codegen_mir<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>,
) {
    assert!(!instance.args.has_infer());

    let tcx = cx.tcx();
    let llfn = cx.get_fn(instance);

    let mut mir = tcx.instance_mir(instance.def);

    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());
    debug!("fn_abi: {:?}", fn_abi);

    if tcx.features().ergonomic_clones() {
        let monomorphized_mir = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(mir.clone()),
        );
        mir = tcx.arena.alloc(optimize_use_clone::<Bx>(cx, monomorphized_mir));
    }

    let debug_context = cx.create_function_debug_context(instance, fn_abi, llfn, &mir);

    let start_llbb = Bx::append_block(cx, llfn, "start");
    let mut start_bx = Bx::build(cx, start_llbb);

    if mir.basic_blocks.iter().any(|bb| {
        bb.is_cleanup || matches!(bb.terminator().unwind(), Some(mir::UnwindAction::Terminate(_)))
    }) {
        start_bx.set_personality_fn(cx.eh_personality());
    }

    let cleanup_kinds =
        base::wants_new_eh_instructions(tcx.sess).then(|| analyze::cleanup_kinds(&mir));

    let cached_llbbs: IndexVec<mir::BasicBlock, CachedLlbb<Bx::BasicBlock>> =
        mir.basic_blocks
            .indices()
            .map(|bb| {
                if bb == mir::START_BLOCK { CachedLlbb::Some(start_llbb) } else { CachedLlbb::None }
            })
            .collect();

    let mut fx = FunctionCx {
        instance,
        mir,
        llfn,
        fn_abi,
        cx,
        personality_slot: None,
        cached_llbbs,
        unreachable_block: None,
        terminate_block: None,
        cleanup_kinds,
        landing_pads: IndexVec::from_elem(None, &mir.basic_blocks),
        funclets: IndexVec::from_fn_n(|_| None, mir.basic_blocks.len()),
        cold_blocks: find_cold_blocks(tcx, mir),
        locals: locals::Locals::empty(),
        debug_context,
        per_local_var_debug_info: None,
        caller_location: None,
    };

    // It may seem like we should iterate over `required_consts` to ensure they all successfully
    // evaluate; however, the `MirUsedCollector` already did that during the collection phase of
    // monomorphization, and if there is an error during collection then codegen never starts -- so
    // we don't have to do it again.

    let (per_local_var_debug_info, consts_debug_info) =
        fx.compute_per_local_var_debug_info(&mut start_bx).unzip();
    fx.per_local_var_debug_info = per_local_var_debug_info;

    let traversal_order = traversal::mono_reachable_reverse_postorder(mir, tcx, instance);
    let memory_locals = analyze::non_ssa_locals(&fx, &traversal_order);

    // Allocate variable and temp allocas
    let local_values = {
        let args = arg_local_refs(&mut start_bx, &mut fx, &memory_locals);

        let mut allocate_local = |local: Local| {
            let decl = &mir.local_decls[local];
            let layout = start_bx.layout_of(fx.monomorphize(decl.ty));
            assert!(!layout.ty.has_erasable_regions());

            if local == mir::RETURN_PLACE {
                match fx.fn_abi.ret.mode {
                    PassMode::Indirect { .. } => {
                        debug!("alloc: {:?} (return place) -> place", local);
                        let llretptr = start_bx.get_param(0);
                        return LocalRef::Place(PlaceRef::new_sized(llretptr, layout));
                    }
                    PassMode::Cast { ref cast, .. } => {
                        debug!("alloc: {:?} (return place) -> place", local);
                        let size = cast.size(&start_bx).max(layout.size);
                        return LocalRef::Place(PlaceRef::alloca_size(&mut start_bx, size, layout));
                    }
                    _ => {}
                };
            }

            if memory_locals.contains(local) {
                debug!("alloc: {:?} -> place", local);
                if layout.is_unsized() {
                    LocalRef::UnsizedPlace(PlaceRef::alloca_unsized_indirect(&mut start_bx, layout))
                } else {
                    LocalRef::Place(PlaceRef::alloca(&mut start_bx, layout))
                }
            } else {
                debug!("alloc: {:?} -> operand", local);
                LocalRef::new_operand(layout)
            }
        };

        let retptr = allocate_local(mir::RETURN_PLACE);
        iter::once(retptr)
            .chain(args.into_iter())
            .chain(mir.vars_and_temps_iter().map(allocate_local))
            .collect()
    };
    fx.initialize_locals(local_values);

    // Apply debuginfo to the newly allocated locals.
    fx.debug_introduce_locals(&mut start_bx, consts_debug_info.unwrap_or_default());

    // If the backend supports coverage, and coverage is enabled for this function,
    // do any necessary start-of-function codegen (e.g. locals for MC/DC bitmaps).
    start_bx.init_coverage(instance);

    // The builders will be created separately for each basic block at `codegen_block`.
    // So drop the builder of `start_llbb` to avoid having two at the same time.
    drop(start_bx);

    let mut unreached_blocks = DenseBitSet::new_filled(mir.basic_blocks.len());
    // Codegen the body of each reachable block using our reverse postorder list.
    for bb in traversal_order {
        fx.codegen_block(bb);
        unreached_blocks.remove(bb);
    }

    // FIXME: These empty unreachable blocks are *mostly* a waste. They are occasionally
    // targets for a SwitchInt terminator, but the reimplementation of the mono-reachable
    // simplification in SwitchInt lowering sometimes misses cases that
    // mono_reachable_reverse_postorder manages to figure out.
    // The solution is to do something like post-mono GVN. But for now we have this hack.
    for bb in unreached_blocks.iter() {
        fx.codegen_block_as_unreachable(bb);
    }
}

// FIXME: Move this function to mir::transform when post-mono MIR passes land.
fn optimize_use_clone<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    mut mir: Body<'tcx>,
) -> Body<'tcx> {
    let tcx = cx.tcx();

    if tcx.features().ergonomic_clones() {
        for bb in mir.basic_blocks.as_mut() {
            let mir::TerminatorKind::Call {
                args,
                destination,
                target,
                call_source: mir::CallSource::Use,
                ..
            } = &bb.terminator().kind
            else {
                continue;
            };

            // CallSource::Use calls always use 1 argument.
            assert_eq!(args.len(), 1);
            let arg = &args[0];

            // These types are easily available from locals, so check that before
            // doing DefId lookups to figure out what we're actually calling.
            let arg_ty = arg.node.ty(&mir.local_decls, tcx);

            let ty::Ref(_region, inner_ty, mir::Mutability::Not) = *arg_ty.kind() else { continue };

            if !tcx.type_is_copy_modulo_regions(cx.typing_env(), inner_ty) {
                continue;
            }

            let Some(arg_place) = arg.node.place() else { continue };

            let destination_block = target.unwrap();

            bb.statements.push(mir::Statement::new(
                bb.terminator().source_info,
                mir::StatementKind::Assign(Box::new((
                    *destination,
                    mir::Rvalue::Use(mir::Operand::Copy(
                        arg_place.project_deeper(&[mir::ProjectionElem::Deref], tcx),
                    )),
                ))),
            ));

            bb.terminator_mut().kind = mir::TerminatorKind::Goto { target: destination_block };
        }
    }

    mir
}

/// Produces, for each argument, a `Value` pointing at the
/// argument's value. As arguments are places, these are always
/// indirect.
fn arg_local_refs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    fx: &mut FunctionCx<'a, 'tcx, Bx>,
    memory_locals: &DenseBitSet<mir::Local>,
) -> Vec<LocalRef<'tcx, Bx::Value>> {
    let mir = fx.mir;
    let mut idx = 0;
    let mut llarg_idx = fx.fn_abi.ret.is_indirect() as usize;

    let mut num_untupled = None;

    let codegen_fn_attrs = bx.tcx().codegen_fn_attrs(fx.instance.def_id());
    let naked = codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED);
    if naked {
        return vec![];
    }

    let args = mir
        .args_iter()
        .enumerate()
        .map(|(arg_index, local)| {
            let arg_decl = &mir.local_decls[local];
            let arg_ty = fx.monomorphize(arg_decl.ty);

            if Some(local) == mir.spread_arg {
                // This argument (e.g., the last argument in the "rust-call" ABI)
                // is a tuple that was spread at the ABI level and now we have
                // to reconstruct it into a tuple local variable, from multiple
                // individual LLVM function arguments.
                let ty::Tuple(tupled_arg_tys) = arg_ty.kind() else {
                    bug!("spread argument isn't a tuple?!");
                };

                let layout = bx.layout_of(arg_ty);

                // FIXME: support unsized params in "rust-call" ABI
                if layout.is_unsized() {
                    span_bug!(
                        arg_decl.source_info.span,
                        "\"rust-call\" ABI does not support unsized params",
                    );
                }

                let place = PlaceRef::alloca(bx, layout);
                for i in 0..tupled_arg_tys.len() {
                    let arg = &fx.fn_abi.args[idx];
                    idx += 1;
                    if let PassMode::Cast { pad_i32: true, .. } = arg.mode {
                        llarg_idx += 1;
                    }
                    let pr_field = place.project_field(bx, i);
                    bx.store_fn_arg(arg, &mut llarg_idx, pr_field);
                }
                assert_eq!(
                    None,
                    num_untupled.replace(tupled_arg_tys.len()),
                    "Replaced existing num_tupled"
                );

                return LocalRef::Place(place);
            }

            if fx.fn_abi.c_variadic && arg_index == fx.fn_abi.args.len() {
                let va_list = PlaceRef::alloca(bx, bx.layout_of(arg_ty));
                bx.va_start(va_list.val.llval);

                return LocalRef::Place(va_list);
            }

            let arg = &fx.fn_abi.args[idx];
            idx += 1;
            if let PassMode::Cast { pad_i32: true, .. } = arg.mode {
                llarg_idx += 1;
            }

            if !memory_locals.contains(local) {
                // We don't have to cast or keep the argument in the alloca.
                // FIXME(eddyb): We should figure out how to use llvm.dbg.value instead
                // of putting everything in allocas just so we can use llvm.dbg.declare.
                let local = |op| LocalRef::Operand(op);
                match arg.mode {
                    PassMode::Ignore => {
                        return local(OperandRef::zero_sized(arg.layout));
                    }
                    PassMode::Direct(_) => {
                        let llarg = bx.get_param(llarg_idx);
                        llarg_idx += 1;
                        return local(OperandRef::from_immediate_or_packed_pair(
                            bx, llarg, arg.layout,
                        ));
                    }
                    PassMode::Pair(..) => {
                        let (a, b) = (bx.get_param(llarg_idx), bx.get_param(llarg_idx + 1));
                        llarg_idx += 2;

                        return local(OperandRef {
                            val: OperandValue::Pair(a, b),
                            layout: arg.layout,
                        });
                    }
                    _ => {}
                }
            }

            match arg.mode {
                // Sized indirect arguments
                PassMode::Indirect { attrs, meta_attrs: None, on_stack: _ } => {
                    // Don't copy an indirect argument to an alloca, the caller already put it
                    // in a temporary alloca and gave it up.
                    // FIXME: lifetimes
                    if let Some(pointee_align) = attrs.pointee_align
                        && pointee_align < arg.layout.align.abi
                    {
                        // ...unless the argument is underaligned, then we need to copy it to
                        // a higher-aligned alloca.
                        let tmp = PlaceRef::alloca(bx, arg.layout);
                        bx.store_fn_arg(arg, &mut llarg_idx, tmp);
                        LocalRef::Place(tmp)
                    } else {
                        let llarg = bx.get_param(llarg_idx);
                        llarg_idx += 1;
                        LocalRef::Place(PlaceRef::new_sized(llarg, arg.layout))
                    }
                }
                // Unsized indirect qrguments
                PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
                    // As the storage for the indirect argument lives during
                    // the whole function call, we just copy the wide pointer.
                    let llarg = bx.get_param(llarg_idx);
                    llarg_idx += 1;
                    let llextra = bx.get_param(llarg_idx);
                    llarg_idx += 1;
                    let indirect_operand = OperandValue::Pair(llarg, llextra);

                    let tmp = PlaceRef::alloca_unsized_indirect(bx, arg.layout);
                    indirect_operand.store(bx, tmp);
                    LocalRef::UnsizedPlace(tmp)
                }
                _ => {
                    let tmp = PlaceRef::alloca(bx, arg.layout);
                    bx.store_fn_arg(arg, &mut llarg_idx, tmp);
                    LocalRef::Place(tmp)
                }
            }
        })
        .collect::<Vec<_>>();

    if fx.instance.def.requires_caller_location(bx.tcx()) {
        let mir_args = if let Some(num_untupled) = num_untupled {
            // Subtract off the tupled argument that gets 'expanded'
            args.len() - 1 + num_untupled
        } else {
            args.len()
        };
        assert_eq!(
            fx.fn_abi.args.len(),
            mir_args + 1,
            "#[track_caller] instance {:?} must have 1 more argument in their ABI than in their MIR",
            fx.instance
        );

        let arg = fx.fn_abi.args.last().unwrap();
        match arg.mode {
            PassMode::Direct(_) => (),
            _ => bug!("caller location must be PassMode::Direct, found {:?}", arg.mode),
        }

        fx.caller_location = Some(OperandRef {
            val: OperandValue::Immediate(bx.get_param(llarg_idx)),
            layout: arg.layout,
        });
    }

    args
}

fn find_cold_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir: &mir::Body<'tcx>,
) -> IndexVec<mir::BasicBlock, bool> {
    let local_decls = &mir.local_decls;

    let mut cold_blocks: IndexVec<mir::BasicBlock, bool> =
        IndexVec::from_elem(false, &mir.basic_blocks);

    // Traverse all basic blocks from end of the function to the start.
    for (bb, bb_data) in traversal::postorder(mir) {
        let terminator = bb_data.terminator();

        match terminator.kind {
            // If a BB ends with a call to a cold function, mark it as cold.
            mir::TerminatorKind::Call { ref func, .. }
            | mir::TerminatorKind::TailCall { ref func, .. }
                if let ty::FnDef(def_id, ..) = *func.ty(local_decls, tcx).kind()
                    && let attrs = tcx.codegen_fn_attrs(def_id)
                    && attrs.flags.contains(CodegenFnAttrFlags::COLD) =>
            {
                cold_blocks[bb] = true;
                continue;
            }

            // If a BB ends with an `unreachable`, also mark it as cold.
            mir::TerminatorKind::Unreachable => {
                cold_blocks[bb] = true;
                continue;
            }

            _ => {}
        }

        // If all successors of a BB are cold and there's at least one of them, mark this BB as cold
        let mut succ = terminator.successors();
        if let Some(first) = succ.next()
            && cold_blocks[first]
            && succ.all(|s| cold_blocks[s])
        {
            cold_blocks[bb] = true;
        }
    }

    cold_blocks
}
