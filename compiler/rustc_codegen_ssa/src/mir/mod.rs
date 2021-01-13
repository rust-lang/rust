use crate::base;
use crate::traits::*;
use rustc_errors::ErrorReported;
use rustc_middle::mir;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::layout::{FnAbiExt, HasTyCtxt, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TypeFoldable};
use rustc_target::abi::call::{FnAbi, PassMode};
use rustc_target::abi::HasDataLayout;

use std::iter;

use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;

use self::analyze::CleanupKind;
use self::debuginfo::{FunctionDebugContext, PerLocalVarDebugInfo};
use self::place::PlaceRef;
use rustc_middle::mir::traversal;

use self::operand::{OperandRef, OperandValue};

/// Master context for codegenning from MIR.
pub struct FunctionCx<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    instance: Instance<'tcx>,

    mir: &'tcx mir::Body<'tcx>,

    debug_context: Option<FunctionDebugContext<Bx::DIScope, Bx::DILocation>>,

    llfn: Bx::Function,

    cx: &'a Bx::CodegenCx,

    fn_abi: FnAbi<'tcx, Ty<'tcx>>,

    /// When unwinding is initiated, we have to store this personality
    /// value somewhere so that we can load it and re-use it in the
    /// resume instruction. The personality is (afaik) some kind of
    /// value used for C++ unwinding, which must filter by type: we
    /// don't really care about it very much. Anyway, this value
    /// contains an alloca into which the personality is stored and
    /// then later loaded when generating the DIVERGE_BLOCK.
    personality_slot: Option<PlaceRef<'tcx, Bx::Value>>,

    /// A `Block` for each MIR `BasicBlock`
    blocks: IndexVec<mir::BasicBlock, Bx::BasicBlock>,

    /// The funclet status of each basic block
    cleanup_kinds: IndexVec<mir::BasicBlock, analyze::CleanupKind>,

    /// When targeting MSVC, this stores the cleanup info for each funclet
    /// BB. This is initialized as we compute the funclets' head block in RPO.
    funclets: IndexVec<mir::BasicBlock, Option<Bx::Funclet>>,

    /// This stores the landing-pad block for a given BB, computed lazily on GNU
    /// and eagerly on MSVC.
    landing_pads: IndexVec<mir::BasicBlock, Option<Bx::BasicBlock>>,

    /// Cached unreachable block
    unreachable_block: Option<Bx::BasicBlock>,

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
    locals: IndexVec<mir::Local, LocalRef<'tcx, Bx::Value>>,

    /// All `VarDebugInfo` from the MIR body, partitioned by `Local`.
    /// This is `None` if no var`#[non_exhaustive]`iable debuginfo/names are needed.
    per_local_var_debug_info:
        Option<IndexVec<mir::Local, Vec<PerLocalVarDebugInfo<'tcx, Bx::DIVariable>>>>,

    /// Caller location propagated if this function has `#[track_caller]`.
    caller_location: Option<OperandRef<'tcx, Bx::Value>>,
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn monomorphize<T>(&self, value: T) -> T
    where
        T: Copy + TypeFoldable<'tcx>,
    {
        debug!("monomorphize: self.instance={:?}", self.instance);
        self.instance.subst_mir_and_normalize_erasing_regions(
            self.cx.tcx(),
            ty::ParamEnv::reveal_all(),
            value,
        )
    }
}

enum LocalRef<'tcx, V> {
    Place(PlaceRef<'tcx, V>),
    /// `UnsizedPlace(p)`: `p` itself is a thin pointer (indirect place).
    /// `*p` is the fat pointer that references the actual unsized place.
    /// Every time it is initialized, we have to reallocate the place
    /// and update the fat pointer. That's the reason why it is indirect.
    UnsizedPlace(PlaceRef<'tcx, V>),
    Operand(Option<OperandRef<'tcx, V>>),
}

impl<'a, 'tcx, V: CodegenObject> LocalRef<'tcx, V> {
    fn new_operand<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> LocalRef<'tcx, V> {
        if layout.is_zst() {
            // Zero-size temporaries aren't always initialized, which
            // doesn't matter because they don't contain data, but
            // we need something in the operand.
            LocalRef::Operand(Some(OperandRef::new_zst(bx, layout)))
        } else {
            LocalRef::Operand(None)
        }
    }
}

///////////////////////////////////////////////////////////////////////////

pub fn codegen_mir<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    cx: &'a Bx::CodegenCx,
    instance: Instance<'tcx>,
) {
    assert!(!instance.substs.needs_infer());

    let llfn = cx.get_fn(instance);

    let mir = cx.tcx().instance_mir(instance.def);

    let fn_abi = FnAbi::of_instance(cx, instance, &[]);
    debug!("fn_abi: {:?}", fn_abi);

    let debug_context = cx.create_function_debug_context(instance, &fn_abi, llfn, &mir);

    let mut bx = Bx::new_block(cx, llfn, "start");

    if mir.basic_blocks().iter().any(|bb| bb.is_cleanup) {
        bx.set_personality_fn(cx.eh_personality());
    }

    bx.sideeffect(false);

    let cleanup_kinds = analyze::cleanup_kinds(&mir);
    // Allocate a `Block` for every basic block, except
    // the start block, if nothing loops back to it.
    let reentrant_start_block = !mir.predecessors()[mir::START_BLOCK].is_empty();
    let block_bxs: IndexVec<mir::BasicBlock, Bx::BasicBlock> = mir
        .basic_blocks()
        .indices()
        .map(|bb| {
            if bb == mir::START_BLOCK && !reentrant_start_block {
                bx.llbb()
            } else {
                bx.build_sibling_block(&format!("{:?}", bb)).llbb()
            }
        })
        .collect();

    let (landing_pads, funclets) = create_funclets(&mir, &mut bx, &cleanup_kinds, &block_bxs);
    let mut fx = FunctionCx {
        instance,
        mir,
        llfn,
        fn_abi,
        cx,
        personality_slot: None,
        blocks: block_bxs,
        unreachable_block: None,
        cleanup_kinds,
        landing_pads,
        funclets,
        locals: IndexVec::new(),
        debug_context,
        per_local_var_debug_info: None,
        caller_location: None,
    };

    fx.per_local_var_debug_info = fx.compute_per_local_var_debug_info(&mut bx);

    for const_ in &mir.required_consts {
        if let Err(err) = fx.eval_mir_constant(const_) {
            match err {
                // errored or at least linted
                ErrorHandled::Reported(ErrorReported) | ErrorHandled::Linted => {}
                ErrorHandled::TooGeneric => {
                    span_bug!(const_.span, "codgen encountered polymorphic constant: {:?}", err)
                }
            }
        }
    }

    let memory_locals = analyze::non_ssa_locals(&fx);

    // Allocate variable and temp allocas
    fx.locals = {
        let args = arg_local_refs(&mut bx, &mut fx, &memory_locals);

        let mut allocate_local = |local| {
            let decl = &mir.local_decls[local];
            let layout = bx.layout_of(fx.monomorphize(decl.ty));
            assert!(!layout.ty.has_erasable_regions());

            if local == mir::RETURN_PLACE && fx.fn_abi.ret.is_indirect() {
                debug!("alloc: {:?} (return place) -> place", local);
                let llretptr = bx.get_param(0);
                return LocalRef::Place(PlaceRef::new_sized(llretptr, layout));
            }

            if memory_locals.contains(local) {
                debug!("alloc: {:?} -> place", local);
                if layout.is_unsized() {
                    LocalRef::UnsizedPlace(PlaceRef::alloca_unsized_indirect(&mut bx, layout))
                } else {
                    LocalRef::Place(PlaceRef::alloca(&mut bx, layout))
                }
            } else {
                debug!("alloc: {:?} -> operand", local);
                LocalRef::new_operand(&mut bx, layout)
            }
        };

        let retptr = allocate_local(mir::RETURN_PLACE);
        iter::once(retptr)
            .chain(args.into_iter())
            .chain(mir.vars_and_temps_iter().map(allocate_local))
            .collect()
    };

    // Apply debuginfo to the newly allocated locals.
    fx.debug_introduce_locals(&mut bx);

    // Branch to the START block, if it's not the entry block.
    if reentrant_start_block {
        bx.br(fx.blocks[mir::START_BLOCK]);
    }

    let rpo = traversal::reverse_postorder(&mir);
    let mut visited = BitSet::new_empty(mir.basic_blocks().len());

    // Codegen the body of each block using reverse postorder
    for (bb, _) in rpo {
        visited.insert(bb.index());
        fx.codegen_block(bb);
    }

    // Remove blocks that haven't been visited, or have no
    // predecessors.
    for bb in mir.basic_blocks().indices() {
        // Unreachable block
        if !visited.contains(bb.index()) {
            debug!("codegen_mir: block {:?} was not visited", bb);
            unsafe {
                bx.delete_basic_block(fx.blocks[bb]);
            }
        }
    }
}

fn create_funclets<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    mir: &'tcx mir::Body<'tcx>,
    bx: &mut Bx,
    cleanup_kinds: &IndexVec<mir::BasicBlock, CleanupKind>,
    block_bxs: &IndexVec<mir::BasicBlock, Bx::BasicBlock>,
) -> (
    IndexVec<mir::BasicBlock, Option<Bx::BasicBlock>>,
    IndexVec<mir::BasicBlock, Option<Bx::Funclet>>,
) {
    block_bxs
        .iter_enumerated()
        .zip(cleanup_kinds)
        .map(|((bb, &llbb), cleanup_kind)| {
            match *cleanup_kind {
                CleanupKind::Funclet if base::wants_msvc_seh(bx.sess()) => {}
                _ => return (None, None),
            }

            let funclet;
            let ret_llbb;
            match mir[bb].terminator.as_ref().map(|t| &t.kind) {
                // This is a basic block that we're aborting the program for,
                // notably in an `extern` function. These basic blocks are inserted
                // so that we assert that `extern` functions do indeed not panic,
                // and if they do we abort the process.
                //
                // On MSVC these are tricky though (where we're doing funclets). If
                // we were to do a cleanuppad (like below) the normal functions like
                // `longjmp` would trigger the abort logic, terminating the
                // program. Instead we insert the equivalent of `catch(...)` for C++
                // which magically doesn't trigger when `longjmp` files over this
                // frame.
                //
                // Lots more discussion can be found on #48251 but this codegen is
                // modeled after clang's for:
                //
                //      try {
                //          foo();
                //      } catch (...) {
                //          bar();
                //      }
                Some(&mir::TerminatorKind::Abort) => {
                    let mut cs_bx = bx.build_sibling_block(&format!("cs_funclet{:?}", bb));
                    let mut cp_bx = bx.build_sibling_block(&format!("cp_funclet{:?}", bb));
                    ret_llbb = cs_bx.llbb();

                    let cs = cs_bx.catch_switch(None, None, 1);
                    cs_bx.add_handler(cs, cp_bx.llbb());

                    // The "null" here is actually a RTTI type descriptor for the
                    // C++ personality function, but `catch (...)` has no type so
                    // it's null. The 64 here is actually a bitfield which
                    // represents that this is a catch-all block.
                    let null = bx.const_null(
                        bx.type_i8p_ext(bx.cx().data_layout().instruction_address_space),
                    );
                    let sixty_four = bx.const_i32(64);
                    funclet = cp_bx.catch_pad(cs, &[null, sixty_four, null]);
                    cp_bx.br(llbb);
                }
                _ => {
                    let mut cleanup_bx = bx.build_sibling_block(&format!("funclet_{:?}", bb));
                    ret_llbb = cleanup_bx.llbb();
                    funclet = cleanup_bx.cleanup_pad(None, &[]);
                    cleanup_bx.br(llbb);
                }
            };

            (Some(ret_llbb), Some(funclet))
        })
        .unzip()
}

/// Produces, for each argument, a `Value` pointing at the
/// argument's value. As arguments are places, these are always
/// indirect.
fn arg_local_refs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    fx: &mut FunctionCx<'a, 'tcx, Bx>,
    memory_locals: &BitSet<mir::Local>,
) -> Vec<LocalRef<'tcx, Bx::Value>> {
    let mir = fx.mir;
    let mut idx = 0;
    let mut llarg_idx = fx.fn_abi.ret.is_indirect() as usize;

    let args = mir
        .args_iter()
        .enumerate()
        .map(|(arg_index, local)| {
            let arg_decl = &mir.local_decls[local];

            if Some(local) == mir.spread_arg {
                // This argument (e.g., the last argument in the "rust-call" ABI)
                // is a tuple that was spread at the ABI level and now we have
                // to reconstruct it into a tuple local variable, from multiple
                // individual LLVM function arguments.

                let arg_ty = fx.monomorphize(arg_decl.ty);
                let tupled_arg_tys = match arg_ty.kind() {
                    ty::Tuple(tys) => tys,
                    _ => bug!("spread argument isn't a tuple?!"),
                };

                let place = PlaceRef::alloca(bx, bx.layout_of(arg_ty));
                for i in 0..tupled_arg_tys.len() {
                    let arg = &fx.fn_abi.args[idx];
                    idx += 1;
                    if arg.pad.is_some() {
                        llarg_idx += 1;
                    }
                    let pr_field = place.project_field(bx, i);
                    bx.store_fn_arg(arg, &mut llarg_idx, pr_field);
                }

                return LocalRef::Place(place);
            }

            if fx.fn_abi.c_variadic && arg_index == fx.fn_abi.args.len() {
                let arg_ty = fx.monomorphize(arg_decl.ty);

                let va_list = PlaceRef::alloca(bx, bx.layout_of(arg_ty));
                bx.va_start(va_list.llval);

                return LocalRef::Place(va_list);
            }

            let arg = &fx.fn_abi.args[idx];
            idx += 1;
            if arg.pad.is_some() {
                llarg_idx += 1;
            }

            if !memory_locals.contains(local) {
                // We don't have to cast or keep the argument in the alloca.
                // FIXME(eddyb): We should figure out how to use llvm.dbg.value instead
                // of putting everything in allocas just so we can use llvm.dbg.declare.
                let local = |op| LocalRef::Operand(Some(op));
                match arg.mode {
                    PassMode::Ignore => {
                        return local(OperandRef::new_zst(bx, arg.layout));
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

            if arg.is_sized_indirect() {
                // Don't copy an indirect argument to an alloca, the caller
                // already put it in a temporary alloca and gave it up.
                // FIXME: lifetimes
                let llarg = bx.get_param(llarg_idx);
                llarg_idx += 1;
                LocalRef::Place(PlaceRef::new_sized(llarg, arg.layout))
            } else if arg.is_unsized_indirect() {
                // As the storage for the indirect argument lives during
                // the whole function call, we just copy the fat pointer.
                let llarg = bx.get_param(llarg_idx);
                llarg_idx += 1;
                let llextra = bx.get_param(llarg_idx);
                llarg_idx += 1;
                let indirect_operand = OperandValue::Pair(llarg, llextra);

                let tmp = PlaceRef::alloca_unsized_indirect(bx, arg.layout);
                indirect_operand.store(bx, tmp);
                LocalRef::UnsizedPlace(tmp)
            } else {
                let tmp = PlaceRef::alloca(bx, arg.layout);
                bx.store_fn_arg(arg, &mut llarg_idx, tmp);
                LocalRef::Place(tmp)
            }
        })
        .collect::<Vec<_>>();

    if fx.instance.def.requires_caller_location(bx.tcx()) {
        assert_eq!(
            fx.fn_abi.args.len(),
            args.len() + 1,
            "#[track_caller] fn's must have 1 more argument in their ABI than in their MIR",
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

mod analyze;
mod block;
pub mod constant;
pub mod coverageinfo;
pub mod debuginfo;
mod intrinsic;
pub mod operand;
pub mod place;
mod rvalue;
mod statement;
