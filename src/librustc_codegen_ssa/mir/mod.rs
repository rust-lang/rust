use rustc::ty::{self, Ty, TypeFoldable, UpvarSubsts, Instance};
use rustc::ty::layout::{TyLayout, HasTyCtxt, FnTypeExt};
use rustc::mir::{self, Body};
use rustc::session::config::DebugInfo;
use rustc_target::abi::call::{FnType, PassMode};
use rustc_target::abi::{Variants, VariantIdx};
use crate::base;
use crate::traits::*;

use syntax_pos::DUMMY_SP;
use syntax::symbol::kw;

use std::iter;

use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;

use self::analyze::CleanupKind;
use self::debuginfo::{VariableAccess, VariableKind, FunctionDebugContext};
use self::place::PlaceRef;
use rustc::mir::traversal;

use self::operand::{OperandRef, OperandValue};

/// Master context for codegenning from MIR.
pub struct FunctionCx<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    instance: Instance<'tcx>,

    mir: &'a mir::Body<'tcx>,

    debug_context: FunctionDebugContext<Bx::DIScope>,

    llfn: Bx::Function,

    cx: &'a Bx::CodegenCx,

    fn_ty: FnType<'tcx, Ty<'tcx>>,

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

    /// Debug information for MIR scopes.
    scopes: IndexVec<mir::SourceScope, debuginfo::DebugScope<Bx::DIScope>>,
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn monomorphize<T>(&self, value: &T) -> T
        where T: TypeFoldable<'tcx>
    {
        self.cx.tcx().subst_and_normalize_erasing_regions(
            self.instance.substs,
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
        layout: TyLayout<'tcx>,
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
    llfn: Bx::Function,
    mir: &'a Body<'tcx>,
    instance: Instance<'tcx>,
    sig: ty::FnSig<'tcx>,
) {
    assert!(!instance.substs.needs_infer());

    let fn_ty = FnType::new(cx, sig, &[]);
    debug!("fn_ty: {:?}", fn_ty);
    let mut debug_context =
        cx.create_function_debug_context(instance, sig, llfn, mir);
    let mut bx = Bx::new_block(cx, llfn, "start");

    if mir.basic_blocks().iter().any(|bb| bb.is_cleanup) {
        bx.set_personality_fn(cx.eh_personality());
    }

    bx.sideeffect();

    let cleanup_kinds = analyze::cleanup_kinds(&mir);
    // Allocate a `Block` for every basic block, except
    // the start block, if nothing loops back to it.
    let reentrant_start_block = !mir.predecessors_for(mir::START_BLOCK).is_empty();
    let block_bxs: IndexVec<mir::BasicBlock, Bx::BasicBlock> =
        mir.basic_blocks().indices().map(|bb| {
            if bb == mir::START_BLOCK && !reentrant_start_block {
                bx.llbb()
            } else {
                bx.build_sibling_block(&format!("{:?}", bb)).llbb()
            }
        }).collect();

    // Compute debuginfo scopes from MIR scopes.
    let scopes = cx.create_mir_scopes(mir, &mut debug_context);
    let (landing_pads, funclets) = create_funclets(mir, &mut bx, &cleanup_kinds, &block_bxs);

    let mut fx = FunctionCx {
        instance,
        mir,
        llfn,
        fn_ty,
        cx,
        personality_slot: None,
        blocks: block_bxs,
        unreachable_block: None,
        cleanup_kinds,
        landing_pads,
        funclets,
        scopes,
        locals: IndexVec::new(),
        debug_context,
    };

    let memory_locals = analyze::non_ssa_locals(&fx);

    // Allocate variable and temp allocas
    fx.locals = {
        let args = arg_local_refs(&mut bx, &fx, &memory_locals);

        let mut allocate_local = |local| {
            let decl = &mir.local_decls[local];
            let layout = bx.layout_of(fx.monomorphize(&decl.ty));
            assert!(!layout.ty.has_erasable_regions());

            if let Some(name) = decl.name {
                // User variable
                let debug_scope = fx.scopes[decl.visibility_scope];
                let dbg = debug_scope.is_valid() &&
                    bx.sess().opts.debuginfo == DebugInfo::Full;

                if !memory_locals.contains(local) && !dbg {
                    debug!("alloc: {:?} ({}) -> operand", local, name);
                    return LocalRef::new_operand(&mut bx, layout);
                }

                debug!("alloc: {:?} ({}) -> place", local, name);
                if layout.is_unsized() {
                    let indirect_place =
                        PlaceRef::alloca_unsized_indirect(&mut bx, layout);
                    bx.set_var_name(indirect_place.llval, name);
                    // FIXME: add an appropriate debuginfo
                    LocalRef::UnsizedPlace(indirect_place)
                } else {
                    let place = PlaceRef::alloca(&mut bx, layout);
                    bx.set_var_name(place.llval, name);
                    if dbg {
                        let (scope, span) = fx.debug_loc(mir::SourceInfo {
                            span: decl.source_info.span,
                            scope: decl.visibility_scope,
                        });
                        bx.declare_local(&fx.debug_context, name, layout.ty, scope.unwrap(),
                            VariableAccess::DirectVariable { alloca: place.llval },
                            VariableKind::LocalVariable, span);
                    }
                    LocalRef::Place(place)
                }
            } else {
                // Temporary or return place
                if local == mir::RETURN_PLACE && fx.fn_ty.ret.is_indirect() {
                    debug!("alloc: {:?} (return place) -> place", local);
                    let llretptr = bx.get_param(0);
                    LocalRef::Place(PlaceRef::new_sized(llretptr, layout))
                } else if memory_locals.contains(local) {
                    debug!("alloc: {:?} -> place", local);
                    if layout.is_unsized() {
                        let indirect_place = PlaceRef::alloca_unsized_indirect(&mut bx, layout);
                        bx.set_var_name(indirect_place.llval, format_args!("{:?}", local));
                        LocalRef::UnsizedPlace(indirect_place)
                    } else {
                        let place = PlaceRef::alloca(&mut bx, layout);
                        bx.set_var_name(place.llval, format_args!("{:?}", local));
                        LocalRef::Place(place)
                    }
                } else {
                    // If this is an immediate local, we do not create an
                    // alloca in advance. Instead we wait until we see the
                    // definition and update the operand there.
                    debug!("alloc: {:?} -> operand", local);
                    LocalRef::new_operand(&mut bx, layout)
                }
            }
        };

        let retptr = allocate_local(mir::RETURN_PLACE);
        iter::once(retptr)
            .chain(args.into_iter())
            .chain(mir.vars_and_temps_iter().map(allocate_local))
            .collect()
    };

    // Branch to the START block, if it's not the entry block.
    if reentrant_start_block {
        bx.br(fx.blocks[mir::START_BLOCK]);
    }

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(&mut fx.debug_context);

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
    mir: &'a Body<'tcx>,
    bx: &mut Bx,
    cleanup_kinds: &IndexVec<mir::BasicBlock, CleanupKind>,
    block_bxs: &IndexVec<mir::BasicBlock, Bx::BasicBlock>,
) -> (
    IndexVec<mir::BasicBlock, Option<Bx::BasicBlock>>,
    IndexVec<mir::BasicBlock, Option<Bx::Funclet>>,
) {
    block_bxs.iter_enumerated().zip(cleanup_kinds).map(|((bb, &llbb), cleanup_kind)| {
        match *cleanup_kind {
            CleanupKind::Funclet if base::wants_msvc_seh(bx.sess()) => {}
            _ => return (None, None)
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
                let null = bx.const_null(bx.type_i8p());
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
    }).unzip()
}

/// Produces, for each argument, a `Value` pointing at the
/// argument's value. As arguments are places, these are always
/// indirect.
fn arg_local_refs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    fx: &FunctionCx<'a, 'tcx, Bx>,
    memory_locals: &BitSet<mir::Local>,
) -> Vec<LocalRef<'tcx, Bx::Value>> {
    let mir = fx.mir;
    let tcx = fx.cx.tcx();
    let mut idx = 0;
    let mut llarg_idx = fx.fn_ty.ret.is_indirect() as usize;

    // Get the argument scope, if it exists and if we need it.
    let arg_scope = fx.scopes[mir::OUTERMOST_SOURCE_SCOPE];
    let arg_scope = if bx.sess().opts.debuginfo == DebugInfo::Full {
        arg_scope.scope_metadata
    } else {
        None
    };

    mir.args_iter().enumerate().map(|(arg_index, local)| {
        let arg_decl = &mir.local_decls[local];

        // FIXME(eddyb) don't allocate a `String` unless it gets used.
        let name = if let Some(name) = arg_decl.name {
            name.as_str().to_string()
        } else {
            format!("{:?}", local)
        };

        if Some(local) == mir.spread_arg {
            // This argument (e.g., the last argument in the "rust-call" ABI)
            // is a tuple that was spread at the ABI level and now we have
            // to reconstruct it into a tuple local variable, from multiple
            // individual LLVM function arguments.

            let arg_ty = fx.monomorphize(&arg_decl.ty);
            let tupled_arg_tys = match arg_ty.kind {
                ty::Tuple(ref tys) => tys,
                _ => bug!("spread argument isn't a tuple?!")
            };

            let place = PlaceRef::alloca(bx, bx.layout_of(arg_ty));
            bx.set_var_name(place.llval, name);
            for i in 0..tupled_arg_tys.len() {
                let arg = &fx.fn_ty.args[idx];
                idx += 1;
                if arg.pad.is_some() {
                    llarg_idx += 1;
                }
                let pr_field = place.project_field(bx, i);
                bx.store_fn_arg(arg, &mut llarg_idx, pr_field);
            }

            // Now that we have one alloca that contains the aggregate value,
            // we can create one debuginfo entry for the argument.
            arg_scope.map(|scope| {
                let variable_access = VariableAccess::DirectVariable {
                    alloca: place.llval
                };
                bx.declare_local(
                    &fx.debug_context,
                    arg_decl.name.unwrap_or(kw::Invalid),
                    arg_ty, scope,
                    variable_access,
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
            });

            return LocalRef::Place(place);
        }

        if fx.fn_ty.c_variadic && arg_index == fx.fn_ty.args.len() {
            let arg_ty = fx.monomorphize(&arg_decl.ty);

            let va_list = PlaceRef::alloca(bx, bx.layout_of(arg_ty));
            bx.set_var_name(va_list.llval, name);
            bx.va_start(va_list.llval);

            arg_scope.map(|scope| {
                let variable_access = VariableAccess::DirectVariable {
                    alloca: va_list.llval
                };
                bx.declare_local(
                    &fx.debug_context,
                    arg_decl.name.unwrap_or(kw::Invalid),
                    va_list.layout.ty,
                    scope,
                    variable_access,
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
            });

            return LocalRef::Place(va_list);
        }

        let arg = &fx.fn_ty.args[idx];
        idx += 1;
        if arg.pad.is_some() {
            llarg_idx += 1;
        }

        if arg_scope.is_none() && !memory_locals.contains(local) {
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
                    bx.set_var_name(llarg, &name);
                    llarg_idx += 1;
                    return local(
                        OperandRef::from_immediate_or_packed_pair(bx, llarg, arg.layout));
                }
                PassMode::Pair(..) => {
                    let (a, b) = (bx.get_param(llarg_idx), bx.get_param(llarg_idx + 1));
                    llarg_idx += 2;

                    // FIXME(eddyb) these are scalar components,
                    // maybe extract the high-level fields?
                    bx.set_var_name(a, format_args!("{}.0", name));
                    bx.set_var_name(b, format_args!("{}.1", name));

                    return local(OperandRef {
                        val: OperandValue::Pair(a, b),
                        layout: arg.layout
                    });
                }
                _ => {}
            }
        }

        let place = if arg.is_sized_indirect() {
            // Don't copy an indirect argument to an alloca, the caller
            // already put it in a temporary alloca and gave it up.
            // FIXME: lifetimes
            let llarg = bx.get_param(llarg_idx);
            bx.set_var_name(llarg, &name);
            llarg_idx += 1;
            PlaceRef::new_sized(llarg, arg.layout)
        } else if arg.is_unsized_indirect() {
            // As the storage for the indirect argument lives during
            // the whole function call, we just copy the fat pointer.
            let llarg = bx.get_param(llarg_idx);
            llarg_idx += 1;
            let llextra = bx.get_param(llarg_idx);
            llarg_idx += 1;
            let indirect_operand = OperandValue::Pair(llarg, llextra);

            let tmp = PlaceRef::alloca_unsized_indirect(bx, arg.layout);
            bx.set_var_name(tmp.llval, name);
            indirect_operand.store(bx, tmp);
            tmp
        } else {
            let tmp = PlaceRef::alloca(bx, arg.layout);
            bx.set_var_name(tmp.llval, name);
            bx.store_fn_arg(arg, &mut llarg_idx, tmp);
            tmp
        };
        let upvar_debuginfo = &mir.__upvar_debuginfo_codegen_only_do_not_use;
        arg_scope.map(|scope| {
            // Is this a regular argument?
            if arg_index > 0 || upvar_debuginfo.is_empty() {
                // The Rust ABI passes indirect variables using a pointer and a manual copy, so we
                // need to insert a deref here, but the C ABI uses a pointer and a copy using the
                // byval attribute, for which LLVM always does the deref itself,
                // so we must not add it.
                let variable_access = VariableAccess::DirectVariable {
                    alloca: place.llval
                };

                bx.declare_local(
                    &fx.debug_context,
                    arg_decl.name.unwrap_or(kw::Invalid),
                    arg.layout.ty,
                    scope,
                    variable_access,
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
                return;
            }

            let pin_did = tcx.lang_items().pin_type();
            // Or is it the closure environment?
            let (closure_layout, env_ref) = match arg.layout.ty.kind {
                ty::RawPtr(ty::TypeAndMut { ty, .. }) |
                ty::Ref(_, ty, _)  => (bx.layout_of(ty), true),
                ty::Adt(def, substs) if Some(def.did) == pin_did => {
                    match substs.type_at(0).kind {
                        ty::Ref(_, ty, _)  => (bx.layout_of(ty), true),
                        _ => (arg.layout, false),
                    }
                }
                _ => (arg.layout, false)
            };

            let (def_id, upvar_substs) = match closure_layout.ty.kind {
                ty::Closure(def_id, substs) => (def_id,
                    UpvarSubsts::Closure(substs)),
                ty::Generator(def_id, substs, _) => (def_id, UpvarSubsts::Generator(substs)),
                _ => bug!("upvar debuginfo with non-closure arg0 type `{}`", closure_layout.ty)
            };
            let upvar_tys = upvar_substs.upvar_tys(def_id, tcx);

            let extra_locals = {
                let upvars = upvar_debuginfo
                    .iter()
                    .zip(upvar_tys)
                    .enumerate()
                    .map(|(i, (upvar, ty))| {
                        (None, i, upvar.debug_name, upvar.by_ref, ty, scope, DUMMY_SP)
                    });

                let generator_fields = mir.generator_layout.as_ref().map(|generator_layout| {
                    let (def_id, gen_substs) = match closure_layout.ty.kind {
                        ty::Generator(def_id, substs, _) => (def_id, substs),
                        _ => bug!("generator layout without generator substs"),
                    };
                    let state_tys = gen_substs.as_generator().state_tys(def_id, tcx);

                    generator_layout.variant_fields.iter()
                        .zip(state_tys)
                        .enumerate()
                        .flat_map(move |(variant_idx, (fields, tys))| {
                            let variant_idx = Some(VariantIdx::from(variant_idx));
                            fields.iter()
                                .zip(tys)
                                .enumerate()
                                .filter_map(move |(i, (field, ty))| {
                                    let decl = &generator_layout.
                                        __local_debuginfo_codegen_only_do_not_use[*field];
                                    if let Some(name) = decl.name {
                                        let ty = fx.monomorphize(&ty);
                                        let (var_scope, var_span) = fx.debug_loc(mir::SourceInfo {
                                            span: decl.source_info.span,
                                            scope: decl.visibility_scope,
                                        });
                                        let var_scope = var_scope.unwrap_or(scope);
                                        Some((variant_idx, i, name, false, ty, var_scope, var_span))
                                    } else {
                                        None
                                    }
                            })
                        })
                }).into_iter().flatten();

                upvars.chain(generator_fields)
            };

            for (variant_idx, field, name, by_ref, ty, var_scope, var_span) in extra_locals {
                let fields = match variant_idx {
                    Some(variant_idx) => {
                        match &closure_layout.variants {
                            Variants::Multiple { variants, .. } => {
                                &variants[variant_idx].fields
                            },
                            _ => bug!("variant index on univariant layout"),
                        }
                    }
                    None => &closure_layout.fields,
                };
                let byte_offset_of_var_in_env = fields.offset(field).bytes();

                let ops = bx.debuginfo_upvar_ops_sequence(byte_offset_of_var_in_env);

                // The environment and the capture can each be indirect.
                let mut ops = if env_ref { &ops[..] } else { &ops[1..] };

                let ty = if let (true, &ty::Ref(_, ty, _)) = (by_ref, &ty.kind) {
                    ty
                } else {
                    ops = &ops[..ops.len() - 1];
                    ty
                };

                let variable_access = VariableAccess::IndirectVariable {
                    alloca: place.llval,
                    address_operations: &ops
                };
                bx.declare_local(
                    &fx.debug_context,
                    name,
                    ty,
                    var_scope,
                    variable_access,
                    VariableKind::LocalVariable,
                    var_span
                );
            }
        });
        if arg.is_unsized_indirect() {
            LocalRef::UnsizedPlace(place)
        } else {
            LocalRef::Place(place)
        }
    }).collect()
}

mod analyze;
mod block;
pub mod constant;
pub mod debuginfo;
pub mod place;
pub mod operand;
mod rvalue;
mod statement;
