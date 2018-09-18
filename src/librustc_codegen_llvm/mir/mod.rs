// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::{C_i32, C_null};
use libc::c_uint;
use llvm::{self, BasicBlock};
use llvm::debuginfo::DIScope;
use rustc::ty::{self, Ty, TypeFoldable, UpvarSubsts};
use rustc::ty::layout::{LayoutOf, TyLayout};
use rustc::mir::{self, Mir};
use rustc::ty::subst::Substs;
use rustc::session::config::DebugInfo;
use base;
use builder::Builder;
use common::{CodegenCx, Funclet};
use debuginfo::{self, declare_local, VariableAccess, VariableKind, FunctionDebugContext};
use monomorphize::Instance;
use abi::{ArgTypeExt, FnType, FnTypeExt, PassMode};
use type_::Type;
use value::Value;

use syntax_pos::{DUMMY_SP, NO_EXPANSION, BytePos, Span};
use syntax::symbol::keywords;

use std::iter;

use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::IndexVec;

pub use self::constant::codegen_static_initializer;

use self::analyze::CleanupKind;
use self::place::PlaceRef;
use rustc::mir::traversal;

use self::operand::{OperandRef, OperandValue};

/// Master context for codegenning from MIR.
pub struct FunctionCx<'a, 'll: 'a, 'tcx: 'll> {
    instance: Instance<'tcx>,

    mir: &'a mir::Mir<'tcx>,

    debug_context: FunctionDebugContext<'ll>,

    llfn: &'ll Value,

    cx: &'a CodegenCx<'ll, 'tcx>,

    fn_ty: FnType<'tcx, Ty<'tcx>>,

    /// When unwinding is initiated, we have to store this personality
    /// value somewhere so that we can load it and re-use it in the
    /// resume instruction. The personality is (afaik) some kind of
    /// value used for C++ unwinding, which must filter by type: we
    /// don't really care about it very much. Anyway, this value
    /// contains an alloca into which the personality is stored and
    /// then later loaded when generating the DIVERGE_BLOCK.
    personality_slot: Option<PlaceRef<'ll, 'tcx>>,

    /// A `Block` for each MIR `BasicBlock`
    blocks: IndexVec<mir::BasicBlock, &'ll BasicBlock>,

    /// The funclet status of each basic block
    cleanup_kinds: IndexVec<mir::BasicBlock, analyze::CleanupKind>,

    /// When targeting MSVC, this stores the cleanup info for each funclet
    /// BB. This is initialized as we compute the funclets' head block in RPO.
    funclets: &'a IndexVec<mir::BasicBlock, Option<Funclet<'ll>>>,

    /// This stores the landing-pad block for a given BB, computed lazily on GNU
    /// and eagerly on MSVC.
    landing_pads: IndexVec<mir::BasicBlock, Option<&'ll BasicBlock>>,

    /// Cached unreachable block
    unreachable_block: Option<&'ll BasicBlock>,

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
    locals: IndexVec<mir::Local, LocalRef<'ll, 'tcx>>,

    /// Debug information for MIR scopes.
    scopes: IndexVec<mir::SourceScope, debuginfo::MirDebugScope<'ll>>,

    /// If this function is being monomorphized, this contains the type substitutions used.
    param_substs: &'tcx Substs<'tcx>,
}

impl FunctionCx<'a, 'll, 'tcx> {
    pub fn monomorphize<T>(&self, value: &T) -> T
        where T: TypeFoldable<'tcx>
    {
        self.cx.tcx.subst_and_normalize_erasing_regions(
            self.param_substs,
            ty::ParamEnv::reveal_all(),
            value,
        )
    }

    pub fn set_debug_loc(&mut self, bx: &Builder<'_, 'll, '_>, source_info: mir::SourceInfo) {
        let (scope, span) = self.debug_loc(source_info);
        debuginfo::set_source_location(&self.debug_context, bx, scope, span);
    }

    pub fn debug_loc(&mut self, source_info: mir::SourceInfo) -> (Option<&'ll DIScope>, Span) {
        // Bail out if debug info emission is not enabled.
        match self.debug_context {
            FunctionDebugContext::DebugInfoDisabled |
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                return (self.scopes[source_info.scope].scope_metadata, source_info.span);
            }
            FunctionDebugContext::RegularContext(_) =>{}
        }

        // In order to have a good line stepping behavior in debugger, we overwrite debug
        // locations of macro expansions with that of the outermost expansion site
        // (unless the crate is being compiled with `-Z debug-macros`).
        if source_info.span.ctxt() == NO_EXPANSION ||
           self.cx.sess().opts.debugging_opts.debug_macros {
            let scope = self.scope_metadata_for_loc(source_info.scope, source_info.span.lo());
            (scope, source_info.span)
        } else {
            // Walk up the macro expansion chain until we reach a non-expanded span.
            // We also stop at the function body level because no line stepping can occur
            // at the level above that.
            let mut span = source_info.span;
            while span.ctxt() != NO_EXPANSION && span.ctxt() != self.mir.span.ctxt() {
                if let Some(info) = span.ctxt().outer().expn_info() {
                    span = info.call_site;
                } else {
                    break;
                }
            }
            let scope = self.scope_metadata_for_loc(source_info.scope, span.lo());
            // Use span of the outermost expansion site, while keeping the original lexical scope.
            (scope, span)
        }
    }

    // DILocations inherit source file name from the parent DIScope.  Due to macro expansions
    // it may so happen that the current span belongs to a different file than the DIScope
    // corresponding to span's containing source scope.  If so, we need to create a DIScope
    // "extension" into that file.
    fn scope_metadata_for_loc(&self, scope_id: mir::SourceScope, pos: BytePos)
                               -> Option<&'ll DIScope> {
        let scope_metadata = self.scopes[scope_id].scope_metadata;
        if pos < self.scopes[scope_id].file_start_pos ||
           pos >= self.scopes[scope_id].file_end_pos {
            let cm = self.cx.sess().source_map();
            let defining_crate = self.debug_context.get_ref(DUMMY_SP).defining_crate;
            Some(debuginfo::extend_scope_to_file(self.cx,
                                            scope_metadata.unwrap(),
                                            &cm.lookup_char_pos(pos).file,
                                            defining_crate))
        } else {
            scope_metadata
        }
    }
}

enum LocalRef<'ll, 'tcx> {
    Place(PlaceRef<'ll, 'tcx>),
    /// `UnsizedPlace(p)`: `p` itself is a thin pointer (indirect place).
    /// `*p` is the fat pointer that references the actual unsized place.
    /// Every time it is initialized, we have to reallocate the place
    /// and update the fat pointer. That's the reason why it is indirect.
    UnsizedPlace(PlaceRef<'ll, 'tcx>),
    Operand(Option<OperandRef<'ll, 'tcx>>),
}

impl LocalRef<'ll, 'tcx> {
    fn new_operand(cx: &CodegenCx<'ll, 'tcx>, layout: TyLayout<'tcx>) -> LocalRef<'ll, 'tcx> {
        if layout.is_zst() {
            // Zero-size temporaries aren't always initialized, which
            // doesn't matter because they don't contain data, but
            // we need something in the operand.
            LocalRef::Operand(Some(OperandRef::new_zst(cx, layout)))
        } else {
            LocalRef::Operand(None)
        }
    }
}

///////////////////////////////////////////////////////////////////////////

pub fn codegen_mir(
    cx: &'a CodegenCx<'ll, 'tcx>,
    llfn: &'ll Value,
    mir: &'a Mir<'tcx>,
    instance: Instance<'tcx>,
    sig: ty::FnSig<'tcx>,
) {
    let fn_ty = FnType::new(cx, sig, &[]);
    debug!("fn_ty: {:?}", fn_ty);
    let debug_context =
        debuginfo::create_function_debug_context(cx, instance, sig, llfn, mir);
    let bx = Builder::new_block(cx, llfn, "start");

    if mir.basic_blocks().iter().any(|bb| bb.is_cleanup) {
        bx.set_personality_fn(cx.eh_personality());
    }

    let cleanup_kinds = analyze::cleanup_kinds(&mir);
    // Allocate a `Block` for every basic block, except
    // the start block, if nothing loops back to it.
    let reentrant_start_block = !mir.predecessors_for(mir::START_BLOCK).is_empty();
    let block_bxs: IndexVec<mir::BasicBlock, &'ll BasicBlock> =
        mir.basic_blocks().indices().map(|bb| {
            if bb == mir::START_BLOCK && !reentrant_start_block {
                bx.llbb()
            } else {
                bx.build_sibling_block(&format!("{:?}", bb)).llbb()
            }
        }).collect();

    // Compute debuginfo scopes from MIR scopes.
    let scopes = debuginfo::create_mir_scopes(cx, mir, &debug_context);
    let (landing_pads, funclets) = create_funclets(mir, &bx, &cleanup_kinds, &block_bxs);

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
        funclets: &funclets,
        scopes,
        locals: IndexVec::new(),
        debug_context,
        param_substs: {
            assert!(!instance.substs.needs_infer());
            instance.substs
        },
    };

    let memory_locals = analyze::non_ssa_locals(&fx);

    // Allocate variable and temp allocas
    fx.locals = {
        let args = arg_local_refs(&bx, &fx, &fx.scopes, &memory_locals);

        let mut allocate_local = |local| {
            let decl = &mir.local_decls[local];
            let layout = bx.cx.layout_of(fx.monomorphize(&decl.ty));
            assert!(!layout.ty.has_erasable_regions());

            if let Some(name) = decl.name {
                // User variable
                let debug_scope = fx.scopes[decl.visibility_scope];
                let dbg = debug_scope.is_valid() && bx.sess().opts.debuginfo == DebugInfo::Full;

                if !memory_locals.contains(local) && !dbg {
                    debug!("alloc: {:?} ({}) -> operand", local, name);
                    return LocalRef::new_operand(bx.cx, layout);
                }

                debug!("alloc: {:?} ({}) -> place", local, name);
                if layout.is_unsized() {
                    let indirect_place =
                        PlaceRef::alloca_unsized_indirect(&bx, layout, &name.as_str());
                    // FIXME: add an appropriate debuginfo
                    LocalRef::UnsizedPlace(indirect_place)
                } else {
                    let place = PlaceRef::alloca(&bx, layout, &name.as_str());
                    if dbg {
                        let (scope, span) = fx.debug_loc(mir::SourceInfo {
                            span: decl.source_info.span,
                            scope: decl.visibility_scope,
                        });
                        declare_local(&bx, &fx.debug_context, name, layout.ty, scope.unwrap(),
                            VariableAccess::DirectVariable { alloca: place.llval },
                            VariableKind::LocalVariable, span);
                    }
                    LocalRef::Place(place)
                }
            } else {
                // Temporary or return place
                if local == mir::RETURN_PLACE && fx.fn_ty.ret.is_indirect() {
                    debug!("alloc: {:?} (return place) -> place", local);
                    let llretptr = llvm::get_param(llfn, 0);
                    LocalRef::Place(PlaceRef::new_sized(llretptr, layout, layout.align))
                } else if memory_locals.contains(local) {
                    debug!("alloc: {:?} -> place", local);
                    if layout.is_unsized() {
                        let indirect_place =
                            PlaceRef::alloca_unsized_indirect(&bx, layout, &format!("{:?}", local));
                        LocalRef::UnsizedPlace(indirect_place)
                    } else {
                        LocalRef::Place(PlaceRef::alloca(&bx, layout, &format!("{:?}", local)))
                    }
                } else {
                    // If this is an immediate local, we do not create an
                    // alloca in advance. Instead we wait until we see the
                    // definition and update the operand there.
                    debug!("alloc: {:?} -> operand", local);
                    LocalRef::new_operand(bx.cx, layout)
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
    debuginfo::start_emitting_source_locations(&fx.debug_context);

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
                llvm::LLVMDeleteBasicBlock(fx.blocks[bb]);
            }
        }
    }
}

fn create_funclets(
    mir: &'a Mir<'tcx>,
    bx: &Builder<'a, 'll, 'tcx>,
    cleanup_kinds: &IndexVec<mir::BasicBlock, CleanupKind>,
    block_bxs: &IndexVec<mir::BasicBlock, &'ll BasicBlock>)
    -> (IndexVec<mir::BasicBlock, Option<&'ll BasicBlock>>,
        IndexVec<mir::BasicBlock, Option<Funclet<'ll>>>)
{
    block_bxs.iter_enumerated().zip(cleanup_kinds).map(|((bb, &llbb), cleanup_kind)| {
        match *cleanup_kind {
            CleanupKind::Funclet if base::wants_msvc_seh(bx.sess()) => {}
            _ => return (None, None)
        }

        let cleanup;
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
                let cs_bx = bx.build_sibling_block(&format!("cs_funclet{:?}", bb));
                let cp_bx = bx.build_sibling_block(&format!("cp_funclet{:?}", bb));
                ret_llbb = cs_bx.llbb();

                let cs = cs_bx.catch_switch(None, None, 1);
                cs_bx.add_handler(cs, cp_bx.llbb());

                // The "null" here is actually a RTTI type descriptor for the
                // C++ personality function, but `catch (...)` has no type so
                // it's null. The 64 here is actually a bitfield which
                // represents that this is a catch-all block.
                let null = C_null(Type::i8p(bx.cx));
                let sixty_four = C_i32(bx.cx, 64);
                cleanup = cp_bx.catch_pad(cs, &[null, sixty_four, null]);
                cp_bx.br(llbb);
            }
            _ => {
                let cleanup_bx = bx.build_sibling_block(&format!("funclet_{:?}", bb));
                ret_llbb = cleanup_bx.llbb();
                cleanup = cleanup_bx.cleanup_pad(None, &[]);
                cleanup_bx.br(llbb);
            }
        };

        (Some(ret_llbb), Some(Funclet::new(cleanup)))
    }).unzip()
}

/// Produce, for each argument, a `Value` pointing at the
/// argument's value. As arguments are places, these are always
/// indirect.
fn arg_local_refs(
    bx: &Builder<'a, 'll, 'tcx>,
    fx: &FunctionCx<'a, 'll, 'tcx>,
    scopes: &IndexVec<mir::SourceScope, debuginfo::MirDebugScope<'ll>>,
    memory_locals: &BitSet<mir::Local>,
) -> Vec<LocalRef<'ll, 'tcx>> {
    let mir = fx.mir;
    let tcx = bx.tcx();
    let mut idx = 0;
    let mut llarg_idx = fx.fn_ty.ret.is_indirect() as usize;

    // Get the argument scope, if it exists and if we need it.
    let arg_scope = scopes[mir::OUTERMOST_SOURCE_SCOPE];
    let arg_scope = if bx.sess().opts.debuginfo == DebugInfo::Full {
        arg_scope.scope_metadata
    } else {
        None
    };

    mir.args_iter().enumerate().map(|(arg_index, local)| {
        let arg_decl = &mir.local_decls[local];

        let name = if let Some(name) = arg_decl.name {
            name.as_str().to_string()
        } else {
            format!("arg{}", arg_index)
        };

        if Some(local) == mir.spread_arg {
            // This argument (e.g. the last argument in the "rust-call" ABI)
            // is a tuple that was spread at the ABI level and now we have
            // to reconstruct it into a tuple local variable, from multiple
            // individual LLVM function arguments.

            let arg_ty = fx.monomorphize(&arg_decl.ty);
            let tupled_arg_tys = match arg_ty.sty {
                ty::Tuple(ref tys) => tys,
                _ => bug!("spread argument isn't a tuple?!")
            };

            let place = PlaceRef::alloca(bx, bx.cx.layout_of(arg_ty), &name);
            for i in 0..tupled_arg_tys.len() {
                let arg = &fx.fn_ty.args[idx];
                idx += 1;
                if arg.pad.is_some() {
                    llarg_idx += 1;
                }
                arg.store_fn_arg(bx, &mut llarg_idx, place.project_field(bx, i));
            }

            // Now that we have one alloca that contains the aggregate value,
            // we can create one debuginfo entry for the argument.
            arg_scope.map(|scope| {
                let variable_access = VariableAccess::DirectVariable {
                    alloca: place.llval
                };
                declare_local(
                    bx,
                    &fx.debug_context,
                    arg_decl.name.unwrap_or(keywords::Invalid.name()),
                    arg_ty, scope,
                    variable_access,
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
            });

            return LocalRef::Place(place);
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
                    return local(OperandRef::new_zst(bx.cx, arg.layout));
                }
                PassMode::Direct(_) => {
                    let llarg = llvm::get_param(bx.llfn(), llarg_idx as c_uint);
                    bx.set_value_name(llarg, &name);
                    llarg_idx += 1;
                    return local(
                        OperandRef::from_immediate_or_packed_pair(bx, llarg, arg.layout));
                }
                PassMode::Pair(..) => {
                    let a = llvm::get_param(bx.llfn(), llarg_idx as c_uint);
                    bx.set_value_name(a, &(name.clone() + ".0"));
                    llarg_idx += 1;

                    let b = llvm::get_param(bx.llfn(), llarg_idx as c_uint);
                    bx.set_value_name(b, &(name + ".1"));
                    llarg_idx += 1;

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
            let llarg = llvm::get_param(bx.llfn(), llarg_idx as c_uint);
            bx.set_value_name(llarg, &name);
            llarg_idx += 1;
            PlaceRef::new_sized(llarg, arg.layout, arg.layout.align)
        } else if arg.is_unsized_indirect() {
            // As the storage for the indirect argument lives during
            // the whole function call, we just copy the fat pointer.
            let llarg = llvm::get_param(bx.llfn(), llarg_idx as c_uint);
            llarg_idx += 1;
            let llextra = llvm::get_param(bx.llfn(), llarg_idx as c_uint);
            llarg_idx += 1;
            let indirect_operand = OperandValue::Pair(llarg, llextra);

            let tmp = PlaceRef::alloca_unsized_indirect(bx, arg.layout, &name);
            indirect_operand.store(&bx, tmp);
            tmp
        } else {
            let tmp = PlaceRef::alloca(bx, arg.layout, &name);
            arg.store_fn_arg(bx, &mut llarg_idx, tmp);
            tmp
        };
        arg_scope.map(|scope| {
            // Is this a regular argument?
            if arg_index > 0 || mir.upvar_decls.is_empty() {
                // The Rust ABI passes indirect variables using a pointer and a manual copy, so we
                // need to insert a deref here, but the C ABI uses a pointer and a copy using the
                // byval attribute, for which LLVM always does the deref itself,
                // so we must not add it.
                let variable_access = VariableAccess::DirectVariable {
                    alloca: place.llval
                };

                declare_local(
                    bx,
                    &fx.debug_context,
                    arg_decl.name.unwrap_or(keywords::Invalid.name()),
                    arg.layout.ty,
                    scope,
                    variable_access,
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
                return;
            }

            // Or is it the closure environment?
            let (closure_layout, env_ref) = match arg.layout.ty.sty {
                ty::RawPtr(ty::TypeAndMut { ty, .. }) |
                ty::Ref(_, ty, _)  => (bx.cx.layout_of(ty), true),
                _ => (arg.layout, false)
            };

            let (def_id, upvar_substs) = match closure_layout.ty.sty {
                ty::Closure(def_id, substs) => (def_id, UpvarSubsts::Closure(substs)),
                ty::Generator(def_id, substs, _) => (def_id, UpvarSubsts::Generator(substs)),
                _ => bug!("upvar_decls with non-closure arg0 type `{}`", closure_layout.ty)
            };
            let upvar_tys = upvar_substs.upvar_tys(def_id, tcx);

            // Store the pointer to closure data in an alloca for debuginfo
            // because that's what the llvm.dbg.declare intrinsic expects.

            // FIXME(eddyb) this shouldn't be necessary but SROA seems to
            // mishandle DW_OP_plus not preceded by DW_OP_deref, i.e. it
            // doesn't actually strip the offset when splitting the closure
            // environment into its components so it ends up out of bounds.
            // (cuviper) It seems to be fine without the alloca on LLVM 6 and later.
            let env_alloca = !env_ref && unsafe { llvm::LLVMRustVersionMajor() < 6 };
            let env_ptr = if env_alloca {
                let scratch = PlaceRef::alloca(bx,
                    bx.cx.layout_of(tcx.mk_mut_ptr(arg.layout.ty)),
                    "__debuginfo_env_ptr");
                bx.store(place.llval, scratch.llval, scratch.align);
                scratch.llval
            } else {
                place.llval
            };

            for (i, (decl, ty)) in mir.upvar_decls.iter().zip(upvar_tys).enumerate() {
                let byte_offset_of_var_in_env = closure_layout.fields.offset(i).bytes();

                let ops = unsafe {
                    [llvm::LLVMRustDIBuilderCreateOpDeref(),
                     llvm::LLVMRustDIBuilderCreateOpPlusUconst(),
                     byte_offset_of_var_in_env as i64,
                     llvm::LLVMRustDIBuilderCreateOpDeref()]
                };

                // The environment and the capture can each be indirect.

                // FIXME(eddyb) see above why we sometimes have to keep
                // a pointer in an alloca for debuginfo atm.
                let mut ops = if env_ref || env_alloca { &ops[..] } else { &ops[1..] };

                let ty = if let (true, &ty::Ref(_, ty, _)) = (decl.by_ref, &ty.sty) {
                    ty
                } else {
                    ops = &ops[..ops.len() - 1];
                    ty
                };

                let variable_access = VariableAccess::IndirectVariable {
                    alloca: env_ptr,
                    address_operations: &ops
                };
                declare_local(
                    bx,
                    &fx.debug_context,
                    decl.debug_name,
                    ty,
                    scope,
                    variable_access,
                    VariableKind::LocalVariable,
                    DUMMY_SP
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
mod constant;
pub mod place;
pub mod operand;
mod rvalue;
mod statement;
