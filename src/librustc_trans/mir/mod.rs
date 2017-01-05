// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::c_uint;
use llvm::{self, ValueRef, BasicBlockRef};
use llvm::debuginfo::DIScope;
use rustc::ty::{self, layout};
use rustc::mir::{self, Mir};
use rustc::mir::tcx::LvalueTy;
use rustc::ty::subst::Substs;
use rustc::infer::TransNormalize;
use rustc::ty::TypeFoldable;
use session::config::FullDebugInfo;
use base;
use builder::Builder;
use common::{self, CrateContext, C_null, Funclet};
use debuginfo::{self, declare_local, VariableAccess, VariableKind, FunctionDebugContext};
use monomorphize::{self, Instance};
use abi::FnType;
use type_of;

use syntax_pos::{DUMMY_SP, NO_EXPANSION, COMMAND_LINE_EXPN, BytePos, Span};
use syntax::symbol::keywords;
use syntax::abi::Abi;

use std::iter;

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};

pub use self::constant::trans_static_initializer;

use self::analyze::CleanupKind;
use self::lvalue::LvalueRef;
use rustc::mir::traversal;

use self::operand::{OperandRef, OperandValue};

/// Master context for translating MIR.
pub struct MirContext<'a, 'tcx:'a> {
    mir: &'a mir::Mir<'tcx>,

    debug_context: debuginfo::FunctionDebugContext,

    llfn: ValueRef,

    ccx: &'a CrateContext<'a, 'tcx>,

    fn_ty: FnType,

    /// When unwinding is initiated, we have to store this personality
    /// value somewhere so that we can load it and re-use it in the
    /// resume instruction. The personality is (afaik) some kind of
    /// value used for C++ unwinding, which must filter by type: we
    /// don't really care about it very much. Anyway, this value
    /// contains an alloca into which the personality is stored and
    /// then later loaded when generating the DIVERGE_BLOCK.
    llpersonalityslot: Option<ValueRef>,

    /// A `Block` for each MIR `BasicBlock`
    blocks: IndexVec<mir::BasicBlock, BasicBlockRef>,

    /// The funclet status of each basic block
    cleanup_kinds: IndexVec<mir::BasicBlock, analyze::CleanupKind>,

    /// This stores the landing-pad block for a given BB, computed lazily on GNU
    /// and eagerly on MSVC.
    landing_pads: IndexVec<mir::BasicBlock, Option<BasicBlockRef>>,

    /// Cached unreachable block
    unreachable_block: Option<BasicBlockRef>,

    /// The location where each MIR arg/var/tmp/ret is stored. This is
    /// usually an `LvalueRef` representing an alloca, but not always:
    /// sometimes we can skip the alloca and just store the value
    /// directly using an `OperandRef`, which makes for tighter LLVM
    /// IR. The conditions for using an `OperandRef` are as follows:
    ///
    /// - the type of the local must be judged "immediate" by `type_is_immediate`
    /// - the operand must never be referenced indirectly
    ///     - we should not take its address using the `&` operator
    ///     - nor should it appear in an lvalue path like `tmp.a`
    /// - the operand must be defined by an rvalue that can generate immediate
    ///   values
    ///
    /// Avoiding allocs can also be important for certain intrinsics,
    /// notably `expect`.
    locals: IndexVec<mir::Local, LocalRef<'tcx>>,

    /// Debug information for MIR scopes.
    scopes: IndexVec<mir::VisibilityScope, debuginfo::MirDebugScope>,

    /// If this function is being monomorphized, this contains the type substitutions used.
    param_substs: &'tcx Substs<'tcx>,
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn monomorphize<T>(&self, value: &T) -> T
        where T: TransNormalize<'tcx> {
        monomorphize::apply_param_substs(self.ccx.shared(), self.param_substs, value)
    }

    pub fn set_debug_loc(&mut self, bcx: &Builder, source_info: mir::SourceInfo) {
        let (scope, span) = self.debug_loc(source_info);
        debuginfo::set_source_location(&self.debug_context, bcx, scope, span);
    }

    pub fn debug_loc(&mut self, source_info: mir::SourceInfo) -> (DIScope, Span) {
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
        if source_info.span.expn_id == NO_EXPANSION ||
            source_info.span.expn_id == COMMAND_LINE_EXPN ||
            self.ccx.sess().opts.debugging_opts.debug_macros {

            let scope = self.scope_metadata_for_loc(source_info.scope, source_info.span.lo);
            (scope, source_info.span)
        } else {
            let cm = self.ccx.sess().codemap();
            // Walk up the macro expansion chain until we reach a non-expanded span.
            let mut span = source_info.span;
            while span.expn_id != NO_EXPANSION && span.expn_id != COMMAND_LINE_EXPN {
                if let Some(callsite_span) = cm.with_expn_info(span.expn_id,
                                                    |ei| ei.map(|ei| ei.call_site.clone())) {
                    span = callsite_span;
                } else {
                    break;
                }
            }
            let scope = self.scope_metadata_for_loc(source_info.scope, span.lo);
            // Use span of the outermost call site, while keeping the original lexical scope
            (scope, span)
        }
    }

    // DILocations inherit source file name from the parent DIScope.  Due to macro expansions
    // it may so happen that the current span belongs to a different file than the DIScope
    // corresponding to span's containing visibility scope.  If so, we need to create a DIScope
    // "extension" into that file.
    fn scope_metadata_for_loc(&self, scope_id: mir::VisibilityScope, pos: BytePos)
                               -> llvm::debuginfo::DIScope {
        let scope_metadata = self.scopes[scope_id].scope_metadata;
        if pos < self.scopes[scope_id].file_start_pos ||
           pos >= self.scopes[scope_id].file_end_pos {
            let cm = self.ccx.sess().codemap();
            debuginfo::extend_scope_to_file(self.ccx, scope_metadata, &cm.lookup_char_pos(pos).file)
        } else {
            scope_metadata
        }
    }
}

enum LocalRef<'tcx> {
    Lvalue(LvalueRef<'tcx>),
    Operand(Option<OperandRef<'tcx>>),
}

impl<'tcx> LocalRef<'tcx> {
    fn new_operand<'a>(ccx: &CrateContext<'a, 'tcx>,
                         ty: ty::Ty<'tcx>) -> LocalRef<'tcx> {
        if common::type_is_zero_size(ccx, ty) {
            // Zero-size temporaries aren't always initialized, which
            // doesn't matter because they don't contain data, but
            // we need something in the operand.
            let llty = type_of::type_of(ccx, ty);
            let val = if common::type_is_imm_pair(ccx, ty) {
                let fields = llty.field_types();
                OperandValue::Pair(C_null(fields[0]), C_null(fields[1]))
            } else {
                OperandValue::Immediate(C_null(llty))
            };
            let op = OperandRef {
                val: val,
                ty: ty
            };
            LocalRef::Operand(Some(op))
        } else {
            LocalRef::Operand(None)
        }
    }
}

///////////////////////////////////////////////////////////////////////////

pub fn trans_mir<'a, 'tcx: 'a>(
    ccx: &'a CrateContext<'a, 'tcx>,
    llfn: ValueRef,
    fn_ty: FnType,
    mir: &'a Mir<'tcx>,
    instance: Instance<'tcx>,
    sig: &ty::FnSig<'tcx>,
    abi: Abi,
) {
    debug!("fn_ty: {:?}", fn_ty);
    let debug_context =
        debuginfo::create_function_debug_context(ccx, instance, sig, abi, llfn, mir);
    let bcx = Builder::new_block(ccx, llfn, "entry-block");

    let cleanup_kinds = analyze::cleanup_kinds(&mir);

    // Allocate a `Block` for every basic block
    let block_bcxs: IndexVec<mir::BasicBlock, BasicBlockRef> =
        mir.basic_blocks().indices().map(|bb| {
            if bb == mir::START_BLOCK {
                bcx.build_sibling_block("start").llbb()
            } else {
                bcx.build_sibling_block(&format!("{:?}", bb)).llbb()
            }
        }).collect();

    // Compute debuginfo scopes from MIR scopes.
    let scopes = debuginfo::create_mir_scopes(ccx, mir, &debug_context);

    let mut mircx = MirContext {
        mir: mir,
        llfn: llfn,
        fn_ty: fn_ty,
        ccx: ccx,
        llpersonalityslot: None,
        blocks: block_bcxs,
        unreachable_block: None,
        cleanup_kinds: cleanup_kinds,
        landing_pads: IndexVec::from_elem(None, mir.basic_blocks()),
        scopes: scopes,
        locals: IndexVec::new(),
        debug_context: debug_context,
        param_substs: {
            assert!(!instance.substs.needs_infer());
            instance.substs
        },
    };

    let lvalue_locals = analyze::lvalue_locals(&mircx);

    // Allocate variable and temp allocas
    mircx.locals = {
        let args = arg_local_refs(&bcx, &mircx, &mircx.scopes, &lvalue_locals);

        let mut allocate_local = |local| {
            let decl = &mir.local_decls[local];
            let ty = mircx.monomorphize(&decl.ty);

            if let Some(name) = decl.name {
                // User variable
                let source_info = decl.source_info.unwrap();
                let debug_scope = mircx.scopes[source_info.scope];
                let dbg = debug_scope.is_valid() && bcx.sess().opts.debuginfo == FullDebugInfo;

                if !lvalue_locals.contains(local.index()) && !dbg {
                    debug!("alloc: {:?} ({}) -> operand", local, name);
                    return LocalRef::new_operand(bcx.ccx, ty);
                }

                debug!("alloc: {:?} ({}) -> lvalue", local, name);
                assert!(!ty.has_erasable_regions());
                let lltemp = bcx.alloca_ty(ty, &name.as_str());
                let lvalue = LvalueRef::new_sized(lltemp, LvalueTy::from_ty(ty));
                if dbg {
                    let (scope, span) = mircx.debug_loc(source_info);
                    declare_local(&bcx, &mircx.debug_context, name, ty, scope,
                        VariableAccess::DirectVariable { alloca: lvalue.llval },
                        VariableKind::LocalVariable, span);
                }
                LocalRef::Lvalue(lvalue)
            } else {
                // Temporary or return pointer
                if local == mir::RETURN_POINTER && mircx.fn_ty.ret.is_indirect() {
                    debug!("alloc: {:?} (return pointer) -> lvalue", local);
                    let llretptr = llvm::get_param(llfn, 0);
                    LocalRef::Lvalue(LvalueRef::new_sized(llretptr, LvalueTy::from_ty(ty)))
                } else if lvalue_locals.contains(local.index()) {
                    debug!("alloc: {:?} -> lvalue", local);
                    assert!(!ty.has_erasable_regions());
                    let lltemp = bcx.alloca_ty(ty, &format!("{:?}", local));
                    LocalRef::Lvalue(LvalueRef::new_sized(lltemp, LvalueTy::from_ty(ty)))
                } else {
                    // If this is an immediate local, we do not create an
                    // alloca in advance. Instead we wait until we see the
                    // definition and update the operand there.
                    debug!("alloc: {:?} -> operand", local);
                    LocalRef::new_operand(bcx.ccx, ty)
                }
            }
        };

        let retptr = allocate_local(mir::RETURN_POINTER);
        iter::once(retptr)
            .chain(args.into_iter())
            .chain(mir.vars_and_temps_iter().map(allocate_local))
            .collect()
    };

    // Branch to the START block
    let start_bcx = mircx.blocks[mir::START_BLOCK];
    bcx.br(start_bcx);

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(&mircx.debug_context);

    let funclets: IndexVec<mir::BasicBlock, Option<Funclet>> =
    mircx.cleanup_kinds.iter_enumerated().map(|(bb, cleanup_kind)| {
        if let CleanupKind::Funclet = *cleanup_kind {
            let bcx = mircx.get_builder(bb);
            bcx.set_personality_fn(mircx.ccx.eh_personality());
            if base::wants_msvc_seh(ccx.sess()) {
                return Some(Funclet::new(bcx.cleanup_pad(None, &[])));
            }
        }

        None
    }).collect();

    let rpo = traversal::reverse_postorder(&mir);
    let mut visited = BitVector::new(mir.basic_blocks().len());

    // Translate the body of each block using reverse postorder
    for (bb, _) in rpo {
        visited.insert(bb.index());
        mircx.trans_block(bb, &funclets);
    }

    // Remove blocks that haven't been visited, or have no
    // predecessors.
    for bb in mir.basic_blocks().indices() {
        // Unreachable block
        if !visited.contains(bb.index()) {
            debug!("trans_mir: block {:?} was not visited", bb);
            unsafe {
                llvm::LLVMDeleteBasicBlock(mircx.blocks[bb]);
            }
        }
    }
}

/// Produce, for each argument, a `ValueRef` pointing at the
/// argument's value. As arguments are lvalues, these are always
/// indirect.
fn arg_local_refs<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                            mircx: &MirContext<'a, 'tcx>,
                            scopes: &IndexVec<mir::VisibilityScope, debuginfo::MirDebugScope>,
                            lvalue_locals: &BitVector)
                            -> Vec<LocalRef<'tcx>> {
    let mir = mircx.mir;
    let tcx = bcx.tcx();
    let mut idx = 0;
    let mut llarg_idx = mircx.fn_ty.ret.is_indirect() as usize;

    // Get the argument scope, if it exists and if we need it.
    let arg_scope = scopes[mir::ARGUMENT_VISIBILITY_SCOPE];
    let arg_scope = if arg_scope.is_valid() && bcx.sess().opts.debuginfo == FullDebugInfo {
        Some(arg_scope.scope_metadata)
    } else {
        None
    };

    mir.args_iter().enumerate().map(|(arg_index, local)| {
        let arg_decl = &mir.local_decls[local];
        let arg_ty = mircx.monomorphize(&arg_decl.ty);

        if Some(local) == mir.spread_arg {
            // This argument (e.g. the last argument in the "rust-call" ABI)
            // is a tuple that was spread at the ABI level and now we have
            // to reconstruct it into a tuple local variable, from multiple
            // individual LLVM function arguments.

            let tupled_arg_tys = match arg_ty.sty {
                ty::TyTuple(ref tys) => tys,
                _ => bug!("spread argument isn't a tuple?!")
            };

            let lltemp = bcx.alloca_ty(arg_ty, &format!("arg{}", arg_index));
            for (i, &tupled_arg_ty) in tupled_arg_tys.iter().enumerate() {
                let dst = bcx.struct_gep(lltemp, i);
                let arg = &mircx.fn_ty.args[idx];
                idx += 1;
                if common::type_is_fat_ptr(bcx.ccx, tupled_arg_ty) {
                    // We pass fat pointers as two words, but inside the tuple
                    // they are the two sub-fields of a single aggregate field.
                    let meta = &mircx.fn_ty.args[idx];
                    idx += 1;
                    arg.store_fn_arg(bcx, &mut llarg_idx, base::get_dataptr(bcx, dst));
                    meta.store_fn_arg(bcx, &mut llarg_idx, base::get_meta(bcx, dst));
                } else {
                    arg.store_fn_arg(bcx, &mut llarg_idx, dst);
                }
            }

            // Now that we have one alloca that contains the aggregate value,
            // we can create one debuginfo entry for the argument.
            arg_scope.map(|scope| {
                let variable_access = VariableAccess::DirectVariable {
                    alloca: lltemp
                };
                declare_local(
                    bcx,
                    &mircx.debug_context,
                    arg_decl.name.unwrap_or(keywords::Invalid.name()),
                    arg_ty, scope,
                    variable_access,
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
            });

            return LocalRef::Lvalue(LvalueRef::new_sized(lltemp, LvalueTy::from_ty(arg_ty)));
        }

        let arg = &mircx.fn_ty.args[idx];
        idx += 1;
        let llval = if arg.is_indirect() && bcx.sess().opts.debuginfo != FullDebugInfo {
            // Don't copy an indirect argument to an alloca, the caller
            // already put it in a temporary alloca and gave it up, unless
            // we emit extra-debug-info, which requires local allocas :(.
            // FIXME: lifetimes
            if arg.pad.is_some() {
                llarg_idx += 1;
            }
            let llarg = llvm::get_param(bcx.llfn(), llarg_idx as c_uint);
            llarg_idx += 1;
            llarg
        } else if !lvalue_locals.contains(local.index()) &&
                  !arg.is_indirect() && arg.cast.is_none() &&
                  arg_scope.is_none() {
            if arg.is_ignore() {
                return LocalRef::new_operand(bcx.ccx, arg_ty);
            }

            // We don't have to cast or keep the argument in the alloca.
            // FIXME(eddyb): We should figure out how to use llvm.dbg.value instead
            // of putting everything in allocas just so we can use llvm.dbg.declare.
            if arg.pad.is_some() {
                llarg_idx += 1;
            }
            let llarg = llvm::get_param(bcx.llfn(), llarg_idx as c_uint);
            llarg_idx += 1;
            let val = if common::type_is_fat_ptr(bcx.ccx, arg_ty) {
                let meta = &mircx.fn_ty.args[idx];
                idx += 1;
                assert_eq!((meta.cast, meta.pad), (None, None));
                let llmeta = llvm::get_param(bcx.llfn(), llarg_idx as c_uint);
                llarg_idx += 1;
                OperandValue::Pair(llarg, llmeta)
            } else {
                OperandValue::Immediate(llarg)
            };
            let operand = OperandRef {
                val: val,
                ty: arg_ty
            };
            return LocalRef::Operand(Some(operand.unpack_if_pair(bcx)));
        } else {
            let lltemp = bcx.alloca_ty(arg_ty, &format!("arg{}", arg_index));
            if common::type_is_fat_ptr(bcx.ccx, arg_ty) {
                // we pass fat pointers as two words, but we want to
                // represent them internally as a pointer to two words,
                // so make an alloca to store them in.
                let meta = &mircx.fn_ty.args[idx];
                idx += 1;
                arg.store_fn_arg(bcx, &mut llarg_idx, base::get_dataptr(bcx, lltemp));
                meta.store_fn_arg(bcx, &mut llarg_idx, base::get_meta(bcx, lltemp));
            } else  {
                // otherwise, arg is passed by value, so make a
                // temporary and store it there
                arg.store_fn_arg(bcx, &mut llarg_idx, lltemp);
            }
            lltemp
        };
        arg_scope.map(|scope| {
            // Is this a regular argument?
            if arg_index > 0 || mir.upvar_decls.is_empty() {
                declare_local(
                    bcx,
                    &mircx.debug_context,
                    arg_decl.name.unwrap_or(keywords::Invalid.name()),
                    arg_ty,
                    scope,
                    VariableAccess::DirectVariable { alloca: llval },
                    VariableKind::ArgumentVariable(arg_index + 1),
                    DUMMY_SP
                );
                return;
            }

            // Or is it the closure environment?
            let (closure_ty, env_ref) = if let ty::TyRef(_, mt) = arg_ty.sty {
                (mt.ty, true)
            } else {
                (arg_ty, false)
            };
            let upvar_tys = if let ty::TyClosure(def_id, substs) = closure_ty.sty {
                substs.upvar_tys(def_id, tcx)
            } else {
                bug!("upvar_decls with non-closure arg0 type `{}`", closure_ty);
            };

            // Store the pointer to closure data in an alloca for debuginfo
            // because that's what the llvm.dbg.declare intrinsic expects.

            // FIXME(eddyb) this shouldn't be necessary but SROA seems to
            // mishandle DW_OP_plus not preceded by DW_OP_deref, i.e. it
            // doesn't actually strip the offset when splitting the closure
            // environment into its components so it ends up out of bounds.
            let env_ptr = if !env_ref {
                let alloc = bcx.alloca(common::val_ty(llval), "__debuginfo_env_ptr");
                bcx.store(llval, alloc, None);
                alloc
            } else {
                llval
            };

            let layout = bcx.ccx.layout_of(closure_ty);
            let offsets = match *layout {
                layout::Univariant { ref variant, .. } => &variant.offsets[..],
                _ => bug!("Closures are only supposed to be Univariant")
            };

            for (i, (decl, ty)) in mir.upvar_decls.iter().zip(upvar_tys).enumerate() {
                let byte_offset_of_var_in_env = offsets[i].bytes();

                let ops = unsafe {
                    [llvm::LLVMRustDIBuilderCreateOpDeref(),
                     llvm::LLVMRustDIBuilderCreateOpPlus(),
                     byte_offset_of_var_in_env as i64,
                     llvm::LLVMRustDIBuilderCreateOpDeref()]
                };

                // The environment and the capture can each be indirect.

                // FIXME(eddyb) see above why we have to keep
                // a pointer in an alloca for debuginfo atm.
                let mut ops = if env_ref || true { &ops[..] } else { &ops[1..] };

                let ty = if let (true, &ty::TyRef(_, mt)) = (decl.by_ref, &ty.sty) {
                    mt.ty
                } else {
                    ops = &ops[..ops.len() - 1];
                    ty
                };

                let variable_access = VariableAccess::IndirectVariable {
                    alloca: env_ptr,
                    address_operations: &ops
                };
                declare_local(
                    bcx,
                    &mircx.debug_context,
                    decl.debug_name,
                    ty,
                    scope,
                    variable_access,
                    VariableKind::CapturedVariable,
                    DUMMY_SP
                );
            }
        });
        LocalRef::Lvalue(LvalueRef::new_sized(llval, LvalueTy::from_ty(arg_ty)))
    }).collect()
}

mod analyze;
mod block;
mod constant;
pub mod lvalue;
mod operand;
mod rvalue;
mod statement;
