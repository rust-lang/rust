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
use llvm::{self, ValueRef};
use llvm::debuginfo::DIScope;
use rustc::ty;
use rustc::mir::repr as mir;
use rustc::mir::tcx::LvalueTy;
use session::config::FullDebugInfo;
use base;
use common::{self, Block, BlockAndBuilder, CrateContext, FunctionContext, C_null};
use debuginfo::{self, declare_local, DebugLoc, VariableAccess, VariableKind};
use machine;
use type_of;

use syntax_pos::DUMMY_SP;
use syntax::parse::token::keywords;

use std::ops::Deref;
use std::rc::Rc;

use basic_block::BasicBlock;

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};

pub use self::constant::trans_static_initializer;

use self::lvalue::{LvalueRef, get_dataptr, get_meta};
use rustc::mir::traversal;

use self::operand::{OperandRef, OperandValue};

#[derive(Clone)]
pub enum CachedMir<'mir, 'tcx: 'mir> {
    Ref(&'mir mir::Mir<'tcx>),
    Owned(Rc<mir::Mir<'tcx>>)
}

impl<'mir, 'tcx: 'mir> Deref for CachedMir<'mir, 'tcx> {
    type Target = mir::Mir<'tcx>;
    fn deref(&self) -> &mir::Mir<'tcx> {
        match *self {
            CachedMir::Ref(r) => r,
            CachedMir::Owned(ref rc) => rc
        }
    }
}

/// Master context for translating MIR.
pub struct MirContext<'bcx, 'tcx:'bcx> {
    mir: CachedMir<'bcx, 'tcx>,

    /// Function context
    fcx: &'bcx common::FunctionContext<'bcx, 'tcx>,

    /// When unwinding is initiated, we have to store this personality
    /// value somewhere so that we can load it and re-use it in the
    /// resume instruction. The personality is (afaik) some kind of
    /// value used for C++ unwinding, which must filter by type: we
    /// don't really care about it very much. Anyway, this value
    /// contains an alloca into which the personality is stored and
    /// then later loaded when generating the DIVERGE_BLOCK.
    llpersonalityslot: Option<ValueRef>,

    /// A `Block` for each MIR `BasicBlock`
    blocks: IndexVec<mir::BasicBlock, Block<'bcx, 'tcx>>,

    /// The funclet status of each basic block
    cleanup_kinds: IndexVec<mir::BasicBlock, analyze::CleanupKind>,

    /// This stores the landing-pad block for a given BB, computed lazily on GNU
    /// and eagerly on MSVC.
    landing_pads: IndexVec<mir::BasicBlock, Option<Block<'bcx, 'tcx>>>,

    /// Cached unreachable block
    unreachable_block: Option<Block<'bcx, 'tcx>>,

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
    scopes: IndexVec<mir::VisibilityScope, DIScope>
}

impl<'blk, 'tcx> MirContext<'blk, 'tcx> {
    pub fn debug_loc(&self, source_info: mir::SourceInfo) -> DebugLoc {
        DebugLoc::ScopeAt(self.scopes[source_info.scope], source_info.span)
    }
}

enum LocalRef<'tcx> {
    Lvalue(LvalueRef<'tcx>),
    Operand(Option<OperandRef<'tcx>>),
}

impl<'tcx> LocalRef<'tcx> {
    fn new_operand<'bcx>(ccx: &CrateContext<'bcx, 'tcx>,
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

pub fn trans_mir<'blk, 'tcx: 'blk>(fcx: &'blk FunctionContext<'blk, 'tcx>) {
    let bcx = fcx.init(false, None).build();
    let mir = bcx.mir();

    // Analyze the temps to determine which must be lvalues
    // FIXME
    let (lvalue_locals, cleanup_kinds) = bcx.with_block(|bcx| {
        (analyze::lvalue_locals(bcx, &mir),
         analyze::cleanup_kinds(bcx, &mir))
    });

    // Compute debuginfo scopes from MIR scopes.
    let scopes = debuginfo::create_mir_scopes(fcx);

    // Allocate variable and temp allocas
    let locals = {
        let args = arg_local_refs(&bcx, &mir, &scopes, &lvalue_locals);
        let vars = mir.var_decls.iter().enumerate().map(|(i, decl)| {
            let ty = bcx.monomorphize(&decl.ty);
            let scope = scopes[decl.source_info.scope];
            let dbg = !scope.is_null() && bcx.sess().opts.debuginfo == FullDebugInfo;

            let local = mir.local_index(&mir::Lvalue::Var(mir::Var::new(i))).unwrap();
            if !lvalue_locals.contains(local.index()) && !dbg {
                return LocalRef::new_operand(bcx.ccx(), ty);
            }

            let lvalue = LvalueRef::alloca(&bcx, ty, &decl.name.as_str());
            if dbg {
                bcx.with_block(|bcx| {
                    declare_local(bcx, decl.name, ty, scope,
                                VariableAccess::DirectVariable { alloca: lvalue.llval },
                                VariableKind::LocalVariable, decl.source_info.span);
                });
            }
            LocalRef::Lvalue(lvalue)
        });

        let locals = mir.temp_decls.iter().enumerate().map(|(i, decl)| {
            (mir::Lvalue::Temp(mir::Temp::new(i)), decl.ty)
        }).chain(mir.return_ty.maybe_converging().map(|ty| (mir::Lvalue::ReturnPointer, ty)));

        args.into_iter().chain(vars).chain(locals.map(|(lvalue, ty)| {
            let ty = bcx.monomorphize(&ty);
            let local = mir.local_index(&lvalue).unwrap();
            if lvalue == mir::Lvalue::ReturnPointer && fcx.fn_ty.ret.is_indirect() {
                let llretptr = llvm::get_param(fcx.llfn, 0);
                LocalRef::Lvalue(LvalueRef::new_sized(llretptr, LvalueTy::from_ty(ty)))
            } else if lvalue_locals.contains(local.index()) {
                LocalRef::Lvalue(LvalueRef::alloca(&bcx, ty, &format!("{:?}", lvalue)))
            } else {
                // If this is an immediate local, we do not create an
                // alloca in advance. Instead we wait until we see the
                // definition and update the operand there.
                LocalRef::new_operand(bcx.ccx(), ty)
            }
        })).collect()
    };

    // Allocate a `Block` for every basic block
    let block_bcxs: IndexVec<mir::BasicBlock, Block<'blk,'tcx>> =
        mir.basic_blocks().indices().map(|bb| {
            if bb == mir::START_BLOCK {
                fcx.new_block("start", None)
            } else {
                fcx.new_block(&format!("{:?}", bb), None)
            }
        }).collect();

    // Branch to the START block
    let start_bcx = block_bcxs[mir::START_BLOCK];
    bcx.br(start_bcx.llbb);

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(fcx);

    let mut mircx = MirContext {
        mir: mir.clone(),
        fcx: fcx,
        llpersonalityslot: None,
        blocks: block_bcxs,
        unreachable_block: None,
        cleanup_kinds: cleanup_kinds,
        landing_pads: IndexVec::from_elem(None, mir.basic_blocks()),
        locals: locals,
        scopes: scopes
    };

    let mut visited = BitVector::new(mir.basic_blocks().len());

    let mut rpo = traversal::reverse_postorder(&mir);

    // Prepare each block for translation.
    for (bb, _) in rpo.by_ref() {
        mircx.init_cpad(bb);
    }
    rpo.reset();

    // Translate the body of each block using reverse postorder
    for (bb, _) in rpo {
        visited.insert(bb.index());
        mircx.trans_block(bb);
    }

    // Remove blocks that haven't been visited, or have no
    // predecessors.
    for bb in mir.basic_blocks().indices() {
        let block = mircx.blocks[bb];
        let block = BasicBlock(block.llbb);
        // Unreachable block
        if !visited.contains(bb.index()) {
            debug!("trans_mir: block {:?} was not visited", bb);
            block.delete();
        }
    }

    DebugLoc::None.apply(fcx);
    fcx.cleanup();
}

/// Produce, for each argument, a `ValueRef` pointing at the
/// argument's value. As arguments are lvalues, these are always
/// indirect.
fn arg_local_refs<'bcx, 'tcx>(bcx: &BlockAndBuilder<'bcx, 'tcx>,
                              mir: &mir::Mir<'tcx>,
                              scopes: &IndexVec<mir::VisibilityScope, DIScope>,
                              lvalue_locals: &BitVector)
                              -> Vec<LocalRef<'tcx>> {
    let fcx = bcx.fcx();
    let tcx = bcx.tcx();
    let mut idx = 0;
    let mut llarg_idx = fcx.fn_ty.ret.is_indirect() as usize;

    // Get the argument scope, if it exists and if we need it.
    let arg_scope = scopes[mir::ARGUMENT_VISIBILITY_SCOPE];
    let arg_scope = if !arg_scope.is_null() && bcx.sess().opts.debuginfo == FullDebugInfo {
        Some(arg_scope)
    } else {
        None
    };

    mir.arg_decls.iter().enumerate().map(|(arg_index, arg_decl)| {
        let arg_ty = bcx.monomorphize(&arg_decl.ty);
        let local = mir.local_index(&mir::Lvalue::Arg(mir::Arg::new(arg_index))).unwrap();
        if arg_decl.spread {
            // This argument (e.g. the last argument in the "rust-call" ABI)
            // is a tuple that was spread at the ABI level and now we have
            // to reconstruct it into a tuple local variable, from multiple
            // individual LLVM function arguments.

            let tupled_arg_tys = match arg_ty.sty {
                ty::TyTuple(ref tys) => tys,
                _ => bug!("spread argument isn't a tuple?!")
            };

            let lltuplety = type_of::type_of(bcx.ccx(), arg_ty);
            let lltemp = bcx.with_block(|bcx| {
                base::alloc_ty(bcx, arg_ty, &format!("arg{}", arg_index))
            });
            for (i, &tupled_arg_ty) in tupled_arg_tys.iter().enumerate() {
                let dst = bcx.struct_gep(lltemp, i);
                let arg = &fcx.fn_ty.args[idx];
                idx += 1;
                if common::type_is_fat_ptr(tcx, tupled_arg_ty) {
                    // We pass fat pointers as two words, but inside the tuple
                    // they are the two sub-fields of a single aggregate field.
                    let meta = &fcx.fn_ty.args[idx];
                    idx += 1;
                    arg.store_fn_arg(bcx, &mut llarg_idx, get_dataptr(bcx, dst));
                    meta.store_fn_arg(bcx, &mut llarg_idx, get_meta(bcx, dst));
                } else {
                    arg.store_fn_arg(bcx, &mut llarg_idx, dst);
                }

                bcx.with_block(|bcx| arg_scope.map(|scope| {
                    let byte_offset_of_var_in_tuple =
                        machine::llelement_offset(bcx.ccx(), lltuplety, i);

                    let ops = unsafe {
                        [llvm::LLVMDIBuilderCreateOpDeref(),
                         llvm::LLVMDIBuilderCreateOpPlus(),
                         byte_offset_of_var_in_tuple as i64]
                    };

                    let variable_access = VariableAccess::IndirectVariable {
                        alloca: lltemp,
                        address_operations: &ops
                    };
                    declare_local(bcx, keywords::Invalid.name(),
                                  tupled_arg_ty, scope, variable_access,
                                  VariableKind::ArgumentVariable(arg_index + i + 1),
                                  bcx.fcx().span.unwrap_or(DUMMY_SP));
                }));
            }
            return LocalRef::Lvalue(LvalueRef::new_sized(lltemp, LvalueTy::from_ty(arg_ty)));
        }

        let arg = &fcx.fn_ty.args[idx];
        idx += 1;
        let llval = if arg.is_indirect() && bcx.sess().opts.debuginfo != FullDebugInfo {
            // Don't copy an indirect argument to an alloca, the caller
            // already put it in a temporary alloca and gave it up, unless
            // we emit extra-debug-info, which requires local allocas :(.
            // FIXME: lifetimes
            if arg.pad.is_some() {
                llarg_idx += 1;
            }
            let llarg = llvm::get_param(fcx.llfn, llarg_idx as c_uint);
            llarg_idx += 1;
            llarg
        } else if !lvalue_locals.contains(local.index()) &&
                  !arg.is_indirect() && arg.cast.is_none() &&
                  arg_scope.is_none() {
            if arg.is_ignore() {
                return LocalRef::new_operand(bcx.ccx(), arg_ty);
            }

            // We don't have to cast or keep the argument in the alloca.
            // FIXME(eddyb): We should figure out how to use llvm.dbg.value instead
            // of putting everything in allocas just so we can use llvm.dbg.declare.
            if arg.pad.is_some() {
                llarg_idx += 1;
            }
            let llarg = llvm::get_param(fcx.llfn, llarg_idx as c_uint);
            llarg_idx += 1;
            let val = if common::type_is_fat_ptr(tcx, arg_ty) {
                let meta = &fcx.fn_ty.args[idx];
                idx += 1;
                assert_eq!((meta.cast, meta.pad), (None, None));
                let llmeta = llvm::get_param(fcx.llfn, llarg_idx as c_uint);
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
            let lltemp = bcx.with_block(|bcx| {
                base::alloc_ty(bcx, arg_ty, &format!("arg{}", arg_index))
            });
            if common::type_is_fat_ptr(tcx, arg_ty) {
                // we pass fat pointers as two words, but we want to
                // represent them internally as a pointer to two words,
                // so make an alloca to store them in.
                let meta = &fcx.fn_ty.args[idx];
                idx += 1;
                arg.store_fn_arg(bcx, &mut llarg_idx, get_dataptr(bcx, lltemp));
                meta.store_fn_arg(bcx, &mut llarg_idx, get_meta(bcx, lltemp));
            } else  {
                // otherwise, arg is passed by value, so make a
                // temporary and store it there
                arg.store_fn_arg(bcx, &mut llarg_idx, lltemp);
            }
            lltemp
        };
        bcx.with_block(|bcx| arg_scope.map(|scope| {
            // Is this a regular argument?
            if arg_index > 0 || mir.upvar_decls.is_empty() {
                declare_local(bcx, arg_decl.debug_name, arg_ty, scope,
                              VariableAccess::DirectVariable { alloca: llval },
                              VariableKind::ArgumentVariable(arg_index + 1),
                              bcx.fcx().span.unwrap_or(DUMMY_SP));
                return;
            }

            // Or is it the closure environment?
            let (closure_ty, env_ref) = if let ty::TyRef(_, mt) = arg_ty.sty {
                (mt.ty, true)
            } else {
                (arg_ty, false)
            };
            let upvar_tys = if let ty::TyClosure(_, ref substs) = closure_ty.sty {
                &substs.upvar_tys[..]
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
                use base::*;
                use build::*;
                use common::*;
                let alloc = alloca(bcx, val_ty(llval), "__debuginfo_env_ptr");
                Store(bcx, llval, alloc);
                alloc
            } else {
                llval
            };

            let llclosurety = type_of::type_of(bcx.ccx(), closure_ty);
            for (i, (decl, ty)) in mir.upvar_decls.iter().zip(upvar_tys).enumerate() {
                let byte_offset_of_var_in_env =
                    machine::llelement_offset(bcx.ccx(), llclosurety, i);

                let ops = unsafe {
                    [llvm::LLVMDIBuilderCreateOpDeref(),
                     llvm::LLVMDIBuilderCreateOpPlus(),
                     byte_offset_of_var_in_env as i64,
                     llvm::LLVMDIBuilderCreateOpDeref()]
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
                declare_local(bcx, decl.debug_name, ty, scope, variable_access,
                              VariableKind::CapturedVariable,
                              bcx.fcx().span.unwrap_or(DUMMY_SP));
            }
        }));
        LocalRef::Lvalue(LvalueRef::new_sized(llval, LvalueTy::from_ty(arg_ty)))
    }).collect()
}

mod analyze;
mod block;
mod constant;
mod lvalue;
mod operand;
mod rvalue;
mod statement;
