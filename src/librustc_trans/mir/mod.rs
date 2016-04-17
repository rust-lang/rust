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
use common::{self, Block, BlockAndBuilder, FunctionContext};
use debuginfo::{self, declare_local, DebugLoc, VariableAccess, VariableKind};
use machine;
use type_of;

use syntax::codemap::DUMMY_SP;
use syntax::parse::token;

use std::ops::Deref;
use std::rc::Rc;

use basic_block::BasicBlock;

use rustc_data_structures::bitvec::BitVector;

use self::lvalue::{LvalueRef, get_dataptr, get_meta};
use rustc_mir::traversal;

use self::operand::OperandRef;

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
    blocks: Vec<Block<'bcx, 'tcx>>,

    /// Cached unreachable block
    unreachable_block: Option<Block<'bcx, 'tcx>>,

    /// An LLVM alloca for each MIR `VarDecl`
    vars: Vec<LvalueRef<'tcx>>,

    /// The location where each MIR `TempDecl` is stored. This is
    /// usually an `LvalueRef` representing an alloca, but not always:
    /// sometimes we can skip the alloca and just store the value
    /// directly using an `OperandRef`, which makes for tighter LLVM
    /// IR. The conditions for using an `OperandRef` are as follows:
    ///
    /// - the type of the temporary must be judged "immediate" by `type_is_immediate`
    /// - the operand must never be referenced indirectly
    ///     - we should not take its address using the `&` operator
    ///     - nor should it appear in an lvalue path like `tmp.a`
    /// - the operand must be defined by an rvalue that can generate immediate
    ///   values
    ///
    /// Avoiding allocs can also be important for certain intrinsics,
    /// notably `expect`.
    temps: Vec<TempRef<'tcx>>,

    /// The arguments to the function; as args are lvalues, these are
    /// always indirect, though we try to avoid creating an alloca
    /// when we can (and just reuse the pointer the caller provided).
    args: Vec<LvalueRef<'tcx>>,

    /// Debug information for MIR scopes.
    scopes: Vec<DIScope>
}

enum TempRef<'tcx> {
    Lvalue(LvalueRef<'tcx>),
    Operand(Option<OperandRef<'tcx>>),
}

///////////////////////////////////////////////////////////////////////////

pub fn trans_mir<'blk, 'tcx: 'blk>(fcx: &'blk FunctionContext<'blk, 'tcx>) {
    let bcx = fcx.init(false, None).build();
    let mir = bcx.mir();

    let mir_blocks = mir.all_basic_blocks();

    // Analyze the temps to determine which must be lvalues
    // FIXME
    let lvalue_temps = bcx.with_block(|bcx| {
      analyze::lvalue_temps(bcx, &mir)
    });

    // Compute debuginfo scopes from MIR scopes.
    let scopes = debuginfo::create_mir_scopes(fcx);

    // Allocate variable and temp allocas
    let args = arg_value_refs(&bcx, &mir, &scopes);
    let vars = mir.var_decls.iter()
                            .map(|decl| (bcx.monomorphize(&decl.ty), decl))
                            .map(|(mty, decl)| {
        let lvalue = LvalueRef::alloca(&bcx, mty, &decl.name.as_str());

        let scope = scopes[decl.scope.index()];
        if !scope.is_null() && bcx.sess().opts.debuginfo == FullDebugInfo {
            bcx.with_block(|bcx| {
                declare_local(bcx, decl.name, mty, scope,
                              VariableAccess::DirectVariable { alloca: lvalue.llval },
                              VariableKind::LocalVariable, decl.span);
            });
        }

        lvalue
    }).collect();
    let temps = mir.temp_decls.iter()
                              .map(|decl| bcx.monomorphize(&decl.ty))
                              .enumerate()
                              .map(|(i, mty)| if lvalue_temps.contains(i) {
                                  TempRef::Lvalue(LvalueRef::alloca(&bcx,
                                                                    mty,
                                                                    &format!("temp{:?}", i)))
                              } else {
                                  // If this is an immediate temp, we do not create an
                                  // alloca in advance. Instead we wait until we see the
                                  // definition and update the operand there.
                                  TempRef::Operand(None)
                              })
                              .collect();

    // Allocate a `Block` for every basic block
    let block_bcxs: Vec<Block<'blk,'tcx>> =
        mir_blocks.iter()
                  .map(|&bb|{
                      if bb == mir::START_BLOCK {
                          fcx.new_block("start", None)
                      } else if bb == mir::END_BLOCK {
                          fcx.new_block("end", None)
                      } else {
                          fcx.new_block(&format!("{:?}", bb), None)
                      }
                  })
                  .collect();

    // Branch to the START block
    let start_bcx = block_bcxs[mir::START_BLOCK.index()];
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
        vars: vars,
        temps: temps,
        args: args,
        scopes: scopes
    };

    let mut visited = BitVector::new(mir_blocks.len());

    let rpo = traversal::reverse_postorder(&mir);
    // Translate the body of each block using reverse postorder
    for (bb, _) in rpo {
        visited.insert(bb.index());
        mircx.trans_block(bb);
    }

    // Remove blocks that haven't been visited, or have no
    // predecessors.
    for &bb in &mir_blocks {
        let block = mircx.blocks[bb.index()];
        let block = BasicBlock(block.llbb);
        // Unreachable block
        if !visited.contains(bb.index()) {
            block.delete();
        } else if block.pred_iter().count() == 0 {
            block.delete();
        }
    }

    DebugLoc::None.apply(fcx);
    fcx.cleanup();
}

/// Produce, for each argument, a `ValueRef` pointing at the
/// argument's value. As arguments are lvalues, these are always
/// indirect.
fn arg_value_refs<'bcx, 'tcx>(bcx: &BlockAndBuilder<'bcx, 'tcx>,
                              mir: &mir::Mir<'tcx>,
                              scopes: &[DIScope])
                              -> Vec<LvalueRef<'tcx>> {
    let fcx = bcx.fcx();
    let tcx = bcx.tcx();
    let mut idx = 0;
    let mut llarg_idx = fcx.fn_ty.ret.is_indirect() as usize;

    // Get the argument scope assuming ScopeId(0) has no parent.
    let arg_scope = mir.scopes.get(0).and_then(|data| {
        let scope = scopes[0];
        if data.parent_scope.is_none() && !scope.is_null() &&
           bcx.sess().opts.debuginfo == FullDebugInfo {
            Some(scope)
        } else {
            None
        }
    });

    mir.arg_decls.iter().enumerate().map(|(arg_index, arg_decl)| {
        let arg_ty = bcx.monomorphize(&arg_decl.ty);
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
                    declare_local(bcx, token::special_idents::invalid.name,
                                  tupled_arg_ty, scope, variable_access,
                                  VariableKind::ArgumentVariable(arg_index + i + 1),
                                  bcx.fcx().span.unwrap_or(DUMMY_SP));
                }));
            }
            return LvalueRef::new_sized(lltemp, LvalueTy::from_ty(arg_ty));
        }

        let arg = &fcx.fn_ty.args[idx];
        idx += 1;
        let llval = if arg.is_indirect() && bcx.sess().opts.debuginfo != FullDebugInfo {
            // Don't copy an indirect argument to an alloca, the caller
            // already put it in a temporary alloca and gave it up, unless
            // we emit extra-debug-info, which requires local allocas :(.
            // FIXME: lifetimes
            let llarg = llvm::get_param(fcx.llfn, llarg_idx as c_uint);
            llarg_idx += 1;
            llarg
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
        LvalueRef::new_sized(llval, LvalueTy::from_ty(arg_ty))
    }).collect()
}

mod analyze;
mod block;
mod constant;
mod drop;
mod lvalue;
mod operand;
mod rvalue;
mod statement;
