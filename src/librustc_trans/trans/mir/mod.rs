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
use middle::ty;
use rustc::mir::repr as mir;
use rustc::mir::tcx::LvalueTy;
use trans::base;
use trans::common::{self, Block, BlockAndBuilder, FunctionContext};

use std::ops::Deref;
use std::rc::Rc;

use self::lvalue::{LvalueRef, get_dataptr, get_meta};
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

// FIXME DebugLoc is always None right now

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
}

enum TempRef<'tcx> {
    Lvalue(LvalueRef<'tcx>),
    Operand(Option<OperandRef<'tcx>>),
}

///////////////////////////////////////////////////////////////////////////

pub fn trans_mir<'blk, 'tcx>(fcx: &'blk FunctionContext<'blk, 'tcx>) {
    let bcx = fcx.init(false, None).build();
    let mir = bcx.mir();

    let mir_blocks = mir.all_basic_blocks();

    // Analyze the temps to determine which must be lvalues
    // FIXME
    let lvalue_temps = bcx.with_block(|bcx| {
      analyze::lvalue_temps(bcx, &mir)
    });

    // Allocate variable and temp allocas
    let vars = mir.var_decls.iter()
                            .map(|decl| (bcx.monomorphize(&decl.ty), decl.name))
                            .map(|(mty, name)| LvalueRef::alloca(&bcx, mty, &name.as_str()))
                            .collect();
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
    let args = arg_value_refs(&bcx, &mir);

    // Allocate a `Block` for every basic block
    let block_bcxs: Vec<Block<'blk,'tcx>> =
        mir_blocks.iter()
                  .map(|&bb|{
                      // FIXME(#30941) this doesn't handle msvc-style exceptions
                      fcx.new_block(&format!("{:?}", bb), None)
                  })
                  .collect();

    // Branch to the START block
    let start_bcx = block_bcxs[mir::START_BLOCK.index()];
    bcx.br(start_bcx.llbb);

    let mut mircx = MirContext {
        mir: mir,
        fcx: fcx,
        llpersonalityslot: None,
        blocks: block_bcxs,
        unreachable_block: None,
        vars: vars,
        temps: temps,
        args: args,
    };

    // Translate the body of each block
    for &bb in &mir_blocks {
        mircx.trans_block(bb);
    }

    fcx.cleanup();
}

/// Produce, for each argument, a `ValueRef` pointing at the
/// argument's value. As arguments are lvalues, these are always
/// indirect.
fn arg_value_refs<'bcx, 'tcx>(bcx: &BlockAndBuilder<'bcx, 'tcx>,
                              mir: &mir::Mir<'tcx>)
                              -> Vec<LvalueRef<'tcx>> {
    let fcx = bcx.fcx();
    let tcx = bcx.tcx();
    let mut idx = 0;
    let mut llarg_idx = fcx.fn_ty.ret.is_indirect() as usize;
    mir.arg_decls.iter().enumerate().map(|(arg_index, arg_decl)| {
        let arg_ty = bcx.monomorphize(&arg_decl.ty);
        if arg_decl.spread {
            // This argument (e.g. the last argument in the "rust-call" ABI)
            // is a tuple that was spread at the ABI level and now we have
            // to reconstruct it into a tuple local variable, from multiple
            // individual LLVM function arguments.

            let tupled_arg_tys = match arg_ty.sty {
                ty::TyTuple(ref tys) => tys,
                _ => unreachable!("spread argument isn't a tuple?!")
            };

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
            }
            return LvalueRef::new_sized(lltemp, LvalueTy::from_ty(arg_ty));
        }

        let arg = &fcx.fn_ty.args[idx];
        idx += 1;
        let llval = if arg.is_indirect() {
            // Don't copy an indirect argument to an alloca, the caller
            // already put it in a temporary alloca and gave it up, unless
            // we emit extra-debug-info, which requires local allocas :(.
            // FIXME: lifetimes, debug info
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
