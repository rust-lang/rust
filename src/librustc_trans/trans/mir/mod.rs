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
use rustc_mir::repr as mir;
use rustc_mir::tcx::LvalueTy;
use trans::base;
use trans::build;
use trans::common::{self, Block};
use trans::debuginfo::DebugLoc;
use trans::expr;
use trans::type_of;

use self::lvalue::LvalueRef;
use self::operand::OperandRef;

// FIXME DebugLoc is always None right now

/// Master context for translating MIR.
pub struct MirContext<'bcx, 'tcx:'bcx> {
    mir: &'bcx mir::Mir<'tcx>,

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

pub fn trans_mir<'bcx, 'tcx>(bcx: Block<'bcx, 'tcx>) {
    let fcx = bcx.fcx;
    let mir = bcx.mir();

    let mir_blocks = bcx.mir().all_basic_blocks();

    // Analyze the temps to determine which must be lvalues
    // FIXME
    let lvalue_temps = analyze::lvalue_temps(bcx, mir);

    // Allocate variable and temp allocas
    let vars = mir.var_decls.iter()
                            .map(|decl| (bcx.monomorphize(&decl.ty), decl.name))
                            .map(|(mty, name)| LvalueRef::alloca(bcx, mty, &name.as_str()))
                            .collect();
    let temps = mir.temp_decls.iter()
                              .map(|decl| bcx.monomorphize(&decl.ty))
                              .enumerate()
                              .map(|(i, mty)| if lvalue_temps.contains(&i) {
                                  TempRef::Lvalue(LvalueRef::alloca(bcx,
                                                                    mty,
                                                                    &format!("temp{:?}", i)))
                              } else {
                                  // If this is an immediate temp, we do not create an
                                  // alloca in advance. Instead we wait until we see the
                                  // definition and update the operand there.
                                  TempRef::Operand(None)
                              })
                              .collect();
    let args = arg_value_refs(bcx, mir);

    // Allocate a `Block` for every basic block
    let block_bcxs: Vec<Block<'bcx,'tcx>> =
        mir_blocks.iter()
                  .map(|&bb| fcx.new_block(false, &format!("{:?}", bb), None))
                  .collect();

    // Branch to the START block
    let start_bcx = block_bcxs[mir::START_BLOCK.index()];
    build::Br(bcx, start_bcx.llbb, DebugLoc::None);

    let mut mircx = MirContext {
        mir: mir,
        llpersonalityslot: None,
        blocks: block_bcxs,
        vars: vars,
        temps: temps,
        args: args,
    };

    // Translate the body of each block
    for &bb in &mir_blocks {
        if bb != mir::DIVERGE_BLOCK {
            mircx.trans_block(bb);
        }
    }

    // Total hack: translate DIVERGE_BLOCK last. This is so that any
    // panics which the fn may do can initialize the
    // `llpersonalityslot` cell. We don't do this up front because the
    // LLVM type of it is (frankly) annoying to compute.
    mircx.trans_block(mir::DIVERGE_BLOCK);
}

/// Produce, for each argument, a `ValueRef` pointing at the
/// argument's value. As arguments are lvalues, these are always
/// indirect.
fn arg_value_refs<'bcx, 'tcx>(bcx: Block<'bcx, 'tcx>,
                              mir: &mir::Mir<'tcx>)
                              -> Vec<LvalueRef<'tcx>> {
    // FIXME tupled_args? I think I'd rather that mapping is done in MIR land though
    let fcx = bcx.fcx;
    let tcx = bcx.tcx();
    let mut idx = fcx.arg_offset() as c_uint;
    mir.arg_decls
       .iter()
       .enumerate()
       .map(|(arg_index, arg_decl)| {
           let arg_ty = bcx.monomorphize(&arg_decl.ty);
           let llval = if type_of::arg_is_indirect(bcx.ccx(), arg_ty) {
               // Don't copy an indirect argument to an alloca, the caller
               // already put it in a temporary alloca and gave it up, unless
               // we emit extra-debug-info, which requires local allocas :(.
               // FIXME: lifetimes, debug info
               let llarg = llvm::get_param(fcx.llfn, idx);
               idx += 1;
               llarg
           } else if common::type_is_fat_ptr(tcx, arg_ty) {
               // we pass fat pointers as two words, but we want to
               // represent them internally as a pointer two two words,
               // so make an alloca to store them in.
               let lldata = llvm::get_param(fcx.llfn, idx);
               let llextra = llvm::get_param(fcx.llfn, idx + 1);
               idx += 2;
               let lltemp = base::alloc_ty(bcx, arg_ty, &format!("arg{}", arg_index));
               build::Store(bcx, lldata, expr::get_dataptr(bcx, lltemp));
               build::Store(bcx, llextra, expr::get_dataptr(bcx, lltemp));
               lltemp
           } else {
               // otherwise, arg is passed by value, so make a
               // temporary and store it there
               let llarg = llvm::get_param(fcx.llfn, idx);
               idx += 1;
               let lltemp = base::alloc_ty(bcx, arg_ty, &format!("arg{}", arg_index));
               build::Store(bcx, llarg, lltemp);
               lltemp
           };
           LvalueRef::new(llval, LvalueTy::from_ty(arg_ty))
       })
       .collect()
}

mod analyze;
mod block;
mod constant;
mod lvalue;
mod rvalue;
mod operand;
mod statement;

