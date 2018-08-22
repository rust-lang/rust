// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains everything needed to instantiate an interpreter.
//! This separation exists to ensure that no fancy miri features like
//! interpreting common C functions leak into CTFE.

use std::hash::Hash;

use rustc::mir::interpret::{AllocId, EvalResult, Scalar, Pointer, AccessKind, GlobalId};
use super::{EvalContext, PlaceTy, OpTy, Memory};

use rustc::mir;
use rustc::ty::{self, layout::TyLayout};
use rustc::ty::layout::Size;
use syntax::source_map::Span;
use syntax::ast::Mutability;

/// Methods of this trait signifies a point where CTFE evaluation would fail
/// and some use case dependent behaviour can instead be applied
pub trait Machine<'mir, 'tcx>: Clone + Eq + Hash {
    /// Additional data that can be accessed via the Memory
    type MemoryData: Clone + Eq + Hash;

    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    type MemoryKinds: ::std::fmt::Debug + PartialEq + Copy + Clone;

    /// Entry point to all function calls.
    ///
    /// Returns Ok(true) when the function was handled completely
    /// e.g. due to missing mir
    ///
    /// Returns Ok(false) if a new stack frame was pushed
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(PlaceTy<'tcx>, mir::BasicBlock)>,
        args: &[OpTy<'tcx>],
        span: Span,
    ) -> EvalResult<'tcx, bool>;

    /// directly process an intrinsic without pushing a stack frame.
    fn call_intrinsic<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx>;

    /// Called for all binary operations except on float types.
    ///
    /// Returns `None` if the operation should be handled by the integer
    /// op code in order to share more code between machines
    ///
    /// Returns a (value, overflowed) pair if the operation succeeded
    fn try_ptr_op<'a>(
        ecx: &EvalContext<'a, 'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: Scalar,
        left_layout: TyLayout<'tcx>,
        right: Scalar,
        right_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, Option<(Scalar, bool)>>;

    /// Called when trying to mark machine defined `MemoryKinds` as static
    fn mark_static_initialized<'a>(
        _mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        _id: AllocId,
        _mutability: Mutability,
    ) -> EvalResult<'tcx, bool>;

    /// Called when requiring a pointer to a static. Non const eval can
    /// create a mutable memory location for `static mut`
    fn init_static<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        cid: GlobalId<'tcx>,
    ) -> EvalResult<'tcx, AllocId>;

    /// Heap allocations via the `box` keyword
    ///
    /// Returns a pointer to the allocated memory
    fn box_alloc<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx>;

    /// Called when trying to access a global declared with a `linkage` attribute
    fn global_item_with_linkage<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        mutability: Mutability,
    ) -> EvalResult<'tcx>;

    fn check_locks<'a>(
        _mem: &Memory<'a, 'mir, 'tcx, Self>,
        _ptr: Pointer,
        _size: Size,
        _access: AccessKind,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    fn add_lock<'a>(
        _mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        _id: AllocId,
    ) {}

    fn free_lock<'a>(
        _mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        _id: AllocId,
        _len: u64,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    fn end_region<'a>(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _reg: Option<::rustc::middle::region::Scope>,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    fn validation_op<'a>(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _op: ::rustc::mir::ValidationOp,
        _operand: &::rustc::mir::ValidationOperand<'tcx, ::rustc::mir::Place<'tcx>>,
    ) -> EvalResult<'tcx> {
        Ok(())
    }
}
