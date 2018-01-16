// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Lints statically known runtime failures

use rustc::mir::*;
use rustc::mir::visit::Visitor;
use rustc::mir::interpret::{Value, PrimVal};
use rustc::middle::const_val::{ConstVal, ConstEvalErr, ErrKind};
use rustc::traits;
use interpret::{eval_body_as_integer, check_body};
use rustc::ty::{TyCtxt, ParamEnv, self};
use rustc::ty::Instance;
use rustc::ty::layout::LayoutOf;
use rustc::hir::def_id::DefId;

pub fn check<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    if tcx.is_closure(def_id) {
        return;
    }
    let generics = tcx.generics_of(def_id);
    // FIXME: miri should be able to eval stuff that doesn't need info
    // from the generics
    if generics.parent_types as usize + generics.types.len() > 0 {
        return;
    }
    let mir = &tcx.optimized_mir(def_id);
    ConstErrVisitor {
        tcx,
        def_id,
        mir,
    }.visit_mir(mir);
    let param_env = ParamEnv::empty(traits::Reveal::All);
    let instance = Instance::mono(tcx, def_id);
    for i in 0.. mir.promoted.len() {
        use rustc_data_structures::indexed_vec::Idx;
        check_body(tcx, instance, Some(Promoted::new(i)), param_env);
    }
}

struct ConstErrVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    mir: &'a Mir<'tcx>,
}

impl<'a, 'tcx> ConstErrVisitor<'a, 'tcx> {
    fn eval_op(&self, op: &Operand<'tcx>) -> Option<u128> {
        let op = match *op {
            Operand::Constant(ref c) => c,
            _ => return None,
        };
        let param_env = ParamEnv::empty(traits::Reveal::All);
        let val = match op.literal {
            Literal::Value { value } => match value.val {
                ConstVal::Value(Value::ByVal(PrimVal::Bytes(b))) => b,
                _ => return None,
            },
            Literal::Promoted { index } => {
                let instance = Instance::mono(self.tcx, self.def_id);
                eval_body_as_integer(self.tcx, param_env, instance, Some(index)).unwrap()
            }
        };
        Some(val)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ConstErrVisitor<'a, 'tcx> {
    fn visit_terminator(&mut self,
                        block: BasicBlock,
                        terminator: &Terminator<'tcx>,
                        location: Location) {
        self.super_terminator(block, terminator, location);
        match terminator.kind {
            TerminatorKind::Assert { cond: Operand::Constant(box Constant {
                literal: Literal::Value {
                    value: &ty::Const {
                        val: ConstVal::Value(Value::ByVal(PrimVal::Bytes(cond))),
                    .. }
                }, ..
            }), expected, ref msg, .. } if (cond == 1) != expected => {
                assert!(cond <= 1);
                // If we know we always panic, and the error message
                // is also constant, then we can produce a warning.

                let kind = match *msg {
                    AssertMessage::BoundsCheck { ref len, ref index } => {
                        let len = match self.eval_op(len) {
                            Some(val) => val,
                            None => return,
                        };
                        let index = match self.eval_op(index) {
                            Some(val) => val,
                            None => return,
                        };
                        ErrKind::IndexOutOfBounds {
                            len: len as u64,
                            index: index as u64
                        }
                    }
                    AssertMessage::Math(ref err) => ErrKind::Math(err.clone()),
                    AssertMessage::GeneratorResumedAfterReturn |
                    // FIXME(oli-obk): can we report a const_err warning here?
                    AssertMessage::GeneratorResumedAfterPanic => return,
                };
                let span = terminator.source_info.span;
                let msg = ConstEvalErr{ span, kind };
                let scope_info = match self.mir.visibility_scope_info {
                    ClearCrossCrate::Set(ref data) => data,
                    ClearCrossCrate::Clear => return,
                };
                let node_id = scope_info[terminator.source_info.scope].lint_root;
                self.tcx.lint_node(::rustc::lint::builtin::CONST_ERR,
                            node_id,
                            msg.span,
                            &msg.description().into_oneline().into_owned());
            },
            _ => {},
        }
    }
    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        self.super_rvalue(rvalue, location);
        use rustc::mir::BinOp;
        match *rvalue {
            Rvalue::BinaryOp(BinOp::Shr, ref lop, ref rop) |
            Rvalue::BinaryOp(BinOp::Shl, ref lop, ref rop) => {
                let val = match self.eval_op(rop) {
                    Some(val) => val,
                    None => return,
                };
                let ty = lop.ty(self.mir, self.tcx);
                let param_env = ParamEnv::empty(traits::Reveal::All);
                let bits = (self.tcx, param_env).layout_of(ty).unwrap().size.bits();
                if val >= bits as u128 {
                    let data = &self.mir[location.block];
                    let stmt_idx = location.statement_index;
                    let source_info = if stmt_idx < data.statements.len() {
                        data.statements[stmt_idx].source_info
                    } else {
                        data.terminator().source_info
                    };
                    let span = source_info.span;
                    let scope_info = match self.mir.visibility_scope_info {
                        ClearCrossCrate::Set(ref data) => data,
                        ClearCrossCrate::Clear => return,
                    };
                    let node_id = scope_info[source_info.scope].lint_root;
                    self.tcx.lint_node(
                        ::rustc::lint::builtin::EXCEEDING_BITSHIFTS,
                        node_id,
                        span,
                        "bitshift exceeds the type's number of bits");
                }
            }
            _ => {}
        }
    }
}
