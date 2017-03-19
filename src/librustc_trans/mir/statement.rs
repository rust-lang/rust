// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::OperandBundleDef;
use rustc::mir;
use rustc::ty;
use rustc_const_eval::{ErrKind, ConstEvalErr, note_const_eval_err};
use rustc::middle::lang_items;

use base;
use asm;
use common::{self, C_bool, C_str_slice, C_u32, C_struct};
use builder::Builder;
use syntax::symbol::Symbol;
use machine::llalign_of_min;
use consts;
use callee;

use super::MirContext;
use super::LocalRef;
use super::super::adt;
use super::super::disr::Disr;

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_statement(&mut self,
                           bcx: Builder<'a, 'tcx>,
                           statement: &mir::Statement<'tcx>)
                           -> Builder<'a, 'tcx> {
        debug!("trans_statement(statement={:?})", statement);

        self.set_debug_loc(&bcx, statement.source_info);
        match statement.kind {
            mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                if let mir::Lvalue::Local(index) = *lvalue {
                    match self.locals[index] {
                        LocalRef::Lvalue(tr_dest) => {
                            self.trans_rvalue(bcx, tr_dest, rvalue)
                        }
                        LocalRef::Operand(None) => {
                            let (bcx, operand) = self.trans_rvalue_operand(bcx, rvalue);
                            self.locals[index] = LocalRef::Operand(Some(operand));
                            bcx
                        }
                        LocalRef::Operand(Some(_)) => {
                            let ty = self.monomorphized_lvalue_ty(lvalue);

                            if !common::type_is_zero_size(bcx.ccx, ty) {
                                span_bug!(statement.source_info.span,
                                          "operand {:?} already assigned",
                                          rvalue);
                            } else {
                                // If the type is zero-sized, it's already been set here,
                                // but we still need to make sure we translate the operand
                                self.trans_rvalue_operand(bcx, rvalue).0
                            }
                        }
                    }
                } else {
                    let tr_dest = self.trans_lvalue(&bcx, lvalue);
                    self.trans_rvalue(bcx, tr_dest, rvalue)
                }
            }
            mir::StatementKind::SetDiscriminant{ref lvalue, variant_index} => {
                let ty = self.monomorphized_lvalue_ty(lvalue);
                let lvalue_transed = self.trans_lvalue(&bcx, lvalue);
                adt::trans_set_discr(&bcx,
                    ty,
                    lvalue_transed.llval,
                    Disr::from(variant_index));
                bcx
            }
            mir::StatementKind::StorageLive(ref lvalue) => {
                self.trans_storage_liveness(bcx, lvalue, base::Lifetime::Start)
            }
            mir::StatementKind::StorageDead(ref lvalue) => {
                self.trans_storage_liveness(bcx, lvalue, base::Lifetime::End)
            }
            mir::StatementKind::InlineAsm { ref asm, ref outputs, ref inputs } => {
                let outputs = outputs.iter().map(|output| {
                    let lvalue = self.trans_lvalue(&bcx, output);
                    (lvalue.llval, lvalue.ty.to_ty(bcx.tcx()))
                }).collect();

                let input_vals = inputs.iter().map(|input| {
                    self.trans_operand(&bcx, input).immediate()
                }).collect();

                asm::trans_inline_asm(&bcx, asm, outputs, input_vals);
                bcx
            }
            mir::StatementKind::Nop => bcx,
        }
    }

    fn trans_storage_liveness(&self,
                              bcx: Builder<'a, 'tcx>,
                              lvalue: &mir::Lvalue<'tcx>,
                              intrinsic: base::Lifetime)
                              -> Builder<'a, 'tcx> {
        if let mir::Lvalue::Local(index) = *lvalue {
            if let LocalRef::Lvalue(tr_lval) = self.locals[index] {
                intrinsic.call(&bcx, tr_lval.llval);
            }
        }
        bcx
    }

    pub fn trans_assert(
        &mut self,
        mut bcx: Builder<'a, 'tcx>,
        cond: &mir::Operand<'tcx>,
        expected: bool,
        msg: &mir::AssertMessage<'tcx>,
        cleanup: Option<mir::Block>,
        cleanup_bundle: Option<&OperandBundleDef>,
        source_info: mir::SourceInfo,
    ) -> Builder<'a, 'tcx> {
        let cond = self.trans_operand(&bcx, cond).immediate();
        let mut const_cond = common::const_to_opt_u128(cond, false).map(|c| c == 1);

        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        // NOTE: Unlike binops, negation doesn't have its own
        // checked operation, just a comparison with the minimum
        // value, so we have to check for the assert message.
        if !bcx.ccx.check_overflow() {
            use rustc_const_math::ConstMathErr::Overflow;
            use rustc_const_math::Op::Neg;

            if let mir::AssertMessage::Math(Overflow(Neg)) = *msg {
                const_cond = Some(expected);
            }
        }

        // Don't translate the panic block if success if known.
        if const_cond == Some(expected) {
            return bcx;
        }

        // Pass the condition through llvm.expect for branch hinting.
        let expect = bcx.ccx.get_intrinsic(&"llvm.expect.i1");
        let cond = bcx.call(expect, &[cond, C_bool(bcx.ccx, expected)], None);

        // Create the failure block and the conditional branch to it.
        let success_block = self.new_block("success");
        let panic_block = self.new_block("panic");
        if expected {
            bcx.cond_br(cond, success_block.llbb(), panic_block.llbb());
        } else {
            bcx.cond_br(cond, panic_block.llbb(), success_block.llbb());
        }

        // After this point, bcx is the block for the call to panic.
        bcx = panic_block;
        self.set_debug_loc(&bcx, source_info);

        // Get the location information.
        let loc = bcx.sess().codemap().lookup_char_pos(source_info.span.lo);
        let filename = Symbol::intern(&loc.file.name).as_str();
        let filename = C_str_slice(bcx.ccx, filename);
        let line = C_u32(bcx.ccx, loc.line as u32);

        // Put together the arguments to the panic entry point.
        let (lang_item, args, const_err) = match *msg {
            mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                let len = self.trans_operand(&mut bcx, len).immediate();
                let index = self.trans_operand(&mut bcx, index).immediate();

                let const_err = common::const_to_opt_u128(len, false)
                    .and_then(|len| common::const_to_opt_u128(index, false)
                        .map(|index| ErrKind::IndexOutOfBounds {
                            len: len as u64,
                            index: index as u64
                        }));

                let file_line = C_struct(bcx.ccx, &[filename, line], false);
                let align = llalign_of_min(bcx.ccx, common::val_ty(file_line));
                let file_line = consts::addr_of(bcx.ccx,
                                                file_line,
                                                align,
                                                "panic_bounds_check_loc");
                (lang_items::PanicBoundsCheckFnLangItem,
                    vec![file_line, index, len],
                    const_err)
            }
            mir::AssertMessage::Math(ref err) => {
                let msg_str = Symbol::intern(err.description()).as_str();
                let msg_str = C_str_slice(bcx.ccx, msg_str);
                let msg_file_line = C_struct(bcx.ccx,
                                                &[msg_str, filename, line],
                                                false);
                let align = llalign_of_min(bcx.ccx, common::val_ty(msg_file_line));
                let msg_file_line = consts::addr_of(bcx.ccx,
                                                    msg_file_line,
                                                    align,
                                                    "panic_loc");
                (lang_items::PanicFnLangItem,
                    vec![msg_file_line],
                    Some(ErrKind::Math(err.clone())))
            }
        };

        // If we know we always panic, and the error message
        // is also constant, then we can produce a warning.
        if const_cond == Some(!expected) {
            if let Some(err) = const_err {
                let err = ConstEvalErr { span: source_info.span, kind: err };
                let mut diag = bcx.tcx().sess.struct_span_warn(
                    source_info.span, "this expression will panic at run-time");
                note_const_eval_err(bcx.tcx(), &err, source_info.span, "expression", &mut diag);
                diag.emit();
            }
        }

        // Obtain the panic entry point.
        let def_id = common::langcall(bcx.tcx(), Some(source_info.span), "", lang_item);
        let instance = ty::Instance::mono(bcx.tcx(), def_id);
        let llfn = callee::get_fn(bcx.ccx, instance);

        // Translate the actual panic invoke/call.
        if let Some(unwind) = cleanup {
            bcx.invoke(llfn,
                        &args,
                        self.unreachable_block(),
                        self.landing_pad_to(unwind),
                        cleanup_bundle);
        } else {
            bcx.call(llfn, &args, cleanup_bundle);
            bcx.unreachable();
        }

        success_block
    }
}
