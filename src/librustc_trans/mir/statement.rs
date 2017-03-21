// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef, OperandBundleDef};
use rustc::mir;
use rustc::ty::{self, layout, TypeFoldable};
use rustc_const_eval::{ErrKind, ConstEvalErr, note_const_eval_err};
use rustc::middle::lang_items;

use base;
use asm;
use common::{self, C_bool, C_str_slice, C_u32, C_struct, C_undef, C_uint};
use builder::Builder;
use syntax::symbol::Symbol;
use machine::llalign_of_min;
use consts;
use callee;
use monomorphize;
use meth;
use tvec;
use abi::{Abi, FnType, ArgType};
use type_::Type;
use type_of::{self, align_of};

use std::cmp;

use super::MirContext;
use super::LocalRef;
use super::super::adt;
use super::super::disr::Disr;
use super::lvalue::{Alignment, LvalueRef};
use super::operand::OperandRef;
use super::operand::OperandValue::{Pair, Ref, Immediate};

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_statement(&mut self,
                           bcx: Builder<'a, 'tcx>,
                           statement: &mir::Statement<'tcx>,
                           cleanup_bundle: Option<&OperandBundleDef>)
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
            mir::StatementKind::Assert { ref cond, expected, ref msg, cleanup } => {
                self.trans_assert(
                    bcx, cond, expected, msg, cleanup, cleanup_bundle, statement.source_info
                )
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
            let old_bcx = bcx;
            bcx = old_bcx.build_sibling_block("assert-next");
            old_bcx.invoke(
                llfn,
                &args,
                bcx.llbb(),
                self.landing_pad_to(unwind),
                cleanup_bundle
            );
        } else {
            bcx.call(llfn, &args, cleanup_bundle);
        }
        bcx.unreachable();

        success_block
    }

    pub fn trans_drop(
        &mut self,
        mut bcx: Builder<'a, 'tcx>,
        location: &mir::Lvalue<'tcx>,
        unwind: Option<mir::Block>,
        cleanup_bundle: Option<&OperandBundleDef>,
        source_info: mir::SourceInfo,
    ) -> Builder<'a, 'tcx> {
        let ty = location.ty(&self.mir, bcx.tcx()).to_ty(bcx.tcx());
        let ty = self.monomorphize(&ty);
        let drop_fn = monomorphize::resolve_drop_in_place(bcx.ccx.shared(), ty);

        if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
            // we don't actually need to drop anything.
            return bcx;
        }

        let lvalue = self.trans_lvalue(&bcx, location);
        let (drop_fn, need_extra) = match ty.sty {
            ty::TyDynamic(..) => (meth::DESTRUCTOR.get_fn(&bcx, lvalue.llextra),
                                    false),
            ty::TyArray(ety, _) | ty::TySlice(ety) => {
                // FIXME: handle panics
                let drop_fn = monomorphize::resolve_drop_in_place(
                    bcx.ccx.shared(), ety);
                let drop_fn = callee::get_fn(bcx.ccx, drop_fn);
                return tvec::slice_for_each(
                    &bcx,
                    lvalue.project_index(&bcx, C_uint(bcx.ccx, 0u64)),
                    ety,
                    lvalue.len(bcx.ccx),
                    |mut bcx, llval, loop_bb| {
                        self.set_debug_loc(&bcx, source_info);
                        if let Some(unwind) = unwind {
                            let old_bcx = bcx;
                            bcx = old_bcx.build_sibling_block("drop-next");
                            old_bcx.invoke(
                                drop_fn,
                                &[llval],
                                bcx.llbb(),
                                self.landing_pad_to(unwind),
                                cleanup_bundle
                            );
                        } else {
                            bcx.call(drop_fn, &[llval], cleanup_bundle);
                        }
                        bcx.br(loop_bb);
                        bcx
                    });
            }
            _ => (callee::get_fn(bcx.ccx, drop_fn), lvalue.has_extra())
        };
        let args = &[lvalue.llval, lvalue.llextra][..1 + need_extra as usize];
        if let Some(unwind) = unwind {
            let old_bcx = bcx;
            bcx = old_bcx.build_sibling_block("drop-next");
            old_bcx.invoke(
                drop_fn,
                args,
                bcx.llbb(),
                self.landing_pad_to(unwind),
                cleanup_bundle
            );
        } else {
            bcx.call(drop_fn, args, cleanup_bundle);
        }

        bcx
    }

    pub fn trans_call(
        &mut self,
        mut bcx: Builder<'a, 'tcx>,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        destination: &Option<(mir::Lvalue<'tcx>, mir::Block)>,
        cleanup: &Option<mir::Block>,
        cleanup_bundle: Option<&OperandBundleDef>,
        source_info: mir::SourceInfo,
    ) -> Builder<'a, 'tcx> {
        // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
        let callee = self.trans_operand(&bcx, func);

        let (instance, mut llfn, sig) = match callee.ty.sty {
            ty::TyFnDef(def_id, substs, sig) => {
                (Some(monomorphize::resolve(bcx.ccx.shared(), def_id, substs)),
                    None,
                    sig)
            }
            ty::TyFnPtr(sig) => {
                (None,
                    Some(callee.immediate()),
                    sig)
            }
            _ => bug!("{} is not callable", callee.ty)
        };
        let def = instance.map(|i| i.def);
        let sig = bcx.tcx().erase_late_bound_regions_and_normalize(&sig);
        let abi = sig.abi;

        // Handle intrinsics old trans wants Expr's for, ourselves.
        let intrinsic = match def {
            Some(ty::InstanceDef::Intrinsic(def_id))
                => Some(bcx.tcx().item_name(def_id).as_str()),
            _ => None
        };
        let intrinsic = intrinsic.as_ref().map(|s| &s[..]);

        if intrinsic == Some("move_val_init") {
            // The first argument is a thin destination pointer.
            let llptr = self.trans_operand(&bcx, &args[0]).immediate();
            let val = self.trans_operand(&bcx, &args[1]);
            self.store_operand(&bcx, llptr, None, val);
            return bcx;
        }

        if intrinsic == Some("transmute") {
            let &(ref dest, _) = destination.as_ref().unwrap();
            self.trans_transmute(&bcx, &args[0], dest);
            return bcx;
        }

        let extra_args = &args[sig.inputs().len()..];
        let extra_args = extra_args.iter().map(|op_arg| {
            let op_ty = op_arg.ty(&self.mir, bcx.tcx());
            self.monomorphize(&op_ty)
        }).collect::<Vec<_>>();

        let fn_ty = match def {
            Some(ty::InstanceDef::Virtual(..)) => {
                FnType::new_vtable(bcx.ccx, sig, &extra_args)
            }
            Some(ty::InstanceDef::DropGlue(_, None)) => {
                // empty drop glue - a nop.
                return bcx;
            }
            _ => FnType::new(bcx.ccx, sig, &extra_args)
        };

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_ty.args.len() + fn_ty.ret.is_indirect() as usize;
        let mut llargs = Vec::with_capacity(arg_count);

        // Prepare the return value destination
        let ret_dest = if let Some((ref dest, _)) = *destination {
            let is_intrinsic = intrinsic.is_some();
            self.make_return_dest(&bcx, dest, &fn_ty.ret, &mut llargs,
                                    is_intrinsic)
        } else {
            ReturnDest::Nothing
        };

        // Split the rust-call tupled arguments off.
        let (first_args, untuple) = if abi == Abi::RustCall && !args.is_empty() {
            let (tup, args) = args.split_last().unwrap();
            (args, Some(tup))
        } else {
            (&args[..], None)
        };

        let is_shuffle = intrinsic.map_or(false, |name| {
            name.starts_with("simd_shuffle")
        });
        let mut idx = 0;
        for arg in first_args {
            // The indices passed to simd_shuffle* in the
            // third argument must be constant. This is
            // checked by const-qualification, which also
            // promotes any complex rvalues to constants.
            if is_shuffle && idx == 2 {
                match *arg {
                    mir::Operand::Consume(_) => {
                        span_bug!(source_info.span, "shuffle indices must be constant");
                    }
                    mir::Operand::Constant(ref constant) => {
                        let val = self.trans_constant(&bcx, constant);
                        llargs.push(val.llval);
                        idx += 1;
                        continue;
                    }
                }
            }

            let op = self.trans_operand(&bcx, arg);
            self.trans_argument(&bcx, op, &mut llargs, &fn_ty,
                                &mut idx, &mut llfn, &def);
        }
        if let Some(tup) = untuple {
            self.trans_arguments_untupled(&bcx, tup, &mut llargs, &fn_ty,
                                            &mut idx, &mut llfn, &def)
        }

        if intrinsic.is_some() && intrinsic != Some("drop_in_place") {
            use intrinsic::trans_intrinsic_call;

            let (dest, llargs) = match ret_dest {
                _ if fn_ty.ret.is_indirect() => {
                    (llargs[0], &llargs[1..])
                }
                ReturnDest::Nothing => {
                    (C_undef(fn_ty.ret.original_ty.ptr_to()), &llargs[..])
                }
                ReturnDest::IndirectOperand(dst, _) |
                ReturnDest::Store(dst) => (dst, &llargs[..]),
                ReturnDest::DirectOperand(_) =>
                    bug!("Cannot use direct operand with an intrinsic call")
            };

            let callee_ty = common::instance_ty(
                bcx.ccx.shared(), instance.as_ref().unwrap());
            trans_intrinsic_call(&bcx, callee_ty, &fn_ty, &llargs, dest,
                                    source_info.span);

            if let ReturnDest::IndirectOperand(dst, _) = ret_dest {
                // Make a fake operand for store_return
                let op = OperandRef {
                    val: Ref(dst, Alignment::AbiAligned),
                    ty: sig.output(),
                };
                self.store_return(&bcx, ret_dest, fn_ty.ret, op);
            }

            return bcx;
        }

        let fn_ptr = match (llfn, instance) {
            (Some(llfn), _) => llfn,
            (None, Some(instance)) => callee::get_fn(bcx.ccx, instance),
            _ => span_bug!(source_info.span, "no llfn for call"),
        };

        let llret = if let &Some(cleanup) = cleanup {
            let old_bcx = bcx;
            bcx = old_bcx.build_sibling_block("call-next");
            self.set_debug_loc(&bcx, source_info);
            old_bcx.invoke(
                fn_ptr,
                &llargs,
                bcx.llbb(),
                self.landing_pad_to(cleanup),
                cleanup_bundle,
            )
        } else {
            bcx.call(fn_ptr, &llargs, cleanup_bundle)
        };

        fn_ty.apply_attrs_callsite(llret);

        let op = OperandRef {
            val: Immediate(llret),
            ty: sig.output(),
        };
        self.store_return(&bcx, ret_dest, fn_ty.ret, op);

        bcx
    }

    fn trans_argument(&mut self,
                      bcx: &Builder<'a, 'tcx>,
                      op: OperandRef<'tcx>,
                      llargs: &mut Vec<ValueRef>,
                      fn_ty: &FnType,
                      next_idx: &mut usize,
                      llfn: &mut Option<ValueRef>,
                      def: &Option<ty::InstanceDef<'tcx>>) {
        if let Pair(a, b) = op.val {
            // Treat the values in a fat pointer separately.
            if common::type_is_fat_ptr(bcx.ccx, op.ty) {
                let (ptr, meta) = (a, b);
                if *next_idx == 0 {
                    if let Some(ty::InstanceDef::Virtual(_, idx)) = *def {
                        let llmeth = meth::VirtualIndex::from_index(idx).get_fn(bcx, meta);
                        let llty = fn_ty.llvm_type(bcx.ccx).ptr_to();
                        *llfn = Some(bcx.pointercast(llmeth, llty));
                    }
                }

                let imm_op = |x| OperandRef {
                    val: Immediate(x),
                    // We won't be checking the type again.
                    ty: bcx.tcx().types.err
                };
                self.trans_argument(bcx, imm_op(ptr), llargs, fn_ty, next_idx, llfn, def);
                self.trans_argument(bcx, imm_op(meta), llargs, fn_ty, next_idx, llfn, def);
                return;
            }
        }

        let arg = &fn_ty.args[*next_idx];
        *next_idx += 1;

        // Fill padding with undef value, where applicable.
        if let Some(ty) = arg.pad {
            llargs.push(C_undef(ty));
        }

        if arg.is_ignore() {
            return;
        }

        // Force by-ref if we have to load through a cast pointer.
        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => {
                if arg.is_indirect() || arg.cast.is_some() {
                    let llscratch = bcx.alloca(arg.original_ty, "arg");
                    self.store_operand(bcx, llscratch, None, op);
                    (llscratch, Alignment::AbiAligned, true)
                } else {
                    (op.pack_if_pair(bcx).immediate(), Alignment::AbiAligned, false)
                }
            }
            Ref(llval, Alignment::Packed) if arg.is_indirect() => {
                // `foo(packed.large_field)`. We can't pass the (unaligned) field directly. I
                // think that ATM (Rust 1.16) we only pass temporaries, but we shouldn't
                // have scary latent bugs around.

                let llscratch = bcx.alloca(arg.original_ty, "arg");
                base::memcpy_ty(bcx, llscratch, llval, op.ty, Some(1));
                (llscratch, Alignment::AbiAligned, true)
            }
            Ref(llval, align) => (llval, align, true)
        };

        if by_ref && !arg.is_indirect() {
            // Have to load the argument, maybe while casting it.
            if arg.original_ty == Type::i1(bcx.ccx) {
                // We store bools as i8 so we need to truncate to i1.
                llval = bcx.load_range_assert(llval, 0, 2, llvm::False, None);
                llval = bcx.trunc(llval, arg.original_ty);
            } else if let Some(ty) = arg.cast {
                llval = bcx.load(bcx.pointercast(llval, ty.ptr_to()),
                                 align.min_with(llalign_of_min(bcx.ccx, arg.ty)));
            } else {
                llval = bcx.load(llval, align.to_align());
            }
        }

        llargs.push(llval);
    }

    fn trans_arguments_untupled(&mut self,
                                bcx: &Builder<'a, 'tcx>,
                                operand: &mir::Operand<'tcx>,
                                llargs: &mut Vec<ValueRef>,
                                fn_ty: &FnType,
                                next_idx: &mut usize,
                                llfn: &mut Option<ValueRef>,
                                def: &Option<ty::InstanceDef<'tcx>>) {
        let tuple = self.trans_operand(bcx, operand);

        let arg_types = match tuple.ty.sty {
            ty::TyTuple(ref tys, _) => tys,
            _ => span_bug!(self.mir.span,
                           "bad final argument to \"rust-call\" fn {:?}", tuple.ty)
        };

        // Handle both by-ref and immediate tuples.
        match tuple.val {
            Ref(llval, align) => {
                for (n, &ty) in arg_types.iter().enumerate() {
                    let ptr = LvalueRef::new_sized_ty(llval, tuple.ty, align);
                    let (ptr, align) = ptr.trans_field_ptr(bcx, n);
                    let val = if common::type_is_fat_ptr(bcx.ccx, ty) {
                        let (lldata, llextra) = base::load_fat_ptr(bcx, ptr, align, ty);
                        Pair(lldata, llextra)
                    } else {
                        // trans_argument will load this if it needs to
                        Ref(ptr, align)
                    };
                    let op = OperandRef {
                        val: val,
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, llfn, def);
                }

            }
            Immediate(llval) => {
                let l = bcx.ccx.layout_of(tuple.ty);
                let v = if let layout::Univariant { ref variant, .. } = *l {
                    variant
                } else {
                    bug!("Not a tuple.");
                };
                for (n, &ty) in arg_types.iter().enumerate() {
                    let mut elem = bcx.extract_value(llval, v.memory_index[n] as usize);
                    // Truncate bools to i1, if needed
                    if ty.is_bool() && common::val_ty(elem) != Type::i1(bcx.ccx) {
                        elem = bcx.trunc(elem, Type::i1(bcx.ccx));
                    }
                    // If the tuple is immediate, the elements are as well
                    let op = OperandRef {
                        val: Immediate(elem),
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, llfn, def);
                }
            }
            Pair(a, b) => {
                let elems = [a, b];
                for (n, &ty) in arg_types.iter().enumerate() {
                    let mut elem = elems[n];
                    // Truncate bools to i1, if needed
                    if ty.is_bool() && common::val_ty(elem) != Type::i1(bcx.ccx) {
                        elem = bcx.trunc(elem, Type::i1(bcx.ccx));
                    }
                    // Pair is always made up of immediates
                    let op = OperandRef {
                        val: Immediate(elem),
                        ty: ty
                    };
                    self.trans_argument(bcx, op, llargs, fn_ty, next_idx, llfn, def);
                }
            }
        }
    }

    fn make_return_dest(&mut self, bcx: &Builder<'a, 'tcx>,
                        dest: &mir::Lvalue<'tcx>, fn_ret_ty: &ArgType,
                        llargs: &mut Vec<ValueRef>, is_intrinsic: bool) -> ReturnDest {
        // If the return is ignored, we can just return a do-nothing ReturnDest
        if fn_ret_ty.is_ignore() {
            return ReturnDest::Nothing;
        }
        let dest = if let mir::Lvalue::Local(index) = *dest {
            let ret_ty = self.monomorphized_lvalue_ty(dest);
            match self.locals[index] {
                LocalRef::Lvalue(dest) => dest,
                LocalRef::Operand(None) => {
                    // Handle temporary lvalues, specifically Operand ones, as
                    // they don't have allocas
                    return if fn_ret_ty.is_indirect() {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = LvalueRef::alloca(bcx, ret_ty, "tmp_ret");
                        llargs.push(tmp.llval);
                        ReturnDest::IndirectOperand(tmp.llval, index)
                    } else if is_intrinsic {
                        // Currently, intrinsics always need a location to store
                        // the result. so we create a temporary alloca for the
                        // result
                        let tmp = LvalueRef::alloca(bcx, ret_ty, "tmp_ret");
                        ReturnDest::IndirectOperand(tmp.llval, index)
                    } else {
                        ReturnDest::DirectOperand(index)
                    };
                }
                LocalRef::Operand(Some(_)) => {
                    bug!("lvalue local already assigned to");
                }
            }
        } else {
            self.trans_lvalue(bcx, dest)
        };
        if fn_ret_ty.is_indirect() {
            match dest.alignment {
                Alignment::AbiAligned => {
                    llargs.push(dest.llval);
                    ReturnDest::Nothing
                },
                Alignment::Packed => {
                    // Currently, MIR code generation does not create calls
                    // that store directly to fields of packed structs (in
                    // fact, the calls it creates write only to temps),
                    //
                    // If someone changes that, please update this code path
                    // to create a temporary.
                    span_bug!(self.mir.span, "can't directly store to unaligned value");
                }
            }
        } else {
            ReturnDest::Store(dest.llval)
        }
    }

    fn trans_transmute(&mut self, bcx: &Builder<'a, 'tcx>,
                       src: &mir::Operand<'tcx>,
                       dst: &mir::Lvalue<'tcx>) {
        if let mir::Lvalue::Local(index) = *dst {
            match self.locals[index] {
                LocalRef::Lvalue(lvalue) => self.trans_transmute_into(bcx, src, &lvalue),
                LocalRef::Operand(None) => {
                    let lvalue_ty = self.monomorphized_lvalue_ty(dst);
                    assert!(!lvalue_ty.has_erasable_regions());
                    let lvalue = LvalueRef::alloca(bcx, lvalue_ty, "transmute_temp");
                    self.trans_transmute_into(bcx, src, &lvalue);
                    let op = self.trans_load(bcx, lvalue.llval, lvalue.alignment, lvalue_ty);
                    self.locals[index] = LocalRef::Operand(Some(op));
                }
                LocalRef::Operand(Some(_)) => {
                    let ty = self.monomorphized_lvalue_ty(dst);
                    assert!(common::type_is_zero_size(bcx.ccx, ty),
                            "assigning to initialized SSAtemp");
                }
            }
        } else {
            let dst = self.trans_lvalue(bcx, dst);
            self.trans_transmute_into(bcx, src, &dst);
        }
    }

    fn trans_transmute_into(&mut self, bcx: &Builder<'a, 'tcx>,
                            src: &mir::Operand<'tcx>,
                            dst: &LvalueRef<'tcx>) {
        let val = self.trans_operand(bcx, src);
        let llty = type_of::type_of(bcx.ccx, val.ty);
        let cast_ptr = bcx.pointercast(dst.llval, llty.ptr_to());
        let in_type = val.ty;
        let out_type = dst.ty.to_ty(bcx.tcx());;
        let llalign = cmp::min(align_of(bcx.ccx, in_type), align_of(bcx.ccx, out_type));
        self.store_operand(bcx, cast_ptr, Some(llalign), val);
    }


    // Stores the return value of a function call into it's final location.
    fn store_return(&mut self,
                    bcx: &Builder<'a, 'tcx>,
                    dest: ReturnDest,
                    ret_ty: ArgType,
                    op: OperandRef<'tcx>) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => ret_ty.store(bcx, op.immediate(), dst),
            IndirectOperand(tmp, index) => {
                let op = self.trans_load(bcx, tmp, Alignment::AbiAligned, op.ty);
                self.locals[index] = LocalRef::Operand(Some(op));
            }
            DirectOperand(index) => {
                // If there is a cast, we have to store and reload.
                let op = if ret_ty.cast.is_some() {
                    let tmp = LvalueRef::alloca(bcx, op.ty, "tmp_ret");
                    ret_ty.store(bcx, op.immediate(), tmp.llval);
                    self.trans_load(bcx, tmp.llval, tmp.alignment, op.ty)
                } else {
                    op.unpack_if_pair(bcx)
                };
                self.locals[index] = LocalRef::Operand(Some(op));
            }
        }
    }
}

enum ReturnDest {
    // Do nothing, the return value is indirect or ignored
    Nothing,
    // Store the return value to the pointer
    Store(ValueRef),
    // Stores an indirect return value to an operand local lvalue
    IndirectOperand(ValueRef, mir::Local),
    // Stores a direct return value to an operand local lvalue
    DirectOperand(mir::Local)
}
