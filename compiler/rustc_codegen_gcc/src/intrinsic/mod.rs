pub mod llvm;
mod simd;

#[cfg(feature="master")]
use std::iter;

use gccjit::{ComparisonOp, Function, RValue, ToRValue, Type, UnaryOp, FunctionType};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::base::wants_msvc_seh;
use rustc_codegen_ssa::common::IntPredicate;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{ArgAbiMethods, BaseTypeMethods, BuilderMethods, ConstMethods, IntrinsicCallMethods};
#[cfg(feature="master")]
use rustc_codegen_ssa::traits::{DerivedTypeMethods, MiscMethods};
use rustc_middle::bug;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_middle::ty::layout::LayoutOf;
#[cfg(feature="master")]
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt};
use rustc_span::{Span, Symbol, symbol::kw, sym};
use rustc_target::abi::HasDataLayout;
use rustc_target::abi::call::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::PanicStrategy;
#[cfg(feature="master")]
use rustc_target::spec::abi::Abi;

use crate::abi::GccType;
#[cfg(feature="master")]
use crate::abi::FnAbiGccExt;
use crate::builder::Builder;
use crate::common::{SignType, TypeReflection};
use crate::context::CodegenCx;
use crate::errors::InvalidMonomorphizationBasicInteger;
use crate::type_of::LayoutGccExt;
use crate::intrinsic::simd::generic_simd_intrinsic;

fn get_simple_intrinsic<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, name: Symbol) -> Option<Function<'gcc>> {
    let gcc_name = match name {
        sym::sqrtf32 => "sqrtf",
        sym::sqrtf64 => "sqrt",
        sym::powif32 => "__builtin_powif",
        sym::powif64 => "__builtin_powi",
        sym::sinf32 => "sinf",
        sym::sinf64 => "sin",
        sym::cosf32 => "cosf",
        sym::cosf64 => "cos",
        sym::powf32 => "powf",
        sym::powf64 => "pow",
        sym::expf32 => "expf",
        sym::expf64 => "exp",
        sym::exp2f32 => "exp2f",
        sym::exp2f64 => "exp2",
        sym::logf32 => "logf",
        sym::logf64 => "log",
        sym::log10f32 => "log10f",
        sym::log10f64 => "log10",
        sym::log2f32 => "log2f",
        sym::log2f64 => "log2",
        sym::fmaf32 => "fmaf",
        sym::fmaf64 => "fma",
        sym::fabsf32 => "fabsf",
        sym::fabsf64 => "fabs",
        sym::minnumf32 => "fminf",
        sym::minnumf64 => "fmin",
        sym::maxnumf32 => "fmaxf",
        sym::maxnumf64 => "fmax",
        sym::copysignf32 => "copysignf",
        sym::copysignf64 => "copysign",
        sym::floorf32 => "floorf",
        sym::floorf64 => "floor",
        sym::ceilf32 => "ceilf",
        sym::ceilf64 => "ceil",
        sym::truncf32 => "truncf",
        sym::truncf64 => "trunc",
        sym::rintf32 => "rintf",
        sym::rintf64 => "rint",
        sym::nearbyintf32 => "nearbyintf",
        sym::nearbyintf64 => "nearbyint",
        sym::roundf32 => "roundf",
        sym::roundf64 => "round",
        sym::roundevenf32 => "roundevenf",
        sym::roundevenf64 => "roundeven",
        sym::abort => "abort",
        _ => return None,
    };
    Some(cx.context.get_builtin_function(&gcc_name))
}

impl<'a, 'gcc, 'tcx> IntrinsicCallMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn codegen_intrinsic_call(&mut self, instance: Instance<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>, args: &[OperandRef<'tcx, RValue<'gcc>>], llresult: RValue<'gcc>, span: Span) {
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, ty::ParamEnv::reveal_all());

        let (def_id, substs) = match *callee_ty.kind() {
            ty::FnDef(def_id, substs) => (def_id, substs),
            _ => bug!("expected fn item type, found {}", callee_ty),
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = tcx.item_name(def_id);
        let name_str = name.as_str();

        let llret_ty = self.layout_of(ret_ty).gcc_type(self);
        let result = PlaceRef::new_sized(llresult, fn_abi.ret.layout);

        let simple = get_simple_intrinsic(self, name);
        let llval =
            match name {
                _ if simple.is_some() => {
                    // FIXME(antoyo): remove this cast when the API supports function.
                    let func = unsafe { std::mem::transmute(simple.expect("simple")) };
                    self.call(self.type_void(), None, func, &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(), None)
                },
                sym::likely => {
                    self.expect(args[0].immediate(), true)
                }
                sym::unlikely => {
                    self.expect(args[0].immediate(), false)
                }
                kw::Try => {
                    try_intrinsic(
                        self,
                        args[0].immediate(),
                        args[1].immediate(),
                        args[2].immediate(),
                        llresult,
                    );
                    return;
                }
                sym::breakpoint => {
                    unimplemented!();
                }
                sym::va_copy => {
                    unimplemented!();
                }
                sym::va_arg => {
                    unimplemented!();
                }

                sym::volatile_load | sym::unaligned_volatile_load => {
                    let tp_ty = substs.type_at(0);
                    let mut ptr = args[0].immediate();
                    if let PassMode::Cast(ty, _) = &fn_abi.ret.mode {
                        ptr = self.pointercast(ptr, self.type_ptr_to(ty.gcc_type(self)));
                    }
                    let load = self.volatile_load(ptr.get_type(), ptr);
                    // TODO(antoyo): set alignment.
                    self.to_immediate(load, self.layout_of(tp_ty))
                }
                sym::volatile_store => {
                    let dst = args[0].deref(self.cx());
                    args[1].val.volatile_store(self, dst);
                    return;
                }
                sym::unaligned_volatile_store => {
                    let dst = args[0].deref(self.cx());
                    args[1].val.unaligned_volatile_store(self, dst);
                    return;
                }
                sym::prefetch_read_data
                    | sym::prefetch_write_data
                    | sym::prefetch_read_instruction
                    | sym::prefetch_write_instruction => {
                        unimplemented!();
                    }
                sym::ctlz
                    | sym::ctlz_nonzero
                    | sym::cttz
                    | sym::cttz_nonzero
                    | sym::ctpop
                    | sym::bswap
                    | sym::bitreverse
                    | sym::rotate_left
                    | sym::rotate_right
                    | sym::saturating_add
                    | sym::saturating_sub => {
                        let ty = arg_tys[0];
                        match int_type_width_signed(ty, self) {
                            Some((width, signed)) => match name {
                                sym::ctlz | sym::cttz => {
                                    let func = self.current_func.borrow().expect("func");
                                    let then_block = func.new_block("then");
                                    let else_block = func.new_block("else");
                                    let after_block = func.new_block("after");

                                    let arg = args[0].immediate();
                                    let result = func.new_local(None, arg.get_type(), "zeros");
                                    let zero = self.cx.gcc_zero(arg.get_type());
                                    let cond = self.gcc_icmp(IntPredicate::IntEQ, arg, zero);
                                    self.llbb().end_with_conditional(None, cond, then_block, else_block);

                                    let zero_result = self.cx.gcc_uint(arg.get_type(), width);
                                    then_block.add_assignment(None, result, zero_result);
                                    then_block.end_with_jump(None, after_block);

                                    // NOTE: since jumps were added in a place
                                    // count_leading_zeroes() does not expect, the current block
                                    // in the state need to be updated.
                                    self.switch_to_block(else_block);

                                    let zeros =
                                        match name {
                                            sym::ctlz => self.count_leading_zeroes(width, arg),
                                            sym::cttz => self.count_trailing_zeroes(width, arg),
                                            _ => unreachable!(),
                                        };
                                    self.llbb().add_assignment(None, result, zeros);
                                    self.llbb().end_with_jump(None, after_block);

                                    // NOTE: since jumps were added in a place rustc does not
                                    // expect, the current block in the state need to be updated.
                                    self.switch_to_block(after_block);

                                    result.to_rvalue()
                                }
                                sym::ctlz_nonzero => {
                                    self.count_leading_zeroes(width, args[0].immediate())
                                },
                                sym::cttz_nonzero => {
                                    self.count_trailing_zeroes(width, args[0].immediate())
                                }
                                sym::ctpop => self.pop_count(args[0].immediate()),
                                sym::bswap => {
                                    if width == 8 {
                                        args[0].immediate() // byte swap a u8/i8 is just a no-op
                                    }
                                    else {
                                        self.gcc_bswap(args[0].immediate(), width)
                                    }
                                },
                                sym::bitreverse => self.bit_reverse(width, args[0].immediate()),
                                sym::rotate_left | sym::rotate_right => {
                                    // TODO(antoyo): implement using algorithm from:
                                    // https://blog.regehr.org/archives/1063
                                    // for other platforms.
                                    let is_left = name == sym::rotate_left;
                                    let val = args[0].immediate();
                                    let raw_shift = args[1].immediate();
                                    if is_left {
                                        self.rotate_left(val, raw_shift, width)
                                    }
                                    else {
                                        self.rotate_right(val, raw_shift, width)
                                    }
                                },
                                sym::saturating_add => {
                                    self.saturating_add(args[0].immediate(), args[1].immediate(), signed, width)
                                },
                                sym::saturating_sub => {
                                    self.saturating_sub(args[0].immediate(), args[1].immediate(), signed, width)
                                },
                                _ => bug!(),
                            },
                            None => {
                                tcx.sess.emit_err(InvalidMonomorphizationBasicInteger { span, name, ty });
                                return;
                            }
                        }
                    }

                sym::raw_eq => {
                    use rustc_target::abi::Abi::*;
                    let tp_ty = substs.type_at(0);
                    let layout = self.layout_of(tp_ty).layout;
                    let _use_integer_compare = match layout.abi() {
                        Scalar(_) | ScalarPair(_, _) => true,
                        Uninhabited | Vector { .. } => false,
                        Aggregate { .. } => {
                            // For rusty ABIs, small aggregates are actually passed
                            // as `RegKind::Integer` (see `FnAbi::adjust_for_abi`),
                            // so we re-use that same threshold here.
                            layout.size() <= self.data_layout().pointer_size * 2
                        }
                    };

                    let a = args[0].immediate();
                    let b = args[1].immediate();
                    if layout.size().bytes() == 0 {
                        self.const_bool(true)
                    }
                    /*else if use_integer_compare {
                        let integer_ty = self.type_ix(layout.size.bits()); // FIXME(antoyo): LLVM creates an integer of 96 bits for [i32; 3], but gcc doesn't support this, so it creates an integer of 128 bits.
                        let ptr_ty = self.type_ptr_to(integer_ty);
                        let a_ptr = self.bitcast(a, ptr_ty);
                        let a_val = self.load(integer_ty, a_ptr, layout.align.abi);
                        let b_ptr = self.bitcast(b, ptr_ty);
                        let b_val = self.load(integer_ty, b_ptr, layout.align.abi);
                        self.icmp(IntPredicate::IntEQ, a_val, b_val)
                    }*/
                    else {
                        let void_ptr_type = self.context.new_type::<*const ()>();
                        let a_ptr = self.bitcast(a, void_ptr_type);
                        let b_ptr = self.bitcast(b, void_ptr_type);
                        let n = self.context.new_cast(None, self.const_usize(layout.size().bytes()), self.sizet_type);
                        let builtin = self.context.get_builtin_function("memcmp");
                        let cmp = self.context.new_call(None, builtin, &[a_ptr, b_ptr, n]);
                        self.icmp(IntPredicate::IntEQ, cmp, self.const_i32(0))
                    }
                }

                sym::black_box => {
                    args[0].val.store(self, result);

                    let block = self.llbb();
                    let extended_asm = block.add_extended_asm(None, "");
                    extended_asm.add_input_operand(None, "r", result.llval);
                    extended_asm.add_clobber("memory");
                    extended_asm.set_volatile_flag(true);

                    // We have copied the value to `result` already.
                    return;
                }

                sym::ptr_mask => {
                    let usize_type = self.context.new_type::<usize>();
                    let void_ptr_type = self.context.new_type::<*const ()>();

                    let ptr = args[0].immediate();
                    let mask = args[1].immediate();

                    let addr = self.bitcast(ptr, usize_type);
                    let masked = self.and(addr, mask);
                    self.bitcast(masked, void_ptr_type)
                },
                
                _ if name_str.starts_with("simd_") => {
                    match generic_simd_intrinsic(self, name, callee_ty, args, ret_ty, llret_ty, span) {
                        Ok(llval) => llval,
                        Err(()) => return,
                    }
                }

                _ => bug!("unknown intrinsic '{}'", name),
            };

        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast(ty, _) = &fn_abi.ret.mode {
                let ptr_llty = self.type_ptr_to(ty.gcc_type(self));
                let ptr = self.pointercast(result.llval, ptr_llty);
                self.store(llval, ptr, result.align);
            }
            else {
                OperandRef::from_immediate_or_packed_pair(self, llval, result.layout)
                    .val
                    .store(self, result);
            }
        }
    }

    fn abort(&mut self) {
        let func = self.context.get_builtin_function("abort");
        let func: RValue<'gcc> = unsafe { std::mem::transmute(func) };
        self.call(self.type_void(), None, func, &[], None);
    }

    fn assume(&mut self, value: Self::Value) {
        // TODO(antoyo): switch to assume when it exists.
        // Or use something like this:
        // #define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
        self.expect(value, true);
    }

    fn expect(&mut self, cond: Self::Value, _expected: bool) -> Self::Value {
        // TODO(antoyo)
        cond
    }

    fn type_test(&mut self, _pointer: Self::Value, _typeid: Self::Value) -> Self::Value {
        // Unsupported.
        self.context.new_rvalue_from_int(self.int_type, 0)
    }

    fn type_checked_load(
        &mut self,
        _llvtable: Self::Value,
        _vtable_byte_offset: u64,
        _typeid: Self::Value,
    ) -> Self::Value {
        // Unsupported.
        self.context.new_rvalue_from_int(self.int_type, 0)
    }

    fn va_start(&mut self, _va_list: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn va_end(&mut self, _va_list: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }
}

impl<'a, 'gcc, 'tcx> ArgAbiMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn store_fn_arg(&mut self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>, idx: &mut usize, dst: PlaceRef<'tcx, Self::Value>) {
        arg_abi.store_fn_arg(self, idx, dst)
    }

    fn store_arg(&mut self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>, val: RValue<'gcc>, dst: PlaceRef<'tcx, RValue<'gcc>>) {
        arg_abi.store(self, val, dst)
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> Type<'gcc> {
        arg_abi.memory_ty(self)
    }
}

pub trait ArgAbiExt<'gcc, 'tcx> {
    fn memory_ty(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    fn store(&self, bx: &mut Builder<'_, 'gcc, 'tcx>, val: RValue<'gcc>, dst: PlaceRef<'tcx, RValue<'gcc>>);
    fn store_fn_arg(&self, bx: &mut Builder<'_, 'gcc, 'tcx>, idx: &mut usize, dst: PlaceRef<'tcx, RValue<'gcc>>);
}

impl<'gcc, 'tcx> ArgAbiExt<'gcc, 'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    /// Gets the LLVM type for a place of the original Rust type of
    /// this argument/return, i.e., the result of `type_of::type_of`.
    fn memory_ty(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        self.layout.gcc_type(cx)
    }

    /// Stores a direct/indirect value described by this ArgAbi into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    fn store(&self, bx: &mut Builder<'_, 'gcc, 'tcx>, val: RValue<'gcc>, dst: PlaceRef<'tcx, RValue<'gcc>>) {
        if self.is_ignore() {
            return;
        }
        if self.is_sized_indirect() {
            OperandValue::Ref(val, None, self.layout.align.abi).store(bx, dst)
        }
        else if self.is_unsized_indirect() {
            bug!("unsized `ArgAbi` must be handled through `store_fn_arg`");
        }
        else if let PassMode::Cast(ref cast, _) = self.mode {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_ptr_llty = bx.type_ptr_to(cast.gcc_type(bx));
                let cast_dst = bx.pointercast(dst.llval, cast_ptr_llty);
                bx.store(val, cast_dst, self.layout.align.abi);
            }
            else {
                // The actual return type is a struct, but the ABI
                // adaptation code has cast it into some scalar type.  The
                // code that follows is the only reliable way I have
                // found to do a transform like i64 -> {i32,i32}.
                // Basically we dump the data onto the stack then memcpy it.
                //
                // Other approaches I tried:
                // - Casting rust ret pointer to the foreign type and using Store
                //   is (a) unsafe if size of foreign type > size of rust type and
                //   (b) runs afoul of strict aliasing rules, yielding invalid
                //   assembly under -O (specifically, the store gets removed).
                // - Truncating foreign type to correct integral type and then
                //   bitcasting to the struct type yields invalid cast errors.

                // We instead thus allocate some scratch space...
                let scratch_size = cast.size(bx);
                let scratch_align = cast.align(bx);
                let llscratch = bx.alloca(cast.gcc_type(bx), scratch_align);
                bx.lifetime_start(llscratch, scratch_size);

                // ... where we first store the value...
                bx.store(val, llscratch, scratch_align);

                // ... and then memcpy it to the intended destination.
                bx.memcpy(
                    dst.llval,
                    self.layout.align.abi,
                    llscratch,
                    scratch_align,
                    bx.const_usize(self.layout.size.bytes()),
                    MemFlags::empty(),
                );

                bx.lifetime_end(llscratch, scratch_size);
            }
        }
        else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    fn store_fn_arg<'a>(&self, bx: &mut Builder<'a, 'gcc, 'tcx>, idx: &mut usize, dst: PlaceRef<'tcx, RValue<'gcc>>) {
        let mut next = || {
            let val = bx.current_func().get_param(*idx as i32);
            *idx += 1;
            val.to_rvalue()
        };
        match self.mode {
            PassMode::Ignore => {},
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            },
            PassMode::Indirect { extra_attrs: Some(_), .. } => {
                OperandValue::Ref(next(), Some(next()), self.layout.align.abi).store(bx, dst);
            },
            PassMode::Direct(_) | PassMode::Indirect { extra_attrs: None, .. } | PassMode::Cast(..) => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            },
        }
    }
}

fn int_type_width_signed<'gcc, 'tcx>(ty: Ty<'tcx>, cx: &CodegenCx<'gcc, 'tcx>) -> Option<(u64, bool)> {
    match ty.kind() {
        ty::Int(t) => Some((
            match t {
                rustc_middle::ty::IntTy::Isize => u64::from(cx.tcx.sess.target.pointer_width),
                rustc_middle::ty::IntTy::I8 => 8,
                rustc_middle::ty::IntTy::I16 => 16,
                rustc_middle::ty::IntTy::I32 => 32,
                rustc_middle::ty::IntTy::I64 => 64,
                rustc_middle::ty::IntTy::I128 => 128,
            },
            true,
        )),
        ty::Uint(t) => Some((
            match t {
                rustc_middle::ty::UintTy::Usize => u64::from(cx.tcx.sess.target.pointer_width),
                rustc_middle::ty::UintTy::U8 => 8,
                rustc_middle::ty::UintTy::U16 => 16,
                rustc_middle::ty::UintTy::U32 => 32,
                rustc_middle::ty::UintTy::U64 => 64,
                rustc_middle::ty::UintTy::U128 => 128,
            },
            false,
        )),
        _ => None,
    }
}

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    fn bit_reverse(&mut self, width: u64, value: RValue<'gcc>) -> RValue<'gcc> {
        let result_type = value.get_type();
        let typ = result_type.to_unsigned(self.cx);

        let value =
            if result_type.is_signed(self.cx) {
                self.gcc_int_cast(value, typ)
            }
            else {
                value
            };

        let context = &self.cx.context;
        let result =
            match width {
                8 => {
                    // First step.
                    let left = self.and(value, context.new_rvalue_from_int(typ, 0xF0));
                    let left = self.lshr(left, context.new_rvalue_from_int(typ, 4));
                    let right = self.and(value, context.new_rvalue_from_int(typ, 0x0F));
                    let right = self.shl(right, context.new_rvalue_from_int(typ, 4));
                    let step1 = self.or(left, right);

                    // Second step.
                    let left = self.and(step1, context.new_rvalue_from_int(typ, 0xCC));
                    let left = self.lshr(left, context.new_rvalue_from_int(typ, 2));
                    let right = self.and(step1, context.new_rvalue_from_int(typ, 0x33));
                    let right = self.shl(right, context.new_rvalue_from_int(typ, 2));
                    let step2 = self.or(left, right);

                    // Third step.
                    let left = self.and(step2, context.new_rvalue_from_int(typ, 0xAA));
                    let left = self.lshr(left, context.new_rvalue_from_int(typ, 1));
                    let right = self.and(step2, context.new_rvalue_from_int(typ, 0x55));
                    let right = self.shl(right, context.new_rvalue_from_int(typ, 1));
                    let step3 = self.or(left, right);

                    step3
                },
                16 => {
                    // First step.
                    let left = self.and(value, context.new_rvalue_from_int(typ, 0x5555));
                    let left = self.shl(left, context.new_rvalue_from_int(typ, 1));
                    let right = self.and(value, context.new_rvalue_from_int(typ, 0xAAAA));
                    let right = self.lshr(right, context.new_rvalue_from_int(typ, 1));
                    let step1 = self.or(left, right);

                    // Second step.
                    let left = self.and(step1, context.new_rvalue_from_int(typ, 0x3333));
                    let left = self.shl(left, context.new_rvalue_from_int(typ, 2));
                    let right = self.and(step1, context.new_rvalue_from_int(typ, 0xCCCC));
                    let right = self.lshr(right, context.new_rvalue_from_int(typ, 2));
                    let step2 = self.or(left, right);

                    // Third step.
                    let left = self.and(step2, context.new_rvalue_from_int(typ, 0x0F0F));
                    let left = self.shl(left, context.new_rvalue_from_int(typ, 4));
                    let right = self.and(step2, context.new_rvalue_from_int(typ, 0xF0F0));
                    let right = self.lshr(right, context.new_rvalue_from_int(typ, 4));
                    let step3 = self.or(left, right);

                    // Fourth step.
                    let left = self.and(step3, context.new_rvalue_from_int(typ, 0x00FF));
                    let left = self.shl(left, context.new_rvalue_from_int(typ, 8));
                    let right = self.and(step3, context.new_rvalue_from_int(typ, 0xFF00));
                    let right = self.lshr(right, context.new_rvalue_from_int(typ, 8));
                    let step4 = self.or(left, right);

                    step4
                },
                32 => {
                    // TODO(antoyo): Refactor with other implementations.
                    // First step.
                    let left = self.and(value, context.new_rvalue_from_long(typ, 0x55555555));
                    let left = self.shl(left, context.new_rvalue_from_long(typ, 1));
                    let right = self.and(value, context.new_rvalue_from_long(typ, 0xAAAAAAAA));
                    let right = self.lshr(right, context.new_rvalue_from_long(typ, 1));
                    let step1 = self.or(left, right);

                    // Second step.
                    let left = self.and(step1, context.new_rvalue_from_long(typ, 0x33333333));
                    let left = self.shl(left, context.new_rvalue_from_long(typ, 2));
                    let right = self.and(step1, context.new_rvalue_from_long(typ, 0xCCCCCCCC));
                    let right = self.lshr(right, context.new_rvalue_from_long(typ, 2));
                    let step2 = self.or(left, right);

                    // Third step.
                    let left = self.and(step2, context.new_rvalue_from_long(typ, 0x0F0F0F0F));
                    let left = self.shl(left, context.new_rvalue_from_long(typ, 4));
                    let right = self.and(step2, context.new_rvalue_from_long(typ, 0xF0F0F0F0));
                    let right = self.lshr(right, context.new_rvalue_from_long(typ, 4));
                    let step3 = self.or(left, right);

                    // Fourth step.
                    let left = self.and(step3, context.new_rvalue_from_long(typ, 0x00FF00FF));
                    let left = self.shl(left, context.new_rvalue_from_long(typ, 8));
                    let right = self.and(step3, context.new_rvalue_from_long(typ, 0xFF00FF00));
                    let right = self.lshr(right, context.new_rvalue_from_long(typ, 8));
                    let step4 = self.or(left, right);

                    // Fifth step.
                    let left = self.and(step4, context.new_rvalue_from_long(typ, 0x0000FFFF));
                    let left = self.shl(left, context.new_rvalue_from_long(typ, 16));
                    let right = self.and(step4, context.new_rvalue_from_long(typ, 0xFFFF0000));
                    let right = self.lshr(right, context.new_rvalue_from_long(typ, 16));
                    let step5 = self.or(left, right);

                    step5
                },
                64 => {
                    // First step.
                    let left = self.shl(value, context.new_rvalue_from_long(typ, 32));
                    let right = self.lshr(value, context.new_rvalue_from_long(typ, 32));
                    let step1 = self.or(left, right);

                    // Second step.
                    let left = self.and(step1, context.new_rvalue_from_long(typ, 0x0001FFFF0001FFFF));
                    let left = self.shl(left, context.new_rvalue_from_long(typ, 15));
                    let right = self.and(step1, context.new_rvalue_from_long(typ, 0xFFFE0000FFFE0000u64 as i64)); // TODO(antoyo): transmute the number instead?
                    let right = self.lshr(right, context.new_rvalue_from_long(typ, 17));
                    let step2 = self.or(left, right);

                    // Third step.
                    let left = self.lshr(step2, context.new_rvalue_from_long(typ, 10));
                    let left = self.xor(step2, left);
                    let temp = self.and(left, context.new_rvalue_from_long(typ, 0x003F801F003F801F));

                    let left = self.shl(temp, context.new_rvalue_from_long(typ, 10));
                    let left = self.or(temp, left);
                    let step3 = self.xor(left, step2);

                    // Fourth step.
                    let left = self.lshr(step3, context.new_rvalue_from_long(typ, 4));
                    let left = self.xor(step3, left);
                    let temp = self.and(left, context.new_rvalue_from_long(typ, 0x0E0384210E038421));

                    let left = self.shl(temp, context.new_rvalue_from_long(typ, 4));
                    let left = self.or(temp, left);
                    let step4 = self.xor(left, step3);

                    // Fifth step.
                    let left = self.lshr(step4, context.new_rvalue_from_long(typ, 2));
                    let left = self.xor(step4, left);
                    let temp = self.and(left, context.new_rvalue_from_long(typ, 0x2248884222488842));

                    let left = self.shl(temp, context.new_rvalue_from_long(typ, 2));
                    let left = self.or(temp, left);
                    let step5 = self.xor(left, step4);

                    step5
                },
                128 => {
                    // TODO(antoyo): find a more efficient implementation?
                    let sixty_four = self.gcc_int(typ, 64);
                    let right_shift = self.gcc_lshr(value, sixty_four);
                    let high = self.gcc_int_cast(right_shift, self.u64_type);
                    let low = self.gcc_int_cast(value, self.u64_type);

                    let reversed_high = self.bit_reverse(64, high);
                    let reversed_low = self.bit_reverse(64, low);

                    let new_low = self.gcc_int_cast(reversed_high, typ);
                    let new_high = self.shl(self.gcc_int_cast(reversed_low, typ), sixty_four);

                    self.gcc_or(new_low, new_high)
                },
                _ => {
                    panic!("cannot bit reverse with width = {}", width);
                },
            };

        self.gcc_int_cast(result, result_type)
    }

    fn count_leading_zeroes(&mut self, width: u64, arg: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): use width?
        let arg_type = arg.get_type();
        let count_leading_zeroes =
            // TODO(antoyo): write a new function Type::is_compatible_with(&Type) and use it here
            // instead of using is_uint().
            if arg_type.is_uint(&self.cx) {
                "__builtin_clz"
            }
            else if arg_type.is_ulong(&self.cx) {
                "__builtin_clzl"
            }
            else if arg_type.is_ulonglong(&self.cx) {
                "__builtin_clzll"
            }
            else if width == 128 {
                // Algorithm from: https://stackoverflow.com/a/28433850/389119
                let array_type = self.context.new_array_type(None, arg_type, 3);
                let result = self.current_func()
                    .new_local(None, array_type, "count_loading_zeroes_results");

                let sixty_four = self.const_uint(arg_type, 64);
                let shift = self.lshr(arg, sixty_four);
                let high = self.gcc_int_cast(shift, self.u64_type);
                let low = self.gcc_int_cast(arg, self.u64_type);

                let zero = self.context.new_rvalue_zero(self.usize_type);
                let one = self.context.new_rvalue_one(self.usize_type);
                let two = self.context.new_rvalue_from_long(self.usize_type, 2);

                let clzll = self.context.get_builtin_function("__builtin_clzll");

                let first_elem = self.context.new_array_access(None, result, zero);
                let first_value = self.gcc_int_cast(self.context.new_call(None, clzll, &[high]), arg_type);
                self.llbb()
                    .add_assignment(None, first_elem, first_value);

                let second_elem = self.context.new_array_access(None, result, one);
                let cast = self.gcc_int_cast(self.context.new_call(None, clzll, &[low]), arg_type);
                let second_value = self.add(cast, sixty_four);
                self.llbb()
                    .add_assignment(None, second_elem, second_value);

                let third_elem = self.context.new_array_access(None, result, two);
                let third_value = self.const_uint(arg_type, 128);
                self.llbb()
                    .add_assignment(None, third_elem, third_value);

                let not_high = self.context.new_unary_op(None, UnaryOp::LogicalNegate, self.u64_type, high);
                let not_low = self.context.new_unary_op(None, UnaryOp::LogicalNegate, self.u64_type, low);
                let not_low_and_not_high = not_low & not_high;
                let index = not_high + not_low_and_not_high;
                // NOTE: the following cast is necessary to avoid a GIMPLE verification failure in
                // gcc.
                // TODO(antoyo): do the correct verification in libgccjit to avoid an error at the
                // compilation stage.
                let index = self.context.new_cast(None, index, self.i32_type);

                let res = self.context.new_array_access(None, result, index);

                return self.gcc_int_cast(res.to_rvalue(), arg_type);
            }
            else {
                let count_leading_zeroes = self.context.get_builtin_function("__builtin_clzll");
                let arg = self.context.new_cast(None, arg, self.ulonglong_type);
                let diff = self.ulonglong_type.get_size() as i64 - arg_type.get_size() as i64;
                let diff = self.context.new_rvalue_from_long(self.int_type, diff * 8);
                let res = self.context.new_call(None, count_leading_zeroes, &[arg]) - diff;
                return self.context.new_cast(None, res, arg_type);
            };
        let count_leading_zeroes = self.context.get_builtin_function(count_leading_zeroes);
        let res = self.context.new_call(None, count_leading_zeroes, &[arg]);
        self.context.new_cast(None, res, arg_type)
    }

    fn count_trailing_zeroes(&mut self, _width: u64, arg: RValue<'gcc>) -> RValue<'gcc> {
        let result_type = arg.get_type();
        let arg =
            if result_type.is_signed(self.cx) {
                let new_type = result_type.to_unsigned(self.cx);
                self.gcc_int_cast(arg, new_type)
            }
            else {
                arg
            };
        let arg_type = arg.get_type();
        let (count_trailing_zeroes, expected_type) =
            // TODO(antoyo): write a new function Type::is_compatible_with(&Type) and use it here
            // instead of using is_uint().
            if arg_type.is_uchar(&self.cx) || arg_type.is_ushort(&self.cx) || arg_type.is_uint(&self.cx) {
                // NOTE: we don't need to & 0xFF for uchar because the result is undefined on zero.
                ("__builtin_ctz", self.cx.uint_type)
            }
            else if arg_type.is_ulong(&self.cx) {
                ("__builtin_ctzl", self.cx.ulong_type)
            }
            else if arg_type.is_ulonglong(&self.cx) {
                ("__builtin_ctzll", self.cx.ulonglong_type)
            }
            else if arg_type.is_u128(&self.cx) {
                // Adapted from the algorithm to count leading zeroes from: https://stackoverflow.com/a/28433850/389119
                let array_type = self.context.new_array_type(None, arg_type, 3);
                let result = self.current_func()
                    .new_local(None, array_type, "count_loading_zeroes_results");

                let sixty_four = self.gcc_int(arg_type, 64);
                let shift = self.gcc_lshr(arg, sixty_four);
                let high = self.gcc_int_cast(shift, self.u64_type);
                let low = self.gcc_int_cast(arg, self.u64_type);

                let zero = self.context.new_rvalue_zero(self.usize_type);
                let one = self.context.new_rvalue_one(self.usize_type);
                let two = self.context.new_rvalue_from_long(self.usize_type, 2);

                let ctzll = self.context.get_builtin_function("__builtin_ctzll");

                let first_elem = self.context.new_array_access(None, result, zero);
                let first_value = self.gcc_int_cast(self.context.new_call(None, ctzll, &[low]), arg_type);
                self.llbb()
                    .add_assignment(None, first_elem, first_value);

                let second_elem = self.context.new_array_access(None, result, one);
                let second_value = self.gcc_add(self.gcc_int_cast(self.context.new_call(None, ctzll, &[high]), arg_type), sixty_four);
                self.llbb()
                    .add_assignment(None, second_elem, second_value);

                let third_elem = self.context.new_array_access(None, result, two);
                let third_value = self.gcc_int(arg_type, 128);
                self.llbb()
                    .add_assignment(None, third_elem, third_value);

                let not_low = self.context.new_unary_op(None, UnaryOp::LogicalNegate, self.u64_type, low);
                let not_high = self.context.new_unary_op(None, UnaryOp::LogicalNegate, self.u64_type, high);
                let not_low_and_not_high = not_low & not_high;
                let index = not_low + not_low_and_not_high;
                // NOTE: the following cast is necessary to avoid a GIMPLE verification failure in
                // gcc.
                // TODO(antoyo): do the correct verification in libgccjit to avoid an error at the
                // compilation stage.
                let index = self.context.new_cast(None, index, self.i32_type);

                let res = self.context.new_array_access(None, result, index);

                return self.gcc_int_cast(res.to_rvalue(), result_type);
            }
            else {
                let count_trailing_zeroes = self.context.get_builtin_function("__builtin_ctzll");
                let arg_size = arg_type.get_size();
                let casted_arg = self.context.new_cast(None, arg, self.ulonglong_type);
                let byte_diff = self.ulonglong_type.get_size() as i64 - arg_size as i64;
                let diff = self.context.new_rvalue_from_long(self.int_type, byte_diff * 8);
                let mask = self.context.new_rvalue_from_long(arg_type, -1); // To get the value with all bits set.
                let masked = mask & self.context.new_unary_op(None, UnaryOp::BitwiseNegate, arg_type, arg);
                let cond = self.context.new_comparison(None, ComparisonOp::Equals, masked, mask);
                let diff = diff * self.context.new_cast(None, cond, self.int_type);
                let res = self.context.new_call(None, count_trailing_zeroes, &[casted_arg]) - diff;
                return self.context.new_cast(None, res, result_type);
            };
        let count_trailing_zeroes = self.context.get_builtin_function(count_trailing_zeroes);
        let arg =
            if arg_type != expected_type {
                self.context.new_cast(None, arg, expected_type)
            }
            else {
                arg
            };
        let res = self.context.new_call(None, count_trailing_zeroes, &[arg]);
        self.context.new_cast(None, res, result_type)
    }

    fn pop_count(&mut self, value: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): use the optimized version with fewer operations.
        let result_type = value.get_type();
        let value_type = result_type.to_unsigned(self.cx);

        let value =
            if result_type.is_signed(self.cx) {
                self.gcc_int_cast(value, value_type)
            }
            else {
                value
            };

        if value_type.is_u128(&self.cx) {
            // TODO(antoyo): implement in the normal algorithm below to have a more efficient
            // implementation (that does not require a call to __popcountdi2).
            let popcount = self.context.get_builtin_function("__builtin_popcountll");
            let sixty_four = self.gcc_int(value_type, 64);
            let right_shift = self.gcc_lshr(value, sixty_four);
            let high = self.gcc_int_cast(right_shift, self.cx.ulonglong_type);
            let high = self.context.new_call(None, popcount, &[high]);
            let low = self.gcc_int_cast(value, self.cx.ulonglong_type);
            let low = self.context.new_call(None, popcount, &[low]);
            let res = high + low;
            return self.gcc_int_cast(res, result_type);
        }

        // First step.
        let mask = self.context.new_rvalue_from_long(value_type, 0x5555555555555555);
        let left = value & mask;
        let shifted = value >> self.context.new_rvalue_from_int(value_type, 1);
        let right = shifted & mask;
        let value = left + right;

        // Second step.
        let mask = self.context.new_rvalue_from_long(value_type, 0x3333333333333333);
        let left = value & mask;
        let shifted = value >> self.context.new_rvalue_from_int(value_type, 2);
        let right = shifted & mask;
        let value = left + right;

        // Third step.
        let mask = self.context.new_rvalue_from_long(value_type, 0x0F0F0F0F0F0F0F0F);
        let left = value & mask;
        let shifted = value >> self.context.new_rvalue_from_int(value_type, 4);
        let right = shifted & mask;
        let value = left + right;

        if value_type.is_u8(&self.cx) {
            return self.context.new_cast(None, value, result_type);
        }

        // Fourth step.
        let mask = self.context.new_rvalue_from_long(value_type, 0x00FF00FF00FF00FF);
        let left = value & mask;
        let shifted = value >> self.context.new_rvalue_from_int(value_type, 8);
        let right = shifted & mask;
        let value = left + right;

        if value_type.is_u16(&self.cx) {
            return self.context.new_cast(None, value, result_type);
        }

        // Fifth step.
        let mask = self.context.new_rvalue_from_long(value_type, 0x0000FFFF0000FFFF);
        let left = value & mask;
        let shifted = value >> self.context.new_rvalue_from_int(value_type, 16);
        let right = shifted & mask;
        let value = left + right;

        if value_type.is_u32(&self.cx) {
            return self.context.new_cast(None, value, result_type);
        }

        // Sixth step.
        let mask = self.context.new_rvalue_from_long(value_type, 0x00000000FFFFFFFF);
        let left = value & mask;
        let shifted = value >> self.context.new_rvalue_from_int(value_type, 32);
        let right = shifted & mask;
        let value = left + right;

        self.context.new_cast(None, value, result_type)
    }

    // Algorithm from: https://blog.regehr.org/archives/1063
    fn rotate_left(&mut self, value: RValue<'gcc>, shift: RValue<'gcc>, width: u64) -> RValue<'gcc> {
        let max = self.const_uint(shift.get_type(), width);
        let shift = self.urem(shift, max);
        let lhs = self.shl(value, shift);
        let result_neg = self.neg(shift);
        let result_and =
            self.and(
                result_neg,
                self.const_uint(shift.get_type(), width - 1),
            );
        let rhs = self.lshr(value, result_and);
        self.or(lhs, rhs)
    }

    // Algorithm from: https://blog.regehr.org/archives/1063
    fn rotate_right(&mut self, value: RValue<'gcc>, shift: RValue<'gcc>, width: u64) -> RValue<'gcc> {
        let max = self.const_uint(shift.get_type(), width);
        let shift = self.urem(shift, max);
        let lhs = self.lshr(value, shift);
        let result_neg = self.neg(shift);
        let result_and =
            self.and(
                result_neg,
                self.const_uint(shift.get_type(), width - 1),
            );
        let rhs = self.shl(value, result_and);
        self.or(lhs, rhs)
    }

    fn saturating_add(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>, signed: bool, width: u64) -> RValue<'gcc> {
        let result_type = lhs.get_type();
        if signed {
            // Based on algorithm from: https://stackoverflow.com/a/56531252/389119
            let func = self.current_func.borrow().expect("func");
            let res = func.new_local(None, result_type, "saturating_sum");
            let supports_native_type = self.is_native_int_type(result_type);
            let overflow =
                if supports_native_type {
                    let func_name =
                        match width {
                            8 => "__builtin_add_overflow",
                            16 => "__builtin_add_overflow",
                            32 => "__builtin_sadd_overflow",
                            64 => "__builtin_saddll_overflow",
                            128 => "__builtin_add_overflow",
                            _ => unreachable!(),
                        };
                    let overflow_func = self.context.get_builtin_function(func_name);
                    self.overflow_call(overflow_func, &[lhs, rhs, res.get_address(None)], None)
                }
                else {
                    let func_name =
                        match width {
                            128 => "__rust_i128_addo",
                            _ => unreachable!(),
                        };
                    let param_a = self.context.new_parameter(None, result_type, "a");
                    let param_b = self.context.new_parameter(None, result_type, "b");
                    let result_field = self.context.new_field(None, result_type, "result");
                    let overflow_field = self.context.new_field(None, self.bool_type, "overflow");
                    let return_type = self.context.new_struct_type(None, "result_overflow", &[result_field, overflow_field]);
                    let func = self.context.new_function(None, FunctionType::Extern, return_type.as_type(), &[param_a, param_b], func_name, false);
                    let result = self.context.new_call(None, func, &[lhs, rhs]);
                    let overflow = result.access_field(None, overflow_field);
                    let int_result = result.access_field(None, result_field);
                    self.llbb().add_assignment(None, res, int_result);
                    overflow
                };

            let then_block = func.new_block("then");
            let after_block = func.new_block("after");

            // Return `result_type`'s maximum or minimum value on overflow
            // NOTE: convert the type to unsigned to have an unsigned shift.
            let unsigned_type = result_type.to_unsigned(&self.cx);
            let shifted = self.gcc_lshr(self.gcc_int_cast(lhs, unsigned_type), self.gcc_int(unsigned_type, width as i64 - 1));
            let uint_max = self.gcc_not(self.gcc_int(unsigned_type, 0));
            let int_max = self.gcc_lshr(uint_max, self.gcc_int(unsigned_type, 1));
            then_block.add_assignment(None, res, self.gcc_int_cast(self.gcc_add(shifted, int_max), result_type));
            then_block.end_with_jump(None, after_block);

            self.llbb().end_with_conditional(None, overflow, then_block, after_block);

            // NOTE: since jumps were added in a place rustc does not
            // expect, the current block in the state need to be updated.
            self.switch_to_block(after_block);

            res.to_rvalue()
        }
        else {
            // Algorithm from: http://locklessinc.com/articles/sat_arithmetic/
            let res = self.gcc_add(lhs, rhs);
            let cond = self.gcc_icmp(IntPredicate::IntULT, res, lhs);
            let value = self.gcc_neg(self.gcc_int_cast(cond, result_type));
            self.gcc_or(res, value)
        }
    }

    // Algorithm from: https://locklessinc.com/articles/sat_arithmetic/
    fn saturating_sub(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>, signed: bool, width: u64) -> RValue<'gcc> {
        let result_type = lhs.get_type();
        if signed {
            // Based on algorithm from: https://stackoverflow.com/a/56531252/389119
            let func = self.current_func.borrow().expect("func");
            let res = func.new_local(None, result_type, "saturating_diff");
            let supports_native_type = self.is_native_int_type(result_type);
            let overflow =
                if supports_native_type {
                    let func_name =
                        match width {
                            8 => "__builtin_sub_overflow",
                            16 => "__builtin_sub_overflow",
                            32 => "__builtin_ssub_overflow",
                            64 => "__builtin_ssubll_overflow",
                            128 => "__builtin_sub_overflow",
                            _ => unreachable!(),
                        };
                    let overflow_func = self.context.get_builtin_function(func_name);
                    self.overflow_call(overflow_func, &[lhs, rhs, res.get_address(None)], None)
                }
                else {
                    let func_name =
                        match width {
                            128 => "__rust_i128_subo",
                            _ => unreachable!(),
                        };
                    let param_a = self.context.new_parameter(None, result_type, "a");
                    let param_b = self.context.new_parameter(None, result_type, "b");
                    let result_field = self.context.new_field(None, result_type, "result");
                    let overflow_field = self.context.new_field(None, self.bool_type, "overflow");
                    let return_type = self.context.new_struct_type(None, "result_overflow", &[result_field, overflow_field]);
                    let func = self.context.new_function(None, FunctionType::Extern, return_type.as_type(), &[param_a, param_b], func_name, false);
                    let result = self.context.new_call(None, func, &[lhs, rhs]);
                    let overflow = result.access_field(None, overflow_field);
                    let int_result = result.access_field(None, result_field);
                    self.llbb().add_assignment(None, res, int_result);
                    overflow
                };

            let then_block = func.new_block("then");
            let after_block = func.new_block("after");

            // Return `result_type`'s maximum or minimum value on overflow
            // NOTE: convert the type to unsigned to have an unsigned shift.
            let unsigned_type = result_type.to_unsigned(&self.cx);
            let shifted = self.gcc_lshr(self.gcc_int_cast(lhs, unsigned_type), self.gcc_int(unsigned_type, width as i64 - 1));
            let uint_max = self.gcc_not(self.gcc_int(unsigned_type, 0));
            let int_max = self.gcc_lshr(uint_max, self.gcc_int(unsigned_type, 1));
            then_block.add_assignment(None, res, self.gcc_int_cast(self.gcc_add(shifted, int_max), result_type));
            then_block.end_with_jump(None, after_block);

            self.llbb().end_with_conditional(None, overflow, then_block, after_block);

            // NOTE: since jumps were added in a place rustc does not
            // expect, the current block in the state need to be updated.
            self.switch_to_block(after_block);

            res.to_rvalue()
        }
        else {
            let res = self.gcc_sub(lhs, rhs);
            let comparison = self.gcc_icmp(IntPredicate::IntULE, res, lhs);
            let value = self.gcc_neg(self.gcc_int_cast(comparison, result_type));
            self.gcc_and(res, value)
        }
    }
}

fn try_intrinsic<'a, 'b, 'gcc, 'tcx>(bx: &'b mut Builder<'a, 'gcc, 'tcx>, try_func: RValue<'gcc>, data: RValue<'gcc>, _catch_func: RValue<'gcc>, dest: RValue<'gcc>) {
    if bx.sess().panic_strategy() == PanicStrategy::Abort {
        bx.call(bx.type_void(), None, try_func, &[data], None);
        // Return 0 unconditionally from the intrinsic call;
        // we can never unwind.
        let ret_align = bx.tcx.data_layout.i32_align.abi;
        bx.store(bx.const_i32(0), dest, ret_align);
    }
    else if wants_msvc_seh(bx.sess()) {
        unimplemented!();
    }
    else {
        #[cfg(feature="master")]
        codegen_gnu_try(bx, try_func, data, _catch_func, dest);
        #[cfg(not(feature="master"))]
        unimplemented!();
    }
}

// Definition of the standard `try` function for Rust using the GNU-like model
// of exceptions (e.g., the normal semantics of LLVM's `landingpad` and `invoke`
// instructions).
//
// This codegen is a little surprising because we always call a shim
// function instead of inlining the call to `invoke` manually here. This is done
// because in LLVM we're only allowed to have one personality per function
// definition. The call to the `try` intrinsic is being inlined into the
// function calling it, and that function may already have other personality
// functions in play. By calling a shim we're guaranteed that our shim will have
// the right personality function.
#[cfg(feature="master")]
fn codegen_gnu_try<'gcc>(bx: &mut Builder<'_, 'gcc, '_>, try_func: RValue<'gcc>, data: RValue<'gcc>, catch_func: RValue<'gcc>, dest: RValue<'gcc>) {
    let cx: &CodegenCx<'gcc, '_> = bx.cx;
    let (llty, func) = get_rust_try_fn(cx, &mut |mut bx| {
        // Codegens the shims described above:
        //
        //   bx:
        //      invoke %try_func(%data) normal %normal unwind %catch
        //
        //   normal:
        //      ret 0
        //
        //   catch:
        //      (%ptr, _) = landingpad
        //      call %catch_func(%data, %ptr)
        //      ret 1
        let then = bx.append_sibling_block("then");
        let catch = bx.append_sibling_block("catch");

        let func = bx.current_func();
        let try_func = func.get_param(0).to_rvalue();
        let data = func.get_param(1).to_rvalue();
        let catch_func = func.get_param(2).to_rvalue();
        let try_func_ty = bx.type_func(&[bx.type_i8p()], bx.type_void());

        let current_block = bx.block.clone();

        bx.switch_to_block(then);
        bx.ret(bx.const_i32(0));

        // Type indicator for the exception being thrown.
        //
        // The value is a pointer to the exception object
        // being thrown.
        bx.switch_to_block(catch);
        bx.set_personality_fn(bx.eh_personality());

        let eh_pointer_builtin = bx.cx.context.get_target_builtin_function("__builtin_eh_pointer");
        let zero = bx.cx.context.new_rvalue_zero(bx.int_type);
        let ptr = bx.cx.context.new_call(None, eh_pointer_builtin, &[zero]);
        let catch_ty = bx.type_func(&[bx.type_i8p(), bx.type_i8p()], bx.type_void());
        bx.call(catch_ty, None, catch_func, &[data, ptr], None);
        bx.ret(bx.const_i32(1));

        // NOTE: the blocks must be filled before adding the try/catch, otherwise gcc will not
        // generate a try/catch.
        // FIXME(antoyo): add a check in the libgccjit API to prevent this.
        bx.switch_to_block(current_block);
        bx.invoke(try_func_ty, None, try_func, &[data], then, catch, None);
    });

    let func = unsafe { std::mem::transmute(func) };

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llty, None, func, &[try_func, data, catch_func], None);
    let i32_align = bx.tcx().data_layout.i32_align.abi;
    bx.store(ret, dest, i32_align);
}


// Helper function used to get a handle to the `__rust_try` function used to
// catch exceptions.
//
// This function is only generated once and is then cached.
#[cfg(feature="master")]
fn get_rust_try_fn<'a, 'gcc, 'tcx>(cx: &'a CodegenCx<'gcc, 'tcx>, codegen: &mut dyn FnMut(Builder<'a, 'gcc, 'tcx>)) -> (Type<'gcc>, Function<'gcc>) {
    if let Some(llfn) = cx.rust_try_fn.get() {
        return llfn;
    }

    // Define the type up front for the signature of the rust_try function.
    let tcx = cx.tcx;
    let i8p = tcx.mk_mut_ptr(tcx.types.i8);
    // `unsafe fn(*mut i8) -> ()`
    let try_fn_ty = tcx.mk_fn_ptr(ty::Binder::dummy(tcx.mk_fn_sig(
        iter::once(i8p),
        tcx.mk_unit(),
        false,
        rustc_hir::Unsafety::Unsafe,
        Abi::Rust,
    )));
    // `unsafe fn(*mut i8, *mut i8) -> ()`
    let catch_fn_ty = tcx.mk_fn_ptr(ty::Binder::dummy(tcx.mk_fn_sig(
        [i8p, i8p].iter().cloned(),
        tcx.mk_unit(),
        false,
        rustc_hir::Unsafety::Unsafe,
        Abi::Rust,
    )));
    // `unsafe fn(unsafe fn(*mut i8) -> (), *mut i8, unsafe fn(*mut i8, *mut i8) -> ()) -> i32`
    let rust_fn_sig = ty::Binder::dummy(cx.tcx.mk_fn_sig(
        [try_fn_ty, i8p, catch_fn_ty],
        tcx.types.i32,
        false,
        rustc_hir::Unsafety::Unsafe,
        Abi::Rust,
    ));
    let rust_try = gen_fn(cx, "__rust_try", rust_fn_sig, codegen);
    cx.rust_try_fn.set(Some(rust_try));
    rust_try
}

// Helper function to give a Block to a closure to codegen a shim function.
// This is currently primarily used for the `try` intrinsic functions above.
#[cfg(feature="master")]
fn gen_fn<'a, 'gcc, 'tcx>(cx: &'a CodegenCx<'gcc, 'tcx>, name: &str, rust_fn_sig: ty::PolyFnSig<'tcx>, codegen: &mut dyn FnMut(Builder<'a, 'gcc, 'tcx>)) -> (Type<'gcc>, Function<'gcc>) {
    let fn_abi = cx.fn_abi_of_fn_ptr(rust_fn_sig, ty::List::empty());
    let (typ, _, _, _) = fn_abi.gcc_type(cx);
    // FIXME(eddyb) find a nicer way to do this.
    cx.linkage.set(FunctionType::Internal);
    let func = cx.declare_fn(name, fn_abi);
    let func_val = unsafe { std::mem::transmute(func) };
    cx.set_frame_pointer_type(func_val);
    cx.apply_target_cpu_attr(func_val);
    let block = Builder::append_block(cx, func_val, "entry-block");
    let bx = Builder::build(cx, block);
    codegen(bx);
    (typ, func)
}
