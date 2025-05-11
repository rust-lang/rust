pub mod llvm;
mod simd;

#[cfg(feature = "master")]
use std::iter;

#[cfg(feature = "master")]
use gccjit::FunctionType;
use gccjit::{ComparisonOp, Function, RValue, ToRValue, Type, UnaryOp};
#[cfg(feature = "master")]
use rustc_abi::ExternAbi;
use rustc_abi::{BackendRepr, HasDataLayout};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::base::wants_msvc_seh;
use rustc_codegen_ssa::common::IntPredicate;
use rustc_codegen_ssa::errors::InvalidMonomorphization;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
#[cfg(feature = "master")]
use rustc_codegen_ssa::traits::MiscCodegenMethods;
use rustc_codegen_ssa::traits::{
    ArgAbiBuilderMethods, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    IntrinsicCallBuilderMethods,
};
use rustc_middle::bug;
#[cfg(feature = "master")]
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt};
use rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Instance, Ty};
use rustc_span::{Span, Symbol, sym};
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::PanicStrategy;

#[cfg(feature = "master")]
use crate::abi::FnAbiGccExt;
use crate::abi::GccType;
use crate::builder::Builder;
use crate::common::{SignType, TypeReflection};
use crate::context::CodegenCx;
use crate::intrinsic::simd::generic_simd_intrinsic;
use crate::type_of::LayoutGccExt;

fn get_simple_intrinsic<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    name: Symbol,
) -> Option<Function<'gcc>> {
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
        // FIXME: calling `fma` from libc without FMA target feature uses expensive software emulation
        sym::fmuladdf32 => "fmaf", // TODO: use gcc intrinsic analogous to llvm.fmuladd.f32
        sym::fmuladdf64 => "fma",  // TODO: use gcc intrinsic analogous to llvm.fmuladd.f64
        sym::fabsf32 => "fabsf",
        sym::fabsf64 => "fabs",
        sym::minnumf32 => "fminf",
        sym::minnumf64 => "fmin",
        sym::minimumf32 => "fminimumf",
        sym::minimumf64 => "fminimum",
        sym::minimumf128 => {
            // GCC doesn't have the intrinsic we want so we use the compiler-builtins one
            // https://docs.rs/compiler_builtins/latest/compiler_builtins/math/full_availability/fn.fminimumf128.html
            let f128_type = cx.type_f128();
            return Some(cx.context.new_function(
                None,
                FunctionType::Extern,
                f128_type,
                &[
                    cx.context.new_parameter(None, f128_type, "a"),
                    cx.context.new_parameter(None, f128_type, "b"),
                ],
                "fminimumf128",
                false,
            ));
        }
        sym::maxnumf32 => "fmaxf",
        sym::maxnumf64 => "fmax",
        sym::maximumf32 => "fmaximumf",
        sym::maximumf64 => "fmaximum",
        sym::maximumf128 => {
            // GCC doesn't have the intrinsic we want so we use the compiler-builtins one
            // https://docs.rs/compiler_builtins/latest/compiler_builtins/math/full_availability/fn.fmaximumf128.html
            let f128_type = cx.type_f128();
            return Some(cx.context.new_function(
                None,
                FunctionType::Extern,
                f128_type,
                &[
                    cx.context.new_parameter(None, f128_type, "a"),
                    cx.context.new_parameter(None, f128_type, "b"),
                ],
                "fmaximumf128",
                false,
            ));
        }
        sym::copysignf32 => "copysignf",
        sym::copysignf64 => "copysign",
        sym::copysignf128 => "copysignl",
        sym::floorf32 => "floorf",
        sym::floorf64 => "floor",
        sym::ceilf32 => "ceilf",
        sym::ceilf64 => "ceil",
        sym::truncf32 => "truncf",
        sym::truncf64 => "trunc",
        // We match the LLVM backend and lower this to `rint`.
        sym::round_ties_even_f32 => "rintf",
        sym::round_ties_even_f64 => "rint",
        sym::roundf32 => "roundf",
        sym::roundf64 => "round",
        sym::abort => "abort",
        _ => return None,
    };
    Some(cx.context.get_builtin_function(gcc_name))
}

impl<'a, 'gcc, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, RValue<'gcc>>],
        llresult: RValue<'gcc>,
        span: Span,
    ) -> Result<(), Instance<'tcx>> {
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, self.typing_env());

        let (def_id, fn_args) = match *callee_ty.kind() {
            ty::FnDef(def_id, fn_args) => (def_id, fn_args),
            _ => bug!("expected fn item type, found {}", callee_ty),
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(self.typing_env(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = tcx.item_name(def_id);
        let name_str = name.as_str();

        let llret_ty = self.layout_of(ret_ty).gcc_type(self);
        let result = PlaceRef::new_sized(llresult, fn_abi.ret.layout);

        let simple = get_simple_intrinsic(self, name);

        // FIXME(tempdragon): Re-enable `clippy::suspicious_else_formatting` if the following issue is solved:
        // https://github.com/rust-lang/rust-clippy/issues/12497
        // and leave `else if use_integer_compare` to be placed "as is".
        #[allow(clippy::suspicious_else_formatting)]
        let value = match name {
            _ if simple.is_some() => {
                let func = simple.expect("simple function");
                self.cx.context.new_call(
                    self.location,
                    func,
                    &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                )
            }
            sym::fmaf16 => {
                // TODO(antoyo): use the correct builtin for f16.
                let func = self.cx.context.get_builtin_function("fmaf");
                let args: Vec<_> = args
                    .iter()
                    .map(|arg| {
                        self.cx.context.new_cast(self.location, arg.immediate(), self.cx.type_f32())
                    })
                    .collect();
                let result = self.cx.context.new_call(self.location, func, &args);
                self.cx.context.new_cast(self.location, result, self.cx.type_f16())
            }
            sym::is_val_statically_known => {
                let a = args[0].immediate();
                let builtin = self.context.get_builtin_function("__builtin_constant_p");
                let res = self.context.new_call(None, builtin, &[a]);
                self.icmp(IntPredicate::IntEQ, res, self.const_i32(0))
            }
            sym::catch_unwind => {
                try_intrinsic(
                    self,
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                    llresult,
                );
                return Ok(());
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
                let tp_ty = fn_args.type_at(0);
                let ptr = args[0].immediate();
                let layout = self.layout_of(tp_ty);
                let load = if let PassMode::Cast { cast: ref ty, pad_i32: _ } = fn_abi.ret.mode {
                    let gcc_ty = ty.gcc_type(self);
                    self.volatile_load(gcc_ty, ptr)
                } else {
                    self.volatile_load(layout.gcc_type(self), ptr)
                };
                // TODO(antoyo): set alignment.
                if let BackendRepr::Scalar(scalar) = layout.backend_repr {
                    self.to_immediate_scalar(load, scalar)
                } else {
                    load
                }
            }
            sym::volatile_store => {
                let dst = args[0].deref(self.cx());
                args[1].val.volatile_store(self, dst);
                return Ok(());
            }
            sym::unaligned_volatile_store => {
                let dst = args[0].deref(self.cx());
                args[1].val.unaligned_volatile_store(self, dst);
                return Ok(());
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
                            let result = func.new_local(None, self.u32_type, "zeros");
                            let zero = self.cx.gcc_zero(arg.get_type());
                            let cond = self.gcc_icmp(IntPredicate::IntEQ, arg, zero);
                            self.llbb().end_with_conditional(None, cond, then_block, else_block);

                            let zero_result = self.cx.gcc_uint(self.u32_type, width);
                            then_block.add_assignment(None, result, zero_result);
                            then_block.end_with_jump(None, after_block);

                            // NOTE: since jumps were added in a place
                            // count_leading_zeroes() does not expect, the current block
                            // in the state need to be updated.
                            self.switch_to_block(else_block);

                            let zeros = match name {
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
                        sym::ctlz_nonzero => self.count_leading_zeroes(width, args[0].immediate()),
                        sym::cttz_nonzero => self.count_trailing_zeroes(width, args[0].immediate()),
                        sym::ctpop => self.pop_count(args[0].immediate()),
                        sym::bswap => {
                            if width == 8 {
                                args[0].immediate() // byte swap a u8/i8 is just a no-op
                            } else {
                                self.gcc_bswap(args[0].immediate(), width)
                            }
                        }
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
                            } else {
                                self.rotate_right(val, raw_shift, width)
                            }
                        }
                        sym::saturating_add => self.saturating_add(
                            args[0].immediate(),
                            args[1].immediate(),
                            signed,
                            width,
                        ),
                        sym::saturating_sub => self.saturating_sub(
                            args[0].immediate(),
                            args[1].immediate(),
                            signed,
                            width,
                        ),
                        _ => bug!(),
                    },
                    None => {
                        tcx.dcx().emit_err(InvalidMonomorphization::BasicIntegerType {
                            span,
                            name,
                            ty,
                        });
                        return Ok(());
                    }
                }
            }

            sym::raw_eq => {
                use rustc_abi::BackendRepr::*;
                let tp_ty = fn_args.type_at(0);
                let layout = self.layout_of(tp_ty).layout;
                let _use_integer_compare = match layout.backend_repr() {
                    Scalar(_) | ScalarPair(_, _) => true,
                    SimdVector { .. } => false,
                    Memory { .. } => {
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
                    let n = self.context.new_cast(
                        None,
                        self.const_usize(layout.size().bytes()),
                        self.sizet_type,
                    );
                    let builtin = self.context.get_builtin_function("memcmp");
                    let cmp = self.context.new_call(None, builtin, &[a_ptr, b_ptr, n]);
                    self.icmp(IntPredicate::IntEQ, cmp, self.const_i32(0))
                }
            }

            sym::compare_bytes => {
                let a = args[0].immediate();
                let b = args[1].immediate();
                let n = args[2].immediate();

                let void_ptr_type = self.context.new_type::<*const ()>();
                let a_ptr = self.bitcast(a, void_ptr_type);
                let b_ptr = self.bitcast(b, void_ptr_type);

                // Here we assume that the `memcmp` provided by the target is a NOP for size 0.
                let builtin = self.context.get_builtin_function("memcmp");
                let cmp = self.context.new_call(None, builtin, &[a_ptr, b_ptr, n]);
                self.sext(cmp, self.type_ix(32))
            }

            sym::black_box => {
                args[0].val.store(self, result);

                let block = self.llbb();
                let extended_asm = block.add_extended_asm(None, "");
                extended_asm.add_input_operand(None, "r", result.val.llval);
                extended_asm.add_clobber("memory");
                extended_asm.set_volatile_flag(true);

                // We have copied the value to `result` already.
                return Ok(());
            }

            sym::ptr_mask => {
                let usize_type = self.context.new_type::<usize>();
                let void_ptr_type = self.context.new_type::<*const ()>();

                let ptr = args[0].immediate();
                let mask = args[1].immediate();

                let addr = self.bitcast(ptr, usize_type);
                let masked = self.and(addr, mask);
                self.bitcast(masked, void_ptr_type)
            }

            _ if name_str.starts_with("simd_") => {
                match generic_simd_intrinsic(self, name, callee_ty, args, ret_ty, llret_ty, span) {
                    Ok(value) => value,
                    Err(()) => return Ok(()),
                }
            }

            // Fall back to default body
            _ => return Err(Instance::new_raw(instance.def_id(), instance.args)),
        };

        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast { cast: ref ty, .. } = fn_abi.ret.mode {
                let ptr_llty = self.type_ptr_to(ty.gcc_type(self));
                let ptr = self.pointercast(result.val.llval, ptr_llty);
                self.store(value, ptr, result.val.align);
            } else {
                OperandRef::from_immediate_or_packed_pair(self, value, result.layout)
                    .val
                    .store(self, result);
            }
        }
        Ok(())
    }

    fn abort(&mut self) {
        let func = self.context.get_builtin_function("abort");
        let func: RValue<'gcc> = unsafe { std::mem::transmute(func) };
        self.call(self.type_void(), None, None, func, &[], None, None);
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

impl<'a, 'gcc, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        arg_abi.store_fn_arg(self, idx, dst)
    }

    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: RValue<'gcc>,
        dst: PlaceRef<'tcx, RValue<'gcc>>,
    ) {
        arg_abi.store(self, val, dst)
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> Type<'gcc> {
        arg_abi.memory_ty(self)
    }
}

pub trait ArgAbiExt<'gcc, 'tcx> {
    fn memory_ty(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    fn store(
        &self,
        bx: &mut Builder<'_, 'gcc, 'tcx>,
        val: RValue<'gcc>,
        dst: PlaceRef<'tcx, RValue<'gcc>>,
    );
    fn store_fn_arg(
        &self,
        bx: &mut Builder<'_, 'gcc, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, RValue<'gcc>>,
    );
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
    fn store(
        &self,
        bx: &mut Builder<'_, 'gcc, 'tcx>,
        val: RValue<'gcc>,
        dst: PlaceRef<'tcx, RValue<'gcc>>,
    ) {
        if self.is_ignore() {
            return;
        }
        if self.is_sized_indirect() {
            OperandValue::Ref(PlaceValue::new_sized(val, self.layout.align.abi)).store(bx, dst)
        } else if self.is_unsized_indirect() {
            bug!("unsized `ArgAbi` must be handled through `store_fn_arg`");
        } else if let PassMode::Cast { ref cast, .. } = self.mode {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_ptr_llty = bx.type_ptr_to(cast.gcc_type(bx));
                let cast_dst = bx.pointercast(dst.val.llval, cast_ptr_llty);
                bx.store(val, cast_dst, self.layout.align.abi);
            } else {
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
                let llscratch = bx.alloca(scratch_size, scratch_align);
                bx.lifetime_start(llscratch, scratch_size);

                // ... where we first store the value...
                bx.store(val, llscratch, scratch_align);

                // ... and then memcpy it to the intended destination.
                bx.memcpy(
                    dst.val.llval,
                    self.layout.align.abi,
                    llscratch,
                    scratch_align,
                    bx.const_usize(self.layout.size.bytes()),
                    MemFlags::empty(),
                );

                bx.lifetime_end(llscratch, scratch_size);
            }
        } else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    fn store_fn_arg<'a>(
        &self,
        bx: &mut Builder<'a, 'gcc, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, RValue<'gcc>>,
    ) {
        let mut next = || {
            let val = bx.current_func().get_param(*idx as i32);
            *idx += 1;
            val.to_rvalue()
        };
        match self.mode {
            PassMode::Ignore => {}
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Indirect { meta_attrs: Some(_), .. } => {
                let place_val = PlaceValue {
                    llval: next(),
                    llextra: Some(next()),
                    align: self.layout.align.abi,
                };
                OperandValue::Ref(place_val).store(bx, dst);
            }
            PassMode::Direct(_)
            | PassMode::Indirect { meta_attrs: None, .. }
            | PassMode::Cast { .. } => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}

fn int_type_width_signed<'gcc, 'tcx>(
    ty: Ty<'tcx>,
    cx: &CodegenCx<'gcc, 'tcx>,
) -> Option<(u64, bool)> {
    match *ty.kind() {
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
            if result_type.is_signed(self.cx) { self.gcc_int_cast(value, typ) } else { value };

        let context = &self.cx.context;
        let result = match width {
            8 | 16 | 32 | 64 => {
                let mask = ((1u128 << width) - 1) as u64;
                let (m0, m1, m2) = if width > 16 {
                    (
                        context.new_rvalue_from_long(typ, (0x5555555555555555u64 & mask) as i64),
                        context.new_rvalue_from_long(typ, (0x3333333333333333u64 & mask) as i64),
                        context.new_rvalue_from_long(typ, (0x0f0f0f0f0f0f0f0fu64 & mask) as i64),
                    )
                } else {
                    (
                        context.new_rvalue_from_int(typ, (0x5555u64 & mask) as i32),
                        context.new_rvalue_from_int(typ, (0x3333u64 & mask) as i32),
                        context.new_rvalue_from_int(typ, (0x0f0fu64 & mask) as i32),
                    )
                };
                let one = context.new_rvalue_from_int(typ, 1);
                let two = context.new_rvalue_from_int(typ, 2);
                let four = context.new_rvalue_from_int(typ, 4);

                // First step.
                let left = self.lshr(value, one);
                let left = self.and(left, m0);
                let right = self.and(value, m0);
                let right = self.shl(right, one);
                let step1 = self.or(left, right);

                // Second step.
                let left = self.lshr(step1, two);
                let left = self.and(left, m1);
                let right = self.and(step1, m1);
                let right = self.shl(right, two);
                let step2 = self.or(left, right);

                // Third step.
                let left = self.lshr(step2, four);
                let left = self.and(left, m2);
                let right = self.and(step2, m2);
                let right = self.shl(right, four);
                let step3 = self.or(left, right);

                // Fourth step.
                if width == 8 { step3 } else { self.gcc_bswap(step3, width) }
            }
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

                self.gcc_or(new_low, new_high, self.location)
            }
            _ => {
                panic!("cannot bit reverse with width = {}", width);
            }
        };

        self.gcc_int_cast(result, result_type)
    }

    fn count_leading_zeroes(&mut self, width: u64, arg: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): use width?
        let arg_type = arg.get_type();
        let result_type = self.u32_type;
        let count_leading_zeroes =
            // TODO(antoyo): write a new function Type::is_compatible_with(&Type) and use it here
            // instead of using is_uint().
            if arg_type.is_uint(self.cx) {
                "__builtin_clz"
            }
            else if arg_type.is_ulong(self.cx) {
                "__builtin_clzl"
            }
            else if arg_type.is_ulonglong(self.cx) {
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
                    .add_assignment(self.location, first_elem, first_value);

                let second_elem = self.context.new_array_access(self.location, result, one);
                let cast = self.gcc_int_cast(self.context.new_call(self.location, clzll, &[low]), arg_type);
                let second_value = self.add(cast, sixty_four);
                self.llbb()
                    .add_assignment(self.location, second_elem, second_value);

                let third_elem = self.context.new_array_access(self.location, result, two);
                let third_value = self.const_uint(arg_type, 128);
                self.llbb()
                    .add_assignment(self.location, third_elem, third_value);

                let not_high = self.context.new_unary_op(self.location, UnaryOp::LogicalNegate, self.u64_type, high);
                let not_low = self.context.new_unary_op(self.location, UnaryOp::LogicalNegate, self.u64_type, low);
                let not_low_and_not_high = not_low & not_high;
                let index = not_high + not_low_and_not_high;
                // NOTE: the following cast is necessary to avoid a GIMPLE verification failure in
                // gcc.
                // TODO(antoyo): do the correct verification in libgccjit to avoid an error at the
                // compilation stage.
                let index = self.context.new_cast(self.location, index, self.i32_type);

                let res = self.context.new_array_access(self.location, result, index);

                return self.gcc_int_cast(res.to_rvalue(), result_type);
            }
            else {
                let count_leading_zeroes = self.context.get_builtin_function("__builtin_clzll");
                let arg = self.context.new_cast(self.location, arg, self.ulonglong_type);
                let diff = self.ulonglong_type.get_size() as i64 - arg_type.get_size() as i64;
                let diff = self.context.new_rvalue_from_long(self.int_type, diff * 8);
                let res = self.context.new_call(self.location, count_leading_zeroes, &[arg]) - diff;
                return self.context.new_cast(self.location, res, result_type);
            };
        let count_leading_zeroes = self.context.get_builtin_function(count_leading_zeroes);
        let res = self.context.new_call(self.location, count_leading_zeroes, &[arg]);
        self.context.new_cast(self.location, res, result_type)
    }

    fn count_trailing_zeroes(&mut self, _width: u64, arg: RValue<'gcc>) -> RValue<'gcc> {
        let arg_type = arg.get_type();
        let result_type = self.u32_type;
        let arg = if arg_type.is_signed(self.cx) {
            let new_type = arg_type.to_unsigned(self.cx);
            self.gcc_int_cast(arg, new_type)
        } else {
            arg
        };
        let arg_type = arg.get_type();
        let (count_trailing_zeroes, expected_type) =
            // TODO(antoyo): write a new function Type::is_compatible_with(&Type) and use it here
            // instead of using is_uint().
            if arg_type.is_uchar(self.cx) || arg_type.is_ushort(self.cx) || arg_type.is_uint(self.cx) {
                // NOTE: we don't need to & 0xFF for uchar because the result is undefined on zero.
                ("__builtin_ctz", self.cx.uint_type)
            }
            else if arg_type.is_ulong(self.cx) {
                ("__builtin_ctzl", self.cx.ulong_type)
            }
            else if arg_type.is_ulonglong(self.cx) {
                ("__builtin_ctzll", self.cx.ulonglong_type)
            }
            else if arg_type.is_u128(self.cx) {
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

                let first_elem = self.context.new_array_access(self.location, result, zero);
                let first_value = self.gcc_int_cast(self.context.new_call(self.location, ctzll, &[low]), arg_type);
                self.llbb()
                    .add_assignment(self.location, first_elem, first_value);

                let second_elem = self.context.new_array_access(self.location, result, one);
                let second_value = self.gcc_add(self.gcc_int_cast(self.context.new_call(self.location, ctzll, &[high]), arg_type), sixty_four);
                self.llbb()
                    .add_assignment(self.location, second_elem, second_value);

                let third_elem = self.context.new_array_access(self.location, result, two);
                let third_value = self.gcc_int(arg_type, 128);
                self.llbb()
                    .add_assignment(self.location, third_elem, third_value);

                let not_low = self.context.new_unary_op(self.location, UnaryOp::LogicalNegate, self.u64_type, low);
                let not_high = self.context.new_unary_op(self.location, UnaryOp::LogicalNegate, self.u64_type, high);
                let not_low_and_not_high = not_low & not_high;
                let index = not_low + not_low_and_not_high;
                // NOTE: the following cast is necessary to avoid a GIMPLE verification failure in
                // gcc.
                // TODO(antoyo): do the correct verification in libgccjit to avoid an error at the
                // compilation stage.
                let index = self.context.new_cast(self.location, index, self.i32_type);

                let res = self.context.new_array_access(self.location, result, index);

                return self.gcc_int_cast(res.to_rvalue(), result_type);
            }
            else {
                let count_trailing_zeroes = self.context.get_builtin_function("__builtin_ctzll");
                let arg_size = arg_type.get_size();
                let casted_arg = self.context.new_cast(self.location, arg, self.ulonglong_type);
                let byte_diff = self.ulonglong_type.get_size() as i64 - arg_size as i64;
                let diff = self.context.new_rvalue_from_long(self.int_type, byte_diff * 8);
                let mask = self.context.new_rvalue_from_long(arg_type, -1); // To get the value with all bits set.
                let masked = mask & self.context.new_unary_op(self.location, UnaryOp::BitwiseNegate, arg_type, arg);
                let cond = self.context.new_comparison(self.location, ComparisonOp::Equals, masked, mask);
                let diff = diff * self.context.new_cast(self.location, cond, self.int_type);
                let res = self.context.new_call(self.location, count_trailing_zeroes, &[casted_arg]) - diff;
                return self.context.new_cast(self.location, res, result_type);
            };
        let count_trailing_zeroes = self.context.get_builtin_function(count_trailing_zeroes);
        let arg = if arg_type != expected_type {
            self.context.new_cast(self.location, arg, expected_type)
        } else {
            arg
        };
        let res = self.context.new_call(self.location, count_trailing_zeroes, &[arg]);
        self.context.new_cast(self.location, res, result_type)
    }

    fn pop_count(&mut self, value: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): use the optimized version with fewer operations.
        let result_type = self.u32_type;
        let arg_type = value.get_type();
        let value_type = arg_type.to_unsigned(self.cx);

        let value =
            if arg_type.is_signed(self.cx) { self.gcc_int_cast(value, value_type) } else { value };

        // only break apart 128-bit ints if they're not natively supported
        // TODO(antoyo): remove this if/when native 128-bit integers land in libgccjit
        if value_type.is_u128(self.cx) && !self.cx.supports_128bit_integers {
            let sixty_four = self.gcc_int(value_type, 64);
            let right_shift = self.gcc_lshr(value, sixty_four);
            let high = self.gcc_int_cast(right_shift, self.cx.ulonglong_type);
            let high = self.pop_count(high);
            let low = self.gcc_int_cast(value, self.cx.ulonglong_type);
            let low = self.pop_count(low);
            let res = high + low;
            return self.gcc_int_cast(res, result_type);
        }

        // Use Wenger's algorithm for population count, gcc's seems to play better with it
        // for (int counter = 0; value != 0; counter++) {
        //     value &= value - 1;
        // }
        let func = self.current_func.borrow().expect("func");
        let loop_head = func.new_block("head");
        let loop_body = func.new_block("body");
        let loop_tail = func.new_block("tail");

        let counter_type = self.int_type;
        let counter = self.current_func().new_local(None, counter_type, "popcount_counter");
        let val = self.current_func().new_local(None, value_type, "popcount_value");
        let zero = self.gcc_zero(counter_type);
        self.llbb().add_assignment(self.location, counter, zero);
        self.llbb().add_assignment(self.location, val, value);
        self.br(loop_head);

        // check if value isn't zero
        self.switch_to_block(loop_head);
        let zero = self.gcc_zero(value_type);
        let cond = self.gcc_icmp(IntPredicate::IntNE, val.to_rvalue(), zero);
        self.cond_br(cond, loop_body, loop_tail);

        // val &= val - 1;
        self.switch_to_block(loop_body);
        let one = self.gcc_int(value_type, 1);
        let sub = self.gcc_sub(val.to_rvalue(), one);
        let op = self.gcc_and(val.to_rvalue(), sub);
        loop_body.add_assignment(self.location, val, op);

        // counter += 1
        let one = self.gcc_int(counter_type, 1);
        let op = self.gcc_add(counter.to_rvalue(), one);
        loop_body.add_assignment(self.location, counter, op);
        self.br(loop_head);

        // end of loop
        self.switch_to_block(loop_tail);
        self.gcc_int_cast(counter.to_rvalue(), result_type)
    }

    // Algorithm from: https://blog.regehr.org/archives/1063
    fn rotate_left(
        &mut self,
        value: RValue<'gcc>,
        shift: RValue<'gcc>,
        width: u64,
    ) -> RValue<'gcc> {
        let max = self.const_uint(shift.get_type(), width);
        let shift = self.urem(shift, max);
        let lhs = self.shl(value, shift);
        let result_neg = self.neg(shift);
        let result_and = self.and(result_neg, self.const_uint(shift.get_type(), width - 1));
        let rhs = self.lshr(value, result_and);
        self.or(lhs, rhs)
    }

    // Algorithm from: https://blog.regehr.org/archives/1063
    fn rotate_right(
        &mut self,
        value: RValue<'gcc>,
        shift: RValue<'gcc>,
        width: u64,
    ) -> RValue<'gcc> {
        let max = self.const_uint(shift.get_type(), width);
        let shift = self.urem(shift, max);
        let lhs = self.lshr(value, shift);
        let result_neg = self.neg(shift);
        let result_and = self.and(result_neg, self.const_uint(shift.get_type(), width - 1));
        let rhs = self.shl(value, result_and);
        self.or(lhs, rhs)
    }

    fn saturating_add(
        &mut self,
        lhs: RValue<'gcc>,
        rhs: RValue<'gcc>,
        signed: bool,
        width: u64,
    ) -> RValue<'gcc> {
        let result_type = lhs.get_type();
        if signed {
            // Based on algorithm from: https://stackoverflow.com/a/56531252/389119
            let func = self.current_func.borrow().expect("func");
            let res = func.new_local(self.location, result_type, "saturating_sum");
            let supports_native_type = self.is_native_int_type(result_type);
            let overflow = if supports_native_type {
                let func_name = match width {
                    8 => "__builtin_add_overflow",
                    16 => "__builtin_add_overflow",
                    32 => "__builtin_sadd_overflow",
                    64 => "__builtin_saddll_overflow",
                    128 => "__builtin_add_overflow",
                    _ => unreachable!(),
                };
                let overflow_func = self.context.get_builtin_function(func_name);
                self.overflow_call(overflow_func, &[lhs, rhs, res.get_address(self.location)], None)
            } else {
                let func_name = match width {
                    128 => "__rust_i128_addo",
                    _ => unreachable!(),
                };
                let (int_result, overflow) =
                    self.operation_with_overflow(func_name, lhs, rhs, width);
                self.llbb().add_assignment(self.location, res, int_result);
                overflow
            };

            let then_block = func.new_block("then");
            let after_block = func.new_block("after");

            // Return `result_type`'s maximum or minimum value on overflow
            // NOTE: convert the type to unsigned to have an unsigned shift.
            let unsigned_type = result_type.to_unsigned(self.cx);
            let shifted = self.gcc_lshr(
                self.gcc_int_cast(lhs, unsigned_type),
                self.gcc_int(unsigned_type, width as i64 - 1),
            );
            let uint_max = self.gcc_not(self.gcc_int(unsigned_type, 0));
            let int_max = self.gcc_lshr(uint_max, self.gcc_int(unsigned_type, 1));
            then_block.add_assignment(
                self.location,
                res,
                self.gcc_int_cast(self.gcc_add(shifted, int_max), result_type),
            );
            then_block.end_with_jump(self.location, after_block);

            self.llbb().end_with_conditional(self.location, overflow, then_block, after_block);

            // NOTE: since jumps were added in a place rustc does not
            // expect, the current block in the state need to be updated.
            self.switch_to_block(after_block);

            res.to_rvalue()
        } else {
            // Algorithm from: http://locklessinc.com/articles/sat_arithmetic/
            let res = self.gcc_add(lhs, rhs);
            let cond = self.gcc_icmp(IntPredicate::IntULT, res, lhs);
            let value = self.gcc_neg(self.gcc_int_cast(cond, result_type));
            self.gcc_or(res, value, self.location)
        }
    }

    // Algorithm from: https://locklessinc.com/articles/sat_arithmetic/
    fn saturating_sub(
        &mut self,
        lhs: RValue<'gcc>,
        rhs: RValue<'gcc>,
        signed: bool,
        width: u64,
    ) -> RValue<'gcc> {
        let result_type = lhs.get_type();
        if signed {
            // Based on algorithm from: https://stackoverflow.com/a/56531252/389119
            let func = self.current_func.borrow().expect("func");
            let res = func.new_local(self.location, result_type, "saturating_diff");
            let supports_native_type = self.is_native_int_type(result_type);
            let overflow = if supports_native_type {
                let func_name = match width {
                    8 => "__builtin_sub_overflow",
                    16 => "__builtin_sub_overflow",
                    32 => "__builtin_ssub_overflow",
                    64 => "__builtin_ssubll_overflow",
                    128 => "__builtin_sub_overflow",
                    _ => unreachable!(),
                };
                let overflow_func = self.context.get_builtin_function(func_name);
                self.overflow_call(overflow_func, &[lhs, rhs, res.get_address(self.location)], None)
            } else {
                let func_name = match width {
                    128 => "__rust_i128_subo",
                    _ => unreachable!(),
                };
                let (int_result, overflow) =
                    self.operation_with_overflow(func_name, lhs, rhs, width);
                self.llbb().add_assignment(self.location, res, int_result);
                overflow
            };

            let then_block = func.new_block("then");
            let after_block = func.new_block("after");

            // Return `result_type`'s maximum or minimum value on overflow
            // NOTE: convert the type to unsigned to have an unsigned shift.
            let unsigned_type = result_type.to_unsigned(self.cx);
            let shifted = self.gcc_lshr(
                self.gcc_int_cast(lhs, unsigned_type),
                self.gcc_int(unsigned_type, width as i64 - 1),
            );
            let uint_max = self.gcc_not(self.gcc_int(unsigned_type, 0));
            let int_max = self.gcc_lshr(uint_max, self.gcc_int(unsigned_type, 1));
            then_block.add_assignment(
                self.location,
                res,
                self.gcc_int_cast(self.gcc_add(shifted, int_max), result_type),
            );
            then_block.end_with_jump(self.location, after_block);

            self.llbb().end_with_conditional(self.location, overflow, then_block, after_block);

            // NOTE: since jumps were added in a place rustc does not
            // expect, the current block in the state need to be updated.
            self.switch_to_block(after_block);

            res.to_rvalue()
        } else {
            let res = self.gcc_sub(lhs, rhs);
            let comparison = self.gcc_icmp(IntPredicate::IntULE, res, lhs);
            let value = self.gcc_neg(self.gcc_int_cast(comparison, result_type));
            self.gcc_and(res, value)
        }
    }
}

fn try_intrinsic<'a, 'b, 'gcc, 'tcx>(
    bx: &'b mut Builder<'a, 'gcc, 'tcx>,
    try_func: RValue<'gcc>,
    data: RValue<'gcc>,
    _catch_func: RValue<'gcc>,
    dest: RValue<'gcc>,
) {
    if bx.sess().panic_strategy() == PanicStrategy::Abort {
        bx.call(bx.type_void(), None, None, try_func, &[data], None, None);
        // Return 0 unconditionally from the intrinsic call;
        // we can never unwind.
        let ret_align = bx.tcx.data_layout.i32_align.abi;
        bx.store(bx.const_i32(0), dest, ret_align);
    } else {
        if wants_msvc_seh(bx.sess()) {
            unimplemented!();
        }
        #[cfg(feature = "master")]
        codegen_gnu_try(bx, try_func, data, _catch_func, dest);
        #[cfg(not(feature = "master"))]
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
#[cfg(feature = "master")]
fn codegen_gnu_try<'gcc>(
    bx: &mut Builder<'_, 'gcc, '_>,
    try_func: RValue<'gcc>,
    data: RValue<'gcc>,
    catch_func: RValue<'gcc>,
    dest: RValue<'gcc>,
) {
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

        let current_block = bx.block;

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
        bx.call(catch_ty, None, None, catch_func, &[data, ptr], None, None);
        bx.ret(bx.const_i32(1));

        // NOTE: the blocks must be filled before adding the try/catch, otherwise gcc will not
        // generate a try/catch.
        // FIXME(antoyo): add a check in the libgccjit API to prevent this.
        bx.switch_to_block(current_block);
        bx.invoke(try_func_ty, None, None, try_func, &[data], then, catch, None, None);
    });

    let func = unsafe { std::mem::transmute::<Function<'gcc>, RValue<'gcc>>(func) };

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llty, None, None, func, &[try_func, data, catch_func], None, None);
    let i32_align = bx.tcx().data_layout.i32_align.abi;
    bx.store(ret, dest, i32_align);
}

// Helper function used to get a handle to the `__rust_try` function used to
// catch exceptions.
//
// This function is only generated once and is then cached.
#[cfg(feature = "master")]
fn get_rust_try_fn<'a, 'gcc, 'tcx>(
    cx: &'a CodegenCx<'gcc, 'tcx>,
    codegen: &mut dyn FnMut(Builder<'a, 'gcc, 'tcx>),
) -> (Type<'gcc>, Function<'gcc>) {
    if let Some(llfn) = cx.rust_try_fn.get() {
        return llfn;
    }

    // Define the type up front for the signature of the rust_try function.
    let tcx = cx.tcx;
    let i8p = Ty::new_mut_ptr(tcx, tcx.types.i8);
    // `unsafe fn(*mut i8) -> ()`
    let try_fn_ty = Ty::new_fn_ptr(
        tcx,
        ty::Binder::dummy(tcx.mk_fn_sig(
            iter::once(i8p),
            tcx.types.unit,
            false,
            rustc_hir::Safety::Unsafe,
            ExternAbi::Rust,
        )),
    );
    // `unsafe fn(*mut i8, *mut i8) -> ()`
    let catch_fn_ty = Ty::new_fn_ptr(
        tcx,
        ty::Binder::dummy(tcx.mk_fn_sig(
            [i8p, i8p].iter().cloned(),
            tcx.types.unit,
            false,
            rustc_hir::Safety::Unsafe,
            ExternAbi::Rust,
        )),
    );
    // `unsafe fn(unsafe fn(*mut i8) -> (), *mut i8, unsafe fn(*mut i8, *mut i8) -> ()) -> i32`
    let rust_fn_sig = ty::Binder::dummy(cx.tcx.mk_fn_sig(
        [try_fn_ty, i8p, catch_fn_ty],
        tcx.types.i32,
        false,
        rustc_hir::Safety::Unsafe,
        ExternAbi::Rust,
    ));
    let rust_try = gen_fn(cx, "__rust_try", rust_fn_sig, codegen);
    cx.rust_try_fn.set(Some(rust_try));
    rust_try
}

// Helper function to give a Block to a closure to codegen a shim function.
// This is currently primarily used for the `try` intrinsic functions above.
#[cfg(feature = "master")]
fn gen_fn<'a, 'gcc, 'tcx>(
    cx: &'a CodegenCx<'gcc, 'tcx>,
    name: &str,
    rust_fn_sig: ty::PolyFnSig<'tcx>,
    codegen: &mut dyn FnMut(Builder<'a, 'gcc, 'tcx>),
) -> (Type<'gcc>, Function<'gcc>) {
    let fn_abi = cx.fn_abi_of_fn_ptr(rust_fn_sig, ty::List::empty());
    let return_type = fn_abi.gcc_type(cx).return_type;
    // FIXME(eddyb) find a nicer way to do this.
    cx.linkage.set(FunctionType::Internal);
    let func = cx.declare_fn(name, fn_abi);
    let func_val = unsafe { std::mem::transmute::<Function<'gcc>, RValue<'gcc>>(func) };
    cx.set_frame_pointer_type(func_val);
    cx.apply_target_cpu_attr(func_val);
    let block = Builder::append_block(cx, func_val, "entry-block");
    let bx = Builder::build(cx, block);
    codegen(bx);
    (return_type, func)
}
