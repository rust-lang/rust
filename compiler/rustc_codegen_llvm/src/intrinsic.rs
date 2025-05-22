use std::assert_matches::assert_matches;
use std::cmp::Ordering;

use rustc_abi::{Align, BackendRepr, ExternAbi, Float, HasDataLayout, Primitive, Size};
use rustc_codegen_ssa::base::{compare_simd_types, wants_msvc_seh, wants_wasm_eh};
use rustc_codegen_ssa::common::{IntPredicate, TypeKind};
use rustc_codegen_ssa::errors::{ExpectedPointerMutability, InvalidMonomorphization};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::traits::*;
use rustc_hir as hir;
use rustc_middle::mir::BinOp;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, GenericArgsRef, Ty};
use rustc_middle::{bug, span_bug};
use rustc_span::{Span, Symbol, sym};
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_target::spec::{HasTargetSpec, PanicStrategy};
use tracing::debug;

use crate::abi::FnAbiLlvmExt;
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::llvm::{self, Metadata};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::va_arg::emit_va_arg;
use crate::value::Value;

fn get_simple_intrinsic<'ll>(
    cx: &CodegenCx<'ll, '_>,
    name: Symbol,
) -> Option<(&'ll Type, &'ll Value)> {
    let llvm_name = match name {
        sym::sqrtf16 => "llvm.sqrt.f16",
        sym::sqrtf32 => "llvm.sqrt.f32",
        sym::sqrtf64 => "llvm.sqrt.f64",
        sym::sqrtf128 => "llvm.sqrt.f128",

        sym::powif16 => "llvm.powi.f16.i32",
        sym::powif32 => "llvm.powi.f32.i32",
        sym::powif64 => "llvm.powi.f64.i32",
        sym::powif128 => "llvm.powi.f128.i32",

        sym::sinf16 => "llvm.sin.f16",
        sym::sinf32 => "llvm.sin.f32",
        sym::sinf64 => "llvm.sin.f64",
        sym::sinf128 => "llvm.sin.f128",

        sym::cosf16 => "llvm.cos.f16",
        sym::cosf32 => "llvm.cos.f32",
        sym::cosf64 => "llvm.cos.f64",
        sym::cosf128 => "llvm.cos.f128",

        sym::powf16 => "llvm.pow.f16",
        sym::powf32 => "llvm.pow.f32",
        sym::powf64 => "llvm.pow.f64",
        sym::powf128 => "llvm.pow.f128",

        sym::expf16 => "llvm.exp.f16",
        sym::expf32 => "llvm.exp.f32",
        sym::expf64 => "llvm.exp.f64",
        sym::expf128 => "llvm.exp.f128",

        sym::exp2f16 => "llvm.exp2.f16",
        sym::exp2f32 => "llvm.exp2.f32",
        sym::exp2f64 => "llvm.exp2.f64",
        sym::exp2f128 => "llvm.exp2.f128",

        sym::logf16 => "llvm.log.f16",
        sym::logf32 => "llvm.log.f32",
        sym::logf64 => "llvm.log.f64",
        sym::logf128 => "llvm.log.f128",

        sym::log10f16 => "llvm.log10.f16",
        sym::log10f32 => "llvm.log10.f32",
        sym::log10f64 => "llvm.log10.f64",
        sym::log10f128 => "llvm.log10.f128",

        sym::log2f16 => "llvm.log2.f16",
        sym::log2f32 => "llvm.log2.f32",
        sym::log2f64 => "llvm.log2.f64",
        sym::log2f128 => "llvm.log2.f128",

        sym::fmaf16 => "llvm.fma.f16",
        sym::fmaf32 => "llvm.fma.f32",
        sym::fmaf64 => "llvm.fma.f64",
        sym::fmaf128 => "llvm.fma.f128",

        sym::fmuladdf16 => "llvm.fmuladd.f16",
        sym::fmuladdf32 => "llvm.fmuladd.f32",
        sym::fmuladdf64 => "llvm.fmuladd.f64",
        sym::fmuladdf128 => "llvm.fmuladd.f128",

        sym::fabsf16 => "llvm.fabs.f16",
        sym::fabsf32 => "llvm.fabs.f32",
        sym::fabsf64 => "llvm.fabs.f64",
        sym::fabsf128 => "llvm.fabs.f128",

        sym::minnumf16 => "llvm.minnum.f16",
        sym::minnumf32 => "llvm.minnum.f32",
        sym::minnumf64 => "llvm.minnum.f64",
        sym::minnumf128 => "llvm.minnum.f128",

        sym::minimumf16 => "llvm.minimum.f16",
        sym::minimumf32 => "llvm.minimum.f32",
        sym::minimumf64 => "llvm.minimum.f64",
        // There are issues on x86_64 and aarch64 with the f128 variant,
        // let's instead use the instrinsic fallback body.
        // sym::minimumf128 => "llvm.minimum.f128",
        sym::maxnumf16 => "llvm.maxnum.f16",
        sym::maxnumf32 => "llvm.maxnum.f32",
        sym::maxnumf64 => "llvm.maxnum.f64",
        sym::maxnumf128 => "llvm.maxnum.f128",

        sym::maximumf16 => "llvm.maximum.f16",
        sym::maximumf32 => "llvm.maximum.f32",
        sym::maximumf64 => "llvm.maximum.f64",
        // There are issues on x86_64 and aarch64 with the f128 variant,
        // let's instead use the instrinsic fallback body.
        // sym::maximumf128 => "llvm.maximum.f128",
        sym::copysignf16 => "llvm.copysign.f16",
        sym::copysignf32 => "llvm.copysign.f32",
        sym::copysignf64 => "llvm.copysign.f64",
        sym::copysignf128 => "llvm.copysign.f128",

        sym::floorf16 => "llvm.floor.f16",
        sym::floorf32 => "llvm.floor.f32",
        sym::floorf64 => "llvm.floor.f64",
        sym::floorf128 => "llvm.floor.f128",

        sym::ceilf16 => "llvm.ceil.f16",
        sym::ceilf32 => "llvm.ceil.f32",
        sym::ceilf64 => "llvm.ceil.f64",
        sym::ceilf128 => "llvm.ceil.f128",

        sym::truncf16 => "llvm.trunc.f16",
        sym::truncf32 => "llvm.trunc.f32",
        sym::truncf64 => "llvm.trunc.f64",
        sym::truncf128 => "llvm.trunc.f128",

        // We could use any of `rint`, `nearbyint`, or `roundeven`
        // for this -- they are all identical in semantics when
        // assuming the default FP environment.
        // `rint` is what we used for $forever.
        sym::round_ties_even_f16 => "llvm.rint.f16",
        sym::round_ties_even_f32 => "llvm.rint.f32",
        sym::round_ties_even_f64 => "llvm.rint.f64",
        sym::round_ties_even_f128 => "llvm.rint.f128",

        sym::roundf16 => "llvm.round.f16",
        sym::roundf32 => "llvm.round.f32",
        sym::roundf64 => "llvm.round.f64",
        sym::roundf128 => "llvm.round.f128",

        sym::ptr_mask => "llvm.ptrmask",

        _ => return None,
    };
    Some(cx.get_intrinsic(llvm_name))
}

impl<'ll, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, &'ll Value>],
        result: PlaceRef<'tcx, &'ll Value>,
        span: Span,
    ) -> Result<(), ty::Instance<'tcx>> {
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, self.typing_env());

        let ty::FnDef(def_id, fn_args) = *callee_ty.kind() else {
            bug!("expected fn item type, found {}", callee_ty);
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(self.typing_env(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = tcx.item_name(def_id);

        let llret_ty = self.layout_of(ret_ty).llvm_type(self);

        let simple = get_simple_intrinsic(self, name);
        let llval = match name {
            _ if simple.is_some() => {
                let (simple_ty, simple_fn) = simple.unwrap();
                self.call(
                    simple_ty,
                    None,
                    None,
                    simple_fn,
                    &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                    None,
                    Some(instance),
                )
            }
            sym::is_val_statically_known => {
                let intrinsic_type = args[0].layout.immediate_llvm_type(self.cx);
                let kind = self.type_kind(intrinsic_type);
                let intrinsic_name = match kind {
                    TypeKind::Pointer | TypeKind::Integer => {
                        Some(format!("llvm.is.constant.{intrinsic_type:?}"))
                    }
                    // LLVM float types' intrinsic names differ from their type names.
                    TypeKind::Half => Some(format!("llvm.is.constant.f16")),
                    TypeKind::Float => Some(format!("llvm.is.constant.f32")),
                    TypeKind::Double => Some(format!("llvm.is.constant.f64")),
                    TypeKind::FP128 => Some(format!("llvm.is.constant.f128")),
                    _ => None,
                };
                if let Some(intrinsic_name) = intrinsic_name {
                    self.call_intrinsic(&intrinsic_name, &[args[0].immediate()])
                } else {
                    self.const_bool(false)
                }
            }
            sym::select_unpredictable => {
                let cond = args[0].immediate();
                assert_eq!(args[1].layout, args[2].layout);
                let select = |bx: &mut Self, true_val, false_val| {
                    let result = bx.select(cond, true_val, false_val);
                    bx.set_unpredictable(&result);
                    result
                };
                match (args[1].val, args[2].val) {
                    (OperandValue::Ref(true_val), OperandValue::Ref(false_val)) => {
                        assert!(true_val.llextra.is_none());
                        assert!(false_val.llextra.is_none());
                        assert_eq!(true_val.align, false_val.align);
                        let ptr = select(self, true_val.llval, false_val.llval);
                        let selected =
                            OperandValue::Ref(PlaceValue::new_sized(ptr, true_val.align));
                        selected.store(self, result);
                        return Ok(());
                    }
                    (OperandValue::Immediate(_), OperandValue::Immediate(_))
                    | (OperandValue::Pair(_, _), OperandValue::Pair(_, _)) => {
                        let true_val = args[1].immediate_or_packed_pair(self);
                        let false_val = args[2].immediate_or_packed_pair(self);
                        select(self, true_val, false_val)
                    }
                    (OperandValue::ZeroSized, OperandValue::ZeroSized) => return Ok(()),
                    _ => span_bug!(span, "Incompatible OperandValue for select_unpredictable"),
                }
            }
            sym::catch_unwind => {
                catch_unwind_intrinsic(
                    self,
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                    result,
                );
                return Ok(());
            }
            sym::breakpoint => self.call_intrinsic("llvm.debugtrap", &[]),
            sym::va_copy => {
                self.call_intrinsic("llvm.va_copy", &[args[0].immediate(), args[1].immediate()])
            }
            sym::va_arg => {
                match result.layout.backend_repr {
                    BackendRepr::Scalar(scalar) => {
                        match scalar.primitive() {
                            Primitive::Int(..) => {
                                if self.cx().size_of(ret_ty).bytes() < 4 {
                                    // `va_arg` should not be called on an integer type
                                    // less than 4 bytes in length. If it is, promote
                                    // the integer to an `i32` and truncate the result
                                    // back to the smaller type.
                                    let promoted_result = emit_va_arg(self, args[0], tcx.types.i32);
                                    self.trunc(promoted_result, llret_ty)
                                } else {
                                    emit_va_arg(self, args[0], ret_ty)
                                }
                            }
                            Primitive::Float(Float::F16) => {
                                bug!("the va_arg intrinsic does not work with `f16`")
                            }
                            Primitive::Float(Float::F64) | Primitive::Pointer(_) => {
                                emit_va_arg(self, args[0], ret_ty)
                            }
                            // `va_arg` should never be used with the return type f32.
                            Primitive::Float(Float::F32) => {
                                bug!("the va_arg intrinsic does not work with `f32`")
                            }
                            Primitive::Float(Float::F128) => {
                                bug!("the va_arg intrinsic does not work with `f128`")
                            }
                        }
                    }
                    _ => bug!("the va_arg intrinsic does not work with non-scalar types"),
                }
            }

            sym::volatile_load | sym::unaligned_volatile_load => {
                let ptr = args[0].immediate();
                let load = self.volatile_load(result.layout.llvm_type(self), ptr);
                let align = if name == sym::unaligned_volatile_load {
                    1
                } else {
                    result.layout.align.abi.bytes() as u32
                };
                unsafe {
                    llvm::LLVMSetAlignment(load, align);
                }
                if !result.layout.is_zst() {
                    self.store_to_place(load, result.val);
                }
                return Ok(());
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
                let (rw, cache_type) = match name {
                    sym::prefetch_read_data => (0, 1),
                    sym::prefetch_write_data => (1, 1),
                    sym::prefetch_read_instruction => (0, 0),
                    sym::prefetch_write_instruction => (1, 0),
                    _ => bug!(),
                };
                self.call_intrinsic(
                    "llvm.prefetch",
                    &[
                        args[0].immediate(),
                        self.const_i32(rw),
                        args[1].immediate(),
                        self.const_i32(cache_type),
                    ],
                )
            }
            sym::carrying_mul_add => {
                let (size, signed) = fn_args.type_at(0).int_size_and_signed(self.tcx);

                let wide_llty = self.type_ix(size.bits() * 2);
                let args = args.as_array().unwrap();
                let [a, b, c, d] = args.map(|a| self.intcast(a.immediate(), wide_llty, signed));

                let wide = if signed {
                    let prod = self.unchecked_smul(a, b);
                    let acc = self.unchecked_sadd(prod, c);
                    self.unchecked_sadd(acc, d)
                } else {
                    let prod = self.unchecked_umul(a, b);
                    let acc = self.unchecked_uadd(prod, c);
                    self.unchecked_uadd(acc, d)
                };

                let narrow_llty = self.type_ix(size.bits());
                let low = self.trunc(wide, narrow_llty);
                let bits_const = self.const_uint(wide_llty, size.bits());
                // No need for ashr when signed; LLVM changes it to lshr anyway.
                let high = self.lshr(wide, bits_const);
                // FIXME: could be `trunc nuw`, even for signed.
                let high = self.trunc(high, narrow_llty);

                let pair_llty = self.type_struct(&[narrow_llty, narrow_llty], false);
                let pair = self.const_poison(pair_llty);
                let pair = self.insert_value(pair, low, 0);
                let pair = self.insert_value(pair, high, 1);
                pair
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
                if !ty.is_integral() {
                    tcx.dcx().emit_err(InvalidMonomorphization::BasicIntegerType {
                        span,
                        name,
                        ty,
                    });
                    return Ok(());
                }
                let (size, signed) = ty.int_size_and_signed(self.tcx);
                let width = size.bits();
                match name {
                    sym::ctlz | sym::cttz => {
                        let y = self.const_bool(false);
                        let ret = self.call_intrinsic(
                            &format!("llvm.{name}.i{width}"),
                            &[args[0].immediate(), y],
                        );

                        self.intcast(ret, llret_ty, false)
                    }
                    sym::ctlz_nonzero => {
                        let y = self.const_bool(true);
                        let llvm_name = &format!("llvm.ctlz.i{width}");
                        let ret = self.call_intrinsic(llvm_name, &[args[0].immediate(), y]);
                        self.intcast(ret, llret_ty, false)
                    }
                    sym::cttz_nonzero => {
                        let y = self.const_bool(true);
                        let llvm_name = &format!("llvm.cttz.i{width}");
                        let ret = self.call_intrinsic(llvm_name, &[args[0].immediate(), y]);
                        self.intcast(ret, llret_ty, false)
                    }
                    sym::ctpop => {
                        let ret = self.call_intrinsic(
                            &format!("llvm.ctpop.i{width}"),
                            &[args[0].immediate()],
                        );
                        self.intcast(ret, llret_ty, false)
                    }
                    sym::bswap => {
                        if width == 8 {
                            args[0].immediate() // byte swap a u8/i8 is just a no-op
                        } else {
                            self.call_intrinsic(
                                &format!("llvm.bswap.i{width}"),
                                &[args[0].immediate()],
                            )
                        }
                    }
                    sym::bitreverse => self.call_intrinsic(
                        &format!("llvm.bitreverse.i{width}"),
                        &[args[0].immediate()],
                    ),
                    sym::rotate_left | sym::rotate_right => {
                        let is_left = name == sym::rotate_left;
                        let val = args[0].immediate();
                        let raw_shift = args[1].immediate();
                        // rotate = funnel shift with first two args the same
                        let llvm_name =
                            &format!("llvm.fsh{}.i{}", if is_left { 'l' } else { 'r' }, width);

                        // llvm expects shift to be the same type as the values, but rust
                        // always uses `u32`.
                        let raw_shift = self.intcast(raw_shift, self.val_ty(val), false);

                        self.call_intrinsic(llvm_name, &[val, val, raw_shift])
                    }
                    sym::saturating_add | sym::saturating_sub => {
                        let is_add = name == sym::saturating_add;
                        let lhs = args[0].immediate();
                        let rhs = args[1].immediate();
                        let llvm_name = &format!(
                            "llvm.{}{}.sat.i{}",
                            if signed { 's' } else { 'u' },
                            if is_add { "add" } else { "sub" },
                            width
                        );
                        self.call_intrinsic(llvm_name, &[lhs, rhs])
                    }
                    _ => bug!(),
                }
            }

            sym::raw_eq => {
                use BackendRepr::*;
                let tp_ty = fn_args.type_at(0);
                let layout = self.layout_of(tp_ty).layout;
                let use_integer_compare = match layout.backend_repr() {
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
                } else if use_integer_compare {
                    let integer_ty = self.type_ix(layout.size().bits());
                    let a_val = self.load(integer_ty, a, layout.align().abi);
                    let b_val = self.load(integer_ty, b, layout.align().abi);
                    self.icmp(IntPredicate::IntEQ, a_val, b_val)
                } else {
                    let n = self.const_usize(layout.size().bytes());
                    let cmp = self.call_intrinsic("memcmp", &[a, b, n]);
                    match self.cx.sess().target.arch.as_ref() {
                        "avr" | "msp430" => self.icmp(IntPredicate::IntEQ, cmp, self.const_i16(0)),
                        _ => self.icmp(IntPredicate::IntEQ, cmp, self.const_i32(0)),
                    }
                }
            }

            sym::compare_bytes => {
                // Here we assume that the `memcmp` provided by the target is a NOP for size 0.
                let cmp = self.call_intrinsic(
                    "memcmp",
                    &[args[0].immediate(), args[1].immediate(), args[2].immediate()],
                );
                // Some targets have `memcmp` returning `i16`, but the intrinsic is always `i32`.
                self.sext(cmp, self.type_ix(32))
            }

            sym::black_box => {
                args[0].val.store(self, result);
                let result_val_span = [result.val.llval];
                // We need to "use" the argument in some way LLVM can't introspect, and on
                // targets that support it we can typically leverage inline assembly to do
                // this. LLVM's interpretation of inline assembly is that it's, well, a black
                // box. This isn't the greatest implementation since it probably deoptimizes
                // more than we want, but it's so far good enough.
                //
                // For zero-sized types, the location pointed to by the result may be
                // uninitialized. Do not "use" the result in this case; instead just clobber
                // the memory.
                let (constraint, inputs): (&str, &[_]) = if result.layout.is_zst() {
                    ("~{memory}", &[])
                } else {
                    ("r,~{memory}", &result_val_span)
                };
                crate::asm::inline_asm_call(
                    self,
                    "",
                    constraint,
                    inputs,
                    self.type_void(),
                    &[],
                    true,
                    false,
                    llvm::AsmDialect::Att,
                    &[span],
                    false,
                    None,
                    None,
                )
                .unwrap_or_else(|| bug!("failed to generate inline asm call for `black_box`"));

                // We have copied the value to `result` already.
                return Ok(());
            }

            _ if name.as_str().starts_with("simd_") => {
                // Unpack non-power-of-2 #[repr(packed, simd)] arguments.
                // This gives them the expected layout of a regular #[repr(simd)] vector.
                let mut loaded_args = Vec::new();
                for (ty, arg) in arg_tys.iter().zip(args) {
                    loaded_args.push(
                        // #[repr(packed, simd)] vectors are passed like arrays (as references,
                        // with reduced alignment and no padding) rather than as immediates.
                        // We can use a vector load to fix the layout and turn the argument
                        // into an immediate.
                        if ty.is_simd()
                            && let OperandValue::Ref(place) = arg.val
                        {
                            let (size, elem_ty) = ty.simd_size_and_type(self.tcx());
                            let elem_ll_ty = match elem_ty.kind() {
                                ty::Float(f) => self.type_float_from_ty(*f),
                                ty::Int(i) => self.type_int_from_ty(*i),
                                ty::Uint(u) => self.type_uint_from_ty(*u),
                                ty::RawPtr(_, _) => self.type_ptr(),
                                _ => unreachable!(),
                            };
                            let loaded =
                                self.load_from_place(self.type_vector(elem_ll_ty, size), place);
                            OperandRef::from_immediate_or_packed_pair(self, loaded, arg.layout)
                        } else {
                            *arg
                        },
                    );
                }

                let llret_ty = if ret_ty.is_simd()
                    && let BackendRepr::Memory { .. } = self.layout_of(ret_ty).layout.backend_repr
                {
                    let (size, elem_ty) = ret_ty.simd_size_and_type(self.tcx());
                    let elem_ll_ty = match elem_ty.kind() {
                        ty::Float(f) => self.type_float_from_ty(*f),
                        ty::Int(i) => self.type_int_from_ty(*i),
                        ty::Uint(u) => self.type_uint_from_ty(*u),
                        ty::RawPtr(_, _) => self.type_ptr(),
                        _ => unreachable!(),
                    };
                    self.type_vector(elem_ll_ty, size)
                } else {
                    llret_ty
                };

                match generic_simd_intrinsic(
                    self,
                    name,
                    callee_ty,
                    fn_args,
                    &loaded_args,
                    ret_ty,
                    llret_ty,
                    span,
                ) {
                    Ok(llval) => llval,
                    // If there was an error, just skip this invocation... we'll abort compilation
                    // anyway, but we can keep codegen'ing to find more errors.
                    Err(()) => return Ok(()),
                }
            }

            _ => {
                debug!("unknown intrinsic '{}' -- falling back to default body", name);
                // Call the fallback body instead of generating the intrinsic code
                return Err(ty::Instance::new_raw(instance.def_id(), instance.args));
            }
        };

        if result.layout.ty.is_bool() {
            OperandRef::from_immediate_or_packed_pair(self, llval, result.layout)
                .val
                .store(self, result);
        } else if !result.layout.ty.is_unit() {
            self.store_to_place(llval, result.val);
        }
        Ok(())
    }

    fn abort(&mut self) {
        self.call_intrinsic("llvm.trap", &[]);
    }

    fn assume(&mut self, val: Self::Value) {
        if self.cx.sess().opts.optimize != rustc_session::config::OptLevel::No {
            self.call_intrinsic("llvm.assume", &[val]);
        }
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        if self.cx.sess().opts.optimize != rustc_session::config::OptLevel::No {
            self.call_intrinsic("llvm.expect.i1", &[cond, self.const_bool(expected)])
        } else {
            cond
        }
    }

    fn type_test(&mut self, pointer: Self::Value, typeid: Self::Metadata) -> Self::Value {
        // Test the called operand using llvm.type.test intrinsic. The LowerTypeTests link-time
        // optimization pass replaces calls to this intrinsic with code to test type membership.
        let typeid = self.get_metadata_value(typeid);
        self.call_intrinsic("llvm.type.test", &[pointer, typeid])
    }

    fn type_checked_load(
        &mut self,
        llvtable: &'ll Value,
        vtable_byte_offset: u64,
        typeid: &'ll Metadata,
    ) -> Self::Value {
        let typeid = self.get_metadata_value(typeid);
        let vtable_byte_offset = self.const_i32(vtable_byte_offset as i32);
        let type_checked_load =
            self.call_intrinsic("llvm.type.checked.load", &[llvtable, vtable_byte_offset, typeid]);
        self.extract_value(type_checked_load, 0)
    }

    fn va_start(&mut self, va_list: &'ll Value) -> &'ll Value {
        self.call_intrinsic("llvm.va_start", &[va_list])
    }

    fn va_end(&mut self, va_list: &'ll Value) -> &'ll Value {
        self.call_intrinsic("llvm.va_end", &[va_list])
    }
}

fn catch_unwind_intrinsic<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    try_func: &'ll Value,
    data: &'ll Value,
    catch_func: &'ll Value,
    dest: PlaceRef<'tcx, &'ll Value>,
) {
    if bx.sess().panic_strategy() == PanicStrategy::Abort {
        let try_func_ty = bx.type_func(&[bx.type_ptr()], bx.type_void());
        bx.call(try_func_ty, None, None, try_func, &[data], None, None);
        // Return 0 unconditionally from the intrinsic call;
        // we can never unwind.
        OperandValue::Immediate(bx.const_i32(0)).store(bx, dest);
    } else if wants_msvc_seh(bx.sess()) {
        codegen_msvc_try(bx, try_func, data, catch_func, dest);
    } else if wants_wasm_eh(bx.sess()) {
        codegen_wasm_try(bx, try_func, data, catch_func, dest);
    } else if bx.sess().target.os == "emscripten" {
        codegen_emcc_try(bx, try_func, data, catch_func, dest);
    } else {
        codegen_gnu_try(bx, try_func, data, catch_func, dest);
    }
}

// MSVC's definition of the `rust_try` function.
//
// This implementation uses the new exception handling instructions in LLVM
// which have support in LLVM for SEH on MSVC targets. Although these
// instructions are meant to work for all targets, as of the time of this
// writing, however, LLVM does not recommend the usage of these new instructions
// as the old ones are still more optimized.
fn codegen_msvc_try<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    try_func: &'ll Value,
    data: &'ll Value,
    catch_func: &'ll Value,
    dest: PlaceRef<'tcx, &'ll Value>,
) {
    let (llty, llfn) = get_rust_try_fn(bx, &mut |mut bx| {
        bx.set_personality_fn(bx.eh_personality());

        let normal = bx.append_sibling_block("normal");
        let catchswitch = bx.append_sibling_block("catchswitch");
        let catchpad_rust = bx.append_sibling_block("catchpad_rust");
        let catchpad_foreign = bx.append_sibling_block("catchpad_foreign");
        let caught = bx.append_sibling_block("caught");

        let try_func = llvm::get_param(bx.llfn(), 0);
        let data = llvm::get_param(bx.llfn(), 1);
        let catch_func = llvm::get_param(bx.llfn(), 2);

        // We're generating an IR snippet that looks like:
        //
        //   declare i32 @rust_try(%try_func, %data, %catch_func) {
        //      %slot = alloca i8*
        //      invoke %try_func(%data) to label %normal unwind label %catchswitch
        //
        //   normal:
        //      ret i32 0
        //
        //   catchswitch:
        //      %cs = catchswitch within none [%catchpad_rust, %catchpad_foreign] unwind to caller
        //
        //   catchpad_rust:
        //      %tok = catchpad within %cs [%type_descriptor, 8, %slot]
        //      %ptr = load %slot
        //      call %catch_func(%data, %ptr)
        //      catchret from %tok to label %caught
        //
        //   catchpad_foreign:
        //      %tok = catchpad within %cs [null, 64, null]
        //      call %catch_func(%data, null)
        //      catchret from %tok to label %caught
        //
        //   caught:
        //      ret i32 1
        //   }
        //
        // This structure follows the basic usage of throw/try/catch in LLVM.
        // For example, compile this C++ snippet to see what LLVM generates:
        //
        //      struct rust_panic {
        //          rust_panic(const rust_panic&);
        //          ~rust_panic();
        //
        //          void* x[2];
        //      };
        //
        //      int __rust_try(
        //          void (*try_func)(void*),
        //          void *data,
        //          void (*catch_func)(void*, void*) noexcept
        //      ) {
        //          try {
        //              try_func(data);
        //              return 0;
        //          } catch(rust_panic& a) {
        //              catch_func(data, &a);
        //              return 1;
        //          } catch(...) {
        //              catch_func(data, NULL);
        //              return 1;
        //          }
        //      }
        //
        // More information can be found in libstd's seh.rs implementation.
        let ptr_size = bx.tcx().data_layout.pointer_size;
        let ptr_align = bx.tcx().data_layout.pointer_align.abi;
        let slot = bx.alloca(ptr_size, ptr_align);
        let try_func_ty = bx.type_func(&[bx.type_ptr()], bx.type_void());
        bx.invoke(try_func_ty, None, None, try_func, &[data], normal, catchswitch, None, None);

        bx.switch_to_block(normal);
        bx.ret(bx.const_i32(0));

        bx.switch_to_block(catchswitch);
        let cs = bx.catch_switch(None, None, &[catchpad_rust, catchpad_foreign]);

        // We can't use the TypeDescriptor defined in libpanic_unwind because it
        // might be in another DLL and the SEH encoding only supports specifying
        // a TypeDescriptor from the current module.
        //
        // However this isn't an issue since the MSVC runtime uses string
        // comparison on the type name to match TypeDescriptors rather than
        // pointer equality.
        //
        // So instead we generate a new TypeDescriptor in each module that uses
        // `try` and let the linker merge duplicate definitions in the same
        // module.
        //
        // When modifying, make sure that the type_name string exactly matches
        // the one used in library/panic_unwind/src/seh.rs.
        let type_info_vtable = bx.declare_global("??_7type_info@@6B@", bx.type_ptr());
        let type_name = bx.const_bytes(b"rust_panic\0");
        let type_info =
            bx.const_struct(&[type_info_vtable, bx.const_null(bx.type_ptr()), type_name], false);
        let tydesc = bx.declare_global(
            &mangle_internal_symbol(bx.tcx, "__rust_panic_type_info"),
            bx.val_ty(type_info),
        );

        llvm::set_linkage(tydesc, llvm::Linkage::LinkOnceODRLinkage);
        if bx.cx.tcx.sess.target.supports_comdat() {
            llvm::SetUniqueComdat(bx.llmod, tydesc);
        }
        llvm::set_initializer(tydesc, type_info);

        // The flag value of 8 indicates that we are catching the exception by
        // reference instead of by value. We can't use catch by value because
        // that requires copying the exception object, which we don't support
        // since our exception object effectively contains a Box.
        //
        // Source: MicrosoftCXXABI::getAddrOfCXXCatchHandlerType in clang
        bx.switch_to_block(catchpad_rust);
        let flags = bx.const_i32(8);
        let funclet = bx.catch_pad(cs, &[tydesc, flags, slot]);
        let ptr = bx.load(bx.type_ptr(), slot, ptr_align);
        let catch_ty = bx.type_func(&[bx.type_ptr(), bx.type_ptr()], bx.type_void());
        bx.call(catch_ty, None, None, catch_func, &[data, ptr], Some(&funclet), None);
        bx.catch_ret(&funclet, caught);

        // The flag value of 64 indicates a "catch-all".
        bx.switch_to_block(catchpad_foreign);
        let flags = bx.const_i32(64);
        let null = bx.const_null(bx.type_ptr());
        let funclet = bx.catch_pad(cs, &[null, flags, null]);
        bx.call(catch_ty, None, None, catch_func, &[data, null], Some(&funclet), None);
        bx.catch_ret(&funclet, caught);

        bx.switch_to_block(caught);
        bx.ret(bx.const_i32(1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llty, None, None, llfn, &[try_func, data, catch_func], None, None);
    OperandValue::Immediate(ret).store(bx, dest);
}

// WASM's definition of the `rust_try` function.
fn codegen_wasm_try<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    try_func: &'ll Value,
    data: &'ll Value,
    catch_func: &'ll Value,
    dest: PlaceRef<'tcx, &'ll Value>,
) {
    let (llty, llfn) = get_rust_try_fn(bx, &mut |mut bx| {
        bx.set_personality_fn(bx.eh_personality());

        let normal = bx.append_sibling_block("normal");
        let catchswitch = bx.append_sibling_block("catchswitch");
        let catchpad = bx.append_sibling_block("catchpad");
        let caught = bx.append_sibling_block("caught");

        let try_func = llvm::get_param(bx.llfn(), 0);
        let data = llvm::get_param(bx.llfn(), 1);
        let catch_func = llvm::get_param(bx.llfn(), 2);

        // We're generating an IR snippet that looks like:
        //
        //   declare i32 @rust_try(%try_func, %data, %catch_func) {
        //      %slot = alloca i8*
        //      invoke %try_func(%data) to label %normal unwind label %catchswitch
        //
        //   normal:
        //      ret i32 0
        //
        //   catchswitch:
        //      %cs = catchswitch within none [%catchpad] unwind to caller
        //
        //   catchpad:
        //      %tok = catchpad within %cs [null]
        //      %ptr = call @llvm.wasm.get.exception(token %tok)
        //      %sel = call @llvm.wasm.get.ehselector(token %tok)
        //      call %catch_func(%data, %ptr)
        //      catchret from %tok to label %caught
        //
        //   caught:
        //      ret i32 1
        //   }
        //
        let try_func_ty = bx.type_func(&[bx.type_ptr()], bx.type_void());
        bx.invoke(try_func_ty, None, None, try_func, &[data], normal, catchswitch, None, None);

        bx.switch_to_block(normal);
        bx.ret(bx.const_i32(0));

        bx.switch_to_block(catchswitch);
        let cs = bx.catch_switch(None, None, &[catchpad]);

        bx.switch_to_block(catchpad);
        let null = bx.const_null(bx.type_ptr());
        let funclet = bx.catch_pad(cs, &[null]);

        let ptr = bx.call_intrinsic("llvm.wasm.get.exception", &[funclet.cleanuppad()]);
        let _sel = bx.call_intrinsic("llvm.wasm.get.ehselector", &[funclet.cleanuppad()]);

        let catch_ty = bx.type_func(&[bx.type_ptr(), bx.type_ptr()], bx.type_void());
        bx.call(catch_ty, None, None, catch_func, &[data, ptr], Some(&funclet), None);
        bx.catch_ret(&funclet, caught);

        bx.switch_to_block(caught);
        bx.ret(bx.const_i32(1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llty, None, None, llfn, &[try_func, data, catch_func], None, None);
    OperandValue::Immediate(ret).store(bx, dest);
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
fn codegen_gnu_try<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    try_func: &'ll Value,
    data: &'ll Value,
    catch_func: &'ll Value,
    dest: PlaceRef<'tcx, &'ll Value>,
) {
    let (llty, llfn) = get_rust_try_fn(bx, &mut |mut bx| {
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

        let try_func = llvm::get_param(bx.llfn(), 0);
        let data = llvm::get_param(bx.llfn(), 1);
        let catch_func = llvm::get_param(bx.llfn(), 2);
        let try_func_ty = bx.type_func(&[bx.type_ptr()], bx.type_void());
        bx.invoke(try_func_ty, None, None, try_func, &[data], then, catch, None, None);

        bx.switch_to_block(then);
        bx.ret(bx.const_i32(0));

        // Type indicator for the exception being thrown.
        //
        // The first value in this tuple is a pointer to the exception object
        // being thrown. The second value is a "selector" indicating which of
        // the landing pad clauses the exception's type had been matched to.
        // rust_try ignores the selector.
        bx.switch_to_block(catch);
        let lpad_ty = bx.type_struct(&[bx.type_ptr(), bx.type_i32()], false);
        let vals = bx.landing_pad(lpad_ty, bx.eh_personality(), 1);
        let tydesc = bx.const_null(bx.type_ptr());
        bx.add_clause(vals, tydesc);
        let ptr = bx.extract_value(vals, 0);
        let catch_ty = bx.type_func(&[bx.type_ptr(), bx.type_ptr()], bx.type_void());
        bx.call(catch_ty, None, None, catch_func, &[data, ptr], None, None);
        bx.ret(bx.const_i32(1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llty, None, None, llfn, &[try_func, data, catch_func], None, None);
    OperandValue::Immediate(ret).store(bx, dest);
}

// Variant of codegen_gnu_try used for emscripten where Rust panics are
// implemented using C++ exceptions. Here we use exceptions of a specific type
// (`struct rust_panic`) to represent Rust panics.
fn codegen_emcc_try<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    try_func: &'ll Value,
    data: &'ll Value,
    catch_func: &'ll Value,
    dest: PlaceRef<'tcx, &'ll Value>,
) {
    let (llty, llfn) = get_rust_try_fn(bx, &mut |mut bx| {
        // Codegens the shims described above:
        //
        //   bx:
        //      invoke %try_func(%data) normal %normal unwind %catch
        //
        //   normal:
        //      ret 0
        //
        //   catch:
        //      (%ptr, %selector) = landingpad
        //      %rust_typeid = @llvm.eh.typeid.for(@_ZTI10rust_panic)
        //      %is_rust_panic = %selector == %rust_typeid
        //      %catch_data = alloca { i8*, i8 }
        //      %catch_data[0] = %ptr
        //      %catch_data[1] = %is_rust_panic
        //      call %catch_func(%data, %catch_data)
        //      ret 1
        let then = bx.append_sibling_block("then");
        let catch = bx.append_sibling_block("catch");

        let try_func = llvm::get_param(bx.llfn(), 0);
        let data = llvm::get_param(bx.llfn(), 1);
        let catch_func = llvm::get_param(bx.llfn(), 2);
        let try_func_ty = bx.type_func(&[bx.type_ptr()], bx.type_void());
        bx.invoke(try_func_ty, None, None, try_func, &[data], then, catch, None, None);

        bx.switch_to_block(then);
        bx.ret(bx.const_i32(0));

        // Type indicator for the exception being thrown.
        //
        // The first value in this tuple is a pointer to the exception object
        // being thrown. The second value is a "selector" indicating which of
        // the landing pad clauses the exception's type had been matched to.
        bx.switch_to_block(catch);
        let tydesc = bx.eh_catch_typeinfo();
        let lpad_ty = bx.type_struct(&[bx.type_ptr(), bx.type_i32()], false);
        let vals = bx.landing_pad(lpad_ty, bx.eh_personality(), 2);
        bx.add_clause(vals, tydesc);
        bx.add_clause(vals, bx.const_null(bx.type_ptr()));
        let ptr = bx.extract_value(vals, 0);
        let selector = bx.extract_value(vals, 1);

        // Check if the typeid we got is the one for a Rust panic.
        let rust_typeid = bx.call_intrinsic("llvm.eh.typeid.for", &[tydesc]);
        let is_rust_panic = bx.icmp(IntPredicate::IntEQ, selector, rust_typeid);
        let is_rust_panic = bx.zext(is_rust_panic, bx.type_bool());

        // We need to pass two values to catch_func (ptr and is_rust_panic), so
        // create an alloca and pass a pointer to that.
        let ptr_size = bx.tcx().data_layout.pointer_size;
        let ptr_align = bx.tcx().data_layout.pointer_align.abi;
        let i8_align = bx.tcx().data_layout.i8_align.abi;
        // Required in order for there to be no padding between the fields.
        assert!(i8_align <= ptr_align);
        let catch_data = bx.alloca(2 * ptr_size, ptr_align);
        bx.store(ptr, catch_data, ptr_align);
        let catch_data_1 = bx.inbounds_ptradd(catch_data, bx.const_usize(ptr_size.bytes()));
        bx.store(is_rust_panic, catch_data_1, i8_align);

        let catch_ty = bx.type_func(&[bx.type_ptr(), bx.type_ptr()], bx.type_void());
        bx.call(catch_ty, None, None, catch_func, &[data, catch_data], None, None);
        bx.ret(bx.const_i32(1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llty, None, None, llfn, &[try_func, data, catch_func], None, None);
    OperandValue::Immediate(ret).store(bx, dest);
}

// Helper function to give a Block to a closure to codegen a shim function.
// This is currently primarily used for the `try` intrinsic functions above.
fn gen_fn<'a, 'll, 'tcx>(
    cx: &'a CodegenCx<'ll, 'tcx>,
    name: &str,
    rust_fn_sig: ty::PolyFnSig<'tcx>,
    codegen: &mut dyn FnMut(Builder<'a, 'll, 'tcx>),
) -> (&'ll Type, &'ll Value) {
    let fn_abi = cx.fn_abi_of_fn_ptr(rust_fn_sig, ty::List::empty());
    let llty = fn_abi.llvm_type(cx);
    let llfn = cx.declare_fn(name, fn_abi, None);
    cx.set_frame_pointer_type(llfn);
    cx.apply_target_cpu_attr(llfn);
    // FIXME(eddyb) find a nicer way to do this.
    llvm::set_linkage(llfn, llvm::Linkage::InternalLinkage);
    let llbb = Builder::append_block(cx, llfn, "entry-block");
    let bx = Builder::build(cx, llbb);
    codegen(bx);
    (llty, llfn)
}

// Helper function used to get a handle to the `__rust_try` function used to
// catch exceptions.
//
// This function is only generated once and is then cached.
fn get_rust_try_fn<'a, 'll, 'tcx>(
    cx: &'a CodegenCx<'ll, 'tcx>,
    codegen: &mut dyn FnMut(Builder<'a, 'll, 'tcx>),
) -> (&'ll Type, &'ll Value) {
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
            [i8p],
            tcx.types.unit,
            false,
            hir::Safety::Unsafe,
            ExternAbi::Rust,
        )),
    );
    // `unsafe fn(*mut i8, *mut i8) -> ()`
    let catch_fn_ty = Ty::new_fn_ptr(
        tcx,
        ty::Binder::dummy(tcx.mk_fn_sig(
            [i8p, i8p],
            tcx.types.unit,
            false,
            hir::Safety::Unsafe,
            ExternAbi::Rust,
        )),
    );
    // `unsafe fn(unsafe fn(*mut i8) -> (), *mut i8, unsafe fn(*mut i8, *mut i8) -> ()) -> i32`
    let rust_fn_sig = ty::Binder::dummy(cx.tcx.mk_fn_sig(
        [try_fn_ty, i8p, catch_fn_ty],
        tcx.types.i32,
        false,
        hir::Safety::Unsafe,
        ExternAbi::Rust,
    ));
    let rust_try = gen_fn(cx, "__rust_try", rust_fn_sig, codegen);
    cx.rust_try_fn.set(Some(rust_try));
    rust_try
}

fn generic_simd_intrinsic<'ll, 'tcx>(
    bx: &mut Builder<'_, 'll, 'tcx>,
    name: Symbol,
    callee_ty: Ty<'tcx>,
    fn_args: GenericArgsRef<'tcx>,
    args: &[OperandRef<'tcx, &'ll Value>],
    ret_ty: Ty<'tcx>,
    llret_ty: &'ll Type,
    span: Span,
) -> Result<&'ll Value, ()> {
    macro_rules! return_error {
        ($diag: expr) => {{
            bx.sess().dcx().emit_err($diag);
            return Err(());
        }};
    }

    macro_rules! require {
        ($cond: expr, $diag: expr) => {
            if !$cond {
                return_error!($diag);
            }
        };
    }

    macro_rules! require_simd {
        ($ty: expr, $variant:ident) => {{
            require!($ty.is_simd(), InvalidMonomorphization::$variant { span, name, ty: $ty });
            $ty.simd_size_and_type(bx.tcx())
        }};
    }

    /// Returns the bitwidth of the `$ty` argument if it is an `Int` or `Uint` type.
    macro_rules! require_int_or_uint_ty {
        ($ty: expr, $diag: expr) => {
            match $ty {
                ty::Int(i) => i.bit_width().unwrap_or_else(|| bx.data_layout().pointer_size.bits()),
                ty::Uint(i) => {
                    i.bit_width().unwrap_or_else(|| bx.data_layout().pointer_size.bits())
                }
                _ => {
                    return_error!($diag);
                }
            }
        };
    }

    /// Converts a vector mask, where each element has a bit width equal to the data elements it is used with,
    /// down to an i1 based mask that can be used by llvm intrinsics.
    ///
    /// The rust simd semantics are that each element should either consist of all ones or all zeroes,
    /// but this information is not available to llvm. Truncating the vector effectively uses the lowest bit,
    /// but codegen for several targets is better if we consider the highest bit by shifting.
    ///
    /// For x86 SSE/AVX targets this is beneficial since most instructions with mask parameters only consider the highest bit.
    /// So even though on llvm level we have an additional shift, in the final assembly there is no shift or truncate and
    /// instead the mask can be used as is.
    ///
    /// For aarch64 and other targets there is a benefit because a mask from the sign bit can be more
    /// efficiently converted to an all ones / all zeroes mask by comparing whether each element is negative.
    fn vector_mask_to_bitmask<'a, 'll, 'tcx>(
        bx: &mut Builder<'a, 'll, 'tcx>,
        i_xn: &'ll Value,
        in_elem_bitwidth: u64,
        in_len: u64,
    ) -> &'ll Value {
        // Shift the MSB to the right by "in_elem_bitwidth - 1" into the first bit position.
        let shift_idx = bx.cx.const_int(bx.type_ix(in_elem_bitwidth), (in_elem_bitwidth - 1) as _);
        let shift_indices = vec![shift_idx; in_len as _];
        let i_xn_msb = bx.lshr(i_xn, bx.const_vector(shift_indices.as_slice()));
        // Truncate vector to an <i1 x N>
        bx.trunc(i_xn_msb, bx.type_vector(bx.type_i1(), in_len))
    }

    let tcx = bx.tcx();
    let sig = tcx.normalize_erasing_late_bound_regions(bx.typing_env(), callee_ty.fn_sig(tcx));
    let arg_tys = sig.inputs();

    // Sanity-check: all vector arguments must be immediates.
    if cfg!(debug_assertions) {
        for (ty, arg) in arg_tys.iter().zip(args) {
            if ty.is_simd() {
                assert_matches!(arg.val, OperandValue::Immediate(_));
            }
        }
    }

    if name == sym::simd_select_bitmask {
        let (len, _) = require_simd!(arg_tys[1], SimdArgument);

        let expected_int_bits = len.max(8).next_power_of_two();
        let expected_bytes = len.div_ceil(8);

        let mask_ty = arg_tys[0];
        let mask = match mask_ty.kind() {
            ty::Int(i) if i.bit_width() == Some(expected_int_bits) => args[0].immediate(),
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => args[0].immediate(),
            ty::Array(elem, len)
                if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len
                        .try_to_target_usize(bx.tcx)
                        .expect("expected monomorphic const in codegen")
                        == expected_bytes =>
            {
                let place = PlaceRef::alloca(bx, args[0].layout);
                args[0].val.store(bx, place);
                let int_ty = bx.type_ix(expected_bytes * 8);
                bx.load(int_ty, place.val.llval, Align::ONE)
            }
            _ => return_error!(InvalidMonomorphization::InvalidBitmask {
                span,
                name,
                mask_ty,
                expected_int_bits,
                expected_bytes
            }),
        };

        let i1 = bx.type_i1();
        let im = bx.type_ix(len);
        let i1xn = bx.type_vector(i1, len);
        let m_im = bx.trunc(mask, im);
        let m_i1s = bx.bitcast(m_im, i1xn);
        return Ok(bx.select(m_i1s, args[1].immediate(), args[2].immediate()));
    }

    // every intrinsic below takes a SIMD vector as its first argument
    let (in_len, in_elem) = require_simd!(arg_tys[0], SimdInput);
    let in_ty = arg_tys[0];

    let comparison = match name {
        sym::simd_eq => Some(BinOp::Eq),
        sym::simd_ne => Some(BinOp::Ne),
        sym::simd_lt => Some(BinOp::Lt),
        sym::simd_le => Some(BinOp::Le),
        sym::simd_gt => Some(BinOp::Gt),
        sym::simd_ge => Some(BinOp::Ge),
        _ => None,
    };

    if let Some(cmp_op) = comparison {
        let (out_len, out_ty) = require_simd!(ret_ty, SimdReturn);

        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );
        require!(
            bx.type_kind(bx.element_type(llret_ty)) == TypeKind::Integer,
            InvalidMonomorphization::ReturnIntegerType { span, name, ret_ty, out_ty }
        );

        return Ok(compare_simd_types(
            bx,
            args[0].immediate(),
            args[1].immediate(),
            in_elem,
            llret_ty,
            cmp_op,
        ));
    }

    if name == sym::simd_shuffle_const_generic {
        let idx = fn_args[2].expect_const().to_value().valtree.unwrap_branch();
        let n = idx.len() as u64;

        let (out_len, out_ty) = require_simd!(ret_ty, SimdReturn);
        require!(
            out_len == n,
            InvalidMonomorphization::ReturnLength { span, name, in_len: n, ret_ty, out_len }
        );
        require!(
            in_elem == out_ty,
            InvalidMonomorphization::ReturnElement { span, name, in_elem, in_ty, ret_ty, out_ty }
        );

        let total_len = in_len * 2;

        let indices: Option<Vec<_>> = idx
            .iter()
            .enumerate()
            .map(|(arg_idx, val)| {
                let idx = val.unwrap_leaf().to_i32();
                if idx >= i32::try_from(total_len).unwrap() {
                    bx.sess().dcx().emit_err(InvalidMonomorphization::SimdIndexOutOfBounds {
                        span,
                        name,
                        arg_idx: arg_idx as u64,
                        total_len: total_len.into(),
                    });
                    None
                } else {
                    Some(bx.const_i32(idx))
                }
            })
            .collect();
        let Some(indices) = indices else {
            return Ok(bx.const_null(llret_ty));
        };

        return Ok(bx.shuffle_vector(
            args[0].immediate(),
            args[1].immediate(),
            bx.const_vector(&indices),
        ));
    }

    if name == sym::simd_shuffle {
        // Make sure this is actually a SIMD vector.
        let idx_ty = args[2].layout.ty;
        let n: u64 = if idx_ty.is_simd()
            && matches!(idx_ty.simd_size_and_type(bx.cx.tcx).1.kind(), ty::Uint(ty::UintTy::U32))
        {
            idx_ty.simd_size_and_type(bx.cx.tcx).0
        } else {
            return_error!(InvalidMonomorphization::SimdShuffle { span, name, ty: idx_ty })
        };

        let (out_len, out_ty) = require_simd!(ret_ty, SimdReturn);
        require!(
            out_len == n,
            InvalidMonomorphization::ReturnLength { span, name, in_len: n, ret_ty, out_len }
        );
        require!(
            in_elem == out_ty,
            InvalidMonomorphization::ReturnElement { span, name, in_elem, in_ty, ret_ty, out_ty }
        );

        let total_len = u128::from(in_len) * 2;

        // Check that the indices are in-bounds.
        let indices = args[2].immediate();
        for i in 0..n {
            let val = bx.const_get_elt(indices, i as u64);
            let idx = bx
                .const_to_opt_u128(val, true)
                .unwrap_or_else(|| bug!("typeck should have already ensured that these are const"));
            if idx >= total_len {
                return_error!(InvalidMonomorphization::SimdIndexOutOfBounds {
                    span,
                    name,
                    arg_idx: i,
                    total_len,
                });
            }
        }

        return Ok(bx.shuffle_vector(args[0].immediate(), args[1].immediate(), indices));
    }

    if name == sym::simd_insert || name == sym::simd_insert_dyn {
        require!(
            in_elem == arg_tys[2],
            InvalidMonomorphization::InsertedType {
                span,
                name,
                in_elem,
                in_ty,
                out_ty: arg_tys[2]
            }
        );

        let index_imm = if name == sym::simd_insert {
            let idx = bx
                .const_to_opt_u128(args[1].immediate(), false)
                .expect("typeck should have ensure that this is a const");
            if idx >= in_len.into() {
                return_error!(InvalidMonomorphization::SimdIndexOutOfBounds {
                    span,
                    name,
                    arg_idx: 1,
                    total_len: in_len.into(),
                });
            }
            bx.const_i32(idx as i32)
        } else {
            args[1].immediate()
        };

        return Ok(bx.insert_element(args[0].immediate(), args[2].immediate(), index_imm));
    }
    if name == sym::simd_extract || name == sym::simd_extract_dyn {
        require!(
            ret_ty == in_elem,
            InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
        );
        let index_imm = if name == sym::simd_extract {
            let idx = bx
                .const_to_opt_u128(args[1].immediate(), false)
                .expect("typeck should have ensure that this is a const");
            if idx >= in_len.into() {
                return_error!(InvalidMonomorphization::SimdIndexOutOfBounds {
                    span,
                    name,
                    arg_idx: 1,
                    total_len: in_len.into(),
                });
            }
            bx.const_i32(idx as i32)
        } else {
            args[1].immediate()
        };

        return Ok(bx.extract_element(args[0].immediate(), index_imm));
    }

    if name == sym::simd_select {
        let m_elem_ty = in_elem;
        let m_len = in_len;
        let (v_len, _) = require_simd!(arg_tys[1], SimdArgument);
        require!(
            m_len == v_len,
            InvalidMonomorphization::MismatchedLengths { span, name, m_len, v_len }
        );
        let in_elem_bitwidth = require_int_or_uint_ty!(
            m_elem_ty.kind(),
            InvalidMonomorphization::MaskWrongElementType { span, name, ty: m_elem_ty }
        );
        let m_i1s = vector_mask_to_bitmask(bx, args[0].immediate(), in_elem_bitwidth, m_len);
        return Ok(bx.select(m_i1s, args[1].immediate(), args[2].immediate()));
    }

    if name == sym::simd_bitmask {
        // The `fn simd_bitmask(vector) -> unsigned integer` intrinsic takes a vector mask and
        // returns one bit for each lane (which must all be `0` or `!0`) in the form of either:
        // * an unsigned integer
        // * an array of `u8`
        // If the vector has less than 8 lanes, a u8 is returned with zeroed trailing bits.
        //
        // The bit order of the result depends on the byte endianness, LSB-first for little
        // endian and MSB-first for big endian.
        let expected_int_bits = in_len.max(8).next_power_of_two();
        let expected_bytes = in_len.div_ceil(8);

        // Integer vector <i{in_bitwidth} x in_len>:
        let in_elem_bitwidth = require_int_or_uint_ty!(
            in_elem.kind(),
            InvalidMonomorphization::MaskWrongElementType { span, name, ty: in_elem }
        );

        let i1xn = vector_mask_to_bitmask(bx, args[0].immediate(), in_elem_bitwidth, in_len);
        // Bitcast <i1 x N> to iN:
        let i_ = bx.bitcast(i1xn, bx.type_ix(in_len));

        match ret_ty.kind() {
            ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => {
                // Zero-extend iN to the bitmask type:
                return Ok(bx.zext(i_, bx.type_ix(expected_int_bits)));
            }
            ty::Array(elem, len)
                if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                    && len
                        .try_to_target_usize(bx.tcx)
                        .expect("expected monomorphic const in codegen")
                        == expected_bytes =>
            {
                // Zero-extend iN to the array length:
                let ze = bx.zext(i_, bx.type_ix(expected_bytes * 8));

                // Convert the integer to a byte array
                let ptr = bx.alloca(Size::from_bytes(expected_bytes), Align::ONE);
                bx.store(ze, ptr, Align::ONE);
                let array_ty = bx.type_array(bx.type_i8(), expected_bytes);
                return Ok(bx.load(array_ty, ptr, Align::ONE));
            }
            _ => return_error!(InvalidMonomorphization::CannotReturn {
                span,
                name,
                ret_ty,
                expected_int_bits,
                expected_bytes
            }),
        }
    }

    fn simd_simple_float_intrinsic<'ll, 'tcx>(
        name: Symbol,
        in_elem: Ty<'_>,
        in_ty: Ty<'_>,
        in_len: u64,
        bx: &mut Builder<'_, 'll, 'tcx>,
        span: Span,
        args: &[OperandRef<'tcx, &'ll Value>],
    ) -> Result<&'ll Value, ()> {
        macro_rules! return_error {
            ($diag: expr) => {{
                bx.sess().dcx().emit_err($diag);
                return Err(());
            }};
        }

        let (elem_ty_str, elem_ty) = if let ty::Float(f) = in_elem.kind() {
            let elem_ty = bx.cx.type_float_from_ty(*f);
            match f.bit_width() {
                16 => ("f16", elem_ty),
                32 => ("f32", elem_ty),
                64 => ("f64", elem_ty),
                128 => ("f128", elem_ty),
                _ => return_error!(InvalidMonomorphization::FloatingPointVector {
                    span,
                    name,
                    f_ty: *f,
                    in_ty,
                }),
            }
        } else {
            return_error!(InvalidMonomorphization::FloatingPointType { span, name, in_ty });
        };

        let vec_ty = bx.type_vector(elem_ty, in_len);

        let (intr_name, fn_ty) = match name {
            sym::simd_ceil => ("ceil", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fabs => ("fabs", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fcos => ("cos", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fexp2 => ("exp2", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fexp => ("exp", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_flog10 => ("log10", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_flog2 => ("log2", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_flog => ("log", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_floor => ("floor", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fma => ("fma", bx.type_func(&[vec_ty, vec_ty, vec_ty], vec_ty)),
            sym::simd_relaxed_fma => ("fmuladd", bx.type_func(&[vec_ty, vec_ty, vec_ty], vec_ty)),
            sym::simd_fsin => ("sin", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_fsqrt => ("sqrt", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_round => ("round", bx.type_func(&[vec_ty], vec_ty)),
            sym::simd_trunc => ("trunc", bx.type_func(&[vec_ty], vec_ty)),
            _ => return_error!(InvalidMonomorphization::UnrecognizedIntrinsic { span, name }),
        };
        let llvm_name = &format!("llvm.{intr_name}.v{in_len}{elem_ty_str}");
        let f = bx.declare_cfn(llvm_name, llvm::UnnamedAddr::No, fn_ty);
        let c = bx.call(
            fn_ty,
            None,
            None,
            f,
            &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
            None,
            None,
        );
        Ok(c)
    }

    if std::matches!(
        name,
        sym::simd_ceil
            | sym::simd_fabs
            | sym::simd_fcos
            | sym::simd_fexp2
            | sym::simd_fexp
            | sym::simd_flog10
            | sym::simd_flog2
            | sym::simd_flog
            | sym::simd_floor
            | sym::simd_fma
            | sym::simd_fsin
            | sym::simd_fsqrt
            | sym::simd_relaxed_fma
            | sym::simd_round
            | sym::simd_trunc
    ) {
        return simd_simple_float_intrinsic(name, in_elem, in_ty, in_len, bx, span, args);
    }

    // FIXME: use:
    //  https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Function.h#L182
    //  https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Intrinsics.h#L81
    fn llvm_vector_str(bx: &Builder<'_, '_, '_>, elem_ty: Ty<'_>, vec_len: u64) -> String {
        match *elem_ty.kind() {
            ty::Int(v) => format!(
                "v{}i{}",
                vec_len,
                // Normalize to prevent crash if v: IntTy::Isize
                v.normalize(bx.target_spec().pointer_width).bit_width().unwrap()
            ),
            ty::Uint(v) => format!(
                "v{}i{}",
                vec_len,
                // Normalize to prevent crash if v: UIntTy::Usize
                v.normalize(bx.target_spec().pointer_width).bit_width().unwrap()
            ),
            ty::Float(v) => format!("v{}f{}", vec_len, v.bit_width()),
            ty::RawPtr(_, _) => format!("v{}p0", vec_len),
            _ => unreachable!(),
        }
    }

    fn llvm_vector_ty<'ll>(cx: &CodegenCx<'ll, '_>, elem_ty: Ty<'_>, vec_len: u64) -> &'ll Type {
        let elem_ty = match *elem_ty.kind() {
            ty::Int(v) => cx.type_int_from_ty(v),
            ty::Uint(v) => cx.type_uint_from_ty(v),
            ty::Float(v) => cx.type_float_from_ty(v),
            ty::RawPtr(_, _) => cx.type_ptr(),
            _ => unreachable!(),
        };
        cx.type_vector(elem_ty, vec_len)
    }

    if name == sym::simd_gather {
        // simd_gather(values: <N x T>, pointers: <N x *_ T>,
        //             mask: <N x i{M}>) -> <N x T>
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (_, element_ty0) = require_simd!(in_ty, SimdFirst);
        let (out_len, element_ty1) = require_simd!(arg_tys[1], SimdSecond);
        // The element type of the third argument must be a signed integer type of any width:
        let (out_len2, element_ty2) = require_simd!(arg_tys[2], SimdThird);
        require_simd!(ret_ty, SimdReturn);

        // Of the same length:
        require!(
            in_len == out_len,
            InvalidMonomorphization::SecondArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[1],
                out_len
            }
        );
        require!(
            in_len == out_len2,
            InvalidMonomorphization::ThirdArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[2],
                out_len: out_len2
            }
        );

        // The return type must match the first argument type
        require!(
            ret_ty == in_ty,
            InvalidMonomorphization::ExpectedReturnType { span, name, in_ty, ret_ty }
        );

        require!(
            matches!(
                *element_ty1.kind(),
                ty::RawPtr(p_ty, _) if p_ty == in_elem && p_ty.kind() == element_ty0.kind()
            ),
            InvalidMonomorphization::ExpectedElementType {
                span,
                name,
                expected_element: element_ty1,
                second_arg: arg_tys[1],
                in_elem,
                in_ty,
                mutability: ExpectedPointerMutability::Not,
            }
        );

        let mask_elem_bitwidth = require_int_or_uint_ty!(
            element_ty2.kind(),
            InvalidMonomorphization::MaskWrongElementType { span, name, ty: element_ty2 }
        );

        // Alignment of T, must be a constant integer value:
        let alignment_ty = bx.type_i32();
        let alignment = bx.const_i32(bx.align_of(in_elem).bytes() as i32);

        // Truncate the mask vector to a vector of i1s:
        let mask = vector_mask_to_bitmask(bx, args[2].immediate(), mask_elem_bitwidth, in_len);
        let mask_ty = bx.type_vector(bx.type_i1(), in_len);

        // Type of the vector of pointers:
        let llvm_pointer_vec_ty = llvm_vector_ty(bx, element_ty1, in_len);
        let llvm_pointer_vec_str = llvm_vector_str(bx, element_ty1, in_len);

        // Type of the vector of elements:
        let llvm_elem_vec_ty = llvm_vector_ty(bx, element_ty0, in_len);
        let llvm_elem_vec_str = llvm_vector_str(bx, element_ty0, in_len);

        let llvm_intrinsic =
            format!("llvm.masked.gather.{llvm_elem_vec_str}.{llvm_pointer_vec_str}");
        let fn_ty = bx.type_func(
            &[llvm_pointer_vec_ty, alignment_ty, mask_ty, llvm_elem_vec_ty],
            llvm_elem_vec_ty,
        );
        let f = bx.declare_cfn(&llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
        let v = bx.call(
            fn_ty,
            None,
            None,
            f,
            &[args[1].immediate(), alignment, mask, args[0].immediate()],
            None,
            None,
        );
        return Ok(v);
    }

    if name == sym::simd_masked_load {
        // simd_masked_load(mask: <N x i{M}>, pointer: *_ T, values: <N x T>) -> <N x T>
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1
        // Loads contiguous elements from memory behind `pointer`, but only for
        // those lanes whose `mask` bit is enabled.
        // The memory addresses corresponding to the “off” lanes are not accessed.

        // The element type of the "mask" argument must be a signed integer type of any width
        let mask_ty = in_ty;
        let (mask_len, mask_elem) = (in_len, in_elem);

        // The second argument must be a pointer matching the element type
        let pointer_ty = arg_tys[1];

        // The last argument is a passthrough vector providing values for disabled lanes
        let values_ty = arg_tys[2];
        let (values_len, values_elem) = require_simd!(values_ty, SimdThird);

        require_simd!(ret_ty, SimdReturn);

        // Of the same length:
        require!(
            values_len == mask_len,
            InvalidMonomorphization::ThirdArgumentLength {
                span,
                name,
                in_len: mask_len,
                in_ty: mask_ty,
                arg_ty: values_ty,
                out_len: values_len
            }
        );

        // The return type must match the last argument type
        require!(
            ret_ty == values_ty,
            InvalidMonomorphization::ExpectedReturnType { span, name, in_ty: values_ty, ret_ty }
        );

        require!(
            matches!(
                *pointer_ty.kind(),
                ty::RawPtr(p_ty, _) if p_ty == values_elem && p_ty.kind() == values_elem.kind()
            ),
            InvalidMonomorphization::ExpectedElementType {
                span,
                name,
                expected_element: values_elem,
                second_arg: pointer_ty,
                in_elem: values_elem,
                in_ty: values_ty,
                mutability: ExpectedPointerMutability::Not,
            }
        );

        let m_elem_bitwidth = require_int_or_uint_ty!(
            mask_elem.kind(),
            InvalidMonomorphization::MaskWrongElementType { span, name, ty: mask_elem }
        );

        let mask = vector_mask_to_bitmask(bx, args[0].immediate(), m_elem_bitwidth, mask_len);
        let mask_ty = bx.type_vector(bx.type_i1(), mask_len);

        // Alignment of T, must be a constant integer value:
        let alignment_ty = bx.type_i32();
        let alignment = bx.const_i32(bx.align_of(values_elem).bytes() as i32);

        let llvm_pointer = bx.type_ptr();

        // Type of the vector of elements:
        let llvm_elem_vec_ty = llvm_vector_ty(bx, values_elem, values_len);
        let llvm_elem_vec_str = llvm_vector_str(bx, values_elem, values_len);

        let llvm_intrinsic = format!("llvm.masked.load.{llvm_elem_vec_str}.p0");
        let fn_ty = bx
            .type_func(&[llvm_pointer, alignment_ty, mask_ty, llvm_elem_vec_ty], llvm_elem_vec_ty);
        let f = bx.declare_cfn(&llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
        let v = bx.call(
            fn_ty,
            None,
            None,
            f,
            &[args[1].immediate(), alignment, mask, args[2].immediate()],
            None,
            None,
        );
        return Ok(v);
    }

    if name == sym::simd_masked_store {
        // simd_masked_store(mask: <N x i{M}>, pointer: *mut T, values: <N x T>) -> ()
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1
        // Stores contiguous elements to memory behind `pointer`, but only for
        // those lanes whose `mask` bit is enabled.
        // The memory addresses corresponding to the “off” lanes are not accessed.

        // The element type of the "mask" argument must be a signed integer type of any width
        let mask_ty = in_ty;
        let (mask_len, mask_elem) = (in_len, in_elem);

        // The second argument must be a pointer matching the element type
        let pointer_ty = arg_tys[1];

        // The last argument specifies the values to store to memory
        let values_ty = arg_tys[2];
        let (values_len, values_elem) = require_simd!(values_ty, SimdThird);

        // Of the same length:
        require!(
            values_len == mask_len,
            InvalidMonomorphization::ThirdArgumentLength {
                span,
                name,
                in_len: mask_len,
                in_ty: mask_ty,
                arg_ty: values_ty,
                out_len: values_len
            }
        );

        // The second argument must be a mutable pointer type matching the element type
        require!(
            matches!(
                *pointer_ty.kind(),
                ty::RawPtr(p_ty, p_mutbl)
                    if p_ty == values_elem && p_ty.kind() == values_elem.kind() && p_mutbl.is_mut()
            ),
            InvalidMonomorphization::ExpectedElementType {
                span,
                name,
                expected_element: values_elem,
                second_arg: pointer_ty,
                in_elem: values_elem,
                in_ty: values_ty,
                mutability: ExpectedPointerMutability::Mut,
            }
        );

        let m_elem_bitwidth = require_int_or_uint_ty!(
            mask_elem.kind(),
            InvalidMonomorphization::MaskWrongElementType { span, name, ty: mask_elem }
        );

        let mask = vector_mask_to_bitmask(bx, args[0].immediate(), m_elem_bitwidth, mask_len);
        let mask_ty = bx.type_vector(bx.type_i1(), mask_len);

        // Alignment of T, must be a constant integer value:
        let alignment_ty = bx.type_i32();
        let alignment = bx.const_i32(bx.align_of(values_elem).bytes() as i32);

        let ret_t = bx.type_void();

        let llvm_pointer = bx.type_ptr();

        // Type of the vector of elements:
        let llvm_elem_vec_ty = llvm_vector_ty(bx, values_elem, values_len);
        let llvm_elem_vec_str = llvm_vector_str(bx, values_elem, values_len);

        let llvm_intrinsic = format!("llvm.masked.store.{llvm_elem_vec_str}.p0");
        let fn_ty = bx.type_func(&[llvm_elem_vec_ty, llvm_pointer, alignment_ty, mask_ty], ret_t);
        let f = bx.declare_cfn(&llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
        let v = bx.call(
            fn_ty,
            None,
            None,
            f,
            &[args[2].immediate(), args[1].immediate(), alignment, mask],
            None,
            None,
        );
        return Ok(v);
    }

    if name == sym::simd_scatter {
        // simd_scatter(values: <N x T>, pointers: <N x *mut T>,
        //             mask: <N x i{M}>) -> ()
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (_, element_ty0) = require_simd!(in_ty, SimdFirst);
        let (element_len1, element_ty1) = require_simd!(arg_tys[1], SimdSecond);
        let (element_len2, element_ty2) = require_simd!(arg_tys[2], SimdThird);

        // Of the same length:
        require!(
            in_len == element_len1,
            InvalidMonomorphization::SecondArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[1],
                out_len: element_len1
            }
        );
        require!(
            in_len == element_len2,
            InvalidMonomorphization::ThirdArgumentLength {
                span,
                name,
                in_len,
                in_ty,
                arg_ty: arg_tys[2],
                out_len: element_len2
            }
        );

        require!(
            matches!(
                *element_ty1.kind(),
                ty::RawPtr(p_ty, p_mutbl)
                    if p_ty == in_elem && p_mutbl.is_mut() && p_ty.kind() == element_ty0.kind()
            ),
            InvalidMonomorphization::ExpectedElementType {
                span,
                name,
                expected_element: element_ty1,
                second_arg: arg_tys[1],
                in_elem,
                in_ty,
                mutability: ExpectedPointerMutability::Mut,
            }
        );

        // The element type of the third argument must be an integer type of any width:
        let mask_elem_bitwidth = require_int_or_uint_ty!(
            element_ty2.kind(),
            InvalidMonomorphization::MaskWrongElementType { span, name, ty: element_ty2 }
        );

        // Alignment of T, must be a constant integer value:
        let alignment_ty = bx.type_i32();
        let alignment = bx.const_i32(bx.align_of(in_elem).bytes() as i32);

        // Truncate the mask vector to a vector of i1s:
        let mask = vector_mask_to_bitmask(bx, args[2].immediate(), mask_elem_bitwidth, in_len);
        let mask_ty = bx.type_vector(bx.type_i1(), in_len);

        let ret_t = bx.type_void();

        // Type of the vector of pointers:
        let llvm_pointer_vec_ty = llvm_vector_ty(bx, element_ty1, in_len);
        let llvm_pointer_vec_str = llvm_vector_str(bx, element_ty1, in_len);

        // Type of the vector of elements:
        let llvm_elem_vec_ty = llvm_vector_ty(bx, element_ty0, in_len);
        let llvm_elem_vec_str = llvm_vector_str(bx, element_ty0, in_len);

        let llvm_intrinsic =
            format!("llvm.masked.scatter.{llvm_elem_vec_str}.{llvm_pointer_vec_str}");
        let fn_ty =
            bx.type_func(&[llvm_elem_vec_ty, llvm_pointer_vec_ty, alignment_ty, mask_ty], ret_t);
        let f = bx.declare_cfn(&llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
        let v = bx.call(
            fn_ty,
            None,
            None,
            f,
            &[args[0].immediate(), args[1].immediate(), alignment, mask],
            None,
            None,
        );
        return Ok(v);
    }

    macro_rules! arith_red {
        ($name:ident : $integer_reduce:ident, $float_reduce:ident, $ordered:expr, $op:ident,
         $identity:expr) => {
            if name == sym::$name {
                require!(
                    ret_ty == in_elem,
                    InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
                );
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.$integer_reduce(args[0].immediate());
                        if $ordered {
                            // if overflow occurs, the result is the
                            // mathematical result modulo 2^n:
                            Ok(bx.$op(args[1].immediate(), r))
                        } else {
                            Ok(bx.$integer_reduce(args[0].immediate()))
                        }
                    }
                    ty::Float(f) => {
                        let acc = if $ordered {
                            // ordered arithmetic reductions take an accumulator
                            args[1].immediate()
                        } else {
                            // unordered arithmetic reductions use the identity accumulator
                            match f.bit_width() {
                                32 => bx.const_real(bx.type_f32(), $identity),
                                64 => bx.const_real(bx.type_f64(), $identity),
                                v => return_error!(
                                    InvalidMonomorphization::UnsupportedSymbolOfSize {
                                        span,
                                        name,
                                        symbol: sym::$name,
                                        in_ty,
                                        in_elem,
                                        size: v,
                                        ret_ty
                                    }
                                ),
                            }
                        };
                        Ok(bx.$float_reduce(acc, args[0].immediate()))
                    }
                    _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                        span,
                        name,
                        symbol: sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    }),
                };
            }
        };
    }

    arith_red!(simd_reduce_add_ordered: vector_reduce_add, vector_reduce_fadd, true, add, -0.0);
    arith_red!(simd_reduce_mul_ordered: vector_reduce_mul, vector_reduce_fmul, true, mul, 1.0);
    arith_red!(
        simd_reduce_add_unordered: vector_reduce_add,
        vector_reduce_fadd_reassoc,
        false,
        add,
        -0.0
    );
    arith_red!(
        simd_reduce_mul_unordered: vector_reduce_mul,
        vector_reduce_fmul_reassoc,
        false,
        mul,
        1.0
    );

    macro_rules! minmax_red {
        ($name:ident: $int_red:ident, $float_red:ident) => {
            if name == sym::$name {
                require!(
                    ret_ty == in_elem,
                    InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
                );
                return match in_elem.kind() {
                    ty::Int(_i) => Ok(bx.$int_red(args[0].immediate(), true)),
                    ty::Uint(_u) => Ok(bx.$int_red(args[0].immediate(), false)),
                    ty::Float(_f) => Ok(bx.$float_red(args[0].immediate())),
                    _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                        span,
                        name,
                        symbol: sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    }),
                };
            }
        };
    }

    minmax_red!(simd_reduce_min: vector_reduce_min, vector_reduce_fmin);
    minmax_red!(simd_reduce_max: vector_reduce_max, vector_reduce_fmax);

    macro_rules! bitwise_red {
        ($name:ident : $red:ident, $boolean:expr) => {
            if name == sym::$name {
                let input = if !$boolean {
                    require!(
                        ret_ty == in_elem,
                        InvalidMonomorphization::ReturnType { span, name, in_elem, in_ty, ret_ty }
                    );
                    args[0].immediate()
                } else {
                    let bitwidth = match in_elem.kind() {
                        ty::Int(i) => {
                            i.bit_width().unwrap_or_else(|| bx.data_layout().pointer_size.bits())
                        }
                        ty::Uint(i) => {
                            i.bit_width().unwrap_or_else(|| bx.data_layout().pointer_size.bits())
                        }
                        _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                            span,
                            name,
                            symbol: sym::$name,
                            in_ty,
                            in_elem,
                            ret_ty
                        }),
                    };

                    vector_mask_to_bitmask(bx, args[0].immediate(), bitwidth, in_len as _)
                };
                return match in_elem.kind() {
                    ty::Int(_) | ty::Uint(_) => {
                        let r = bx.$red(input);
                        Ok(if !$boolean { r } else { bx.zext(r, bx.type_bool()) })
                    }
                    _ => return_error!(InvalidMonomorphization::UnsupportedSymbol {
                        span,
                        name,
                        symbol: sym::$name,
                        in_ty,
                        in_elem,
                        ret_ty
                    }),
                };
            }
        };
    }

    bitwise_red!(simd_reduce_and: vector_reduce_and, false);
    bitwise_red!(simd_reduce_or: vector_reduce_or, false);
    bitwise_red!(simd_reduce_xor: vector_reduce_xor, false);
    bitwise_red!(simd_reduce_all: vector_reduce_and, true);
    bitwise_red!(simd_reduce_any: vector_reduce_or, true);

    if name == sym::simd_cast_ptr {
        let (out_len, out_elem) = require_simd!(ret_ty, SimdReturn);
        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );

        match in_elem.kind() {
            ty::RawPtr(p_ty, _) => {
                let metadata = p_ty.ptr_metadata_ty(bx.tcx, |ty| {
                    bx.tcx.normalize_erasing_regions(bx.typing_env(), ty)
                });
                require!(
                    metadata.is_unit(),
                    InvalidMonomorphization::CastWidePointer { span, name, ty: in_elem }
                );
            }
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: in_elem })
            }
        }
        match out_elem.kind() {
            ty::RawPtr(p_ty, _) => {
                let metadata = p_ty.ptr_metadata_ty(bx.tcx, |ty| {
                    bx.tcx.normalize_erasing_regions(bx.typing_env(), ty)
                });
                require!(
                    metadata.is_unit(),
                    InvalidMonomorphization::CastWidePointer { span, name, ty: out_elem }
                );
            }
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: out_elem })
            }
        }

        return Ok(args[0].immediate());
    }

    if name == sym::simd_expose_provenance {
        let (out_len, out_elem) = require_simd!(ret_ty, SimdReturn);
        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );

        match in_elem.kind() {
            ty::RawPtr(_, _) => {}
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: in_elem })
            }
        }
        match out_elem.kind() {
            ty::Uint(ty::UintTy::Usize) => {}
            _ => return_error!(InvalidMonomorphization::ExpectedUsize { span, name, ty: out_elem }),
        }

        return Ok(bx.ptrtoint(args[0].immediate(), llret_ty));
    }

    if name == sym::simd_with_exposed_provenance {
        let (out_len, out_elem) = require_simd!(ret_ty, SimdReturn);
        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );

        match in_elem.kind() {
            ty::Uint(ty::UintTy::Usize) => {}
            _ => return_error!(InvalidMonomorphization::ExpectedUsize { span, name, ty: in_elem }),
        }
        match out_elem.kind() {
            ty::RawPtr(_, _) => {}
            _ => {
                return_error!(InvalidMonomorphization::ExpectedPointer { span, name, ty: out_elem })
            }
        }

        return Ok(bx.inttoptr(args[0].immediate(), llret_ty));
    }

    if name == sym::simd_cast || name == sym::simd_as {
        let (out_len, out_elem) = require_simd!(ret_ty, SimdReturn);
        require!(
            in_len == out_len,
            InvalidMonomorphization::ReturnLengthInputType {
                span,
                name,
                in_len,
                in_ty,
                ret_ty,
                out_len
            }
        );
        // casting cares about nominal type, not just structural type
        if in_elem == out_elem {
            return Ok(args[0].immediate());
        }

        #[derive(Copy, Clone)]
        enum Sign {
            Unsigned,
            Signed,
        }
        use Sign::*;

        enum Style {
            Float,
            Int(Sign),
            Unsupported,
        }

        let (in_style, in_width) = match in_elem.kind() {
            // vectors of pointer-sized integers should've been
            // disallowed before here, so this unwrap is safe.
            ty::Int(i) => (
                Style::Int(Signed),
                i.normalize(bx.tcx().sess.target.pointer_width).bit_width().unwrap(),
            ),
            ty::Uint(u) => (
                Style::Int(Unsigned),
                u.normalize(bx.tcx().sess.target.pointer_width).bit_width().unwrap(),
            ),
            ty::Float(f) => (Style::Float, f.bit_width()),
            _ => (Style::Unsupported, 0),
        };
        let (out_style, out_width) = match out_elem.kind() {
            ty::Int(i) => (
                Style::Int(Signed),
                i.normalize(bx.tcx().sess.target.pointer_width).bit_width().unwrap(),
            ),
            ty::Uint(u) => (
                Style::Int(Unsigned),
                u.normalize(bx.tcx().sess.target.pointer_width).bit_width().unwrap(),
            ),
            ty::Float(f) => (Style::Float, f.bit_width()),
            _ => (Style::Unsupported, 0),
        };

        match (in_style, out_style) {
            (Style::Int(sign), Style::Int(_)) => {
                return Ok(match in_width.cmp(&out_width) {
                    Ordering::Greater => bx.trunc(args[0].immediate(), llret_ty),
                    Ordering::Equal => args[0].immediate(),
                    Ordering::Less => match sign {
                        Sign::Signed => bx.sext(args[0].immediate(), llret_ty),
                        Sign::Unsigned => bx.zext(args[0].immediate(), llret_ty),
                    },
                });
            }
            (Style::Int(Sign::Signed), Style::Float) => {
                return Ok(bx.sitofp(args[0].immediate(), llret_ty));
            }
            (Style::Int(Sign::Unsigned), Style::Float) => {
                return Ok(bx.uitofp(args[0].immediate(), llret_ty));
            }
            (Style::Float, Style::Int(sign)) => {
                return Ok(match (sign, name == sym::simd_as) {
                    (Sign::Unsigned, false) => bx.fptoui(args[0].immediate(), llret_ty),
                    (Sign::Signed, false) => bx.fptosi(args[0].immediate(), llret_ty),
                    (_, true) => bx.cast_float_to_int(
                        matches!(sign, Sign::Signed),
                        args[0].immediate(),
                        llret_ty,
                    ),
                });
            }
            (Style::Float, Style::Float) => {
                return Ok(match in_width.cmp(&out_width) {
                    Ordering::Greater => bx.fptrunc(args[0].immediate(), llret_ty),
                    Ordering::Equal => args[0].immediate(),
                    Ordering::Less => bx.fpext(args[0].immediate(), llret_ty),
                });
            }
            _ => { /* Unsupported. Fallthrough. */ }
        }
        return_error!(InvalidMonomorphization::UnsupportedCast {
            span,
            name,
            in_ty,
            in_elem,
            ret_ty,
            out_elem
        });
    }
    macro_rules! arith_binary {
        ($($name: ident: $($($p: ident),* => $call: ident),*;)*) => {
            $(if name == sym::$name {
                match in_elem.kind() {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate(), args[1].immediate()))
                    })*
                    _ => {},
                }
                return_error!(
                    InvalidMonomorphization::UnsupportedOperation { span, name, in_ty, in_elem }
                );
            })*
        }
    }
    arith_binary! {
        simd_add: Uint, Int => add, Float => fadd;
        simd_sub: Uint, Int => sub, Float => fsub;
        simd_mul: Uint, Int => mul, Float => fmul;
        simd_div: Uint => udiv, Int => sdiv, Float => fdiv;
        simd_rem: Uint => urem, Int => srem, Float => frem;
        simd_shl: Uint, Int => shl;
        simd_shr: Uint => lshr, Int => ashr;
        simd_and: Uint, Int => and;
        simd_or: Uint, Int => or;
        simd_xor: Uint, Int => xor;
        simd_fmax: Float => maxnum;
        simd_fmin: Float => minnum;

    }
    macro_rules! arith_unary {
        ($($name: ident: $($($p: ident),* => $call: ident),*;)*) => {
            $(if name == sym::$name {
                match in_elem.kind() {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate()))
                    })*
                    _ => {},
                }
                return_error!(
                    InvalidMonomorphization::UnsupportedOperation { span, name, in_ty, in_elem }
                );
            })*
        }
    }
    arith_unary! {
        simd_neg: Int => neg, Float => fneg;
    }

    // Unary integer intrinsics
    if matches!(
        name,
        sym::simd_bswap | sym::simd_bitreverse | sym::simd_ctlz | sym::simd_ctpop | sym::simd_cttz
    ) {
        let vec_ty = bx.cx.type_vector(
            match *in_elem.kind() {
                ty::Int(i) => bx.cx.type_int_from_ty(i),
                ty::Uint(i) => bx.cx.type_uint_from_ty(i),
                _ => return_error!(InvalidMonomorphization::UnsupportedOperation {
                    span,
                    name,
                    in_ty,
                    in_elem
                }),
            },
            in_len as u64,
        );
        let intrinsic_name = match name {
            sym::simd_bswap => "bswap",
            sym::simd_bitreverse => "bitreverse",
            sym::simd_ctlz => "ctlz",
            sym::simd_ctpop => "ctpop",
            sym::simd_cttz => "cttz",
            _ => unreachable!(),
        };
        let int_size = in_elem.int_size_and_signed(bx.tcx()).0.bits();
        let llvm_intrinsic = &format!("llvm.{}.v{}i{}", intrinsic_name, in_len, int_size,);

        return match name {
            // byte swap is no-op for i8/u8
            sym::simd_bswap if int_size == 8 => Ok(args[0].immediate()),
            sym::simd_ctlz | sym::simd_cttz => {
                // for the (int, i1 immediate) pair, the second arg adds `(0, true) => poison`
                let fn_ty = bx.type_func(&[vec_ty, bx.type_i1()], vec_ty);
                let dont_poison_on_zero = bx.const_int(bx.type_i1(), 0);
                let f = bx.declare_cfn(llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
                Ok(bx.call(
                    fn_ty,
                    None,
                    None,
                    f,
                    &[args[0].immediate(), dont_poison_on_zero],
                    None,
                    None,
                ))
            }
            sym::simd_bswap | sym::simd_bitreverse | sym::simd_ctpop => {
                // simple unary argument cases
                let fn_ty = bx.type_func(&[vec_ty], vec_ty);
                let f = bx.declare_cfn(llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
                Ok(bx.call(fn_ty, None, None, f, &[args[0].immediate()], None, None))
            }
            _ => unreachable!(),
        };
    }

    if name == sym::simd_arith_offset {
        // This also checks that the first operand is a ptr type.
        let pointee = in_elem.builtin_deref(true).unwrap_or_else(|| {
            span_bug!(span, "must be called with a vector of pointer types as first argument")
        });
        let layout = bx.layout_of(pointee);
        let ptrs = args[0].immediate();
        // The second argument must be a ptr-sized integer.
        // (We don't care about the signedness, this is wrapping anyway.)
        let (_offsets_len, offsets_elem) = arg_tys[1].simd_size_and_type(bx.tcx());
        if !matches!(offsets_elem.kind(), ty::Int(ty::IntTy::Isize) | ty::Uint(ty::UintTy::Usize)) {
            span_bug!(
                span,
                "must be called with a vector of pointer-sized integers as second argument"
            );
        }
        let offsets = args[1].immediate();

        return Ok(bx.gep(bx.backend_type(layout), ptrs, &[offsets]));
    }

    if name == sym::simd_saturating_add || name == sym::simd_saturating_sub {
        let lhs = args[0].immediate();
        let rhs = args[1].immediate();
        let is_add = name == sym::simd_saturating_add;
        let ptr_bits = bx.tcx().data_layout.pointer_size.bits() as _;
        let (signed, elem_width, elem_ty) = match *in_elem.kind() {
            ty::Int(i) => (true, i.bit_width().unwrap_or(ptr_bits), bx.cx.type_int_from_ty(i)),
            ty::Uint(i) => (false, i.bit_width().unwrap_or(ptr_bits), bx.cx.type_uint_from_ty(i)),
            _ => {
                return_error!(InvalidMonomorphization::ExpectedVectorElementType {
                    span,
                    name,
                    expected_element: arg_tys[0].simd_size_and_type(bx.tcx()).1,
                    vector_type: arg_tys[0]
                });
            }
        };
        let llvm_intrinsic = &format!(
            "llvm.{}{}.sat.v{}i{}",
            if signed { 's' } else { 'u' },
            if is_add { "add" } else { "sub" },
            in_len,
            elem_width
        );
        let vec_ty = bx.cx.type_vector(elem_ty, in_len as u64);

        let fn_ty = bx.type_func(&[vec_ty, vec_ty], vec_ty);
        let f = bx.declare_cfn(llvm_intrinsic, llvm::UnnamedAddr::No, fn_ty);
        let v = bx.call(fn_ty, None, None, f, &[lhs, rhs], None, None);
        return Ok(v);
    }

    span_bug!(span, "unknown SIMD intrinsic");
}
