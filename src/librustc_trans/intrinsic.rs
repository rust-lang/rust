// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_upper_case_globals)]

use intrinsics::{self, Intrinsic};
use llvm;
use llvm::{ValueRef};
use abi::{Abi, FnType, PassMode};
use mir::place::PlaceRef;
use mir::operand::{OperandRef, OperandValue};
use base::*;
use common::*;
use declare;
use glue;
use type_::Type;
use type_of::LayoutLlvmExt;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{HasDataLayout, LayoutOf};
use rustc::hir;
use syntax::ast;
use syntax::symbol::Symbol;
use builder::Builder;

use rustc::session::Session;
use syntax_pos::Span;

use std::cmp::Ordering;
use std::iter;

fn get_simple_intrinsic(cx: &CodegenCx, name: &str) -> Option<ValueRef> {
    let llvm_name = match name {
        "sqrtf32" => "llvm.sqrt.f32",
        "sqrtf64" => "llvm.sqrt.f64",
        "powif32" => "llvm.powi.f32",
        "powif64" => "llvm.powi.f64",
        "sinf32" => "llvm.sin.f32",
        "sinf64" => "llvm.sin.f64",
        "cosf32" => "llvm.cos.f32",
        "cosf64" => "llvm.cos.f64",
        "powf32" => "llvm.pow.f32",
        "powf64" => "llvm.pow.f64",
        "expf32" => "llvm.exp.f32",
        "expf64" => "llvm.exp.f64",
        "exp2f32" => "llvm.exp2.f32",
        "exp2f64" => "llvm.exp2.f64",
        "logf32" => "llvm.log.f32",
        "logf64" => "llvm.log.f64",
        "log10f32" => "llvm.log10.f32",
        "log10f64" => "llvm.log10.f64",
        "log2f32" => "llvm.log2.f32",
        "log2f64" => "llvm.log2.f64",
        "fmaf32" => "llvm.fma.f32",
        "fmaf64" => "llvm.fma.f64",
        "fabsf32" => "llvm.fabs.f32",
        "fabsf64" => "llvm.fabs.f64",
        "copysignf32" => "llvm.copysign.f32",
        "copysignf64" => "llvm.copysign.f64",
        "floorf32" => "llvm.floor.f32",
        "floorf64" => "llvm.floor.f64",
        "ceilf32" => "llvm.ceil.f32",
        "ceilf64" => "llvm.ceil.f64",
        "truncf32" => "llvm.trunc.f32",
        "truncf64" => "llvm.trunc.f64",
        "rintf32" => "llvm.rint.f32",
        "rintf64" => "llvm.rint.f64",
        "nearbyintf32" => "llvm.nearbyint.f32",
        "nearbyintf64" => "llvm.nearbyint.f64",
        "roundf32" => "llvm.round.f32",
        "roundf64" => "llvm.round.f64",
        "assume" => "llvm.assume",
        "abort" => "llvm.trap",
        _ => return None
    };
    Some(cx.get_intrinsic(&llvm_name))
}

/// Remember to add all intrinsics here, in librustc_typeck/check/mod.rs,
/// and in libcore/intrinsics.rs; if you need access to any llvm intrinsics,
/// add them to librustc_trans/trans/context.rs
pub fn trans_intrinsic_call<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                                      callee_ty: Ty<'tcx>,
                                      fn_ty: &FnType<'tcx>,
                                      args: &[OperandRef<'tcx>],
                                      llresult: ValueRef,
                                      span: Span) {
    let cx = bx.cx;
    let tcx = cx.tcx;

    let (def_id, substs) = match callee_ty.sty {
        ty::TyFnDef(def_id, substs) => (def_id, substs),
        _ => bug!("expected fn item type, found {}", callee_ty)
    };

    let sig = callee_ty.fn_sig(tcx);
    let sig = tcx.erase_late_bound_regions_and_normalize(&sig);
    let arg_tys = sig.inputs();
    let ret_ty = sig.output();
    let name = &*tcx.item_name(def_id);

    let llret_ty = cx.layout_of(ret_ty).llvm_type(cx);
    let result = PlaceRef::new_sized(llresult, fn_ty.ret.layout, fn_ty.ret.layout.align);

    let simple = get_simple_intrinsic(cx, name);
    let llval = match name {
        _ if simple.is_some() => {
            bx.call(simple.unwrap(),
                     &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                     None)
        }
        "unreachable" => {
            return;
        },
        "likely" => {
            let expect = cx.get_intrinsic(&("llvm.expect.i1"));
            bx.call(expect, &[args[0].immediate(), C_bool(cx, true)], None)
        }
        "unlikely" => {
            let expect = cx.get_intrinsic(&("llvm.expect.i1"));
            bx.call(expect, &[args[0].immediate(), C_bool(cx, false)], None)
        }
        "try" => {
            try_intrinsic(bx, cx,
                          args[0].immediate(),
                          args[1].immediate(),
                          args[2].immediate(),
                          llresult);
            return;
        }
        "breakpoint" => {
            let llfn = cx.get_intrinsic(&("llvm.debugtrap"));
            bx.call(llfn, &[], None)
        }
        "size_of" => {
            let tp_ty = substs.type_at(0);
            C_usize(cx, cx.size_of(tp_ty).bytes())
        }
        "size_of_val" => {
            let tp_ty = substs.type_at(0);
            if let OperandValue::Pair(_, meta) = args[0].val {
                let (llsize, _) =
                    glue::size_and_align_of_dst(bx, tp_ty, meta);
                llsize
            } else {
                C_usize(cx, cx.size_of(tp_ty).bytes())
            }
        }
        "min_align_of" => {
            let tp_ty = substs.type_at(0);
            C_usize(cx, cx.align_of(tp_ty).abi())
        }
        "min_align_of_val" => {
            let tp_ty = substs.type_at(0);
            if let OperandValue::Pair(_, meta) = args[0].val {
                let (_, llalign) =
                    glue::size_and_align_of_dst(bx, tp_ty, meta);
                llalign
            } else {
                C_usize(cx, cx.align_of(tp_ty).abi())
            }
        }
        "pref_align_of" => {
            let tp_ty = substs.type_at(0);
            C_usize(cx, cx.align_of(tp_ty).pref())
        }
        "type_name" => {
            let tp_ty = substs.type_at(0);
            let ty_name = Symbol::intern(&tp_ty.to_string()).as_str();
            C_str_slice(cx, ty_name)
        }
        "type_id" => {
            C_u64(cx, cx.tcx.type_id_hash(substs.type_at(0)))
        }
        "init" => {
            let ty = substs.type_at(0);
            if !cx.layout_of(ty).is_zst() {
                // Just zero out the stack slot.
                // If we store a zero constant, LLVM will drown in vreg allocation for large data
                // structures, and the generated code will be awful. (A telltale sign of this is
                // large quantities of `mov [byte ptr foo],0` in the generated code.)
                memset_intrinsic(bx, false, ty, llresult, C_u8(cx, 0), C_usize(cx, 1));
            }
            return;
        }
        // Effectively no-ops
        "uninit" => {
            return;
        }
        "needs_drop" => {
            let tp_ty = substs.type_at(0);

            C_bool(cx, bx.cx.type_needs_drop(tp_ty))
        }
        "offset" => {
            let ptr = args[0].immediate();
            let offset = args[1].immediate();
            bx.inbounds_gep(ptr, &[offset])
        }
        "arith_offset" => {
            let ptr = args[0].immediate();
            let offset = args[1].immediate();
            bx.gep(ptr, &[offset])
        }

        "copy_nonoverlapping" => {
            copy_intrinsic(bx, false, false, substs.type_at(0),
                           args[1].immediate(), args[0].immediate(), args[2].immediate())
        }
        "copy" => {
            copy_intrinsic(bx, true, false, substs.type_at(0),
                           args[1].immediate(), args[0].immediate(), args[2].immediate())
        }
        "write_bytes" => {
            memset_intrinsic(bx, false, substs.type_at(0),
                             args[0].immediate(), args[1].immediate(), args[2].immediate())
        }

        "volatile_copy_nonoverlapping_memory" => {
            copy_intrinsic(bx, false, true, substs.type_at(0),
                           args[0].immediate(), args[1].immediate(), args[2].immediate())
        }
        "volatile_copy_memory" => {
            copy_intrinsic(bx, true, true, substs.type_at(0),
                           args[0].immediate(), args[1].immediate(), args[2].immediate())
        }
        "volatile_set_memory" => {
            memset_intrinsic(bx, true, substs.type_at(0),
                             args[0].immediate(), args[1].immediate(), args[2].immediate())
        }
        "volatile_load" => {
            let tp_ty = substs.type_at(0);
            let mut ptr = args[0].immediate();
            if let PassMode::Cast(ty) = fn_ty.ret.mode {
                ptr = bx.pointercast(ptr, ty.llvm_type(cx).ptr_to());
            }
            let load = bx.volatile_load(ptr);
            unsafe {
                llvm::LLVMSetAlignment(load, cx.align_of(tp_ty).abi() as u32);
            }
            to_immediate(bx, load, cx.layout_of(tp_ty))
        },
        "volatile_store" => {
            let tp_ty = substs.type_at(0);
            let dst = args[0].deref(bx.cx);
            if let OperandValue::Pair(a, b) = args[1].val {
                bx.volatile_store(a, dst.project_field(bx, 0).llval);
                bx.volatile_store(b, dst.project_field(bx, 1).llval);
            } else {
                let val = if let OperandValue::Ref(ptr, align) = args[1].val {
                    bx.load(ptr, align)
                } else {
                    if dst.layout.is_zst() {
                        return;
                    }
                    from_immediate(bx, args[1].immediate())
                };
                let ptr = bx.pointercast(dst.llval, val_ty(val).ptr_to());
                let store = bx.volatile_store(val, ptr);
                unsafe {
                    llvm::LLVMSetAlignment(store, cx.align_of(tp_ty).abi() as u32);
                }
            }
            return;
        },
        "prefetch_read_data" | "prefetch_write_data" |
        "prefetch_read_instruction" | "prefetch_write_instruction" => {
            let expect = cx.get_intrinsic(&("llvm.prefetch"));
            let (rw, cache_type) = match name {
                "prefetch_read_data" => (0, 1),
                "prefetch_write_data" => (1, 1),
                "prefetch_read_instruction" => (0, 0),
                "prefetch_write_instruction" => (1, 0),
                _ => bug!()
            };
            bx.call(expect, &[
                args[0].immediate(),
                C_i32(cx, rw),
                args[1].immediate(),
                C_i32(cx, cache_type)
            ], None)
        },
        "ctlz" | "ctlz_nonzero" | "cttz" | "cttz_nonzero" | "ctpop" | "bswap" |
        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" |
        "overflowing_add" | "overflowing_sub" | "overflowing_mul" |
        "unchecked_div" | "unchecked_rem" | "unchecked_shl" | "unchecked_shr" => {
            let ty = arg_tys[0];
            match int_type_width_signed(ty, cx) {
                Some((width, signed)) =>
                    match name {
                        "ctlz" | "cttz" => {
                            let y = C_bool(bx.cx, false);
                            let llfn = cx.get_intrinsic(&format!("llvm.{}.i{}", name, width));
                            bx.call(llfn, &[args[0].immediate(), y], None)
                        }
                        "ctlz_nonzero" | "cttz_nonzero" => {
                            let y = C_bool(bx.cx, true);
                            let llvm_name = &format!("llvm.{}.i{}", &name[..4], width);
                            let llfn = cx.get_intrinsic(llvm_name);
                            bx.call(llfn, &[args[0].immediate(), y], None)
                        }
                        "ctpop" => bx.call(cx.get_intrinsic(&format!("llvm.ctpop.i{}", width)),
                                        &[args[0].immediate()], None),
                        "bswap" => {
                            if width == 8 {
                                args[0].immediate() // byte swap a u8/i8 is just a no-op
                            } else {
                                bx.call(cx.get_intrinsic(&format!("llvm.bswap.i{}", width)),
                                        &[args[0].immediate()], None)
                            }
                        }
                        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" => {
                            let intrinsic = format!("llvm.{}{}.with.overflow.i{}",
                                                    if signed { 's' } else { 'u' },
                                                    &name[..3], width);
                            let llfn = bx.cx.get_intrinsic(&intrinsic);

                            // Convert `i1` to a `bool`, and write it to the out parameter
                            let pair = bx.call(llfn, &[
                                args[0].immediate(),
                                args[1].immediate()
                            ], None);
                            let val = bx.extract_value(pair, 0);
                            let overflow = bx.zext(bx.extract_value(pair, 1), Type::bool(cx));

                            let dest = result.project_field(bx, 0);
                            bx.store(val, dest.llval, dest.align);
                            let dest = result.project_field(bx, 1);
                            bx.store(overflow, dest.llval, dest.align);

                            return;
                        },
                        "overflowing_add" => bx.add(args[0].immediate(), args[1].immediate()),
                        "overflowing_sub" => bx.sub(args[0].immediate(), args[1].immediate()),
                        "overflowing_mul" => bx.mul(args[0].immediate(), args[1].immediate()),
                        "unchecked_div" =>
                            if signed {
                                bx.sdiv(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.udiv(args[0].immediate(), args[1].immediate())
                            },
                        "unchecked_rem" =>
                            if signed {
                                bx.srem(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.urem(args[0].immediate(), args[1].immediate())
                            },
                        "unchecked_shl" => bx.shl(args[0].immediate(), args[1].immediate()),
                        "unchecked_shr" =>
                            if signed {
                                bx.ashr(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.lshr(args[0].immediate(), args[1].immediate())
                            },
                        _ => bug!(),
                    },
                None => {
                    span_invalid_monomorphization_error(
                        tcx.sess, span,
                        &format!("invalid monomorphization of `{}` intrinsic: \
                                  expected basic integer type, found `{}`", name, ty));
                    return;
                }
            }

        },
        "fadd_fast" | "fsub_fast" | "fmul_fast" | "fdiv_fast" | "frem_fast" => {
            let sty = &arg_tys[0].sty;
            match float_type_width(sty) {
                Some(_width) =>
                    match name {
                        "fadd_fast" => bx.fadd_fast(args[0].immediate(), args[1].immediate()),
                        "fsub_fast" => bx.fsub_fast(args[0].immediate(), args[1].immediate()),
                        "fmul_fast" => bx.fmul_fast(args[0].immediate(), args[1].immediate()),
                        "fdiv_fast" => bx.fdiv_fast(args[0].immediate(), args[1].immediate()),
                        "frem_fast" => bx.frem_fast(args[0].immediate(), args[1].immediate()),
                        _ => bug!(),
                    },
                None => {
                    span_invalid_monomorphization_error(
                        tcx.sess, span,
                        &format!("invalid monomorphization of `{}` intrinsic: \
                                  expected basic float type, found `{}`", name, sty));
                    return;
                }
            }

        },

        "discriminant_value" => {
            args[0].deref(bx.cx).trans_get_discr(bx, ret_ty)
        }

        "align_offset" => {
            // `ptr as usize`
            let ptr_val = bx.ptrtoint(args[0].immediate(), bx.cx.isize_ty);
            // `ptr_val % align`
            let align = args[1].immediate();
            let offset = bx.urem(ptr_val, align);
            let zero = C_null(bx.cx.isize_ty);
            // `offset == 0`
            let is_zero = bx.icmp(llvm::IntPredicate::IntEQ, offset, zero);
            // `if offset == 0 { 0 } else { align - offset }`
            bx.select(is_zero, zero, bx.sub(align, offset))
        }
        name if name.starts_with("simd_") => {
            match generic_simd_intrinsic(bx, name,
                                         callee_ty,
                                         args,
                                         ret_ty, llret_ty,
                                         span) {
                Ok(llval) => llval,
                Err(()) => return
            }
        }
        // This requires that atomic intrinsics follow a specific naming pattern:
        // "atomic_<operation>[_<ordering>]", and no ordering means SeqCst
        name if name.starts_with("atomic_") => {
            use llvm::AtomicOrdering::*;

            let split: Vec<&str> = name.split('_').collect();

            let is_cxchg = split[1] == "cxchg" || split[1] == "cxchgweak";
            let (order, failorder) = match split.len() {
                2 => (SequentiallyConsistent, SequentiallyConsistent),
                3 => match split[2] {
                    "unordered" => (Unordered, Unordered),
                    "relaxed" => (Monotonic, Monotonic),
                    "acq"     => (Acquire, Acquire),
                    "rel"     => (Release, Monotonic),
                    "acqrel"  => (AcquireRelease, Acquire),
                    "failrelaxed" if is_cxchg =>
                        (SequentiallyConsistent, Monotonic),
                    "failacq" if is_cxchg =>
                        (SequentiallyConsistent, Acquire),
                    _ => cx.sess().fatal("unknown ordering in atomic intrinsic")
                },
                4 => match (split[2], split[3]) {
                    ("acq", "failrelaxed") if is_cxchg =>
                        (Acquire, Monotonic),
                    ("acqrel", "failrelaxed") if is_cxchg =>
                        (AcquireRelease, Monotonic),
                    _ => cx.sess().fatal("unknown ordering in atomic intrinsic")
                },
                _ => cx.sess().fatal("Atomic intrinsic not in correct format"),
            };

            let invalid_monomorphization = |ty| {
                span_invalid_monomorphization_error(tcx.sess, span,
                    &format!("invalid monomorphization of `{}` intrinsic: \
                              expected basic integer type, found `{}`", name, ty));
            };

            match split[1] {
                "cxchg" | "cxchgweak" => {
                    let ty = substs.type_at(0);
                    if int_type_width_signed(ty, cx).is_some() {
                        let weak = if split[1] == "cxchgweak" { llvm::True } else { llvm::False };
                        let pair = bx.atomic_cmpxchg(
                            args[0].immediate(),
                            args[1].immediate(),
                            args[2].immediate(),
                            order,
                            failorder,
                            weak);
                        let val = bx.extract_value(pair, 0);
                        let success = bx.zext(bx.extract_value(pair, 1), Type::bool(bx.cx));

                        let dest = result.project_field(bx, 0);
                        bx.store(val, dest.llval, dest.align);
                        let dest = result.project_field(bx, 1);
                        bx.store(success, dest.llval, dest.align);
                        return;
                    } else {
                        return invalid_monomorphization(ty);
                    }
                }

                "load" => {
                    let ty = substs.type_at(0);
                    if int_type_width_signed(ty, cx).is_some() {
                        let align = cx.align_of(ty);
                        bx.atomic_load(args[0].immediate(), order, align)
                    } else {
                        return invalid_monomorphization(ty);
                    }
                }

                "store" => {
                    let ty = substs.type_at(0);
                    if int_type_width_signed(ty, cx).is_some() {
                        let align = cx.align_of(ty);
                        bx.atomic_store(args[1].immediate(), args[0].immediate(), order, align);
                        return;
                    } else {
                        return invalid_monomorphization(ty);
                    }
                }

                "fence" => {
                    bx.atomic_fence(order, llvm::SynchronizationScope::CrossThread);
                    return;
                }

                "singlethreadfence" => {
                    bx.atomic_fence(order, llvm::SynchronizationScope::SingleThread);
                    return;
                }

                // These are all AtomicRMW ops
                op => {
                    let atom_op = match op {
                        "xchg"  => llvm::AtomicXchg,
                        "xadd"  => llvm::AtomicAdd,
                        "xsub"  => llvm::AtomicSub,
                        "and"   => llvm::AtomicAnd,
                        "nand"  => llvm::AtomicNand,
                        "or"    => llvm::AtomicOr,
                        "xor"   => llvm::AtomicXor,
                        "max"   => llvm::AtomicMax,
                        "min"   => llvm::AtomicMin,
                        "umax"  => llvm::AtomicUMax,
                        "umin"  => llvm::AtomicUMin,
                        _ => cx.sess().fatal("unknown atomic operation")
                    };

                    let ty = substs.type_at(0);
                    if int_type_width_signed(ty, cx).is_some() {
                        bx.atomic_rmw(atom_op, args[0].immediate(), args[1].immediate(), order)
                    } else {
                        return invalid_monomorphization(ty);
                    }
                }
            }
        }

        "nontemporal_store" => {
            let tp_ty = substs.type_at(0);
            let dst = args[0].deref(bx.cx);
            let val = if let OperandValue::Ref(ptr, align) = args[1].val {
                bx.load(ptr, align)
            } else {
                from_immediate(bx, args[1].immediate())
            };
            let ptr = bx.pointercast(dst.llval, val_ty(val).ptr_to());
            let store = bx.nontemporal_store(val, ptr);
            unsafe {
                llvm::LLVMSetAlignment(store, cx.align_of(tp_ty).abi() as u32);
            }
            return
        }

        _ => {
            let intr = match Intrinsic::find(&name) {
                Some(intr) => intr,
                None => bug!("unknown intrinsic '{}'", name),
            };
            fn one<T>(x: Vec<T>) -> T {
                assert_eq!(x.len(), 1);
                x.into_iter().next().unwrap()
            }
            fn ty_to_type(cx: &CodegenCx, t: &intrinsics::Type) -> Vec<Type> {
                use intrinsics::Type::*;
                match *t {
                    Void => vec![Type::void(cx)],
                    Integer(_signed, _width, llvm_width) => {
                        vec![Type::ix(cx, llvm_width as u64)]
                    }
                    Float(x) => {
                        match x {
                            32 => vec![Type::f32(cx)],
                            64 => vec![Type::f64(cx)],
                            _ => bug!()
                        }
                    }
                    Pointer(ref t, ref llvm_elem, _const) => {
                        let t = llvm_elem.as_ref().unwrap_or(t);
                        let elem = one(ty_to_type(cx, t));
                        vec![elem.ptr_to()]
                    }
                    Vector(ref t, ref llvm_elem, length) => {
                        let t = llvm_elem.as_ref().unwrap_or(t);
                        let elem = one(ty_to_type(cx, t));
                        vec![Type::vector(&elem, length as u64)]
                    }
                    Aggregate(false, ref contents) => {
                        let elems = contents.iter()
                                            .map(|t| one(ty_to_type(cx, t)))
                                            .collect::<Vec<_>>();
                        vec![Type::struct_(cx, &elems, false)]
                    }
                    Aggregate(true, ref contents) => {
                        contents.iter()
                                .flat_map(|t| ty_to_type(cx, t))
                                .collect()
                    }
                }
            }

            // This allows an argument list like `foo, (bar, baz),
            // qux` to be converted into `foo, bar, baz, qux`, integer
            // arguments to be truncated as needed and pointers to be
            // cast.
            fn modify_as_needed<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                                          t: &intrinsics::Type,
                                          arg: &OperandRef<'tcx>)
                                          -> Vec<ValueRef>
            {
                match *t {
                    intrinsics::Type::Aggregate(true, ref contents) => {
                        // We found a tuple that needs squishing! So
                        // run over the tuple and load each field.
                        //
                        // This assumes the type is "simple", i.e. no
                        // destructors, and the contents are SIMD
                        // etc.
                        assert!(!bx.cx.type_needs_drop(arg.layout.ty));
                        let (ptr, align) = match arg.val {
                            OperandValue::Ref(ptr, align) => (ptr, align),
                            _ => bug!()
                        };
                        let arg = PlaceRef::new_sized(ptr, arg.layout, align);
                        (0..contents.len()).map(|i| {
                            arg.project_field(bx, i).load(bx).immediate()
                        }).collect()
                    }
                    intrinsics::Type::Pointer(_, Some(ref llvm_elem), _) => {
                        let llvm_elem = one(ty_to_type(bx.cx, llvm_elem));
                        vec![bx.pointercast(arg.immediate(), llvm_elem.ptr_to())]
                    }
                    intrinsics::Type::Vector(_, Some(ref llvm_elem), length) => {
                        let llvm_elem = one(ty_to_type(bx.cx, llvm_elem));
                        vec![bx.bitcast(arg.immediate(), Type::vector(&llvm_elem, length as u64))]
                    }
                    intrinsics::Type::Integer(_, width, llvm_width) if width != llvm_width => {
                        // the LLVM intrinsic uses a smaller integer
                        // size than the C intrinsic's signature, so
                        // we have to trim it down here.
                        vec![bx.trunc(arg.immediate(), Type::ix(bx.cx, llvm_width as u64))]
                    }
                    _ => vec![arg.immediate()],
                }
            }


            let inputs = intr.inputs.iter()
                                    .flat_map(|t| ty_to_type(cx, t))
                                    .collect::<Vec<_>>();

            let outputs = one(ty_to_type(cx, &intr.output));

            let llargs: Vec<_> = intr.inputs.iter().zip(args).flat_map(|(t, arg)| {
                modify_as_needed(bx, t, arg)
            }).collect();
            assert_eq!(inputs.len(), llargs.len());

            let val = match intr.definition {
                intrinsics::IntrinsicDef::Named(name) => {
                    let f = declare::declare_cfn(cx,
                                                 name,
                                                 Type::func(&inputs, &outputs));
                    bx.call(f, &llargs, None)
                }
            };

            match *intr.output {
                intrinsics::Type::Aggregate(flatten, ref elems) => {
                    // the output is a tuple so we need to munge it properly
                    assert!(!flatten);

                    for i in 0..elems.len() {
                        let dest = result.project_field(bx, i);
                        let val = bx.extract_value(val, i as u64);
                        bx.store(val, dest.llval, dest.align);
                    }
                    return;
                }
                _ => val,
            }
        }
    };

    if !fn_ty.ret.is_ignore() {
        if let PassMode::Cast(ty) = fn_ty.ret.mode {
            let ptr = bx.pointercast(result.llval, ty.llvm_type(cx).ptr_to());
            bx.store(llval, ptr, result.align);
        } else {
            OperandRef::from_immediate_or_packed_pair(bx, llval, result.layout)
                .val.store(bx, result);
        }
    }
}

fn copy_intrinsic<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                            allow_overlap: bool,
                            volatile: bool,
                            ty: Ty<'tcx>,
                            dst: ValueRef,
                            src: ValueRef,
                            count: ValueRef)
                            -> ValueRef {
    let cx = bx.cx;
    let (size, align) = cx.size_and_align_of(ty);
    let size = C_usize(cx, size.bytes());
    let align = C_i32(cx, align.abi() as i32);

    let operation = if allow_overlap {
        "memmove"
    } else {
        "memcpy"
    };

    let name = format!("llvm.{}.p0i8.p0i8.i{}", operation,
                       cx.data_layout().pointer_size.bits());

    let dst_ptr = bx.pointercast(dst, Type::i8p(cx));
    let src_ptr = bx.pointercast(src, Type::i8p(cx));
    let llfn = cx.get_intrinsic(&name);

    bx.call(llfn,
        &[dst_ptr,
        src_ptr,
        bx.mul(size, count),
        align,
        C_bool(cx, volatile)],
        None)
}

fn memset_intrinsic<'a, 'tcx>(
    bx: &Builder<'a, 'tcx>,
    volatile: bool,
    ty: Ty<'tcx>,
    dst: ValueRef,
    val: ValueRef,
    count: ValueRef
) -> ValueRef {
    let cx = bx.cx;
    let (size, align) = cx.size_and_align_of(ty);
    let size = C_usize(cx, size.bytes());
    let align = C_i32(cx, align.abi() as i32);
    let dst = bx.pointercast(dst, Type::i8p(cx));
    call_memset(bx, dst, val, bx.mul(size, count), align, volatile)
}

fn try_intrinsic<'a, 'tcx>(
    bx: &Builder<'a, 'tcx>,
    cx: &CodegenCx,
    func: ValueRef,
    data: ValueRef,
    local_ptr: ValueRef,
    dest: ValueRef,
) {
    if bx.sess().no_landing_pads() {
        bx.call(func, &[data], None);
        let ptr_align = bx.tcx().data_layout.pointer_align;
        bx.store(C_null(Type::i8p(&bx.cx)), dest, ptr_align);
    } else if wants_msvc_seh(bx.sess()) {
        trans_msvc_try(bx, cx, func, data, local_ptr, dest);
    } else {
        trans_gnu_try(bx, cx, func, data, local_ptr, dest);
    }
}

// MSVC's definition of the `rust_try` function.
//
// This implementation uses the new exception handling instructions in LLVM
// which have support in LLVM for SEH on MSVC targets. Although these
// instructions are meant to work for all targets, as of the time of this
// writing, however, LLVM does not recommend the usage of these new instructions
// as the old ones are still more optimized.
fn trans_msvc_try<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                            cx: &CodegenCx,
                            func: ValueRef,
                            data: ValueRef,
                            local_ptr: ValueRef,
                            dest: ValueRef) {
    let llfn = get_rust_try_fn(cx, &mut |bx| {
        let cx = bx.cx;

        bx.set_personality_fn(bx.cx.eh_personality());

        let normal = bx.build_sibling_block("normal");
        let catchswitch = bx.build_sibling_block("catchswitch");
        let catchpad = bx.build_sibling_block("catchpad");
        let caught = bx.build_sibling_block("caught");

        let func = llvm::get_param(bx.llfn(), 0);
        let data = llvm::get_param(bx.llfn(), 1);
        let local_ptr = llvm::get_param(bx.llfn(), 2);

        // We're generating an IR snippet that looks like:
        //
        //   declare i32 @rust_try(%func, %data, %ptr) {
        //      %slot = alloca i64*
        //      invoke %func(%data) to label %normal unwind label %catchswitch
        //
        //   normal:
        //      ret i32 0
        //
        //   catchswitch:
        //      %cs = catchswitch within none [%catchpad] unwind to caller
        //
        //   catchpad:
        //      %tok = catchpad within %cs [%type_descriptor, 0, %slot]
        //      %ptr[0] = %slot[0]
        //      %ptr[1] = %slot[1]
        //      catchret from %tok to label %caught
        //
        //   caught:
        //      ret i32 1
        //   }
        //
        // This structure follows the basic usage of throw/try/catch in LLVM.
        // For example, compile this C++ snippet to see what LLVM generates:
        //
        //      #include <stdint.h>
        //
        //      int bar(void (*foo)(void), uint64_t *ret) {
        //          try {
        //              foo();
        //              return 0;
        //          } catch(uint64_t a[2]) {
        //              ret[0] = a[0];
        //              ret[1] = a[1];
        //              return 1;
        //          }
        //      }
        //
        // More information can be found in libstd's seh.rs implementation.
        let i64p = Type::i64(cx).ptr_to();
        let ptr_align = bx.tcx().data_layout.pointer_align;
        let slot = bx.alloca(i64p, "slot", ptr_align);
        bx.invoke(func, &[data], normal.llbb(), catchswitch.llbb(),
            None);

        normal.ret(C_i32(cx, 0));

        let cs = catchswitch.catch_switch(None, None, 1);
        catchswitch.add_handler(cs, catchpad.llbb());

        let tcx = cx.tcx;
        let tydesc = match tcx.lang_items().msvc_try_filter() {
            Some(did) => ::consts::get_static(cx, did),
            None => bug!("msvc_try_filter not defined"),
        };
        let tok = catchpad.catch_pad(cs, &[tydesc, C_i32(cx, 0), slot]);
        let addr = catchpad.load(slot, ptr_align);

        let i64_align = bx.tcx().data_layout.i64_align;
        let arg1 = catchpad.load(addr, i64_align);
        let val1 = C_i32(cx, 1);
        let arg2 = catchpad.load(catchpad.inbounds_gep(addr, &[val1]), i64_align);
        let local_ptr = catchpad.bitcast(local_ptr, i64p);
        catchpad.store(arg1, local_ptr, i64_align);
        catchpad.store(arg2, catchpad.inbounds_gep(local_ptr, &[val1]), i64_align);
        catchpad.catch_ret(tok, caught.llbb());

        caught.ret(C_i32(cx, 1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llfn, &[func, data, local_ptr], None);
    let i32_align = bx.tcx().data_layout.i32_align;
    bx.store(ret, dest, i32_align);
}

// Definition of the standard "try" function for Rust using the GNU-like model
// of exceptions (e.g. the normal semantics of LLVM's landingpad and invoke
// instructions).
//
// This translation is a little surprising because we always call a shim
// function instead of inlining the call to `invoke` manually here. This is done
// because in LLVM we're only allowed to have one personality per function
// definition. The call to the `try` intrinsic is being inlined into the
// function calling it, and that function may already have other personality
// functions in play. By calling a shim we're guaranteed that our shim will have
// the right personality function.
fn trans_gnu_try<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                           cx: &CodegenCx,
                           func: ValueRef,
                           data: ValueRef,
                           local_ptr: ValueRef,
                           dest: ValueRef) {
    let llfn = get_rust_try_fn(cx, &mut |bx| {
        let cx = bx.cx;

        // Translates the shims described above:
        //
        //   bx:
        //      invoke %func(%args...) normal %normal unwind %catch
        //
        //   normal:
        //      ret 0
        //
        //   catch:
        //      (ptr, _) = landingpad
        //      store ptr, %local_ptr
        //      ret 1
        //
        // Note that the `local_ptr` data passed into the `try` intrinsic is
        // expected to be `*mut *mut u8` for this to actually work, but that's
        // managed by the standard library.

        let then = bx.build_sibling_block("then");
        let catch = bx.build_sibling_block("catch");

        let func = llvm::get_param(bx.llfn(), 0);
        let data = llvm::get_param(bx.llfn(), 1);
        let local_ptr = llvm::get_param(bx.llfn(), 2);
        bx.invoke(func, &[data], then.llbb(), catch.llbb(), None);
        then.ret(C_i32(cx, 0));

        // Type indicator for the exception being thrown.
        //
        // The first value in this tuple is a pointer to the exception object
        // being thrown.  The second value is a "selector" indicating which of
        // the landing pad clauses the exception's type had been matched to.
        // rust_try ignores the selector.
        let lpad_ty = Type::struct_(cx, &[Type::i8p(cx), Type::i32(cx)],
                                    false);
        let vals = catch.landing_pad(lpad_ty, bx.cx.eh_personality(), 1);
        catch.add_clause(vals, C_null(Type::i8p(cx)));
        let ptr = catch.extract_value(vals, 0);
        let ptr_align = bx.tcx().data_layout.pointer_align;
        catch.store(ptr, catch.bitcast(local_ptr, Type::i8p(cx).ptr_to()), ptr_align);
        catch.ret(C_i32(cx, 1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bx.call(llfn, &[func, data, local_ptr], None);
    let i32_align = bx.tcx().data_layout.i32_align;
    bx.store(ret, dest, i32_align);
}

// Helper function to give a Block to a closure to translate a shim function.
// This is currently primarily used for the `try` intrinsic functions above.
fn gen_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                    name: &str,
                    inputs: Vec<Ty<'tcx>>,
                    output: Ty<'tcx>,
                    trans: &mut for<'b> FnMut(Builder<'b, 'tcx>))
                    -> ValueRef {
    let rust_fn_ty = cx.tcx.mk_fn_ptr(ty::Binder(cx.tcx.mk_fn_sig(
        inputs.into_iter(),
        output,
        false,
        hir::Unsafety::Unsafe,
        Abi::Rust
    )));
    let llfn = declare::define_internal_fn(cx, name, rust_fn_ty);
    let bx = Builder::new_block(cx, llfn, "entry-block");
    trans(bx);
    llfn
}

// Helper function used to get a handle to the `__rust_try` function used to
// catch exceptions.
//
// This function is only generated once and is then cached.
fn get_rust_try_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                             trans: &mut for<'b> FnMut(Builder<'b, 'tcx>))
                             -> ValueRef {
    if let Some(llfn) = cx.rust_try_fn.get() {
        return llfn;
    }

    // Define the type up front for the signature of the rust_try function.
    let tcx = cx.tcx;
    let i8p = tcx.mk_mut_ptr(tcx.types.i8);
    let fn_ty = tcx.mk_fn_ptr(ty::Binder(tcx.mk_fn_sig(
        iter::once(i8p),
        tcx.mk_nil(),
        false,
        hir::Unsafety::Unsafe,
        Abi::Rust
    )));
    let output = tcx.types.i32;
    let rust_try = gen_fn(cx, "__rust_try", vec![fn_ty, i8p, i8p], output, trans);
    cx.rust_try_fn.set(Some(rust_try));
    return rust_try
}

fn span_invalid_monomorphization_error(a: &Session, b: Span, c: &str) {
    span_err!(a, b, E0511, "{}", c);
}

fn generic_simd_intrinsic<'a, 'tcx>(
    bx: &Builder<'a, 'tcx>,
    name: &str,
    callee_ty: Ty<'tcx>,
    args: &[OperandRef<'tcx>],
    ret_ty: Ty<'tcx>,
    llret_ty: Type,
    span: Span
) -> Result<ValueRef, ()> {
    // macros for error handling:
    macro_rules! emit_error {
        ($msg: tt) => {
            emit_error!($msg, )
        };
        ($msg: tt, $($fmt: tt)*) => {
            span_invalid_monomorphization_error(
                bx.sess(), span,
                &format!(concat!("invalid monomorphization of `{}` intrinsic: ",
                                 $msg),
                         name, $($fmt)*));
        }
    }
    macro_rules! require {
        ($cond: expr, $($fmt: tt)*) => {
            if !$cond {
                emit_error!($($fmt)*);
                return Err(());
            }
        }
    }
    macro_rules! require_simd {
        ($ty: expr, $position: expr) => {
            require!($ty.is_simd(), "expected SIMD {} type, found non-SIMD `{}`", $position, $ty)
        }
    }



    let tcx = bx.tcx();
    let sig = tcx.erase_late_bound_regions_and_normalize(&callee_ty.fn_sig(tcx));
    let arg_tys = sig.inputs();

    // every intrinsic takes a SIMD vector as its first argument
    require_simd!(arg_tys[0], "input");
    let in_ty = arg_tys[0];
    let in_elem = arg_tys[0].simd_type(tcx);
    let in_len = arg_tys[0].simd_size(tcx);

    let comparison = match name {
        "simd_eq" => Some(hir::BiEq),
        "simd_ne" => Some(hir::BiNe),
        "simd_lt" => Some(hir::BiLt),
        "simd_le" => Some(hir::BiLe),
        "simd_gt" => Some(hir::BiGt),
        "simd_ge" => Some(hir::BiGe),
        _ => None
    };

    if let Some(cmp_op) = comparison {
        require_simd!(ret_ty, "return");

        let out_len = ret_ty.simd_size(tcx);
        require!(in_len == out_len,
                 "expected return type with length {} (same as input type `{}`), \
                  found `{}` with length {}",
                 in_len, in_ty,
                 ret_ty, out_len);
        require!(llret_ty.element_type().kind() == llvm::Integer,
                 "expected return type with integer elements, found `{}` with non-integer `{}`",
                 ret_ty,
                 ret_ty.simd_type(tcx));

        return Ok(compare_simd_types(bx,
                                     args[0].immediate(),
                                     args[1].immediate(),
                                     in_elem,
                                     llret_ty,
                                     cmp_op))
    }

    if name.starts_with("simd_shuffle") {
        let n: usize = match name["simd_shuffle".len()..].parse() {
            Ok(n) => n,
            Err(_) => span_bug!(span,
                                "bad `simd_shuffle` instruction only caught in trans?")
        };

        require_simd!(ret_ty, "return");

        let out_len = ret_ty.simd_size(tcx);
        require!(out_len == n,
                 "expected return type of length {}, found `{}` with length {}",
                 n, ret_ty, out_len);
        require!(in_elem == ret_ty.simd_type(tcx),
                 "expected return element type `{}` (element of input `{}`), \
                  found `{}` with element type `{}`",
                 in_elem, in_ty,
                 ret_ty, ret_ty.simd_type(tcx));

        let total_len = in_len as u128 * 2;

        let vector = args[2].immediate();

        let indices: Option<Vec<_>> = (0..n)
            .map(|i| {
                let arg_idx = i;
                let val = const_get_elt(vector, i as u64);
                match const_to_opt_u128(val, true) {
                    None => {
                        emit_error!("shuffle index #{} is not a constant", arg_idx);
                        None
                    }
                    Some(idx) if idx >= total_len => {
                        emit_error!("shuffle index #{} is out of bounds (limit {})",
                                    arg_idx, total_len);
                        None
                    }
                    Some(idx) => Some(C_i32(bx.cx, idx as i32)),
                }
            })
            .collect();
        let indices = match indices {
            Some(i) => i,
            None => return Ok(C_null(llret_ty))
        };

        return Ok(bx.shuffle_vector(args[0].immediate(),
                                     args[1].immediate(),
                                     C_vector(&indices)))
    }

    if name == "simd_insert" {
        require!(in_elem == arg_tys[2],
                 "expected inserted type `{}` (element of input `{}`), found `{}`",
                 in_elem, in_ty, arg_tys[2]);
        return Ok(bx.insert_element(args[0].immediate(),
                                     args[2].immediate(),
                                     args[1].immediate()))
    }
    if name == "simd_extract" {
        require!(ret_ty == in_elem,
                 "expected return type `{}` (element of input `{}`), found `{}`",
                 in_elem, in_ty, ret_ty);
        return Ok(bx.extract_element(args[0].immediate(), args[1].immediate()))
    }

    if name == "simd_cast" {
        require_simd!(ret_ty, "return");
        let out_len = ret_ty.simd_size(tcx);
        require!(in_len == out_len,
                 "expected return type with length {} (same as input type `{}`), \
                  found `{}` with length {}",
                 in_len, in_ty,
                 ret_ty, out_len);
        // casting cares about nominal type, not just structural type
        let out_elem = ret_ty.simd_type(tcx);

        if in_elem == out_elem { return Ok(args[0].immediate()); }

        enum Style { Float, Int(/* is signed? */ bool), Unsupported }

        let (in_style, in_width) = match in_elem.sty {
            // vectors of pointer-sized integers should've been
            // disallowed before here, so this unwrap is safe.
            ty::TyInt(i) => (Style::Int(true), i.bit_width().unwrap()),
            ty::TyUint(u) => (Style::Int(false), u.bit_width().unwrap()),
            ty::TyFloat(f) => (Style::Float, f.bit_width()),
            _ => (Style::Unsupported, 0)
        };
        let (out_style, out_width) = match out_elem.sty {
            ty::TyInt(i) => (Style::Int(true), i.bit_width().unwrap()),
            ty::TyUint(u) => (Style::Int(false), u.bit_width().unwrap()),
            ty::TyFloat(f) => (Style::Float, f.bit_width()),
            _ => (Style::Unsupported, 0)
        };

        match (in_style, out_style) {
            (Style::Int(in_is_signed), Style::Int(_)) => {
                return Ok(match in_width.cmp(&out_width) {
                    Ordering::Greater => bx.trunc(args[0].immediate(), llret_ty),
                    Ordering::Equal => args[0].immediate(),
                    Ordering::Less => if in_is_signed {
                        bx.sext(args[0].immediate(), llret_ty)
                    } else {
                        bx.zext(args[0].immediate(), llret_ty)
                    }
                })
            }
            (Style::Int(in_is_signed), Style::Float) => {
                return Ok(if in_is_signed {
                    bx.sitofp(args[0].immediate(), llret_ty)
                } else {
                    bx.uitofp(args[0].immediate(), llret_ty)
                })
            }
            (Style::Float, Style::Int(out_is_signed)) => {
                return Ok(if out_is_signed {
                    bx.fptosi(args[0].immediate(), llret_ty)
                } else {
                    bx.fptoui(args[0].immediate(), llret_ty)
                })
            }
            (Style::Float, Style::Float) => {
                return Ok(match in_width.cmp(&out_width) {
                    Ordering::Greater => bx.fptrunc(args[0].immediate(), llret_ty),
                    Ordering::Equal => args[0].immediate(),
                    Ordering::Less => bx.fpext(args[0].immediate(), llret_ty)
                })
            }
            _ => {/* Unsupported. Fallthrough. */}
        }
        require!(false,
                 "unsupported cast from `{}` with element `{}` to `{}` with element `{}`",
                 in_ty, in_elem,
                 ret_ty, out_elem);
    }
    macro_rules! arith {
        ($($name: ident: $($($p: ident),* => $call: ident),*;)*) => {
            $(if name == stringify!($name) {
                match in_elem.sty {
                    $($(ty::$p(_))|* => {
                        return Ok(bx.$call(args[0].immediate(), args[1].immediate()))
                    })*
                    _ => {},
                }
                require!(false,
                            "unsupported operation on `{}` with element `{}`",
                            in_ty,
                            in_elem)
            })*
        }
    }
    arith! {
        simd_add: TyUint, TyInt => add, TyFloat => fadd;
        simd_sub: TyUint, TyInt => sub, TyFloat => fsub;
        simd_mul: TyUint, TyInt => mul, TyFloat => fmul;
        simd_div: TyUint => udiv, TyInt => sdiv, TyFloat => fdiv;
        simd_rem: TyUint => urem, TyInt => srem, TyFloat => frem;
        simd_shl: TyUint, TyInt => shl;
        simd_shr: TyUint => lshr, TyInt => ashr;
        simd_and: TyUint, TyInt => and;
        simd_or: TyUint, TyInt => or;
        simd_xor: TyUint, TyInt => xor;
    }
    span_bug!(span, "unknown SIMD intrinsic");
}

// Returns the width of an int Ty, and if it's signed or not
// Returns None if the type is not an integer
// FIXME: there’s multiple of this functions, investigate using some of the already existing
// stuffs.
fn int_type_width_signed(ty: Ty, cx: &CodegenCx) -> Option<(u64, bool)> {
    match ty.sty {
        ty::TyInt(t) => Some((match t {
            ast::IntTy::Isize => {
                match &cx.tcx.sess.target.target.target_pointer_width[..] {
                    "16" => 16,
                    "32" => 32,
                    "64" => 64,
                    tws => bug!("Unsupported target word size for isize: {}", tws),
                }
            },
            ast::IntTy::I8 => 8,
            ast::IntTy::I16 => 16,
            ast::IntTy::I32 => 32,
            ast::IntTy::I64 => 64,
            ast::IntTy::I128 => 128,
        }, true)),
        ty::TyUint(t) => Some((match t {
            ast::UintTy::Usize => {
                match &cx.tcx.sess.target.target.target_pointer_width[..] {
                    "16" => 16,
                    "32" => 32,
                    "64" => 64,
                    tws => bug!("Unsupported target word size for usize: {}", tws),
                }
            },
            ast::UintTy::U8 => 8,
            ast::UintTy::U16 => 16,
            ast::UintTy::U32 => 32,
            ast::UintTy::U64 => 64,
            ast::UintTy::U128 => 128,
        }, false)),
        _ => None,
    }
}

// Returns the width of a float TypeVariant
// Returns None if the type is not a float
fn float_type_width<'tcx>(sty: &ty::TypeVariants<'tcx>)
        -> Option<u64> {
    use rustc::ty::TyFloat;
    match *sty {
        TyFloat(t) => Some(match t {
            ast::FloatTy::F32 => 32,
            ast::FloatTy::F64 => 64,
        }),
        _ => None,
    }
}
