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
use abi::{Abi, FnType, LlvmType, PassMode};
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
/// add them to librustc_codegen_llvm/context.rs
pub fn codegen_intrinsic_call<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                                      callee_ty: Ty<'tcx>,
                                      fn_ty: &FnType<'tcx, Ty<'tcx>>,
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
    let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
    let arg_tys = sig.inputs();
    let ret_ty = sig.output();
    let name = &*tcx.item_name(def_id).as_str();

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
            let dst = args[0].deref(bx.cx);
            args[1].val.volatile_store(bx, dst);
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
        "bitreverse" | "add_with_overflow" | "sub_with_overflow" |
        "mul_with_overflow" | "overflowing_add" | "overflowing_sub" | "overflowing_mul" |
        "unchecked_div" | "unchecked_rem" | "unchecked_shl" | "unchecked_shr" | "exact_div" => {
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
                        "bitreverse" => {
                            bx.call(cx.get_intrinsic(&format!("llvm.bitreverse.i{}", width)),
                                &[args[0].immediate()], None)
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
                        "exact_div" =>
                            if signed {
                                bx.exactsdiv(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.exactudiv(args[0].immediate(), args[1].immediate())
                            },
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
            args[0].deref(bx.cx).codegen_get_discr(bx, ret_ty)
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
            let dst = args[0].deref(bx.cx);
            args[1].val.nontemporal_store(bx, dst);
            return;
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
        codegen_msvc_try(bx, cx, func, data, local_ptr, dest);
    } else {
        codegen_gnu_try(bx, cx, func, data, local_ptr, dest);
    }
}

// MSVC's definition of the `rust_try` function.
//
// This implementation uses the new exception handling instructions in LLVM
// which have support in LLVM for SEH on MSVC targets. Although these
// instructions are meant to work for all targets, as of the time of this
// writing, however, LLVM does not recommend the usage of these new instructions
// as the old ones are still more optimized.
fn codegen_msvc_try<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
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
// This codegen is a little surprising because we always call a shim
// function instead of inlining the call to `invoke` manually here. This is done
// because in LLVM we're only allowed to have one personality per function
// definition. The call to the `try` intrinsic is being inlined into the
// function calling it, and that function may already have other personality
// functions in play. By calling a shim we're guaranteed that our shim will have
// the right personality function.
fn codegen_gnu_try<'a, 'tcx>(bx: &Builder<'a, 'tcx>,
                           cx: &CodegenCx,
                           func: ValueRef,
                           data: ValueRef,
                           local_ptr: ValueRef,
                           dest: ValueRef) {
    let llfn = get_rust_try_fn(cx, &mut |bx| {
        let cx = bx.cx;

        // Codegens the shims described above:
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

// Helper function to give a Block to a closure to codegen a shim function.
// This is currently primarily used for the `try` intrinsic functions above.
fn gen_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                    name: &str,
                    inputs: Vec<Ty<'tcx>>,
                    output: Ty<'tcx>,
                    codegen: &mut for<'b> FnMut(Builder<'b, 'tcx>))
                    -> ValueRef {
    let rust_fn_ty = cx.tcx.mk_fn_ptr(ty::Binder::bind(cx.tcx.mk_fn_sig(
        inputs.into_iter(),
        output,
        false,
        hir::Unsafety::Unsafe,
        Abi::Rust
    )));
    let llfn = declare::define_internal_fn(cx, name, rust_fn_ty);
    let bx = Builder::new_block(cx, llfn, "entry-block");
    codegen(bx);
    llfn
}

// Helper function used to get a handle to the `__rust_try` function used to
// catch exceptions.
//
// This function is only generated once and is then cached.
fn get_rust_try_fn<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                             codegen: &mut for<'b> FnMut(Builder<'b, 'tcx>))
                             -> ValueRef {
    if let Some(llfn) = cx.rust_try_fn.get() {
        return llfn;
    }

    // Define the type up front for the signature of the rust_try function.
    let tcx = cx.tcx;
    let i8p = tcx.mk_mut_ptr(tcx.types.i8);
    let fn_ty = tcx.mk_fn_ptr(ty::Binder::bind(tcx.mk_fn_sig(
        iter::once(i8p),
        tcx.mk_nil(),
        false,
        hir::Unsafety::Unsafe,
        Abi::Rust
    )));
    let output = tcx.types.i32;
    let rust_try = gen_fn(cx, "__rust_try", vec![fn_ty, i8p, i8p], output, codegen);
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
    macro_rules! return_error {
        ($($fmt: tt)*) => {
            {
                emit_error!($($fmt)*);
                return Err(());
            }
        }
    }

    macro_rules! require {
        ($cond: expr, $($fmt: tt)*) => {
            if !$cond {
                return_error!($($fmt)*);
            }
        };
    }
    macro_rules! require_simd {
        ($ty: expr, $position: expr) => {
            require!($ty.is_simd(), "expected SIMD {} type, found non-SIMD `{}`", $position, $ty)
        }
    }



    let tcx = bx.tcx();
    let sig = tcx.normalize_erasing_late_bound_regions(
        ty::ParamEnv::reveal_all(),
        &callee_ty.fn_sig(tcx),
    );
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
                                "bad `simd_shuffle` instruction only caught in codegen?")
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

    if name == "simd_select" {
        let m_elem_ty = in_elem;
        let m_len = in_len;
        let v_len = arg_tys[1].simd_size(tcx);
        require!(m_len == v_len,
                 "mismatched lengths: mask length `{}` != other vector length `{}`",
                 m_len, v_len
        );
        match m_elem_ty.sty {
            ty::TyInt(_) => {},
            _ => {
                return_error!("mask element type is `{}`, expected `i_`", m_elem_ty);
            }
        }
        // truncate the mask to a vector of i1s
        let i1 = Type::i1(bx.cx);
        let i1xn = Type::vector(&i1, m_len as u64);
        let m_i1s = bx.trunc(args[0].immediate(), i1xn);
        return Ok(bx.select(m_i1s, args[1].immediate(), args[2].immediate()));
    }

    fn simd_simple_float_intrinsic<'a, 'tcx>(name: &str,
                                             in_elem: &::rustc::ty::TyS,
                                             in_ty: &::rustc::ty::TyS,
                                             in_len: usize,
                                             bx: &Builder<'a, 'tcx>,
                                             span: Span,
                                             args: &[OperandRef<'tcx>])
                                             -> Result<ValueRef, ()> {
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
        macro_rules! return_error {
            ($($fmt: tt)*) => {
                {
                    emit_error!($($fmt)*);
                    return Err(());
                }
            }
        }
        let ety = match in_elem.sty {
            ty::TyFloat(f) if f.bit_width() == 32 => {
                if in_len < 2 || in_len > 16 {
                    return_error!(
                        "unsupported floating-point vector `{}` with length `{}` \
                         out-of-range [2, 16]",
                        in_ty, in_len);
                }
                "f32"
            },
            ty::TyFloat(f) if f.bit_width() == 64 => {
                if in_len < 2 || in_len > 8 {
                    return_error!("unsupported floating-point vector `{}` with length `{}` \
                                   out-of-range [2, 8]",
                                  in_ty, in_len);
                }
                "f64"
            },
            ty::TyFloat(f) => {
                return_error!("unsupported element type `{}` of floating-point vector `{}`",
                              f, in_ty);
            },
            _ => {
                return_error!("`{}` is not a floating-point type", in_ty);
            }
        };

        let llvm_name = &format!("llvm.{0}.v{1}{2}", name, in_len, ety);
        let intrinsic = bx.cx.get_intrinsic(&llvm_name);
        let c = bx.call(intrinsic,
                        &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                        None);
        unsafe { llvm::LLVMRustSetHasUnsafeAlgebra(c) };
        return Ok(c);
    }

    if name == "simd_fsqrt" {
        return simd_simple_float_intrinsic("sqrt", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fsin" {
        return simd_simple_float_intrinsic("sin", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fcos" {
        return simd_simple_float_intrinsic("cos", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fabs" {
        return simd_simple_float_intrinsic("fabs", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_floor" {
        return simd_simple_float_intrinsic("floor", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_ceil" {
        return simd_simple_float_intrinsic("ceil", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fexp" {
        return simd_simple_float_intrinsic("exp", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fexp2" {
        return simd_simple_float_intrinsic("exp2", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_flog10" {
        return simd_simple_float_intrinsic("log10", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_flog2" {
        return simd_simple_float_intrinsic("log2", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_flog" {
        return simd_simple_float_intrinsic("log", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fpowi" {
        return simd_simple_float_intrinsic("powi", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fpow"  {
        return simd_simple_float_intrinsic("pow", in_elem, in_ty, in_len, bx, span, args);
    }

    if name == "simd_fma" {
        return simd_simple_float_intrinsic("fma", in_elem, in_ty, in_len, bx, span, args);
    }

    // FIXME: use:
    //  https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Function.h#L182
    //  https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Intrinsics.h#L81
    fn llvm_vector_str(elem_ty: ty::Ty, vec_len: usize, no_pointers: usize) -> String {
        let p0s: String = "p0".repeat(no_pointers);
        match elem_ty.sty {
            ty::TyInt(v) => format!("v{}{}i{}", vec_len, p0s, v.bit_width().unwrap()),
            ty::TyUint(v) => format!("v{}{}i{}", vec_len, p0s, v.bit_width().unwrap()),
            ty::TyFloat(v) => format!("v{}{}f{}", vec_len, p0s, v.bit_width()),
            _ => unreachable!(),
        }
    }

    fn llvm_vector_ty(cx: &CodegenCx, elem_ty: ty::Ty, vec_len: usize,
                      mut no_pointers: usize) -> Type {
        // FIXME: use cx.layout_of(ty).llvm_type() ?
        let mut elem_ty = match elem_ty.sty {
            ty::TyInt(v) => Type::int_from_ty(cx, v),
            ty::TyUint(v) => Type::uint_from_ty(cx, v),
            ty::TyFloat(v) => Type::float_from_ty(cx, v),
            _ => unreachable!(),
        };
        while no_pointers > 0 {
            elem_ty = elem_ty.ptr_to();
            no_pointers -= 1;
        }
        Type::vector(&elem_ty, vec_len as u64)
    }


    if name == "simd_gather"  {
        // simd_gather(values: <N x T>, pointers: <N x *_ T>,
        //             mask: <N x i{M}>) -> <N x T>
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        require_simd!(in_ty, "first");
        require_simd!(arg_tys[1], "second");
        require_simd!(arg_tys[2], "third");
        require_simd!(ret_ty, "return");

        // Of the same length:
        require!(in_len == arg_tys[1].simd_size(tcx),
                 "expected {} argument with length {} (same as input type `{}`), \
                  found `{}` with length {}", "second", in_len, in_ty, arg_tys[1],
                 arg_tys[1].simd_size(tcx));
        require!(in_len == arg_tys[2].simd_size(tcx),
                 "expected {} argument with length {} (same as input type `{}`), \
                  found `{}` with length {}", "third", in_len, in_ty, arg_tys[2],
                 arg_tys[2].simd_size(tcx));

        // The return type must match the first argument type
        require!(ret_ty == in_ty,
                 "expected return type `{}`, found `{}`",
                 in_ty, ret_ty);

        // This counts how many pointers
        fn ptr_count(t: ty::Ty) -> usize {
            match t.sty {
                ty::TyRawPtr(p) => 1 + ptr_count(p.ty),
                _ => 0,
            }
        }

        // Non-ptr type
        fn non_ptr(t: ty::Ty) -> ty::Ty {
            match t.sty {
                ty::TyRawPtr(p) => non_ptr(p.ty),
                _ => t,
            }
        }

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (pointer_count, underlying_ty) = match arg_tys[1].simd_type(tcx).sty {
            ty::TyRawPtr(p) if p.ty == in_elem => (ptr_count(arg_tys[1].simd_type(tcx)),
                                                   non_ptr(arg_tys[1].simd_type(tcx))),
            _ => {
                require!(false, "expected element type `{}` of second argument `{}` \
                                 to be a pointer to the element type `{}` of the first \
                                 argument `{}`, found `{}` != `*_ {}`",
                         arg_tys[1].simd_type(tcx).sty, arg_tys[1], in_elem, in_ty,
                         arg_tys[1].simd_type(tcx).sty, in_elem);
                unreachable!();
            }
        };
        assert!(pointer_count > 0);
        assert!(pointer_count - 1 == ptr_count(arg_tys[0].simd_type(tcx)));
        assert_eq!(underlying_ty, non_ptr(arg_tys[0].simd_type(tcx)));

        // The element type of the third argument must be a signed integer type of any width:
        match arg_tys[2].simd_type(tcx).sty {
            ty::TyInt(_) => (),
            _ => {
                require!(false, "expected element type `{}` of third argument `{}` \
                                 to be a signed integer type",
                         arg_tys[2].simd_type(tcx).sty, arg_tys[2]);
            }
        }

        // Alignment of T, must be a constant integer value:
        let alignment_ty = Type::i32(bx.cx);
        let alignment = C_i32(bx.cx, bx.cx.align_of(in_elem).abi() as i32);

        // Truncate the mask vector to a vector of i1s:
        let (mask, mask_ty) = {
            let i1 = Type::i1(bx.cx);
            let i1xn = Type::vector(&i1, in_len as u64);
            (bx.trunc(args[2].immediate(), i1xn), i1xn)
        };

        // Type of the vector of pointers:
        let llvm_pointer_vec_ty = llvm_vector_ty(bx.cx, underlying_ty, in_len, pointer_count);
        let llvm_pointer_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count);

        // Type of the vector of elements:
        let llvm_elem_vec_ty = llvm_vector_ty(bx.cx, underlying_ty, in_len, pointer_count - 1);
        let llvm_elem_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count - 1);

        let llvm_intrinsic = format!("llvm.masked.gather.{}.{}",
                                     llvm_elem_vec_str, llvm_pointer_vec_str);
        let f = declare::declare_cfn(bx.cx, &llvm_intrinsic,
                                     Type::func(&[llvm_pointer_vec_ty, alignment_ty, mask_ty,
                                                  llvm_elem_vec_ty], &llvm_elem_vec_ty));
        llvm::SetUnnamedAddr(f, false);
        let v = bx.call(f, &[args[1].immediate(), alignment, mask, args[0].immediate()],
                        None);
        return Ok(v);
    }

    if name == "simd_scatter"  {
        // simd_scatter(values: <N x T>, pointers: <N x *mut T>,
        //             mask: <N x i{M}>) -> ()
        // * N: number of elements in the input vectors
        // * T: type of the element to load
        // * M: any integer width is supported, will be truncated to i1

        // All types must be simd vector types
        require_simd!(in_ty, "first");
        require_simd!(arg_tys[1], "second");
        require_simd!(arg_tys[2], "third");

        // Of the same length:
        require!(in_len == arg_tys[1].simd_size(tcx),
                 "expected {} argument with length {} (same as input type `{}`), \
                  found `{}` with length {}", "second", in_len, in_ty, arg_tys[1],
                 arg_tys[1].simd_size(tcx));
        require!(in_len == arg_tys[2].simd_size(tcx),
                 "expected {} argument with length {} (same as input type `{}`), \
                  found `{}` with length {}", "third", in_len, in_ty, arg_tys[2],
                 arg_tys[2].simd_size(tcx));

        // This counts how many pointers
        fn ptr_count(t: ty::Ty) -> usize {
            match t.sty {
                ty::TyRawPtr(p) => 1 + ptr_count(p.ty),
                _ => 0,
            }
        }

        // Non-ptr type
        fn non_ptr(t: ty::Ty) -> ty::Ty {
            match t.sty {
                ty::TyRawPtr(p) => non_ptr(p.ty),
                _ => t,
            }
        }

        // The second argument must be a simd vector with an element type that's a pointer
        // to the element type of the first argument
        let (pointer_count, underlying_ty) = match arg_tys[1].simd_type(tcx).sty {
            ty::TyRawPtr(p) if p.ty == in_elem && p.mutbl == hir::MutMutable
                => (ptr_count(arg_tys[1].simd_type(tcx)),
                    non_ptr(arg_tys[1].simd_type(tcx))),
            _ => {
                require!(false, "expected element type `{}` of second argument `{}` \
                                 to be a pointer to the element type `{}` of the first \
                                 argument `{}`, found `{}` != `*mut {}`",
                         arg_tys[1].simd_type(tcx).sty, arg_tys[1], in_elem, in_ty,
                         arg_tys[1].simd_type(tcx).sty, in_elem);
                unreachable!();
            }
        };
        assert!(pointer_count > 0);
        assert!(pointer_count - 1 == ptr_count(arg_tys[0].simd_type(tcx)));
        assert_eq!(underlying_ty, non_ptr(arg_tys[0].simd_type(tcx)));

        // The element type of the third argument must be a signed integer type of any width:
        match arg_tys[2].simd_type(tcx).sty {
            ty::TyInt(_) => (),
            _ => {
                require!(false, "expected element type `{}` of third argument `{}` \
                                 to be a signed integer type",
                         arg_tys[2].simd_type(tcx).sty, arg_tys[2]);
            }
        }

        // Alignment of T, must be a constant integer value:
        let alignment_ty = Type::i32(bx.cx);
        let alignment = C_i32(bx.cx, bx.cx.align_of(in_elem).abi() as i32);

        // Truncate the mask vector to a vector of i1s:
        let (mask, mask_ty) = {
            let i1 = Type::i1(bx.cx);
            let i1xn = Type::vector(&i1, in_len as u64);
            (bx.trunc(args[2].immediate(), i1xn), i1xn)
        };

        let ret_t = Type::void(bx.cx);

        // Type of the vector of pointers:
        let llvm_pointer_vec_ty = llvm_vector_ty(bx.cx, underlying_ty, in_len, pointer_count);
        let llvm_pointer_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count);

        // Type of the vector of elements:
        let llvm_elem_vec_ty = llvm_vector_ty(bx.cx, underlying_ty, in_len, pointer_count - 1);
        let llvm_elem_vec_str = llvm_vector_str(underlying_ty, in_len, pointer_count - 1);

        let llvm_intrinsic = format!("llvm.masked.scatter.{}.{}",
                                     llvm_elem_vec_str, llvm_pointer_vec_str);
        let f = declare::declare_cfn(bx.cx, &llvm_intrinsic,
                                     Type::func(&[llvm_elem_vec_ty,
                                                  llvm_pointer_vec_ty,
                                                  alignment_ty,
                                                  mask_ty], &ret_t));
        llvm::SetUnnamedAddr(f, false);
        let v = bx.call(f, &[args[0].immediate(), args[1].immediate(), alignment, mask],
                        None);
        return Ok(v);
    }

    macro_rules! arith_red {
        ($name:tt : $integer_reduce:ident, $float_reduce:ident, $ordered:expr) => {
            if name == $name {
                require!(ret_ty == in_elem,
                         "expected return type `{}` (element of input `{}`), found `{}`",
                         in_elem, in_ty, ret_ty);
                return match in_elem.sty {
                    ty::TyInt(_) | ty::TyUint(_) => {
                        let r = bx.$integer_reduce(args[0].immediate());
                        if $ordered {
                            // if overflow occurs, the result is the
                            // mathematical result modulo 2^n:
                            if name.contains("mul") {
                                Ok(bx.mul(args[1].immediate(), r))
                            } else {
                                Ok(bx.add(args[1].immediate(), r))
                            }
                        } else {
                            Ok(bx.$integer_reduce(args[0].immediate()))
                        }
                    },
                    ty::TyFloat(f) => {
                        // ordered arithmetic reductions take an accumulator
                        let acc = if $ordered {
                            let acc = args[1].immediate();
                            // FIXME: https://bugs.llvm.org/show_bug.cgi?id=36734
                            // * if the accumulator of the fadd isn't 0, incorrect
                            //   code is generated
                            // * if the accumulator of the fmul isn't 1, incorrect
                            //   code is generated
                            match const_get_real(acc) {
                                None => return_error!("accumulator of {} is not a constant", $name),
                                Some((v, loses_info)) => {
                                    if $name.contains("mul") && v != 1.0_f64 {
                                        return_error!("accumulator of {} is not 1.0", $name);
                                    } else if $name.contains("add") && v != 0.0_f64 {
                                        return_error!("accumulator of {} is not 0.0", $name);
                                    } else if loses_info {
                                        return_error!("accumulator of {} loses information", $name);
                                    }
                                }
                            }
                            acc
                        } else {
                            // unordered arithmetic reductions do not:
                            match f.bit_width() {
                                32 => C_undef(Type::f32(bx.cx)),
                                64 => C_undef(Type::f64(bx.cx)),
                                v => {
                                    return_error!(r#"
unsupported {} from `{}` with element `{}` of size `{}` to `{}`"#,
                                        $name, in_ty, in_elem, v, ret_ty
                                    )
                                }
                            }

                        };
                        Ok(bx.$float_reduce(acc, args[0].immediate()))
                    }
                    _ => {
                        return_error!(
                            "unsupported {} from `{}` with element `{}` to `{}`",
                            $name, in_ty, in_elem, ret_ty
                        )
                    },
                }
            }
        }
    }

    arith_red!("simd_reduce_add_ordered": vector_reduce_add, vector_reduce_fadd_fast, true);
    arith_red!("simd_reduce_mul_ordered": vector_reduce_mul, vector_reduce_fmul_fast, true);
    arith_red!("simd_reduce_add_unordered": vector_reduce_add, vector_reduce_fadd_fast, false);
    arith_red!("simd_reduce_mul_unordered": vector_reduce_mul, vector_reduce_fmul_fast, false);

    macro_rules! minmax_red {
        ($name:tt: $int_red:ident, $float_red:ident) => {
            if name == $name {
                require!(ret_ty == in_elem,
                         "expected return type `{}` (element of input `{}`), found `{}`",
                         in_elem, in_ty, ret_ty);
                return match in_elem.sty {
                    ty::TyInt(_i) => {
                        Ok(bx.$int_red(args[0].immediate(), true))
                    },
                    ty::TyUint(_u) => {
                        Ok(bx.$int_red(args[0].immediate(), false))
                    },
                    ty::TyFloat(_f) => {
                        Ok(bx.$float_red(args[0].immediate()))
                    }
                    _ => {
                        return_error!("unsupported {} from `{}` with element `{}` to `{}`",
                                      $name, in_ty, in_elem, ret_ty)
                    },
                }
            }

        }
    }

    minmax_red!("simd_reduce_min": vector_reduce_min, vector_reduce_fmin);
    minmax_red!("simd_reduce_max": vector_reduce_max, vector_reduce_fmax);

    minmax_red!("simd_reduce_min_nanless": vector_reduce_min, vector_reduce_fmin_fast);
    minmax_red!("simd_reduce_max_nanless": vector_reduce_max, vector_reduce_fmax_fast);

    macro_rules! bitwise_red {
        ($name:tt : $red:ident, $boolean:expr) => {
            if name == $name {
                let input = if !$boolean {
                    require!(ret_ty == in_elem,
                             "expected return type `{}` (element of input `{}`), found `{}`",
                             in_elem, in_ty, ret_ty);
                    args[0].immediate()
                } else {
                    match in_elem.sty {
                        ty::TyInt(_) | ty::TyUint(_) => {},
                        _ => {
                            return_error!("unsupported {} from `{}` with element `{}` to `{}`",
                                          $name, in_ty, in_elem, ret_ty)
                        }
                    }

                    // boolean reductions operate on vectors of i1s:
                    let i1 = Type::i1(bx.cx);
                    let i1xn = Type::vector(&i1, in_len as u64);
                    bx.trunc(args[0].immediate(), i1xn)
                };
                return match in_elem.sty {
                    ty::TyInt(_) | ty::TyUint(_) => {
                        let r = bx.$red(input);
                        Ok(
                            if !$boolean {
                                r
                            } else {
                                bx.zext(r, Type::bool(bx.cx))
                            }
                        )
                    },
                    _ => {
                        return_error!("unsupported {} from `{}` with element `{}` to `{}`",
                                      $name, in_ty, in_elem, ret_ty)
                    },
                }
            }
        }
    }

    bitwise_red!("simd_reduce_and": vector_reduce_and, false);
    bitwise_red!("simd_reduce_or": vector_reduce_or, false);
    bitwise_red!("simd_reduce_xor": vector_reduce_xor, false);
    bitwise_red!("simd_reduce_all": vector_reduce_and, true);
    bitwise_red!("simd_reduce_any": vector_reduce_or, true);

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
        simd_fmax: TyFloat => maxnum;
        simd_fmin: TyFloat => minnum;
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
