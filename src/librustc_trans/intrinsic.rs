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
use libc;
use llvm;
use llvm::{ValueRef};
use abi::{Abi, FnType};
use adt;
use mir::lvalue::LvalueRef;
use base::*;
use common::*;
use declare;
use glue;
use type_of;
use machine;
use type_::Type;
use rustc::ty::{self, Ty};
use rustc::hir;
use syntax::ast;
use syntax::symbol::Symbol;
use builder::Builder;

use rustc::session::Session;
use syntax_pos::Span;

use rustc_i128::u128;

use std::cmp::Ordering;
use std::iter;

fn get_simple_intrinsic(ccx: &CrateContext, name: &str) -> Option<ValueRef> {
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
    Some(ccx.get_intrinsic(&llvm_name))
}

/// Remember to add all intrinsics here, in librustc_typeck/check/mod.rs,
/// and in libcore/intrinsics.rs; if you need access to any llvm intrinsics,
/// add them to librustc_trans/trans/context.rs
pub fn trans_intrinsic_call<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                                      callee_ty: Ty<'tcx>,
                                      fn_ty: &FnType,
                                      llargs: &[ValueRef],
                                      llresult: ValueRef,
                                      span: Span) {
    let ccx = bcx.ccx;
    let tcx = ccx.tcx();

    let (def_id, substs, fty) = match callee_ty.sty {
        ty::TyFnDef(def_id, substs, ref fty) => (def_id, substs, fty),
        _ => bug!("expected fn item type, found {}", callee_ty)
    };

    let sig = tcx.erase_late_bound_regions_and_normalize(&fty.sig);
    let arg_tys = sig.inputs();
    let ret_ty = sig.output();
    let name = &*tcx.item_name(def_id).as_str();

    let llret_ty = type_of::type_of(ccx, ret_ty);

    let simple = get_simple_intrinsic(ccx, name);
    let llval = match name {
        _ if simple.is_some() => {
            bcx.call(simple.unwrap(), &llargs, None)
        }
        "unreachable" => {
            return;
        },
        "likely" => {
            let expect = ccx.get_intrinsic(&("llvm.expect.i1"));
            bcx.call(expect, &[llargs[0], C_bool(ccx, true)], None)
        }
        "unlikely" => {
            let expect = ccx.get_intrinsic(&("llvm.expect.i1"));
            bcx.call(expect, &[llargs[0], C_bool(ccx, false)], None)
        }
        "try" => {
            try_intrinsic(bcx, ccx, llargs[0], llargs[1], llargs[2], llresult);
            C_nil(ccx)
        }
        "breakpoint" => {
            let llfn = ccx.get_intrinsic(&("llvm.debugtrap"));
            bcx.call(llfn, &[], None)
        }
        "size_of" => {
            let tp_ty = substs.type_at(0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            C_uint(ccx, machine::llsize_of_alloc(ccx, lltp_ty))
        }
        "size_of_val" => {
            let tp_ty = substs.type_at(0);
            if !bcx.ccx.shared().type_is_sized(tp_ty) {
                let (llsize, _) =
                    glue::size_and_align_of_dst(bcx, tp_ty, llargs[1]);
                llsize
            } else {
                let lltp_ty = type_of::type_of(ccx, tp_ty);
                C_uint(ccx, machine::llsize_of_alloc(ccx, lltp_ty))
            }
        }
        "min_align_of" => {
            let tp_ty = substs.type_at(0);
            C_uint(ccx, type_of::align_of(ccx, tp_ty))
        }
        "min_align_of_val" => {
            let tp_ty = substs.type_at(0);
            if !bcx.ccx.shared().type_is_sized(tp_ty) {
                let (_, llalign) =
                    glue::size_and_align_of_dst(bcx, tp_ty, llargs[1]);
                llalign
            } else {
                C_uint(ccx, type_of::align_of(ccx, tp_ty))
            }
        }
        "pref_align_of" => {
            let tp_ty = substs.type_at(0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            C_uint(ccx, machine::llalign_of_pref(ccx, lltp_ty))
        }
        "type_name" => {
            let tp_ty = substs.type_at(0);
            let ty_name = Symbol::intern(&tp_ty.to_string()).as_str();
            C_str_slice(ccx, ty_name)
        }
        "type_id" => {
            C_u64(ccx, ccx.tcx().type_id_hash(substs.type_at(0)))
        }
        "init" => {
            let ty = substs.type_at(0);
            if !type_is_zero_size(ccx, ty) {
                // Just zero out the stack slot.
                // If we store a zero constant, LLVM will drown in vreg allocation for large data
                // structures, and the generated code will be awful. (A telltale sign of this is
                // large quantities of `mov [byte ptr foo],0` in the generated code.)
                memset_intrinsic(bcx, false, ty, llresult, C_u8(ccx, 0), C_uint(ccx, 1usize));
            }
            C_nil(ccx)
        }
        // Effectively no-ops
        "uninit" | "forget" => {
            C_nil(ccx)
        }
        "needs_drop" => {
            let tp_ty = substs.type_at(0);

            C_bool(ccx, bcx.ccx.shared().type_needs_drop(tp_ty))
        }
        "offset" => {
            let ptr = llargs[0];
            let offset = llargs[1];
            bcx.inbounds_gep(ptr, &[offset])
        }
        "arith_offset" => {
            let ptr = llargs[0];
            let offset = llargs[1];
            bcx.gep(ptr, &[offset])
        }

        "copy_nonoverlapping" => {
            copy_intrinsic(bcx, false, false, substs.type_at(0), llargs[1], llargs[0], llargs[2])
        }
        "copy" => {
            copy_intrinsic(bcx, true, false, substs.type_at(0), llargs[1], llargs[0], llargs[2])
        }
        "write_bytes" => {
            memset_intrinsic(bcx, false, substs.type_at(0), llargs[0], llargs[1], llargs[2])
        }

        "volatile_copy_nonoverlapping_memory" => {
            copy_intrinsic(bcx, false, true, substs.type_at(0), llargs[0], llargs[1], llargs[2])
        }
        "volatile_copy_memory" => {
            copy_intrinsic(bcx, true, true, substs.type_at(0), llargs[0], llargs[1], llargs[2])
        }
        "volatile_set_memory" => {
            memset_intrinsic(bcx, true, substs.type_at(0), llargs[0], llargs[1], llargs[2])
        }
        "volatile_load" => {
            let tp_ty = substs.type_at(0);
            let mut ptr = llargs[0];
            if let Some(ty) = fn_ty.ret.cast {
                ptr = bcx.pointercast(ptr, ty.ptr_to());
            }
            let load = bcx.volatile_load(ptr);
            unsafe {
                llvm::LLVMSetAlignment(load, type_of::align_of(ccx, tp_ty));
            }
            to_immediate(bcx, load, tp_ty)
        },
        "volatile_store" => {
            let tp_ty = substs.type_at(0);
            if type_is_fat_ptr(bcx.ccx, tp_ty) {
                bcx.volatile_store(llargs[1], get_dataptr(bcx, llargs[0]));
                bcx.volatile_store(llargs[2], get_meta(bcx, llargs[0]));
            } else {
                let val = if fn_ty.args[1].is_indirect() {
                    bcx.load(llargs[1])
                } else {
                    from_immediate(bcx, llargs[1])
                };
                let ptr = bcx.pointercast(llargs[0], val_ty(val).ptr_to());
                let store = bcx.volatile_store(val, ptr);
                unsafe {
                    llvm::LLVMSetAlignment(store, type_of::align_of(ccx, tp_ty));
                }
            }
            C_nil(ccx)
        },

        "ctlz" | "cttz" | "ctpop" | "bswap" |
        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" |
        "overflowing_add" | "overflowing_sub" | "overflowing_mul" |
        "unchecked_div" | "unchecked_rem" => {
            let sty = &arg_tys[0].sty;
            match int_type_width_signed(sty, ccx) {
                Some((width, signed)) =>
                    match name {
                        "ctlz" | "cttz" => {
                            let y = C_bool(bcx.ccx, false);
                            let llfn = ccx.get_intrinsic(&format!("llvm.{}.i{}", name, width));
                            bcx.call(llfn, &[llargs[0], y], None)
                        }
                        "ctpop" => bcx.call(ccx.get_intrinsic(&format!("llvm.ctpop.i{}", width)),
                                        &llargs, None),
                        "bswap" => {
                            if width == 8 {
                                llargs[0] // byte swap a u8/i8 is just a no-op
                            } else {
                                bcx.call(ccx.get_intrinsic(&format!("llvm.bswap.i{}", width)),
                                        &llargs, None)
                            }
                        }
                        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" => {
                            let intrinsic = format!("llvm.{}{}.with.overflow.i{}",
                                                    if signed { 's' } else { 'u' },
                                                    &name[..3], width);
                            let llfn = bcx.ccx.get_intrinsic(&intrinsic);

                            // Convert `i1` to a `bool`, and write it to the out parameter
                            let val = bcx.call(llfn, &[llargs[0], llargs[1]], None);
                            let result = bcx.extract_value(val, 0);
                            let overflow = bcx.zext(bcx.extract_value(val, 1), Type::bool(ccx));
                            bcx.store(result, bcx.struct_gep(llresult, 0), None);
                            bcx.store(overflow, bcx.struct_gep(llresult, 1), None);

                            C_nil(bcx.ccx)
                        },
                        "overflowing_add" => bcx.add(llargs[0], llargs[1]),
                        "overflowing_sub" => bcx.sub(llargs[0], llargs[1]),
                        "overflowing_mul" => bcx.mul(llargs[0], llargs[1]),
                        "unchecked_div" =>
                            if signed {
                                bcx.sdiv(llargs[0], llargs[1])
                            } else {
                                bcx.udiv(llargs[0], llargs[1])
                            },
                        "unchecked_rem" =>
                            if signed {
                                bcx.srem(llargs[0], llargs[1])
                            } else {
                                bcx.urem(llargs[0], llargs[1])
                            },
                        _ => bug!(),
                    },
                None => {
                    span_invalid_monomorphization_error(
                        tcx.sess, span,
                        &format!("invalid monomorphization of `{}` intrinsic: \
                                  expected basic integer type, found `{}`", name, sty));
                        C_nil(ccx)
                }
            }

        },
        "fadd_fast" | "fsub_fast" | "fmul_fast" | "fdiv_fast" | "frem_fast" => {
            let sty = &arg_tys[0].sty;
            match float_type_width(sty) {
                Some(_width) =>
                    match name {
                        "fadd_fast" => bcx.fadd_fast(llargs[0], llargs[1]),
                        "fsub_fast" => bcx.fsub_fast(llargs[0], llargs[1]),
                        "fmul_fast" => bcx.fmul_fast(llargs[0], llargs[1]),
                        "fdiv_fast" => bcx.fdiv_fast(llargs[0], llargs[1]),
                        "frem_fast" => bcx.frem_fast(llargs[0], llargs[1]),
                        _ => bug!(),
                    },
                None => {
                    span_invalid_monomorphization_error(
                        tcx.sess, span,
                        &format!("invalid monomorphization of `{}` intrinsic: \
                                  expected basic float type, found `{}`", name, sty));
                        C_nil(ccx)
                }
            }

        },

        "discriminant_value" => {
            let val_ty = substs.type_at(0);
            match val_ty.sty {
                ty::TyAdt(adt, ..) if adt.is_enum() => {
                    adt::trans_get_discr(bcx, val_ty, llargs[0],
                                         Some(llret_ty), true)
                }
                _ => C_null(llret_ty)
            }
        }
        name if name.starts_with("simd_") => {
            generic_simd_intrinsic(bcx, name,
                                   callee_ty,
                                   &llargs,
                                   ret_ty, llret_ty,
                                   span)
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
                    _ => ccx.sess().fatal("unknown ordering in atomic intrinsic")
                },
                4 => match (split[2], split[3]) {
                    ("acq", "failrelaxed") if is_cxchg =>
                        (Acquire, Monotonic),
                    ("acqrel", "failrelaxed") if is_cxchg =>
                        (AcquireRelease, Monotonic),
                    _ => ccx.sess().fatal("unknown ordering in atomic intrinsic")
                },
                _ => ccx.sess().fatal("Atomic intrinsic not in correct format"),
            };

            let invalid_monomorphization = |sty| {
                span_invalid_monomorphization_error(tcx.sess, span,
                    &format!("invalid monomorphization of `{}` intrinsic: \
                              expected basic integer type, found `{}`", name, sty));
            };

            match split[1] {
                "cxchg" | "cxchgweak" => {
                    let sty = &substs.type_at(0).sty;
                    if int_type_width_signed(sty, ccx).is_some() {
                        let weak = if split[1] == "cxchgweak" { llvm::True } else { llvm::False };
                        let val = bcx.atomic_cmpxchg(llargs[0], llargs[1], llargs[2], order,
                            failorder, weak);
                        let result = bcx.extract_value(val, 0);
                        let success = bcx.zext(bcx.extract_value(val, 1), Type::bool(bcx.ccx));
                        bcx.store(result, bcx.struct_gep(llresult, 0), None);
                        bcx.store(success, bcx.struct_gep(llresult, 1), None);
                    } else {
                        invalid_monomorphization(sty);
                    }
                    C_nil(ccx)
                }

                "load" => {
                    let sty = &substs.type_at(0).sty;
                    if int_type_width_signed(sty, ccx).is_some() {
                        bcx.atomic_load(llargs[0], order)
                    } else {
                        invalid_monomorphization(sty);
                        C_nil(ccx)
                    }
                }

                "store" => {
                    let sty = &substs.type_at(0).sty;
                    if int_type_width_signed(sty, ccx).is_some() {
                        bcx.atomic_store(llargs[1], llargs[0], order);
                    } else {
                        invalid_monomorphization(sty);
                    }
                    C_nil(ccx)
                }

                "fence" => {
                    bcx.atomic_fence(order, llvm::SynchronizationScope::CrossThread);
                    C_nil(ccx)
                }

                "singlethreadfence" => {
                    bcx.atomic_fence(order, llvm::SynchronizationScope::SingleThread);
                    C_nil(ccx)
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
                        _ => ccx.sess().fatal("unknown atomic operation")
                    };

                    let sty = &substs.type_at(0).sty;
                    if int_type_width_signed(sty, ccx).is_some() {
                        bcx.atomic_rmw(atom_op, llargs[0], llargs[1], order)
                    } else {
                        invalid_monomorphization(sty);
                        C_nil(ccx)
                    }
                }
            }
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
            fn ty_to_type(ccx: &CrateContext, t: &intrinsics::Type,
                          any_changes_needed: &mut bool) -> Vec<Type> {
                use intrinsics::Type::*;
                match *t {
                    Void => vec![Type::void(ccx)],
                    Integer(_signed, width, llvm_width) => {
                        *any_changes_needed |= width != llvm_width;
                        vec![Type::ix(ccx, llvm_width as u64)]
                    }
                    Float(x) => {
                        match x {
                            32 => vec![Type::f32(ccx)],
                            64 => vec![Type::f64(ccx)],
                            _ => bug!()
                        }
                    }
                    Pointer(ref t, ref llvm_elem, _const) => {
                        *any_changes_needed |= llvm_elem.is_some();

                        let t = llvm_elem.as_ref().unwrap_or(t);
                        let elem = one(ty_to_type(ccx, t, any_changes_needed));
                        vec![elem.ptr_to()]
                    }
                    Vector(ref t, ref llvm_elem, length) => {
                        *any_changes_needed |= llvm_elem.is_some();

                        let t = llvm_elem.as_ref().unwrap_or(t);
                        let elem = one(ty_to_type(ccx, t, any_changes_needed));
                        vec![Type::vector(&elem, length as u64)]
                    }
                    Aggregate(false, ref contents) => {
                        let elems = contents.iter()
                                            .map(|t| one(ty_to_type(ccx, t, any_changes_needed)))
                                            .collect::<Vec<_>>();
                        vec![Type::struct_(ccx, &elems, false)]
                    }
                    Aggregate(true, ref contents) => {
                        *any_changes_needed = true;
                        contents.iter()
                                .flat_map(|t| ty_to_type(ccx, t, any_changes_needed))
                                .collect()
                    }
                }
            }

            // This allows an argument list like `foo, (bar, baz),
            // qux` to be converted into `foo, bar, baz, qux`, integer
            // arguments to be truncated as needed and pointers to be
            // cast.
            fn modify_as_needed<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                                          t: &intrinsics::Type,
                                          arg_type: Ty<'tcx>,
                                          llarg: ValueRef)
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
                        assert!(!bcx.ccx.shared().type_needs_drop(arg_type));
                        let arg = LvalueRef::new_sized_ty(llarg, arg_type);
                        (0..contents.len()).map(|i| bcx.load(arg.trans_field_ptr(bcx, i))).collect()
                    }
                    intrinsics::Type::Pointer(_, Some(ref llvm_elem), _) => {
                        let llvm_elem = one(ty_to_type(bcx.ccx, llvm_elem, &mut false));
                        vec![bcx.pointercast(llarg, llvm_elem.ptr_to())]
                    }
                    intrinsics::Type::Vector(_, Some(ref llvm_elem), length) => {
                        let llvm_elem = one(ty_to_type(bcx.ccx, llvm_elem, &mut false));
                        vec![bcx.bitcast(llarg, Type::vector(&llvm_elem, length as u64))]
                    }
                    intrinsics::Type::Integer(_, width, llvm_width) if width != llvm_width => {
                        // the LLVM intrinsic uses a smaller integer
                        // size than the C intrinsic's signature, so
                        // we have to trim it down here.
                        vec![bcx.trunc(llarg, Type::ix(bcx.ccx, llvm_width as u64))]
                    }
                    _ => vec![llarg],
                }
            }


            let mut any_changes_needed = false;
            let inputs = intr.inputs.iter()
                                    .flat_map(|t| ty_to_type(ccx, t, &mut any_changes_needed))
                                    .collect::<Vec<_>>();

            let mut out_changes = false;
            let outputs = one(ty_to_type(ccx, &intr.output, &mut out_changes));
            // outputting a flattened aggregate is nonsense
            assert!(!out_changes);

            let llargs = if !any_changes_needed {
                // no aggregates to flatten, so no change needed
                llargs.to_vec()
            } else {
                // there are some aggregates that need to be flattened
                // in the LLVM call, so we need to run over the types
                // again to find them and extract the arguments
                intr.inputs.iter()
                           .zip(llargs)
                           .zip(arg_tys)
                           .flat_map(|((t, llarg), ty)| modify_as_needed(bcx, t, ty, *llarg))
                           .collect()
            };
            assert_eq!(inputs.len(), llargs.len());

            let val = match intr.definition {
                intrinsics::IntrinsicDef::Named(name) => {
                    let f = declare::declare_cfn(ccx,
                                                 name,
                                                 Type::func(&inputs, &outputs));
                    bcx.call(f, &llargs, None)
                }
            };

            match *intr.output {
                intrinsics::Type::Aggregate(flatten, ref elems) => {
                    // the output is a tuple so we need to munge it properly
                    assert!(!flatten);

                    for i in 0..elems.len() {
                        let val = bcx.extract_value(val, i);
                        bcx.store(val, bcx.struct_gep(llresult, i), None);
                    }
                    C_nil(ccx)
                }
                _ => val,
            }
        }
    };

    if val_ty(llval) != Type::void(ccx) && machine::llsize_of_alloc(ccx, val_ty(llval)) != 0 {
        if let Some(ty) = fn_ty.ret.cast {
            let ptr = bcx.pointercast(llresult, ty.ptr_to());
            bcx.store(llval, ptr, Some(type_of::align_of(ccx, ret_ty)));
        } else {
            store_ty(bcx, llval, llresult, ret_ty);
        }
    }
}

fn copy_intrinsic<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                            allow_overlap: bool,
                            volatile: bool,
                            tp_ty: Ty<'tcx>,
                            dst: ValueRef,
                            src: ValueRef,
                            count: ValueRef)
                            -> ValueRef {
    let ccx = bcx.ccx;
    let lltp_ty = type_of::type_of(ccx, tp_ty);
    let align = C_i32(ccx, type_of::align_of(ccx, tp_ty) as i32);
    let size = machine::llsize_of(ccx, lltp_ty);
    let int_size = machine::llbitsize_of_real(ccx, ccx.int_type());

    let operation = if allow_overlap {
        "memmove"
    } else {
        "memcpy"
    };

    let name = format!("llvm.{}.p0i8.p0i8.i{}", operation, int_size);

    let dst_ptr = bcx.pointercast(dst, Type::i8p(ccx));
    let src_ptr = bcx.pointercast(src, Type::i8p(ccx));
    let llfn = ccx.get_intrinsic(&name);

    bcx.call(llfn,
        &[dst_ptr,
        src_ptr,
        bcx.mul(size, count),
        align,
        C_bool(ccx, volatile)],
        None)
}

fn memset_intrinsic<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    volatile: bool,
    ty: Ty<'tcx>,
    dst: ValueRef,
    val: ValueRef,
    count: ValueRef
) -> ValueRef {
    let ccx = bcx.ccx;
    let align = C_i32(ccx, type_of::align_of(ccx, ty) as i32);
    let lltp_ty = type_of::type_of(ccx, ty);
    let size = machine::llsize_of(ccx, lltp_ty);
    let dst = bcx.pointercast(dst, Type::i8p(ccx));
    call_memset(bcx, dst, val, bcx.mul(size, count), align, volatile)
}

fn try_intrinsic<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    ccx: &CrateContext,
    func: ValueRef,
    data: ValueRef,
    local_ptr: ValueRef,
    dest: ValueRef,
) {
    if bcx.sess().no_landing_pads() {
        bcx.call(func, &[data], None);
        bcx.store(C_null(Type::i8p(&bcx.ccx)), dest, None);
    } else if wants_msvc_seh(bcx.sess()) {
        trans_msvc_try(bcx, ccx, func, data, local_ptr, dest);
    } else {
        trans_gnu_try(bcx, ccx, func, data, local_ptr, dest);
    }
}

// MSVC's definition of the `rust_try` function.
//
// This implementation uses the new exception handling instructions in LLVM
// which have support in LLVM for SEH on MSVC targets. Although these
// instructions are meant to work for all targets, as of the time of this
// writing, however, LLVM does not recommend the usage of these new instructions
// as the old ones are still more optimized.
fn trans_msvc_try<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                            ccx: &CrateContext,
                            func: ValueRef,
                            data: ValueRef,
                            local_ptr: ValueRef,
                            dest: ValueRef) {
    let llfn = get_rust_try_fn(ccx, &mut |bcx| {
        let ccx = bcx.ccx;

        bcx.set_personality_fn(bcx.ccx.eh_personality());

        let normal = bcx.build_sibling_block("normal");
        let catchswitch = bcx.build_sibling_block("catchswitch");
        let catchpad = bcx.build_sibling_block("catchpad");
        let caught = bcx.build_sibling_block("caught");

        let func = llvm::get_param(bcx.llfn(), 0);
        let data = llvm::get_param(bcx.llfn(), 1);
        let local_ptr = llvm::get_param(bcx.llfn(), 2);

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
        let i64p = Type::i64(ccx).ptr_to();
        let slot = bcx.alloca(i64p, "slot");
        bcx.invoke(func, &[data], normal.llbb(), catchswitch.llbb(),
            None);

        normal.ret(C_i32(ccx, 0));

        let cs = catchswitch.catch_switch(None, None, 1);
        catchswitch.add_handler(cs, catchpad.llbb());

        let tcx = ccx.tcx();
        let tydesc = match tcx.lang_items.msvc_try_filter() {
            Some(did) => ::consts::get_static(ccx, did),
            None => bug!("msvc_try_filter not defined"),
        };
        let tok = catchpad.catch_pad(cs, &[tydesc, C_i32(ccx, 0), slot]);
        let addr = catchpad.load(slot);
        let arg1 = catchpad.load(addr);
        let val1 = C_i32(ccx, 1);
        let arg2 = catchpad.load(catchpad.inbounds_gep(addr, &[val1]));
        let local_ptr = catchpad.bitcast(local_ptr, i64p);
        catchpad.store(arg1, local_ptr, None);
        catchpad.store(arg2, catchpad.inbounds_gep(local_ptr, &[val1]), None);
        catchpad.catch_ret(tok, caught.llbb());

        caught.ret(C_i32(ccx, 1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bcx.call(llfn, &[func, data, local_ptr], None);
    bcx.store(ret, dest, None);
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
fn trans_gnu_try<'a, 'tcx>(bcx: &Builder<'a, 'tcx>,
                           ccx: &CrateContext,
                           func: ValueRef,
                           data: ValueRef,
                           local_ptr: ValueRef,
                           dest: ValueRef) {
    let llfn = get_rust_try_fn(ccx, &mut |bcx| {
        let ccx = bcx.ccx;

        // Translates the shims described above:
        //
        //   bcx:
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

        let then = bcx.build_sibling_block("then");
        let catch = bcx.build_sibling_block("catch");

        let func = llvm::get_param(bcx.llfn(), 0);
        let data = llvm::get_param(bcx.llfn(), 1);
        let local_ptr = llvm::get_param(bcx.llfn(), 2);
        bcx.invoke(func, &[data], then.llbb(), catch.llbb(), None);
        then.ret(C_i32(ccx, 0));

        // Type indicator for the exception being thrown.
        //
        // The first value in this tuple is a pointer to the exception object
        // being thrown.  The second value is a "selector" indicating which of
        // the landing pad clauses the exception's type had been matched to.
        // rust_try ignores the selector.
        let lpad_ty = Type::struct_(ccx, &[Type::i8p(ccx), Type::i32(ccx)],
                                    false);
        let vals = catch.landing_pad(lpad_ty, bcx.ccx.eh_personality(), 1, catch.llfn());
        catch.add_clause(vals, C_null(Type::i8p(ccx)));
        let ptr = catch.extract_value(vals, 0);
        catch.store(ptr, catch.bitcast(local_ptr, Type::i8p(ccx).ptr_to()), None);
        catch.ret(C_i32(ccx, 1));
    });

    // Note that no invoke is used here because by definition this function
    // can't panic (that's what it's catching).
    let ret = bcx.call(llfn, &[func, data, local_ptr], None);
    bcx.store(ret, dest, None);
}

// Helper function to give a Block to a closure to translate a shim function.
// This is currently primarily used for the `try` intrinsic functions above.
fn gen_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                    name: &str,
                    inputs: Vec<Ty<'tcx>>,
                    output: Ty<'tcx>,
                    trans: &mut for<'b> FnMut(Builder<'b, 'tcx>))
                    -> ValueRef {
    let sig = ccx.tcx().mk_fn_sig(inputs.into_iter(), output, false);

    let rust_fn_ty = ccx.tcx().mk_fn_ptr(ccx.tcx().mk_bare_fn(ty::BareFnTy {
        unsafety: hir::Unsafety::Unsafe,
        abi: Abi::Rust,
        sig: ty::Binder(sig)
    }));
    let llfn = declare::define_internal_fn(ccx, name, rust_fn_ty);
    let bcx = Builder::new_block(ccx, llfn, "entry-block");
    trans(bcx);
    llfn
}

// Helper function used to get a handle to the `__rust_try` function used to
// catch exceptions.
//
// This function is only generated once and is then cached.
fn get_rust_try_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                             trans: &mut for<'b> FnMut(Builder<'b, 'tcx>))
                             -> ValueRef {
    if let Some(llfn) = ccx.rust_try_fn().get() {
        return llfn;
    }

    // Define the type up front for the signature of the rust_try function.
    let tcx = ccx.tcx();
    let i8p = tcx.mk_mut_ptr(tcx.types.i8);
    let fn_ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: hir::Unsafety::Unsafe,
        abi: Abi::Rust,
        sig: ty::Binder(tcx.mk_fn_sig(iter::once(i8p), tcx.mk_nil(), false)),
    }));
    let output = tcx.types.i32;
    let rust_try = gen_fn(ccx, "__rust_try", vec![fn_ty, i8p, i8p], output, trans);
    ccx.rust_try_fn().set(Some(rust_try));
    return rust_try
}

fn span_invalid_monomorphization_error(a: &Session, b: Span, c: &str) {
    span_err!(a, b, E0511, "{}", c);
}

fn generic_simd_intrinsic<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    name: &str,
    callee_ty: Ty<'tcx>,
    llargs: &[ValueRef],
    ret_ty: Ty<'tcx>,
    llret_ty: Type,
    span: Span
) -> ValueRef {
    // macros for error handling:
    macro_rules! emit_error {
        ($msg: tt) => {
            emit_error!($msg, )
        };
        ($msg: tt, $($fmt: tt)*) => {
            span_invalid_monomorphization_error(
                bcx.sess(), span,
                &format!(concat!("invalid monomorphization of `{}` intrinsic: ",
                                 $msg),
                         name, $($fmt)*));
        }
    }
    macro_rules! require {
        ($cond: expr, $($fmt: tt)*) => {
            if !$cond {
                emit_error!($($fmt)*);
                return C_nil(bcx.ccx)
            }
        }
    }
    macro_rules! require_simd {
        ($ty: expr, $position: expr) => {
            require!($ty.is_simd(), "expected SIMD {} type, found non-SIMD `{}`", $position, $ty)
        }
    }



    let tcx = bcx.tcx();
    let sig = tcx.erase_late_bound_regions_and_normalize(callee_ty.fn_sig());
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

        return compare_simd_types(bcx,
                                  llargs[0],
                                  llargs[1],
                                  in_elem,
                                  llret_ty,
                                  cmp_op)
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

        let vector = llargs[2];

        let indices: Option<Vec<_>> = (0..n)
            .map(|i| {
                let arg_idx = i;
                let val = const_get_elt(vector, &[i as libc::c_uint]);
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
                    Some(idx) => Some(C_i32(bcx.ccx, idx as i32)),
                }
            })
            .collect();
        let indices = match indices {
            Some(i) => i,
            None => return C_null(llret_ty)
        };

        return bcx.shuffle_vector(llargs[0], llargs[1], C_vector(&indices))
    }

    if name == "simd_insert" {
        require!(in_elem == arg_tys[2],
                 "expected inserted type `{}` (element of input `{}`), found `{}`",
                 in_elem, in_ty, arg_tys[2]);
        return bcx.insert_element(llargs[0], llargs[2], llargs[1])
    }
    if name == "simd_extract" {
        require!(ret_ty == in_elem,
                 "expected return type `{}` (element of input `{}`), found `{}`",
                 in_elem, in_ty, ret_ty);
        return bcx.extract_element(llargs[0], llargs[1])
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

        if in_elem == out_elem { return llargs[0]; }

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
                return match in_width.cmp(&out_width) {
                    Ordering::Greater => bcx.trunc(llargs[0], llret_ty),
                    Ordering::Equal => llargs[0],
                    Ordering::Less => if in_is_signed {
                        bcx.sext(llargs[0], llret_ty)
                    } else {
                        bcx.zext(llargs[0], llret_ty)
                    }
                }
            }
            (Style::Int(in_is_signed), Style::Float) => {
                return if in_is_signed {
                    bcx.sitofp(llargs[0], llret_ty)
                } else {
                    bcx.uitofp(llargs[0], llret_ty)
                }
            }
            (Style::Float, Style::Int(out_is_signed)) => {
                return if out_is_signed {
                    bcx.fptosi(llargs[0], llret_ty)
                } else {
                    bcx.fptoui(llargs[0], llret_ty)
                }
            }
            (Style::Float, Style::Float) => {
                return match in_width.cmp(&out_width) {
                    Ordering::Greater => bcx.fptrunc(llargs[0], llret_ty),
                    Ordering::Equal => llargs[0],
                    Ordering::Less => bcx.fpext(llargs[0], llret_ty)
                }
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
            $(
                if name == stringify!($name) {
                    match in_elem.sty {
                        $(
                            $(ty::$p(_))|* => {
                                return bcx.$call(llargs[0], llargs[1])
                            }
                            )*
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
        simd_div: TyFloat => fdiv;
        simd_shl: TyUint, TyInt => shl;
        simd_shr: TyUint => lshr, TyInt => ashr;
        simd_and: TyUint, TyInt => and;
        simd_or: TyUint, TyInt => or;
        simd_xor: TyUint, TyInt => xor;
    }
    span_bug!(span, "unknown SIMD intrinsic");
}

// Returns the width of an int TypeVariant, and if it's signed or not
// Returns None if the type is not an integer
// FIXME: there’s multiple of this functions, investigate using some of the already existing
// stuffs.
fn int_type_width_signed<'tcx>(sty: &ty::TypeVariants<'tcx>, ccx: &CrateContext)
        -> Option<(u64, bool)> {
    use rustc::ty::{TyInt, TyUint};
    match *sty {
        TyInt(t) => Some((match t {
            ast::IntTy::Is => {
                match &ccx.tcx().sess.target.target.target_pointer_width[..] {
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
        TyUint(t) => Some((match t {
            ast::UintTy::Us => {
                match &ccx.tcx().sess.target.target.target_pointer_width[..] {
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
