// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_uppercase_pattern_statics)]

use llvm;
use llvm::{SequentiallyConsistent, Acquire, Release, Xchg, ValueRef};
use middle::subst;
use middle::subst::FnSpace;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::trans::machine;
use middle::trans::machine::llsize_of;
use middle::trans::type_::Type;
use middle::ty;
use syntax::abi::RustIntrinsic;
use syntax::ast;
use syntax::parse::token;
use util::ppaux::ty_to_string;

pub fn get_simple_intrinsic(ccx: &CrateContext, item: &ast::ForeignItem) -> Option<ValueRef> {
    let name = match token::get_ident(item.ident).get() {
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
        "ctpop8" => "llvm.ctpop.i8",
        "ctpop16" => "llvm.ctpop.i16",
        "ctpop32" => "llvm.ctpop.i32",
        "ctpop64" => "llvm.ctpop.i64",
        "bswap16" => "llvm.bswap.i16",
        "bswap32" => "llvm.bswap.i32",
        "bswap64" => "llvm.bswap.i64",
        _ => return None
    };
    Some(ccx.get_intrinsic(&name))
}

/// Performs late verification that intrinsics are used correctly. At present,
/// the only intrinsic that needs such verification is `transmute`.
pub fn check_intrinsics(ccx: &CrateContext) {
    for transmute_restriction in ccx.tcx
                                    .transmute_restrictions
                                    .borrow()
                                    .iter() {
        let llfromtype = type_of::sizing_type_of(ccx,
                                                 transmute_restriction.from);
        let lltotype = type_of::sizing_type_of(ccx,
                                               transmute_restriction.to);
        let from_type_size = machine::llbitsize_of_real(ccx, llfromtype);
        let to_type_size = machine::llbitsize_of_real(ccx, lltotype);
        if from_type_size != to_type_size {
            ccx.sess()
               .span_err(transmute_restriction.span,
                format!("transmute called on types with different sizes: \
                         {} ({} bit{}) to {} ({} bit{})",
                        ty_to_string(ccx.tcx(), transmute_restriction.from),
                        from_type_size as uint,
                        if from_type_size == 1 {
                            ""
                        } else {
                            "s"
                        },
                        ty_to_string(ccx.tcx(), transmute_restriction.to),
                        to_type_size as uint,
                        if to_type_size == 1 {
                            ""
                        } else {
                            "s"
                        }).as_slice());
        }
        if ty::type_is_fat_ptr(ccx.tcx(), transmute_restriction.to) ||
           ty::type_is_fat_ptr(ccx.tcx(), transmute_restriction.from) {
            ccx.sess()
               .add_lint(::lint::builtin::TRANSMUTE_FAT_PTR,
                         transmute_restriction.id,
                         transmute_restriction.span,
                         format!("Transmuting fat pointer types; {} to {}.\
                                  Beware of relying on the compiler's representation",
                                 ty_to_string(ccx.tcx(), transmute_restriction.from),
                                 ty_to_string(ccx.tcx(), transmute_restriction.to)));
        }
    }
    ccx.sess().abort_if_errors();
}

pub fn trans_intrinsic_call<'a>(mut bcx: &'a Block<'a>, node: ast::NodeId,
                                callee_ty: ty::t, cleanup_scope: cleanup::CustomScopeIndex,
                                args: callee::CallArgs, dest: expr::Dest,
                                substs: subst::Substs, call_info: NodeInfo) -> Result<'a> {

    let fcx = bcx.fcx;
    let ccx = fcx.ccx;
    let tcx = bcx.tcx();

    let ret_ty = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref f) => f.sig.output,
        _ => fail!("expected bare_fn in trans_intrinsic_call")
    };
    let llret_ty = type_of::type_of(ccx, ret_ty);
    let foreign_item = tcx.map.expect_foreign_item(node);
    let name = token::get_ident(foreign_item.ident);

    // For `transmute` we can just trans the input expr directly into dest
    if name.get() == "transmute" {
        match args {
            callee::ArgExprs(arg_exprs) => {
                assert_eq!(arg_exprs.len(), 1);

                let (in_type, out_type) = (*substs.types.get(FnSpace, 0),
                                           *substs.types.get(FnSpace, 1));
                let llintype = type_of::type_of(ccx, in_type);
                let llouttype = type_of::type_of(ccx, out_type);

                let in_type_size = machine::llbitsize_of_real(ccx, llintype);
                let out_type_size = machine::llbitsize_of_real(ccx, llouttype);

                // This should be caught by the intrinsicck pass
                assert_eq!(in_type_size, out_type_size);

                // We need to cast the dest so the types work out
                let dest = match dest {
                    expr::SaveIn(d) => expr::SaveIn(PointerCast(bcx, d, llintype.ptr_to())),
                    expr::Ignore => expr::Ignore
                };
                bcx = expr::trans_into(bcx, &*arg_exprs[0], dest);

                fcx.pop_custom_cleanup_scope(cleanup_scope);

                return match dest {
                    expr::SaveIn(d) => Result::new(bcx, d),
                    expr::Ignore => Result::new(bcx, C_undef(llret_ty.ptr_to()))
                };

            }

            _ => {
                ccx.sess().bug("expected expr as argument for transmute");
            }
        }
    }

    // Get location to store the result. If the user does
    // not care about the result, just make a stack slot
    let llresult = match dest {
        expr::SaveIn(d) => d,
        expr::Ignore => {
            if !type_is_zero_size(ccx, ret_ty) {
                alloc_ty(bcx, ret_ty, "intrinsic_result")
            } else {
                C_undef(llret_ty.ptr_to())
            }
        }
    };

    // Push the arguments.
    let mut llargs = Vec::new();
    bcx = callee::trans_args(bcx,
                             args,
                             callee_ty,
                             &mut llargs,
                             cleanup::CustomScope(cleanup_scope),
                             false,
                             RustIntrinsic);

    fcx.pop_custom_cleanup_scope(cleanup_scope);

    let simple = get_simple_intrinsic(ccx, &*foreign_item);

    let llval = match (simple, name.get()) {
        (Some(llfn), _) => {
            Call(bcx, llfn, llargs.as_slice(), None)
        }
        (_, "abort") => {
            let llfn = ccx.get_intrinsic(&("llvm.trap"));
            let v = Call(bcx, llfn, [], None);
            Unreachable(bcx);
            v
        }
        (_, "breakpoint") => {
            let llfn = ccx.get_intrinsic(&("llvm.debugtrap"));
            Call(bcx, llfn, [], None)
        }
        (_, "size_of") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            C_uint(ccx, machine::llsize_of_real(ccx, lltp_ty) as uint)
        }
        (_, "min_align_of") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            C_uint(ccx, type_of::align_of(ccx, tp_ty) as uint)
        }
        (_, "pref_align_of") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            C_uint(ccx, machine::llalign_of_pref(ccx, lltp_ty) as uint)
        }
        (_, "move_val_init") => {
            // Create a datum reflecting the value being moved.
            // Use `appropriate_mode` so that the datum is by ref
            // if the value is non-immediate. Note that, with
            // intrinsics, there are no argument cleanups to
            // concern ourselves with, so we can use an rvalue datum.
            let tp_ty = *substs.types.get(FnSpace, 0);
            let mode = appropriate_rvalue_mode(ccx, tp_ty);
            let src = Datum {
                val: *llargs.get(1),
                ty: tp_ty,
                kind: Rvalue::new(mode)
            };
            bcx = src.store_to(bcx, *llargs.get(0));
            C_nil(ccx)
        }
        (_, "get_tydesc") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            let static_ti = get_tydesc(ccx, tp_ty);
            glue::lazily_emit_visit_glue(ccx, &*static_ti);

            // FIXME (#3730): ideally this shouldn't need a cast,
            // but there's a circularity between translating rust types to llvm
            // types and having a tydesc type available. So I can't directly access
            // the llvm type of intrinsic::TyDesc struct.
            PointerCast(bcx, static_ti.tydesc, llret_ty)
        }
        (_, "type_id") => {
            let hash = ty::hash_crate_independent(
                ccx.tcx(),
                *substs.types.get(FnSpace, 0),
                &ccx.link_meta.crate_hash);
            // NB: This needs to be kept in lockstep with the TypeId struct in
            //     the intrinsic module
            C_named_struct(llret_ty, [C_u64(ccx, hash)])
        }
        (_, "init") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            if return_type_is_void(ccx, tp_ty) {
                C_nil(ccx)
            } else {
                C_null(lltp_ty)
            }
        }
        // Effectively no-ops
        (_, "uninit") | (_, "forget") => {
            C_nil(ccx)
        }
        (_, "needs_drop") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            C_bool(ccx, ty::type_needs_drop(ccx.tcx(), tp_ty))
        }
        (_, "owns_managed") => {
            let tp_ty = *substs.types.get(FnSpace, 0);
            C_bool(ccx, ty::type_contents(ccx.tcx(), tp_ty).owns_managed())
        }
        (_, "visit_tydesc") => {
            let td = *llargs.get(0);
            let visitor = *llargs.get(1);
            let td = PointerCast(bcx, td, ccx.tydesc_type().ptr_to());
            glue::call_visit_glue(bcx, visitor, td);
            C_nil(ccx)
        }
        (_, "offset") => {
            let ptr = *llargs.get(0);
            let offset = *llargs.get(1);
            InBoundsGEP(bcx, ptr, [offset])
        }

        (_, "copy_nonoverlapping_memory") => {
            copy_intrinsic(bcx, false, false, *substs.types.get(FnSpace, 0),
                           *llargs.get(0), *llargs.get(1), *llargs.get(2))
        }
        (_, "copy_memory") => {
            copy_intrinsic(bcx, true, false, *substs.types.get(FnSpace, 0),
                           *llargs.get(0), *llargs.get(1), *llargs.get(2))
        }
        (_, "set_memory") => {
            memset_intrinsic(bcx, false, *substs.types.get(FnSpace, 0),
                             *llargs.get(0), *llargs.get(1), *llargs.get(2))
        }

        (_, "volatile_copy_nonoverlapping_memory") => {
            copy_intrinsic(bcx, false, true, *substs.types.get(FnSpace, 0),
                           *llargs.get(0), *llargs.get(1), *llargs.get(2))
        }
        (_, "volatile_copy_memory") => {
            copy_intrinsic(bcx, true, true, *substs.types.get(FnSpace, 0),
                           *llargs.get(0), *llargs.get(1), *llargs.get(2))
        }
        (_, "volatile_set_memory") => {
            memset_intrinsic(bcx, true, *substs.types.get(FnSpace, 0),
                             *llargs.get(0), *llargs.get(1), *llargs.get(2))
        }
        (_, "volatile_load") => {
            VolatileLoad(bcx, *llargs.get(0))
        },
        (_, "volatile_store") => {
            VolatileStore(bcx, *llargs.get(1), *llargs.get(0));
            C_nil(ccx)
        },

        (_, "ctlz8") => count_zeros_intrinsic(bcx, "llvm.ctlz.i8", *llargs.get(0)),
        (_, "ctlz16") => count_zeros_intrinsic(bcx, "llvm.ctlz.i16", *llargs.get(0)),
        (_, "ctlz32") => count_zeros_intrinsic(bcx, "llvm.ctlz.i32", *llargs.get(0)),
        (_, "ctlz64") => count_zeros_intrinsic(bcx, "llvm.ctlz.i64", *llargs.get(0)),
        (_, "cttz8") => count_zeros_intrinsic(bcx, "llvm.cttz.i8", *llargs.get(0)),
        (_, "cttz16") => count_zeros_intrinsic(bcx, "llvm.cttz.i16", *llargs.get(0)),
        (_, "cttz32") => count_zeros_intrinsic(bcx, "llvm.cttz.i32", *llargs.get(0)),
        (_, "cttz64") => count_zeros_intrinsic(bcx, "llvm.cttz.i64", *llargs.get(0)),

        (_, "i8_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.sadd.with.overflow.i8", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i16_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.sadd.with.overflow.i16", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i32_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.sadd.with.overflow.i32", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i64_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.sadd.with.overflow.i64", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),

        (_, "u8_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.uadd.with.overflow.i8", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "u16_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.uadd.with.overflow.i16", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "u32_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.uadd.with.overflow.i32", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "u64_add_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.uadd.with.overflow.i64", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),

        (_, "i8_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.ssub.with.overflow.i8", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i16_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.ssub.with.overflow.i16", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i32_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.ssub.with.overflow.i32", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i64_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.ssub.with.overflow.i64", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),

        (_, "u8_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.usub.with.overflow.i8", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "u16_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.usub.with.overflow.i16", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "u32_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.usub.with.overflow.i32", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "u64_sub_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.usub.with.overflow.i64", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),

        (_, "i8_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.smul.with.overflow.i8", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i16_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.smul.with.overflow.i16", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i32_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.smul.with.overflow.i32", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),
        (_, "i64_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.smul.with.overflow.i64", ret_ty,
                                   *llargs.get(0), *llargs.get(1)),

        (_, "u8_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.umul.with.overflow.i8", ret_ty,
                                    *llargs.get(0), *llargs.get(1)),
        (_, "u16_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.umul.with.overflow.i16", ret_ty,
                                    *llargs.get(0), *llargs.get(1)),
        (_, "u32_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.umul.with.overflow.i32", ret_ty,
                                    *llargs.get(0), *llargs.get(1)),
        (_, "u64_mul_with_overflow") =>
            with_overflow_intrinsic(bcx, "llvm.umul.with.overflow.i64", ret_ty,
                                    *llargs.get(0), *llargs.get(1)),

        (_, "return_address") => {
            if !fcx.caller_expects_out_pointer {
                tcx.sess.span_err(call_info.span,
                                  "invalid use of `return_address` intrinsic: function \
                                   does not use out pointer");
                C_null(Type::i8p(ccx))
            } else {
                PointerCast(bcx, llvm::get_param(fcx.llfn, 0), Type::i8p(ccx))
            }
        }

        // This requires that atomic intrinsics follow a specific naming pattern:
        // "atomic_<operation>[_<ordering>]", and no ordering means SeqCst
        (_, name) if name.starts_with("atomic_") => {
            let split: Vec<&str> = name.split('_').collect();
            assert!(split.len() >= 2, "Atomic intrinsic not correct format");

            let order = if split.len() == 2 {
                llvm::SequentiallyConsistent
            } else {
                match *split.get(2) {
                    "relaxed" => llvm::Monotonic,
                    "acq"     => llvm::Acquire,
                    "rel"     => llvm::Release,
                    "acqrel"  => llvm::AcquireRelease,
                    _ => ccx.sess().fatal("unknown ordering in atomic intrinsic")
                }
            };

            match *split.get(1) {
                "cxchg" => {
                    // See include/llvm/IR/Instructions.h for their implementation
                    // of this, I assume that it's good enough for us to use for
                    // now.
                    let strongest_failure_ordering = match order {
                        llvm::NotAtomic | llvm::Unordered =>
                            ccx.sess().fatal("cmpxchg must be atomic"),

                        llvm::Monotonic | llvm::Release =>
                            llvm::Monotonic,

                        llvm::Acquire | llvm::AcquireRelease =>
                            llvm::Acquire,

                        llvm::SequentiallyConsistent =>
                            llvm::SequentiallyConsistent
                    };

                    let res = AtomicCmpXchg(bcx, *llargs.get(0), *llargs.get(1),
                                            *llargs.get(2), order,
                                            strongest_failure_ordering);
                    if unsafe { llvm::LLVMVersionMinor() >= 5 } {
                        ExtractValue(bcx, res, 0)
                    } else {
                        res
                    }
                }

                "load" => {
                    AtomicLoad(bcx, *llargs.get(0), order)
                }
                "store" => {
                    AtomicStore(bcx, *llargs.get(1), *llargs.get(0), order);
                    C_nil(ccx)
                }

                "fence" => {
                    AtomicFence(bcx, order);
                    C_nil(ccx)
                }

                // These are all AtomicRMW ops
                op => {
                    let atom_op = match op {
                        "xchg"  => llvm::Xchg,
                        "xadd"  => llvm::Add,
                        "xsub"  => llvm::Sub,
                        "and"   => llvm::And,
                        "nand"  => llvm::Nand,
                        "or"    => llvm::Or,
                        "xor"   => llvm::Xor,
                        "max"   => llvm::Max,
                        "min"   => llvm::Min,
                        "umax"  => llvm::UMax,
                        "umin"  => llvm::UMin,
                        _ => ccx.sess().fatal("unknown atomic operation")
                    };

                    AtomicRMW(bcx, atom_op, *llargs.get(0), *llargs.get(1), order)
                }
            }

        }

        (_, _) => ccx.sess().span_bug(foreign_item.span, "unknown intrinsic")
    };

    if val_ty(llval) != Type::void(ccx) &&
       machine::llsize_of_alloc(ccx, val_ty(llval)) != 0 {
        store_ty(bcx, llval, llresult, ret_ty);
    }

    // If we made a temporary stack slot, let's clean it up
    match dest {
        expr::Ignore => {
            bcx = glue::drop_ty(bcx, llresult, ret_ty);
        }
        expr::SaveIn(_) => {}
    }

    Result::new(bcx, llresult)
}

fn copy_intrinsic(bcx: &Block, allow_overlap: bool, volatile: bool,
                  tp_ty: ty::t, dst: ValueRef, src: ValueRef, count: ValueRef) -> ValueRef {
    let ccx = bcx.ccx();
    let lltp_ty = type_of::type_of(ccx, tp_ty);
    let align = C_i32(ccx, type_of::align_of(ccx, tp_ty) as i32);
    let size = machine::llsize_of(ccx, lltp_ty);
    let int_size = machine::llbitsize_of_real(ccx, ccx.int_type);
    let name = if allow_overlap {
        if int_size == 32 {
            "llvm.memmove.p0i8.p0i8.i32"
        } else {
            "llvm.memmove.p0i8.p0i8.i64"
        }
    } else {
        if int_size == 32 {
            "llvm.memcpy.p0i8.p0i8.i32"
        } else {
            "llvm.memcpy.p0i8.p0i8.i64"
        }
    };

    let dst_ptr = PointerCast(bcx, dst, Type::i8p(ccx));
    let src_ptr = PointerCast(bcx, src, Type::i8p(ccx));
    let llfn = ccx.get_intrinsic(&name);

    Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align,
                     C_bool(ccx, volatile)], None)
}

fn memset_intrinsic(bcx: &Block, volatile: bool, tp_ty: ty::t,
                    dst: ValueRef, val: ValueRef, count: ValueRef) -> ValueRef {
    let ccx = bcx.ccx();
    let lltp_ty = type_of::type_of(ccx, tp_ty);
    let align = C_i32(ccx, type_of::align_of(ccx, tp_ty) as i32);
    let size = machine::llsize_of(ccx, lltp_ty);
    let name = if machine::llbitsize_of_real(ccx, ccx.int_type) == 32 {
        "llvm.memset.p0i8.i32"
    } else {
        "llvm.memset.p0i8.i64"
    };

    let dst_ptr = PointerCast(bcx, dst, Type::i8p(ccx));
    let llfn = ccx.get_intrinsic(&name);

    Call(bcx, llfn, [dst_ptr, val, Mul(bcx, size, count), align,
                     C_bool(ccx, volatile)], None)
}

fn count_zeros_intrinsic(bcx: &Block, name: &'static str, val: ValueRef) -> ValueRef {
    let y = C_bool(bcx.ccx(), false);
    let llfn = bcx.ccx().get_intrinsic(&name);
    Call(bcx, llfn, [val, y], None)
}

fn with_overflow_intrinsic(bcx: &Block, name: &'static str, t: ty::t,
                           a: ValueRef, b: ValueRef) -> ValueRef {
    let llfn = bcx.ccx().get_intrinsic(&name);

    // Convert `i1` to a `bool`, and write it to the out parameter
    let val = Call(bcx, llfn, [a, b], None);
    let result = ExtractValue(bcx, val, 0);
    let overflow = ZExt(bcx, ExtractValue(bcx, val, 1), Type::bool(bcx.ccx()));
    let ret = C_undef(type_of::type_of(bcx.ccx(), t));
    let ret = InsertValue(bcx, ret, result, 0);
    let ret = InsertValue(bcx, ret, overflow, 1);

    ret
}
