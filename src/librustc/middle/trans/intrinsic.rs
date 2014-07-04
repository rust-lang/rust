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

use arena::TypedArena;
use lib::llvm::{SequentiallyConsistent, Acquire, Release, Xchg};
use lib::llvm::{ValueRef, Pointer, Array, Struct};
use lib;
use middle::subst::FnSpace;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::glue;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::trans::machine;
use middle::trans::machine::llsize_of;
use middle::trans::type_::Type;
use middle::ty;
use syntax::ast;
use syntax::ast_map;
use syntax::parse::token;
use util::ppaux::ty_to_str;

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

pub fn trans_intrinsic(ccx: &CrateContext,
                       decl: ValueRef,
                       item: &ast::ForeignItem,
                       substs: &param_substs,
                       ref_id: Option<ast::NodeId>) {
    debug!("trans_intrinsic(item.ident={})", token::get_ident(item.ident));

    fn with_overflow_instrinsic(bcx: &Block, name: &'static str, t: ty::t) {
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let a = get_param(bcx.fcx.llfn, first_real_arg);
        let b = get_param(bcx.fcx.llfn, first_real_arg + 1);
        let llfn = bcx.ccx().get_intrinsic(&name);

        let val = Call(bcx, llfn, [a, b], []);

        if type_is_immediate(bcx.ccx(), t) {
            Ret(bcx, val);
        } else {
            let retptr = get_param(bcx.fcx.llfn, bcx.fcx.out_arg_pos());
            Store(bcx, val, retptr);
            RetVoid(bcx);
        }
    }

    fn volatile_load_intrinsic(bcx: &Block) {
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let src = get_param(bcx.fcx.llfn, first_real_arg);

        let val = VolatileLoad(bcx, src);
        Ret(bcx, val);
    }

    fn volatile_store_intrinsic(bcx: &Block) {
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let dst = get_param(bcx.fcx.llfn, first_real_arg);
        let val = get_param(bcx.fcx.llfn, first_real_arg + 1);

        VolatileStore(bcx, val, dst);
        RetVoid(bcx);
    }

    fn copy_intrinsic(bcx: &Block, allow_overlap: bool, volatile: bool, tp_ty: ty::t) {
        let ccx = bcx.ccx();
        let lltp_ty = type_of::type_of(ccx, tp_ty);
        let align = C_i32(ccx, machine::llalign_of_min(ccx, lltp_ty) as i32);
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

        let decl = bcx.fcx.llfn;
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), Type::i8p(ccx));
        let src_ptr = PointerCast(bcx, get_param(decl, first_real_arg + 1), Type::i8p(ccx));
        let count = get_param(decl, first_real_arg + 2);
        let llfn = ccx.get_intrinsic(&name);
        Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align, C_i1(ccx, volatile)], []);
        RetVoid(bcx);
    }

    fn memset_intrinsic(bcx: &Block, volatile: bool, tp_ty: ty::t) {
        let ccx = bcx.ccx();
        let lltp_ty = type_of::type_of(ccx, tp_ty);
        let align = C_i32(ccx, machine::llalign_of_min(ccx, lltp_ty) as i32);
        let size = machine::llsize_of(ccx, lltp_ty);
        let name = if machine::llbitsize_of_real(ccx, ccx.int_type) == 32 {
            "llvm.memset.p0i8.i32"
        } else {
            "llvm.memset.p0i8.i64"
        };

        let decl = bcx.fcx.llfn;
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), Type::i8p(ccx));
        let val = get_param(decl, first_real_arg + 1);
        let count = get_param(decl, first_real_arg + 2);
        let llfn = ccx.get_intrinsic(&name);
        Call(bcx, llfn, [dst_ptr, val, Mul(bcx, size, count), align, C_i1(ccx, volatile)], []);
        RetVoid(bcx);
    }

    fn count_zeros_intrinsic(bcx: &Block, name: &'static str) {
        let x = get_param(bcx.fcx.llfn, bcx.fcx.arg_pos(0u));
        let y = C_i1(bcx.ccx(), false);
        let llfn = bcx.ccx().get_intrinsic(&name);
        let llcall = Call(bcx, llfn, [x, y], []);
        Ret(bcx, llcall);
    }

    let output_type = ty::ty_fn_ret(ty::node_id_to_type(ccx.tcx(), item.id));

    let arena = TypedArena::new();
    let fcx = new_fn_ctxt(ccx, decl, item.id, false, output_type,
                          substs, Some(item.span), &arena);
    let mut bcx = init_function(&fcx, true, output_type);

    set_always_inline(fcx.llfn);

    let first_real_arg = fcx.arg_pos(0u);

    let name = token::get_ident(item.ident);

    // This requires that atomic intrinsics follow a specific naming pattern:
    // "atomic_<operation>[_<ordering>], and no ordering means SeqCst
    if name.get().starts_with("atomic_") {
        let split: Vec<&str> = name.get().split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic not correct format");
        let order = if split.len() == 2 {
            lib::llvm::SequentiallyConsistent
        } else {
            match *split.get(2) {
                "relaxed" => lib::llvm::Monotonic,
                "acq"     => lib::llvm::Acquire,
                "rel"     => lib::llvm::Release,
                "acqrel"  => lib::llvm::AcquireRelease,
                _ => ccx.sess().fatal("unknown ordering in atomic intrinsic")
            }
        };

        match *split.get(1) {
            "cxchg" => {
                // See include/llvm/IR/Instructions.h for their implementation
                // of this, I assume that it's good enough for us to use for
                // now.
                let strongest_failure_ordering = match order {
                    lib::llvm::NotAtomic | lib::llvm::Unordered =>
                        ccx.sess().fatal("cmpxchg must be atomic"),
                    lib::llvm::Monotonic | lib::llvm::Release =>
                        lib::llvm::Monotonic,
                    lib::llvm::Acquire | lib::llvm::AcquireRelease =>
                        lib::llvm::Acquire,
                    lib::llvm::SequentiallyConsistent =>
                        lib::llvm::SequentiallyConsistent,
                };
                let res = AtomicCmpXchg(bcx, get_param(decl, first_real_arg),
                                        get_param(decl, first_real_arg + 1u),
                                        get_param(decl, first_real_arg + 2u),
                                        order, strongest_failure_ordering);
                if unsafe { lib::llvm::llvm::LLVMVersionMinor() >= 5 } {
                    Ret(bcx, ExtractValue(bcx, res, 0));
                } else {
                    Ret(bcx, res);
                }
            }
            "load" => {
                let old = AtomicLoad(bcx, get_param(decl, first_real_arg),
                                     order);
                Ret(bcx, old);
            }
            "store" => {
                AtomicStore(bcx, get_param(decl, first_real_arg + 1u),
                            get_param(decl, first_real_arg),
                            order);
                RetVoid(bcx);
            }
            "fence" => {
                AtomicFence(bcx, order);
                RetVoid(bcx);
            }
            op => {
                // These are all AtomicRMW ops
                let atom_op = match op {
                    "xchg"  => lib::llvm::Xchg,
                    "xadd"  => lib::llvm::Add,
                    "xsub"  => lib::llvm::Sub,
                    "and"   => lib::llvm::And,
                    "nand"  => lib::llvm::Nand,
                    "or"    => lib::llvm::Or,
                    "xor"   => lib::llvm::Xor,
                    "max"   => lib::llvm::Max,
                    "min"   => lib::llvm::Min,
                    "umax"  => lib::llvm::UMax,
                    "umin"  => lib::llvm::UMin,
                    _ => ccx.sess().fatal("unknown atomic operation")
                };

                let old = AtomicRMW(bcx, atom_op, get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    order);
                Ret(bcx, old);
            }
        }

        fcx.cleanup();
        return;
    }

    match name.get() {
        "abort" => {
            let llfn = bcx.ccx().get_intrinsic(&("llvm.trap"));
            Call(bcx, llfn, [], []);
            Unreachable(bcx);
        }
        "breakpoint" => {
            let llfn = bcx.ccx().get_intrinsic(&("llvm.debugtrap"));
            Call(bcx, llfn, [], []);
            RetVoid(bcx);
        }
        "size_of" => {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Ret(bcx, C_uint(ccx, machine::llsize_of_real(ccx, lltp_ty) as uint));
        }
        "move_val_init" => {
            // Create a datum reflecting the value being moved.
            // Use `appropriate_mode` so that the datum is by ref
            // if the value is non-immediate. Note that, with
            // intrinsics, there are no argument cleanups to
            // concern ourselves with, so we can use an rvalue datum.
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            let mode = appropriate_rvalue_mode(ccx, tp_ty);
            let src = Datum {val: get_param(decl, first_real_arg + 1u),
                             ty: tp_ty,
                             kind: Rvalue::new(mode)};
            bcx = src.store_to(bcx, get_param(decl, first_real_arg));
            RetVoid(bcx);
        }
        "min_align_of" => {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Ret(bcx, C_uint(ccx, machine::llalign_of_min(ccx, lltp_ty) as uint));
        }
        "pref_align_of"=> {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Ret(bcx, C_uint(ccx, machine::llalign_of_pref(ccx, lltp_ty) as uint));
        }
        "get_tydesc" => {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            let static_ti = get_tydesc(ccx, tp_ty);
            glue::lazily_emit_visit_glue(ccx, &*static_ti);

            // FIXME (#3730): ideally this shouldn't need a cast,
            // but there's a circularity between translating rust types to llvm
            // types and having a tydesc type available. So I can't directly access
            // the llvm type of intrinsic::TyDesc struct.
            let userland_tydesc_ty = type_of::type_of(ccx, output_type);
            let td = PointerCast(bcx, static_ti.tydesc, userland_tydesc_ty);
            Ret(bcx, td);
        }
        "type_id" => {
            let hash = ty::hash_crate_independent(
                ccx.tcx(),
                *substs.substs.types.get(FnSpace, 0),
                &ccx.link_meta.crate_hash);
            // NB: This needs to be kept in lockstep with the TypeId struct in
            //     libstd/unstable/intrinsics.rs
            let val = C_named_struct(type_of::type_of(ccx, output_type),
                                     [C_u64(ccx, hash)]);
            match bcx.fcx.llretptr.get() {
                Some(ptr) => {
                    Store(bcx, val, ptr);
                    RetVoid(bcx);
                },
                None => Ret(bcx, val)
            }
        }
        "init" => {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            match bcx.fcx.llretptr.get() {
                Some(ptr) => { Store(bcx, C_null(lltp_ty), ptr); RetVoid(bcx); }
                None if ty::type_is_nil(tp_ty) => RetVoid(bcx),
                None => Ret(bcx, C_null(lltp_ty)),
            }
        }
        "uninit" => {
            // Do nothing, this is effectively a no-op
            let retty = *substs.substs.types.get(FnSpace, 0);
            if type_is_immediate(ccx, retty) && !return_type_is_void(ccx, retty) {
                unsafe {
                    Ret(bcx, lib::llvm::llvm::LLVMGetUndef(type_of(ccx, retty).to_ref()));
                }
            } else {
                RetVoid(bcx)
            }
        }
        "forget" => {
            RetVoid(bcx);
        }
        "transmute" => {
            let (in_type, out_type) = (*substs.substs.types.get(FnSpace, 0),
                                       *substs.substs.types.get(FnSpace, 1));
            let llintype = type_of::type_of(ccx, in_type);
            let llouttype = type_of::type_of(ccx, out_type);

            let in_type_size = machine::llbitsize_of_real(ccx, llintype);
            let out_type_size = machine::llbitsize_of_real(ccx, llouttype);
            if in_type_size != out_type_size {
                let sp = match ccx.tcx.map.get(ref_id.unwrap()) {
                    ast_map::NodeExpr(e) => e.span,
                    _ => fail!("transmute has non-expr arg"),
                };
                ccx.sess().span_bug(sp,
                    format!("transmute called on types with different sizes: \
                             {} ({} bit{}) to \
                             {} ({} bit{})",
                            ty_to_str(ccx.tcx(), in_type),
                            in_type_size,
                            if in_type_size == 1 {""} else {"s"},
                            ty_to_str(ccx.tcx(), out_type),
                            out_type_size,
                            if out_type_size == 1 {""} else {"s"}).as_slice());
            }

            if !return_type_is_void(ccx, out_type) {
                let llsrcval = get_param(decl, first_real_arg);
                if type_is_immediate(ccx, in_type) {
                    match fcx.llretptr.get() {
                        Some(llretptr) => {
                            Store(bcx, llsrcval, PointerCast(bcx, llretptr, llintype.ptr_to()));
                            RetVoid(bcx);
                        }
                        None => match (llintype.kind(), llouttype.kind()) {
                            (Pointer, other) | (other, Pointer) if other != Pointer => {
                                let tmp = Alloca(bcx, llouttype, "");
                                Store(bcx, llsrcval, PointerCast(bcx, tmp, llintype.ptr_to()));
                                Ret(bcx, Load(bcx, tmp));
                            }
                            (Array, _) | (_, Array) | (Struct, _) | (_, Struct) => {
                                let tmp = Alloca(bcx, llouttype, "");
                                Store(bcx, llsrcval, PointerCast(bcx, tmp, llintype.ptr_to()));
                                Ret(bcx, Load(bcx, tmp));
                            }
                            _ => {
                                let llbitcast = BitCast(bcx, llsrcval, llouttype);
                                Ret(bcx, llbitcast)
                            }
                        }
                    }
                } else if type_is_immediate(ccx, out_type) {
                    let llsrcptr = PointerCast(bcx, llsrcval, llouttype.ptr_to());
                    let ll_load = Load(bcx, llsrcptr);
                    Ret(bcx, ll_load);
                } else {
                    // NB: Do not use a Load and Store here. This causes massive
                    // code bloat when `transmute` is used on large structural
                    // types.
                    let lldestptr = fcx.llretptr.get().unwrap();
                    let lldestptr = PointerCast(bcx, lldestptr, Type::i8p(ccx));
                    let llsrcptr = PointerCast(bcx, llsrcval, Type::i8p(ccx));

                    let llsize = llsize_of(ccx, llintype);
                    call_memcpy(bcx, lldestptr, llsrcptr, llsize, 1);
                    RetVoid(bcx);
                };
            } else {
                RetVoid(bcx);
            }
        }
        "needs_drop" => {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            Ret(bcx, C_bool(ccx, ty::type_needs_drop(ccx.tcx(), tp_ty)));
        }
        "owns_managed" => {
            let tp_ty = *substs.substs.types.get(FnSpace, 0);
            Ret(bcx, C_bool(ccx, ty::type_contents(ccx.tcx(), tp_ty).owns_managed()));
        }
        "visit_tydesc" => {
            let td = get_param(decl, first_real_arg);
            let visitor = get_param(decl, first_real_arg + 1u);
            let td = PointerCast(bcx, td, ccx.tydesc_type().ptr_to());
            glue::call_visit_glue(bcx, visitor, td, None);
            RetVoid(bcx);
        }
        "offset" => {
            let ptr = get_param(decl, first_real_arg);
            let offset = get_param(decl, first_real_arg + 1);
            let lladdr = InBoundsGEP(bcx, ptr, [offset]);
            Ret(bcx, lladdr);
        }
        "copy_nonoverlapping_memory" => {
            copy_intrinsic(bcx, false, false, *substs.substs.types.get(FnSpace, 0))
        }
        "copy_memory" => {
            copy_intrinsic(bcx, true, false, *substs.substs.types.get(FnSpace, 0))
        }
        "set_memory" => {
            memset_intrinsic(bcx, false, *substs.substs.types.get(FnSpace, 0))
        }

        "volatile_copy_nonoverlapping_memory" => {
            copy_intrinsic(bcx, false, true, *substs.substs.types.get(FnSpace, 0))
        }

        "volatile_copy_memory" => {
            copy_intrinsic(bcx, true, true, *substs.substs.types.get(FnSpace, 0))
        }

        "volatile_set_memory" => {
            memset_intrinsic(bcx, true, *substs.substs.types.get(FnSpace, 0))
        }

        "ctlz8" => count_zeros_intrinsic(bcx, "llvm.ctlz.i8"),
        "ctlz16" => count_zeros_intrinsic(bcx, "llvm.ctlz.i16"),
        "ctlz32" => count_zeros_intrinsic(bcx, "llvm.ctlz.i32"),
        "ctlz64" => count_zeros_intrinsic(bcx, "llvm.ctlz.i64"),
        "cttz8" => count_zeros_intrinsic(bcx, "llvm.cttz.i8"),
        "cttz16" => count_zeros_intrinsic(bcx, "llvm.cttz.i16"),
        "cttz32" => count_zeros_intrinsic(bcx, "llvm.cttz.i32"),
        "cttz64" => count_zeros_intrinsic(bcx, "llvm.cttz.i64"),

        "volatile_load" => volatile_load_intrinsic(bcx),
        "volatile_store" => volatile_store_intrinsic(bcx),

        "i8_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i8", output_type),
        "i16_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i16", output_type),
        "i32_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i32", output_type),
        "i64_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i64", output_type),

        "u8_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i8", output_type),
        "u16_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i16", output_type),
        "u32_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i32", output_type),
        "u64_add_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i64", output_type),

        "i8_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i8", output_type),
        "i16_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i16", output_type),
        "i32_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i32", output_type),
        "i64_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i64", output_type),

        "u8_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i8", output_type),
        "u16_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i16", output_type),
        "u32_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i32", output_type),
        "u64_sub_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i64", output_type),

        "i8_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i8", output_type),
        "i16_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i16", output_type),
        "i32_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i32", output_type),
        "i64_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i64", output_type),

        "u8_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i8", output_type),
        "u16_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i16", output_type),
        "u32_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i32", output_type),
        "u64_mul_with_overflow" =>
            with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i64", output_type),

        _ => {
            // Could we make this an enum rather than a string? does it get
            // checked earlier?
            ccx.sess().span_bug(item.span, "unknown intrinsic");
        }
    }
    fcx.cleanup();
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
                        ty_to_str(ccx.tcx(), transmute_restriction.from),
                        from_type_size as uint,
                        if from_type_size == 1 {
                            ""
                        } else {
                            "s"
                        },
                        ty_to_str(ccx.tcx(), transmute_restriction.to),
                        to_type_size as uint,
                        if to_type_size == 1 {
                            ""
                        } else {
                            "s"
                        }).as_slice());
        }
    }
    ccx.sess().abort_if_errors();
}

