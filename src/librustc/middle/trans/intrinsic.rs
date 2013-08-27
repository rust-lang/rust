// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::{abi};
use lib::llvm::{SequentiallyConsistent, Acquire, Release, Xchg};
use lib::llvm::{ValueRef, Pointer};
use lib;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee::*;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::trans::expr::Ignore;
use middle::trans::machine;
use middle::trans::glue;
use middle::ty::FnSig;
use middle::ty;
use syntax::ast;
use syntax::ast_map;
use syntax::attr;
use syntax::opt_vec;
use util::ppaux::{ty_to_str};
use middle::trans::machine::llsize_of;
use middle::trans::type_::Type;

pub fn trans_intrinsic(ccx: @mut CrateContext,
                       decl: ValueRef,
                       item: &ast::foreign_item,
                       path: ast_map::path,
                       substs: @param_substs,
                       attributes: &[ast::Attribute],
                       ref_id: Option<ast::NodeId>) {
    debug!("trans_intrinsic(item.ident=%s)", ccx.sess.str_of(item.ident));

    fn with_overflow_instrinsic(bcx: @mut Block, name: &'static str) {
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let a = get_param(bcx.fcx.llfn, first_real_arg);
        let b = get_param(bcx.fcx.llfn, first_real_arg + 1);
        let llfn = bcx.ccx().intrinsics.get_copy(&name);

        // convert `i1` to a `bool`, and write to the out parameter
        let val = Call(bcx, llfn, [a, b]);
        let result = ExtractValue(bcx, val, 0);
        let overflow = ZExt(bcx, ExtractValue(bcx, val, 1), Type::bool());
        let retptr = get_param(bcx.fcx.llfn, bcx.fcx.out_arg_pos());
        let ret = Load(bcx, retptr);
        let ret = InsertValue(bcx, ret, result, 0);
        let ret = InsertValue(bcx, ret, overflow, 1);
        Store(bcx, ret, retptr);
        RetVoid(bcx)
    }

    fn memcpy_intrinsic(bcx: @mut Block, name: &'static str, tp_ty: ty::t, sizebits: u8) {
        let ccx = bcx.ccx();
        let lltp_ty = type_of::type_of(ccx, tp_ty);
        let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
        let size = match sizebits {
            32 => C_i32(machine::llsize_of_real(ccx, lltp_ty) as i32),
            64 => C_i64(machine::llsize_of_real(ccx, lltp_ty) as i64),
            _ => ccx.sess.fatal("Invalid value for sizebits")
        };

        let decl = bcx.fcx.llfn;
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), Type::i8p());
        let src_ptr = PointerCast(bcx, get_param(decl, first_real_arg + 1), Type::i8p());
        let count = get_param(decl, first_real_arg + 2);
        let volatile = C_i1(false);
        let llfn = bcx.ccx().intrinsics.get_copy(&name);
        Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align, volatile]);
        RetVoid(bcx);
    }

    fn memset_intrinsic(bcx: @mut Block, name: &'static str, tp_ty: ty::t, sizebits: u8) {
        let ccx = bcx.ccx();
        let lltp_ty = type_of::type_of(ccx, tp_ty);
        let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
        let size = match sizebits {
            32 => C_i32(machine::llsize_of_real(ccx, lltp_ty) as i32),
            64 => C_i64(machine::llsize_of_real(ccx, lltp_ty) as i64),
            _ => ccx.sess.fatal("Invalid value for sizebits")
        };

        let decl = bcx.fcx.llfn;
        let first_real_arg = bcx.fcx.arg_pos(0u);
        let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), Type::i8p());
        let val = get_param(decl, first_real_arg + 1);
        let count = get_param(decl, first_real_arg + 2);
        let volatile = C_i1(false);
        let llfn = bcx.ccx().intrinsics.get_copy(&name);
        Call(bcx, llfn, [dst_ptr, val, Mul(bcx, size, count), align, volatile]);
        RetVoid(bcx);
    }

    fn count_zeros_intrinsic(bcx: @mut Block, name: &'static str) {
        let x = get_param(bcx.fcx.llfn, bcx.fcx.arg_pos(0u));
        let y = C_i1(false);
        let llfn = bcx.ccx().intrinsics.get_copy(&name);
        Ret(bcx, Call(bcx, llfn, [x, y]));
    }

    let output_type = ty::ty_fn_ret(ty::node_id_to_type(ccx.tcx, item.id));

    let fcx = new_fn_ctxt_w_id(ccx,
                               path,
                               decl,
                               item.id,
                               output_type,
                               true,
                               Some(substs),
                               None,
                               Some(item.span));

    set_always_inline(fcx.llfn);

    // Set the fixed stack segment flag if necessary.
    if attr::contains_name(attributes, "fixed_stack_segment") {
        set_fixed_stack_segment(fcx.llfn);
    }

    let mut bcx = fcx.entry_bcx.unwrap();
    let first_real_arg = fcx.arg_pos(0u);

    let nm = ccx.sess.str_of(item.ident);
    let name = nm.as_slice();

    // This requires that atomic intrinsics follow a specific naming pattern:
    // "atomic_<operation>[_<ordering>], and no ordering means SeqCst
    if name.starts_with("atomic_") {
        let split : ~[&str] = name.split_iter('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic not correct format");
        let order = if split.len() == 2 {
            lib::llvm::SequentiallyConsistent
        } else {
            match split[2] {
                "relaxed" => lib::llvm::Monotonic,
                "acq"     => lib::llvm::Acquire,
                "rel"     => lib::llvm::Release,
                "acqrel"  => lib::llvm::AcquireRelease,
                _ => ccx.sess.fatal("Unknown ordering in atomic intrinsic")
            }
        };

        match split[1] {
            "cxchg" => {
                let old = AtomicCmpXchg(bcx, get_param(decl, first_real_arg),
                                        get_param(decl, first_real_arg + 1u),
                                        get_param(decl, first_real_arg + 2u),
                                        order);
                Ret(bcx, old);
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
                    _ => ccx.sess.fatal("Unknown atomic operation")
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

    match name {
        "size_of" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Ret(bcx, C_uint(ccx, machine::llsize_of_real(ccx, lltp_ty)));
        }
        "move_val_init" => {
            // See comments for `"move_val"`.
            let tp_ty = substs.tys[0];
            let mode = appropriate_mode(ccx.tcx, tp_ty);
            let src = Datum {val: get_param(decl, first_real_arg + 1u),
                             ty: tp_ty, mode: mode};
            bcx = src.move_to(bcx, INIT, get_param(decl, first_real_arg));
            RetVoid(bcx);
        }
        "min_align_of" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Ret(bcx, C_uint(ccx, machine::llalign_of_min(ccx, lltp_ty)));
        }
        "pref_align_of"=> {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Ret(bcx, C_uint(ccx, machine::llalign_of_pref(ccx, lltp_ty)));
        }
        "get_tydesc" => {
            let tp_ty = substs.tys[0];
            let static_ti = get_tydesc(ccx, tp_ty);
            glue::lazily_emit_all_tydesc_glue(ccx, static_ti);

            // FIXME (#3730): ideally this shouldn't need a cast,
            // but there's a circularity between translating rust types to llvm
            // types and having a tydesc type available. So I can't directly access
            // the llvm type of intrinsic::TyDesc struct.
            let userland_tydesc_ty = type_of::type_of(ccx, output_type);
            let td = PointerCast(bcx, static_ti.tydesc, userland_tydesc_ty);
            Ret(bcx, td);
        }
        "init" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            match bcx.fcx.llretptr {
                Some(ptr) => { Store(bcx, C_null(lltp_ty), ptr); RetVoid(bcx); }
                None if ty::type_is_nil(tp_ty) => RetVoid(bcx),
                None => Ret(bcx, C_null(lltp_ty)),
            }
        }
        "uninit" => {
            // Do nothing, this is effectively a no-op
            let retty = substs.tys[0];
            if ty::type_is_immediate(ccx.tcx, retty) && !ty::type_is_nil(retty) {
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
            let (in_type, out_type) = (substs.tys[0], substs.tys[1]);
            let llintype = type_of::type_of(ccx, in_type);
            let llouttype = type_of::type_of(ccx, out_type);

            let in_type_size = machine::llbitsize_of_real(ccx, llintype);
            let out_type_size = machine::llbitsize_of_real(ccx, llouttype);
            if in_type_size != out_type_size {
                let sp = match ccx.tcx.items.get_copy(&ref_id.unwrap()) {
                    ast_map::node_expr(e) => e.span,
                    _ => fail!("transmute has non-expr arg"),
                };
                let pluralize = |n| if 1u == n { "" } else { "s" };
                ccx.sess.span_fatal(sp,
                                    fmt!("transmute called on types with \
                                          different sizes: %s (%u bit%s) to \
                                          %s (%u bit%s)",
                                         ty_to_str(ccx.tcx, in_type),
                                         in_type_size,
                                         pluralize(in_type_size),
                                         ty_to_str(ccx.tcx, out_type),
                                         out_type_size,
                                         pluralize(out_type_size)));
            }

            if !ty::type_is_voidish(out_type) {
                let llsrcval = get_param(decl, first_real_arg);
                if ty::type_is_immediate(ccx.tcx, in_type) {
                    match fcx.llretptr {
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
                            _ => Ret(bcx, BitCast(bcx, llsrcval, llouttype))
                        }
                    }
                } else if ty::type_is_immediate(ccx.tcx, out_type) {
                    let llsrcptr = PointerCast(bcx, llsrcval, llouttype.ptr_to());
                    Ret(bcx, Load(bcx, llsrcptr));
                } else {
                    // NB: Do not use a Load and Store here. This causes massive
                    // code bloat when `transmute` is used on large structural
                    // types.
                    let lldestptr = fcx.llretptr.unwrap();
                    let lldestptr = PointerCast(bcx, lldestptr, Type::i8p());
                    let llsrcptr = PointerCast(bcx, llsrcval, Type::i8p());

                    let llsize = llsize_of(ccx, llintype);
                    call_memcpy(bcx, lldestptr, llsrcptr, llsize, 1);
                    RetVoid(bcx);
                };
            } else {
                RetVoid(bcx);
            }
        }
        "needs_drop" => {
            let tp_ty = substs.tys[0];
            Ret(bcx, C_bool(ty::type_needs_drop(ccx.tcx, tp_ty)));
        }
        "contains_managed" => {
            let tp_ty = substs.tys[0];
            Ret(bcx, C_bool(ty::type_contents(ccx.tcx, tp_ty).contains_managed()));
        }
        "visit_tydesc" => {
            let td = get_param(decl, first_real_arg);
            let visitor = get_param(decl, first_real_arg + 1u);
            let td = PointerCast(bcx, td, ccx.tydesc_type.ptr_to());
            glue::call_tydesc_glue_full(bcx, visitor, td,
                                        abi::tydesc_field_visit_glue, None);
            RetVoid(bcx);
        }
        "offset" => {
            let ptr = get_param(decl, first_real_arg);
            let offset = get_param(decl, first_real_arg + 1);
            Ret(bcx, InBoundsGEP(bcx, ptr, [offset]));
        }
        "memcpy32" => memcpy_intrinsic(bcx, "llvm.memcpy.p0i8.p0i8.i32", substs.tys[0], 32),
        "memcpy64" => memcpy_intrinsic(bcx, "llvm.memcpy.p0i8.p0i8.i64", substs.tys[0], 64),
        "memmove32" => memcpy_intrinsic(bcx, "llvm.memmove.p0i8.p0i8.i32", substs.tys[0], 32),
        "memmove64" => memcpy_intrinsic(bcx, "llvm.memmove.p0i8.p0i8.i64", substs.tys[0], 64),
        "memset32" => memset_intrinsic(bcx, "llvm.memset.p0i8.i32", substs.tys[0], 32),
        "memset64" => memset_intrinsic(bcx, "llvm.memset.p0i8.i64", substs.tys[0], 64),
        "ctlz8" => count_zeros_intrinsic(bcx, "llvm.ctlz.i8"),
        "ctlz16" => count_zeros_intrinsic(bcx, "llvm.ctlz.i16"),
        "ctlz32" => count_zeros_intrinsic(bcx, "llvm.ctlz.i32"),
        "ctlz64" => count_zeros_intrinsic(bcx, "llvm.ctlz.i64"),
        "cttz8" => count_zeros_intrinsic(bcx, "llvm.cttz.i8"),
        "cttz16" => count_zeros_intrinsic(bcx, "llvm.cttz.i16"),
        "cttz32" => count_zeros_intrinsic(bcx, "llvm.cttz.i32"),
        "cttz64" => count_zeros_intrinsic(bcx, "llvm.cttz.i64"),

        "i8_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i8"),
        "i16_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i16"),
        "i32_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i32"),
        "i64_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.sadd.with.overflow.i64"),

        "u8_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i8"),
        "u16_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i16"),
        "u32_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i32"),
        "u64_add_with_overflow" => with_overflow_instrinsic(bcx, "llvm.uadd.with.overflow.i64"),

        "i8_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i8"),
        "i16_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i16"),
        "i32_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i32"),
        "i64_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.ssub.with.overflow.i64"),

        "u8_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i8"),
        "u16_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i16"),
        "u32_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i32"),
        "u64_sub_with_overflow" => with_overflow_instrinsic(bcx, "llvm.usub.with.overflow.i64"),

        "i8_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i8"),
        "i16_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i16"),
        "i32_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i32"),
        "i64_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.smul.with.overflow.i64"),

        "u8_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i8"),
        "u16_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i16"),
        "u32_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i32"),
        "u64_mul_with_overflow" => with_overflow_instrinsic(bcx, "llvm.umul.with.overflow.i64"),

        _ => {
            // Could we make this an enum rather than a string? does it get
            // checked earlier?
            ccx.sess.span_bug(item.span, "unknown intrinsic");
        }
    }
    fcx.cleanup();
}
