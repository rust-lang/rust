// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;
use back::{abi};
use lib::llvm::{SequentiallyConsistent, Acquire, Release, Xchg};
use lib::llvm::{ValueRef};
use lib;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee::*;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::trans::expr::Ignore;
use middle::ty::FnSig;
use middle::ty;
use syntax::ast;
use syntax::ast_map;
use syntax::attr;
use syntax::opt_vec;
use middle::trans::machine;
use middle::trans::glue;
use util::ppaux::{ty_to_str};
use middle::trans::machine::llsize_of;

pub fn trans_intrinsic(ccx: @CrateContext,
                       decl: ValueRef,
                       item: @ast::foreign_item,
                       path: ast_map::path,
                       substs: @param_substs,
                       attributes: &[ast::attribute],
                       ref_id: Option<ast::node_id>) {
    debug!("trans_intrinsic(item.ident=%s)", *ccx.sess.str_of(item.ident));

    let output_type = ty::ty_fn_ret(ty::node_id_to_type(ccx.tcx, item.id));

    let fcx = new_fn_ctxt_w_id(ccx,
                               path,
                               decl,
                               item.id,
                               output_type,
                               None,
                               Some(substs),
                               Some(item.span));

    // Set the fixed stack segment flag if necessary.
    if attr::attrs_contains_name(attributes, "fixed_stack_segment") {
        set_fixed_stack_segment(fcx.llfn);
    }

    let mut bcx = top_scope_block(fcx, None);
    let lltop = bcx.llbb;
    match *ccx.sess.str_of(item.ident) {
        ~"atomic_cxchg" => {
            let old = AtomicCmpXchg(bcx,
                                    get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    get_param(decl, first_real_arg + 2u),
                                    SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_cxchg_acq" => {
            let old = AtomicCmpXchg(bcx,
                                    get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    get_param(decl, first_real_arg + 2u),
                                    Acquire);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_cxchg_rel" => {
            let old = AtomicCmpXchg(bcx,
                                    get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    get_param(decl, first_real_arg + 2u),
                                    Release);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_load" => {
            let old = AtomicLoad(bcx,
                                 get_param(decl, first_real_arg),
                                 SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_load_acq" => {
            let old = AtomicLoad(bcx,
                                 get_param(decl, first_real_arg),
                                 Acquire);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_store" => {
            AtomicStore(bcx,
                        get_param(decl, first_real_arg + 1u),
                        get_param(decl, first_real_arg),
                        SequentiallyConsistent);
        }
        ~"atomic_store_rel" => {
            AtomicStore(bcx,
                        get_param(decl, first_real_arg + 1u),
                        get_param(decl, first_real_arg),
                        Release);
        }
        ~"atomic_xchg" => {
            let old = AtomicRMW(bcx, Xchg,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xchg_acq" => {
            let old = AtomicRMW(bcx, Xchg,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Acquire);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xchg_rel" => {
            let old = AtomicRMW(bcx, Xchg,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Release);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xadd" => {
            let old = AtomicRMW(bcx, lib::llvm::Add,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xadd_acq" => {
            let old = AtomicRMW(bcx, lib::llvm::Add,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Acquire);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xadd_rel" => {
            let old = AtomicRMW(bcx, lib::llvm::Add,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Release);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xsub" => {
            let old = AtomicRMW(bcx, lib::llvm::Sub,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xsub_acq" => {
            let old = AtomicRMW(bcx, lib::llvm::Sub,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Acquire);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"atomic_xsub_rel" => {
            let old = AtomicRMW(bcx, lib::llvm::Sub,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Release);
            Store(bcx, old, fcx.llretptr.get());
        }
        ~"size_of" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Store(bcx, C_uint(ccx, machine::llsize_of_real(ccx, lltp_ty)),
                  fcx.llretptr.get());
        }
        ~"move_val" => {
            // Create a datum reflecting the value being moved:
            //
            // - the datum will be by ref if the value is non-immediate;
            //
            // - the datum has a RevokeClean source because, that way,
            //   the `move_to()` method does not feel compelled to
            //   zero out the memory where the datum resides.  Zeroing
            //   is not necessary since, for intrinsics, there is no
            //   cleanup to concern ourselves with.
            let tp_ty = substs.tys[0];
            let mode = appropriate_mode(tp_ty);
            let src = Datum {val: get_param(decl, first_real_arg + 1u),
                             ty: tp_ty, mode: mode, source: RevokeClean};
            bcx = src.move_to(bcx, DROP_EXISTING,
                              get_param(decl, first_real_arg));
        }
        ~"move_val_init" => {
            // See comments for `"move_val"`.
            let tp_ty = substs.tys[0];
            let mode = appropriate_mode(tp_ty);
            let src = Datum {val: get_param(decl, first_real_arg + 1u),
                             ty: tp_ty, mode: mode, source: RevokeClean};
            bcx = src.move_to(bcx, INIT, get_param(decl, first_real_arg));
        }
        ~"min_align_of" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Store(bcx, C_uint(ccx, machine::llalign_of_min(ccx, lltp_ty)),
                  fcx.llretptr.get());
        }
        ~"pref_align_of"=> {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Store(bcx, C_uint(ccx, machine::llalign_of_pref(ccx, lltp_ty)),
                  fcx.llretptr.get());
        }
        ~"get_tydesc" => {
            let tp_ty = substs.tys[0];
            let static_ti = get_tydesc(ccx, tp_ty);
            glue::lazily_emit_all_tydesc_glue(ccx, static_ti);

            // FIXME (#3727): change this to T_ptr(ccx.tydesc_ty) when the
            // core::sys copy of the get_tydesc interface dies off.
            let td = PointerCast(bcx, static_ti.tydesc, T_ptr(T_nil()));
            Store(bcx, td, fcx.llretptr.get());
        }
        ~"init" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            if !ty::type_is_nil(tp_ty) {
                Store(bcx, C_null(lltp_ty), fcx.llretptr.get());
            }
        }
        ~"uninit" => {
            // Do nothing, this is effectively a no-op
        }
        ~"forget" => {}
        ~"transmute" => {
            let (in_type, out_type) = (substs.tys[0], substs.tys[1]);
            let llintype = type_of::type_of(ccx, in_type);
            let llouttype = type_of::type_of(ccx, out_type);

            let in_type_size = machine::llbitsize_of_real(ccx, llintype);
            let out_type_size = machine::llbitsize_of_real(ccx, llouttype);
            if in_type_size != out_type_size {
                let sp = match ccx.tcx.items.get_copy(&ref_id.get()) {
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

            if !ty::type_is_nil(out_type) {
                // NB: Do not use a Load and Store here. This causes massive
                // code bloat when `transmute` is used on large structural
                // types.
                let lldestptr = fcx.llretptr.get();
                let lldestptr = PointerCast(bcx, lldestptr, T_ptr(T_i8()));

                let llsrcval = get_param(decl, first_real_arg);
                let llsrcptr = if ty::type_is_immediate(in_type) {
                    let llsrcptr = alloca(bcx, llintype);
                    Store(bcx, llsrcval, llsrcptr);
                    llsrcptr
                } else {
                    llsrcval
                };
                let llsrcptr = PointerCast(bcx, llsrcptr, T_ptr(T_i8()));

                let llsize = llsize_of(ccx, llintype);
                call_memcpy(bcx, lldestptr, llsrcptr, llsize, 1);
            }
        }
        ~"needs_drop" => {
            let tp_ty = substs.tys[0];
            Store(bcx,
                  C_bool(ty::type_needs_drop(ccx.tcx, tp_ty)),
                  fcx.llretptr.get());
        }
        ~"visit_tydesc" => {
            let td = get_param(decl, first_real_arg);
            let visitor = get_param(decl, first_real_arg + 1u);
            //let llvisitorptr = alloca(bcx, val_ty(visitor));
            //Store(bcx, visitor, llvisitorptr);
            let td = PointerCast(bcx, td, T_ptr(ccx.tydesc_type));
            glue::call_tydesc_glue_full(bcx,
                                        visitor,
                                        td,
                                        abi::tydesc_field_visit_glue,
                                        None);
        }
        ~"frame_address" => {
            let frameaddress = *ccx.intrinsics.get(&~"llvm.frameaddress");
            let frameaddress_val = Call(bcx, frameaddress, [C_i32(0i32)]);
            let star_u8 = ty::mk_imm_ptr(
                bcx.tcx(),
                ty::mk_mach_uint(ast::ty_u8));
            let fty = ty::mk_closure(bcx.tcx(), ty::ClosureTy {
                purity: ast::impure_fn,
                sigil: ast::BorrowedSigil,
                onceness: ast::Many,
                region: ty::re_bound(ty::br_anon(0)),
                bounds: ty::EmptyBuiltinBounds(),
                sig: FnSig {
                    bound_lifetime_names: opt_vec::Empty,
                    inputs: ~[ star_u8 ],
                    output: ty::mk_nil()
                }
            });
            let datum = Datum {val: get_param(decl, first_real_arg),
                               mode: ByRef, ty: fty, source: ZeroMem};
            let arg_vals = ~[frameaddress_val];
            bcx = trans_call_inner(
                bcx, None, fty, ty::mk_nil(),
                |bcx| Callee {bcx: bcx, data: Closure(datum)},
                ArgVals(arg_vals), Ignore, DontAutorefArg);
        }
        ~"morestack_addr" => {
            // XXX This is a hack to grab the address of this particular
            // native function. There should be a general in-language
            // way to do this
            let llfty = type_of_rust_fn(bcx.ccx(), [], ty::mk_nil());
            let morestack_addr = decl_cdecl_fn(
                bcx.ccx().llmod, "__morestack", llfty);
            let morestack_addr = PointerCast(bcx, morestack_addr,
                                             T_ptr(T_nil()));
            Store(bcx, morestack_addr, fcx.llretptr.get());
        }
        ~"memcpy32" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
            let size = C_i32(machine::llsize_of_real(ccx, lltp_ty) as i32);

            let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), T_ptr(T_i8()));
            let src_ptr = PointerCast(bcx, get_param(decl, first_real_arg + 1), T_ptr(T_i8()));
            let count = get_param(decl, first_real_arg + 2);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(&~"llvm.memcpy.p0i8.p0i8.i32");
            Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align, volatile]);
        }
        ~"memcpy64" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
            let size = C_i64(machine::llsize_of_real(ccx, lltp_ty) as i64);

            let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), T_ptr(T_i8()));
            let src_ptr = PointerCast(bcx, get_param(decl, first_real_arg + 1), T_ptr(T_i8()));
            let count = get_param(decl, first_real_arg + 2);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(&~"llvm.memcpy.p0i8.p0i8.i64");
            Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align, volatile]);
        }
        ~"memmove32" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
            let size = C_i32(machine::llsize_of_real(ccx, lltp_ty) as i32);

            let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), T_ptr(T_i8()));
            let src_ptr = PointerCast(bcx, get_param(decl, first_real_arg + 1), T_ptr(T_i8()));
            let count = get_param(decl, first_real_arg + 2);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(&~"llvm.memmove.p0i8.p0i8.i32");
            Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align, volatile]);
        }
        ~"memmove64" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
            let size = C_i64(machine::llsize_of_real(ccx, lltp_ty) as i64);

            let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), T_ptr(T_i8()));
            let src_ptr = PointerCast(bcx, get_param(decl, first_real_arg + 1), T_ptr(T_i8()));
            let count = get_param(decl, first_real_arg + 2);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(&~"llvm.memmove.p0i8.p0i8.i64");
            Call(bcx, llfn, [dst_ptr, src_ptr, Mul(bcx, size, count), align, volatile]);
        }
        ~"memset32" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
            let size = C_i32(machine::llsize_of_real(ccx, lltp_ty) as i32);

            let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), T_ptr(T_i8()));
            let val = get_param(decl, first_real_arg + 1);
            let count = get_param(decl, first_real_arg + 2);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(&~"llvm.memset.p0i8.i32");
            Call(bcx, llfn, [dst_ptr, val, Mul(bcx, size, count), align, volatile]);
        }
        ~"memset64" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let align = C_i32(machine::llalign_of_min(ccx, lltp_ty) as i32);
            let size = C_i64(machine::llsize_of_real(ccx, lltp_ty) as i64);

            let dst_ptr = PointerCast(bcx, get_param(decl, first_real_arg), T_ptr(T_i8()));
            let val = get_param(decl, first_real_arg + 1);
            let count = get_param(decl, first_real_arg + 2);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(&~"llvm.memset.p0i8.i64");
            Call(bcx, llfn, [dst_ptr, val, Mul(bcx, size, count), align, volatile]);
        }
        ~"sqrtf32" => {
            let x = get_param(decl, first_real_arg);
            let sqrtf = *ccx.intrinsics.get(&~"llvm.sqrt.f32");
            Store(bcx, Call(bcx, sqrtf, [x]), fcx.llretptr.get());
        }
        ~"sqrtf64" => {
            let x = get_param(decl, first_real_arg);
            let sqrtf = *ccx.intrinsics.get(&~"llvm.sqrt.f64");
            Store(bcx, Call(bcx, sqrtf, [x]), fcx.llretptr.get());
        }
        ~"powif32" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powif = *ccx.intrinsics.get(&~"llvm.powi.f32");
            Store(bcx, Call(bcx, powif, [a, x]), fcx.llretptr.get());
        }
        ~"powif64" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powif = *ccx.intrinsics.get(&~"llvm.powi.f64");
            Store(bcx, Call(bcx, powif, [a, x]), fcx.llretptr.get());
        }
        ~"sinf32" => {
            let x = get_param(decl, first_real_arg);
            let sinf = *ccx.intrinsics.get(&~"llvm.sin.f32");
            Store(bcx, Call(bcx, sinf, [x]), fcx.llretptr.get());
        }
        ~"sinf64" => {
            let x = get_param(decl, first_real_arg);
            let sinf = *ccx.intrinsics.get(&~"llvm.sin.f64");
            Store(bcx, Call(bcx, sinf, [x]), fcx.llretptr.get());
        }
        ~"cosf32" => {
            let x = get_param(decl, first_real_arg);
            let cosf = *ccx.intrinsics.get(&~"llvm.cos.f32");
            Store(bcx, Call(bcx, cosf, [x]), fcx.llretptr.get());
        }
        ~"cosf64" => {
            let x = get_param(decl, first_real_arg);
            let cosf = *ccx.intrinsics.get(&~"llvm.cos.f64");
            Store(bcx, Call(bcx, cosf, [x]), fcx.llretptr.get());
        }
        ~"powf32" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powf = *ccx.intrinsics.get(&~"llvm.pow.f32");
            Store(bcx, Call(bcx, powf, [a, x]), fcx.llretptr.get());
        }
        ~"powf64" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powf = *ccx.intrinsics.get(&~"llvm.pow.f64");
            Store(bcx, Call(bcx, powf, [a, x]), fcx.llretptr.get());
        }
        ~"expf32" => {
            let x = get_param(decl, first_real_arg);
            let expf = *ccx.intrinsics.get(&~"llvm.exp.f32");
            Store(bcx, Call(bcx, expf, [x]), fcx.llretptr.get());
        }
        ~"expf64" => {
            let x = get_param(decl, first_real_arg);
            let expf = *ccx.intrinsics.get(&~"llvm.exp.f64");
            Store(bcx, Call(bcx, expf, [x]), fcx.llretptr.get());
        }
        ~"exp2f32" => {
            let x = get_param(decl, first_real_arg);
            let exp2f = *ccx.intrinsics.get(&~"llvm.exp2.f32");
            Store(bcx, Call(bcx, exp2f, [x]), fcx.llretptr.get());
        }
        ~"exp2f64" => {
            let x = get_param(decl, first_real_arg);
            let exp2f = *ccx.intrinsics.get(&~"llvm.exp2.f64");
            Store(bcx, Call(bcx, exp2f, [x]), fcx.llretptr.get());
        }
        ~"logf32" => {
            let x = get_param(decl, first_real_arg);
            let logf = *ccx.intrinsics.get(&~"llvm.log.f32");
            Store(bcx, Call(bcx, logf, [x]), fcx.llretptr.get());
        }
        ~"logf64" => {
            let x = get_param(decl, first_real_arg);
            let logf = *ccx.intrinsics.get(&~"llvm.log.f64");
            Store(bcx, Call(bcx, logf, [x]), fcx.llretptr.get());
        }
        ~"log10f32" => {
            let x = get_param(decl, first_real_arg);
            let log10f = *ccx.intrinsics.get(&~"llvm.log10.f32");
            Store(bcx, Call(bcx, log10f, [x]), fcx.llretptr.get());
        }
        ~"log10f64" => {
            let x = get_param(decl, first_real_arg);
            let log10f = *ccx.intrinsics.get(&~"llvm.log10.f64");
            Store(bcx, Call(bcx, log10f, [x]), fcx.llretptr.get());
        }
        ~"log2f32" => {
            let x = get_param(decl, first_real_arg);
            let log2f = *ccx.intrinsics.get(&~"llvm.log2.f32");
            Store(bcx, Call(bcx, log2f, [x]), fcx.llretptr.get());
        }
        ~"log2f64" => {
            let x = get_param(decl, first_real_arg);
            let log2f = *ccx.intrinsics.get(&~"llvm.log2.f64");
            Store(bcx, Call(bcx, log2f, [x]), fcx.llretptr.get());
        }
        ~"fmaf32" => {
            let a = get_param(decl, first_real_arg);
            let b = get_param(decl, first_real_arg + 1u);
            let c = get_param(decl, first_real_arg + 2u);
            let fmaf = *ccx.intrinsics.get(&~"llvm.fma.f32");
            Store(bcx, Call(bcx, fmaf, [a, b, c]), fcx.llretptr.get());
        }
        ~"fmaf64" => {
            let a = get_param(decl, first_real_arg);
            let b = get_param(decl, first_real_arg + 1u);
            let c = get_param(decl, first_real_arg + 2u);
            let fmaf = *ccx.intrinsics.get(&~"llvm.fma.f64");
            Store(bcx, Call(bcx, fmaf, [a, b, c]), fcx.llretptr.get());
        }
        ~"fabsf32" => {
            let x = get_param(decl, first_real_arg);
            let fabsf = *ccx.intrinsics.get(&~"llvm.fabs.f32");
            Store(bcx, Call(bcx, fabsf, [x]), fcx.llretptr.get());
        }
        ~"fabsf64" => {
            let x = get_param(decl, first_real_arg);
            let fabsf = *ccx.intrinsics.get(&~"llvm.fabs.f64");
            Store(bcx, Call(bcx, fabsf, [x]), fcx.llretptr.get());
        }
        ~"floorf32" => {
            let x = get_param(decl, first_real_arg);
            let floorf = *ccx.intrinsics.get(&~"llvm.floor.f32");
            Store(bcx, Call(bcx, floorf, [x]), fcx.llretptr.get());
        }
        ~"floorf64" => {
            let x = get_param(decl, first_real_arg);
            let floorf = *ccx.intrinsics.get(&~"llvm.floor.f64");
            Store(bcx, Call(bcx, floorf, [x]), fcx.llretptr.get());
        }
        ~"ceilf32" => {
            let x = get_param(decl, first_real_arg);
            let ceilf = *ccx.intrinsics.get(&~"llvm.ceil.f32");
            Store(bcx, Call(bcx, ceilf, [x]), fcx.llretptr.get());
        }
        ~"ceilf64" => {
            let x = get_param(decl, first_real_arg);
            let ceilf = *ccx.intrinsics.get(&~"llvm.ceil.f64");
            Store(bcx, Call(bcx, ceilf, [x]), fcx.llretptr.get());
        }
        ~"truncf32" => {
            let x = get_param(decl, first_real_arg);
            let truncf = *ccx.intrinsics.get(&~"llvm.trunc.f32");
            Store(bcx, Call(bcx, truncf, [x]), fcx.llretptr.get());
        }
        ~"truncf64" => {
            let x = get_param(decl, first_real_arg);
            let truncf = *ccx.intrinsics.get(&~"llvm.trunc.f64");
            Store(bcx, Call(bcx, truncf, [x]), fcx.llretptr.get());
        }
        ~"ctpop8" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i8");
            Store(bcx, Call(bcx, ctpop, [x]), fcx.llretptr.get())
        }
        ~"ctpop16" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i16");
            Store(bcx, Call(bcx, ctpop, [x]), fcx.llretptr.get())
        }
        ~"ctpop32" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i32");
            Store(bcx, Call(bcx, ctpop, [x]), fcx.llretptr.get())
        }
        ~"ctpop64" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i64");
            Store(bcx, Call(bcx, ctpop, [x]), fcx.llretptr.get())
        }
        ~"ctlz8" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i8");
            Store(bcx, Call(bcx, ctlz, [x, y]), fcx.llretptr.get())
        }
        ~"ctlz16" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i16");
            Store(bcx, Call(bcx, ctlz, [x, y]), fcx.llretptr.get())
        }
        ~"ctlz32" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i32");
            Store(bcx, Call(bcx, ctlz, [x, y]), fcx.llretptr.get())
        }
        ~"ctlz64" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i64");
            Store(bcx, Call(bcx, ctlz, [x, y]), fcx.llretptr.get())
        }
        ~"cttz8" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i8");
            Store(bcx, Call(bcx, cttz, [x, y]), fcx.llretptr.get())
        }
        ~"cttz16" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i16");
            Store(bcx, Call(bcx, cttz, [x, y]), fcx.llretptr.get())
        }
        ~"cttz32" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i32");
            Store(bcx, Call(bcx, cttz, [x, y]), fcx.llretptr.get())
        }
        ~"cttz64" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i64");
            Store(bcx, Call(bcx, cttz, [x, y]), fcx.llretptr.get())
        }
        ~"bswap16" => {
            let x = get_param(decl, first_real_arg);
            let cttz = *ccx.intrinsics.get(&~"llvm.bswap.i16");
            Store(bcx, Call(bcx, cttz, [x]), fcx.llretptr.get())
        }
        ~"bswap32" => {
            let x = get_param(decl, first_real_arg);
            let cttz = *ccx.intrinsics.get(&~"llvm.bswap.i32");
            Store(bcx, Call(bcx, cttz, [x]), fcx.llretptr.get())
        }
        ~"bswap64" => {
            let x = get_param(decl, first_real_arg);
            let cttz = *ccx.intrinsics.get(&~"llvm.bswap.i64");
            Store(bcx, Call(bcx, cttz, [x]), fcx.llretptr.get())
        }
        _ => {
            // Could we make this an enum rather than a string? does it get
            // checked earlier?
            ccx.sess.span_bug(item.span, "unknown intrinsic");
        }
    }
    build_return(bcx);
    finish_fn(fcx, lltop);
}
