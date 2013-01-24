// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// trans.rs: Translate the completed AST to the LLVM IR.
//
// Some functions here, such as trans_block and trans_expr, return a value --
// the result of the translation to LLVM -- while others, such as trans_fn,
// trans_impl, and trans_item, are called only for the side effect of adding a
// particular definition to the LLVM IR output we're producing.
//
// Hopefully useful general knowledge about trans:
//
//   * There's no way to find out the ty::t type of a ValueRef.  Doing so
//     would be "trying to get the eggs out of an omelette" (credit:
//     pcwalton).  You can, instead, find out its TypeRef by calling val_ty,
//     but many TypeRefs correspond to one ty::t; for instance, tup(int, int,
//     int) and rec(x=int, y=int, z=int) will have the same TypeRef.

use core::prelude::*;

use back::link::{mangle_exported_name};
use back::link::{mangle_internal_name_by_path_and_seq};
use back::link::{mangle_internal_name_by_path};
use back::link::{mangle_internal_name_by_seq};
use back::link::{mangle_internal_name_by_type_only};
use back::{link, abi, upcall};
use driver::session;
use driver::session::Session;
use lib::llvm::{ModuleRef, ValueRef, TypeRef, BasicBlockRef};
use lib::llvm::{True, False};
use lib::llvm::{llvm, mk_target_data, mk_type_names};
use lib;
use metadata::common::link_meta;
use metadata::{csearch, cstore, decoder, encoder};
use middle::astencode;
use middle::borrowck::RootInfo;
use middle::pat_util::*;
use middle::resolve;
use middle::trans::_match;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::controlflow;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::foreign;
use middle::trans::glue;
use middle::trans::inline;
use middle::trans::meth;
use middle::trans::monomorphize;
use middle::trans::reachable;
use middle::trans::shape::*;
use middle::trans::tvec;
use middle::trans::type_of::*;
use util::common::indenter;
use util::ppaux::{ty_to_str, ty_to_short_str};
use util::ppaux;

use core::either;
use core::hash;
use core::int;
use core::io;
use core::libc::{c_uint, c_ulonglong};
use core::option::{is_none, is_some};
use core::option;
use core::uint;
use std::map::HashMap;
use std::smallintmap;
use std::{map, time, list};
use syntax::ast_map::{path, path_elt_to_str, path_mod, path_name};
use syntax::ast_util::{def_id_of_def, local_def, path_to_ident};
use syntax::attr;
use syntax::codemap::span;
use syntax::diagnostic::expect;
use syntax::parse::token::special_idents;
use syntax::print::pprust::{expr_to_str, stmt_to_str, path_to_str};
use syntax::visit;
use syntax::visit::vt;
use syntax::{ast, ast_util, codemap, ast_map};

struct icx_popper {
    ccx: @crate_ctxt,
    drop {
      if self.ccx.sess.count_llvm_insns() {
          self.ccx.stats.llvm_insn_ctxt.pop();
      }
    }
}

fn icx_popper(ccx: @crate_ctxt) -> icx_popper {
    icx_popper {
        ccx: ccx
    }
}

trait get_insn_ctxt {
    fn insn_ctxt(s: &str) -> icx_popper;
}

impl @crate_ctxt: get_insn_ctxt {
    fn insn_ctxt(s: &str) -> icx_popper {
        debug!("new insn_ctxt: %s", s);
        if self.sess.count_llvm_insns() {
            self.stats.llvm_insn_ctxt.push(str::from_slice(s));
        }
        icx_popper(self)
    }
}

impl block: get_insn_ctxt {
    fn insn_ctxt(s: &str) -> icx_popper {
        self.ccx().insn_ctxt(s)
    }
}

impl fn_ctxt: get_insn_ctxt {
    fn insn_ctxt(s: &str) -> icx_popper {
        self.ccx.insn_ctxt(s)
    }
}

fn log_fn_time(ccx: @crate_ctxt, +name: ~str, start: time::Timespec,
               end: time::Timespec) {
    let elapsed = 1000 * ((end.sec - start.sec) as int) +
        ((end.nsec as int) - (start.nsec as int)) / 1000000;
    ccx.stats.fn_times.push({ident: name, time: elapsed});
}

fn decl_fn(llmod: ModuleRef, name: ~str, cc: lib::llvm::CallConv,
           llty: TypeRef) -> ValueRef {
    let llfn: ValueRef = str::as_c_str(name, |buf| {
        unsafe {
            llvm::LLVMGetOrInsertFunction(llmod, buf, llty)
        }
    });
    unsafe {
        lib::llvm::SetFunctionCallConv(llfn, cc);
    }
    return llfn;
}

fn decl_cdecl_fn(llmod: ModuleRef, +name: ~str, llty: TypeRef) -> ValueRef {
    return decl_fn(llmod, name, lib::llvm::CCallConv, llty);
}

// Only use this if you are going to actually define the function. It's
// not valid to simply declare a function as internal.
fn decl_internal_cdecl_fn(llmod: ModuleRef, +name: ~str, llty: TypeRef) ->
   ValueRef {
    let llfn = decl_cdecl_fn(llmod, name, llty);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    return llfn;
}

fn get_extern_fn(externs: HashMap<~str, ValueRef>,
                 llmod: ModuleRef,
                 +name: ~str,
                 cc: lib::llvm::CallConv,
                 ty: TypeRef) -> ValueRef {
    // XXX: Bad copy.
    if externs.contains_key(copy name) { return externs.get(name); }
    // XXX: Bad copy.
    let f = decl_fn(llmod, copy name, cc, ty);
    externs.insert(name, f);
    return f;
}

fn get_extern_const(externs: HashMap<~str, ValueRef>, llmod: ModuleRef,
                    +name: ~str, ty: TypeRef) -> ValueRef {
    unsafe {
        // XXX: Bad copy.
        if externs.contains_key(copy name) { return externs.get(name); }
        let c = str::as_c_str(name, |buf| {
            llvm::LLVMAddGlobal(llmod, ty, buf)
        });
        externs.insert(name, c);
        return c;
    }
}

fn get_simple_extern_fn(cx: block,
                        externs: HashMap<~str, ValueRef>,
                        llmod: ModuleRef,
                        +name: ~str,
                        n_args: int) -> ValueRef {
    let _icx = cx.insn_ctxt("get_simple_extern_fn");
    let ccx = cx.fcx.ccx;
    let inputs = vec::from_elem(n_args as uint, ccx.int_type);
    let output = ccx.int_type;
    let t = T_fn(inputs, output);
    return get_extern_fn(externs, llmod, name, lib::llvm::CCallConv, t);
}

fn trans_foreign_call(cx: block, externs: HashMap<~str, ValueRef>,
                      llmod: ModuleRef, +name: ~str, args: ~[ValueRef]) ->
   ValueRef {
    let _icx = cx.insn_ctxt("trans_foreign_call");
    let n = args.len() as int;
    let llforeign: ValueRef =
        get_simple_extern_fn(cx, externs, llmod, name, n);
    return Call(cx, llforeign, args);
}

fn umax(cx: block, a: ValueRef, b: ValueRef) -> ValueRef {
    let _icx = cx.insn_ctxt("umax");
    let cond = ICmp(cx, lib::llvm::IntULT, a, b);
    return Select(cx, cond, b, a);
}

fn umin(cx: block, a: ValueRef, b: ValueRef) -> ValueRef {
    let _icx = cx.insn_ctxt("umin");
    let cond = ICmp(cx, lib::llvm::IntULT, a, b);
    return Select(cx, cond, a, b);
}

// Given a pointer p, returns a pointer sz(p) (i.e., inc'd by sz bytes).
// The type of the returned pointer is always i8*.  If you care about the
// return type, use bump_ptr().
fn ptr_offs(bcx: block, base: ValueRef, sz: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("ptr_offs");
    let raw = PointerCast(bcx, base, T_ptr(T_i8()));
    InBoundsGEP(bcx, raw, ~[sz])
}

// Increment a pointer by a given amount and then cast it to be a pointer
// to a given type.
fn bump_ptr(bcx: block, t: ty::t, base: ValueRef, sz: ValueRef) ->
   ValueRef {
    let _icx = bcx.insn_ctxt("bump_ptr");
    let ccx = bcx.ccx();
    let bumped = ptr_offs(bcx, base, sz);
    let typ = T_ptr(type_of(ccx, t));
    PointerCast(bcx, bumped, typ)
}

// Replacement for the LLVM 'GEP' instruction when field indexing into a enum.
// @llblobptr is the data part of a enum value; its actual type
// is meaningless, as it will be cast away.
fn GEP_enum(bcx: block, llblobptr: ValueRef, enum_id: ast::def_id,
            variant_id: ast::def_id, ty_substs: ~[ty::t],
            ix: uint) -> ValueRef {
    let _icx = bcx.insn_ctxt("GEP_enum");
    let ccx = bcx.ccx();
    let variant = ty::enum_variant_with_id(ccx.tcx, enum_id, variant_id);
    assert ix < variant.args.len();

    let arg_lltys = vec::map(variant.args, |aty| {
        type_of(ccx, ty::subst_tps(ccx.tcx, ty_substs, None, *aty))
    });
    let typed_blobptr = PointerCast(bcx, llblobptr,
                                    T_ptr(T_struct(arg_lltys)));
    GEPi(bcx, typed_blobptr, [0u, ix])
}

// Returns a pointer to the body for the box. The box may be an opaque
// box. The result will be casted to the type of body_t, if it is statically
// known.
//
// The runtime equivalent is box_body() in "rust_internal.h".
fn opaque_box_body(bcx: block,
                   body_t: ty::t,
                   boxptr: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("opaque_box_body");
    let ccx = bcx.ccx();
    let boxptr = PointerCast(bcx, boxptr, T_ptr(T_box_header(ccx)));
    let bodyptr = GEPi(bcx, boxptr, [1u]);
    PointerCast(bcx, bodyptr, T_ptr(type_of(ccx, body_t)))
}

// malloc_raw_dyn: allocates a box to contain a given type, but with a
// potentially dynamic size.
fn malloc_raw_dyn(bcx: block,
                  t: ty::t,
                  heap: heap,
                  size: ValueRef) -> Result {
    let _icx = bcx.insn_ctxt("malloc_raw");
    let ccx = bcx.ccx();

    let (mk_fn, langcall) = match heap {
        heap_shared => {
            (ty::mk_imm_box, bcx.tcx().lang_items.malloc_fn())
        }
        heap_exchange => {
            (ty::mk_imm_uniq, bcx.tcx().lang_items.exchange_malloc_fn())
        }
    };

    // Grab the TypeRef type of box_ptr_ty.
    let box_ptr_ty = mk_fn(bcx.tcx(), t);
    let llty = type_of(ccx, box_ptr_ty);

    // Get the tydesc for the body:
    let static_ti = get_tydesc(ccx, t);
    glue::lazily_emit_all_tydesc_glue(ccx, static_ti);

    // Allocate space:
    let tydesc = PointerCast(bcx, static_ti.tydesc, T_ptr(T_i8()));
    let rval = alloca_zeroed(bcx, T_ptr(T_i8()));
    let bcx = callee::trans_rtcall_or_lang_call(
        bcx,
        langcall,
        ~[tydesc, size],
        expr::SaveIn(rval));
    return rslt(bcx, PointerCast(bcx, Load(bcx, rval), llty));
}

/**
* Get the type of a box in the default address space.
*
* Shared box pointers live in address space 1 so the GC strategy can find
* them. Before taking a pointer to the inside of a box it should be cast into
* address space 0. Otherwise the resulting (non-box) pointer will be in the
* wrong address space and thus be the wrong type.
*/
fn non_gc_box_cast(bcx: block, val: ValueRef) -> ValueRef {
    unsafe {
        debug!("non_gc_box_cast");
        add_comment(bcx, ~"non_gc_box_cast");
        assert(llvm::LLVMGetPointerAddressSpace(val_ty(val)) ==
                gc_box_addrspace || bcx.unreachable);
        let non_gc_t = T_ptr(llvm::LLVMGetElementType(val_ty(val)));
        PointerCast(bcx, val, non_gc_t)
    }
}

// malloc_raw: expects an unboxed type and returns a pointer to
// enough space for a box of that type.  This includes a rust_opaque_box
// header.
fn malloc_raw(bcx: block, t: ty::t, heap: heap) -> Result {
    malloc_raw_dyn(bcx, t, heap, llsize_of(bcx.ccx(), type_of(bcx.ccx(), t)))
}

// malloc_general_dyn: usefully wraps malloc_raw_dyn; allocates a box,
// and pulls out the body
fn malloc_general_dyn(bcx: block, t: ty::t, heap: heap, size: ValueRef)
    -> {bcx: block, box: ValueRef, body: ValueRef} {
    let _icx = bcx.insn_ctxt("malloc_general");
    let Result {bcx: bcx, val: llbox} = malloc_raw_dyn(bcx, t, heap, size);
    let non_gc_box = non_gc_box_cast(bcx, llbox);
    let body = GEPi(bcx, non_gc_box, [0u, abi::box_field_body]);
    return {bcx: bcx, box: llbox, body: body};
}

fn malloc_general(bcx: block, t: ty::t, heap: heap)
    -> {bcx: block, box: ValueRef, body: ValueRef} {
    malloc_general_dyn(bcx, t, heap,
                       llsize_of(bcx.ccx(), type_of(bcx.ccx(), t)))
}
fn malloc_boxed(bcx: block, t: ty::t)
    -> {bcx: block, box: ValueRef, body: ValueRef} {
    malloc_general(bcx, t, heap_shared)
}
fn malloc_unique(bcx: block, t: ty::t)
    -> {bcx: block, box: ValueRef, body: ValueRef} {
    malloc_general(bcx, t, heap_exchange)
}

// Type descriptor and type glue stuff

fn get_tydesc_simple(ccx: @crate_ctxt, t: ty::t) -> ValueRef {
    get_tydesc(ccx, t).tydesc
}

fn get_tydesc(ccx: @crate_ctxt, t: ty::t) -> @tydesc_info {
    match ccx.tydescs.find(t) {
      Some(inf) => inf,
      _ => {
        ccx.stats.n_static_tydescs += 1u;
        let inf = glue::declare_tydesc(ccx, t);
        ccx.tydescs.insert(t, inf);
        inf
      }
    }
}

fn set_no_inline(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttr(f,
                                  lib::llvm::NoInlineAttribute as c_ulonglong,
                                  0u as c_ulonglong);
    }
}

fn set_no_unwind(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttr(f,
                                  lib::llvm::NoUnwindAttribute as c_ulonglong,
                                  0u as c_ulonglong);
    }
}

// Tell LLVM to emit the information necessary to unwind the stack for the
// function f.
fn set_uwtable(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttr(f,
                                  lib::llvm::UWTableAttribute as c_ulonglong,
                                  0u as c_ulonglong);
    }
}

fn set_inline_hint(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttr(f, lib::llvm::InlineHintAttribute
                                  as c_ulonglong, 0u as c_ulonglong);
    }
}

fn set_inline_hint_if_appr(attrs: ~[ast::attribute],
                           llfn: ValueRef) {
    match attr::find_inline_attr(attrs) {
      attr::ia_hint => set_inline_hint(llfn),
      attr::ia_always => set_always_inline(llfn),
      attr::ia_never => set_no_inline(llfn),
      attr::ia_none => { /* fallthrough */ }
    }
}

fn set_always_inline(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttr(f, lib::llvm::AlwaysInlineAttribute
                                  as c_ulonglong, 0u as c_ulonglong);
    }
}

fn set_custom_stack_growth_fn(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttr(f, 0u as c_ulonglong, 1u as c_ulonglong);
    }
}

fn set_glue_inlining(f: ValueRef, t: ty::t) {
    if ty::type_is_structural(t) {
        set_no_inline(f);
    } else { set_always_inline(f); }
}

// Double-check that we never ask LLVM to declare the same symbol twice. It
// silently mangles such symbols, breaking our linkage model.
fn note_unique_llvm_symbol(ccx: @crate_ctxt, +sym: ~str) {
    // XXX: Bad copy.
    if ccx.all_llvm_symbols.contains_key(copy sym) {
        ccx.sess.bug(~"duplicate LLVM symbol: " + sym);
    }
    ccx.all_llvm_symbols.insert(sym, ());
}


fn get_res_dtor(ccx: @crate_ctxt, did: ast::def_id,
                parent_id: ast::def_id, substs: ~[ty::t])
   -> ValueRef {
    let _icx = ccx.insn_ctxt("trans_res_dtor");
    if (substs.is_not_empty()) {
        let did = if did.crate != ast::local_crate {
            inline::maybe_instantiate_inline(ccx, did, true)
        } else { did };
        assert did.crate == ast::local_crate;
        monomorphize::monomorphic_fn(ccx, did, substs, None, None, None).val
    } else if did.crate == ast::local_crate {
        get_item_val(ccx, did.node)
    } else {
        let tcx = ccx.tcx;
        let name = csearch::get_symbol(ccx.sess.cstore, did);
        let class_ty = ty::subst_tps(tcx, substs, None,
                          ty::lookup_item_type(tcx, parent_id).ty);
        let llty = type_of_dtor(ccx, class_ty);
        get_extern_fn(ccx.externs, ccx.llmod, name, lib::llvm::CCallConv,
                      llty)
    }
}

// Structural comparison: a rather involved form of glue.
fn maybe_name_value(cx: @crate_ctxt, v: ValueRef, s: ~str) {
    if cx.sess.opts.save_temps {
        let _: () = str::as_c_str(s, |buf| {
            unsafe {
                llvm::LLVMSetValueName(v, buf)
            }
        });
    }
}


// Used only for creating scalar comparison glue.
enum scalar_type { nil_type, signed_int, unsigned_int, floating_point, }

fn compare_scalar_types(cx: block, lhs: ValueRef, rhs: ValueRef,
                        t: ty::t, op: ast::binop) -> Result {
    let f = |a| compare_scalar_values(cx, lhs, rhs, a, op);

    match ty::get(t).sty {
        ty::ty_nil => rslt(cx, f(nil_type)),
        ty::ty_bool | ty::ty_ptr(_) => rslt(cx, f(unsigned_int)),
        ty::ty_int(_) => rslt(cx, f(signed_int)),
        ty::ty_uint(_) => rslt(cx, f(unsigned_int)),
        ty::ty_float(_) => rslt(cx, f(floating_point)),
        ty::ty_type => {
            rslt(
                controlflow::trans_fail(
                    cx, None,
                    ~"attempt to compare values of type type"),
                C_nil())
        }
        _ => {
            // Should never get here, because t is scalar.
            cx.sess().bug(~"non-scalar type passed to \
                            compare_scalar_types")
        }
    }
}


// A helper function to do the actual comparison of scalar values.
fn compare_scalar_values(cx: block, lhs: ValueRef, rhs: ValueRef,
                         nt: scalar_type, op: ast::binop) -> ValueRef {
    let _icx = cx.insn_ctxt("compare_scalar_values");
    fn die(cx: block) -> ! {
        cx.tcx().sess.bug(~"compare_scalar_values: must be a\
          comparison operator");
    }
    match nt {
      nil_type => {
        // We don't need to do actual comparisons for nil.
        // () == () holds but () < () does not.
        match op {
          ast::eq | ast::le | ast::ge => return C_bool(true),
          ast::ne | ast::lt | ast::gt => return C_bool(false),
          // refinements would be nice
          _ => die(cx)
        }
      }
      floating_point => {
        let cmp = match op {
          ast::eq => lib::llvm::RealOEQ,
          ast::ne => lib::llvm::RealUNE,
          ast::lt => lib::llvm::RealOLT,
          ast::le => lib::llvm::RealOLE,
          ast::gt => lib::llvm::RealOGT,
          ast::ge => lib::llvm::RealOGE,
          _ => die(cx)
        };
        return FCmp(cx, cmp, lhs, rhs);
      }
      signed_int => {
        let cmp = match op {
          ast::eq => lib::llvm::IntEQ,
          ast::ne => lib::llvm::IntNE,
          ast::lt => lib::llvm::IntSLT,
          ast::le => lib::llvm::IntSLE,
          ast::gt => lib::llvm::IntSGT,
          ast::ge => lib::llvm::IntSGE,
          _ => die(cx)
        };
        return ICmp(cx, cmp, lhs, rhs);
      }
      unsigned_int => {
        let cmp = match op {
          ast::eq => lib::llvm::IntEQ,
          ast::ne => lib::llvm::IntNE,
          ast::lt => lib::llvm::IntULT,
          ast::le => lib::llvm::IntULE,
          ast::gt => lib::llvm::IntUGT,
          ast::ge => lib::llvm::IntUGE,
          _ => die(cx)
        };
        return ICmp(cx, cmp, lhs, rhs);
      }
    }
}

type val_pair_fn = fn@(block, ValueRef, ValueRef) -> block;
type val_and_ty_fn = fn@(block, ValueRef, ty::t) -> block;

fn load_inbounds(cx: block, p: ValueRef, idxs: &[uint]) -> ValueRef {
    return Load(cx, GEPi(cx, p, idxs));
}

fn store_inbounds(cx: block, v: ValueRef, p: ValueRef, idxs: &[uint]) {
    Store(cx, v, GEPi(cx, p, idxs));
}

// Iterates through the elements of a structural type.
fn iter_structural_ty(cx: block, av: ValueRef, t: ty::t,
                      f: val_and_ty_fn) -> block {
    let _icx = cx.insn_ctxt("iter_structural_ty");

    fn iter_variant(cx: block, a_tup: ValueRef,
                    variant: ty::VariantInfo,
                    tps: ~[ty::t], tid: ast::def_id,
                    f: val_and_ty_fn) -> block {
        let _icx = cx.insn_ctxt("iter_variant");
        if variant.args.len() == 0u { return cx; }
        let fn_ty = variant.ctor_ty;
        let ccx = cx.ccx();
        let mut cx = cx;
        match ty::get(fn_ty).sty {
          ty::ty_fn(ref fn_ty) => {
            let mut j = 0u;
            let v_id = variant.id;
            for vec::each(fn_ty.sig.inputs) |a| {
                let llfldp_a = GEP_enum(cx, a_tup, tid, v_id,
                                        /*bad*/copy tps, j);
                // XXX: Is "None" right here?
                let ty_subst = ty::subst_tps(ccx.tcx, tps, None, a.ty);
                cx = f(cx, llfldp_a, ty_subst);
                j += 1u;
            }
          }
          _ => cx.tcx().sess.bug(fmt!("iter_variant: not a function type: \
                                       %s (variant name = %s)",
                                      cx.ty_to_str(fn_ty),
                                      cx.sess().str_of(variant.name)))
        }
        return cx;
    }

    let mut cx = cx;
    match /*bad*/copy ty::get(t).sty {
      ty::ty_rec(*) | ty::ty_struct(*) => {
          do expr::with_field_tys(cx.tcx(), t, None) |_has_dtor, field_tys| {
              for vec::eachi(field_tys) |i, field_ty| {
                  let llfld_a = GEPi(cx, av, struct_field(i));
                  cx = f(cx, llfld_a, field_ty.mt.ty);
              }
          }
      }
      ty::ty_estr(ty::vstore_fixed(_)) |
      ty::ty_evec(_, ty::vstore_fixed(_)) => {
        let (base, len) = tvec::get_base_and_len(cx, av, t);
        cx = tvec::iter_vec_raw(cx, base, t, len, f);
      }
      ty::ty_tup(args) => {
        for vec::eachi(args) |i, arg| {
            let llfld_a = GEPi(cx, av, [0u, i]);
            cx = f(cx, llfld_a, *arg);
        }
      }
      ty::ty_enum(tid, ref substs) => {
        let variants = ty::enum_variants(cx.tcx(), tid);
        let n_variants = (*variants).len();

        // Cast the enums to types we can GEP into.
        if n_variants == 1u {
            return iter_variant(cx,
                                av,
                                variants[0],
                                /*bad*/copy substs.tps,
                                tid,
                                f);
        }

        let ccx = cx.ccx();
        let llenumty = T_opaque_enum_ptr(ccx);
        let av_enum = PointerCast(cx, av, llenumty);
        let lldiscrim_a_ptr = GEPi(cx, av_enum, [0u, 0u]);
        let llunion_a_ptr = GEPi(cx, av_enum, [0u, 1u]);
        let lldiscrim_a = Load(cx, lldiscrim_a_ptr);

        // NB: we must hit the discriminant first so that structural
        // comparison know not to proceed when the discriminants differ.
        cx = f(cx, lldiscrim_a_ptr, ty::mk_int(cx.tcx()));
        let unr_cx = sub_block(cx, ~"enum-iter-unr");
        Unreachable(unr_cx);
        let llswitch = Switch(cx, lldiscrim_a, unr_cx.llbb, n_variants);
        let next_cx = sub_block(cx, ~"enum-iter-next");
        for vec::each(*variants) |variant| {
            let variant_cx =
                sub_block(cx,
                                   ~"enum-iter-variant-" +
                                       int::to_str(variant.disr_val, 10u));
            AddCase(llswitch, C_int(ccx, variant.disr_val), variant_cx.llbb);
            let variant_cx =
                iter_variant(variant_cx, llunion_a_ptr, *variant,
                             /*bad*/copy (*substs).tps, tid, f);
            Br(variant_cx, next_cx.llbb);
        }
        return next_cx;
      }
      _ => cx.sess().unimpl(~"type in iter_structural_ty")
    }
    return cx;
}

fn cast_shift_expr_rhs(cx: block, op: ast::binop,
                       lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs,
                   |a,b| Trunc(cx, a, b),
                   |a,b| ZExt(cx, a, b))
}

fn cast_shift_const_rhs(op: ast::binop,
                        lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    unsafe {
        cast_shift_rhs(op, lhs, rhs,
                       |a, b| unsafe { llvm::LLVMConstTrunc(a, b) },
                       |a, b| unsafe { llvm::LLVMConstZExt(a, b) })
    }
}

fn cast_shift_rhs(op: ast::binop,
                  lhs: ValueRef, rhs: ValueRef,
                  trunc: fn(ValueRef, TypeRef) -> ValueRef,
                  zext: fn(ValueRef, TypeRef) -> ValueRef
                 ) -> ValueRef {
    // Shifts may have any size int on the rhs
    unsafe {
        if ast_util::is_shift_binop(op) {
            let rhs_llty = val_ty(rhs);
            let lhs_llty = val_ty(lhs);
            let rhs_sz = llvm::LLVMGetIntTypeWidth(rhs_llty);
            let lhs_sz = llvm::LLVMGetIntTypeWidth(lhs_llty);
            if lhs_sz < rhs_sz {
                trunc(rhs, lhs_llty)
            } else if lhs_sz > rhs_sz {
                // FIXME (#1877: If shifting by negative
                // values becomes not undefined then this is wrong.
                zext(rhs, lhs_llty)
            } else {
                rhs
            }
        } else {
            rhs
        }
    }
}

fn fail_if_zero(cx: block, span: span, divmod: ast::binop,
                rhs: ValueRef, rhs_t: ty::t) -> block {
    let text = if divmod == ast::div {
        ~"divide by zero"
    } else {
        ~"modulo zero"
    };
    let is_zero = match ty::get(rhs_t).sty {
      ty::ty_int(t) => {
        let zero = C_integral(T_int_ty(cx.ccx(), t), 0u64, False);
        ICmp(cx, lib::llvm::IntEQ, rhs, zero)
      }
      ty::ty_uint(t) => {
        let zero = C_integral(T_uint_ty(cx.ccx(), t), 0u64, False);
        ICmp(cx, lib::llvm::IntEQ, rhs, zero)
      }
      _ => {
        cx.tcx().sess.bug(~"fail-if-zero on unexpected type: " +
                          ty_to_str(cx.ccx().tcx, rhs_t));
      }
    };
    do with_cond(cx, is_zero) |bcx| {
        controlflow::trans_fail(bcx, Some(span), /*bad*/copy text)
    }
}

fn null_env_ptr(bcx: block) -> ValueRef {
    C_null(T_opaque_box_ptr(bcx.ccx()))
}

fn trans_external_path(ccx: @crate_ctxt, did: ast::def_id, t: ty::t)
    -> ValueRef {
    let name = csearch::get_symbol(ccx.sess.cstore, did);
    match ty::get(t).sty {
      ty::ty_fn(_) => {
        let llty = type_of_fn_from_ty(ccx, t);
        return get_extern_fn(ccx.externs, ccx.llmod, name,
                          lib::llvm::CCallConv, llty);
      }
      _ => {
        let llty = type_of(ccx, t);
        return get_extern_const(ccx.externs, ccx.llmod, name, llty);
      }
    };
}

fn get_discrim_val(cx: @crate_ctxt, span: span, enum_did: ast::def_id,
                   variant_did: ast::def_id) -> ValueRef {
    // Can't use `discrims` from the crate context here because
    // those discriminants have an extra level of indirection,
    // and there's no LLVM constant load instruction.
    let mut lldiscrim_opt = None;
    for ty::enum_variants(cx.tcx, enum_did).each |variant_info| {
        if variant_info.id == variant_did {
            lldiscrim_opt = Some(C_int(cx,
                                       variant_info.disr_val));
            break;
        }
    }

    match lldiscrim_opt {
        None => {
            cx.tcx.sess.span_bug(span, ~"didn't find discriminant?!");
        }
        Some(found_lldiscrim) => {
            found_lldiscrim
        }
    }
}

fn lookup_discriminant(ccx: @crate_ctxt, vid: ast::def_id) -> ValueRef {
    unsafe {
        let _icx = ccx.insn_ctxt("lookup_discriminant");
        match ccx.discrims.find(vid) {
            None => {
                // It's an external discriminant that we haven't seen yet.
                assert (vid.crate != ast::local_crate);
                let sym = csearch::get_symbol(ccx.sess.cstore, vid);
                let gvar = str::as_c_str(sym, |buf| {
                    llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
                });
                lib::llvm::SetLinkage(gvar, lib::llvm::ExternalLinkage);
                llvm::LLVMSetGlobalConstant(gvar, True);
                ccx.discrims.insert(vid, gvar);
                return gvar;
            }
            Some(llval) => return llval,
        }
    }
}

fn invoke(bcx: block, llfn: ValueRef, +llargs: ~[ValueRef]) -> block {
    let _icx = bcx.insn_ctxt("invoke_");
    if bcx.unreachable { return bcx; }
    if need_invoke(bcx) {
        log(debug, ~"invoking");
        let normal_bcx = sub_block(bcx, ~"normal return");
        Invoke(bcx, llfn, llargs, normal_bcx.llbb, get_landing_pad(bcx));
        return normal_bcx;
    } else {
        log(debug, ~"calling");
        Call(bcx, llfn, llargs);
        return bcx;
    }
}

fn need_invoke(bcx: block) -> bool {
    if (bcx.ccx().sess.opts.debugging_opts & session::no_landing_pads != 0) {
        return false;
    }

    // Avoid using invoke if we are already inside a landing pad.
    if bcx.is_lpad {
        return false;
    }

    if have_cached_lpad(bcx) {
        return true;
    }

    // Walk the scopes to look for cleanups
    let mut cur = bcx;
    loop {
        match cur.kind {
          block_scope(ref inf) => {
            for vec::each((*inf).cleanups) |cleanup| {
                match *cleanup {
                  clean(_, cleanup_type) | clean_temp(_, _, cleanup_type) => {
                    if cleanup_type == normal_exit_and_unwind {
                        return true;
                    }
                  }
                }
            }
          }
          _ => ()
        }
        cur = match cur.parent {
          Some(next) => next,
          None => return false
        }
    }
}

fn have_cached_lpad(bcx: block) -> bool {
    let mut res = false;
    do in_lpad_scope_cx(bcx) |inf| {
        match inf.landing_pad {
          Some(_) => res = true,
          None => res = false
        }
    }
    return res;
}

fn in_lpad_scope_cx(bcx: block, f: fn(scope_info)) {
    let mut bcx = bcx;
    loop {
        match bcx.kind {
          block_scope(ref inf) => {
            if (*inf).cleanups.len() > 0u || bcx.parent.is_none() {
                f((*inf)); return;
            }
          }
          _ => ()
        }
        bcx = block_parent(bcx);
    }
}

fn get_landing_pad(bcx: block) -> BasicBlockRef {
    let _icx = bcx.insn_ctxt("get_landing_pad");

    let mut cached = None, pad_bcx = bcx; // Guaranteed to be set below
    do in_lpad_scope_cx(bcx) |inf| {
        // If there is a valid landing pad still around, use it
        match copy inf.landing_pad {
          Some(target) => cached = Some(target),
          None => {
            pad_bcx = lpad_block(bcx, ~"unwind");
            inf.landing_pad = Some(pad_bcx.llbb);
          }
        }
    }
    // Can't return from block above
    match cached { Some(b) => return b, None => () }
    // The landing pad return type (the type being propagated). Not sure what
    // this represents but it's determined by the personality function and
    // this is what the EH proposal example uses.
    let llretty = T_struct(~[T_ptr(T_i8()), T_i32()]);
    // The exception handling personality function. This is the C++
    // personality function __gxx_personality_v0, wrapped in our naming
    // convention.
    let personality = bcx.ccx().upcalls.rust_personality;
    // The only landing pad clause will be 'cleanup'
    let llretval = LandingPad(pad_bcx, llretty, personality, 1u);
    // The landing pad block is a cleanup
    SetCleanup(pad_bcx, llretval);

    // Because we may have unwound across a stack boundary, we must call into
    // the runtime to figure out which stack segment we are on and place the
    // stack limit back into the TLS.
    Call(pad_bcx, bcx.ccx().upcalls.reset_stack_limit, ~[]);

    // We store the retval in a function-central alloca, so that calls to
    // Resume can find it.
    match copy bcx.fcx.personality {
      Some(addr) => Store(pad_bcx, llretval, addr),
      None => {
        let addr = alloca(pad_bcx, val_ty(llretval));
        bcx.fcx.personality = Some(addr);
        Store(pad_bcx, llretval, addr);
      }
    }

    // Unwind all parent scopes, and finish with a Resume instr
    cleanup_and_leave(pad_bcx, None, None);
    return pad_bcx.llbb;
}

// Arranges for the value found in `*root_loc` to be dropped once the scope
// associated with `scope_id` exits.  This is used to keep boxes live when
// there are extant region pointers pointing at the interior.
//
// Note that `root_loc` is not the value itself but rather a pointer to the
// value.  Generally it in alloca'd value.  The reason for this is that the
// value is initialized in an inner block but may be freed in some outer
// block, so an SSA value that is valid in the inner block may not be valid in
// the outer block.  In fact, the inner block may not even execute.  Rather
// than generate the full SSA form, we just use an alloca'd value.
fn add_root_cleanup(bcx: block,
                    root_info: RootInfo,
                    root_loc: ValueRef,
                    ty: ty::t) {

    debug!("add_root_cleanup(bcx=%s, \
                             scope=%d, \
                             freezes=%?, \
                             root_loc=%s, \
                             ty=%s)",
           bcx.to_str(),
           root_info.scope,
           root_info.freezes,
           val_str(bcx.ccx().tn, root_loc),
           ppaux::ty_to_str(bcx.ccx().tcx, ty));

    let bcx_scope = find_bcx_for_scope(bcx, root_info.scope);
    if root_info.freezes {
        add_clean_frozen_root(bcx_scope, root_loc, ty);
    } else {
        add_clean_temp_mem(bcx_scope, root_loc, ty);
    }

    fn find_bcx_for_scope(bcx: block, scope_id: ast::node_id) -> block {
        let mut bcx_sid = bcx;
        loop {
            bcx_sid = match bcx_sid.node_info {
              Some({id, _}) if id == scope_id => {
                return bcx_sid
              }
              _ => {
                match bcx_sid.parent {
                  None => bcx.tcx().sess.bug(
                      fmt!("no enclosing scope with id %d", scope_id)),
                  Some(bcx_par) => bcx_par
                }
              }
            }
        }
    }
}

fn do_spill(bcx: block, v: ValueRef, t: ty::t) -> ValueRef {
    if ty::type_is_bot(t) {
        return C_null(T_ptr(T_i8()));
    }
    let llptr = alloc_ty(bcx, t);
    Store(bcx, v, llptr);
    return llptr;
}

// Since this function does *not* root, it is the caller's responsibility to
// ensure that the referent is pointed to by a root.
// [Note-arg-mode]
// ++ mode is temporary, due to how borrowck treats enums. With hope,
// will go away anyway when we get rid of modes.
fn do_spill_noroot(++cx: block, v: ValueRef) -> ValueRef {
    let llptr = alloca(cx, val_ty(v));
    Store(cx, v, llptr);
    return llptr;
}

fn spill_if_immediate(cx: block, v: ValueRef, t: ty::t) -> ValueRef {
    let _icx = cx.insn_ctxt("spill_if_immediate");
    if ty::type_is_immediate(t) { return do_spill(cx, v, t); }
    return v;
}

fn load_if_immediate(cx: block, v: ValueRef, t: ty::t) -> ValueRef {
    let _icx = cx.insn_ctxt("load_if_immediate");
    if ty::type_is_immediate(t) { return Load(cx, v); }
    return v;
}

fn trans_trace(bcx: block, sp_opt: Option<span>, +trace_str: ~str) {
    if !bcx.sess().trace() { return; }
    let _icx = bcx.insn_ctxt("trans_trace");
    // XXX: Bad copy.
    add_comment(bcx, copy trace_str);
    let V_trace_str = C_cstr(bcx.ccx(), trace_str);
    let {V_filename, V_line} = match sp_opt {
      Some(sp) => {
        let sess = bcx.sess();
        let loc = sess.parse_sess.cm.lookup_char_pos(sp.lo);
        {V_filename: C_cstr(bcx.ccx(), /*bad*/copy loc.file.name),
         V_line: loc.line as int}
      }
      None => {
        {V_filename: C_cstr(bcx.ccx(), ~"<runtime>"),
         V_line: 0}
      }
    };
    let ccx = bcx.ccx();
    let V_trace_str = PointerCast(bcx, V_trace_str, T_ptr(T_i8()));
    let V_filename = PointerCast(bcx, V_filename, T_ptr(T_i8()));
    let args = ~[V_trace_str, V_filename, C_int(ccx, V_line)];
    Call(bcx, ccx.upcalls.trace, args);
}

fn build_return(bcx: block) {
    let _icx = bcx.insn_ctxt("build_return");
    Br(bcx, bcx.fcx.llreturn);
}

fn ignore_lhs(_bcx: block, local: @ast::local) -> bool {
    match local.node.pat.node {
        ast::pat_wild => true, _ => false
    }
}

fn init_local(bcx: block, local: @ast::local) -> block {

    debug!("init_local(bcx=%s, local.id=%?)",
           bcx.to_str(), local.node.id);
    let _indenter = indenter();

    let _icx = bcx.insn_ctxt("init_local");
    let ty = node_id_type(bcx, local.node.id);

    debug!("ty=%s", bcx.ty_to_str(ty));

    if ignore_lhs(bcx, local) {
        // Handle let _ = e; just like e;
        match local.node.init {
            Some(init) => {
              return expr::trans_into(bcx, init, expr::Ignore);
            }
            None => { return bcx; }
        }
    }

    let llptr = match bcx.fcx.lllocals.find(local.node.id) {
      Some(local_mem(v)) => v,
      _ => { bcx.tcx().sess.span_bug(local.span,
                        ~"init_local: Someone forgot to document why it's\
                         safe to assume local.node.init must be local_mem!");
        }
    };

    let mut bcx = bcx;
    match local.node.init {
        Some(init) => {
            bcx = expr::trans_into(bcx, init, expr::SaveIn(llptr));
        }
        _ => {
            zero_mem(bcx, llptr, ty);
        }
    }

    // Make a note to drop this slot on the way out.
    debug!("adding clean for %?/%s to bcx=%s",
           local.node.id, bcx.ty_to_str(ty),
           bcx.to_str());
    add_clean(bcx, llptr, ty);

    return _match::bind_irrefutable_pat(bcx,
                                       local.node.pat,
                                       llptr,
                                       false,
                                       _match::BindLocal);
}

fn trans_stmt(cx: block, s: ast::stmt) -> block {
    let _icx = cx.insn_ctxt("trans_stmt");
    debug!("trans_stmt(%s)", stmt_to_str(s, cx.tcx().sess.intr()));

    if !cx.sess().no_asm_comments() {
        add_span_comment(cx, s.span, stmt_to_str(s, cx.ccx().sess.intr()));
    }

    let mut bcx = cx;
    debuginfo::update_source_pos(cx, s.span);

    match s.node {
        ast::stmt_expr(e, _) | ast::stmt_semi(e, _) => {
            bcx = expr::trans_into(cx, e, expr::Ignore);
        }
        ast::stmt_decl(d, _) => {
            match /*bad*/copy d.node {
                ast::decl_local(locals) => {
                    for vec::each(locals) |local| {
                        bcx = init_local(bcx, *local);
                        if cx.sess().opts.extra_debuginfo {
                            debuginfo::create_local_var(bcx, *local);
                        }
                    }
                }
                ast::decl_item(i) => trans_item(cx.fcx.ccx, *i)
            }
        }
        ast::stmt_mac(*) => cx.tcx().sess.bug(~"unexpanded macro")
    }

    return bcx;
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block(cx: fn_ctxt, parent: Option<block>, +kind: block_kind,
             is_lpad: bool, +name: ~str, opt_node_info: Option<node_info>)
    -> block {

    let s = if cx.ccx.sess.opts.save_temps || cx.ccx.sess.opts.debuginfo {
        (cx.ccx.names)(name)
    } else {
        special_idents::invalid
    };
    unsafe {
        let llbb: BasicBlockRef = str::as_c_str(cx.ccx.sess.str_of(s), |buf| {
            llvm::LLVMAppendBasicBlock(cx.llfn, buf)
        });
        let bcx = mk_block(llbb,
                           parent,
                           move kind,
                           is_lpad,
                           opt_node_info,
                           cx);
        do option::iter(&parent) |cx| {
            if cx.unreachable { Unreachable(bcx); }
        };
        return bcx;
    }
}

fn simple_block_scope() -> block_kind {
    block_scope(scope_info {
        loop_break: None,
        loop_label: None,
        mut cleanups: ~[],
        mut cleanup_paths: ~[],
        mut landing_pad: None
    })
}

// Use this when you're at the top block of a function or the like.
fn top_scope_block(fcx: fn_ctxt, opt_node_info: Option<node_info>) -> block {
    return new_block(fcx, None, simple_block_scope(), false,
                  ~"function top level", opt_node_info);
}

fn scope_block(bcx: block,
               opt_node_info: Option<node_info>,
               +n: ~str) -> block {
    return new_block(bcx.fcx, Some(bcx), simple_block_scope(), bcx.is_lpad,
                  n, opt_node_info);
}

fn loop_scope_block(bcx: block, loop_break: block, loop_label: Option<ident>,
                    +n: ~str, opt_node_info: Option<node_info>) -> block {
    return new_block(bcx.fcx, Some(bcx), block_scope(scope_info {
        loop_break: Some(loop_break),
        loop_label: loop_label,
        mut cleanups: ~[],
        mut cleanup_paths: ~[],
        mut landing_pad: None
    }), bcx.is_lpad, n, opt_node_info);
}

// Use this when creating a block for the inside of a landing pad.
fn lpad_block(bcx: block, +n: ~str) -> block {
    new_block(bcx.fcx, Some(bcx), block_non_scope, true, n, None)
}

// Use this when you're making a general CFG BB within a scope.
fn sub_block(bcx: block, +n: ~str) -> block {
    new_block(bcx.fcx, Some(bcx), block_non_scope, bcx.is_lpad, n, None)
}

fn raw_block(fcx: fn_ctxt, is_lpad: bool, llbb: BasicBlockRef) -> block {
    mk_block(llbb, None, block_non_scope, is_lpad, None, fcx)
}


// trans_block_cleanups: Go through all the cleanups attached to this
// block and execute them.
//
// When translating a block that introduces new variables during its scope, we
// need to make sure those variables go out of scope when the block ends.  We
// do that by running a 'cleanup' function for each variable.
// trans_block_cleanups runs all the cleanup functions for the block.
fn trans_block_cleanups(bcx: block, +cleanups: ~[cleanup]) -> block {
    trans_block_cleanups_(bcx, cleanups, false)
}

fn trans_block_cleanups_(bcx: block,
                         +cleanups: ~[cleanup],
                         /* cleanup_cx: block, */ is_lpad: bool) ->
   block {
    let _icx = bcx.insn_ctxt("trans_block_cleanups");
    // NB: Don't short-circuit even if this block is unreachable because
    // GC-based cleanup needs to the see that the roots are live.
    let no_lpads =
        bcx.ccx().sess.opts.debugging_opts & session::no_landing_pads != 0;
    if bcx.unreachable && !no_lpads { return bcx; }
    let mut bcx = bcx;
    for vec::rev_each(cleanups) |cu| {
        match *cu {
            clean(cfn, cleanup_type) | clean_temp(_, cfn, cleanup_type) => {
                // Some types don't need to be cleaned up during
                // landing pads because they can be freed en mass later
                if cleanup_type == normal_exit_and_unwind || !is_lpad {
                    bcx = cfn(bcx);
                }
            }
        }
    }
    return bcx;
}

// In the last argument, Some(block) mean jump to this block, and none means
// this is a landing pad and leaving should be accomplished with a resume
// instruction.
fn cleanup_and_leave(bcx: block,
                     upto: Option<BasicBlockRef>,
                     leave: Option<BasicBlockRef>) {
    let _icx = bcx.insn_ctxt("cleanup_and_leave");
    let mut cur = bcx, bcx = bcx;
    let is_lpad = leave == None;
    loop {
        debug!("cleanup_and_leave: leaving %s", cur.to_str());

        if bcx.sess().trace() {
            trans_trace(
                bcx, None,
                fmt!("cleanup_and_leave(%s)", cur.to_str()));
        }

        match cur.kind {
          block_scope(ref inf) if (*inf).cleanups.len() > 0u => {
            for vec::find((*inf).cleanup_paths,
                          |cp| cp.target == leave).each |cp| {
                Br(bcx, cp.dest);
                return;
            }
            let sub_cx = sub_block(bcx, ~"cleanup");
            Br(bcx, sub_cx.llbb);
            (*inf).cleanup_paths.push({target: leave, dest: sub_cx.llbb});
            bcx = trans_block_cleanups_(sub_cx, block_cleanups(cur), is_lpad);
          }
          _ => ()
        }
        match upto {
          Some(bb) => { if cur.llbb == bb { break; } }
          _ => ()
        }
        cur = match cur.parent {
          Some(next) => next,
          None => { assert upto.is_none(); break; }
        };
    }
    match leave {
      Some(target) => Br(bcx, target),
      None => { Resume(bcx, Load(bcx, bcx.fcx.personality.get())); }
    }
}

fn cleanup_and_Br(bcx: block, upto: block,
                  target: BasicBlockRef) {
    let _icx = bcx.insn_ctxt("cleanup_and_Br");
    cleanup_and_leave(bcx, Some(upto.llbb), Some(target));
}

fn leave_block(bcx: block, out_of: block) -> block {
    let _icx = bcx.insn_ctxt("leave_block");
    let next_cx = sub_block(block_parent(out_of), ~"next");
    if bcx.unreachable { Unreachable(next_cx); }
    cleanup_and_Br(bcx, out_of, next_cx.llbb);
    next_cx
}

fn with_scope(bcx: block, opt_node_info: Option<node_info>,
              +name: ~str, f: fn(block) -> block) -> block {
    let _icx = bcx.insn_ctxt("with_scope");

    debug!("with_scope(bcx=%s, opt_node_info=%?, name=%s)",
           bcx.to_str(), opt_node_info, name);
    let _indenter = indenter();

    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    leave_block(f(scope_cx), scope_cx)
}

fn with_scope_result(bcx: block,
                     opt_node_info: Option<node_info>,
                     +name: ~str,
                     f: fn(block) -> Result)
                  -> Result {
    let _icx = bcx.insn_ctxt("with_scope_result");
    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    let Result {bcx, val} = f(scope_cx);
    rslt(leave_block(bcx, scope_cx), val)
}

fn with_scope_datumblock(bcx: block, opt_node_info: Option<node_info>,
                         +name: ~str, f: fn(block) -> datum::DatumBlock)
    -> datum::DatumBlock
{
    use middle::trans::datum::DatumBlock;

    let _icx = bcx.insn_ctxt("with_scope_result");
    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    let DatumBlock {bcx, datum} = f(scope_cx);
    DatumBlock {bcx: leave_block(bcx, scope_cx), datum: datum}
}

fn block_locals(b: ast::blk, it: fn(@ast::local)) {
    for vec::each(b.node.stmts) |s| {
        match s.node {
          ast::stmt_decl(d, _) => {
            match /*bad*/copy d.node {
              ast::decl_local(locals) => {
                for vec::each(locals) |local| {
                    it(*local);
                }
              }
              _ => {/* fall through */ }
            }
          }
          _ => {/* fall through */ }
        }
    }
}

fn alloc_local(cx: block, local: @ast::local) -> block {
    let _icx = cx.insn_ctxt("alloc_local");
    let t = node_id_type(cx, local.node.id);
    let simple_name = match local.node.pat.node {
      ast::pat_ident(_, pth, None) => Some(path_to_ident(pth)),
      _ => None
    };
    let val = alloc_ty(cx, t);
    if cx.sess().opts.debuginfo {
        do option::iter(&simple_name) |name| {
            str::as_c_str(cx.ccx().sess.str_of(*name), |buf| {
                unsafe {
                    llvm::LLVMSetValueName(val, buf)
                }
            });
        }
    }
    cx.fcx.lllocals.insert(local.node.id, local_mem(val));
    return cx;
}


fn with_cond(bcx: block, val: ValueRef, f: fn(block) -> block) -> block {
    let _icx = bcx.insn_ctxt("with_cond");
    let next_cx = base::sub_block(bcx, ~"next");
    let cond_cx = base::sub_block(bcx, ~"cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb);
    let after_cx = f(cond_cx);
    if !after_cx.terminated { Br(after_cx, next_cx.llbb); }
    next_cx
}

fn call_memcpy(cx: block, dst: ValueRef, src: ValueRef,
                n_bytes: ValueRef) {
    // FIXME (Related to #1645, I think?): Provide LLVM with better
    // alignment information when the alignment is statically known (it must
    // be nothing more than a constant int, or LLVM complains -- not even a
    // constant element of a tydesc works).
    let _icx = cx.insn_ctxt("call_memcpy");
    let ccx = cx.ccx();
    let key = match ccx.sess.targ_cfg.arch {
      session::arch_x86 | session::arch_arm => ~"llvm.memcpy.p0i8.p0i8.i32",
      session::arch_x86_64 => ~"llvm.memcpy.p0i8.p0i8.i64"
    };
    let memcpy = ccx.intrinsics.get(key);
    let src_ptr = PointerCast(cx, src, T_ptr(T_i8()));
    let dst_ptr = PointerCast(cx, dst, T_ptr(T_i8()));
    let size = IntCast(cx, n_bytes, ccx.int_type);
    let align = C_i32(1i32);
    let volatile = C_bool(false);
    Call(cx, memcpy, ~[dst_ptr, src_ptr, size, align, volatile]);
}

fn memcpy_ty(bcx: block, dst: ValueRef, src: ValueRef, t: ty::t) {
    let _icx = bcx.insn_ctxt("memcpy_ty");
    let ccx = bcx.ccx();
    if ty::type_is_structural(t) {
        let llsz = llsize_of(ccx, type_of::type_of(ccx, t));
        call_memcpy(bcx, dst, src, llsz);
    } else {
        Store(bcx, Load(bcx, src), dst);
    }
}

fn zero_mem(cx: block, llptr: ValueRef, t: ty::t) {
    let _icx = cx.insn_ctxt("zero_mem");
    let bcx = cx;
    let ccx = cx.ccx();
    let llty = type_of::type_of(ccx, t);
    memzero(bcx, llptr, llty);
}

// Always use this function instead of storing a zero constant to the memory
// in question. If you store a zero constant, LLVM will drown in vreg
// allocation for large data structures, and the generated code will be
// awful. (A telltale sign of this is large quantities of
// `mov [byte ptr foo],0` in the generated code.)
fn memzero(cx: block, llptr: ValueRef, llty: TypeRef) {
    let _icx = cx.insn_ctxt("memzero");
    let ccx = cx.ccx();

    let intrinsic_key;
    match ccx.sess.targ_cfg.arch {
        session::arch_x86 | session::arch_arm => {
            intrinsic_key = ~"llvm.memset.p0i8.i32";
        }
        session::arch_x86_64 => {
            intrinsic_key = ~"llvm.memset.p0i8.i64";
        }
    }

    let llintrinsicfn = ccx.intrinsics.get(intrinsic_key);
    let llptr = PointerCast(cx, llptr, T_ptr(T_i8()));
    let llzeroval = C_u8(0);
    let size = IntCast(cx, shape::llsize_of(ccx, llty), ccx.int_type);
    let align = C_i32(1i32);
    let volatile = C_bool(false);
    Call(cx, llintrinsicfn, ~[llptr, llzeroval, size, align, volatile]);
}

fn alloc_ty(bcx: block, t: ty::t) -> ValueRef {
    let _icx = bcx.insn_ctxt("alloc_ty");
    let ccx = bcx.ccx();
    let llty = type_of::type_of(ccx, t);
    if ty::type_has_params(t) { log(error, ty_to_str(ccx.tcx, t)); }
    assert !ty::type_has_params(t);
    let val = alloca(bcx, llty);
    return val;
}

fn alloca(cx: block, t: TypeRef) -> ValueRef {
    alloca_maybe_zeroed(cx, t, false)
}

fn alloca_zeroed(cx: block, t: TypeRef) -> ValueRef {
    alloca_maybe_zeroed(cx, t, true)
}

fn alloca_maybe_zeroed(cx: block, t: TypeRef, zero: bool) -> ValueRef {
    let _icx = cx.insn_ctxt("alloca");
    if cx.unreachable {
        unsafe {
            return llvm::LLVMGetUndef(t);
        }
    }
    let initcx = base::raw_block(cx.fcx, false, cx.fcx.llstaticallocas);
    let p = Alloca(initcx, t);
    if zero { memzero(initcx, p, t); }
    return p;
}

fn arrayalloca(cx: block, t: TypeRef, v: ValueRef) -> ValueRef {
    let _icx = cx.insn_ctxt("arrayalloca");
    if cx.unreachable {
        unsafe {
            return llvm::LLVMGetUndef(t);
        }
    }
    return ArrayAlloca(
        base::raw_block(cx.fcx, false, cx.fcx.llstaticallocas), t, v);
}

// Creates the standard set of basic blocks for a function
fn mk_standard_basic_blocks(llfn: ValueRef) ->
   {sa: BasicBlockRef, rt: BasicBlockRef} {
    unsafe {
        {sa: str::as_c_str(~"static_allocas",
                           |buf| llvm::LLVMAppendBasicBlock(llfn, buf)),
         rt: str::as_c_str(~"return",
                           |buf| llvm::LLVMAppendBasicBlock(llfn, buf))}
    }
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn new_fn_ctxt_w_id(ccx: @crate_ctxt,
                    +path: path,
                    llfndecl: ValueRef,
                    id: ast::node_id,
                    impl_id: Option<ast::def_id>,
                    +param_substs: Option<param_substs>,
                    sp: Option<span>) -> fn_ctxt {
    let llbbs = mk_standard_basic_blocks(llfndecl);
    return @fn_ctxt_ {
          llfn: llfndecl,
          llenv: unsafe { llvm::LLVMGetParam(llfndecl, 1u as c_uint) },
          llretptr: unsafe { llvm::LLVMGetParam(llfndecl, 0u as c_uint) },
          mut llstaticallocas: llbbs.sa,
          mut llloadenv: None,
          mut llreturn: llbbs.rt,
          mut llself: None,
          mut personality: None,
          mut loop_ret: None,
          llargs: HashMap(),
          lllocals: HashMap(),
          llupvars: HashMap(),
          id: id,
          impl_id: impl_id,
          param_substs: param_substs,
          span: sp,
          path: path,
          ccx: ccx
    };
}

fn new_fn_ctxt(ccx: @crate_ctxt,
               +path: path,
               llfndecl: ValueRef,
               sp: Option<span>)
            -> fn_ctxt {
    return new_fn_ctxt_w_id(ccx, path, llfndecl, -1, None, None, sp);
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

// create_llargs_for_fn_args: Creates a mapping from incoming arguments to
// allocas created for them.
//
// When we translate a function, we need to map its incoming arguments to the
// spaces that have been created for them (by code in the llallocas field of
// the function's fn_ctxt).  create_llargs_for_fn_args populates the llargs
// field of the fn_ctxt with
fn create_llargs_for_fn_args(cx: fn_ctxt,
                             ty_self: self_arg,
                             args: ~[ast::arg]) -> ~[ValueRef] {
    let _icx = cx.insn_ctxt("create_llargs_for_fn_args");

    match ty_self {
      impl_self(tt) => {
        cx.llself = Some(ValSelfData {
            v: cx.llenv,
            t: tt,
            is_owned: false
        });
      }
      impl_owned_self(tt) => {
        cx.llself = Some(ValSelfData {
            v: cx.llenv,
            t: tt,
            is_owned: true
        });
      }
      no_self => ()
    }

    // Return an array containing the ValueRefs that we get from
    // llvm::LLVMGetParam for each argument.
    vec::from_fn(args.len(), |i| {
        unsafe {
            let arg_n = first_real_arg + i;
            llvm::LLVMGetParam(cx.llfn, arg_n as c_uint)
        }
    })
}

fn copy_args_to_allocas(fcx: fn_ctxt,
                        bcx: block,
                        args: &[ast::arg],
                        raw_llargs: &[ValueRef],
                        arg_tys: &[ty::arg]) -> block {
    let _icx = fcx.insn_ctxt("copy_args_to_allocas");
    let tcx = bcx.tcx();
    let mut bcx = bcx;

    match fcx.llself {
      Some(copy slf) => {
        // We really should do this regardless of whether self is owned, but
        // it doesn't work right with default method impls yet. (FIXME: #2794)
        if slf.is_owned {
            let self_val = PointerCast(bcx, slf.v,
                                       T_ptr(type_of(bcx.ccx(), slf.t)));
            fcx.llself = Some(ValSelfData {v: self_val, ..slf});
            add_clean(bcx, self_val, slf.t);
        }
      }
      _ => {}
    }

    for uint::range(0, arg_tys.len()) |arg_n| {
        let arg_ty = &arg_tys[arg_n];
        let raw_llarg = raw_llargs[arg_n];
        let arg_id = args[arg_n].id;

        // For certain mode/type combinations, the raw llarg values are passed
        // by value.  However, within the fn body itself, we want to always
        // have all locals and arguments be by-ref so that we can cancel the
        // cleanup and for better interaction with LLVM's debug info.  So, if
        // the argument would be passed by value, we store it into an alloca.
        // This alloca should be optimized away by LLVM's mem-to-reg pass in
        // the event it's not truly needed.
        let llarg;
        match ty::resolved_mode(tcx, arg_ty.mode) {
            ast::by_ref => {
                llarg = raw_llarg;
            }
            ast::by_move | ast::by_copy => {
                // only by value if immediate:
                if datum::appropriate_mode(arg_ty.ty).is_by_value() {
                    let alloc = alloc_ty(bcx, arg_ty.ty);
                    Store(bcx, raw_llarg, alloc);
                    llarg = alloc;
                } else {
                    llarg = raw_llarg;
                }

                add_clean(bcx, llarg, arg_ty.ty);
            }
            ast::by_val => {
                // always by value, also not owned, so don't add a cleanup:
                let alloc = alloc_ty(bcx, arg_ty.ty);
                Store(bcx, raw_llarg, alloc);
                llarg = alloc;
            }
        }

        bcx = _match::bind_irrefutable_pat(bcx,
                                          args[arg_n].pat,
                                          llarg,
                                          false,
                                          _match::BindArgument);

        fcx.llargs.insert(arg_id, local_mem(llarg));

        if fcx.ccx.sess.opts.extra_debuginfo {
            debuginfo::create_arg(bcx, args[arg_n], args[arg_n].ty.span);
        }
    }

    return bcx;
}

// Ties up the llstaticallocas -> llloadenv -> lltop edges,
// and builds the return block.
fn finish_fn(fcx: fn_ctxt, lltop: BasicBlockRef) {
    let _icx = fcx.insn_ctxt("finish_fn");
    tie_up_header_blocks(fcx, lltop);
    let ret_cx = raw_block(fcx, false, fcx.llreturn);
    RetVoid(ret_cx);
}

fn tie_up_header_blocks(fcx: fn_ctxt, lltop: BasicBlockRef) {
    let _icx = fcx.insn_ctxt("tie_up_header_blocks");
    match fcx.llloadenv {
        Some(copy ll) => {
            Br(raw_block(fcx, false, fcx.llstaticallocas), ll);
            Br(raw_block(fcx, false, ll), lltop);
        }
        None => {
            Br(raw_block(fcx, false, fcx.llstaticallocas), lltop);
        }
    }
}

enum self_arg { impl_self(ty::t), impl_owned_self(ty::t), no_self, }

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
fn trans_closure(ccx: @crate_ctxt,
                 +path: path,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 llfndecl: ValueRef,
                 ty_self: self_arg,
                 +param_substs: Option<param_substs>,
                 id: ast::node_id,
                 impl_id: Option<ast::def_id>,
                 maybe_load_env: fn(fn_ctxt),
                 finish: fn(block)) {
    ccx.stats.n_closures += 1;
    let _icx = ccx.insn_ctxt("trans_closure");
    set_uwtable(llfndecl);

    // Set up arguments to the function.
    let fcx = new_fn_ctxt_w_id(ccx, path, llfndecl, id, impl_id, param_substs,
                                  Some(body.span));
    let raw_llargs = create_llargs_for_fn_args(fcx, ty_self,
                                               /*bad*/copy decl.inputs);

    // Set GC for function.
    if ccx.sess.opts.gc {
        do str::as_c_str("generic") |strategy| {
            unsafe {
                llvm::LLVMSetGC(fcx.llfn, strategy);
            }
        }
        ccx.uses_gc = true;
    }

    // Create the first basic block in the function and keep a handle on it to
    //  pass to finish_fn later.
    let bcx_top = top_scope_block(fcx, body.info());
    let mut bcx = bcx_top;
    let lltop = bcx.llbb;
    let block_ty = node_id_type(bcx, body.node.id);

    let arg_tys = ty::ty_fn_args(node_id_type(bcx, id));
    bcx = copy_args_to_allocas(fcx, bcx, decl.inputs, raw_llargs, arg_tys);

    maybe_load_env(fcx);

    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).
    if body.node.expr.is_none() || ty::type_is_bot(block_ty) ||
        ty::type_is_nil(block_ty)
    {
        bcx = controlflow::trans_block(bcx, body, expr::Ignore);
    } else {
        bcx = controlflow::trans_block(bcx, body, expr::SaveIn(fcx.llretptr));
    }

    finish(bcx);
    cleanup_and_Br(bcx, bcx_top, fcx.llreturn);

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(fcx, lltop);
}

// trans_fn: creates an LLVM function corresponding to a source language
// function.
fn trans_fn(ccx: @crate_ctxt,
            +path: path,
            decl: ast::fn_decl,
            body: ast::blk,
            llfndecl: ValueRef,
            ty_self: self_arg,
            +param_substs: Option<param_substs>,
            id: ast::node_id,
            impl_id: Option<ast::def_id>) {
    let do_time = ccx.sess.trans_stats();
    let start = if do_time { time::get_time() }
                else { time::Timespec::new(0, 0) };
    debug!("trans_fn(ty_self=%?)", ty_self);
    let _icx = ccx.insn_ctxt("trans_fn");
    ccx.stats.n_fns += 1;
    // XXX: Bad copy of `path`.
    trans_closure(ccx, copy path, decl, body, llfndecl, ty_self,
                  param_substs, id, impl_id,
                  |fcx| {
                      if ccx.sess.opts.extra_debuginfo {
                          debuginfo::create_function(fcx);
                      }
                  },
                  |_bcx| { });
    if do_time {
        let end = time::get_time();
        log_fn_time(ccx, path_str(ccx.sess, path), start, end);
    }
}

fn trans_enum_variant(ccx: @crate_ctxt,
                      enum_id: ast::node_id,
                      variant: ast::variant,
                      args: ~[ast::variant_arg],
                      disr: int,
                      is_degen: bool,
                      +param_substs: Option<param_substs>,
                      llfndecl: ValueRef) {
    let _icx = ccx.insn_ctxt("trans_enum_variant");
    // Translate variant arguments to function arguments.
    let fn_args = do args.map |varg| {
        ast::arg {
            mode: ast::expl(ast::by_copy),
            is_mutbl: false,
            ty: varg.ty,
            pat: ast_util::ident_to_pat(
                ccx.tcx.sess.next_node_id(),
                ast_util::dummy_sp(),
                special_idents::arg),
            id: varg.id,
        }
    };
    // XXX: Bad copy of `param_substs`.
    let fcx = new_fn_ctxt_w_id(ccx, ~[], llfndecl, variant.node.id, None,
                               copy param_substs, None);
    // XXX: Bad copy.
    let raw_llargs = create_llargs_for_fn_args(fcx, no_self, copy fn_args);
    let ty_param_substs = match param_substs {
      Some(ref substs) => /*bad*/copy substs.tys,
      None => ~[]
    };
    let bcx = top_scope_block(fcx, None), lltop = bcx.llbb;
    let arg_tys = ty::ty_fn_args(node_id_type(bcx, variant.node.id));
    let bcx = copy_args_to_allocas(fcx, bcx, fn_args, raw_llargs, arg_tys);

    // Cast the enum to a type we can GEP into.
    let llblobptr = if is_degen {
        fcx.llretptr
    } else {
        let llenumptr =
            PointerCast(bcx, fcx.llretptr, T_opaque_enum_ptr(ccx));
        let lldiscrimptr = GEPi(bcx, llenumptr, [0u, 0u]);
        Store(bcx, C_int(ccx, disr), lldiscrimptr);
        GEPi(bcx, llenumptr, [0u, 1u])
    };
    let t_id = local_def(enum_id);
    let v_id = local_def(variant.node.id);
    for vec::eachi(args) |i, va| {
        let lldestptr = GEP_enum(bcx, llblobptr, t_id, v_id,
                                 /*bad*/copy ty_param_substs, i);
        // If this argument to this function is a enum, it'll have come in to
        // this function as an opaque blob due to the way that type_of()
        // works. So we have to cast to the destination's view of the type.
        let llarg = match fcx.llargs.find(va.id) {
            Some(local_mem(x)) => x,
            _ => fail ~"trans_enum_variant: how do we know this works?",
        };
        let arg_ty = arg_tys[i].ty;
        memcpy_ty(bcx, lldestptr, llarg, arg_ty);
    }
    build_return(bcx);
    finish_fn(fcx, lltop);
}

// NB: In theory this should be merged with the function above. But the AST
// structures are completely different, so very little code would be shared.
fn trans_tuple_struct(ccx: @crate_ctxt,
                      fields: ~[@ast::struct_field],
                      ctor_id: ast::node_id,
                      +param_substs: Option<param_substs>,
                      llfndecl: ValueRef) {
    let _icx = ccx.insn_ctxt("trans_tuple_struct");

    // Translate struct fields to function arguments.
    let fn_args = do fields.map |field| {
        ast::arg {
            mode: ast::expl(ast::by_copy),
            is_mutbl: false,
            ty: field.node.ty,
            pat: ast_util::ident_to_pat(ccx.tcx.sess.next_node_id(),
                                        ast_util::dummy_sp(),
                                        special_idents::arg),
            id: field.node.id
        }
    };

    let fcx = new_fn_ctxt_w_id(ccx, ~[], llfndecl, ctor_id, None,
                               param_substs, None);

    // XXX: Bad copy.
    let raw_llargs = create_llargs_for_fn_args(fcx, no_self, copy fn_args);

    let bcx = top_scope_block(fcx, None);
    let lltop = bcx.llbb;
    let arg_tys = ty::ty_fn_args(node_id_type(bcx, ctor_id));
    let bcx = copy_args_to_allocas(fcx, bcx, fn_args, raw_llargs, arg_tys);

    for fields.eachi |i, field| {
        let lldestptr = GEPi(bcx, fcx.llretptr, [0, 0, i]);
        let llarg = match fcx.llargs.get(field.node.id) {
            local_mem(x) => x,
            _ => {
                ccx.tcx.sess.bug(~"trans_tuple_struct: llarg wasn't \
                                   local_mem")
            }
        };
        let arg_ty = arg_tys[i].ty;
        memcpy_ty(bcx, lldestptr, llarg, arg_ty);
    }

    build_return(bcx);
    finish_fn(fcx, lltop);
}

fn trans_struct_dtor(ccx: @crate_ctxt,
                     +path: path,
                     body: ast::blk,
                     dtor_id: ast::node_id,
                     +psubsts: Option<param_substs>,
                     hash_id: Option<mono_id>,
                     parent_id: ast::def_id)
                  -> ValueRef {
  let tcx = ccx.tcx;
  /* Look up the parent class's def_id */
  let mut class_ty = ty::lookup_item_type(tcx, parent_id).ty;
  /* Substitute in the class type if necessary */
    do option::iter(&psubsts) |ss| {
    class_ty = ty::subst_tps(tcx, ss.tys, ss.self_ty, class_ty);
  }

  /* The dtor takes a (null) output pointer, and a self argument,
     and returns () */
  let lldty = type_of_dtor(ccx, class_ty);

  // XXX: Bad copies.
  let s = get_dtor_symbol(ccx, copy path, dtor_id, copy psubsts);

  /* Register the dtor as a function. It has external linkage */
  let lldecl = decl_internal_cdecl_fn(ccx.llmod, s, lldty);
  lib::llvm::SetLinkage(lldecl, lib::llvm::ExternalLinkage);

  /* If we're monomorphizing, register the monomorphized decl
     for the dtor */
    do option::iter(&hash_id) |h_id| {
    ccx.monomorphized.insert(*h_id, lldecl);
  }
  /* Translate the dtor body */
  trans_fn(ccx, path, ast_util::dtor_dec(),
           body, lldecl, impl_self(class_ty), psubsts, dtor_id, None);
  lldecl
}

fn trans_enum_def(ccx: @crate_ctxt, enum_definition: ast::enum_def,
                  id: ast::node_id, tps: ~[ast::ty_param], degen: bool,
                  path: @ast_map::path, vi: @~[ty::VariantInfo],
                  i: &mut uint) {
    for vec::each(enum_definition.variants) |variant| {
        let disr_val = vi[*i].disr_val;
        *i += 1;

        match variant.node.kind {
            ast::tuple_variant_kind(ref args) if args.len() > 0 => {
                let llfn = get_item_val(ccx, variant.node.id);
                trans_enum_variant(ccx, id, *variant, /*bad*/copy *args,
                                   disr_val, degen, None, llfn);
            }
            ast::tuple_variant_kind(_) => {
                // Nothing to do.
            }
            ast::struct_variant_kind(struct_def) => {
                trans_struct_def(ccx, struct_def, /*bad*/copy tps, path,
                                 variant.node.id);
            }
            ast::enum_variant_kind(ref enum_definition) => {
                trans_enum_def(ccx,
                               *enum_definition,
                               id,
                               /*bad*/copy tps,
                               degen,
                               path,
                               vi,
                               &mut *i);
            }
        }
    }
}

fn trans_item(ccx: @crate_ctxt, item: ast::item) {
    let _icx = ccx.insn_ctxt("trans_item");
    let path = match ccx.tcx.items.get(item.id) {
        ast_map::node_item(_, p) => p,
        // tjc: ?
        _ => fail ~"trans_item",
    };
    match /*bad*/copy item.node {
      // XXX: Bad copies.
      ast::item_fn(copy decl, purity, copy tps, ref body) => {
        if purity == ast::extern_fn  {
            let llfndecl = get_item_val(ccx, item.id);
            foreign::trans_foreign_fn(ccx,
                                     vec::append(
                                         /*bad*/copy *path,
                                         ~[path_name(item.ident)]),
                                     decl, (*body), llfndecl, item.id);
        } else if tps.is_empty() {
            let llfndecl = get_item_val(ccx, item.id);
            trans_fn(ccx,
                     vec::append(/*bad*/copy *path, ~[path_name(item.ident)]),
                     decl, (*body), llfndecl, no_self, None, item.id, None);
        } else {
            for vec::each((*body).node.stmts) |stmt| {
                match stmt.node {
                  ast::stmt_decl(@ast::spanned { node: ast::decl_item(i),
                                                 _ }, _) => {
                    trans_item(ccx, *i);
                  }
                  _ => ()
                }
            }
        }
      }
      ast::item_impl(tps, _, _, ms) => {
        meth::trans_impl(ccx, /*bad*/copy *path, item.ident, ms, tps, None,
                         item.id);
      }
      ast::item_mod(m) => {
        trans_mod(ccx, m);
      }
      ast::item_enum(ref enum_definition, ref tps) => {
        if tps.len() == 0u {
            let degen = (*enum_definition).variants.len() == 1u;
            let vi = ty::enum_variants(ccx.tcx, local_def(item.id));
            let mut i = 0;
            trans_enum_def(ccx, (*enum_definition), item.id, /*bad*/copy *tps,
                           degen, path, vi, &mut i);
        }
      }
      ast::item_const(_, expr) => consts::trans_const(ccx, expr, item.id),
      ast::item_foreign_mod(foreign_mod) => {
        let abi = match attr::foreign_abi(item.attrs) {
          either::Right(abi_) => abi_,
          either::Left(ref msg) => ccx.sess.span_fatal(item.span,
                                                       /*bad*/copy *msg)
        };
        foreign::trans_foreign_mod(ccx, foreign_mod, abi);
      }
      ast::item_struct(struct_def, tps) => {
        trans_struct_def(ccx, struct_def, tps, path, item.id);
      }
      _ => {/* fall through */ }
    }
}

fn trans_struct_def(ccx: @crate_ctxt, struct_def: @ast::struct_def,
                    tps: ~[ast::ty_param], path: @ast_map::path,
                    id: ast::node_id) {
    // If there are type parameters, the destructor and constructor will be
    // monomorphized, so we don't translate them here.
    if tps.len() == 0u {
        // Translate the destructor.
        do option::iter(&struct_def.dtor) |dtor| {
            trans_struct_dtor(ccx, /*bad*/copy *path, dtor.node.body,
                             dtor.node.id, None, None, local_def(id));
        };

        // If this is a tuple-like struct, translate the constructor.
        match struct_def.ctor_id {
            // We only need to translate a constructor if there are fields;
            // otherwise this is a unit-like struct.
            Some(ctor_id) if struct_def.fields.len() > 0 => {
                let llfndecl = get_item_val(ccx, ctor_id);
                trans_tuple_struct(ccx, /*bad*/copy struct_def.fields,
                                   ctor_id, None, llfndecl);
            }
            Some(_) | None => {}
        }
    }
}

// Translate a module. Doing this amounts to translating the items in the
// module; there ends up being no artifact (aside from linkage names) of
// separate modules in the compiled program.  That's because modules exist
// only as a convenience for humans working with the code, to organize names
// and control visibility.
fn trans_mod(ccx: @crate_ctxt, m: ast::_mod) {
    let _icx = ccx.insn_ctxt("trans_mod");
    for vec::each(m.items) |item| {
        trans_item(ccx, **item);
    }
}

fn get_pair_fn_ty(llpairty: TypeRef) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the pair.
    return struct_elt(llpairty, 0u);
}

fn register_fn(ccx: @crate_ctxt,
               sp: span,
               +path: path,
               node_id: ast::node_id,
               attrs: &[ast::attribute])
            -> ValueRef {
    let t = ty::node_id_to_type(ccx.tcx, node_id);
    register_fn_full(ccx, sp, path, node_id, attrs, t)
}

fn register_fn_full(ccx: @crate_ctxt,
                    sp: span,
                    +path: path,
                    node_id: ast::node_id,
                    attrs: &[ast::attribute],
                    node_type: ty::t)
                 -> ValueRef {
    let llfty = type_of_fn_from_ty(ccx, node_type);
    register_fn_fuller(ccx, sp, path, node_id, attrs, node_type,
                       lib::llvm::CCallConv, llfty)
}

fn register_fn_fuller(ccx: @crate_ctxt,
                      sp: span,
                      +path: path,
                      node_id: ast::node_id,
                      attrs: &[ast::attribute],
                      node_type: ty::t,
                      cc: lib::llvm::CallConv,
                      llfty: TypeRef)
                   -> ValueRef {
    debug!("register_fn_fuller creating fn for item %d with path %s",
           node_id,
           ast_map::path_to_str(path, ccx.sess.parse_sess.interner));

    let ps = if attr::attrs_contains_name(attrs, "no_mangle") {
        path_elt_to_str(path.last(), ccx.sess.parse_sess.interner)
    } else {
        mangle_exported_name(ccx, /*bad*/copy path, node_type)
    };

    // XXX: Bad copy.
    let llfn: ValueRef = decl_fn(ccx.llmod, copy ps, cc, llfty);
    ccx.item_symbols.insert(node_id, ps);

    // FIXME #4404 android JNI hacks
    let is_main = is_main_fn(&ccx.sess, node_id) &&
                     (!ccx.sess.building_library ||
                      (ccx.sess.building_library &&
                       ccx.sess.targ_cfg.os == session::os_android));
    if is_main { create_main_wrapper(ccx, sp, llfn); }
    llfn
}

fn is_main_fn(sess: &Session, node_id: ast::node_id) -> bool {
    match sess.main_fn {
        Some((main_id, _)) => node_id == main_id,
        None => false
    }
}

// Create a _rust_main(args: ~[str]) function which will be called from the
// runtime rust_start function
fn create_main_wrapper(ccx: @crate_ctxt, _sp: span, main_llfn: ValueRef) {

    let llfn = create_main(ccx, main_llfn);
    create_entry_fn(ccx, llfn);

    fn create_main(ccx: @crate_ctxt, main_llfn: ValueRef) -> ValueRef {
        let unit_ty = ty::mk_estr(ccx.tcx, ty::vstore_uniq);
        let vecarg_ty: ty::arg =
            {mode: ast::expl(ast::by_val),
             ty: ty::mk_evec(ccx.tcx, ty::mt {ty: unit_ty, mutbl: ast::m_imm},
                             ty::vstore_uniq)};
        let nt = ty::mk_nil(ccx.tcx);
        let llfty = type_of_fn(ccx, ~[vecarg_ty], nt);
        let llfdecl = decl_fn(ccx.llmod, ~"_rust_main",
                              lib::llvm::CCallConv, llfty);

        let fcx = new_fn_ctxt(ccx, ~[], llfdecl, None);

        let bcx = top_scope_block(fcx, None);
        let lltop = bcx.llbb;

        // Call main.
        let lloutputarg = unsafe { llvm::LLVMGetParam(llfdecl, 0 as c_uint) };
        let llenvarg = unsafe { llvm::LLVMGetParam(llfdecl, 1 as c_uint) };
        let mut args = ~[lloutputarg, llenvarg];
        Call(bcx, main_llfn, args);

        build_return(bcx);
        finish_fn(fcx, lltop);
        return llfdecl;
    }

    fn create_entry_fn(ccx: @crate_ctxt, rust_main: ValueRef) {
        #[cfg(windows)]
        fn main_name() -> ~str { return ~"WinMain@16"; }
        #[cfg(unix)]
        fn main_name() -> ~str { return ~"main"; }
        let llfty = T_fn(~[ccx.int_type, ccx.int_type], ccx.int_type);

        // FIXME #4404 android JNI hacks
        let llfn = if ccx.sess.building_library {
            decl_cdecl_fn(ccx.llmod, ~"amain", llfty)
        } else {
            decl_cdecl_fn(ccx.llmod, main_name(), llfty)
        };
        let llbb = str::as_c_str(~"top", |buf| {
            unsafe {
                llvm::LLVMAppendBasicBlock(llfn, buf)
            }
        });
        let bld = ccx.builder.B;
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bld, llbb);
        }
        let crate_map = ccx.crate_map;
        let start_ty = T_fn(~[val_ty(rust_main), ccx.int_type, ccx.int_type,
                             val_ty(crate_map)], ccx.int_type);
        let start = decl_cdecl_fn(ccx.llmod, ~"rust_start", start_ty);

        let args = unsafe {
            if ccx.sess.building_library {
                ~[rust_main,
                  llvm::LLVMConstInt(T_i32(), 0u as c_ulonglong, False),
                  llvm::LLVMConstInt(T_i32(), 0u as c_ulonglong, False),
                  crate_map]
            } else {
                ~[rust_main, llvm::LLVMGetParam(llfn, 0 as c_uint),
                  llvm::LLVMGetParam(llfn, 1 as c_uint), crate_map]
            }
        };

        let result = unsafe {
            llvm::LLVMBuildCall(bld, start, vec::raw::to_ptr(args),
                                args.len() as c_uint, noname())
        };
        unsafe {
            llvm::LLVMBuildRet(bld, result);
        }
    }
}

fn fill_fn_pair(bcx: block, pair: ValueRef, llfn: ValueRef,
                llenvptr: ValueRef) {
    let ccx = bcx.ccx();
    let code_cell = GEPi(bcx, pair, [0u, abi::fn_field_code]);
    Store(bcx, llfn, code_cell);
    let env_cell = GEPi(bcx, pair, [0u, abi::fn_field_box]);
    let llenvblobptr = PointerCast(bcx, llenvptr, T_opaque_box_ptr(ccx));
    Store(bcx, llenvblobptr, env_cell);
}

fn item_path(ccx: @crate_ctxt, i: @ast::item) -> path {
    vec::append(
        /*bad*/copy *match ccx.tcx.items.get(i.id) {
            ast_map::node_item(_, p) => p,
                // separate map for paths?
            _ => fail ~"item_path"
        },
        ~[path_name(i.ident)])
}

/* If there's already a symbol for the dtor with <id> and substs <substs>,
   return it; otherwise, create one and register it, returning it as well */
fn get_dtor_symbol(ccx: @crate_ctxt,
                   +path: path,
                   id: ast::node_id,
                   +substs: Option<param_substs>)
                -> ~str {
  let t = ty::node_id_to_type(ccx.tcx, id);
  match ccx.item_symbols.find(id) {
     Some(ref s) => (/*bad*/copy *s),
     None if substs.is_none() => {
       let s = mangle_exported_name(
           ccx,
           vec::append(path, ~[path_name((ccx.names)(~"dtor"))]),
           t);
       // XXX: Bad copy, use `@str`?
       ccx.item_symbols.insert(id, copy s);
       s
     }
     None   => {
       // Monomorphizing, so just make a symbol, don't add
       // this to item_symbols
       match substs {
         Some(ss) => {
           let mono_ty = ty::subst_tps(ccx.tcx, ss.tys, ss.self_ty, t);
           mangle_exported_name(
               ccx,
               vec::append(path,
                           ~[path_name((ccx.names)(~"dtor"))]),
               mono_ty)
         }
         None => {
             ccx.sess.bug(fmt!("get_dtor_symbol: not monomorphizing and \
               couldn't find a symbol for dtor %?", path));
         }
       }
     }
  }
}

fn get_item_val(ccx: @crate_ctxt, id: ast::node_id) -> ValueRef {
    debug!("get_item_val(id=`%?`)", id);
    let tcx = ccx.tcx;
    match ccx.item_vals.find(id) {
      Some(v) => v,
      None => {

        let mut exprt = false;
        let val = match ccx.tcx.items.get(id) {
          ast_map::node_item(i, pth) => {
            let my_path = vec::append(/*bad*/copy *pth,
                                      ~[path_name(i.ident)]);
            match i.node {
              ast::item_const(_, expr) => {
                let typ = ty::node_id_to_type(ccx.tcx, i.id);
                let s = mangle_exported_name(ccx, my_path, typ);
                // We need the translated value here, because for enums the
                // LLVM type is not fully determined by the Rust type.
                let v = consts::const_expr(ccx, expr);
                ccx.const_values.insert(id, v);
                unsafe {
                    let llty = llvm::LLVMTypeOf(v);
                    let g = str::as_c_str(s, |buf| {
                        llvm::LLVMAddGlobal(ccx.llmod, llty, buf)
                    });
                    ccx.item_symbols.insert(i.id, s);
                    g
                }
              }
              ast::item_fn(_, purity, _, _) => {
                let llfn = if purity != ast::extern_fn {
                    register_fn(ccx, i.span, my_path, i.id, i.attrs)
                } else {
                    foreign::register_foreign_fn(ccx,
                                                 i.span,
                                                 my_path,
                                                 i.id,
                                                 i.attrs)
                };
                set_inline_hint_if_appr(/*bad*/copy i.attrs, llfn);
                llfn
              }
              _ => fail ~"get_item_val: weird result in table"
            }
          }
          ast_map::node_trait_method(trait_method, _, pth) => {
            debug!("get_item_val(): processing a node_trait_method");
            match *trait_method {
              ast::required(_) => {
                ccx.sess.bug(~"unexpected variant: required trait method in \
                               get_item_val()");
              }
              ast::provided(m) => {
                exprt = true;
                register_method(ccx, id, pth, m)
              }
            }
          }
          ast_map::node_method(m, _, pth) => {
            exprt = true;
            register_method(ccx, id, pth, m)
          }
          ast_map::node_foreign_item(ni, _, pth) => {
            exprt = true;
            match ni.node {
                ast::foreign_item_fn(*) => {
                    register_fn(ccx, ni.span,
                                vec::append(/*bad*/copy *pth,
                                            ~[path_name(ni.ident)]),
                                ni.id,
                                ni.attrs)
                }
                ast::foreign_item_const(*) => {
                    let typ = ty::node_id_to_type(ccx.tcx, ni.id);
                    let ident = ccx.sess.parse_sess.interner.get(ni.ident);
                    let g = do str::as_c_str(*ident) |buf| {
                        unsafe {
                            llvm::LLVMAddGlobal(ccx.llmod,
                                                type_of(ccx, typ),
                                                buf)
                        }
                    };
                    g
                }
            }
          }
          ast_map::node_dtor(_, dt, parent_id, pt) => {
            /*
                Don't just call register_fn, since we don't want to add
                the implicit self argument automatically (we want to make sure
                it has the right type)
            */
            // Want parent_id and not id, because id is the dtor's type
            let class_ty = ty::lookup_item_type(tcx, parent_id).ty;
            // This code shouldn't be reached if the class is generic
            assert !ty::type_has_params(class_ty);
            let lldty = unsafe {
                T_fn(~[
                    T_ptr(type_of(ccx, ty::mk_nil(tcx))),
                    T_ptr(type_of(ccx, class_ty))
                ],
                llvm::LLVMVoidType())
            };
            let s = get_dtor_symbol(ccx, /*bad*/copy *pt, dt.node.id, None);

            /* Make the declaration for the dtor */
            let llfn = decl_internal_cdecl_fn(ccx.llmod, s, lldty);
            lib::llvm::SetLinkage(llfn, lib::llvm::ExternalLinkage);
            llfn
          }

          ast_map::node_variant(ref v, enm, pth) => {
            let llfn;
            match /*bad*/copy (*v).node.kind {
                ast::tuple_variant_kind(args) => {
                    assert args.len() != 0u;
                    let pth = vec::append(/*bad*/copy *pth,
                                          ~[path_name(enm.ident),
                                            path_name((*v).node.name)]);
                    llfn = match enm.node {
                      ast::item_enum(_, _) => {
                        register_fn(ccx, (*v).span, pth, id, enm.attrs)
                      }
                      _ => fail ~"node_variant, shouldn't happen"
                    };
                }
                ast::struct_variant_kind(_) => {
                    fail ~"struct variant kind unexpected in get_item_val"
                }
                ast::enum_variant_kind(_) => {
                    fail ~"enum variant kind unexpected in get_item_val"
                }
            }
            set_inline_hint(llfn);
            llfn
          }

          ast_map::node_struct_ctor(struct_def, struct_item, struct_path) => {
            // Only register the constructor if this is a tuple-like struct.
            match struct_def.ctor_id {
                None => {
                    ccx.tcx.sess.bug(~"attempt to register a constructor of \
                                       a non-tuple-like struct")
                }
                Some(ctor_id) => {
                    let llfn = register_fn(ccx,
                                           struct_item.span,
                                           /*bad*/copy *struct_path,
                                           ctor_id,
                                           struct_item.attrs);
                    set_inline_hint(llfn);
                    llfn
                }
            }
          }

          _ => {
            ccx.sess.bug(~"get_item_val(): unexpected variant")
          }
        };
        if !(exprt || ccx.reachable.contains_key(id)) {
            lib::llvm::SetLinkage(val, lib::llvm::InternalLinkage);
        }
        ccx.item_vals.insert(id, val);
        val
      }
    }
}

fn register_method(ccx: @crate_ctxt, id: ast::node_id, pth: @ast_map::path,
                m: @ast::method) -> ValueRef {
    let mty = ty::node_id_to_type(ccx.tcx, id);
    let pth = vec::append(/*bad*/copy *pth, ~[path_name((ccx.names)(~"meth")),
                                  path_name(m.ident)]);
    let llfn = register_fn_full(ccx, m.span, pth, id, m.attrs, mty);
    set_inline_hint_if_appr(/*bad*/copy m.attrs, llfn);
    llfn
}

// The constant translation pass.
fn trans_constant(ccx: @crate_ctxt, it: @ast::item) {
    let _icx = ccx.insn_ctxt("trans_constant");
    match it.node {
      ast::item_enum(ref enum_definition, _) => {
        let vi = ty::enum_variants(ccx.tcx,
                                   ast::def_id { crate: ast::local_crate,
                                                 node: it.id });
        let mut i = 0;
        let path = item_path(ccx, it);
        for vec::each((*enum_definition).variants) |variant| {
            let p = vec::append(/*bad*/copy path, ~[
                path_name(variant.node.name),
                path_name(special_idents::descrim)
            ]);
            let s = mangle_exported_name(ccx, p, ty::mk_int(ccx.tcx));
            let disr_val = vi[i].disr_val;
            // XXX: Bad copy.
            note_unique_llvm_symbol(ccx, copy s);
            let discrim_gvar = str::as_c_str(s, |buf| {
                unsafe {
                    llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
                }
            });
            unsafe {
                llvm::LLVMSetInitializer(discrim_gvar, C_int(ccx, disr_val));
                llvm::LLVMSetGlobalConstant(discrim_gvar, True);
            }
            ccx.discrims.insert(
                local_def(variant.node.id), discrim_gvar);
            ccx.discrim_symbols.insert(variant.node.id, s);
            i += 1;
        }
      }
      _ => ()
    }
}

fn trans_constants(ccx: @crate_ctxt, crate: &ast::crate) {
    visit::visit_crate(
        *crate, (),
        visit::mk_simple_visitor(@visit::SimpleVisitor {
            visit_item: |a| trans_constant(ccx, a),
            ..*visit::default_simple_visitor()
        }));
}

fn vp2i(cx: block, v: ValueRef) -> ValueRef {
    let ccx = cx.ccx();
    return PtrToInt(cx, v, ccx.int_type);
}

fn p2i(ccx: @crate_ctxt, v: ValueRef) -> ValueRef {
    unsafe {
        return llvm::LLVMConstPtrToInt(v, ccx.int_type);
    }
}

fn declare_intrinsics(llmod: ModuleRef) -> HashMap<~str, ValueRef> {
    let T_memcpy32_args: ~[TypeRef] =
        ~[T_ptr(T_i8()), T_ptr(T_i8()), T_i32(), T_i32(), T_i1()];
    let T_memcpy64_args: ~[TypeRef] =
        ~[T_ptr(T_i8()), T_ptr(T_i8()), T_i64(), T_i32(), T_i1()];
    let T_memset32_args: ~[TypeRef] =
        ~[T_ptr(T_i8()), T_i8(), T_i32(), T_i32(), T_i1()];
    let T_memset64_args: ~[TypeRef] =
        ~[T_ptr(T_i8()), T_i8(), T_i64(), T_i32(), T_i1()];
    let T_trap_args: ~[TypeRef] = ~[];
    let T_frameaddress_args: ~[TypeRef] = ~[T_i32()];
    let gcroot =
        decl_cdecl_fn(llmod, ~"llvm.gcroot",
                      T_fn(~[T_ptr(T_ptr(T_i8())), T_ptr(T_i8())],
                           T_void()));
    let gcread =
        decl_cdecl_fn(llmod, ~"llvm.gcread",
                      T_fn(~[T_ptr(T_i8()), T_ptr(T_ptr(T_i8()))],
                           T_void()));
    let memcpy32 =
        decl_cdecl_fn(llmod, ~"llvm.memcpy.p0i8.p0i8.i32",
                      T_fn(T_memcpy32_args, T_void()));
    let memcpy64 =
        decl_cdecl_fn(llmod, ~"llvm.memcpy.p0i8.p0i8.i64",
                      T_fn(T_memcpy64_args, T_void()));
    let memset32 =
        decl_cdecl_fn(llmod, ~"llvm.memset.p0i8.i32",
                      T_fn(T_memset32_args, T_void()));
    let memset64 =
        decl_cdecl_fn(llmod, ~"llvm.memset.p0i8.i64",
                      T_fn(T_memset64_args, T_void()));
    let trap = decl_cdecl_fn(llmod, ~"llvm.trap", T_fn(T_trap_args,
                                                      T_void()));
    let frameaddress = decl_cdecl_fn(llmod, ~"llvm.frameaddress",
                                     T_fn(T_frameaddress_args,
                                          T_ptr(T_i8())));
    let sqrtf32 = decl_cdecl_fn(llmod, ~"llvm.sqrt.f32",
                                T_fn(~[T_f32()], T_f32()));
    let sqrtf64 = decl_cdecl_fn(llmod, ~"llvm.sqrt.f64",
                                T_fn(~[T_f64()], T_f64()));
    let powif32 = decl_cdecl_fn(llmod, ~"llvm.powi.f32",
                                T_fn(~[T_f32(), T_i32()], T_f32()));
    let powif64 = decl_cdecl_fn(llmod, ~"llvm.powi.f64",
                                T_fn(~[T_f64(), T_i32()], T_f64()));
    let sinf32 = decl_cdecl_fn(llmod, ~"llvm.sin.f32",
                                T_fn(~[T_f32()], T_f32()));
    let sinf64 = decl_cdecl_fn(llmod, ~"llvm.sin.f64",
                                T_fn(~[T_f64()], T_f64()));
    let cosf32 = decl_cdecl_fn(llmod, ~"llvm.cos.f32",
                                T_fn(~[T_f32()], T_f32()));
    let cosf64 = decl_cdecl_fn(llmod, ~"llvm.cos.f64",
                                T_fn(~[T_f64()], T_f64()));
    let powf32 = decl_cdecl_fn(llmod, ~"llvm.pow.f32",
                                T_fn(~[T_f32(), T_f32()], T_f32()));
    let powf64 = decl_cdecl_fn(llmod, ~"llvm.pow.f64",
                                T_fn(~[T_f64(), T_f64()], T_f64()));
    let expf32 = decl_cdecl_fn(llmod, ~"llvm.exp.f32",
                                T_fn(~[T_f32()], T_f32()));
    let expf64 = decl_cdecl_fn(llmod, ~"llvm.exp.f64",
                                T_fn(~[T_f64()], T_f64()));
    let exp2f32 = decl_cdecl_fn(llmod, ~"llvm.exp2.f32",
                                T_fn(~[T_f32()], T_f32()));
    let exp2f64 = decl_cdecl_fn(llmod, ~"llvm.exp2.f64",
                                T_fn(~[T_f64()], T_f64()));
    let logf32 = decl_cdecl_fn(llmod, ~"llvm.log.f32",
                                T_fn(~[T_f32()], T_f32()));
    let logf64 = decl_cdecl_fn(llmod, ~"llvm.log.f64",
                                T_fn(~[T_f64()], T_f64()));
    let log10f32 = decl_cdecl_fn(llmod, ~"llvm.log10.f32",
                                T_fn(~[T_f32()], T_f32()));
    let log10f64 = decl_cdecl_fn(llmod, ~"llvm.log10.f64",
                                T_fn(~[T_f64()], T_f64()));
    let log2f32 = decl_cdecl_fn(llmod, ~"llvm.log2.f32",
                                T_fn(~[T_f32()], T_f32()));
    let log2f64 = decl_cdecl_fn(llmod, ~"llvm.log2.f64",
                                T_fn(~[T_f64()], T_f64()));
    let fmaf32 = decl_cdecl_fn(llmod, ~"llvm.fma.f32",
                                T_fn(~[T_f32(), T_f32(), T_f32()], T_f32()));
    let fmaf64 = decl_cdecl_fn(llmod, ~"llvm.fma.f64",
                                T_fn(~[T_f64(), T_f64(), T_f64()], T_f64()));
    let fabsf32 = decl_cdecl_fn(llmod, ~"llvm.fabs.f32",
                                T_fn(~[T_f32()], T_f32()));
    let fabsf64 = decl_cdecl_fn(llmod, ~"llvm.fabs.f64",
                                T_fn(~[T_f64()], T_f64()));
    let floorf32 = decl_cdecl_fn(llmod, ~"llvm.floor.f32",
                                T_fn(~[T_f32()], T_f32()));
    let floorf64 = decl_cdecl_fn(llmod, ~"llvm.floor.f64",
                                T_fn(~[T_f64()], T_f64()));
    let ceilf32 = decl_cdecl_fn(llmod, ~"llvm.ceil.f32",
                                T_fn(~[T_f32()], T_f32()));
    let ceilf64 = decl_cdecl_fn(llmod, ~"llvm.ceil.f64",
                                T_fn(~[T_f64()], T_f64()));
    let truncf32 = decl_cdecl_fn(llmod, ~"llvm.trunc.f32",
                                T_fn(~[T_f32()], T_f32()));
    let truncf64 = decl_cdecl_fn(llmod, ~"llvm.trunc.f64",
                                T_fn(~[T_f64()], T_f64()));
    let ctpop8 = decl_cdecl_fn(llmod, ~"llvm.ctpop.i8",
                                T_fn(~[T_i8()], T_i8()));
    let ctpop16 = decl_cdecl_fn(llmod, ~"llvm.ctpop.i16",
                                T_fn(~[T_i16()], T_i16()));
    let ctpop32 = decl_cdecl_fn(llmod, ~"llvm.ctpop.i32",
                                T_fn(~[T_i32()], T_i32()));
    let ctpop64 = decl_cdecl_fn(llmod, ~"llvm.ctpop.i64",
                                T_fn(~[T_i64()], T_i64()));
    let ctlz8 = decl_cdecl_fn(llmod, ~"llvm.ctlz.i8",
                                T_fn(~[T_i8(), T_i1()], T_i8()));
    let ctlz16 = decl_cdecl_fn(llmod, ~"llvm.ctlz.i16",
                                T_fn(~[T_i16(), T_i1()], T_i16()));
    let ctlz32 = decl_cdecl_fn(llmod, ~"llvm.ctlz.i32",
                                T_fn(~[T_i32(), T_i1()], T_i32()));
    let ctlz64 = decl_cdecl_fn(llmod, ~"llvm.ctlz.i64",
                                T_fn(~[T_i64(), T_i1()], T_i64()));
    let cttz8 = decl_cdecl_fn(llmod, ~"llvm.cttz.i8",
                                T_fn(~[T_i8(), T_i1()], T_i8()));
    let cttz16 = decl_cdecl_fn(llmod, ~"llvm.cttz.i16",
                                T_fn(~[T_i16(), T_i1()], T_i16()));
    let cttz32 = decl_cdecl_fn(llmod, ~"llvm.cttz.i32",
                                T_fn(~[T_i32(), T_i1()], T_i32()));
    let cttz64 = decl_cdecl_fn(llmod, ~"llvm.cttz.i64",
                                T_fn(~[T_i64(), T_i1()], T_i64()));
    let bswap16 = decl_cdecl_fn(llmod, ~"llvm.bswap.i16",
                                T_fn(~[T_i16()], T_i16()));
    let bswap32 = decl_cdecl_fn(llmod, ~"llvm.bswap.i32",
                                T_fn(~[T_i32()], T_i32()));
    let bswap64 = decl_cdecl_fn(llmod, ~"llvm.bswap.i64",
                                T_fn(~[T_i64()], T_i64()));

    let intrinsics = HashMap();
    intrinsics.insert(~"llvm.gcroot", gcroot);
    intrinsics.insert(~"llvm.gcread", gcread);
    intrinsics.insert(~"llvm.memcpy.p0i8.p0i8.i32", memcpy32);
    intrinsics.insert(~"llvm.memcpy.p0i8.p0i8.i64", memcpy64);
    intrinsics.insert(~"llvm.memset.p0i8.i32", memset32);
    intrinsics.insert(~"llvm.memset.p0i8.i64", memset64);
    intrinsics.insert(~"llvm.trap", trap);
    intrinsics.insert(~"llvm.frameaddress", frameaddress);
    intrinsics.insert(~"llvm.sqrt.f32", sqrtf32);
    intrinsics.insert(~"llvm.sqrt.f64", sqrtf64);
    intrinsics.insert(~"llvm.powi.f32", powif32);
    intrinsics.insert(~"llvm.powi.f64", powif64);
    intrinsics.insert(~"llvm.sin.f32", sinf32);
    intrinsics.insert(~"llvm.sin.f64", sinf64);
    intrinsics.insert(~"llvm.cos.f32", cosf32);
    intrinsics.insert(~"llvm.cos.f64", cosf64);
    intrinsics.insert(~"llvm.pow.f32", powf32);
    intrinsics.insert(~"llvm.pow.f64", powf64);
    intrinsics.insert(~"llvm.exp.f32", expf32);
    intrinsics.insert(~"llvm.exp.f64", expf64);
    intrinsics.insert(~"llvm.exp2.f32", exp2f32);
    intrinsics.insert(~"llvm.exp2.f64", exp2f64);
    intrinsics.insert(~"llvm.log.f32", logf32);
    intrinsics.insert(~"llvm.log.f64", logf64);
    intrinsics.insert(~"llvm.log10.f32", log10f32);
    intrinsics.insert(~"llvm.log10.f64", log10f64);
    intrinsics.insert(~"llvm.log2.f32", log2f32);
    intrinsics.insert(~"llvm.log2.f64", log2f64);
    intrinsics.insert(~"llvm.fma.f32", fmaf32);
    intrinsics.insert(~"llvm.fma.f64", fmaf64);
    intrinsics.insert(~"llvm.fabs.f32", fabsf32);
    intrinsics.insert(~"llvm.fabs.f64", fabsf64);
    intrinsics.insert(~"llvm.floor.f32", floorf32);
    intrinsics.insert(~"llvm.floor.f64", floorf64);
    intrinsics.insert(~"llvm.ceil.f32", ceilf32);
    intrinsics.insert(~"llvm.ceil.f64", ceilf64);
    intrinsics.insert(~"llvm.trunc.f32", truncf32);
    intrinsics.insert(~"llvm.trunc.f64", truncf64);
    intrinsics.insert(~"llvm.ctpop.i8", ctpop8);
    intrinsics.insert(~"llvm.ctpop.i16", ctpop16);
    intrinsics.insert(~"llvm.ctpop.i32", ctpop32);
    intrinsics.insert(~"llvm.ctpop.i64", ctpop64);
    intrinsics.insert(~"llvm.ctlz.i8", ctlz8);
    intrinsics.insert(~"llvm.ctlz.i16", ctlz16);
    intrinsics.insert(~"llvm.ctlz.i32", ctlz32);
    intrinsics.insert(~"llvm.ctlz.i64", ctlz64);
    intrinsics.insert(~"llvm.cttz.i8", cttz8);
    intrinsics.insert(~"llvm.cttz.i16", cttz16);
    intrinsics.insert(~"llvm.cttz.i32", cttz32);
    intrinsics.insert(~"llvm.cttz.i64", cttz64);
    intrinsics.insert(~"llvm.bswap.i16", bswap16);
    intrinsics.insert(~"llvm.bswap.i32", bswap32);
    intrinsics.insert(~"llvm.bswap.i64", bswap64);

    return intrinsics;
}

fn declare_dbg_intrinsics(llmod: ModuleRef,
                          intrinsics: HashMap<~str, ValueRef>) {
    let declare =
        decl_cdecl_fn(llmod, ~"llvm.dbg.declare",
                      T_fn(~[T_metadata(), T_metadata()], T_void()));
    let value =
        decl_cdecl_fn(llmod, ~"llvm.dbg.value",
                      T_fn(~[T_metadata(), T_i64(), T_metadata()],
                           T_void()));
    intrinsics.insert(~"llvm.dbg.declare", declare);
    intrinsics.insert(~"llvm.dbg.value", value);
}

fn trap(bcx: block) {
    let v: ~[ValueRef] = ~[];
    match bcx.ccx().intrinsics.find(~"llvm.trap") {
      Some(x) => { Call(bcx, x, v); },
      _ => bcx.sess().bug(~"unbound llvm.trap in trap")
    }
}

fn decl_gc_metadata(ccx: @crate_ctxt, llmod_id: ~str) {
    if !ccx.sess.opts.gc || !ccx.uses_gc {
        return;
    }

    let gc_metadata_name = ~"_gc_module_metadata_" + llmod_id;
    let gc_metadata = do str::as_c_str(gc_metadata_name) |buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod, T_i32(), buf)
        }
    };
    unsafe {
        llvm::LLVMSetGlobalConstant(gc_metadata, True);
        lib::llvm::SetLinkage(gc_metadata, lib::llvm::ExternalLinkage);
        ccx.module_data.insert(~"_gc_module_metadata", gc_metadata);
    }
}

fn create_module_map(ccx: @crate_ctxt) -> ValueRef {
    let elttype = T_struct(~[ccx.int_type, ccx.int_type]);
    let maptype = T_array(elttype, ccx.module_data.size() + 1u);
    let map = str::as_c_str(~"_rust_mod_map", |buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod, maptype, buf)
        }
    });
    unsafe {
        lib::llvm::SetLinkage(map, lib::llvm::InternalLinkage);
    }
    let mut elts: ~[ValueRef] = ~[];
    for ccx.module_data.each |key, val| {
        let elt = C_struct(~[p2i(ccx, C_cstr(ccx, key)),
                            p2i(ccx, val)]);
        elts.push(elt);
    }
    let term = C_struct(~[C_int(ccx, 0), C_int(ccx, 0)]);
    elts.push(term);
    unsafe {
        llvm::LLVMSetInitializer(map, C_array(elttype, elts));
    }
    return map;
}


fn decl_crate_map(sess: session::Session, mapmeta: link_meta,
                  llmod: ModuleRef) -> ValueRef {
    let targ_cfg = sess.targ_cfg;
    let int_type = T_int(targ_cfg);
    let mut n_subcrates = 1;
    let cstore = sess.cstore;
    while cstore::have_crate_data(cstore, n_subcrates) { n_subcrates += 1; }
    let mapname = if sess.building_library {
        mapmeta.name.to_owned() + ~"_" + mapmeta.vers.to_owned() + ~"_"
            + mapmeta.extras_hash.to_owned()
    } else { ~"toplevel" };
    let sym_name = ~"_rust_crate_map_" + mapname;
    let arrtype = T_array(int_type, n_subcrates as uint);
    let maptype = T_struct(~[T_i32(), T_ptr(T_i8()), int_type, arrtype]);
    let map = str::as_c_str(sym_name, |buf| {
        unsafe {
            llvm::LLVMAddGlobal(llmod, maptype, buf)
        }
    });
    lib::llvm::SetLinkage(map, lib::llvm::ExternalLinkage);
    return map;
}

fn fill_crate_map(ccx: @crate_ctxt, map: ValueRef) {
    let mut subcrates: ~[ValueRef] = ~[];
    let mut i = 1;
    let cstore = ccx.sess.cstore;
    while cstore::have_crate_data(cstore, i) {
        let cdata = cstore::get_crate_data(cstore, i);
        let nm = ~"_rust_crate_map_" + cdata.name +
            ~"_" + cstore::get_crate_vers(cstore, i) +
            ~"_" + cstore::get_crate_hash(cstore, i);
        let cr = str::as_c_str(nm, |buf| {
            unsafe {
                llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
            }
        });
        subcrates.push(p2i(ccx, cr));
        i += 1;
    }
    subcrates.push(C_int(ccx, 0));

    let llannihilatefn;
    let annihilate_def_id = ccx.tcx.lang_items.annihilate_fn();
    if annihilate_def_id.crate == ast::local_crate {
        llannihilatefn = get_item_val(ccx, annihilate_def_id.node);
    } else {
        let annihilate_fn_type = csearch::get_type(ccx.tcx,
                                                   annihilate_def_id).ty;
        llannihilatefn = trans_external_path(ccx,
                                             annihilate_def_id,
                                             annihilate_fn_type);
    }

    unsafe {
        llvm::LLVMSetInitializer(map, C_struct(
            ~[C_i32(1),
              lib::llvm::llvm::LLVMConstPointerCast(llannihilatefn,
                                                    T_ptr(T_i8())),
              p2i(ccx, create_module_map(ccx)),
              C_array(ccx.int_type, subcrates)]));
    }
}

fn crate_ctxt_to_encode_parms(cx: @crate_ctxt) -> encoder::encode_parms {
    let encode_inlined_item: encoder::encode_inlined_item =
        |ecx, ebml_w, path, ii|
        astencode::encode_inlined_item(ecx, ebml_w, path, ii, cx.maps);

    return {
        diag: cx.sess.diagnostic(),
        tcx: cx.tcx,
        reachable: cx.reachable,
        reexports2: cx.exp_map2,
        item_symbols: cx.item_symbols,
        discrim_symbols: cx.discrim_symbols,
        link_meta: /*bad*/copy cx.link_meta,
        cstore: cx.sess.cstore,
        encode_inlined_item: encode_inlined_item
    };
}

fn write_metadata(cx: @crate_ctxt, crate: &ast::crate) {
    if !cx.sess.building_library { return; }
    let encode_parms = crate_ctxt_to_encode_parms(cx);
    let llmeta = C_bytes(encoder::encode_metadata(encode_parms, crate));
    let llconst = C_struct(~[llmeta]);
    let mut llglobal = str::as_c_str(~"rust_metadata", |buf| {
        unsafe {
            llvm::LLVMAddGlobal(cx.llmod, val_ty(llconst), buf)
        }
    });
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        str::as_c_str(cx.sess.targ_cfg.target_strs.meta_sect_name, |buf| {
            llvm::LLVMSetSection(llglobal, buf)
        });
        lib::llvm::SetLinkage(llglobal, lib::llvm::InternalLinkage);

        let t_ptr_i8 = T_ptr(T_i8());
        llglobal = llvm::LLVMConstBitCast(llglobal, t_ptr_i8);
        let llvm_used = str::as_c_str(~"llvm.used", |buf| {
            llvm::LLVMAddGlobal(cx.llmod, T_array(t_ptr_i8, 1u), buf)
        });
        lib::llvm::SetLinkage(llvm_used, lib::llvm::AppendingLinkage);
        llvm::LLVMSetInitializer(llvm_used, C_array(t_ptr_i8, ~[llglobal]));
    }
}

// Writes the current ABI version into the crate.
fn write_abi_version(ccx: @crate_ctxt) {
    mk_global(ccx, ~"rust_abi_version", C_uint(ccx, abi::abi_version),
                     false);
}

fn trans_crate(sess: session::Session,
               crate: @ast::crate,
               tcx: ty::ctxt,
               output: &Path,
               emap2: resolve::ExportMap2,
               maps: astencode::maps) -> (ModuleRef, link_meta) {

    let symbol_hasher = @hash::default_state();
    let link_meta =
        link::build_link_meta(sess, crate, output, symbol_hasher);
    let reachable = reachable::find_reachable(crate.node.module, emap2, tcx,
                                              maps.method_map);

    // Append ".rc" to crate name as LLVM module identifier.
    //
    // LLVM code generator emits a ".file filename" directive
    // for ELF backends. Value of the "filename" is set as the
    // LLVM module identifier.  Due to a LLVM MC bug[1], LLVM
    // crashes if the module identifer is same as other symbols
    // such as a function name in the module.
    // 1. http://llvm.org/bugs/show_bug.cgi?id=11479
    let llmod_id = link_meta.name.to_owned() + ~".rc";

    unsafe {
        let llmod = str::as_c_str(llmod_id, |buf| {
            llvm::LLVMModuleCreateWithNameInContext
                (buf, llvm::LLVMGetGlobalContext())
        });
        let data_layout = /*bad*/copy sess.targ_cfg.target_strs.data_layout;
        let targ_triple = /*bad*/copy sess.targ_cfg.target_strs.target_triple;
        let _: () =
            str::as_c_str(data_layout,
                        |buf| llvm::LLVMSetDataLayout(llmod, buf));
        let _: () =
            str::as_c_str(targ_triple,
                        |buf| llvm::LLVMSetTarget(llmod, buf));
        let targ_cfg = sess.targ_cfg;
        let td = mk_target_data(
            /*bad*/copy sess.targ_cfg.target_strs.data_layout);
        let tn = mk_type_names();
        let intrinsics = declare_intrinsics(llmod);
        if sess.opts.extra_debuginfo {
            declare_dbg_intrinsics(llmod, intrinsics);
        }
        let int_type = T_int(targ_cfg);
        let float_type = T_float(targ_cfg);
        let task_type = T_task(targ_cfg);
        let taskptr_type = T_ptr(task_type);
        lib::llvm::associate_type(tn, @"taskptr", taskptr_type);
        let tydesc_type = T_tydesc(targ_cfg);
        lib::llvm::associate_type(tn, @"tydesc", tydesc_type);
        let crate_map = decl_crate_map(sess, link_meta, llmod);
        let dbg_cx = if sess.opts.debuginfo {
            Some(debuginfo::mk_ctxt(copy llmod_id, sess.parse_sess.interner))
        } else {
            None
        };

        let ccx = @crate_ctxt {
              sess: sess,
              llmod: llmod,
              td: td,
              tn: tn,
              externs: HashMap(),
              intrinsics: intrinsics,
              item_vals: HashMap(),
              exp_map2: emap2,
              reachable: reachable,
              item_symbols: HashMap(),
              link_meta: link_meta,
              enum_sizes: ty::new_ty_hash(),
              discrims: HashMap(),
              discrim_symbols: HashMap(),
              tydescs: ty::new_ty_hash(),
              mut finished_tydescs: false,
              external: HashMap(),
              monomorphized: HashMap(),
              monomorphizing: HashMap(),
              type_use_cache: HashMap(),
              vtables: map::HashMap(),
              const_cstr_cache: HashMap(),
              const_globals: HashMap(),
              const_values: HashMap(),
              module_data: HashMap(),
              lltypes: ty::new_ty_hash(),
              names: new_namegen(sess.parse_sess.interner),
              next_addrspace: new_addrspace_gen(),
              symbol_hasher: symbol_hasher,
              type_hashcodes: ty::new_ty_hash(),
              type_short_names: ty::new_ty_hash(),
              all_llvm_symbols: HashMap(),
              tcx: tcx,
              maps: maps,
              stats:
                  {mut n_static_tydescs: 0u,
                   mut n_glues_created: 0u,
                   mut n_null_glues: 0u,
                   mut n_real_glues: 0u,
                   mut n_fns: 0u,
                   mut n_monos: 0u,
                   mut n_inlines: 0u,
                   mut n_closures: 0u,
                   llvm_insn_ctxt: @mut ~[],
                   llvm_insns: HashMap(),
                   fn_times: @mut ~[]},
              upcalls: upcall::declare_upcalls(targ_cfg, llmod),
              tydesc_type: tydesc_type,
              int_type: int_type,
              float_type: float_type,
              task_type: task_type,
              opaque_vec_type: T_opaque_vec(targ_cfg),
              builder: BuilderRef_res(unsafe { llvm::LLVMCreateBuilder() }),
              shape_cx: mk_ctxt(llmod),
              crate_map: crate_map,
              mut uses_gc: false,
              dbg_cx: dbg_cx,
              mut do_not_commit_warning_issued: false
        };

        {
            let _icx = ccx.insn_ctxt("data");
            trans_constants(ccx, crate);
        }

        {
            let _icx = ccx.insn_ctxt("text");
            trans_mod(ccx, crate.node.module);
        }

        decl_gc_metadata(ccx, llmod_id);
        fill_crate_map(ccx, crate_map);
        glue::emit_tydescs(ccx);
        write_abi_version(ccx);

        // Translate the metadata.
        write_metadata(ccx, crate);
        if ccx.sess.trans_stats() {
            io::println(~"--- trans stats ---");
            io::println(fmt!("n_static_tydescs: %u",
                             ccx.stats.n_static_tydescs));
            io::println(fmt!("n_glues_created: %u",
                             ccx.stats.n_glues_created));
            io::println(fmt!("n_null_glues: %u", ccx.stats.n_null_glues));
            io::println(fmt!("n_real_glues: %u", ccx.stats.n_real_glues));

            io::println(fmt!("n_fns: %u", ccx.stats.n_fns));
            io::println(fmt!("n_monos: %u", ccx.stats.n_monos));
            io::println(fmt!("n_inlines: %u", ccx.stats.n_inlines));
            io::println(fmt!("n_closures: %u", ccx.stats.n_closures));
        }

        if ccx.sess.count_llvm_insns() {
            for ccx.stats.llvm_insns.each |k, v| {
                io::println(fmt!("%-7u %s", v, k));
            }
        }
        return (llmod, link_meta);
    }
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
