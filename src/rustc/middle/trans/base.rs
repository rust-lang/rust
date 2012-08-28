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

use libc::{c_uint, c_ulonglong};
use std::{map, time, list};
use std::map::hashmap;
use std::map::{int_hash, str_hash};
use driver::session;
use session::session;
use syntax::attr;
use back::{link, abi, upcall};
use syntax::{ast, ast_util, codemap, ast_map};
use ast_util::{local_def, path_to_ident};
use syntax::visit;
use syntax::codemap::span;
use syntax::print::pprust::{expr_to_str, stmt_to_str, path_to_str};
use pat_util::*;
use visit::vt;
use util::common::is_main_name;
use lib::llvm::{llvm, mk_target_data, mk_type_names};
use lib::llvm::{ModuleRef, ValueRef, TypeRef, BasicBlockRef};
use lib::llvm::{True, False};
use link::{mangle_internal_name_by_type_only,
              mangle_internal_name_by_seq,
              mangle_internal_name_by_path,
              mangle_internal_name_by_path_and_seq,
              mangle_exported_name};
use metadata::{csearch, cstore, decoder, encoder};
use metadata::common::link_meta;
use util::ppaux;
use util::ppaux::{ty_to_str, ty_to_short_str};
use syntax::diagnostic::expect;
use util::common::indenter;

use build::*;
use shape::*;
use type_of::*;
use common::*;
use syntax::ast_map::{path, path_mod, path_name};
use syntax::parse::token::special_idents;

use std::smallintmap;
use option::{is_none, is_some};

struct icx_popper {
    let ccx: @crate_ctxt;
    new(ccx: @crate_ctxt) { self.ccx = ccx; }
    drop {
      if self.ccx.sess.count_llvm_insns() {
          vec::pop(*(self.ccx.stats.llvm_insn_ctxt));
      }
    }
}

trait get_insn_ctxt {
    fn insn_ctxt(s: &str) -> icx_popper;
}

impl @crate_ctxt: get_insn_ctxt {
    fn insn_ctxt(s: &str) -> icx_popper {
        debug!("new insn_ctxt: %s", s);
        if self.sess.count_llvm_insns() {
            vec::push(*self.stats.llvm_insn_ctxt, str::from_slice(s));
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

fn log_fn_time(ccx: @crate_ctxt, name: ~str, start: time::Timespec,
               end: time::Timespec) {
    let elapsed = 1000 * ((end.sec - start.sec) as int) +
        ((end.nsec as int) - (start.nsec as int)) / 1000000;
    vec::push(*ccx.stats.fn_times, {ident: name, time: elapsed});
}

fn decl_fn(llmod: ModuleRef, name: ~str, cc: lib::llvm::CallConv,
           llty: TypeRef) -> ValueRef {
    let llfn: ValueRef = str::as_c_str(name, |buf| {
        llvm::LLVMGetOrInsertFunction(llmod, buf, llty)
    });
    lib::llvm::SetFunctionCallConv(llfn, cc);
    return llfn;
}

fn decl_cdecl_fn(llmod: ModuleRef, name: ~str, llty: TypeRef) -> ValueRef {
    return decl_fn(llmod, name, lib::llvm::CCallConv, llty);
}

// Only use this if you are going to actually define the function. It's
// not valid to simply declare a function as internal.
fn decl_internal_cdecl_fn(llmod: ModuleRef, name: ~str, llty: TypeRef) ->
   ValueRef {
    let llfn = decl_cdecl_fn(llmod, name, llty);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    return llfn;
}

fn get_extern_fn(externs: hashmap<~str, ValueRef>,
                 llmod: ModuleRef, name: ~str,
                 cc: lib::llvm::CallConv, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { return externs.get(name); }
    let f = decl_fn(llmod, name, cc, ty);
    externs.insert(name, f);
    return f;
}

fn get_extern_const(externs: hashmap<~str, ValueRef>, llmod: ModuleRef,
                    name: ~str, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { return externs.get(name); }
    let c = str::as_c_str(name, |buf| llvm::LLVMAddGlobal(llmod, ty, buf));
    externs.insert(name, c);
    return c;
}

fn get_simple_extern_fn(cx: block,
                        externs: hashmap<~str, ValueRef>,
                        llmod: ModuleRef,
                        name: ~str, n_args: int) -> ValueRef {
    let _icx = cx.insn_ctxt("get_simple_extern_fn");
    let ccx = cx.fcx.ccx;
    let inputs = vec::from_elem(n_args as uint, ccx.int_type);
    let output = ccx.int_type;
    let t = T_fn(inputs, output);
    return get_extern_fn(externs, llmod, name, lib::llvm::CCallConv, t);
}

fn trans_foreign_call(cx: block, externs: hashmap<~str, ValueRef>,
                      llmod: ModuleRef, name: ~str, args: ~[ValueRef]) ->
   ValueRef {
    let _icx = cx.insn_ctxt("trans_foreign_call");
    let n = args.len() as int;
    let llforeign: ValueRef =
        get_simple_extern_fn(cx, externs, llmod, name, n);
    let mut call_args: ~[ValueRef] = ~[];
    for vec::each(args) |a| {
        vec::push(call_args, a);
    }
    return Call(cx, llforeign, call_args);
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
        type_of(ccx, ty::subst_tps(ccx.tcx, ty_substs, aty))
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
fn malloc_raw_dyn(bcx: block, t: ty::t, heap: heap,
                  size: ValueRef) -> Result {
    let _icx = bcx.insn_ctxt("malloc_raw");
    let ccx = bcx.ccx();

    let (mk_fn, rtcall) = match heap {
      heap_shared => (ty::mk_imm_box, ~"malloc"),
      heap_exchange => (ty::mk_imm_uniq, ~"exchange_malloc")
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
    let bcx = callee::trans_rtcall(bcx, rtcall, ~[tydesc, size],
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
    debug!("non_gc_box_cast");
    add_comment(bcx, ~"non_gc_box_cast");
    assert(llvm::LLVMGetPointerAddressSpace(val_ty(val)) == gc_box_addrspace);
    let non_gc_t = T_ptr(llvm::LLVMGetElementType(val_ty(val)));
    PointerCast(bcx, val, non_gc_t)
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
    llvm::LLVMAddFunctionAttr(f, lib::llvm::NoInlineAttribute as c_ulonglong,
                              0u as c_ulonglong);
}

fn set_no_unwind(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::NoUnwindAttribute as c_ulonglong,
                              0u as c_ulonglong);
}

// Tell LLVM to emit the information necessary to unwind the stack for the
// function f.
fn set_uwtable(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::UWTableAttribute as c_ulonglong,
                              0u as c_ulonglong);
}

fn set_inline_hint(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::InlineHintAttribute
                              as c_ulonglong, 0u as c_ulonglong);
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
    llvm::LLVMAddFunctionAttr(f, lib::llvm::AlwaysInlineAttribute
                              as c_ulonglong, 0u as c_ulonglong);
}

fn set_custom_stack_growth_fn(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, 0u as c_ulonglong, 1u as c_ulonglong);
}

fn set_glue_inlining(f: ValueRef, t: ty::t) {
    if ty::type_is_structural(t) {
        set_no_inline(f);
    } else { set_always_inline(f); }
}

// Double-check that we never ask LLVM to declare the same symbol twice. It
// silently mangles such symbols, breaking our linkage model.
fn note_unique_llvm_symbol(ccx: @crate_ctxt, sym: ~str) {
    if ccx.all_llvm_symbols.contains_key(sym) {
        ccx.sess.bug(~"duplicate LLVM symbol: " + sym);
    }
    ccx.all_llvm_symbols.insert(sym, ());
}


fn get_res_dtor(ccx: @crate_ctxt, did: ast::def_id,
                parent_id: ast::def_id, substs: ~[ty::t])
   -> ValueRef {
    let _icx = ccx.insn_ctxt("trans_res_dtor");
    if (substs.len() > 0u) {
        let did = if did.crate != ast::local_crate {
            inline::maybe_instantiate_inline(ccx, did)
        } else { did };
        assert did.crate == ast::local_crate;
        monomorphize::monomorphic_fn(ccx, did, substs, None, None).val
    } else if did.crate == ast::local_crate {
        get_item_val(ccx, did.node)
    } else {
        let tcx = ccx.tcx;
        let name = csearch::get_symbol(ccx.sess.cstore, did);
        let class_ty = ty::subst_tps(tcx, substs,
                          ty::lookup_item_type(tcx, parent_id).ty);
        let llty = type_of_dtor(ccx, class_ty);
        get_extern_fn(ccx.externs, ccx.llmod, name, lib::llvm::CCallConv,
                      llty)
    }
}

// Structural comparison: a rather involved form of glue.
fn maybe_name_value(cx: @crate_ctxt, v: ValueRef, s: ~str) {
    if cx.sess.opts.save_temps {
        let _: () = str::as_c_str(s, |buf| llvm::LLVMSetValueName(v, buf));
    }
}


// Used only for creating scalar comparison glue.
enum scalar_type { nil_type, signed_int, unsigned_int, floating_point, }

fn compare_scalar_types(cx: block, lhs: ValueRef, rhs: ValueRef,
                        t: ty::t, op: ast::binop) -> Result {
    let f = |a| compare_scalar_values(cx, lhs, rhs, a, op);

    match ty::get(t).struct {
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
    fn die_(cx: block) -> ! {
        cx.tcx().sess.bug(~"compare_scalar_values: must be a\
          comparison operator");
    }
    let die = fn@() -> ! { die_(cx) };
    match nt {
      nil_type => {
        // We don't need to do actual comparisons for nil.
        // () == () holds but () < () does not.
        match op {
          ast::eq | ast::le | ast::ge => return C_bool(true),
          ast::ne | ast::lt | ast::gt => return C_bool(false),
          // refinements would be nice
          _ => die()
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
          _ => die()
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
          _ => die()
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
          _ => die()
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
                    variant: ty::variant_info,
                    tps: ~[ty::t], tid: ast::def_id,
                    f: val_and_ty_fn) -> block {
        let _icx = cx.insn_ctxt("iter_variant");
        if variant.args.len() == 0u { return cx; }
        let fn_ty = variant.ctor_ty;
        let ccx = cx.ccx();
        let mut cx = cx;
        match ty::get(fn_ty).struct {
          ty::ty_fn({inputs: args, _}) => {
            let mut j = 0u;
            let v_id = variant.id;
            for vec::each(args) |a| {
                let llfldp_a = GEP_enum(cx, a_tup, tid, v_id, tps, j);
                let ty_subst = ty::subst_tps(ccx.tcx, tps, a.ty);
                cx = f(cx, llfldp_a, ty_subst);
                j += 1u;
            }
          }
          _ => cx.tcx().sess.bug(~"iter_variant: not a function type")
        }
        return cx;
    }

    /*
    Typestate constraint that shows the unimpl case doesn't happen?
    */
    let mut cx = cx;
    match ty::get(t).struct {
      ty::ty_rec(*) | ty::ty_class(*) => {
          do expr::with_field_tys(cx.tcx(), t) |_has_dtor, field_tys| {
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
            cx = f(cx, llfld_a, arg);
        }
      }
      ty::ty_enum(tid, substs) => {
        let variants = ty::enum_variants(cx.tcx(), tid);
        let n_variants = (*variants).len();

        // Cast the enums to types we can GEP into.
        if n_variants == 1u {
            return iter_variant(cx, av, variants[0],
                             substs.tps, tid, f);
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
                iter_variant(variant_cx, llunion_a_ptr, variant,
                             substs.tps, tid, f);
            Br(variant_cx, next_cx.llbb);
        }
        return next_cx;
      }
      _ => cx.sess().unimpl(~"type in iter_structural_ty")
    }
    return cx;
}

fn trans_compare(cx: block, op: ast::binop, lhs: ValueRef,
                 _lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> Result {
    let _icx = cx.insn_ctxt("trans_compare");
    if ty::type_is_scalar(rhs_t) {
      let rs = compare_scalar_types(cx, lhs, rhs, rhs_t, op);
      return rslt(rs.bcx, rs.val);
    }

    // Determine the operation we need.
    let llop = {
        match op {
          ast::eq | ast::ne => C_u8(abi::cmp_glue_op_eq),
          ast::lt | ast::ge => C_u8(abi::cmp_glue_op_lt),
          ast::le | ast::gt => C_u8(abi::cmp_glue_op_le),
          _ => cx.tcx().sess.bug(~"trans_compare got non-comparison-op")
        }
    };

    let cmpval = glue::call_cmp_glue(cx, lhs, rhs, rhs_t, llop);

    // Invert the result if necessary.
    match op {
      ast::eq | ast::lt | ast::le => rslt(cx, cmpval),
      ast::ne | ast::ge | ast::gt => rslt(cx, Not(cx, cmpval)),
      _ => cx.tcx().sess.bug(~"trans_compare got non-comparison-op")
    }
}

fn cast_shift_expr_rhs(cx: block, op: ast::binop,
                       lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs,
                   |a,b| Trunc(cx, a, b),
                   |a,b| ZExt(cx, a, b))
}

fn cast_shift_const_rhs(op: ast::binop,
                        lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs,
                   llvm::LLVMConstTrunc, llvm::LLVMConstZExt)
}

fn cast_shift_rhs(op: ast::binop,
                  lhs: ValueRef, rhs: ValueRef,
                  trunc: fn(ValueRef, TypeRef) -> ValueRef,
                  zext: fn(ValueRef, TypeRef) -> ValueRef
                 ) -> ValueRef {
    // Shifts may have any size int on the rhs
    if ast_util::is_shift_binop(op) {
        let rhs_llty = val_ty(rhs);
        let lhs_llty = val_ty(lhs);
        let rhs_sz = llvm::LLVMGetIntTypeWidth(rhs_llty);
        let lhs_sz = llvm::LLVMGetIntTypeWidth(lhs_llty);
        if lhs_sz < rhs_sz {
            trunc(rhs, lhs_llty)
        } else if lhs_sz > rhs_sz {
            // FIXME (See discussion at #1570): If shifting by negative
            // values becomes not undefined then this is wrong.
            zext(rhs, lhs_llty)
        } else {
            rhs
        }
    } else {
        rhs
    }
}

fn fail_if_zero(cx: block, span: span, divmod: ast::binop,
                rhs: ValueRef, rhs_t: ty::t) -> block {
    let text = if divmod == ast::div {
        ~"divide by zero"
    } else {
        ~"modulo zero"
    };
    let is_zero = match ty::get(rhs_t).struct {
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
        controlflow::trans_fail(bcx, Some(span), text)
    }
}

fn null_env_ptr(bcx: block) -> ValueRef {
    C_null(T_opaque_box_ptr(bcx.ccx()))
}

fn trans_external_path(ccx: @crate_ctxt, did: ast::def_id, t: ty::t)
    -> ValueRef {
    let name = csearch::get_symbol(ccx.sess.cstore, did);
    match ty::get(t).struct {
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

fn lookup_discriminant(ccx: @crate_ctxt, vid: ast::def_id) -> ValueRef {
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

fn invoke(bcx: block, llfn: ValueRef, llargs: ~[ValueRef]) -> block {
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
          block_scope(inf) => {
            for vec::each(inf.cleanups) |cleanup| {
                match cleanup {
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
          block_scope(inf) => {
            if inf.cleanups.len() > 0u || is_none(bcx.parent) {
                f(inf); return;
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
fn add_root_cleanup(bcx: block, scope_id: ast::node_id,
                    root_loc: ValueRef, ty: ty::t) {

    debug!("add_root_cleanup(bcx=%s, scope_id=%d, root_loc=%s, ty=%s)",
           bcx.to_str(), scope_id, val_str(bcx.ccx().tn, root_loc),
           ppaux::ty_to_str(bcx.ccx().tcx, ty));

    let bcx_scope = find_bcx_for_scope(bcx, scope_id);
    add_clean_temp_mem(bcx_scope, root_loc, ty);

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

fn trans_trace(bcx: block, sp_opt: Option<span>, trace_str: ~str) {
    if !bcx.sess().trace() { return; }
    let _icx = bcx.insn_ctxt("trans_trace");
    add_comment(bcx, trace_str);
    let V_trace_str = C_cstr(bcx.ccx(), trace_str);
    let {V_filename, V_line} = match sp_opt {
      Some(sp) => {
        let sess = bcx.sess();
        let loc = codemap::lookup_char_pos(sess.parse_sess.cm, sp.lo);
        {V_filename: C_cstr(bcx.ccx(), loc.file.name),
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
              return expr::trans_into(bcx, init.expr, expr::Ignore);
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
            if init.op == ast::init_assign || !bcx.expr_is_lval(init.expr) {
                bcx = expr::trans_into(bcx, init.expr, expr::SaveIn(llptr));
            } else { // This is a move from an lval, perform an actual move
                let init_datumblock = expr::trans_to_datum(bcx, init.expr);
                bcx = init_datumblock.move_to(datum::INIT, llptr);
            }
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

    return alt::bind_irrefutable_pat(bcx, local.node.pat, llptr, false);
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
            match d.node {
                ast::decl_local(locals) => {
                    for vec::each(locals) |local| {
                        bcx = init_local(bcx, local);
                        if cx.sess().opts.extra_debuginfo {
                            debuginfo::create_local_var(bcx, local);
                        }
                    }
                }
                ast::decl_item(i) => trans_item(cx.fcx.ccx, *i)
            }
        }
    }

    return bcx;
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block(cx: fn_ctxt, parent: Option<block>, +kind: block_kind,
             is_lpad: bool, name: ~str, opt_node_info: Option<node_info>)
    -> block {

    let s = if cx.ccx.sess.opts.save_temps || cx.ccx.sess.opts.debuginfo {
        cx.ccx.names(name)
    } else { special_idents::invalid };
    let llbb: BasicBlockRef = str::as_c_str(cx.ccx.sess.str_of(s), |buf| {
        llvm::LLVMAppendBasicBlock(cx.llfn, buf)
    });
    let bcx = mk_block(llbb, parent, kind, is_lpad, opt_node_info, cx);
    do option::iter(parent) |cx| {
        if cx.unreachable { Unreachable(bcx); }
    };
    return bcx;
}

fn simple_block_scope() -> block_kind {
    block_scope({loop_break: None, mut cleanups: ~[],
                 mut cleanup_paths: ~[], mut landing_pad: None})
}

// Use this when you're at the top block of a function or the like.
fn top_scope_block(fcx: fn_ctxt, opt_node_info: Option<node_info>) -> block {
    return new_block(fcx, None, simple_block_scope(), false,
                  ~"function top level", opt_node_info);
}

fn scope_block(bcx: block,
               opt_node_info: Option<node_info>,
               n: ~str) -> block {
    return new_block(bcx.fcx, Some(bcx), simple_block_scope(), bcx.is_lpad,
                  n, opt_node_info);
}

fn loop_scope_block(bcx: block, loop_break: block, n: ~str,
                    opt_node_info: Option<node_info>) -> block {
    return new_block(bcx.fcx, Some(bcx), block_scope({
        loop_break: Some(loop_break),
        mut cleanups: ~[],
        mut cleanup_paths: ~[],
        mut landing_pad: None
    }), bcx.is_lpad, n, opt_node_info);
}

// Use this when creating a block for the inside of a landing pad.
fn lpad_block(bcx: block, n: ~str) -> block {
    new_block(bcx.fcx, Some(bcx), block_non_scope, true, n, None)
}

// Use this when you're making a general CFG BB within a scope.
fn sub_block(bcx: block, n: ~str) -> block {
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
    do vec::riter(cleanups) |cu| {
            match cu {
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
fn cleanup_and_leave(bcx: block, upto: Option<BasicBlockRef>,
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
          block_scope(inf) if inf.cleanups.len() > 0u => {
            for vec::find(inf.cleanup_paths,
                          |cp| cp.target == leave).each |cp| {
                Br(bcx, cp.dest);
                return;
            }
            let sub_cx = sub_block(bcx, ~"cleanup");
            Br(bcx, sub_cx.llbb);
            vec::push(inf.cleanup_paths, {target: leave, dest: sub_cx.llbb});
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
          None => { assert is_none(upto); break; }
        };
    }
    match leave {
      Some(target) => Br(bcx, target),
      None => { Resume(bcx, Load(bcx, option::get(bcx.fcx.personality))); }
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
              name: ~str, f: fn(block) -> block) -> block {
    let _icx = bcx.insn_ctxt("with_scope");

    debug!("with_scope(bcx=%s, opt_node_info=%?, name=%s)",
           bcx.to_str(), opt_node_info, name);
    let _indenter = indenter();

    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    leave_block(f(scope_cx), scope_cx)
}

fn with_scope_result(bcx: block, opt_node_info: Option<node_info>,
                     name: ~str, f: fn(block) -> Result)
    -> Result
{
    let _icx = bcx.insn_ctxt("with_scope_result");
    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    let Result {bcx, val} = f(scope_cx);
    rslt(leave_block(bcx, scope_cx), val)
}

fn with_scope_datumblock(bcx: block, opt_node_info: Option<node_info>,
                         name: ~str, f: fn(block) -> datum::DatumBlock)
    -> datum::DatumBlock
{
    import datum::DatumBlock;

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
            match d.node {
              ast::decl_local(locals) => {
                for vec::each(locals) |local| { it(local); }
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
        do option::iter(simple_name) |name| {
            str::as_c_str(cx.ccx().sess.str_of(name), |buf| {
                llvm::LLVMSetValueName(val, buf)
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

fn call_memmove(cx: block, dst: ValueRef, src: ValueRef,
                n_bytes: ValueRef) {
    // FIXME (Related to #1645, I think?): Provide LLVM with better
    // alignment information when the alignment is statically known (it must
    // be nothing more than a constant int, or LLVM complains -- not even a
    // constant element of a tydesc works).
    let _icx = cx.insn_ctxt("call_memmove");
    let ccx = cx.ccx();
    let key = match ccx.sess.targ_cfg.arch {
      session::arch_x86 | session::arch_arm => ~"llvm.memmove.p0i8.p0i8.i32",
      session::arch_x86_64 => ~"llvm.memmove.p0i8.p0i8.i64"
    };
    let memmove = ccx.intrinsics.get(key);
    let src_ptr = PointerCast(cx, src, T_ptr(T_i8()));
    let dst_ptr = PointerCast(cx, dst, T_ptr(T_i8()));
    let size = IntCast(cx, n_bytes, ccx.int_type);
    let align = C_i32(1i32);
    let volatile = C_bool(false);
    Call(cx, memmove, ~[dst_ptr, src_ptr, size, align, volatile]);
}

fn memmove_ty(bcx: block, dst: ValueRef, src: ValueRef, t: ty::t) {
    let _icx = bcx.insn_ctxt("memmove_ty");
    let ccx = bcx.ccx();
    if ty::type_is_structural(t) {
        let llsz = llsize_of(ccx, type_of::type_of(ccx, t));
        call_memmove(bcx, dst, src, llsz);
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
    if cx.unreachable { return llvm::LLVMGetUndef(t); }
    let initcx = base::raw_block(cx.fcx, false, cx.fcx.llstaticallocas);
    let p = Alloca(initcx, t);
    if zero { memzero(initcx, p, t); }
    return p;
}

fn arrayalloca(cx: block, t: TypeRef, v: ValueRef) -> ValueRef {
    let _icx = cx.insn_ctxt("arrayalloca");
    if cx.unreachable { return llvm::LLVMGetUndef(t); }
    return ArrayAlloca(
        base::raw_block(cx.fcx, false, cx.fcx.llstaticallocas), t, v);
}

// Creates the standard set of basic blocks for a function
fn mk_standard_basic_blocks(llfn: ValueRef) ->
   {sa: BasicBlockRef, ca: BasicBlockRef, rt: BasicBlockRef} {
    {sa: str::as_c_str(~"static_allocas",
                       |buf| llvm::LLVMAppendBasicBlock(llfn, buf)),
     ca: str::as_c_str(~"load_env",
                       |buf| llvm::LLVMAppendBasicBlock(llfn, buf)),
     rt: str::as_c_str(~"return",
                       |buf| llvm::LLVMAppendBasicBlock(llfn, buf))}
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn new_fn_ctxt_w_id(ccx: @crate_ctxt, path: path,
                    llfndecl: ValueRef, id: ast::node_id,
                    param_substs: Option<param_substs>,
                    sp: Option<span>) -> fn_ctxt {
    let llbbs = mk_standard_basic_blocks(llfndecl);
    return @{llfn: llfndecl,
          llenv: llvm::LLVMGetParam(llfndecl, 1u as c_uint),
          llretptr: llvm::LLVMGetParam(llfndecl, 0u as c_uint),
          mut llstaticallocas: llbbs.sa,
          mut llloadenv: llbbs.ca,
          mut llreturn: llbbs.rt,
          mut llself: None,
          mut personality: None,
          mut loop_ret: None,
          llargs: int_hash::<local_val>(),
          lllocals: int_hash::<local_val>(),
          llupvars: int_hash::<ValueRef>(),
          id: id,
          param_substs: param_substs,
          span: sp,
          path: path,
          ccx: ccx};
}

fn new_fn_ctxt(ccx: @crate_ctxt, path: path, llfndecl: ValueRef,
               sp: Option<span>) -> fn_ctxt {
    return new_fn_ctxt_w_id(ccx, path, llfndecl, -1, None, sp);
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
                             args: ~[ast::arg]) {
    let _icx = cx.insn_ctxt("create_llargs_for_fn_args");
    // Skip the implicit arguments 0, and 1.
    let mut arg_n = first_real_arg;
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

    // Populate the llargs field of the function context with the ValueRefs
    // that we get from llvm::LLVMGetParam for each argument.
    for vec::each(args) |arg| {
        let llarg = llvm::LLVMGetParam(cx.llfn, arg_n as c_uint);
        assert (llarg as int != 0);
        // Note that this uses local_mem even for things passed by value.
        // copy_args_to_allocas will overwrite the table entry with local_imm
        // before it's actually used.
        cx.llargs.insert(arg.id, local_mem(llarg));
        arg_n += 1u;
    }
}

fn copy_args_to_allocas(fcx: fn_ctxt, bcx: block, args: ~[ast::arg],
                        arg_tys: ~[ty::arg]) -> block {
    let _icx = fcx.insn_ctxt("copy_args_to_allocas");
    let tcx = bcx.tcx();
    let mut arg_n: uint = 0u, bcx = bcx;
    let epic_fail = fn@() -> ! {
        tcx.sess.bug(~"someone forgot\
                to document an invariant in copy_args_to_allocas!");
    };

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

    for vec::each(arg_tys) |arg| {
        let id = args[arg_n].id;
        let argval = match fcx.llargs.get(id) {
          local_mem(v) => v,
          _ => epic_fail()
        };
        match ty::resolved_mode(tcx, arg.mode) {
          ast::by_mutbl_ref => (),
          ast::by_move | ast::by_copy => add_clean(bcx, argval, arg.ty),
          ast::by_val => {
            if !ty::type_is_immediate(arg.ty) {
                let alloc = alloc_ty(bcx, arg.ty);
                Store(bcx, argval, alloc);
                fcx.llargs.insert(id, local_mem(alloc));
            } else {
                fcx.llargs.insert(id, local_imm(argval));
            }
          }
          ast::by_ref => ()
        }
        if fcx.ccx.sess.opts.extra_debuginfo {
            debuginfo::create_arg(bcx, args[arg_n], args[arg_n].ty.span);
        }
        arg_n += 1u;
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
    Br(raw_block(fcx, false, fcx.llstaticallocas), fcx.llloadenv);
    Br(raw_block(fcx, false, fcx.llloadenv), lltop);
}

enum self_arg { impl_self(ty::t), impl_owned_self(ty::t), no_self, }

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
fn trans_closure(ccx: @crate_ctxt, path: path, decl: ast::fn_decl,
                 body: ast::blk, llfndecl: ValueRef,
                 ty_self: self_arg,
                 param_substs: Option<param_substs>,
                 id: ast::node_id,
                 maybe_load_env: fn(fn_ctxt),
                 finish: fn(block)) {
    let _icx = ccx.insn_ctxt("trans_closure");
    set_uwtable(llfndecl);

    // Set up arguments to the function.
    let fcx = new_fn_ctxt_w_id(ccx, path, llfndecl, id, param_substs,
                                  Some(body.span));
    create_llargs_for_fn_args(fcx, ty_self, decl.inputs);

    // Set GC for function.
    if ccx.sess.opts.gc {
        do str::as_c_str("generic") |strategy| {
            llvm::LLVMSetGC(fcx.llfn, strategy);
        }
    }

    // Create the first basic block in the function and keep a handle on it to
    //  pass to finish_fn later.
    let bcx_top = top_scope_block(fcx, body.info());
    let mut bcx = bcx_top;
    let lltop = bcx.llbb;
    let block_ty = node_id_type(bcx, body.node.id);

    let arg_tys = ty::ty_fn_args(node_id_type(bcx, id));
    bcx = copy_args_to_allocas(fcx, bcx, decl.inputs, arg_tys);

    maybe_load_env(fcx);

    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).

    if !ccx.class_ctors.contains_key(id) // hack --
       /* avoids the need for special cases to assign a type to
          the constructor body (since it has no explicit return) */
      &&
      (option::is_none(body.node.expr) ||
       ty::type_is_bot(block_ty) ||
       ty::type_is_nil(block_ty))  {
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
            path: path,
            decl: ast::fn_decl,
            body: ast::blk,
            llfndecl: ValueRef,
            ty_self: self_arg,
            param_substs: Option<param_substs>,
            id: ast::node_id) {
    let do_time = ccx.sess.trans_stats();
    let start = if do_time { time::get_time() }
                else { {sec: 0i64, nsec: 0i32} };
    let _icx = ccx.insn_ctxt("trans_fn");
    trans_closure(ccx, path, decl, body, llfndecl, ty_self,
                  param_substs, id,
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
                      disr: int, is_degen: bool,
                      param_substs: Option<param_substs>,
                      llfndecl: ValueRef) {
    let _icx = ccx.insn_ctxt("trans_enum_variant");
    // Translate variant arguments to function arguments.
    let fn_args = vec::map(args, |varg|
        {mode: ast::expl(ast::by_copy),
         ty: varg.ty,
         ident: special_idents::arg,
         id: varg.id});
    let fcx = new_fn_ctxt_w_id(ccx, ~[], llfndecl, variant.node.id,
                               param_substs, None);
    create_llargs_for_fn_args(fcx, no_self, fn_args);
    let ty_param_substs = match param_substs {
      Some(substs) => substs.tys,
      None => ~[]
    };
    let bcx = top_scope_block(fcx, None), lltop = bcx.llbb;
    let arg_tys = ty::ty_fn_args(node_id_type(bcx, variant.node.id));
    let bcx = copy_args_to_allocas(fcx, bcx, fn_args, arg_tys);

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
                                 ty_param_substs, i);
        // If this argument to this function is a enum, it'll have come in to
        // this function as an opaque blob due to the way that type_of()
        // works. So we have to cast to the destination's view of the type.
        let llarg = match fcx.llargs.find(va.id) {
            Some(local_mem(x)) => x,
            _ => fail ~"trans_enum_variant: how do we know this works?",
        };
        let arg_ty = arg_tys[i].ty;
        memmove_ty(bcx, lldestptr, llarg, arg_ty);
    }
    build_return(bcx);
    finish_fn(fcx, lltop);
}

fn trans_class_ctor(ccx: @crate_ctxt, path: path, decl: ast::fn_decl,
                    body: ast::blk, llctor_decl: ValueRef,
                    psubsts: param_substs, ctor_id: ast::node_id,
                    parent_id: ast::def_id, sp: span) {
    // Add ctor to the ctor map
    ccx.class_ctors.insert(ctor_id, parent_id);

    // Translate the ctor

    // Set up the type for the result of the ctor
    // kludgy -- this wouldn't be necessary if the typechecker
    // special-cased constructors, then we could just look up
    // the ctor's return type.
    let rslt_ty =  ty::mk_class(ccx.tcx, parent_id,
                                dummy_substs(psubsts.tys));

    // Make the fn context
    let fcx = new_fn_ctxt_w_id(ccx, path, llctor_decl, ctor_id,
                               Some(psubsts), Some(sp));
    create_llargs_for_fn_args(fcx, no_self, decl.inputs);
    let mut bcx_top = top_scope_block(fcx, body.info());
    let lltop = bcx_top.llbb;
    let arg_tys = ty::ty_fn_args(node_id_type(bcx_top, ctor_id));
    bcx_top = copy_args_to_allocas(fcx, bcx_top, decl.inputs, arg_tys);

    // Create a temporary for `self` that we will return at the end
    let selfdatum = datum::scratch_datum(bcx_top, rslt_ty, true);

    // Initialize dtor flag (if any) to 1
    if option::is_some(ty::ty_dtor(bcx_top.tcx(), parent_id)) {
        let flag = GEPi(bcx_top, selfdatum.val, [0, 1]);
        Store(bcx_top, C_u8(1), flag);
    }

    // initialize fields to zero
    let mut bcx = bcx_top;

    // note we don't want to take *or* drop self.
    fcx.llself = Some(ValSelfData {v: selfdatum.val,
                                   t: rslt_ty,
                                   is_owned: false});

    // Translate the body of the ctor
    bcx = controlflow::trans_block(bcx, body, expr::Ignore);

    // Generate the return expression
    bcx = selfdatum.move_to(bcx, datum::INIT, fcx.llretptr);

    cleanup_and_leave(bcx, None, Some(fcx.llreturn));
    Unreachable(bcx);
    finish_fn(fcx, lltop);
}

fn trans_class_dtor(ccx: @crate_ctxt, path: path,
    body: ast::blk, dtor_id: ast::node_id,
    psubsts: Option<param_substs>,
    hash_id: Option<mono_id>, parent_id: ast::def_id)
    -> ValueRef {
  let tcx = ccx.tcx;
  /* Look up the parent class's def_id */
  let mut class_ty = ty::lookup_item_type(tcx, parent_id).ty;
  /* Substitute in the class type if necessary */
    do option::iter(psubsts) |ss| {
    class_ty = ty::subst_tps(tcx, ss.tys, class_ty);
  }

  /* The dtor takes a (null) output pointer, and a self argument,
     and returns () */
  let lldty = T_fn(~[T_ptr(type_of(ccx, ty::mk_nil(tcx))),
                    T_ptr(type_of(ccx, class_ty))],
                   llvm::LLVMVoidType());

  let s = get_dtor_symbol(ccx, path, dtor_id, psubsts);

  /* Register the dtor as a function. It has external linkage */
  let lldecl = decl_internal_cdecl_fn(ccx.llmod, s, lldty);
  lib::llvm::SetLinkage(lldecl, lib::llvm::ExternalLinkage);

  /* If we're monomorphizing, register the monomorphized decl
     for the dtor */
    do option::iter(hash_id) |h_id| {
    ccx.monomorphized.insert(h_id, lldecl);
  }
  /* Translate the dtor body */
  trans_fn(ccx, path, ast_util::dtor_dec(),
           body, lldecl, impl_self(class_ty), psubsts, dtor_id);
  lldecl
}

fn trans_enum_def(ccx: @crate_ctxt, enum_definition: ast::enum_def,
                  id: ast::node_id, tps: ~[ast::ty_param], degen: bool,
                  path: @ast_map::path, vi: @~[ty::variant_info],
                  i: &mut uint) {
    for vec::each(enum_definition.variants) |variant| {
        let disr_val = vi[*i].disr_val;
        *i += 1;

        match variant.node.kind {
            ast::tuple_variant_kind(args) if args.len() > 0 => {
                let llfn = get_item_val(ccx, variant.node.id);
                trans_enum_variant(ccx, id, variant, args, disr_val,
                                   degen, None, llfn);
            }
            ast::tuple_variant_kind(_) => {
                // Nothing to do.
            }
            ast::struct_variant_kind(struct_def) => {
                trans_struct_def(ccx, struct_def, tps, path,
                                 variant.node.name, variant.node.id);
            }
            ast::enum_variant_kind(enum_definition) => {
                trans_enum_def(ccx, enum_definition, id, tps, degen, path, vi,
                               i);
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
    match item.node {
      ast::item_fn(decl, purity, tps, body) => {
        if purity == ast::extern_fn  {
            let llfndecl = get_item_val(ccx, item.id);
            foreign::trans_foreign_fn(ccx,
                                     vec::append(
                                         *path,
                                         ~[path_name(item.ident)]),
                                     decl, body, llfndecl, item.id);
        } else if tps.len() == 0u {
            let llfndecl = get_item_val(ccx, item.id);
            trans_fn(ccx,
                     vec::append(*path, ~[path_name(item.ident)]),
                     decl, body, llfndecl, no_self, None, item.id);
        } else {
            for vec::each(body.node.stmts) |stmt| {
                match stmt.node {
                  ast::stmt_decl(@{node: ast::decl_item(i), _}, _) => {
                    trans_item(ccx, *i);
                  }
                  _ => ()
                }
            }
        }
      }
      ast::item_impl(tps, _, _, ms) => {
        impl::trans_impl(ccx, *path, item.ident, ms, tps);
      }
      ast::item_mod(m) => {
        trans_mod(ccx, m);
      }
      ast::item_enum(enum_definition, tps) => {
        if tps.len() == 0u {
            let degen = enum_definition.variants.len() == 1u;
            let vi = ty::enum_variants(ccx.tcx, local_def(item.id));
            let mut i = 0;
            trans_enum_def(ccx, enum_definition, item.id, tps, degen, path,
                           vi, &mut i);
        }
      }
      ast::item_const(_, expr) => consts::trans_const(ccx, expr, item.id),
      ast::item_foreign_mod(foreign_mod) => {
        let abi = match attr::foreign_abi(item.attrs) {
          either::Right(abi_) => abi_,
          either::Left(msg) => ccx.sess.span_fatal(item.span, msg)
        };
        foreign::trans_foreign_mod(ccx, foreign_mod, abi);
      }
      ast::item_class(struct_def, tps) => {
        trans_struct_def(ccx, struct_def, tps, path, item.ident, item.id);
      }
      ast::item_trait(tps, _, trait_methods) => {
        trans_trait(ccx, tps, trait_methods, path, item.ident);
      }
      _ => {/* fall through */ }
    }
}

fn trans_struct_def(ccx: @crate_ctxt, struct_def: @ast::struct_def,
                    tps: ~[ast::ty_param], path: @ast_map::path,
                    ident: ast::ident, id: ast::node_id) {
    if tps.len() == 0u {
      let psubsts = {tys: ty::ty_params_to_tys(ccx.tcx, tps),
                     vtables: None,
                     bounds: @~[]};
      do option::iter(struct_def.ctor) |ctor| {
        trans_class_ctor(ccx, *path, ctor.node.dec, ctor.node.body,
                         get_item_val(ccx, ctor.node.id), psubsts,
                         ctor.node.id, local_def(id), ctor.span);
      }
      do option::iter(struct_def.dtor) |dtor| {
         trans_class_dtor(ccx, *path, dtor.node.body,
           dtor.node.id, None, None, local_def(id));
      };
    }
    // If there are ty params, the ctor will get monomorphized

    // Translate methods
    impl::trans_impl(ccx, *path, ident, struct_def.methods, tps);
}

fn trans_trait(ccx: @crate_ctxt, tps: ~[ast::ty_param],
               trait_methods: ~[ast::trait_method],
               path: @ast_map::path, ident: ast::ident) {
    // Translate any methods that have provided implementations
    let (_, provided_methods) = ast_util::split_trait_methods(trait_methods);
    impl::trans_impl(ccx, *path, ident, provided_methods, tps);
}

// Translate a module. Doing this amounts to translating the items in the
// module; there ends up being no artifact (aside from linkage names) of
// separate modules in the compiled program.  That's because modules exist
// only as a convenience for humans working with the code, to organize names
// and control visibility.
fn trans_mod(ccx: @crate_ctxt, m: ast::_mod) {
    let _icx = ccx.insn_ctxt("trans_mod");
    for vec::each(m.items) |item| { trans_item(ccx, *item); }
}

fn get_pair_fn_ty(llpairty: TypeRef) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the pair.
    return struct_elt(llpairty, 0u);
}

fn register_fn(ccx: @crate_ctxt, sp: span, path: path,
               node_id: ast::node_id) -> ValueRef {
    let t = ty::node_id_to_type(ccx.tcx, node_id);
    register_fn_full(ccx, sp, path, node_id, t)
}

fn register_fn_full(ccx: @crate_ctxt, sp: span, path: path,
                    node_id: ast::node_id, node_type: ty::t) -> ValueRef {
    let llfty = type_of_fn_from_ty(ccx, node_type);
    register_fn_fuller(ccx, sp, path, node_id, node_type,
                       lib::llvm::CCallConv, llfty)
}

fn register_fn_fuller(ccx: @crate_ctxt, sp: span, path: path,
                      node_id: ast::node_id, node_type: ty::t,
                      cc: lib::llvm::CallConv, llfty: TypeRef) -> ValueRef {
    let ps: ~str = mangle_exported_name(ccx, path, node_type);
    let llfn: ValueRef = decl_fn(ccx.llmod, ps, cc, llfty);
    ccx.item_symbols.insert(node_id, ps);

    debug!("register_fn_fuller created fn %s for item %d with path %s",
           val_str(ccx.tn, llfn), node_id,
           ast_map::path_to_str(path, ccx.sess.parse_sess.interner));

    let is_main = is_main_name(path) && !ccx.sess.building_library;
    if is_main { create_main_wrapper(ccx, sp, llfn, node_type); }
    llfn
}

// Create a _rust_main(args: ~[str]) function which will be called from the
// runtime rust_start function
fn create_main_wrapper(ccx: @crate_ctxt, sp: span, main_llfn: ValueRef,
                       main_node_type: ty::t) {

    if ccx.main_fn != None::<ValueRef> {
        ccx.sess.span_fatal(sp, ~"multiple 'main' functions");
    }

    let main_takes_argv =
        // invariant!
        match ty::get(main_node_type).struct {
          ty::ty_fn({inputs, _}) => inputs.len() != 0u,
          _ => ccx.sess.span_fatal(sp, ~"main has a non-function type")
        };

    let llfn = create_main(ccx, main_llfn, main_takes_argv);
    ccx.main_fn = Some(llfn);
    create_entry_fn(ccx, llfn);

    fn create_main(ccx: @crate_ctxt, main_llfn: ValueRef,
                   takes_argv: bool) -> ValueRef {
        let unit_ty = ty::mk_estr(ccx.tcx, ty::vstore_uniq);
        let vecarg_ty: ty::arg =
            {mode: ast::expl(ast::by_val),
             ty: ty::mk_evec(ccx.tcx, {ty: unit_ty, mutbl: ast::m_imm},
                             ty::vstore_uniq)};
        let nt = ty::mk_nil(ccx.tcx);
        let llfty = type_of_fn(ccx, ~[vecarg_ty], nt);
        let llfdecl = decl_fn(ccx.llmod, ~"_rust_main",
                              lib::llvm::CCallConv, llfty);

        let fcx = new_fn_ctxt(ccx, ~[], llfdecl, None);

        let bcx = top_scope_block(fcx, None);
        let lltop = bcx.llbb;

        let lloutputarg = llvm::LLVMGetParam(llfdecl, 0 as c_uint);
        let llenvarg = llvm::LLVMGetParam(llfdecl, 1 as c_uint);
        let mut args = ~[lloutputarg, llenvarg];
        if takes_argv {
            vec::push(args, llvm::LLVMGetParam(llfdecl, 2 as c_uint));
        }
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
        let llfn = decl_cdecl_fn(ccx.llmod, main_name(), llfty);
        let llbb = str::as_c_str(~"top", |buf| {
            llvm::LLVMAppendBasicBlock(llfn, buf)
        });
        let bld = ccx.builder.B;
        llvm::LLVMPositionBuilderAtEnd(bld, llbb);
        let crate_map = ccx.crate_map;
        let start_ty = T_fn(~[val_ty(rust_main), ccx.int_type, ccx.int_type,
                             val_ty(crate_map)], ccx.int_type);
        let start = decl_cdecl_fn(ccx.llmod, ~"rust_start", start_ty);

        let args = ~[rust_main, llvm::LLVMGetParam(llfn, 0 as c_uint),
                    llvm::LLVMGetParam(llfn, 1 as c_uint), crate_map];
        let result = unsafe {
            llvm::LLVMBuildCall(bld, start, vec::unsafe::to_ptr(args),
                                args.len() as c_uint, noname())
        };
        llvm::LLVMBuildRet(bld, result);
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
        *match ccx.tcx.items.get(i.id) {
            ast_map::node_item(_, p) => p,
                // separate map for paths?
            _ => fail ~"item_path"
        },
        ~[path_name(i.ident)])
}

/* If there's already a symbol for the dtor with <id> and substs <substs>,
   return it; otherwise, create one and register it, returning it as well */
fn get_dtor_symbol(ccx: @crate_ctxt, path: path, id: ast::node_id,
                   substs: Option<param_substs>) -> ~str {
  let t = ty::node_id_to_type(ccx.tcx, id);
  match ccx.item_symbols.find(id) {
     Some(s) => s,
     None if is_none(substs) => {
       let s = mangle_exported_name(
           ccx,
           vec::append(path, ~[path_name(ccx.names(~"dtor"))]),
           t);
       ccx.item_symbols.insert(id, s);
       s
     }
     None   => {
       // Monomorphizing, so just make a symbol, don't add
       // this to item_symbols
       match substs {
         Some(ss) => {
           let mono_ty = ty::subst_tps(ccx.tcx, ss.tys, t);
           mangle_exported_name(
               ccx,
               vec::append(path,
                           ~[path_name(ccx.names(~"dtor"))]),
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
            let my_path = vec::append(*pth, ~[path_name(i.ident)]);
            match i.node {
              ast::item_const(_, _) => {
                let typ = ty::node_id_to_type(ccx.tcx, i.id);
                let s = mangle_exported_name(ccx, my_path, typ);
                let g = str::as_c_str(s, |buf| {
                    llvm::LLVMAddGlobal(ccx.llmod, type_of(ccx, typ), buf)
                });
                ccx.item_symbols.insert(i.id, s);
                g
              }
              ast::item_fn(_, purity, _, _) => {
                let llfn = if purity != ast::extern_fn {
                    register_fn(ccx, i.span, my_path, i.id)
                } else {
                    foreign::register_foreign_fn(ccx, i.span, my_path, i.id)
                };
                set_inline_hint_if_appr(i.attrs, llfn);
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
                                vec::append(*pth, ~[path_name(ni.ident)]),
                                ni.id)
                }
                ast::foreign_item_const(*) => {
                    let typ = ty::node_id_to_type(ccx.tcx, ni.id);
                    let ident = ccx.sess.parse_sess.interner.get(ni.ident);
                    let g = do str::as_c_str(*ident) |buf| {
                        llvm::LLVMAddGlobal(ccx.llmod, type_of(ccx, typ), buf)
                    };
                    g
                }
            }
          }
          ast_map::node_ctor(nm, _, ctor, _, pt) => {
            let my_path = vec::append(*pt, ~[path_name(nm)]);
            register_fn(ccx, ctor.span, my_path, ctor.node.id)
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
            let lldty = T_fn(~[T_ptr(type_of(ccx, ty::mk_nil(tcx))),
                    T_ptr(type_of(ccx, class_ty))],
                                   llvm::LLVMVoidType());
            let s = get_dtor_symbol(ccx, *pt, dt.node.id, None);

            /* Make the declaration for the dtor */
            let llfn = decl_internal_cdecl_fn(ccx.llmod, s, lldty);
            lib::llvm::SetLinkage(llfn, lib::llvm::ExternalLinkage);
            llfn
          }

          ast_map::node_variant(v, enm, pth) => {
            let llfn;
            match v.node.kind {
                ast::tuple_variant_kind(args) => {
                    assert args.len() != 0u;
                    let pth = vec::append(*pth,
                                          ~[path_name(enm.ident),
                                            path_name(v.node.name)]);
                    llfn = match enm.node {
                      ast::item_enum(_, _) => {
                        register_fn(ccx, v.span, pth, id)
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
          _ => {
            ccx.sess.bug(~"get_item_val(): unexpected variant");
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
    let pth = vec::append(*pth, ~[path_name(ccx.names(~"meth")),
                                  path_name(m.ident)]);
    let llfn = register_fn_full(ccx, m.span, pth, id, mty);
    set_inline_hint_if_appr(m.attrs, llfn);
    llfn
}

// The constant translation pass.
fn trans_constant(ccx: @crate_ctxt, it: @ast::item) {
    let _icx = ccx.insn_ctxt("trans_constant");
    match it.node {
      ast::item_enum(enum_definition, _) => {
        let vi = ty::enum_variants(ccx.tcx, {crate: ast::local_crate,
                                             node: it.id});
        let mut i = 0;
        let path = item_path(ccx, it);
        for vec::each(enum_definition.variants) |variant| {
            let p = vec::append(path, ~[path_name(variant.node.name),
                                        path_name(special_idents::descrim)]);
            let s = mangle_exported_name(ccx, p, ty::mk_int(ccx.tcx));
            let disr_val = vi[i].disr_val;
            note_unique_llvm_symbol(ccx, s);
            let discrim_gvar = str::as_c_str(s, |buf| {
                llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
            });
            llvm::LLVMSetInitializer(discrim_gvar, C_int(ccx, disr_val));
            llvm::LLVMSetGlobalConstant(discrim_gvar, True);
            ccx.discrims.insert(
                local_def(variant.node.id), discrim_gvar);
            ccx.discrim_symbols.insert(variant.node.id, s);
            i += 1;
        }
      }
      _ => ()
    }
}

fn trans_constants(ccx: @crate_ctxt, crate: @ast::crate) {
    visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
        visit_item: |a| trans_constant(ccx, a),
        ..*visit::default_simple_visitor()
    }));
}

fn vp2i(cx: block, v: ValueRef) -> ValueRef {
    let ccx = cx.ccx();
    return PtrToInt(cx, v, ccx.int_type);
}

fn p2i(ccx: @crate_ctxt, v: ValueRef) -> ValueRef {
    return llvm::LLVMConstPtrToInt(v, ccx.int_type);
}

fn declare_intrinsics(llmod: ModuleRef) -> hashmap<~str, ValueRef> {
    let T_memmove32_args: ~[TypeRef] =
        ~[T_ptr(T_i8()), T_ptr(T_i8()), T_i32(), T_i32(), T_i1()];
    let T_memmove64_args: ~[TypeRef] =
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
    let memmove32 =
        decl_cdecl_fn(llmod, ~"llvm.memmove.p0i8.p0i8.i32",
                      T_fn(T_memmove32_args, T_void()));
    let memmove64 =
        decl_cdecl_fn(llmod, ~"llvm.memmove.p0i8.p0i8.i64",
                      T_fn(T_memmove64_args, T_void()));
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
    let intrinsics = str_hash::<ValueRef>();
    intrinsics.insert(~"llvm.gcroot", gcroot);
    intrinsics.insert(~"llvm.gcread", gcread);
    intrinsics.insert(~"llvm.memmove.p0i8.p0i8.i32", memmove32);
    intrinsics.insert(~"llvm.memmove.p0i8.p0i8.i64", memmove64);
    intrinsics.insert(~"llvm.memset.p0i8.i32", memset32);
    intrinsics.insert(~"llvm.memset.p0i8.i64", memset64);
    intrinsics.insert(~"llvm.trap", trap);
    intrinsics.insert(~"llvm.frameaddress", frameaddress);
    return intrinsics;
}

fn declare_dbg_intrinsics(llmod: ModuleRef,
                          intrinsics: hashmap<~str, ValueRef>) {
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

fn push_rtcall(ccx: @crate_ctxt, name: ~str, did: ast::def_id) {
    if ccx.rtcalls.contains_key(name) {
        fail fmt!("multiple definitions for runtime call %s", name);
    }
    ccx.rtcalls.insert(name, did);
}

fn gather_local_rtcalls(ccx: @crate_ctxt, crate: @ast::crate) {
    visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
        visit_item: |item| match item.node {
          ast::item_fn(*) => {
            let attr_metas = attr::attr_metas(
                attr::find_attrs_by_name(item.attrs, ~"rt"));
            do vec::iter(attr_metas) |attr_meta| {
                match attr::get_meta_item_list(attr_meta) {
                  Some(list) => {
                    let name = attr::get_meta_item_name(vec::head(list));
                    push_rtcall(ccx, name, {crate: ast::local_crate,
                                            node: item.id});
                  }
                  None => ()
                }
            }
          }
          _ => ()
        },
        ..*visit::default_simple_visitor()
    }));
}

fn gather_external_rtcalls(ccx: @crate_ctxt) {
    do cstore::iter_crate_data(ccx.sess.cstore) |_cnum, cmeta| {
        do decoder::each_path(ccx.sess.intr(), cmeta) |path| {
            let pathname = path.path_string;
            match path.def_like {
              decoder::dl_def(d) => {
                match d {
                  ast::def_fn(did, _) => {
                    // FIXME (#2861): This should really iterate attributes
                    // like gather_local_rtcalls, but we'll need to
                    // export attributes in metadata/encoder before we can do
                    // that.
                    let sentinel = ~"rt::rt_";
                    let slen = str::len(sentinel);
                    if str::starts_with(pathname, sentinel) {
                        let name = str::substr(pathname,
                                               slen, str::len(pathname)-slen);
                        push_rtcall(ccx, name, did);
                    }
                  }
                  _ => ()
                }
              }
              _ => ()
            }
            true
        }
    }
}

fn gather_rtcalls(ccx: @crate_ctxt, crate: @ast::crate) {
    gather_local_rtcalls(ccx, crate);
    gather_external_rtcalls(ccx);

    // FIXME (#2861): Check for other rtcalls too, once they are
    // supported. Also probably want to check type signature so we don't crash
    // in some obscure place in LLVM if the user provides the wrong signature
    // for an rtcall.
    let expected_rtcalls =
        ~[~"exchange_free", ~"exchange_malloc", ~"fail", ~"free", ~"malloc"];
    for vec::each(expected_rtcalls) |name| {
        if !ccx.rtcalls.contains_key(name) {
            fail fmt!("no definition for runtime call %s", name);
        }
    }
}

fn create_module_map(ccx: @crate_ctxt) -> ValueRef {
    let elttype = T_struct(~[ccx.int_type, ccx.int_type]);
    let maptype = T_array(elttype, ccx.module_data.size() + 1u);
    let map = str::as_c_str(~"_rust_mod_map", |buf| {
        llvm::LLVMAddGlobal(ccx.llmod, maptype, buf)
    });
    lib::llvm::SetLinkage(map, lib::llvm::InternalLinkage);
    let mut elts: ~[ValueRef] = ~[];
    for ccx.module_data.each |key, val| {
        let elt = C_struct(~[p2i(ccx, C_cstr(ccx, key)),
                            p2i(ccx, val)]);
        vec::push(elts, elt);
    }
    let term = C_struct(~[C_int(ccx, 0), C_int(ccx, 0)]);
    vec::push(elts, term);
    llvm::LLVMSetInitializer(map, C_array(elttype, elts));
    return map;
}


fn decl_crate_map(sess: session::session, mapmeta: link_meta,
                  llmod: ModuleRef) -> ValueRef {
    let targ_cfg = sess.targ_cfg;
    let int_type = T_int(targ_cfg);
    let mut n_subcrates = 1;
    let cstore = sess.cstore;
    while cstore::have_crate_data(cstore, n_subcrates) { n_subcrates += 1; }
    let mapname = if sess.building_library {
        mapmeta.name + ~"_" + mapmeta.vers + ~"_" + mapmeta.extras_hash
    } else { ~"toplevel" };
    let sym_name = ~"_rust_crate_map_" + mapname;
    let arrtype = T_array(int_type, n_subcrates as uint);
    let maptype = T_struct(~[int_type, arrtype]);
    let map = str::as_c_str(sym_name, |buf| {
        llvm::LLVMAddGlobal(llmod, maptype, buf)
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
            llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
        });
        vec::push(subcrates, p2i(ccx, cr));
        i += 1;
    }
    vec::push(subcrates, C_int(ccx, 0));
    llvm::LLVMSetInitializer(map, C_struct(
        ~[p2i(ccx, create_module_map(ccx)),
         C_array(ccx.int_type, subcrates)]));
}

fn crate_ctxt_to_encode_parms(cx: @crate_ctxt)
    -> encoder::encode_parms {

    let encode_inlined_item =
        |a,b,c,d| astencode::encode_inlined_item(a, b, c, d, cx.maps);

    return {
        diag: cx.sess.diagnostic(),
        tcx: cx.tcx,
        reachable: cx.reachable,
        reexports: reexports(cx),
        reexports2: cx.exp_map2,
        item_symbols: cx.item_symbols,
        discrim_symbols: cx.discrim_symbols,
        link_meta: cx.link_meta,
        cstore: cx.sess.cstore,
        encode_inlined_item: encode_inlined_item
    };

    fn reexports(cx: @crate_ctxt) -> ~[(~str, ast::def_id)] {
        let mut reexports = ~[];
        for cx.exp_map.each |exp_id, defs| {
            for defs.each |def| {
                if !def.reexp { again; }
                let path = match cx.tcx.items.get(exp_id) {
                  ast_map::node_export(_, path) => {
                      ast_map::path_to_str(*path, cx.sess.parse_sess.interner)
                  }
                  _ => fail ~"reexports"
                };
                vec::push(reexports, (path, def.id));
            }
        }
        return reexports;
    }
}

fn write_metadata(cx: @crate_ctxt, crate: @ast::crate) {
    if !cx.sess.building_library { return; }
    let encode_parms = crate_ctxt_to_encode_parms(cx);
    let llmeta = C_bytes(encoder::encode_metadata(encode_parms, crate));
    let llconst = C_struct(~[llmeta]);
    let mut llglobal = str::as_c_str(~"rust_metadata", |buf| {
        llvm::LLVMAddGlobal(cx.llmod, val_ty(llconst), buf)
    });
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

// Writes the current ABI version into the crate.
fn write_abi_version(ccx: @crate_ctxt) {
    mk_global(ccx, ~"rust_abi_version", C_uint(ccx, abi::abi_version),
                     false);
}

fn trans_crate(sess: session::session,
               crate: @ast::crate,
               tcx: ty::ctxt,
               output: &Path,
               emap: resolve::ExportMap,
               emap2: resolve::ExportMap2,
               maps: astencode::maps)
            -> (ModuleRef, link_meta) {

    let symbol_hasher = @hash::default_state();
    let link_meta =
        link::build_link_meta(sess, *crate, output, symbol_hasher);
    let reachable = reachable::find_reachable(crate.node.module, emap, tcx,
                                              maps.method_map);

    // Append ".rc" to crate name as LLVM module identifier.
    //
    // LLVM code generator emits a ".file filename" directive
    // for ELF backends. Value of the "filename" is set as the
    // LLVM module identifier.  Due to a LLVM MC bug[1], LLVM
    // crashes if the module identifer is same as other symbols
    // such as a function name in the module.
    // 1. http://llvm.org/bugs/show_bug.cgi?id=11479
    let llmod_id = link_meta.name + ~".rc";

    let llmod = str::as_c_str(llmod_id, |buf| {
        llvm::LLVMModuleCreateWithNameInContext
            (buf, llvm::LLVMGetGlobalContext())
    });
    let data_layout = sess.targ_cfg.target_strs.data_layout;
    let targ_triple = sess.targ_cfg.target_strs.target_triple;
    let _: () =
        str::as_c_str(data_layout,
                    |buf| llvm::LLVMSetDataLayout(llmod, buf));
    let _: () =
        str::as_c_str(targ_triple,
                    |buf| llvm::LLVMSetTarget(llmod, buf));
    let targ_cfg = sess.targ_cfg;
    let td = mk_target_data(sess.targ_cfg.target_strs.data_layout);
    let tn = mk_type_names();
    let intrinsics = declare_intrinsics(llmod);
    if sess.opts.extra_debuginfo {
        declare_dbg_intrinsics(llmod, intrinsics);
    }
    let int_type = T_int(targ_cfg);
    let float_type = T_float(targ_cfg);
    let task_type = T_task(targ_cfg);
    let taskptr_type = T_ptr(task_type);
    lib::llvm::associate_type(tn, ~"taskptr", taskptr_type);
    let tydesc_type = T_tydesc(targ_cfg);
    lib::llvm::associate_type(tn, ~"tydesc", tydesc_type);
    let crate_map = decl_crate_map(sess, link_meta, llmod);
    let dbg_cx = if sess.opts.debuginfo {
        option::Some(debuginfo::mk_ctxt(llmod_id, sess.parse_sess.interner))
    } else {
        option::None
    };

    let ccx =
        @{sess: sess,
          llmod: llmod,
          td: td,
          tn: tn,
          externs: str_hash::<ValueRef>(),
          intrinsics: intrinsics,
          item_vals: int_hash::<ValueRef>(),
          exp_map: emap,
          exp_map2: emap2,
          reachable: reachable,
          item_symbols: int_hash::<~str>(),
          mut main_fn: None::<ValueRef>,
          link_meta: link_meta,
          enum_sizes: ty::new_ty_hash(),
          discrims: ast_util::new_def_hash::<ValueRef>(),
          discrim_symbols: int_hash::<~str>(),
          tydescs: ty::new_ty_hash(),
          mut finished_tydescs: false,
          external: ast_util::new_def_hash(),
          monomorphized: map::hashmap(hash_mono_id, sys::shape_eq),
          monomorphizing: ast_util::new_def_hash(),
          type_use_cache: ast_util::new_def_hash(),
          vtables: map::hashmap(hash_mono_id, sys::shape_eq),
          const_cstr_cache: map::str_hash(),
          const_globals: int_hash::<ValueRef>(),
          module_data: str_hash::<ValueRef>(),
          lltypes: ty::new_ty_hash(),
          names: new_namegen(sess.parse_sess.interner),
          next_addrspace: new_addrspace_gen(),
          symbol_hasher: symbol_hasher,
          type_hashcodes: ty::new_ty_hash(),
          type_short_names: ty::new_ty_hash(),
          all_llvm_symbols: str_hash::<()>(),
          tcx: tcx,
          maps: maps,
          stats:
              {mut n_static_tydescs: 0u,
               mut n_glues_created: 0u,
               mut n_null_glues: 0u,
               mut n_real_glues: 0u,
               llvm_insn_ctxt: @mut ~[],
               llvm_insns: str_hash(),
               fn_times: @mut ~[]},
          upcalls:
              upcall::declare_upcalls(targ_cfg, tn, tydesc_type,
                                      llmod),
          rtcalls: str_hash::<ast::def_id>(),
          tydesc_type: tydesc_type,
          int_type: int_type,
          float_type: float_type,
          task_type: task_type,
          opaque_vec_type: T_opaque_vec(targ_cfg),
          builder: BuilderRef_res(llvm::LLVMCreateBuilder()),
          shape_cx: mk_ctxt(llmod),
          crate_map: crate_map,
          dbg_cx: dbg_cx,
          class_ctors: int_hash::<ast::def_id>(),
          mut do_not_commit_warning_issued: false};


    gather_rtcalls(ccx, crate);

    {
        let _icx = ccx.insn_ctxt("data");
        trans_constants(ccx, crate);
    }

    {
        let _icx = ccx.insn_ctxt("text");
        trans_mod(ccx, crate.node.module);
    }

    fill_crate_map(ccx, crate_map);
    // NB: Must call force_declare_tydescs before emit_tydescs to break
    // cyclical dependency with shape code! See shape.rs for details.
    force_declare_tydescs(ccx);
    glue::emit_tydescs(ccx);
    gen_shape_tables(ccx);
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

        // FIXME (#2280): this temporary shouldn't be
        // necessary, but seems to be, for borrowing.
        let times = copy *ccx.stats.fn_times;
        for vec::each(times) |timing| {
            io::println(fmt!("time: %s took %d ms", timing.ident,
                             timing.time));
        }
    }

    if ccx.sess.count_llvm_insns() {
        for ccx.stats.llvm_insns.each |k, v| {
            io::println(fmt!("%-7u %s", v, k));
        }
    }
    return (llmod, link_meta);
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
