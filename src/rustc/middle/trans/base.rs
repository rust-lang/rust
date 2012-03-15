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

import libc::c_uint;
import std::{map, time};
import std::map::hashmap;
import std::map::{int_hash, str_hash};
import driver::session;
import session::session;
import front::attr;
import back::{link, abi, upcall};
import syntax::{ast, ast_util, codemap};
import ast_util::inlined_item_methods;
import ast_util::local_def;
import syntax::visit;
import syntax::codemap::span;
import syntax::print::pprust::{expr_to_str, stmt_to_str, path_to_str};
import pat_util::*;
import visit::vt;
import util::common::*;
import lib::llvm::{llvm, mk_target_data, mk_type_names};
import lib::llvm::{ModuleRef, ValueRef, TypeRef, BasicBlockRef};
import lib::llvm::{True, False};
import link::{mangle_internal_name_by_type_only,
              mangle_internal_name_by_seq,
              mangle_internal_name_by_path,
              mangle_internal_name_by_path_and_seq,
              mangle_exported_name};
import metadata::{csearch, cstore};
import util::ppaux::{ty_to_str, ty_to_short_str};

import common::*;
import build::*;
import shape::*;
import type_of::*;
import type_of::type_of; // Issue #1873
import ast_map::{path, path_mod, path_name};

import std::smallintmap;

// Destinations

// These are passed around by the code generating functions to track the
// destination of a computation's value.

enum dest {
    by_val(@mutable ValueRef),
    save_in(ValueRef),
    ignore,
}

fn dest_str(ccx: @crate_ctxt, d: dest) -> str {
    alt d {
      by_val(v) { #fmt["by_val(%s)", val_str(ccx.tn, *v)] }
      save_in(v) { #fmt["save_in(%s)", val_str(ccx.tn, v)] }
      ignore { "ignore" }
    }
}

fn empty_dest_cell() -> @mutable ValueRef {
    ret @mutable llvm::LLVMGetUndef(T_nil());
}

fn dup_for_join(dest: dest) -> dest {
    alt dest {
      by_val(_) { by_val(empty_dest_cell()) }
      _ { dest }
    }
}

fn join_returns(parent_cx: block, in_cxs: [block],
                in_ds: [dest], out_dest: dest) -> block {
    let out = sub_block(parent_cx, "join");
    let reachable = false, i = 0u, phi = none;
    for cx in in_cxs {
        if !cx.unreachable {
            Br(cx, out.llbb);
            reachable = true;
            alt in_ds[i] {
              by_val(cell) {
                if option::is_none(phi) {
                    phi = some(EmptyPhi(out, val_ty(*cell)));
                }
                AddIncomingToPhi(option::get(phi), *cell, cx.llbb);
              }
              _ {}
            }
        }
        i += 1u;
    }
    if !reachable {
        Unreachable(out);
    } else {
        alt out_dest {
          by_val(cell) { *cell = option::get(phi); }
          _ {}
        }
    }
    ret out;
}

// Used to put an immediate value in a dest.
fn store_in_dest(bcx: block, val: ValueRef, dest: dest) -> block {
    alt dest {
      ignore {}
      by_val(cell) { *cell = val; }
      save_in(addr) { Store(bcx, val, addr); }
    }
    ret bcx;
}

fn get_dest_addr(dest: dest) -> ValueRef {
    alt dest {
       save_in(a) { a }
       _ { fail "get_dest_addr: not a save_in"; }
    }
}

// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
fn sanitize(s: str) -> str {
    let result = "";
    for c: u8 in s {
        if c == '@' as u8 {
            result += "boxed_";
        } else {
            if c == ',' as u8 {
                result += "_";
            } else {
                if c == '{' as u8 || c == '(' as u8 {
                    result += "_of_";
                } else {
                    if c != 10u8 && c != '}' as u8 && c != ')' as u8 &&
                           c != ' ' as u8 && c != '\t' as u8 && c != ';' as u8
                       {
                        let v = [c];
                        result += str::from_bytes(v);
                    }
                }
            }
        }
    }
    ret result;
}


fn log_fn_time(ccx: @crate_ctxt, name: str, start: time::timeval,
               end: time::timeval) {
    let elapsed = 1000 * ((end.sec - start.sec) as int) +
        ((end.usec as int) - (start.usec as int)) / 1000;
    *ccx.stats.fn_times += [{ident: name, time: elapsed}];
}


fn decl_fn(llmod: ModuleRef, name: str, cc: lib::llvm::CallConv,
           llty: TypeRef) -> ValueRef {
    let llfn: ValueRef = str::as_c_str(name, {|buf|
        llvm::LLVMGetOrInsertFunction(llmod, buf, llty)
    });
    lib::llvm::SetFunctionCallConv(llfn, cc);
    ret llfn;
}

fn decl_cdecl_fn(llmod: ModuleRef, name: str, llty: TypeRef) -> ValueRef {
    ret decl_fn(llmod, name, lib::llvm::CCallConv, llty);
}


// Only use this if you are going to actually define the function. It's
// not valid to simply declare a function as internal.
fn decl_internal_cdecl_fn(llmod: ModuleRef, name: str, llty: TypeRef) ->
   ValueRef {
    let llfn = decl_cdecl_fn(llmod, name, llty);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    ret llfn;
}

fn get_extern_fn(externs: hashmap<str, ValueRef>, llmod: ModuleRef, name: str,
                 cc: lib::llvm::CallConv, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { ret externs.get(name); }
    let f = decl_fn(llmod, name, cc, ty);
    externs.insert(name, f);
    ret f;
}

fn get_extern_const(externs: hashmap<str, ValueRef>, llmod: ModuleRef,
                    name: str, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { ret externs.get(name); }
    let c = str::as_c_str(name, {|buf| llvm::LLVMAddGlobal(llmod, ty, buf) });
    externs.insert(name, c);
    ret c;
}

fn get_simple_extern_fn(cx: block,
                        externs: hashmap<str, ValueRef>,
                        llmod: ModuleRef,
                        name: str, n_args: int) -> ValueRef {
    let ccx = cx.fcx.ccx;
    let inputs = vec::from_elem(n_args as uint, ccx.int_type);
    let output = ccx.int_type;
    let t = T_fn(inputs, output);
    ret get_extern_fn(externs, llmod, name, lib::llvm::CCallConv, t);
}

fn trans_native_call(cx: block, externs: hashmap<str, ValueRef>,
                     llmod: ModuleRef, name: str, args: [ValueRef]) ->
   ValueRef {
    let n = args.len() as int;
    let llnative: ValueRef =
        get_simple_extern_fn(cx, externs, llmod, name, n);
    let call_args: [ValueRef] = [];
    for a: ValueRef in args {
        call_args += [ZExtOrBitCast(cx, a, cx.ccx().int_type)];
    }
    ret Call(cx, llnative, call_args);
}

fn trans_free(cx: block, v: ValueRef) -> block {
    Call(cx, cx.ccx().upcalls.free, [PointerCast(cx, v, T_ptr(T_i8()))]);
    cx
}

fn trans_shared_free(cx: block, v: ValueRef) -> block {
    Call(cx, cx.ccx().upcalls.shared_free,
         [PointerCast(cx, v, T_ptr(T_i8()))]);
    ret cx;
}

fn umax(cx: block, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = ICmp(cx, lib::llvm::IntULT, a, b);
    ret Select(cx, cond, b, a);
}

fn umin(cx: block, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = ICmp(cx, lib::llvm::IntULT, a, b);
    ret Select(cx, cond, a, b);
}

fn alloca(cx: block, t: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(t); }
    ret Alloca(raw_block(cx.fcx, cx.fcx.llstaticallocas), t);
}

// Given a pointer p, returns a pointer sz(p) (i.e., inc'd by sz bytes).
// The type of the returned pointer is always i8*.  If you care about the
// return type, use bump_ptr().
fn ptr_offs(bcx: block, base: ValueRef, sz: ValueRef) -> ValueRef {
    let raw = PointerCast(bcx, base, T_ptr(T_i8()));
    InBoundsGEP(bcx, raw, [sz])
}

// Increment a pointer by a given amount and then cast it to be a pointer
// to a given type.
fn bump_ptr(bcx: block, t: ty::t, base: ValueRef, sz: ValueRef) ->
   ValueRef {
    let ccx = bcx.ccx();
    let bumped = ptr_offs(bcx, base, sz);
    let typ = T_ptr(type_of(ccx, t));
    PointerCast(bcx, bumped, typ)
}

// Replacement for the LLVM 'GEP' instruction when field indexing into a enum.
// @llblobptr is the data part of a enum value; its actual type
// is meaningless, as it will be cast away.
fn GEP_enum(bcx: block, llblobptr: ValueRef, enum_id: ast::def_id,
            variant_id: ast::def_id, ty_substs: [ty::t],
            ix: uint) -> result {
    let ccx = bcx.ccx();
    let variant = ty::enum_variant_with_id(ccx.tcx, enum_id, variant_id);
    assert ix < variant.args.len();

    let arg_lltys = vec::map(variant.args, {|aty|
        type_of(ccx, ty::substitute_type_params(ccx.tcx, ty_substs, aty))
    });
    let typed_blobptr = PointerCast(bcx, llblobptr,
                                    T_ptr(T_struct(arg_lltys)));
    rslt(bcx, GEPi(bcx, typed_blobptr, [0, ix as int]))
}

// trans_shared_malloc: expects a type indicating which pointer type we want
// and a size indicating how much space we want malloc'd.
fn trans_shared_malloc(cx: block, llptr_ty: TypeRef, llsize: ValueRef)
   -> result {
    let rval = Call(cx, cx.ccx().upcalls.shared_malloc, [llsize]);
    ret rslt(cx, PointerCast(cx, rval, llptr_ty));
}

// Returns a pointer to the body for the box. The box may be an opaque
// box. The result will be casted to the type of body_t, if it is statically
// known.
//
// The runtime equivalent is box_body() in "rust_internal.h".
fn opaque_box_body(bcx: block,
                      body_t: ty::t,
                      boxptr: ValueRef) -> ValueRef {
    let ccx = bcx.ccx();
    let boxptr = PointerCast(bcx, boxptr, T_ptr(T_box_header(ccx)));
    let bodyptr = GEPi(bcx, boxptr, [1]);
    PointerCast(bcx, bodyptr, T_ptr(type_of(ccx, body_t)))
}

// trans_malloc_boxed_raw: expects an unboxed type and returns a pointer to
// enough space for a box of that type.  This includes a rust_opaque_box
// header.
fn trans_malloc_boxed_raw(bcx: block, t: ty::t,
                          &static_ti: option<@tydesc_info>) -> result {
    let bcx = bcx, ccx = bcx.ccx();

    // Grab the TypeRef type of box_ptr, because that's what trans_raw_malloc
    // wants.
    let box_ptr = ty::mk_imm_box(bcx.tcx(), t);
    let llty = type_of(ccx, box_ptr);

    // Get the tydesc for the body:
    let {bcx, val: lltydesc} = get_tydesc(bcx, t, static_ti);
    lazily_emit_all_tydesc_glue(ccx, static_ti);

    // Allocate space:
    let rval = Call(bcx, ccx.upcalls.malloc, [lltydesc]);
    ret rslt(bcx, PointerCast(bcx, rval, llty));
}

// trans_malloc_boxed: usefully wraps trans_malloc_box_raw; allocates a box,
// initializes the reference count to 1, and pulls out the body and rc
fn trans_malloc_boxed(bcx: block, t: ty::t) ->
   {bcx: block, box: ValueRef, body: ValueRef} {
    let ti = none;
    let {bcx, val:box} = trans_malloc_boxed_raw(bcx, t, ti);
    let body = GEPi(bcx, box, [0, abi::box_field_body]);
    ret {bcx: bcx, box: box, body: body};
}

// Type descriptor and type glue stuff

fn get_tydesc_simple(bcx: block, t: ty::t) -> result {
    let ti = none;
    get_tydesc(bcx, t, ti)
}

fn get_tydesc(cx: block, t: ty::t,
              &static_ti: option<@tydesc_info>) -> result {
    assert !ty::type_has_params(t);
    // Otherwise, generate a tydesc if necessary, and return it.
    let info = get_static_tydesc(cx.ccx(), t);
    static_ti = some(info);
    ret rslt(cx, info.tydesc);
}

fn get_static_tydesc(ccx: @crate_ctxt, t: ty::t) -> @tydesc_info {
    alt ccx.tydescs.find(t) {
      some(info) { ret info; }
      none {
        ccx.stats.n_static_tydescs += 1u;
        let info = declare_tydesc(ccx, t);
        ccx.tydescs.insert(t, info);
        ret info;
      }
    }
}

fn set_no_inline(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::NoInlineAttribute as c_uint,
                              0u as c_uint);
}

// Tell LLVM to emit the information necessary to unwind the stack for the
// function f.
fn set_uwtable(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::UWTableAttribute as c_uint,
                              0u as c_uint);
}

fn set_inline_hint(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::InlineHintAttribute as c_uint,
                              0u as c_uint);
}

fn set_inline_hint_if_appr(attrs: [ast::attribute],
                           llfn: ValueRef) {
    alt attr::find_inline_attr(attrs) {
      attr::ia_hint { set_inline_hint(llfn); }
      attr::ia_always { set_always_inline(llfn); }
      attr::ia_none { /* fallthrough */ }
    }
}

fn set_always_inline(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f, lib::llvm::AlwaysInlineAttribute as c_uint,
                              0u as c_uint);
}

fn set_custom_stack_growth_fn(f: ValueRef) {
    // FIXME: Remove this hack to work around the lack of u64 in the FFI.
    llvm::LLVMAddFunctionAttr(f, 0u as c_uint, 1u as c_uint);
}

fn set_glue_inlining(f: ValueRef, t: ty::t) {
    if ty::type_is_structural(t) {
        set_no_inline(f);
    } else { set_always_inline(f); }
}


// Generates the declaration for (but doesn't emit) a type descriptor.
fn declare_tydesc(ccx: @crate_ctxt, t: ty::t) -> @tydesc_info {
    log(debug, "+++ declare_tydesc " + ty_to_str(ccx.tcx, t));
    let llsize;
    let llalign;
    let llty = type_of(ccx, t);
    llsize = llsize_of(ccx, llty);
    llalign = llalign_of(ccx, llty);
    let name;
    if ccx.sess.opts.debuginfo {
        name = mangle_internal_name_by_type_only(ccx, t, "tydesc");
        name = sanitize(name);
    } else { name = mangle_internal_name_by_seq(ccx, "tydesc"); }
    let gvar = str::as_c_str(name, {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type, buf)
    });
    let info =
        @{ty: t,
          tydesc: gvar,
          size: llsize,
          align: llalign,
          mutable take_glue: none,
          mutable drop_glue: none,
          mutable free_glue: none};
    log(debug, "--- declare_tydesc " + ty_to_str(ccx.tcx, t));
    ret info;
}

type glue_helper = fn@(block, ValueRef, ty::t);

fn declare_generic_glue(ccx: @crate_ctxt, t: ty::t, llfnty: TypeRef,
                        name: str) -> ValueRef {
    let name = name;
    let fn_nm;
    if ccx.sess.opts.debuginfo {
        fn_nm = mangle_internal_name_by_type_only(ccx, t, "glue_" + name);
        fn_nm = sanitize(fn_nm);
    } else { fn_nm = mangle_internal_name_by_seq(ccx, "glue_" + name); }
    let llfn = decl_cdecl_fn(ccx.llmod, fn_nm, llfnty);
    set_glue_inlining(llfn, t);
    ret llfn;
}

fn make_generic_glue_inner(ccx: @crate_ctxt, t: ty::t,
                           llfn: ValueRef, helper: glue_helper) -> ValueRef {
    let fcx = new_fn_ctxt(ccx, [], llfn, none);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    ccx.stats.n_glues_created += 1u;
    // Any nontrivial glue is with values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.

    let llty = T_ptr(type_of(ccx, t));

    let bcx = top_scope_block(fcx, none);
    let lltop = bcx.llbb;
    let llrawptr0 = llvm::LLVMGetParam(llfn, 3u as c_uint);
    let llval0 = BitCast(bcx, llrawptr0, llty);
    helper(bcx, llval0, t);
    finish_fn(fcx, lltop);
    ret llfn;
}

fn make_generic_glue(ccx: @crate_ctxt, t: ty::t, llfn: ValueRef,
                     helper: glue_helper, name: str)
    -> ValueRef {
    if !ccx.sess.opts.stats {
        ret make_generic_glue_inner(ccx, t, llfn, helper);
    }

    let start = time::get_time();
    let llval = make_generic_glue_inner(ccx, t, llfn, helper);
    let end = time::get_time();
    log_fn_time(ccx, "glue " + name + " " + ty_to_short_str(ccx.tcx, t),
                start, end);
    ret llval;
}

fn emit_tydescs(ccx: @crate_ctxt) {
    ccx.tydescs.items {|key, val|
        let glue_fn_ty = T_ptr(T_glue_fn(ccx));
        let ti = val;
        let take_glue =
            alt ti.take_glue {
              none { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let drop_glue =
            alt ti.drop_glue {
              none { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let free_glue =
            alt ti.free_glue {
              none { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };

        let shape = shape_of(ccx, key, []);
        let shape_tables =
            llvm::LLVMConstPointerCast(ccx.shape_cx.llshapetables,
                                       T_ptr(T_i8()));

        let tydesc =
            C_named_struct(ccx.tydesc_type,
                           [C_null(T_ptr(T_ptr(ccx.tydesc_type))),
                            ti.size, // size
                            ti.align, // align
                            take_glue, // take_glue
                            drop_glue, // drop_glue
                            free_glue, // free_glue
                            C_null(T_ptr(T_i8())), // unused
                            C_null(glue_fn_ty), // sever_glue
                            C_null(glue_fn_ty), // mark_glue
                            C_null(glue_fn_ty), // unused
                            C_null(T_ptr(T_i8())), // cmp_glue
                            C_shape(ccx, shape), // shape
                            shape_tables, // shape_tables
                            C_int(ccx, 0), // n_params
                            C_int(ccx, 0)]); // n_obj_params

        let gvar = ti.tydesc;
        llvm::LLVMSetInitializer(gvar, tydesc);
        llvm::LLVMSetGlobalConstant(gvar, True);
        lib::llvm::SetLinkage(gvar, lib::llvm::InternalLinkage);
    };
}

fn make_take_glue(cx: block, v: ValueRef, t: ty::t) {
    let bcx = cx;
    // NB: v is a *pointer* to type t here, not a direct value.
    bcx = alt ty::get(t).struct {
      ty::ty_box(_) | ty::ty_opaque_box {
        incr_refcnt_of_boxed(bcx, Load(bcx, v))
      }
      ty::ty_uniq(_) {
        let r = uniq::duplicate(bcx, Load(bcx, v), t);
        Store(r.bcx, r.val, v);
        r.bcx
      }
      ty::ty_vec(_) | ty::ty_str {
        let r = tvec::duplicate(bcx, Load(bcx, v), t);
        Store(r.bcx, r.val, v);
        r.bcx
      }
      ty::ty_fn(_) {
        closure::make_fn_glue(bcx, v, t, take_ty)
      }
      ty::ty_iface(_, _) {
        let box = Load(bcx, GEPi(bcx, v, [0, 1]));
        incr_refcnt_of_boxed(bcx, box)
      }
      ty::ty_opaque_closure_ptr(ck) {
        closure::make_opaque_cbox_take_glue(bcx, ck, v)
      }
      _ if ty::type_is_structural(t) {
        iter_structural_ty(bcx, v, t, take_ty)
      }
      _ { bcx }
    };

    build_return(bcx);
}

fn incr_refcnt_of_boxed(cx: block, box_ptr: ValueRef) -> block {
    let ccx = cx.ccx();
    maybe_validate_box(cx, box_ptr);
    let rc_ptr = GEPi(cx, box_ptr, [0, abi::box_field_refcnt]);
    let rc = Load(cx, rc_ptr);
    rc = Add(cx, rc, C_int(ccx, 1));
    Store(cx, rc, rc_ptr);
    ret cx;
}

fn make_free_glue(bcx: block, v: ValueRef, t: ty::t) {
    // v is a pointer to the actual box component of the type here. The
    // ValueRef will have the wrong type here (make_generic_glue is casting
    // everything to a pointer to the type that the glue acts on).
    let ccx = bcx.ccx();
    let bcx = alt ty::get(t).struct {
      ty::ty_box(body_mt) {
        let v = PointerCast(bcx, v, type_of(ccx, t));
        let body = GEPi(bcx, v, [0, abi::box_field_body]);
        let bcx = drop_ty(bcx, body, body_mt.ty);
        trans_free(bcx, v)
      }
      ty::ty_opaque_box {
        let v = PointerCast(bcx, v, type_of(ccx, t));
        let td = Load(bcx, GEPi(bcx, v, [0, abi::box_field_tydesc]));
        let valptr = GEPi(bcx, v, [0, abi::box_field_body]);
        call_tydesc_glue_full(bcx, valptr, td, abi::tydesc_field_drop_glue,
                              none);
        trans_free(bcx, v)
      }
      ty::ty_uniq(content_mt) {
        let v = PointerCast(bcx, v, type_of(ccx, t));
        uniq::make_free_glue(bcx, v, t)
      }
      ty::ty_vec(_) | ty::ty_str {
        tvec::make_free_glue(bcx, PointerCast(bcx, v, type_of(ccx, t)), t)
      }
      ty::ty_fn(_) {
        closure::make_fn_glue(bcx, v, t, free_ty)
      }
      ty::ty_opaque_closure_ptr(ck) {
        closure::make_opaque_cbox_free_glue(bcx, ck, v)
      }
      _ { bcx }
    };
    build_return(bcx);
}

fn make_drop_glue(bcx: block, v0: ValueRef, t: ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let ccx = bcx.ccx();
    let bcx = alt ty::get(t).struct {
      ty::ty_box(_) | ty::ty_opaque_box {
        decr_refcnt_maybe_free(bcx, Load(bcx, v0), t)
      }
      ty::ty_uniq(_) | ty::ty_vec(_) | ty::ty_str {
        free_ty(bcx, Load(bcx, v0), t)
      }
      ty::ty_res(did, inner, tps) {
        trans_res_drop(bcx, v0, did, inner, tps)
      }
      ty::ty_fn(_) {
        closure::make_fn_glue(bcx, v0, t, drop_ty)
      }
      ty::ty_iface(_, _) {
        let box = Load(bcx, GEPi(bcx, v0, [0, 1]));
        decr_refcnt_maybe_free(bcx, box, ty::mk_opaque_box(ccx.tcx))
      }
      ty::ty_opaque_closure_ptr(ck) {
        closure::make_opaque_cbox_drop_glue(bcx, ck, v0)
      }
      _ {
        if ty::type_needs_drop(ccx.tcx, t) &&
            ty::type_is_structural(t) {
            iter_structural_ty(bcx, v0, t, drop_ty)
        } else { bcx }
      }
    };
    build_return(bcx);
}

fn get_res_dtor(ccx: @crate_ctxt, did: ast::def_id, substs: [ty::t])
   -> ValueRef {
    let did = if did.crate != ast::local_crate && substs.len() > 0u {
        maybe_instantiate_inline(ccx, did)
    } else { did };
    assert did.crate == ast::local_crate;
    monomorphic_fn(ccx, did, substs, none).val
}

fn trans_res_drop(bcx: block, rs: ValueRef, did: ast::def_id,
                  inner_t: ty::t, tps: [ty::t]) -> block {
    let ccx = bcx.ccx();
    let inner_t_s = ty::substitute_type_params(ccx.tcx, tps, inner_t);

    let drop_flag = GEPi(bcx, rs, [0, 0]);
    with_cond(bcx, IsNotNull(bcx, Load(bcx, drop_flag))) {|bcx|
        let valptr = GEPi(bcx, rs, [0, 1]);
        // Find and call the actual destructor.
        let dtor_addr = get_res_dtor(ccx, did, tps);
        let args = [bcx.fcx.llretptr, null_env_ptr(bcx)];
        // Kludge to work around the fact that we know the precise type of the
        // value here, but the dtor expects a type that might have opaque
        // boxes and such.
        let val_llty = lib::llvm::fn_ty_param_tys
            (llvm::LLVMGetElementType
             (llvm::LLVMTypeOf(dtor_addr)))[args.len()];
        let val_cast = BitCast(bcx, valptr, val_llty);
        Call(bcx, dtor_addr, args + [val_cast]);

        let bcx = drop_ty(bcx, valptr, inner_t_s);
        Store(bcx, C_u8(0u), drop_flag);
        bcx
    }
}

fn maybe_validate_box(_cx: block, _box_ptr: ValueRef) {
    // Uncomment this when debugging annoying use-after-free
    // bugs.  But do not commit with this uncommented!  Big performance hit.

    // let cx = _cx, box_ptr = _box_ptr;
    // let ccx = cx.ccx();
    // warn_not_to_commit(ccx, "validate_box() is uncommented");
    // let raw_box_ptr = PointerCast(cx, box_ptr, T_ptr(T_i8()));
    // Call(cx, ccx.upcalls.validate_box, [raw_box_ptr]);
}

fn decr_refcnt_maybe_free(bcx: block, box_ptr: ValueRef, t: ty::t) -> block {
    let ccx = bcx.ccx();
    maybe_validate_box(bcx, box_ptr);

    let llbox_ty = T_opaque_box_ptr(ccx);
    let box_ptr = PointerCast(bcx, box_ptr, llbox_ty);
    with_cond(bcx, IsNotNull(bcx, box_ptr)) {|bcx|
        let rc_ptr = GEPi(bcx, box_ptr, [0, abi::box_field_refcnt]);
        let rc = Sub(bcx, Load(bcx, rc_ptr), C_int(ccx, 1));
        Store(bcx, rc, rc_ptr);
        let zero_test = ICmp(bcx, lib::llvm::IntEQ, C_int(ccx, 0), rc);
        with_cond(bcx, zero_test) {|bcx| free_ty(bcx, box_ptr, t)}
    }
}

// Structural comparison: a rather involved form of glue.
fn maybe_name_value(cx: @crate_ctxt, v: ValueRef, s: str) {
    if cx.sess.opts.save_temps {
        let _: () = str::as_c_str(s, {|buf| llvm::LLVMSetValueName(v, buf) });
    }
}


// Used only for creating scalar comparison glue.
enum scalar_type { nil_type, signed_int, unsigned_int, floating_point, }


fn compare_scalar_types(cx: block, lhs: ValueRef, rhs: ValueRef,
                        t: ty::t, op: ast::binop) -> result {
    let f = bind compare_scalar_values(cx, lhs, rhs, _, op);

    alt ty::get(t).struct {
      ty::ty_nil { ret rslt(cx, f(nil_type)); }
      ty::ty_bool | ty::ty_ptr(_) { ret rslt(cx, f(unsigned_int)); }
      ty::ty_int(_) { ret rslt(cx, f(signed_int)); }
      ty::ty_uint(_) { ret rslt(cx, f(unsigned_int)); }
      ty::ty_float(_) { ret rslt(cx, f(floating_point)); }
      ty::ty_type {
        ret rslt(trans_fail(cx, none,
                            "attempt to compare values of type type"),
                 C_nil());
      }
      _ {
        // Should never get here, because t is scalar.
        cx.sess().bug("non-scalar type passed to \
                                 compare_scalar_types");
      }
    }
}


// A helper function to do the actual comparison of scalar values.
fn compare_scalar_values(cx: block, lhs: ValueRef, rhs: ValueRef,
                         nt: scalar_type, op: ast::binop) -> ValueRef {
    fn die_(cx: block) -> ! {
        cx.tcx().sess.bug("compare_scalar_values: must be a\
          comparison operator");
    }
    let die = bind die_(cx);
    alt nt {
      nil_type {
        // We don't need to do actual comparisons for nil.
        // () == () holds but () < () does not.
        alt op {
          ast::eq | ast::le | ast::ge { ret C_bool(true); }
          ast::ne | ast::lt | ast::gt { ret C_bool(false); }
          // refinements would be nice
          _ { die(); }
        }
      }
      floating_point {
        let cmp = alt op {
          ast::eq { lib::llvm::RealOEQ }
          ast::ne { lib::llvm::RealUNE }
          ast::lt { lib::llvm::RealOLT }
          ast::le { lib::llvm::RealOLE }
          ast::gt { lib::llvm::RealOGT }
          ast::ge { lib::llvm::RealOGE }
          _ { die(); }
        };
        ret FCmp(cx, cmp, lhs, rhs);
      }
      signed_int {
        let cmp = alt op {
          ast::eq { lib::llvm::IntEQ }
          ast::ne { lib::llvm::IntNE }
          ast::lt { lib::llvm::IntSLT }
          ast::le { lib::llvm::IntSLE }
          ast::gt { lib::llvm::IntSGT }
          ast::ge { lib::llvm::IntSGE }
          _ { die(); }
        };
        ret ICmp(cx, cmp, lhs, rhs);
      }
      unsigned_int {
        let cmp = alt op {
          ast::eq { lib::llvm::IntEQ }
          ast::ne { lib::llvm::IntNE }
          ast::lt { lib::llvm::IntULT }
          ast::le { lib::llvm::IntULE }
          ast::gt { lib::llvm::IntUGT }
          ast::ge { lib::llvm::IntUGE }
          _ { die(); }
        };
        ret ICmp(cx, cmp, lhs, rhs);
      }
    }
}

type val_pair_fn = fn@(block, ValueRef, ValueRef) -> block;
type val_and_ty_fn = fn@(block, ValueRef, ty::t) -> block;

fn load_inbounds(cx: block, p: ValueRef, idxs: [int]) -> ValueRef {
    ret Load(cx, GEPi(cx, p, idxs));
}

fn store_inbounds(cx: block, v: ValueRef, p: ValueRef,
                  idxs: [int]) {
    Store(cx, v, GEPi(cx, p, idxs));
}

// Iterates through the elements of a structural type.
fn iter_structural_ty(cx: block, av: ValueRef, t: ty::t,
                      f: val_and_ty_fn) -> block {
    fn iter_variant(cx: block, a_tup: ValueRef,
                    variant: ty::variant_info, tps: [ty::t], tid: ast::def_id,
                    f: val_and_ty_fn) -> block {
        if variant.args.len() == 0u { ret cx; }
        let fn_ty = variant.ctor_ty;
        let ccx = cx.ccx();
        let cx = cx;
        alt ty::get(fn_ty).struct {
          ty::ty_fn({inputs: args, _}) {
            let j = 0u;
            let v_id = variant.id;
            for a: ty::arg in args {
                let rslt = GEP_enum(cx, a_tup, tid, v_id, tps, j);
                let llfldp_a = rslt.val;
                cx = rslt.bcx;
                let ty_subst = ty::substitute_type_params(ccx.tcx, tps, a.ty);
                cx = f(cx, llfldp_a, ty_subst);
                j += 1u;
            }
          }
          _ { cx.tcx().sess.bug("iter_variant: not a function type"); }
        }
        ret cx;
    }

    /*
    Typestate constraint that shows the unimpl case doesn't happen?
    */
    let cx = cx;
    alt ty::get(t).struct {
      ty::ty_rec(fields) {
        let i: int = 0;
        for fld: ty::field in fields {
            let llfld_a = GEPi(cx, av, [0, i]);
            cx = f(cx, llfld_a, fld.mt.ty);
            i += 1;
        }
      }
      ty::ty_tup(args) {
        let i = 0;
        for arg in args {
            let llfld_a = GEPi(cx, av, [0, i]);
            cx = f(cx, llfld_a, arg);
            i += 1;
        }
      }
      ty::ty_res(_, inner, tps) {
        let tcx = cx.tcx();
        let inner1 = ty::substitute_type_params(tcx, tps, inner);
        let llfld_a = GEPi(cx, av, [0, 1]);
        ret f(cx, llfld_a, inner1);
      }
      ty::ty_enum(tid, tps) {
        let variants = ty::enum_variants(cx.tcx(), tid);
        let n_variants = (*variants).len();

        // Cast the enums to types we can GEP into.
        if n_variants == 1u {
            ret iter_variant(cx, av, variants[0], tps, tid, f);
        }

        let ccx = cx.ccx();
        let llenumty = T_opaque_enum_ptr(ccx);
        let av_enum = PointerCast(cx, av, llenumty);
        let lldiscrim_a_ptr = GEPi(cx, av_enum, [0, 0]);
        let llunion_a_ptr = GEPi(cx, av_enum, [0, 1]);
        let lldiscrim_a = Load(cx, lldiscrim_a_ptr);

        // NB: we must hit the discriminant first so that structural
        // comparison know not to proceed when the discriminants differ.
        cx = f(cx, lldiscrim_a_ptr, ty::mk_int(cx.tcx()));
        let unr_cx = sub_block(cx, "enum-iter-unr");
        Unreachable(unr_cx);
        let llswitch = Switch(cx, lldiscrim_a, unr_cx.llbb, n_variants);
        let next_cx = sub_block(cx, "enum-iter-next");
        for variant: ty::variant_info in *variants {
            let variant_cx =
                sub_block(cx,
                                   "enum-iter-variant-" +
                                       int::to_str(variant.disr_val, 10u));
            AddCase(llswitch, C_int(ccx, variant.disr_val), variant_cx.llbb);
            variant_cx =
                iter_variant(variant_cx, llunion_a_ptr, variant, tps, tid, f);
            Br(variant_cx, next_cx.llbb);
        }
        ret next_cx;
      }
      ty::ty_class(did, tps) {
          // a class is like a record type
        let i: int = 0;
        for fld: ty::field in ty::class_items_as_fields(cx.tcx(), did) {
            let llfld_a = GEPi(cx, av, [0, i]);
            cx = f(cx, llfld_a, fld.mt.ty);
            i += 1;
        }
      }
      _ { cx.sess().unimpl("type in iter_structural_ty"); }
    }
    ret cx;
}

fn lazily_emit_all_tydesc_glue(ccx: @crate_ctxt,
                               static_ti: option<@tydesc_info>) {
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_take_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_drop_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_free_glue, static_ti);
}

fn lazily_emit_tydesc_glue(ccx: @crate_ctxt, field: int,
                           static_ti: option<@tydesc_info>) {
    alt static_ti {
      none { }
      some(ti) {
        if field == abi::tydesc_field_take_glue {
            alt ti.take_glue {
              some(_) { }
              none {
                #debug("+++ lazily_emit_tydesc_glue TAKE %s",
                       ty_to_str(ccx.tcx, ti.ty));
                let glue_fn = declare_generic_glue
                    (ccx, ti.ty, T_glue_fn(ccx), "take");
                ti.take_glue = some(glue_fn);
                make_generic_glue(ccx, ti.ty, glue_fn,
                                  make_take_glue, "take");
                #debug("--- lazily_emit_tydesc_glue TAKE %s",
                       ty_to_str(ccx.tcx, ti.ty));
              }
            }
        } else if field == abi::tydesc_field_drop_glue {
            alt ti.drop_glue {
              some(_) { }
              none {
                #debug("+++ lazily_emit_tydesc_glue DROP %s",
                       ty_to_str(ccx.tcx, ti.ty));
                let glue_fn =
                    declare_generic_glue(ccx, ti.ty, T_glue_fn(ccx), "drop");
                ti.drop_glue = some(glue_fn);
                make_generic_glue(ccx, ti.ty, glue_fn,
                                  make_drop_glue, "drop");
                #debug("--- lazily_emit_tydesc_glue DROP %s",
                       ty_to_str(ccx.tcx, ti.ty));
              }
            }
        } else if field == abi::tydesc_field_free_glue {
            alt ti.free_glue {
              some(_) { }
              none {
                #debug("+++ lazily_emit_tydesc_glue FREE %s",
                       ty_to_str(ccx.tcx, ti.ty));
                let glue_fn =
                    declare_generic_glue(ccx, ti.ty, T_glue_fn(ccx), "free");
                ti.free_glue = some(glue_fn);
                make_generic_glue(ccx, ti.ty, glue_fn,
                                  make_free_glue, "free");
                #debug("--- lazily_emit_tydesc_glue FREE %s",
                       ty_to_str(ccx.tcx, ti.ty));
              }
            }
        }
      }
    }
}

fn call_tydesc_glue_full(cx: block, v: ValueRef, tydesc: ValueRef,
                         field: int, static_ti: option<@tydesc_info>) {
    lazily_emit_tydesc_glue(cx.ccx(), field, static_ti);
    if cx.unreachable { ret; }

    let static_glue_fn = none;
    alt static_ti {
      none {/* no-op */ }
      some(sti) {
        if field == abi::tydesc_field_take_glue {
            static_glue_fn = sti.take_glue;
        } else if field == abi::tydesc_field_drop_glue {
            static_glue_fn = sti.drop_glue;
        } else if field == abi::tydesc_field_free_glue {
            static_glue_fn = sti.free_glue;
        }
      }
    }

    let llrawptr = PointerCast(cx, v, T_ptr(T_i8()));
    let lltydescs =
        GEPi(cx, tydesc, [0, abi::tydesc_field_first_param]);
    lltydescs = Load(cx, lltydescs);

    let llfn;
    alt static_glue_fn {
      none {
        let llfnptr = GEPi(cx, tydesc, [0, field]);
        llfn = Load(cx, llfnptr);
      }
      some(sgf) { llfn = sgf; }
    }

    Call(cx, llfn, [C_null(T_ptr(T_nil())), C_null(T_ptr(T_nil())),
                    lltydescs, llrawptr]);
}

fn call_tydesc_glue(cx: block, v: ValueRef, t: ty::t, field: int) ->
   block {
    let ti: option<@tydesc_info> = none::<@tydesc_info>;
    let {bcx: bcx, val: td} = get_tydesc(cx, t, ti);
    call_tydesc_glue_full(bcx, v, td, field, ti);
    ret bcx;
}

fn call_cmp_glue(cx: block, lhs: ValueRef, rhs: ValueRef, t: ty::t,
                 llop: ValueRef) -> result {
    // We can't use call_tydesc_glue_full() and friends here because compare
    // glue has a special signature.

    let bcx = cx;

    let r = spill_if_immediate(bcx, lhs, t);
    let lllhs = r.val;
    bcx = r.bcx;
    r = spill_if_immediate(bcx, rhs, t);
    let llrhs = r.val;
    bcx = r.bcx;

    let llrawlhsptr = BitCast(bcx, lllhs, T_ptr(T_i8()));
    let llrawrhsptr = BitCast(bcx, llrhs, T_ptr(T_i8()));
    r = get_tydesc_simple(bcx, t);
    let lltydesc = r.val;
    bcx = r.bcx;
    let lltydescs =
        GEPi(bcx, lltydesc, [0, abi::tydesc_field_first_param]);
    lltydescs = Load(bcx, lltydescs);

    let llfn = bcx.ccx().upcalls.cmp_type;

    let llcmpresultptr = alloca(bcx, T_i1());
    Call(bcx, llfn, [llcmpresultptr, lltydesc, lltydescs,
                     llrawlhsptr, llrawrhsptr, llop]);
    ret rslt(bcx, Load(bcx, llcmpresultptr));
}

fn take_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    if ty::type_needs_drop(cx.tcx(), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_take_glue);
    }
    ret cx;
}

fn drop_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    if ty::type_needs_drop(cx.tcx(), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_drop_glue);
    }
    ret cx;
}

fn drop_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> block {
    alt ty::get(t).struct {
      ty::ty_uniq(_) | ty::ty_vec(_) | ty::ty_str { free_ty(bcx, v, t) }
      ty::ty_box(_) | ty::ty_opaque_box {
        decr_refcnt_maybe_free(bcx, v, t)
      }
      _ { bcx.tcx().sess.bug("drop_ty_immediate: non-box ty"); }
    }
}

fn take_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> result {
    alt ty::get(t).struct {
      ty::ty_box(_) | ty::ty_opaque_box {
        rslt(incr_refcnt_of_boxed(bcx, v), v)
      }
      ty::ty_uniq(_) {
        uniq::duplicate(bcx, v, t)
      }
      ty::ty_str | ty::ty_vec(_) { tvec::duplicate(bcx, v, t) }
      _ { rslt(bcx, v) }
    }
}

fn free_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    if ty::type_needs_drop(cx.tcx(), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_free_glue);
    }
    ret cx;
}

fn call_memmove(cx: block, dst: ValueRef, src: ValueRef,
                n_bytes: ValueRef) -> result {
    // FIXME: Provide LLVM with better alignment information when the
    // alignment is statically known (it must be nothing more than a constant
    // int, or LLVM complains -- not even a constant element of a tydesc
    // works).

    let ccx = cx.ccx();
    let key = alt ccx.sess.targ_cfg.arch {
      session::arch_x86 | session::arch_arm { "llvm.memmove.p0i8.p0i8.i32" }
      session::arch_x86_64 { "llvm.memmove.p0i8.p0i8.i64" }
    };
    let i = ccx.intrinsics;
    assert (i.contains_key(key));
    let memmove = i.get(key);
    let src_ptr = PointerCast(cx, src, T_ptr(T_i8()));
    let dst_ptr = PointerCast(cx, dst, T_ptr(T_i8()));
    let size = IntCast(cx, n_bytes, ccx.int_type);
    let align = C_i32(1i32);
    let volatile = C_bool(false);
    let ret_val = Call(cx, memmove, [dst_ptr, src_ptr, size,
                                     align, volatile]);
    ret rslt(cx, ret_val);
}

fn memmove_ty(bcx: block, dst: ValueRef, src: ValueRef, t: ty::t) ->
    block {
    let ccx = bcx.ccx();
    if ty::type_is_structural(t) {
        let llsz = llsize_of(ccx, type_of(ccx, t));
        ret call_memmove(bcx, dst, src, llsz).bcx;
    }
    Store(bcx, Load(bcx, src), dst);
    ret bcx;
}

enum copy_action { INIT, DROP_EXISTING, }

// These are the types that are passed by pointer.
fn type_is_structural_or_param(t: ty::t) -> bool {
    if ty::type_is_structural(t) { ret true; }
    alt ty::get(t).struct {
      ty::ty_param(_, _) { ret true; }
      _ { ret false; }
    }
}

fn copy_val(cx: block, action: copy_action, dst: ValueRef,
            src: ValueRef, t: ty::t) -> block {
    if action == DROP_EXISTING &&
        (type_is_structural_or_param(t) ||
         ty::type_is_unique(t)) {
        let dstcmp = load_if_immediate(cx, dst, t);
        let cast = PointerCast(cx, dstcmp, val_ty(src));
        // Self-copy check
        with_cond(cx, ICmp(cx, lib::llvm::IntNE, cast, src)) {|bcx|
            copy_val_no_check(bcx, action, dst, src, t)
        }
    } else {
        copy_val_no_check(cx, action, dst, src, t)
    }
}

fn copy_val_no_check(bcx: block, action: copy_action, dst: ValueRef,
                     src: ValueRef, t: ty::t) -> block {
    let ccx = bcx.ccx(), bcx = bcx;
    if ty::type_is_scalar(t) {
        Store(bcx, src, dst);
        ret bcx;
    }
    if ty::type_is_nil(t) || ty::type_is_bot(t) { ret bcx; }
    if ty::type_is_boxed(t) || ty::type_is_vec(t) ||
       ty::type_is_unique_box(t) {
        if action == DROP_EXISTING { bcx = drop_ty(bcx, dst, t); }
        Store(bcx, src, dst);
        ret take_ty(bcx, dst, t);
    }
    if type_is_structural_or_param(t) {
        if action == DROP_EXISTING { bcx = drop_ty(bcx, dst, t); }
        bcx = memmove_ty(bcx, dst, src, t);
        ret take_ty(bcx, dst, t);
    }
    ccx.sess.bug("unexpected type in trans::copy_val_no_check: " +
                     ty_to_str(ccx.tcx, t));
}


// This works like copy_val, except that it deinitializes the source.
// Since it needs to zero out the source, src also needs to be an lval.
// FIXME: We always zero out the source. Ideally we would detect the
// case where a variable is always deinitialized by block exit and thus
// doesn't need to be dropped.
fn move_val(cx: block, action: copy_action, dst: ValueRef,
            src: lval_result, t: ty::t) -> block {
    let src_val = src.val;
    let tcx = cx.tcx(), cx = cx;
    if ty::type_is_scalar(t) {
        if src.kind == owned { src_val = Load(cx, src_val); }
        Store(cx, src_val, dst);
        ret cx;
    } else if ty::type_is_nil(t) || ty::type_is_bot(t) {
        ret cx;
    } else if ty::type_is_boxed(t) || ty::type_is_unique(t) {
        if src.kind == owned { src_val = Load(cx, src_val); }
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t); }
        Store(cx, src_val, dst);
        if src.kind == owned { ret zero_alloca(cx, src.val, t); }
        // If we're here, it must be a temporary.
        revoke_clean(cx, src_val);
        ret cx;
    } else if type_is_structural_or_param(t) {
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t); }
        cx = memmove_ty(cx, dst, src_val, t);
        if src.kind == owned { ret zero_alloca(cx, src_val, t); }
        // If we're here, it must be a temporary.
        revoke_clean(cx, src_val);
        ret cx;
    }
    cx.sess().bug("unexpected type in trans::move_val: " +
                  ty_to_str(tcx, t));
}

fn store_temp_expr(cx: block, action: copy_action, dst: ValueRef,
                   src: lval_result, t: ty::t, last_use: bool)
    -> block {
    // Lvals in memory are not temporaries. Copy them.
    if src.kind != temporary && !last_use {
        let v = if src.kind == owned {
                    load_if_immediate(cx, src.val, t)
                } else {
                    src.val
                };
        ret copy_val(cx, action, dst, v, t);
    }
    ret move_val(cx, action, dst, src, t);
}

fn trans_crate_lit(cx: @crate_ctxt, lit: ast::lit) -> ValueRef {
    alt lit.node {
      ast::lit_int(i, t) { C_integral(T_int_ty(cx, t), i as u64, True) }
      ast::lit_uint(u, t) { C_integral(T_uint_ty(cx, t), u, False) }
      ast::lit_float(fs, t) { C_floating(fs, T_float_ty(cx, t)) }
      ast::lit_bool(b) { C_bool(b) }
      ast::lit_nil { C_nil() }
      ast::lit_str(s) {
        cx.sess.span_unimpl(lit.span, "unique string in this context");
      }
    }
}

fn trans_lit(cx: block, lit: ast::lit, dest: dest) -> block {
    if dest == ignore { ret cx; }
    alt lit.node {
      ast::lit_str(s) { ret tvec::trans_str(cx, s, dest); }
      _ {
        ret store_in_dest(cx, trans_crate_lit(cx.ccx(), lit), dest);
      }
    }
}

fn trans_unary(bcx: block, op: ast::unop, e: @ast::expr,
               un_expr: @ast::expr, dest: dest) -> block {
    // Check for user-defined method call
    alt bcx.ccx().maps.method_map.find(un_expr.id) {
      some(origin) {
        let callee_id = ast_util::op_expr_callee_id(un_expr);
        let fty = node_id_type(bcx, callee_id);
        ret trans_call_inner(bcx, fty, {|bcx|
            impl::trans_method_callee(bcx, callee_id, e, origin)
        }, [], un_expr.id, dest);
      }
      _ {}
    }

    if dest == ignore { ret trans_expr(bcx, e, ignore); }
    let e_ty = expr_ty(bcx, e);
    alt op {
      ast::not {
        let {bcx, val} = trans_temp_expr(bcx, e);
        ret store_in_dest(bcx, Not(bcx, val), dest);
      }
      ast::neg {
        let {bcx, val} = trans_temp_expr(bcx, e);
        let neg = if ty::type_is_fp(e_ty) {
            FNeg(bcx, val)
        } else { Neg(bcx, val) };
        ret store_in_dest(bcx, neg, dest);
      }
      ast::box(_) {
        let {bcx, box, body} = trans_malloc_boxed(bcx, e_ty);
        add_clean_free(bcx, box, false);
        // Cast the body type to the type of the value. This is needed to
        // make enums work, since enums have a different LLVM type depending
        // on whether they're boxed or not
        let ccx = bcx.ccx();
        let llety = T_ptr(type_of(ccx, e_ty));
        body = PointerCast(bcx, body, llety);
        bcx = trans_expr_save_in(bcx, e, body);
        revoke_clean(bcx, box);
        ret store_in_dest(bcx, box, dest);
      }
      ast::uniq(_) {
        ret uniq::trans_uniq(bcx, e, un_expr.id, dest);
      }
      ast::deref {
        bcx.sess().bug("deref expressions should have been \
                               translated using trans_lval(), not \
                               trans_unary()");
      }
    }
}

fn trans_addr_of(cx: block, e: @ast::expr, dest: dest) -> block {
    let {bcx, val, kind} = trans_temp_lval(cx, e);
    let ety = expr_ty(cx, e);
    let is_immediate = ty::type_is_immediate(ety);
    if (kind == temporary && is_immediate) || kind == owned_imm {
        let {bcx: bcx2, val: val2} = do_spill(cx, val, ety);
        bcx = bcx2; val = val2;
    }
    ret store_in_dest(bcx, val, dest);
}

fn trans_compare(cx: block, op: ast::binop, lhs: ValueRef,
                 _lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> result {
    if ty::type_is_scalar(rhs_t) {
      let rs = compare_scalar_types(cx, lhs, rhs, rhs_t, op);
      ret rslt(rs.bcx, rs.val);
    }

    // Determine the operation we need.
    let llop;
    alt op {
      ast::eq | ast::ne { llop = C_u8(abi::cmp_glue_op_eq); }
      ast::lt | ast::ge { llop = C_u8(abi::cmp_glue_op_lt); }
      ast::le | ast::gt { llop = C_u8(abi::cmp_glue_op_le); }
      _ { cx.tcx().sess.bug("trans_compare got non-comparison-op"); }
    }

    let rs = call_cmp_glue(cx, lhs, rhs, rhs_t, llop);

    // Invert the result if necessary.
    alt op {
      ast::eq | ast::lt | ast::le { ret rslt(rs.bcx, rs.val); }
      ast::ne | ast::ge | ast::gt {
        ret rslt(rs.bcx, Not(rs.bcx, rs.val));
      }
      _ { cx.tcx().sess.bug("trans_compare got\
              non-comparison-op"); }
    }
}

fn cast_shift_expr_rhs(cx: block, op: ast::binop,
                       lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs,
                   bind Trunc(cx, _, _), bind ZExt(cx, _, _))
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
            // FIXME: If shifting by negative values becomes not undefined
            // then this is wrong.
            zext(rhs, lhs_llty)
        } else {
            rhs
        }
    } else {
        rhs
    }
}

// Important to get types for both lhs and rhs, because one might be _|_
// and the other not.
fn trans_eager_binop(cx: block, op: ast::binop, lhs: ValueRef,
                     lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t, dest: dest)
    -> block {
    if dest == ignore { ret cx; }
    let intype = lhs_t;
    if ty::type_is_bot(intype) { intype = rhs_t; }
    let is_float = ty::type_is_fp(intype);

    let rhs = cast_shift_expr_rhs(cx, op, lhs, rhs);

    if op == ast::add && ty::type_is_sequence(intype) {
        ret tvec::trans_add(cx, intype, lhs, rhs, dest);
    }
    let cx = cx, val = alt op {
      ast::add {
        if is_float { FAdd(cx, lhs, rhs) }
        else { Add(cx, lhs, rhs) }
      }
      ast::subtract {
        if is_float { FSub(cx, lhs, rhs) }
        else { Sub(cx, lhs, rhs) }
      }
      ast::mul {
        if is_float { FMul(cx, lhs, rhs) }
        else { Mul(cx, lhs, rhs) }
      }
      ast::div {
        if is_float { FDiv(cx, lhs, rhs) }
        else if ty::type_is_signed(intype) {
            SDiv(cx, lhs, rhs)
        } else { UDiv(cx, lhs, rhs) }
      }
      ast::rem {
        if is_float { FRem(cx, lhs, rhs) }
        else if ty::type_is_signed(intype) {
            SRem(cx, lhs, rhs)
        } else { URem(cx, lhs, rhs) }
      }
      ast::bitor { Or(cx, lhs, rhs) }
      ast::bitand { And(cx, lhs, rhs) }
      ast::bitxor { Xor(cx, lhs, rhs) }
      ast::lsl { Shl(cx, lhs, rhs) }
      ast::lsr { LShr(cx, lhs, rhs) }
      ast::asr { AShr(cx, lhs, rhs) }
      _ {
        let cmpr = trans_compare(cx, op, lhs, lhs_t, rhs, rhs_t);
        cx = cmpr.bcx;
        cmpr.val
      }
    };
    ret store_in_dest(cx, val, dest);
}

fn trans_assign_op(bcx: block, ex: @ast::expr, op: ast::binop,
                   dst: @ast::expr, src: @ast::expr) -> block {
    let t = expr_ty(bcx, src);
    let lhs_res = trans_lval(bcx, dst);
    assert (lhs_res.kind == owned);

    // A user-defined operator method
    alt bcx.ccx().maps.method_map.find(ex.id) {
      some(origin) {
        let callee_id = ast_util::op_expr_callee_id(ex);
        let fty = node_id_type(bcx, callee_id);
        ret trans_call_inner(bcx, fty, {|bcx|
            // FIXME provide the already-computed address, not the expr
            impl::trans_method_callee(bcx, callee_id, dst, origin)
        }, [src], ex.id, save_in(lhs_res.val));
      }
      _ {}
    }

    // Special case for `+= [x]`
    alt ty::get(t).struct {
      ty::ty_vec(_) {
        alt src.node {
          ast::expr_vec(args, _) {
            ret tvec::trans_append_literal(lhs_res.bcx,
                                           lhs_res.val, t, args);
          }
          _ { }
        }
      }
      _ { }
    }
    let {bcx, val: rhs_val} = trans_temp_expr(lhs_res.bcx, src);
    if ty::type_is_sequence(t) {
        alt op {
          ast::add {
            ret tvec::trans_append(bcx, t, lhs_res.val, rhs_val);
          }
          _ { }
        }
    }
    ret trans_eager_binop(bcx, op, Load(bcx, lhs_res.val), t, rhs_val, t,
                          save_in(lhs_res.val));
}

fn autoderef(cx: block, v: ValueRef, t: ty::t) -> result_t {
    let v1: ValueRef = v;
    let t1: ty::t = t;
    let ccx = cx.ccx();
    loop {
        alt ty::get(t1).struct {
          ty::ty_box(mt) {
            let body = GEPi(cx, v1, [0, abi::box_field_body]);
            t1 = mt.ty;

            // Since we're changing levels of box indirection, we may have
            // to cast this pointer, since statically-sized enum types have
            // different types depending on whether they're behind a box
            // or not.
            let llty = type_of(ccx, t1);
            v1 = PointerCast(cx, body, T_ptr(llty));
          }
          ty::ty_uniq(_) {
            let derefed = uniq::autoderef(v1, t1);
            t1 = derefed.t;
            v1 = derefed.v;
          }
          ty::ty_rptr(_, mt) {
            t1 = mt.ty;
            v1 = v;
          }
          ty::ty_res(did, inner, tps) {
            t1 = ty::substitute_type_params(ccx.tcx, tps, inner);
            v1 = GEPi(cx, v1, [0, 1]);
          }
          ty::ty_enum(did, tps) {
            let variants = ty::enum_variants(ccx.tcx, did);
            if (*variants).len() != 1u || variants[0].args.len() != 1u {
                break;
            }
            t1 =
                ty::substitute_type_params(ccx.tcx, tps, variants[0].args[0]);
            v1 = PointerCast(cx, v1, T_ptr(type_of(ccx, t1)));
          }
          _ { break; }
        }
        v1 = load_if_immediate(cx, v1, t1);
    }
    ret {bcx: cx, val: v1, ty: t1};
}

// refinement types would obviate the need for this
enum lazy_binop_ty { lazy_and, lazy_or }

fn trans_lazy_binop(bcx: block, op: lazy_binop_ty, a: @ast::expr,
                    b: @ast::expr, dest: dest) -> block {

    let {bcx: past_lhs, val: lhs} = with_scope_result(bcx, "lhs")
        {|bcx| trans_temp_expr(bcx, a)};
    if past_lhs.unreachable { ret past_lhs; }
    let join = sub_block(bcx, "join"), before_rhs = sub_block(bcx, "rhs");

    alt op {
      lazy_and { CondBr(past_lhs, lhs, before_rhs.llbb, join.llbb); }
      lazy_or { CondBr(past_lhs, lhs, join.llbb, before_rhs.llbb); }
    }
    let {bcx: past_rhs, val: rhs} = with_scope_result(before_rhs, "rhs")
        {|bcx| trans_temp_expr(bcx, b)};

    if past_rhs.unreachable { ret store_in_dest(join, lhs, dest); }
    Br(past_rhs, join.llbb);
    let phi = Phi(join, T_bool(), [lhs, rhs], [past_lhs.llbb, past_rhs.llbb]);
    ret store_in_dest(join, phi, dest);
}

fn trans_binary(bcx: block, op: ast::binop, lhs: @ast::expr,
                rhs: @ast::expr, dest: dest, ex: @ast::expr) -> block {
    // User-defined operators
    alt bcx.ccx().maps.method_map.find(ex.id) {
      some(origin) {
        let callee_id = ast_util::op_expr_callee_id(ex);
        let fty = node_id_type(bcx, callee_id);
        ret trans_call_inner(bcx, fty, {|bcx|
            impl::trans_method_callee(bcx, callee_id, lhs, origin)
        }, [rhs], ex.id, dest);
      }
      _ {}
    }

    // First couple cases are lazy:
    alt op {
      ast::and {
        ret trans_lazy_binop(bcx, lazy_and, lhs, rhs, dest);
      }
      ast::or {
        ret trans_lazy_binop(bcx, lazy_or, lhs, rhs, dest);
      }
      _ {
        // Remaining cases are eager:
        let lhs_res = trans_temp_expr(bcx, lhs);
        let rhs_res = trans_temp_expr(lhs_res.bcx, rhs);
        ret trans_eager_binop(rhs_res.bcx, op, lhs_res.val,
                              expr_ty(bcx, lhs), rhs_res.val,
                              expr_ty(bcx, rhs), dest);
      }
    }
}

fn trans_if(cx: block, cond: @ast::expr, thn: ast::blk,
            els: option<@ast::expr>, dest: dest)
    -> block {
    let {bcx, val: cond_val} = trans_temp_expr(cx, cond);

    let then_dest = dup_for_join(dest);
    let else_dest = dup_for_join(dest);
    let then_cx = scope_block(bcx, "then");
    then_cx.block_span = some(thn.span);
    let else_cx = scope_block(bcx, "else");
    option::may(els) {|e| else_cx.block_span = some(e.span); }
    CondBr(bcx, cond_val, then_cx.llbb, else_cx.llbb);
    let then_bcx = trans_block(then_cx, thn, then_dest);
    then_bcx = trans_block_cleanups(then_bcx, then_cx);
    // Calling trans_block directly instead of trans_expr
    // because trans_expr will create another scope block
    // context for the block, but we've already got the
    // 'else' context
    let else_bcx = alt els {
      some(elexpr) {
        alt elexpr.node {
          ast::expr_if(_, _, _) {
            let elseif_blk = ast_util::block_from_expr(elexpr);
            trans_block(else_cx, elseif_blk, else_dest)
          }
          ast::expr_block(blk) {
            trans_block(else_cx, blk, else_dest)
          }
          // would be nice to have a constraint on ifs
          _ { cx.tcx().sess.bug("strange alternative in if"); }
        }
      }
      _ { else_cx }
    };
    else_bcx = trans_block_cleanups(else_bcx, else_cx);
    ret join_returns(cx, [then_bcx, else_bcx], [then_dest, else_dest], dest);
}

fn trans_for(cx: block, local: @ast::local, seq: @ast::expr,
             body: ast::blk) -> block {
    fn inner(bcx: block, local: @ast::local, curr: ValueRef, t: ty::t,
             body: ast::blk, outer_next_cx: block) -> block {
        let next_cx = sub_block(bcx, "next");
        let scope_cx = loop_scope_block(bcx, cont_other(next_cx),
                                        outer_next_cx, "for loop scope",
                                        body.span);
        Br(bcx, scope_cx.llbb);
        let curr = PointerCast(bcx, curr,
                               T_ptr(type_of(bcx.ccx(), t)));
        let bcx = alt::bind_irrefutable_pat(scope_cx, local.node.pat,
                                                  curr, false);
        bcx = trans_block(bcx, body, ignore);
        cleanup_and_Br(bcx, scope_cx, next_cx.llbb);
        ret next_cx;
    }
    let ccx = cx.ccx();
    let next_cx = sub_block(cx, "next");
    let seq_ty = expr_ty(cx, seq);
    let {bcx: bcx, val: seq} = trans_temp_expr(cx, seq);
    let seq = PointerCast(bcx, seq, T_ptr(ccx.opaque_vec_type));
    let fill = tvec::get_fill(bcx, seq);
    if ty::type_is_str(seq_ty) {
        fill = Sub(bcx, fill, C_int(ccx, 1));
    }
    let bcx = tvec::iter_vec_raw(bcx, seq, seq_ty, fill,
                                 bind inner(_, local, _, _, body, next_cx));
    Br(bcx, next_cx.llbb);
    ret next_cx;
}

fn trans_while(cx: block, cond: @ast::expr, body: ast::blk)
    -> block {
    let next_cx = sub_block(cx, "while next");
    let loop_cx = loop_scope_block(cx, cont_self, next_cx,
                                   "while loop", body.span);
    let cond_cx = scope_block(loop_cx, "while loop cond");
    let body_cx = scope_block(loop_cx, "while loop body");
    Br(cx, loop_cx.llbb);
    Br(loop_cx, cond_cx.llbb);
    let cond_res = trans_temp_expr(cond_cx, cond);
    let cond_bcx = trans_block_cleanups(cond_res.bcx, cond_cx);
    CondBr(cond_bcx, cond_res.val, body_cx.llbb, next_cx.llbb);
    let body_end = trans_block(body_cx, body, ignore);
    cleanup_and_Br(body_end, body_cx, cond_cx.llbb);
    ret next_cx;
}

fn trans_do_while(cx: block, body: ast::blk, cond: @ast::expr) ->
    block {
    let next_cx = sub_block(cx, "next");
    let body_cx =
        loop_scope_block(cx, cont_self, next_cx,
                                  "do-while loop body", body.span);
    let body_end = trans_block(body_cx, body, ignore);
    let cond_cx = scope_block(body_cx, "do-while cond");
    cleanup_and_Br(body_end, body_cx, cond_cx.llbb);
    let cond_res = trans_temp_expr(cond_cx, cond);
    let cond_bcx = trans_block_cleanups(cond_res.bcx, cond_cx);
    CondBr(cond_bcx, cond_res.val, body_cx.llbb, next_cx.llbb);
    Br(cx, body_cx.llbb);
    ret next_cx;
}

fn trans_loop(cx:block, body: ast::blk) -> block {
    let next_cx = sub_block(cx, "next");
    let body_cx =
        loop_scope_block(cx, cont_self, next_cx,
                                  "infinite loop body", body.span);
    let body_end = trans_block(body_cx, body, ignore);
    cleanup_and_Br(body_end, body_cx, body_cx.llbb);
    Br(cx, body_cx.llbb);
    ret next_cx;
}

enum lval_kind {
    temporary, //< Temporary value passed by value if of immediate type
    owned,     //< Non-temporary value passed by pointer
    owned_imm, //< Non-temporary value passed by value
}
type local_var_result = {val: ValueRef, kind: lval_kind};
type lval_result = {bcx: block, val: ValueRef, kind: lval_kind};
enum callee_env {
    null_env,
    is_closure,
    self_env(ValueRef, ty::t, option<ValueRef>),
}
type lval_maybe_callee = {bcx: block,
                          val: ValueRef,
                          kind: lval_kind,
                          env: callee_env,
                          // Tydescs to pass. Only used to call intrinsics
                          tds: option<[ValueRef]>};

fn null_env_ptr(bcx: block) -> ValueRef {
    C_null(T_opaque_box_ptr(bcx.ccx()))
}

fn lval_from_local_var(bcx: block, r: local_var_result) -> lval_result {
    ret { bcx: bcx, val: r.val, kind: r.kind };
}

fn lval_owned(bcx: block, val: ValueRef) -> lval_result {
    ret {bcx: bcx, val: val, kind: owned};
}
fn lval_temp(bcx: block, val: ValueRef) -> lval_result {
    ret {bcx: bcx, val: val, kind: temporary};
}

fn lval_no_env(bcx: block, val: ValueRef, kind: lval_kind)
    -> lval_maybe_callee {
    ret {bcx: bcx, val: val, kind: kind, env: is_closure, tds: none};
}

fn trans_external_path(ccx: @crate_ctxt, did: ast::def_id, t: ty::t)
    -> ValueRef {
    let name = csearch::get_symbol(ccx.sess.cstore, did);
    let llty = alt ty::get(t).struct {
      ty::ty_fn(_) { type_of_fn_from_ty(ccx, t) }
      _ { type_of(ccx, t) }
    };
    ret get_extern_const(ccx.externs, ccx.llmod, name, llty);
}

fn normalize_for_monomorphization(tcx: ty::ctxt, ty: ty::t) -> option<ty::t> {
    // FIXME[mono] could do this recursively. is that worthwhile?
    alt ty::get(ty).struct {
      ty::ty_box(mt) { some(ty::mk_opaque_box(tcx)) }
      ty::ty_fn(fty) { some(ty::mk_fn(tcx, {proto: fty.proto,
                                            inputs: [],
                                            output: ty::mk_nil(tcx),
                                            ret_style: ast::return_val,
                                            constraints: []})) }
      ty::ty_iface(_, _) { some(ty::mk_fn(tcx, {proto: ast::proto_box,
                                                inputs: [],
                                                output: ty::mk_nil(tcx),
                                                ret_style: ast::return_val,
                                                constraints: []})) }
      ty::ty_ptr(_) { some(ty::mk_uint(tcx)) }
      _ { none }
    }
}

fn make_mono_id(ccx: @crate_ctxt, item: ast::def_id, substs: [ty::t],
                vtables: option<typeck::vtable_res>,
                param_uses: option<[type_use::type_uses]>) -> mono_id {
    let precise_param_ids = alt vtables {
      some(vts) {
        let bounds = ty::lookup_item_type(ccx.tcx, item).bounds;
        let i = 0u;
        vec::map2(*bounds, substs, {|bounds, subst|
            let v = [];
            for bound in *bounds {
                alt bound {
                  ty::bound_iface(_) {
                    v += [impl::vtable_id(ccx, vts[i])];
                    i += 1u;
                  }
                  _ {}
                }
            }
            mono_precise(subst, if v.len() > 0u { some(v) } else { none })
        })
      }
      none {
        vec::map(substs, {|subst| mono_precise(subst, none)})
      }
    };
    let param_ids = alt param_uses {
      some(uses) {
        vec::map2(precise_param_ids, uses, {|id, uses|
            alt check id {
              mono_precise(_, some(_)) { id }
              mono_precise(subst, none) {
                if uses == 0u { mono_any }
                else if uses == type_use::use_repr &&
                        !ty::type_needs_drop(ccx.tcx, subst) {
                    let llty = type_of(ccx, subst);
                    let size = shape::llsize_of_real(ccx, llty);
                    let align = shape::llalign_of_real(ccx, llty);
                    // Special value for nil to prevent problems with undef
                    // return pointers.
                    if size == 1u && ty::type_is_nil(subst) {
                        mono_repr(0u, 0u)
                    } else { mono_repr(size, align) }
                } else { id }
              }
            }
        })
      }
      none { precise_param_ids }
    };
    @{def: item, params: param_ids}
}

fn monomorphic_fn(ccx: @crate_ctxt, fn_id: ast::def_id, real_substs: [ty::t],
                  vtables: option<typeck::vtable_res>)
    -> {val: ValueRef, must_cast: bool, intrinsic: bool} {
    let mut must_cast = false;
    let substs = vec::map(real_substs, {|t|
        alt normalize_for_monomorphization(ccx.tcx, t) {
          some(t) { must_cast = true; t }
          none { t }
        }
    });

    let param_uses = type_use::type_uses_for(ccx, fn_id, substs.len());
    let hash_id = make_mono_id(ccx, fn_id, substs, vtables, some(param_uses));
    if vec::any(hash_id.params,
                {|p| alt p { mono_precise(_, _) { false } _ { true } } }) {
        must_cast = true;
    }
    alt ccx.monomorphized.find(hash_id) {
      some(val) {
        ret {val: val, must_cast: must_cast, intrinsic: false};
      }
      none {}
    }

    let tpt = ty::lookup_item_type(ccx.tcx, fn_id);
    let item_ty = tpt.ty;

    let map_node = ccx.tcx.items.get(fn_id.node);
    // Get the path so that we can create a symbol
    let (pt, name) = alt map_node {
      ast_map::node_item(i, pt) {
        alt i.node {
          ast::item_res(_, _, _, dtor_id, _) {
            item_ty = ty::node_id_to_type(ccx.tcx, dtor_id);
          }
          _ {}
        }
        (pt, i.ident)
      }
      ast_map::node_variant(v, _, pt) { (pt, v.node.name) }
      ast_map::node_method(m, _, pt) { (pt, m.ident) }
      ast_map::node_native_item(_, abi, _) {
        // Natives don't have to be monomorphized.
        ret {val: get_item_val(ccx, fn_id.node),
             must_cast: true,
             intrinsic: abi == ast::native_abi_rust_intrinsic};
      }
      ast_map::node_ctor(i) {
        alt check ccx.tcx.items.get(i.id) {
          ast_map::node_item(i, pt) { (pt, i.ident) }
        }
      }
      _ { fail "unexpected node type"; }
    };
    let mono_ty = ty::substitute_type_params(ccx.tcx, substs, item_ty);
    let llfty = type_of_fn_from_ty(ccx, mono_ty);

    let pt = *pt + [path_name(ccx.names(name))];
    let s = mangle_exported_name(ccx, pt, mono_ty);
    let lldecl = decl_cdecl_fn(ccx.llmod, s, llfty);
    ccx.monomorphized.insert(hash_id, lldecl);

    let psubsts = some({tys: substs, vtables: vtables, bounds: tpt.bounds});
    alt check map_node {
      ast_map::node_item(i@@{node: ast::item_fn(decl, _, body), _}, _) {
        set_inline_hint_if_appr(i.attrs, lldecl);
        trans_fn(ccx, pt, decl, body, lldecl, no_self, psubsts, fn_id.node,
                 none);
      }
      ast_map::node_item(@{node: ast::item_res(d, _, body, d_id, _), _}, _) {
        trans_fn(ccx, pt, d, body, lldecl, no_self, psubsts, d_id, none);
      }
      ast_map::node_variant(v, enum_item, _) {
        let tvs = ty::enum_variants(ccx.tcx, local_def(enum_item.id));
        let this_tv = option::get(vec::find(*tvs, {|tv|
            tv.id.node == fn_id.node}));
        set_inline_hint(lldecl);
        trans_enum_variant(ccx, enum_item.id, v, this_tv.disr_val,
                           (*tvs).len() == 1u, psubsts, lldecl);
      }
      ast_map::node_method(mth, impl_def_id, _) {
        set_inline_hint_if_appr(mth.attrs, lldecl);
        let selfty = ty::node_id_to_type(ccx.tcx, mth.self_id);
        let selfty = ty::substitute_type_params(ccx.tcx, substs, selfty);
        trans_fn(ccx, pt, mth.decl, mth.body, lldecl,
                 impl_self(selfty), psubsts, fn_id.node, none);
      }
      ast_map::node_ctor(i) {
        alt check i.node {
          ast::item_res(decl, _, _, _, _) {
            set_inline_hint(lldecl);
            trans_res_ctor(ccx, pt, decl, fn_id.node, psubsts, lldecl);
          }
          ast::item_class(_, _, ctor) {
            ccx.sess.unimpl("monomorphic class constructor");
          }
        }
      }
    }
    {val: lldecl, must_cast: must_cast, intrinsic: false}
}

fn maybe_instantiate_inline(ccx: @crate_ctxt, fn_id: ast::def_id)
    -> ast::def_id {
    alt ccx.external.find(fn_id) {
      some(some(node_id)) {
        // Already inline
        #debug["maybe_instantiate_inline(%s): already inline as node id %d",
               ty::item_path_str(ccx.tcx, fn_id), node_id];
        local_def(node_id)
      }
      some(none) { fn_id } // Not inlinable
      none { // Not seen yet
        alt check csearch::maybe_get_item_ast(ccx.tcx, ccx.maps, fn_id) {
          csearch::not_found {
            ccx.external.insert(fn_id, none);
            fn_id
          }
          csearch::found(ast::ii_item(item)) {
            ccx.external.insert(fn_id, some(item.id));
            trans_item(ccx, *item);
            local_def(item.id)
          }
          csearch::found_parent(parent_id, ast::ii_item(item)) {
            ccx.external.insert(parent_id, some(item.id));
            let my_id = 0;
            alt check item.node {
              ast::item_enum(_, _) {
                let vs_here = ty::enum_variants(ccx.tcx, local_def(item.id));
                let vs_there = ty::enum_variants(ccx.tcx, parent_id);
                vec::iter2(*vs_here, *vs_there) {|here, there|
                    if there.id == fn_id { my_id = here.id.node; }
                    ccx.external.insert(there.id, some(here.id.node));
                }
              }
              ast::item_res(_, _, _, _, ctor_id) { my_id = ctor_id; }
            }
            trans_item(ccx, *item);
            local_def(my_id)
          }
          csearch::found(ast::ii_method(impl_did, mth)) {
            ccx.external.insert(fn_id, some(mth.id));
            compute_ii_method_info(ccx, impl_did, mth) {|ty, bounds, path|
                if bounds.len() == 0u {
                    let llfn = get_item_val(ccx, mth.id);
                    trans_fn(ccx, path, mth.decl, mth.body,
                             llfn, impl_self(ty), none, mth.id, none);
                }
            }
            local_def(mth.id)
          }
        }
      }
    }
}

fn lval_intrinsic_fn(bcx: block, val: ValueRef, tys: [ty::t],
                     id: ast::node_id) -> lval_maybe_callee {
    fn add_tydesc_params(ccx: @crate_ctxt, llfty: TypeRef, n: uint)
        -> TypeRef {
        let out_ty = llvm::LLVMGetReturnType(llfty);
        let n_args = llvm::LLVMCountParamTypes(llfty);
        let args = vec::from_elem(n_args as uint, 0 as TypeRef);
        unsafe { llvm::LLVMGetParamTypes(llfty, vec::unsafe::to_ptr(args)); }
        T_fn(vec::slice(args, 0u, first_real_arg) +
             vec::from_elem(n, T_ptr(ccx.tydesc_type)) +
             vec::tailn(args, first_real_arg), out_ty)
    }

    let bcx = bcx, ccx = bcx.ccx();
    let tds = vec::map(tys, {|t|
        let ti = none, td_res = get_tydesc(bcx, t, ti);
        bcx = td_res.bcx;
        lazily_emit_all_tydesc_glue(ccx, ti);
        td_res.val
    });
    let llfty = type_of_fn_from_ty(ccx, node_id_type(bcx, id));
    let val = PointerCast(bcx, val, T_ptr(add_tydesc_params(
        ccx, llfty, tys.len())));
    {bcx: bcx, val: val, kind: owned, env: null_env, tds: some(tds)}
}

fn lval_static_fn(bcx: block, fn_id: ast::def_id, id: ast::node_id)
    -> lval_maybe_callee {
    let vts = option::map(bcx.ccx().maps.vtable_map.find(id), {|vts|
        impl::resolve_vtables_in_fn_ctxt(bcx.fcx, vts)
    });
    lval_static_fn_inner(bcx, fn_id, id, node_id_type_params(bcx, id), vts)
}

fn lval_static_fn_inner(bcx: block, fn_id: ast::def_id, id: ast::node_id,
                        tys: [ty::t], vtables: option<typeck::vtable_res>)
    -> lval_maybe_callee {
    let ccx = bcx.ccx(), tcx = ccx.tcx;
    let tpt = ty::lookup_item_type(tcx, fn_id);

    // Check whether this fn has an inlined copy and, if so, redirect fn_id to
    // the local id of the inlined copy.
    let fn_id = if fn_id.crate != ast::local_crate {
        maybe_instantiate_inline(ccx, fn_id)
    } else { fn_id };

    if fn_id.crate == ast::local_crate && tys.len() > 0u {
        let {val, must_cast, intrinsic} = monomorphic_fn(ccx, fn_id, tys,
                                                         vtables);
        if intrinsic { ret lval_intrinsic_fn(bcx, val, tys, id); }
        if must_cast {
            val = PointerCast(bcx, val, T_ptr(type_of_fn_from_ty(
                ccx, node_id_type(bcx, id))));
        }
        ret {bcx: bcx, val: val, kind: owned, env: null_env, tds: none};
    }

    let val = if fn_id.crate == ast::local_crate {
        // Internal reference.
        get_item_val(ccx, fn_id.node)
    } else {
        // External reference.
        trans_external_path(ccx, fn_id, tpt.ty)
    };
    if tys.len() > 0u {
        // This is supposed to be an external native function.
        // Unfortunately, I found no easy/cheap way to assert that.
        if csearch::item_is_intrinsic(ccx.sess.cstore, fn_id) {
            ret lval_intrinsic_fn(bcx, val, tys, id);
        }
        val = PointerCast(bcx, val, T_ptr(type_of_fn_from_ty(
            ccx, node_id_type(bcx, id))));
    }

    // FIXME: Need to support external crust functions
    if fn_id.crate == ast::local_crate {
        alt bcx.tcx().def_map.find(id) {
          some(ast::def_fn(_, ast::crust_fn)) {
            // Crust functions are just opaque pointers
            let val = PointerCast(bcx, val, T_ptr(T_i8()));
            ret lval_no_env(bcx, val, owned_imm);
          }
          _ { }
        }
    }

    ret {bcx: bcx, val: val, kind: owned, env: null_env, tds: none};
}

fn lookup_discriminant(ccx: @crate_ctxt, vid: ast::def_id) -> ValueRef {
    alt ccx.discrims.find(vid) {
      none {
        // It's an external discriminant that we haven't seen yet.
        assert (vid.crate != ast::local_crate);
        let sym = csearch::get_symbol(ccx.sess.cstore, vid);
        let gvar = str::as_c_str(sym, {|buf|
            llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
        });
        lib::llvm::SetLinkage(gvar, lib::llvm::ExternalLinkage);
        llvm::LLVMSetGlobalConstant(gvar, True);
        ccx.discrims.insert(vid, gvar);
        ret gvar;
      }
      some(llval) { ret llval; }
    }
}

fn trans_local_var(cx: block, def: ast::def) -> local_var_result {
    fn take_local(table: hashmap<ast::node_id, local_val>,
                  id: ast::node_id) -> local_var_result {
        alt table.find(id) {
          some(local_mem(v)) { {val: v, kind: owned} }
          some(local_imm(v)) { {val: v, kind: owned_imm} }
          r { fail("take_local: internal error"); }
        }
    }
    alt def {
      ast::def_upvar(nid, _, _) {
        assert (cx.fcx.llupvars.contains_key(nid));
        ret { val: cx.fcx.llupvars.get(nid), kind: owned };
      }
      ast::def_arg(nid, _) {
        assert (cx.fcx.llargs.contains_key(nid));
        ret take_local(cx.fcx.llargs, nid);
      }
      ast::def_local(nid, _) | ast::def_binding(nid) {
        assert (cx.fcx.lllocals.contains_key(nid));
        ret take_local(cx.fcx.lllocals, nid);
      }
      ast::def_self(_) {
        let slf = option::get(cx.fcx.llself);
        let ptr = PointerCast(cx, slf.v,
                              T_ptr(type_of(cx.ccx(), slf.t)));
        ret {val: ptr, kind: owned};
      }
      _ {
        cx.sess().unimpl(#fmt("unsupported def type in trans_local_def: %?",
                              def));
      }
    }
}

// The third argument (path) ends up getting used when the id
// refers to a field within the enclosing class, since the name
// gets turned into a record field name.
fn trans_path(cx: block, id: ast::node_id, path: @ast::path)
    -> lval_maybe_callee {
    alt cx.tcx().def_map.find(id) {
      none { cx.sess().bug("trans_path: unbound node ID"); }
      some(df) {
          ret trans_var(cx, df, id, path);
      }
    }
}

fn trans_var(cx: block, def: ast::def, id: ast::node_id, path: @ast::path)
    -> lval_maybe_callee {
    let ccx = cx.ccx();
    alt def {
      ast::def_fn(did, _) {
        ret lval_static_fn(cx, did, id);
      }
      ast::def_variant(tid, vid) {
        if ty::enum_variant_with_id(ccx.tcx, tid, vid).args.len() > 0u {
            // N-ary variant.
            ret lval_static_fn(cx, vid, id);
        } else {
            // Nullary variant.
            let enum_ty = node_id_type(cx, id);
            let alloc_result = alloc_ty(cx, enum_ty);
            let llenumblob = alloc_result.val;
            let llenumty = type_of_enum(ccx, tid, enum_ty);
            let bcx = alloc_result.bcx;
            let llenumptr = PointerCast(bcx, llenumblob, T_ptr(llenumty));
            let lldiscrimptr = GEPi(bcx, llenumptr, [0, 0]);
            let lldiscrim_gv = lookup_discriminant(bcx.fcx.ccx, vid);
            let lldiscrim = Load(bcx, lldiscrim_gv);
            Store(bcx, lldiscrim, lldiscrimptr);
            ret lval_no_env(bcx, llenumptr, temporary);
        }
      }
      ast::def_const(did) {
        if did.crate == ast::local_crate {
            ret lval_no_env(cx, get_item_val(ccx, did.node), owned);
        } else {
            let tp = node_id_type(cx, id);
            let val = trans_external_path(ccx, did, tp);
            ret lval_no_env(cx, load_if_immediate(cx, val, tp), owned_imm);
        }
      }
      ast::def_class_field(parent, did) {
          // base is implicitly "Self"
          alt cx.fcx.self_id {
            some(slf) {
                let {bcx, val, kind} = trans_rec_field(cx, slf,
                    // TODO: only supporting local objects for now
                                          path_to_ident(path));
                ret lval_no_env(bcx, val, kind);
            }
            _ { cx.sess().bug("unbound self param in class"); }
          }
     }
      _ {
        let loc = trans_local_var(cx, def);
        ret lval_no_env(cx, loc.val, loc.kind);
      }
    }
}

fn trans_rec_field(bcx: block, base: @ast::expr,
                   field: ast::ident) -> lval_result {
    let {bcx, val} = trans_temp_expr(bcx, base);
    let {bcx, val, ty} = autoderef(bcx, val, expr_ty(bcx, base));
    let fields = alt ty::get(ty).struct {
            ty::ty_rec(fs) { fs }
            ty::ty_class(did,_) { ty::class_items_as_fields(bcx.tcx(), did) }
            // Constraint?
            _ { bcx.tcx().sess.span_bug(base.span, "trans_rec_field:\
                 base expr has non-record type"); }
        };
    let ix = option::get(ty::field_idx(field, fields));
    let val = GEPi(bcx, val, [0, ix as int]);
    ret {bcx: bcx, val: val, kind: owned};
}

fn trans_index(cx: block, ex: @ast::expr, base: @ast::expr,
               idx: @ast::expr) -> lval_result {
    let base_ty = expr_ty(cx, base);
    let exp = trans_temp_expr(cx, base);
    let lv = autoderef(exp.bcx, exp.val, base_ty);
    let ix = trans_temp_expr(lv.bcx, idx);
    let v = lv.val;
    let bcx = ix.bcx;
    let ccx = cx.ccx();

    // Cast to an LLVM integer. Rust is less strict than LLVM in this regard.
    let ix_val;
    let ix_size = llsize_of_real(cx.ccx(), val_ty(ix.val));
    let int_size = llsize_of_real(cx.ccx(), ccx.int_type);
    if ix_size < int_size {
        ix_val = ZExt(bcx, ix.val, ccx.int_type);
    } else if ix_size > int_size {
        ix_val = Trunc(bcx, ix.val, ccx.int_type);
    } else { ix_val = ix.val; }

    let unit_ty = node_id_type(cx, ex.id);
    let llunitty = type_of(ccx, unit_ty);
    let unit_sz = llsize_of(ccx, llunitty);
    maybe_name_value(cx.ccx(), unit_sz, "unit_sz");
    let scaled_ix = Mul(bcx, ix_val, unit_sz);
    maybe_name_value(cx.ccx(), scaled_ix, "scaled_ix");
    let lim = tvec::get_fill(bcx, v);
    let body = tvec::get_dataptr(bcx, v, type_of(ccx, unit_ty));
    let bounds_check = ICmp(bcx, lib::llvm::IntUGE, scaled_ix, lim);
    bcx = with_cond(bcx, bounds_check) {|bcx|
        // fail: bad bounds check.
        trans_fail(bcx, some(ex.span), "bounds check")
    };
    let elt = InBoundsGEP(bcx, body, [ix_val]);
    ret lval_owned(bcx, PointerCast(bcx, elt, T_ptr(llunitty)));
}

fn expr_is_lval(bcx: block, e: @ast::expr) -> bool {
    let ccx = bcx.ccx();
    ty::expr_is_lval(ccx.maps.method_map, e)
}

fn trans_callee(bcx: block, e: @ast::expr) -> lval_maybe_callee {
    alt e.node {
      ast::expr_path(path) { ret trans_path(bcx, e.id, path); }
      ast::expr_field(base, ident, _) {
        // Lval means this is a record field, so not a method
        if !expr_is_lval(bcx, e) {
            alt bcx.ccx().maps.method_map.find(e.id) {
              some(origin) { // An impl method
                ret impl::trans_method_callee(bcx, e.id, base, origin);
              }
              _ {
                bcx.ccx().sess.span_bug(e.span, "trans_callee: weird expr");
              }
            }
        }
      }
      _ {}
    }
    let lv = trans_temp_lval(bcx, e);
    ret lval_no_env(lv.bcx, lv.val, lv.kind);
}

// Use this when you know you are compiling an lval.
// The additional bool returned indicates whether it's mem (that is
// represented as an alloca or heap, hence needs a 'load' to be used as an
// immediate).
fn trans_lval(cx: block, e: @ast::expr) -> lval_result {
    alt e.node {
      ast::expr_path(p) {
          let v = trans_path(cx, e.id, p);
        ret lval_maybe_callee_to_lval(v, expr_ty(cx, e));
      }
      ast::expr_field(base, ident, _) {
        ret trans_rec_field(cx, base, ident);
      }
      ast::expr_index(base, idx) {
        ret trans_index(cx, e, base, idx);
      }
      ast::expr_unary(ast::deref, base) {
        let ccx = cx.ccx();
        let sub = trans_temp_expr(cx, base);
        let t = expr_ty(cx, base);
        let val = alt check ty::get(t).struct {
          ty::ty_box(_) {
            GEPi(sub.bcx, sub.val, [0, abi::box_field_body])
          }
          ty::ty_res(_, _, _) {
            GEPi(sub.bcx, sub.val, [0, 1])
          }
          ty::ty_enum(_, _) {
            let ety = expr_ty(cx, e);
            let ellty = T_ptr(type_of(ccx, ety));
            PointerCast(sub.bcx, sub.val, ellty)
          }
          ty::ty_ptr(_) | ty::ty_uniq(_) | ty::ty_rptr(_,_) { sub.val }
        };
        ret lval_owned(sub.bcx, val);
      }
      _ { cx.sess().span_bug(e.span, "non-lval in trans_lval"); }
    }
}

fn lval_maybe_callee_to_lval(c: lval_maybe_callee, ty: ty::t) -> lval_result {
    let must_bind = alt c.env { self_env(_, _, _) { true } _ { false } };
    if must_bind {
        let n_args = ty::ty_fn_args(ty).len();
        let args = vec::from_elem(n_args, none);
        let space = alloc_ty(c.bcx, ty);
        let bcx = closure::trans_bind_1(space.bcx, ty, c, args, ty,
                                              save_in(space.val));
        add_clean_temp(bcx, space.val, ty);
        {bcx: bcx, val: space.val, kind: temporary}
    } else {
        alt check c.env {
          is_closure { {bcx: c.bcx, val: c.val, kind: c.kind} }
          null_env {
            let llfnty = llvm::LLVMGetElementType(val_ty(c.val));
            let llfn = create_real_fn_pair(c.bcx, llfnty, c.val,
                                           null_env_ptr(c.bcx));
            {bcx: c.bcx, val: llfn, kind: temporary}
          }
        }
    }
}

fn int_cast(bcx: block, lldsttype: TypeRef, llsrctype: TypeRef,
            llsrc: ValueRef, signed: bool) -> ValueRef {
    let srcsz = llvm::LLVMGetIntTypeWidth(llsrctype);
    let dstsz = llvm::LLVMGetIntTypeWidth(lldsttype);
    ret if dstsz == srcsz {
            BitCast(bcx, llsrc, lldsttype)
        } else if srcsz > dstsz {
            TruncOrBitCast(bcx, llsrc, lldsttype)
        } else if signed {
            SExtOrBitCast(bcx, llsrc, lldsttype)
        } else { ZExtOrBitCast(bcx, llsrc, lldsttype) };
}

fn float_cast(bcx: block, lldsttype: TypeRef, llsrctype: TypeRef,
              llsrc: ValueRef) -> ValueRef {
    let srcsz = lib::llvm::float_width(llsrctype);
    let dstsz = lib::llvm::float_width(lldsttype);
    ret if dstsz > srcsz {
            FPExt(bcx, llsrc, lldsttype)
        } else if srcsz > dstsz {
            FPTrunc(bcx, llsrc, lldsttype)
        } else { llsrc };
}

enum cast_kind { cast_pointer, cast_integral, cast_float,
                 cast_enum, cast_other, }
fn cast_type_kind(t: ty::t) -> cast_kind {
    if ty::type_is_fp(t) { cast_float }
    else if ty::type_is_unsafe_ptr(t) { cast_pointer }
    else if ty::type_is_integral(t) { cast_integral }
    else if ty::type_is_enum(t) { cast_enum }
    else { cast_other }
}


fn trans_cast(cx: block, e: @ast::expr, id: ast::node_id,
              dest: dest) -> block {
    let ccx = cx.ccx();
    let t_out = node_id_type(cx, id);
    alt ty::get(t_out).struct {
      ty::ty_iface(_, _) { ret impl::trans_cast(cx, e, id, dest); }
      _ {}
    }
    let e_res = trans_temp_expr(cx, e);
    let ll_t_in = val_ty(e_res.val);
    let t_in = expr_ty(cx, e);
    let ll_t_out = type_of(ccx, t_out);

    let k_in = cast_type_kind(t_in);
    let k_out = cast_type_kind(t_out);
    let s_in = k_in == cast_integral && ty::type_is_signed(t_in);

    let newval =
        alt {in: k_in, out: k_out} {
          {in: cast_integral, out: cast_integral} {
            int_cast(e_res.bcx, ll_t_out, ll_t_in, e_res.val, s_in)
          }
          {in: cast_float, out: cast_float} {
            float_cast(e_res.bcx, ll_t_out, ll_t_in, e_res.val)
          }
          {in: cast_integral, out: cast_float} {
            if s_in {
                SIToFP(e_res.bcx, e_res.val, ll_t_out)
            } else { UIToFP(e_res.bcx, e_res.val, ll_t_out) }
          }
          {in: cast_float, out: cast_integral} {
            if ty::type_is_signed(t_out) {
                FPToSI(e_res.bcx, e_res.val, ll_t_out)
            } else { FPToUI(e_res.bcx, e_res.val, ll_t_out) }
          }
          {in: cast_integral, out: cast_pointer} {
            IntToPtr(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: cast_pointer, out: cast_integral} {
            PtrToInt(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: cast_pointer, out: cast_pointer} {
            PointerCast(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: cast_enum, out: cast_integral} |
          {in: cast_enum, out: cast_float} {
            let cx = e_res.bcx;
            let llenumty = T_opaque_enum_ptr(ccx);
            let av_enum = PointerCast(cx, e_res.val, llenumty);
            let lldiscrim_a_ptr = GEPi(cx, av_enum, [0, 0]);
            let lldiscrim_a = Load(cx, lldiscrim_a_ptr);
            alt k_out {
              cast_integral {int_cast(e_res.bcx, ll_t_out,
                                      val_ty(lldiscrim_a), lldiscrim_a, true)}
              cast_float {SIToFP(e_res.bcx, lldiscrim_a, ll_t_out)}
              _ { ccx.sess.bug("translating unsupported cast.") }
            }
          }
          _ { ccx.sess.bug("translating unsupported cast.") }
        };
    ret store_in_dest(e_res.bcx, newval, dest);
}

// temp_cleanups: cleanups that should run only if failure occurs before the
// call takes place:
fn trans_arg_expr(cx: block, arg: ty::arg, lldestty: TypeRef, e: @ast::expr,
                  &temp_cleanups: [ValueRef]) -> result {
    let ccx = cx.ccx();
    let e_ty = expr_ty(cx, e);
    let is_bot = ty::type_is_bot(e_ty);
    let lv = trans_temp_lval(cx, e);
    let bcx = lv.bcx;
    let val = lv.val;
    let arg_mode = ty::resolved_mode(ccx.tcx, arg.mode);
    if is_bot {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty (the callee's expected type).
        val = llvm::LLVMGetUndef(lldestty);
    } else if arg_mode == ast::by_ref || arg_mode == ast::by_val {
        let copied = false, imm = ty::type_is_immediate(e_ty);
        if arg_mode == ast::by_ref && lv.kind != owned && imm {
            val = do_spill_noroot(bcx, val);
            copied = true;
        }
        if ccx.maps.copy_map.contains_key(e.id) && lv.kind != temporary {
            if !copied {
                let alloc = alloc_ty(bcx, e_ty);
                bcx = copy_val(alloc.bcx, INIT, alloc.val,
                               load_if_immediate(alloc.bcx, val, e_ty), e_ty);
                val = alloc.val;
            } else { bcx = take_ty(bcx, val, e_ty); }
            add_clean(bcx, val, e_ty);
        }
        if arg_mode == ast::by_val && (lv.kind == owned || !imm) {
            val = Load(bcx, val);
        }
    } else if arg_mode == ast::by_copy || arg_mode == ast::by_move {
        let {bcx: cx, val: alloc} = alloc_ty(bcx, e_ty);
        let move_out = arg_mode == ast::by_move ||
            ccx.maps.last_uses.contains_key(e.id);
        bcx = cx;
        if lv.kind == temporary { revoke_clean(bcx, val); }
        if lv.kind == owned || !ty::type_is_immediate(e_ty) {
            bcx = memmove_ty(bcx, alloc, val, e_ty);
            if move_out && ty::type_needs_drop(ccx.tcx, e_ty) {
                bcx = zero_alloca(bcx, val, e_ty);
            }
        } else { Store(bcx, val, alloc); }
        val = alloc;
        if lv.kind != temporary && !move_out {
            bcx = take_ty(bcx, val, e_ty);
        }

        // In the event that failure occurs before the call actually
        // happens, have to cleanup this copy:
        add_clean_temp_mem(bcx, val, e_ty);
        temp_cleanups += [val];
    } else if ty::type_is_immediate(e_ty) && lv.kind != owned {
        let r = do_spill(bcx, val, e_ty);
        val = r.val;
        bcx = r.bcx;
    }

    if !is_bot && arg.ty != e_ty || ty::type_has_params(arg.ty) {
        val = PointerCast(bcx, val, lldestty);
    }
    ret rslt(bcx, val);
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn trans_args(cx: block, llenv: ValueRef, es: [@ast::expr], fn_ty: ty::t,
              dest: dest, generic_intrinsic: bool)
    -> {bcx: block, args: [ValueRef], retslot: ValueRef} {

    let temp_cleanups = [];
    let args = ty::ty_fn_args(fn_ty);
    let llargs: [ValueRef] = [];

    let ccx = cx.ccx();
    let bcx = cx;

    let retty = ty::ty_fn_ret(fn_ty);
    // Arg 0: Output pointer.
    let llretslot = alt dest {
      ignore {
        if ty::type_is_nil(retty) && !generic_intrinsic {
            llvm::LLVMGetUndef(T_ptr(T_nil()))
        } else {
            let {bcx: cx, val} = alloc_ty(bcx, retty);
            bcx = cx;
            val
        }
      }
      save_in(dst) { dst }
      by_val(_) {
          let {bcx: cx, val} = alloc_ty(bcx, retty);
          bcx = cx;
          val
      }
    };

    llargs += [llretslot];

    // Arg 1: Env (closure-bindings / self value)
    llargs += [llenv];

    // ... then explicit args.

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    let arg_tys = type_of_explicit_args(ccx, args);
    let i = 0u;
    for e: @ast::expr in es {
        let r = trans_arg_expr(bcx, args[i], arg_tys[i], e, temp_cleanups);
        bcx = r.bcx;
        llargs += [r.val];
        i += 1u;
    }

    // now that all arguments have been successfully built, we can revoke any
    // temporary cleanups, as they are only needed if argument construction
    // should fail (for example, cleanup of copy mode args).
    vec::iter(temp_cleanups) {|c|
        revoke_clean(bcx, c)
    }

    ret {bcx: bcx,
         args: llargs,
         retslot: llretslot};
}

fn trans_call(in_cx: block, f: @ast::expr,
              args: [@ast::expr], id: ast::node_id, dest: dest)
    -> block {
    trans_call_inner(in_cx, expr_ty(in_cx, f),
                     {|cx| trans_callee(cx, f)}, args, id, dest)
}

fn trans_call_inner(in_cx: block, fn_expr_ty: ty::t,
                    get_callee: fn(block) -> lval_maybe_callee,
                    args: [@ast::expr], id: ast::node_id, dest: dest)
    -> block {
    with_scope(in_cx, "call") {|cx|
        let f_res = get_callee(cx);
        let bcx = f_res.bcx, ccx = cx.ccx();

        let faddr = f_res.val;
        let llenv = alt f_res.env {
          null_env {
            llvm::LLVMGetUndef(T_opaque_box_ptr(ccx))
          }
          self_env(e, _, _) {
            PointerCast(bcx, e, T_opaque_box_ptr(ccx))
          }
          is_closure {
            // It's a closure. Have to fetch the elements
            if f_res.kind == owned {
                faddr = load_if_immediate(bcx, faddr, fn_expr_ty);
            }
            let pair = faddr;
            faddr = GEPi(bcx, pair, [0, abi::fn_field_code]);
            faddr = Load(bcx, faddr);
            let llclosure = GEPi(bcx, pair, [0, abi::fn_field_box]);
            Load(bcx, llclosure)
          }
        };

        let ret_ty = node_id_type(bcx, id);
        let args_res = trans_args(bcx, llenv, args, fn_expr_ty, dest,
                                  option::is_some(f_res.tds));
        bcx = args_res.bcx;
        let llargs = args_res.args;
        option::may(f_res.tds) {|vals|
            llargs = vec::slice(llargs, 0u, first_real_arg) + vals +
                vec::tailn(llargs, first_real_arg);
        }

        let llretslot = args_res.retslot;

        /* If the block is terminated,
        then one or more of the args has
        type _|_. Since that means it diverges, the code
        for the call itself is unreachable. */
        bcx = invoke_full(bcx, faddr, llargs);
        alt dest {
          ignore {
            if llvm::LLVMIsUndef(llretslot) != lib::llvm::True {
                bcx = drop_ty(bcx, llretslot, ret_ty);
            }
          }
          save_in(_) { } // Already saved by callee
          by_val(cell) {
            *cell = Load(bcx, llretslot);
          }
        }
        if ty::type_is_bot(ret_ty) { Unreachable(bcx); }
        bcx
    }
}

fn invoke(bcx: block, llfn: ValueRef,
          llargs: [ValueRef]) -> block {
    ret invoke_(bcx, llfn, llargs, Invoke);
}

fn invoke_full(bcx: block, llfn: ValueRef, llargs: [ValueRef])
    -> block {
    ret invoke_(bcx, llfn, llargs, Invoke);
}

fn invoke_(bcx: block, llfn: ValueRef, llargs: [ValueRef],
           invoker: fn(block, ValueRef, [ValueRef],
                       BasicBlockRef, BasicBlockRef)) -> block {
    // FIXME: May be worth turning this into a plain call when there are no
    // cleanups to run
    if bcx.unreachable { ret bcx; }
    let normal_bcx = sub_block(bcx, "normal return");
    /*std::io::println("fn: " + lib::llvm::type_to_str(bcx.ccx().tn,
                     val_ty(llfn)));
    for a in llargs {
        std::io::println(" a: " + lib::llvm::type_to_str(bcx.ccx().tn,
                         val_ty(a)));
    }*/
    invoker(bcx, llfn, llargs, normal_bcx.llbb, get_landing_pad(bcx));
    ret normal_bcx;
}

fn get_landing_pad(bcx: block) -> BasicBlockRef {
    fn in_lpad_scope_cx(bcx: block, f: fn(scope_info)) {
        let bcx = bcx;
        loop {
            alt bcx.kind {
              block_scope(info) {
                if info.cleanups.len() > 0u || bcx.parent == parent_none {
                    f(info); ret;
                }
              }
              _ {}
            }
            bcx = block_parent(bcx);
        }
    }

    let cached = none, pad_bcx = bcx; // Guaranteed to be set below
    in_lpad_scope_cx(bcx) {|info|
        // If there is a valid landing pad still around, use it
        alt info.landing_pad {
          some(target) { cached = some(target); ret; }
          none {}
        }
        pad_bcx = sub_block(bcx, "unwind");
        info.landing_pad = some(pad_bcx.llbb);
    }
    alt cached { some(b) { ret b; } none {} } // Can't return from block above
    // The landing pad return type (the type being propagated). Not sure what
    // this represents but it's determined by the personality function and
    // this is what the EH proposal example uses.
    let llretty = T_struct([T_ptr(T_i8()), T_i32()]);
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
    Call(pad_bcx, bcx.ccx().upcalls.reset_stack_limit, []);

    // We store the retval in a function-central alloca, so that calls to
    // Resume can find it.
    alt bcx.fcx.personality {
      some(addr) { Store(pad_bcx, llretval, addr); }
      none {
        let addr = alloca(pad_bcx, val_ty(llretval));
        bcx.fcx.personality = some(addr);
        Store(pad_bcx, llretval, addr);
      }
    }

    // Unwind all parent scopes, and finish with a Resume instr
    cleanup_and_leave(pad_bcx, none, none);
    ret pad_bcx.llbb;
}

fn trans_tup(bcx: block, elts: [@ast::expr], dest: dest) -> block {
    let bcx = bcx;
    let addr = alt dest {
      ignore {
        for ex in elts { bcx = trans_expr(bcx, ex, ignore); }
        ret bcx;
      }
      save_in(pos) { pos }
      _ { bcx.tcx().sess.bug("trans_tup: weird dest"); }
    };
    let temp_cleanups = [], i = 0;
    for e in elts {
        let dst = GEPi(bcx, addr, [0, i]);
        let e_ty = expr_ty(bcx, e);
        bcx = trans_expr_save_in(bcx, e, dst);
        add_clean_temp_mem(bcx, dst, e_ty);
        temp_cleanups += [dst];
        i += 1;
    }
    for cleanup in temp_cleanups { revoke_clean(bcx, cleanup); }
    ret bcx;
}

fn trans_rec(bcx: block, fields: [ast::field],
             base: option<@ast::expr>, id: ast::node_id,
             dest: dest) -> block {
    let t = node_id_type(bcx, id);
    let bcx = bcx;
    let addr = alt dest {
      ignore {
        for fld in fields {
            bcx = trans_expr(bcx, fld.node.expr, ignore);
        }
        ret bcx;
      }
      save_in(pos) { pos }
      _ { bcx.tcx().sess.bug("trans_rec: weird dest"); }
    };

    let ty_fields = alt ty::get(t).struct {
      ty::ty_rec(f) { f }
      _ { bcx.tcx().sess.bug("trans_rec: id doesn't\
           have a record type") } };
    let temp_cleanups = [];
    for fld in fields {
        let ix = option::get(vec::position(ty_fields, {|ft|
            str::eq(fld.node.ident, ft.ident)
        }));
        let dst = GEPi(bcx, addr, [0, ix as int]);
        bcx = trans_expr_save_in(bcx, fld.node.expr, dst);
        add_clean_temp_mem(bcx, dst, ty_fields[ix].mt.ty);
        temp_cleanups += [dst];
    }
    alt base {
      some(bexp) {
        let {bcx: cx, val: base_val} = trans_temp_expr(bcx, bexp), i = 0;
        bcx = cx;
        // Copy over inherited fields
        for tf in ty_fields {
            if !vec::any(fields, {|f| str::eq(f.node.ident, tf.ident)}) {
                let dst = GEPi(bcx, addr, [0, i]);
                let base = GEPi(bcx, base_val, [0, i]);
                let val = load_if_immediate(bcx, base, tf.mt.ty);
                bcx = copy_val(bcx, INIT, dst, val, tf.mt.ty);
            }
            i += 1;
        }
      }
      none {}
    };

    // Now revoke the cleanups as we pass responsibility for the data
    // structure on to the caller
    for cleanup in temp_cleanups { revoke_clean(bcx, cleanup); }
    ret bcx;
}

// Store the result of an expression in the given memory location, ensuring
// that nil or bot expressions get ignore rather than save_in as destination.
fn trans_expr_save_in(bcx: block, e: @ast::expr, dest: ValueRef)
    -> block {
    let t = expr_ty(bcx, e);
    let do_ignore = ty::type_is_bot(t) || ty::type_is_nil(t);
    ret trans_expr(bcx, e, if do_ignore { ignore } else { save_in(dest) });
}

// Call this to compile an expression that you need as an intermediate value,
// and you want to know whether you're dealing with an lval or not (the kind
// field in the returned struct). For non-intermediates, use trans_expr or
// trans_expr_save_in. For intermediates where you don't care about lval-ness,
// use trans_temp_expr.
fn trans_temp_lval(bcx: block, e: @ast::expr) -> lval_result {
    let bcx = bcx;
    if expr_is_lval(bcx, e) {
        ret trans_lval(bcx, e);
    } else {
        let ty = expr_ty(bcx, e);
        if ty::type_is_nil(ty) || ty::type_is_bot(ty) {
            bcx = trans_expr(bcx, e, ignore);
            ret {bcx: bcx, val: C_nil(), kind: temporary};
        } else if ty::type_is_immediate(ty) {
            let cell = empty_dest_cell();
            bcx = trans_expr(bcx, e, by_val(cell));
            add_clean_temp(bcx, *cell, ty);
            ret {bcx: bcx, val: *cell, kind: temporary};
        } else {
            let {bcx, val: scratch} = alloc_ty(bcx, ty);
            bcx = trans_expr_save_in(bcx, e, scratch);
            add_clean_temp(bcx, scratch, ty);
            ret {bcx: bcx, val: scratch, kind: temporary};
        }
    }
}

// Use only for intermediate values. See trans_expr and trans_expr_save_in for
// expressions that must 'end up somewhere' (or get ignored).
fn trans_temp_expr(bcx: block, e: @ast::expr) -> result {
    let {bcx, val, kind} = trans_temp_lval(bcx, e);
    if kind == owned {
        val = load_if_immediate(bcx, val, expr_ty(bcx, e));
    }
    ret {bcx: bcx, val: val};
}

// Translate an expression, with the dest argument deciding what happens with
// the result. Invariants:
// - exprs returning nil or bot always get dest=ignore
// - exprs with non-immediate type never get dest=by_val
fn trans_expr(bcx: block, e: @ast::expr, dest: dest) -> block {
    let tcx = bcx.tcx();
    debuginfo::update_source_pos(bcx, e.span);

    #debug["trans_expr(e=%s,e.id=%d,dest=%s,ty=%s)",
           expr_to_str(e),
           e.id,
           dest_str(bcx.ccx(), dest),
           ty_to_str(tcx, expr_ty(bcx, e))];

    if expr_is_lval(bcx, e) {
        ret lval_to_dps(bcx, e, dest);
    }

    alt e.node {
      ast::expr_if(cond, thn, els) | ast::expr_if_check(cond, thn, els) {
        ret trans_if(bcx, cond, thn, els, dest);
      }
      ast::expr_alt(expr, arms, mode) {
        ret alt::trans_alt(bcx, expr, arms, mode, dest);
      }
      ast::expr_block(blk) {
        ret with_scope(bcx, "block-expr body") {|bcx|
            bcx.block_span = some(blk.span);
            trans_block(bcx, blk, dest)
        };
      }
      ast::expr_rec(args, base) {
        ret trans_rec(bcx, args, base, e.id, dest);
      }
      ast::expr_tup(args) { ret trans_tup(bcx, args, dest); }
      ast::expr_lit(lit) { ret trans_lit(bcx, *lit, dest); }
      ast::expr_vec(args, _) { ret tvec::trans_vec(bcx, args, e.id, dest); }
      ast::expr_binary(op, lhs, rhs) {
        ret trans_binary(bcx, op, lhs, rhs, dest, e);
      }
      ast::expr_unary(op, x) {
        assert op != ast::deref; // lvals are handled above
        ret trans_unary(bcx, op, x, e, dest);
      }
      ast::expr_addr_of(_, x) { ret trans_addr_of(bcx, x, dest); }
      ast::expr_fn(proto, decl, body, cap_clause) {
        ret closure::trans_expr_fn(
            bcx, proto, decl, body, e.span, e.id, *cap_clause, dest);
      }
      ast::expr_fn_block(decl, body) {
        alt ty::get(expr_ty(bcx, e)).struct {
          ty::ty_fn({proto, _}) {
            #debug("translating fn_block %s with type %s",
                   expr_to_str(e), ty_to_str(tcx, expr_ty(bcx, e)));
            let cap_clause = { copies: [], moves: [] };
            ret closure::trans_expr_fn(
                bcx, proto, decl, body, e.span, e.id, cap_clause, dest);
          }
          _ {
            fail "type of fn block is not a function!";
          }
        }
      }
      ast::expr_bind(f, args) {
        ret closure::trans_bind(
            bcx, f, args, e.id, dest);
      }
      ast::expr_copy(a) {
        if !expr_is_lval(bcx, a) {
            ret trans_expr(bcx, a, dest);
        }
        else { ret lval_to_dps(bcx, a, dest); }
      }
      ast::expr_cast(val, _) { ret trans_cast(bcx, val, e.id, dest); }
      ast::expr_call(f, args, _) {
        ret trans_call(bcx, f, args, e.id, dest);
      }
      ast::expr_field(base, _, _) {
        if dest == ignore { ret trans_expr(bcx, base, ignore); }
        let callee = trans_callee(bcx, e), ty = expr_ty(bcx, e);
        let lv = lval_maybe_callee_to_lval(callee, ty);
        revoke_clean(lv.bcx, lv.val);
        ret memmove_ty(lv.bcx, get_dest_addr(dest), lv.val, ty);
      }
      ast::expr_index(base, idx) {
        // If it is here, it's not an lval, so this is a user-defined index op
        let origin = bcx.ccx().maps.method_map.get(e.id);
        let callee_id = ast_util::op_expr_callee_id(e);
        let fty = node_id_type(bcx, callee_id);
        ret trans_call_inner(bcx, fty, {|bcx|
            impl::trans_method_callee(bcx, callee_id, base, origin)
        }, [idx], e.id, dest);
      }

      // These return nothing
      ast::expr_break {
        assert dest == ignore;
        ret trans_break(bcx);
      }
      ast::expr_cont {
        assert dest == ignore;
        ret trans_cont(bcx);
      }
      ast::expr_ret(ex) {
        assert dest == ignore;
        ret trans_ret(bcx, ex);
      }
      ast::expr_be(ex) {
        ret trans_be(bcx, ex);
      }
      ast::expr_fail(expr) {
        assert dest == ignore;
        ret trans_fail_expr(bcx, some(e.span), expr);
      }
      ast::expr_log(_, lvl, a) {
        assert dest == ignore;
        ret trans_log(lvl, bcx, a);
      }
      ast::expr_assert(a) {
        assert dest == ignore;
        ret trans_check_expr(bcx, a, "Assertion");
      }
      ast::expr_check(ast::checked_expr, a) {
        assert dest == ignore;
        ret trans_check_expr(bcx, a, "Predicate");
      }
      ast::expr_check(ast::claimed_expr, a) {
        assert dest == ignore;
        /* Claims are turned on and off by a global variable
           that the RTS sets. This case generates code to
           check the value of that variable, doing nothing
           if it's set to false and acting like a check
           otherwise. */
        let c = get_extern_const(bcx.ccx().externs, bcx.ccx().llmod,
                                 "check_claims", T_bool());
        ret with_cond(bcx, Load(bcx, c)) {|bcx|
            trans_check_expr(bcx, a, "Claim")
        };
      }
      ast::expr_for(decl, seq, body) {
        assert dest == ignore;
        ret trans_for(bcx, decl, seq, body);
      }
      ast::expr_while(cond, body) {
        assert dest == ignore;
        ret trans_while(bcx, cond, body);
      }
      ast::expr_loop(body) {
        assert dest == ignore;
        ret trans_loop(bcx, body);
      }
      ast::expr_do_while(body, cond) {
        assert dest == ignore;
        ret trans_do_while(bcx, body, cond);
      }
      ast::expr_assign(dst, src) {
        assert dest == ignore;
        let src_r = trans_temp_lval(bcx, src);
        let {bcx, val: addr, kind} = trans_lval(src_r.bcx, dst);
        assert kind == owned;
        ret store_temp_expr(bcx, DROP_EXISTING, addr, src_r,
                            expr_ty(bcx, src),
                            bcx.ccx().maps.last_uses.contains_key(src.id));
      }
      ast::expr_move(dst, src) {
        // FIXME: calculate copy init-ness in typestate.
        assert dest == ignore;
        let src_r = trans_temp_lval(bcx, src);
        let {bcx, val: addr, kind} = trans_lval(src_r.bcx, dst);
        assert kind == owned;
        ret move_val(bcx, DROP_EXISTING, addr, src_r,
                     expr_ty(bcx, src));
      }
      ast::expr_swap(dst, src) {
        assert dest == ignore;
        let lhs_res = trans_lval(bcx, dst);
        assert lhs_res.kind == owned;
        let rhs_res = trans_lval(lhs_res.bcx, src);
        let t = expr_ty(bcx, src);
        let {bcx: bcx, val: tmp_alloc} = alloc_ty(rhs_res.bcx, t);
        // Swap through a temporary.
        bcx = move_val(bcx, INIT, tmp_alloc, lhs_res, t);
        bcx = move_val(bcx, INIT, lhs_res.val, rhs_res, t);
        ret move_val(bcx, INIT, rhs_res.val, lval_owned(bcx, tmp_alloc), t);
      }
      ast::expr_assign_op(op, dst, src) {
        assert dest == ignore;
        ret trans_assign_op(bcx, e, op, dst, src);
      }
      _ { bcx.tcx().sess.span_bug(e.span, "trans_expr reached \
             fall-through case"); }

    }
}

fn lval_to_dps(bcx: block, e: @ast::expr, dest: dest) -> block {
    let lv = trans_lval(bcx, e), ccx = bcx.ccx();
    let {bcx, val, kind} = lv;
    let last_use = kind == owned && ccx.maps.last_uses.contains_key(e.id);
    let ty = expr_ty(bcx, e);
    alt dest {
      by_val(cell) {
        if kind == temporary {
            revoke_clean(bcx, val);
            *cell = val;
        } else if last_use {
            *cell = Load(bcx, val);
            if ty::type_needs_drop(ccx.tcx, ty) {
                bcx = zero_alloca(bcx, val, ty);
            }
        } else {
            if kind == owned { val = Load(bcx, val); }
            let {bcx: cx, val} = take_ty_immediate(bcx, val, ty);
            *cell = val;
            bcx = cx;
        }
      }
      save_in(loc) {
        bcx = store_temp_expr(bcx, INIT, loc, lv, ty, last_use);
      }
      ignore {}
    }
    ret bcx;
}

fn do_spill(cx: block, v: ValueRef, t: ty::t) -> result {
    // We have a value but we have to spill it, and root it, to pass by alias.
    let bcx = cx;

    if ty::type_is_bot(t) {
        ret rslt(bcx, C_null(T_ptr(T_i8())));
    }

    let r = alloc_ty(bcx, t);
    bcx = r.bcx;
    let llptr = r.val;

    Store(bcx, v, llptr);

    ret rslt(bcx, llptr);
}

// Since this function does *not* root, it is the caller's responsibility to
// ensure that the referent is pointed to by a root.
fn do_spill_noroot(cx: block, v: ValueRef) -> ValueRef {
    let llptr = alloca(cx, val_ty(v));
    Store(cx, v, llptr);
    ret llptr;
}

fn spill_if_immediate(cx: block, v: ValueRef, t: ty::t) -> result {
    if ty::type_is_immediate(t) { ret do_spill(cx, v, t); }
    ret rslt(cx, v);
}

fn load_if_immediate(cx: block, v: ValueRef, t: ty::t) -> ValueRef {
    if ty::type_is_immediate(t) { ret Load(cx, v); }
    ret v;
}

fn trans_log(lvl: @ast::expr, bcx: block, e: @ast::expr) -> block {
    let ccx = bcx.ccx();
    if ty::type_is_bot(expr_ty(bcx, lvl)) {
       ret trans_expr(bcx, lvl, ignore);
    }

    let modpath = [path_mod(ccx.link_meta.name)] +
        vec::filter(bcx.fcx.path, {|e|
            alt e { path_mod(_) { true } _ { false } }
        });
    let modname = path_str(modpath);

    let global = if ccx.module_data.contains_key(modname) {
        ccx.module_data.get(modname)
    } else {
        let s = link::mangle_internal_name_by_path_and_seq(
            ccx, modpath, "loglevel");
        let global = str::as_c_str(s, {|buf|
            llvm::LLVMAddGlobal(ccx.llmod, T_i32(), buf)
        });
        llvm::LLVMSetGlobalConstant(global, False);
        llvm::LLVMSetInitializer(global, C_null(T_i32()));
        lib::llvm::SetLinkage(global, lib::llvm::InternalLinkage);
        ccx.module_data.insert(modname, global);
        global
    };
    let current_level = Load(bcx, global);
    let {bcx, val: level} = with_scope_result(bcx, "level") {|bcx|
        trans_temp_expr(bcx, lvl)
    };

    with_cond(bcx, ICmp(bcx, lib::llvm::IntUGE, current_level, level)) {|bcx|
        with_scope(bcx, "log") {|bcx|
            let {bcx, val, _} = trans_temp_expr(bcx, e);
            let e_ty = expr_ty(bcx, e);
            let {bcx, val: tydesc} = get_tydesc_simple(bcx, e_ty);
            // Call the polymorphic log function.
            let {bcx, val} = spill_if_immediate(bcx, val, e_ty);
            let val = PointerCast(bcx, val, T_ptr(T_i8()));
            Call(bcx, ccx.upcalls.log_type, [tydesc, val, level]);
            bcx
        }
    }
}

fn trans_check_expr(bcx: block, e: @ast::expr, s: str) -> block {
    let expr_str = s + " " + expr_to_str(e) + " failed";
    let {bcx, val} = with_scope_result(bcx, "check") {|bcx|
        trans_temp_expr(bcx, e)
    };
    with_cond(bcx, Not(bcx, val)) {|bcx|
        trans_fail(bcx, some(e.span), expr_str)
    }
}

fn trans_fail_expr(bcx: block, sp_opt: option<span>,
                   fail_expr: option<@ast::expr>) -> block {
    let bcx = bcx;
    alt fail_expr {
      some(expr) {
        let ccx = bcx.ccx(), tcx = ccx.tcx;
        let expr_res = trans_temp_expr(bcx, expr);
        let e_ty = expr_ty(bcx, expr);
        bcx = expr_res.bcx;

        if ty::type_is_str(e_ty) {
            let data = tvec::get_dataptr(
                bcx, expr_res.val, type_of(
                    ccx, ty::mk_mach_uint(tcx, ast::ty_u8)));
            ret trans_fail_value(bcx, sp_opt, data);
        } else if bcx.unreachable || ty::type_is_bot(e_ty) {
            ret bcx;
        } else {
            bcx.sess().span_bug(
                expr.span, "fail called with unsupported type " +
                ty_to_str(tcx, e_ty));
        }
      }
      _ { ret trans_fail(bcx, sp_opt, "explicit failure"); }
    }
}

fn trans_fail(bcx: block, sp_opt: option<span>, fail_str: str) ->
    block {
    let V_fail_str = C_cstr(bcx.ccx(), fail_str);
    ret trans_fail_value(bcx, sp_opt, V_fail_str);
}

fn trans_fail_value(bcx: block, sp_opt: option<span>,
                    V_fail_str: ValueRef) -> block {
    let ccx = bcx.ccx();
    let V_filename;
    let V_line;
    alt sp_opt {
      some(sp) {
        let sess = bcx.sess();
        let loc = codemap::lookup_char_pos(sess.parse_sess.cm, sp.lo);
        V_filename = C_cstr(bcx.ccx(), loc.file.name);
        V_line = loc.line as int;
      }
      none { V_filename = C_cstr(bcx.ccx(), "<runtime>"); V_line = 0; }
    }
    let V_str = PointerCast(bcx, V_fail_str, T_ptr(T_i8()));
    V_filename = PointerCast(bcx, V_filename, T_ptr(T_i8()));
    let args = [V_str, V_filename, C_int(ccx, V_line)];
    let bcx = invoke(bcx, bcx.ccx().upcalls._fail, args);
    Unreachable(bcx);
    ret bcx;
}

fn trans_break_cont(bcx: block, to_end: bool)
    -> block {
    // Locate closest loop block, outputting cleanup as we go.
    let unwind = bcx, target = bcx;
    loop {
        alt unwind.kind {
          block_scope({is_loop: some({cnt, brk}), _}) {
            target = if to_end {
                brk
            } else {
                alt cnt {
                  cont_other(o) { o }
                  cont_self { unwind }
                }
            };
            break;
          }
          _ {}
        }
        unwind = alt check unwind.parent {
          parent_some(cx) { cx }
          parent_none {
            bcx.sess().bug
                (if to_end { "break" } else { "cont" } + " outside a loop");
          }
        };
    }
    cleanup_and_Br(bcx, unwind, target.llbb);
    Unreachable(bcx);
    ret bcx;
}

fn trans_break(cx: block) -> block {
    ret trans_break_cont(cx, true);
}

fn trans_cont(cx: block) -> block {
    ret trans_break_cont(cx, false);
}

fn trans_ret(bcx: block, e: option<@ast::expr>) -> block {
    let bcx = bcx;
    alt e {
      some(x) { bcx = trans_expr_save_in(bcx, x, bcx.fcx.llretptr); }
      _ {}
    }
    cleanup_and_leave(bcx, none, some(bcx.fcx.llreturn));
    Unreachable(bcx);
    ret bcx;
}

fn build_return(bcx: block) { Br(bcx, bcx.fcx.llreturn); }

fn trans_be(cx: block, e: @ast::expr) -> block {
    // FIXME: Turn this into a real tail call once
    // calling convention issues are settled
    ret trans_ret(cx, some(e));
}

fn init_local(bcx: block, local: @ast::local) -> block {
    let ty = node_id_type(bcx, local.node.id);
    let llptr = alt bcx.fcx.lllocals.find(local.node.id) {
      some(local_mem(v)) { v }
      some(_) { bcx.tcx().sess.span_bug(local.span,
                        "init_local: Someone forgot to document why it's\
                         safe to assume local.node.init must be local_mem!");
      }
      // This is a local that is kept immediate
      none {
        let initexpr = alt local.node.init {
                some({expr, _}) { expr }
                none { bcx.tcx().sess.span_bug(local.span,
                        "init_local: Someone forgot to document why it's\
                         safe to assume local.node.init isn't none!"); }
            };
        let {bcx, val, kind} = trans_temp_lval(bcx, initexpr);
        if kind != temporary {
            if kind == owned { val = Load(bcx, val); }
            let rs = take_ty_immediate(bcx, val, ty);
            bcx = rs.bcx; val = rs.val;
            add_clean_temp(bcx, val, ty);
        }
        bcx.fcx.lllocals.insert(local.node.pat.id, local_imm(val));
        ret bcx;
      }
    };

    let bcx = bcx;
    alt local.node.init {
      some(init) {
        if init.op == ast::init_assign || !expr_is_lval(bcx, init.expr) {
            bcx = trans_expr_save_in(bcx, init.expr, llptr);
        } else { // This is a move from an lval, must perform an actual move
            let sub = trans_lval(bcx, init.expr);
            bcx = move_val(sub.bcx, INIT, llptr, sub, ty);
        }
      }
      _ { bcx = zero_alloca(bcx, llptr, ty); }
    }
    // Make a note to drop this slot on the way out.
    add_clean(bcx, llptr, ty);
    ret alt::bind_irrefutable_pat(bcx, local.node.pat, llptr, false);
}

fn zero_alloca(cx: block, llptr: ValueRef, t: ty::t)
    -> block {
    let bcx = cx;
    let ccx = cx.ccx();
    let llty = type_of(ccx, t);
    Store(bcx, C_null(llty), llptr);
    ret bcx;
}

fn trans_stmt(cx: block, s: ast::stmt) -> block {
    #debug["trans_stmt(%s)", stmt_to_str(s)];

    if (!cx.sess().opts.no_asm_comments) {
        add_span_comment(cx, s.span, stmt_to_str(s));
    }

    let bcx = cx;
    debuginfo::update_source_pos(cx, s.span);

    alt s.node {
      ast::stmt_expr(e, _) | ast::stmt_semi(e, _) {
        bcx = trans_expr(cx, e, ignore);
      }
      ast::stmt_decl(d, _) {
        alt d.node {
          ast::decl_local(locals) {
            for local in locals {
                bcx = init_local(bcx, local);
                if cx.sess().opts.extra_debuginfo {
                    debuginfo::create_local_var(bcx, local);
                }
            }
          }
          ast::decl_item(i) { trans_item(cx.fcx.ccx, *i); }
        }
      }
      _ { cx.sess().unimpl("stmt variant"); }
    }

    ret bcx;
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block(cx: fn_ctxt, parent: block_parent, kind: block_kind,
             name: str, block_span: option<span>) -> block {
    let s = "";
    if cx.ccx.sess.opts.save_temps || cx.ccx.sess.opts.debuginfo {
        s = cx.ccx.names(name);
    }
    let llbb: BasicBlockRef = str::as_c_str(s, {|buf|
        llvm::LLVMAppendBasicBlock(cx.llfn, buf)
    });
    let bcx = @{llbb: llbb,
                mutable terminated: false,
                mutable unreachable: false,
                parent: parent,
                kind: kind,
                mutable block_span: block_span,
                fcx: cx};
    alt parent {
      parent_some(cx) {
        if cx.unreachable { Unreachable(bcx); }
      }
      _ {}
    }
    ret bcx;
}

fn simple_block_scope() -> block_kind {
    block_scope({is_loop: none, mutable cleanups: [],
                 mutable cleanup_paths: [], mutable landing_pad: none})
}

// Use this when you're at the top block of a function or the like.
fn top_scope_block(fcx: fn_ctxt, sp: option<span>) -> block {
    ret new_block(fcx, parent_none, simple_block_scope(),
                  "function top level", sp);
}

fn scope_block(bcx: block, n: str) -> block {
    ret new_block(bcx.fcx, parent_some(bcx), simple_block_scope(),
                  n, none);
}

fn loop_scope_block(bcx: block, _cont: loop_cont,
                    _break: block, n: str, sp: span)
    -> block {
    ret new_block(bcx.fcx, parent_some(bcx), block_scope({
        is_loop: some({cnt: _cont, brk: _break}),
        mutable cleanups: [],
        mutable cleanup_paths: [],
        mutable landing_pad: none
    }), n, some(sp));
}


// Use this when you're making a general CFG BB within a scope.
fn sub_block(bcx: block, n: str) -> block {
    ret new_block(bcx.fcx, parent_some(bcx), block_non_scope, n, none);
}

fn raw_block(fcx: fn_ctxt, llbb: BasicBlockRef) -> block {
    ret @{llbb: llbb,
          mutable terminated: false,
          mutable unreachable: false,
          parent: parent_none,
          kind: block_non_scope,
          mutable block_span: none,
          fcx: fcx};
}


// trans_block_cleanups: Go through all the cleanups attached to this
// block and execute them.
//
// When translating a block that introdces new variables during its scope, we
// need to make sure those variables go out of scope when the block ends.  We
// do that by running a 'cleanup' function for each variable.
// trans_block_cleanups runs all the cleanup functions for the block.
fn trans_block_cleanups(bcx: block, cleanup_cx: block) ->
   block {
    if bcx.unreachable { ret bcx; }
    let bcx = bcx;
    alt check cleanup_cx.kind {
      block_scope({cleanups, _}) {
        vec::riter(cleanups) {|cu|
            alt cu { clean(cfn) | clean_temp(_, cfn) { bcx = cfn(bcx); } }
        }
      }
    }
    ret bcx;
}

// In the last argument, some(block) mean jump to this block, and none means
// this is a landing pad and leaving should be accomplished with a resume
// instruction.
fn cleanup_and_leave(bcx: block, upto: option<BasicBlockRef>,
                     leave: option<BasicBlockRef>) {
    let cur = bcx, bcx = bcx;
    loop {
        alt cur.kind {
          block_scope(info) if info.cleanups.len() > 0u {
            for exists in info.cleanup_paths {
                if exists.target == leave {
                    Br(bcx, exists.dest);
                    ret;
                }
            }
            let sub_cx = sub_block(bcx, "cleanup");
            Br(bcx, sub_cx.llbb);
            info.cleanup_paths += [{target: leave, dest: sub_cx.llbb}];
            bcx = trans_block_cleanups(sub_cx, cur);
          }
          _ {}
        }
        alt upto {
          some(bb) { if cur.llbb == bb { break; } }
          _ {}
        }
        cur = alt cur.parent {
          parent_some(next) { next }
          parent_none { assert option::is_none(upto); break; }
        };
    }
    alt leave {
      some(target) { Br(bcx, target); }
      none { Resume(bcx, Load(bcx, option::get(bcx.fcx.personality))); }
    }
}

fn cleanup_and_Br(bcx: block, upto: block,
                  target: BasicBlockRef) {
    cleanup_and_leave(bcx, some(upto.llbb), some(target));
}

fn leave_block(bcx: block, out_of: block) -> block {
    let next_cx = sub_block(block_parent(out_of), "next");
    if bcx.unreachable { Unreachable(next_cx); }
    cleanup_and_Br(bcx, out_of, next_cx.llbb);
    next_cx
}

fn with_scope(bcx: block, name: str, f: fn(block) -> block) -> block {
    let scope_cx = scope_block(bcx, name);
    Br(bcx, scope_cx.llbb);
    leave_block(f(scope_cx), scope_cx)
}

fn with_scope_result(bcx: block, name: str, f: fn(block) -> result)
    -> result {
    let scope_cx = scope_block(bcx, name);
    Br(bcx, scope_cx.llbb);
    let {bcx, val} = f(scope_cx);
    {bcx: leave_block(bcx, scope_cx), val: val}
}

fn with_cond(bcx: block, val: ValueRef, f: fn(block) -> block) -> block {
    let next_cx = sub_block(bcx, "next"), cond_cx = sub_block(bcx, "cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb);
    let after_cx = f(cond_cx);
    if !after_cx.terminated { Br(after_cx, next_cx.llbb); }
    next_cx
}

fn block_locals(b: ast::blk, it: fn(@ast::local)) {
    for s: @ast::stmt in b.node.stmts {
        alt s.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locals) {
                for local in locals { it(local); }
              }
              _ {/* fall through */ }
            }
          }
          _ {/* fall through */ }
        }
    }
}

fn alloc_ty(cx: block, t: ty::t) -> result {
    let bcx = cx, ccx = cx.ccx();
    let llty = type_of(ccx, t);
    assert !ty::type_has_params(t);
    let val = alloca(bcx, llty);

    // NB: since we've pushed all size calculations in this
    // function up to the alloca block, we actually return the
    // block passed into us unmodified; it doesn't really
    // have to be passed-and-returned here, but it fits
    // past caller conventions and may well make sense again,
    // so we leave it as-is.

    ret rslt(cx, val);
}

fn alloc_local(cx: block, local: @ast::local) -> block {
    let t = node_id_type(cx, local.node.id);
    let simple_name = alt local.node.pat.node {
      ast::pat_ident(pth, none) { some(path_to_ident(pth)) }
      _ { none }
    };
    // Do not allocate space for locals that can be kept immediate.
    let ccx = cx.ccx();
    if option::is_some(simple_name) &&
       !ccx.maps.mutbl_map.contains_key(local.node.pat.id) &&
       !ccx.maps.last_uses.contains_key(local.node.pat.id) &&
       ty::type_is_immediate(t) {
        alt local.node.init {
          some({op: ast::init_assign, _}) { ret cx; }
          _ {}
        }
    }
    let {bcx, val} = alloc_ty(cx, t);
    if cx.sess().opts.debuginfo {
        option::may(simple_name) {|name|
            str::as_c_str(name, {|buf|
                llvm::LLVMSetValueName(val, buf)
            });
        }
    }
    cx.fcx.lllocals.insert(local.node.id, local_mem(val));
    ret bcx;
}

fn trans_block(bcx: block, b: ast::blk, dest: dest)
    -> block {
    let bcx = bcx;
    block_locals(b) {|local| bcx = alloc_local(bcx, local); };
    for s: @ast::stmt in b.node.stmts {
        debuginfo::update_source_pos(bcx, b.span);
        bcx = trans_stmt(bcx, *s);
    }
    alt b.node.expr {
      some(e) {
        let bt = ty::type_is_bot(expr_ty(bcx, e));
        debuginfo::update_source_pos(bcx, e.span);
        bcx = trans_expr(bcx, e, if bt { ignore } else { dest });
      }
      _ { assert dest == ignore || bcx.unreachable; }
    }
    ret bcx;
}

// Creates the standard set of basic blocks for a function
fn mk_standard_basic_blocks(llfn: ValueRef) ->
   {sa: BasicBlockRef, ca: BasicBlockRef, rt: BasicBlockRef} {
    {sa: str::as_c_str("static_allocas", {|buf|
        llvm::LLVMAppendBasicBlock(llfn, buf) }),
     ca: str::as_c_str("load_env", {|buf|
         llvm::LLVMAppendBasicBlock(llfn, buf) }),
     rt: str::as_c_str("return", {|buf|
         llvm::LLVMAppendBasicBlock(llfn, buf) })}
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn new_fn_ctxt_w_id(ccx: @crate_ctxt, path: path,
                    llfndecl: ValueRef, id: ast::node_id,
                    maybe_self_id: option<@ast::expr>,
                    param_substs: option<param_substs>,
                    sp: option<span>) -> fn_ctxt {
    let llbbs = mk_standard_basic_blocks(llfndecl);
    ret @{llfn: llfndecl,
          llenv: llvm::LLVMGetParam(llfndecl, 1u as c_uint),
          llretptr: llvm::LLVMGetParam(llfndecl, 0u as c_uint),
          mutable llstaticallocas: llbbs.sa,
          mutable llloadenv: llbbs.ca,
          mutable llreturn: llbbs.rt,
          mutable llself: none,
          mutable personality: none,
          llargs: int_hash::<local_val>(),
          lllocals: int_hash::<local_val>(),
          llupvars: int_hash::<ValueRef>(),
          id: id,
          self_id: maybe_self_id,
          param_substs: param_substs,
          span: sp,
          path: path,
          ccx: ccx};
}

fn new_fn_ctxt(ccx: @crate_ctxt, path: path, llfndecl: ValueRef,
               sp: option<span>) -> fn_ctxt {
    ret new_fn_ctxt_w_id(ccx, path, llfndecl, -1, none, none, sp);
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
                             args: [ast::arg]) {
    // Skip the implicit arguments 0, and 1.
    let arg_n = first_real_arg;
    alt ty_self {
      impl_self(tt) {
        cx.llself = some({v: cx.llenv, t: tt});
      }
      no_self {}
    }

    // Populate the llargs field of the function context with the ValueRefs
    // that we get from llvm::LLVMGetParam for each argument.
    for arg: ast::arg in args {
        let llarg = llvm::LLVMGetParam(cx.llfn, arg_n as c_uint);
        assert (llarg as int != 0);
        // Note that this uses local_mem even for things passed by value.
        // copy_args_to_allocas will overwrite the table entry with local_imm
        // before it's actually used.
        cx.llargs.insert(arg.id, local_mem(llarg));
        arg_n += 1u;
    }
}

fn copy_args_to_allocas(fcx: fn_ctxt, bcx: block, args: [ast::arg],
                        arg_tys: [ty::arg]) -> block {
    let tcx = bcx.tcx();
    let arg_n: uint = 0u, bcx = bcx;
    let epic_fail = fn@() -> ! {
        tcx.sess.bug("someone forgot\
                to document an invariant in copy_args_to_allocas!");
    };
    for arg in arg_tys {
        let id = args[arg_n].id;
        let argval = alt fcx.llargs.get(id) { local_mem(v) { v }
                                              _ { epic_fail() } };
        alt ty::resolved_mode(tcx, arg.mode) {
          ast::by_mutbl_ref { }
          ast::by_move | ast::by_copy { add_clean(bcx, argval, arg.ty); }
          ast::by_val {
            if !ty::type_is_immediate(arg.ty) {
                let {bcx: cx, val: alloc} = alloc_ty(bcx, arg.ty);
                bcx = cx;
                Store(bcx, argval, alloc);
                fcx.llargs.insert(id, local_mem(alloc));
            } else {
                fcx.llargs.insert(id, local_imm(argval));
            }
          }
          ast::by_ref {}
        }
        if fcx.ccx.sess.opts.extra_debuginfo {
            debuginfo::create_arg(bcx, args[arg_n], args[arg_n].ty.span);
        }
        arg_n += 1u;
    }
    ret bcx;
}

// Ties up the llstaticallocas -> llloadenv -> lltop edges,
// and builds the return block.
fn finish_fn(fcx: fn_ctxt, lltop: BasicBlockRef) {
    tie_up_header_blocks(fcx, lltop);
    let ret_cx = raw_block(fcx, fcx.llreturn);
    RetVoid(ret_cx);
}

fn tie_up_header_blocks(fcx: fn_ctxt, lltop: BasicBlockRef) {
    Br(raw_block(fcx, fcx.llstaticallocas), fcx.llloadenv);
    Br(raw_block(fcx, fcx.llloadenv), lltop);
}

enum self_arg { impl_self(ty::t), no_self, }

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
fn trans_closure(ccx: @crate_ctxt, path: path, decl: ast::fn_decl,
                 body: ast::blk, llfndecl: ValueRef,
                 ty_self: self_arg,
                 param_substs: option<param_substs>,
                 id: ast::node_id, maybe_self_id: option<@ast::expr>,
                 maybe_load_env: fn(fn_ctxt)) {
    set_uwtable(llfndecl);

    // Set up arguments to the function.
            let fcx = new_fn_ctxt_w_id(ccx, path, llfndecl, id, maybe_self_id,
                  param_substs, some(body.span));
    create_llargs_for_fn_args(fcx, ty_self, decl.inputs);

    // Create the first basic block in the function and keep a handle on it to
    //  pass to finish_fn later.
    let bcx_top = top_scope_block(fcx, some(body.span)), bcx = bcx_top;
    let lltop = bcx.llbb;
    let block_ty = node_id_type(bcx, body.node.id);

    let arg_tys = ty::ty_fn_args(node_id_type(bcx, id));
    bcx = copy_args_to_allocas(fcx, bcx, decl.inputs, arg_tys);

    maybe_load_env(fcx);

    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).

    if option::is_none(maybe_self_id) // hack --
       /* avoids the need for special cases to assign a type to
          the constructor body (since it has no explicit return) */
      &&
      (option::is_none(body.node.expr) ||
       ty::type_is_bot(block_ty) ||
       ty::type_is_nil(block_ty))  {
        bcx = trans_block(bcx, body, ignore);
    } else {
        bcx = trans_block(bcx, body, save_in(fcx.llretptr));
    }
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
            param_substs: option<param_substs>,
            id: ast::node_id,
            maybe_self_id: option<@ast::expr>) {
    let do_time = ccx.sess.opts.stats;
    let start = if do_time { time::get_time() }
                else { {sec: 0u32, usec: 0u32} };
    trans_closure(ccx, path, decl, body, llfndecl, ty_self,
                  param_substs, id, maybe_self_id, {|fcx|
        if ccx.sess.opts.extra_debuginfo {
            debuginfo::create_function(fcx);
        }
    });
    if do_time {
        let end = time::get_time();
        log_fn_time(ccx, path_str(path), start, end);
    }
}

fn trans_res_ctor(ccx: @crate_ctxt, path: path, dtor: ast::fn_decl,
                  ctor_id: ast::node_id,
                  param_substs: option<param_substs>, llfndecl: ValueRef) {
    // Create a function for the constructor
    let fcx = new_fn_ctxt_w_id(ccx, path, llfndecl, ctor_id,
                               none, param_substs, none);
    create_llargs_for_fn_args(fcx, no_self, dtor.inputs);
    let bcx = top_scope_block(fcx, none), lltop = bcx.llbb;
    let fty = node_id_type(bcx, ctor_id);
    let arg_t = ty::ty_fn_args(fty)[0].ty;
    let arg = alt fcx.llargs.find(dtor.inputs[0].id) {
      some(local_mem(x)) { x }
      _ { ccx.sess.bug("Someone forgot to document an invariant \
            in trans_res_ctor"); }
    };
    let llretptr = fcx.llretptr;

    let dst = GEPi(bcx, llretptr, [0, 1]);
    bcx = memmove_ty(bcx, dst, arg, arg_t);
    let flag = GEPi(bcx, llretptr, [0, 0]);
    let one = C_u8(1u);
    Store(bcx, one, flag);
    build_return(bcx);
    finish_fn(fcx, lltop);
}


fn trans_enum_variant(ccx: @crate_ctxt, enum_id: ast::node_id,
                      variant: ast::variant, disr: int, is_degen: bool,
                      param_substs: option<param_substs>,
                      llfndecl: ValueRef) {
    // Translate variant arguments to function arguments.
    let fn_args = [], i = 0u;
    for varg in variant.node.args {
        fn_args += [{mode: ast::expl(ast::by_copy),
                     ty: varg.ty,
                     ident: "arg" + uint::to_str(i, 10u),
                     id: varg.id}];
    }
    let fcx = new_fn_ctxt_w_id(ccx, [], llfndecl, variant.node.id, none,
                               param_substs, none);
    create_llargs_for_fn_args(fcx, no_self, fn_args);
    let ty_param_substs = alt param_substs {
      some(substs) { substs.tys }
      none { [] }
    };
    let bcx = top_scope_block(fcx, none), lltop = bcx.llbb;
    let arg_tys = ty::ty_fn_args(node_id_type(bcx, variant.node.id));
    bcx = copy_args_to_allocas(fcx, bcx, fn_args, arg_tys);

    // Cast the enum to a type we can GEP into.
    let llblobptr = if is_degen {
        fcx.llretptr
    } else {
        let llenumptr =
            PointerCast(bcx, fcx.llretptr, T_opaque_enum_ptr(ccx));
        let lldiscrimptr = GEPi(bcx, llenumptr, [0, 0]);
        Store(bcx, C_int(ccx, disr), lldiscrimptr);
        GEPi(bcx, llenumptr, [0, 1])
    };
    let i = 0u;
    let t_id = local_def(enum_id);
    let v_id = local_def(variant.node.id);
    for va: ast::variant_arg in variant.node.args {
        let rslt = GEP_enum(bcx, llblobptr, t_id, v_id, ty_param_substs, i);
        bcx = rslt.bcx;
        let lldestptr = rslt.val;
        // If this argument to this function is a enum, it'll have come in to
        // this function as an opaque blob due to the way that type_of()
        // works. So we have to cast to the destination's view of the type.
        let llarg = alt check fcx.llargs.find(va.id) {
          some(local_mem(x)) { x }
        };
        let arg_ty = arg_tys[i].ty;
        bcx = memmove_ty(bcx, lldestptr, llarg, arg_ty);
        i += 1u;
    }
    build_return(bcx);
    finish_fn(fcx, lltop);
}


// FIXME: this should do some structural hash-consing to avoid
// duplicate constants. I think. Maybe LLVM has a magical mode
// that does so later on?
fn trans_const_expr(cx: @crate_ctxt, e: @ast::expr) -> ValueRef {
    alt e.node {
      ast::expr_lit(lit) { ret trans_crate_lit(cx, *lit); }
      ast::expr_binary(b, e1, e2) {
        let te1 = trans_const_expr(cx, e1);
        let te2 = trans_const_expr(cx, e2);

        let te2 = cast_shift_const_rhs(b, te1, te2);

        /* Neither type is bottom, and we expect them to be unified already,
         * so the following is safe. */
        let ty = ty::expr_ty(cx.tcx, e1);
        let is_float = ty::type_is_fp(ty);
        let signed = ty::type_is_signed(ty);
        ret alt b {
          ast::add    {
            if is_float { llvm::LLVMConstFAdd(te1, te2) }
            else        { llvm::LLVMConstAdd(te1, te2) }
          }
          ast::subtract {
            if is_float { llvm::LLVMConstFSub(te1, te2) }
            else        { llvm::LLVMConstSub(te1, te2) }
          }
          ast::mul    {
            if is_float { llvm::LLVMConstFMul(te1, te2) }
            else        { llvm::LLVMConstMul(te1, te2) }
          }
          ast::div    {
            if is_float    { llvm::LLVMConstFDiv(te1, te2) }
            else if signed { llvm::LLVMConstSDiv(te1, te2) }
            else           { llvm::LLVMConstUDiv(te1, te2) }
          }
          ast::rem    {
            if is_float    { llvm::LLVMConstFRem(te1, te2) }
            else if signed { llvm::LLVMConstSRem(te1, te2) }
            else           { llvm::LLVMConstURem(te1, te2) }
          }
          ast::and    |
          ast::or     { cx.sess.span_unimpl(e.span, "binop logic"); }
          ast::bitxor { llvm::LLVMConstXor(te1, te2) }
          ast::bitand { llvm::LLVMConstAnd(te1, te2) }
          ast::bitor  { llvm::LLVMConstOr(te1, te2) }
          ast::lsl    { llvm::LLVMConstShl(te1, te2) }
          ast::lsr    { llvm::LLVMConstLShr(te1, te2) }
          ast::asr    { llvm::LLVMConstAShr(te1, te2) }
          ast::eq     |
          ast::lt     |
          ast::le     |
          ast::ne     |
          ast::ge     |
          ast::gt     { cx.sess.span_unimpl(e.span, "binop comparator"); }
        }
      }
      ast::expr_unary(u, e) {
        let te = trans_const_expr(cx, e);
        let ty = ty::expr_ty(cx.tcx, e);
        let is_float = ty::type_is_fp(ty);
        ret alt u {
          ast::box(_)  |
          ast::uniq(_) |
          ast::deref   { cx.sess.span_bug(e.span,
                           "bad unop type in trans_const_expr"); }
          ast::not    { llvm::LLVMConstNot(te) }
          ast::neg    {
            if is_float { llvm::LLVMConstFNeg(te) }
            else        { llvm::LLVMConstNeg(te) }
          }
        }
      }
      ast::expr_cast(base, tp) {
        let ety = ty::expr_ty(cx.tcx, e), llty = type_of(cx, ety);
        let basety = ty::expr_ty(cx.tcx, base);
        let v = trans_const_expr(cx, base);
        alt check (cast_type_kind(basety), cast_type_kind(ety)) {
          (cast_integral, cast_integral) {
            let s = if ty::type_is_signed(basety) { True } else { False };
            llvm::LLVMConstIntCast(v, llty, s)
          }
          (cast_integral, cast_float) {
            if ty::type_is_signed(basety) { llvm::LLVMConstSIToFP(v, llty) }
            else { llvm::LLVMConstUIToFP(v, llty) }
          }
          (cast_float, cast_float) { llvm::LLVMConstFPCast(v, llty) }
          (cast_float, cast_integral) {
            if ty::type_is_signed(ety) { llvm::LLVMConstFPToSI(v, llty) }
            else { llvm::LLVMConstFPToUI(v, llty) }
          }
        }
      }
      _ { cx.sess.span_bug(e.span,
            "bad constant expression type in trans_const_expr"); }
    }
}

fn trans_const(ccx: @crate_ctxt, e: @ast::expr, id: ast::node_id) {
    let v = trans_const_expr(ccx, e);

    // The scalars come back as 1st class LLVM vals
    // which we have to stick into global constants.
    let g = get_item_val(ccx, id);
    llvm::LLVMSetInitializer(g, v);
    llvm::LLVMSetGlobalConstant(g, True);
}

fn trans_item(ccx: @crate_ctxt, item: ast::item) {
    let path = alt check ccx.tcx.items.get(item.id) {
      ast_map::node_item(_, p) { p }
    };
    alt item.node {
      ast::item_fn(decl, tps, body) {
        if decl.purity == ast::crust_fn  {
            let llfndecl = get_item_val(ccx, item.id);
            native::trans_crust_fn(ccx, *path + [path_name(item.ident)],
                                   decl, body, llfndecl, item.id);
        } else if tps.len() == 0u {
            let llfndecl = get_item_val(ccx, item.id);
            trans_fn(ccx, *path + [path_name(item.ident)], decl, body,
                     llfndecl, no_self, none, item.id, none);
        } else {
            for stmt in body.node.stmts {
                alt stmt.node {
                  ast::stmt_decl(@{node: ast::decl_item(i), _}, _) {
                    trans_item(ccx, *i);
                  }
                  _ {}
                }
            }
        }
      }
      ast::item_impl(tps, _, _, ms) {
        impl::trans_impl(ccx, *path, item.ident, ms, tps);
      }
      ast::item_res(decl, tps, body, dtor_id, ctor_id) {
        if tps.len() == 0u {
            let llctor_decl = get_item_val(ccx, ctor_id);
            trans_res_ctor(ccx, *path, decl, ctor_id, none, llctor_decl);

            let lldtor_decl = get_item_val(ccx, item.id);
            trans_fn(ccx, *path + [path_name(item.ident)], decl, body,
                     lldtor_decl, no_self, none, dtor_id, none);
        }
      }
      ast::item_mod(m) {
        trans_mod(ccx, m);
      }
      ast::item_enum(variants, tps) {
        if tps.len() == 0u {
            let degen = variants.len() == 1u;
            let vi = ty::enum_variants(ccx.tcx, local_def(item.id));
            let i = 0;
            for variant: ast::variant in variants {
                if variant.node.args.len() > 0u {
                    let llfn = get_item_val(ccx, variant.node.id);
                    trans_enum_variant(ccx, item.id, variant,
                                       vi[i].disr_val, degen,
                                       none, llfn);
                }
                i += 1;
            }
        }
      }
      ast::item_const(_, expr) { trans_const(ccx, expr, item.id); }
      ast::item_native_mod(native_mod) {
        let abi = alt attr::native_abi(item.attrs) {
          either::right(abi_) { abi_ }
          either::left(msg) { ccx.sess.span_fatal(item.span, msg) }
        };
        native::trans_native_mod(ccx, native_mod, abi);
      }
      ast::item_class(tps, items, ctor) {
        // FIXME factor our ctor translation, call from monomorphic_fn
        let llctor_decl = get_item_val(ccx, ctor.node.id);
        // Translate the ctor
        // First, add a preamble that
        // generates a new name, obj:
        // let obj = { ... } (uninit record fields)
        let rslt_path_ = {global: false,
                          idents: ["obj"],
                          types: []}; // ??
        let rslt_path = @{node: rslt_path_,
                          span: ctor.node.body.span};
        let rslt_id : ast::node_id = ccx.sess.next_node_id();
        let rslt_pat : @ast::pat =
            @{id: ccx.sess.next_node_id(),
              node: ast::pat_ident(rslt_path, none),
              span: ctor.node.body.span};
        // Set up obj's type
        let rslt_ast_ty : @ast::ty = @{id: ccx.sess.next_node_id(),
                                       node: ast::ty_infer,
                                       span: ctor.node.body.span};
        // kludgy
        let ty_args = [], i = 0u;
        for tp in tps {
            ty_args += [ty::mk_param(ccx.tcx, i,
                                     local_def(tps[i].id))];
        }
        let rslt_ty =  ty::mk_class(ccx.tcx,
                                    local_def(item.id),
                                    ty_args);
        // Set up a local for obj
        let rslt_loc_ : ast::local_ = {is_mutbl: true,
                                       ty: rslt_ast_ty, // ???
                                       pat: rslt_pat,
                                       init: none::<ast::initializer>,
                                       id: rslt_id};
        // Register a type for obj
        smallintmap::insert(*ccx.tcx.node_types,
                            rslt_loc_.id as uint, rslt_ty);
        // Create the decl statement that initializers obj
        let rslt_loc : @ast::local =
            @{node: rslt_loc_, span: ctor.node.body.span};
        let rslt_decl_ : ast::decl_ = ast::decl_local([rslt_loc]);
        let rslt_decl : @ast::decl
            = @{node: rslt_decl_, span: ctor.node.body.span};
        let prologue = @{node: ast::stmt_decl(rslt_decl,
                                              ccx.sess.next_node_id()),
                         span: ctor.node.body.span};
        let rslt_node_id = ccx.sess.next_node_id();
        ccx.tcx.def_map.insert(rslt_node_id,
                               ast::def_local(rslt_loc_.id, true));
        // And give the statement a type
        smallintmap::insert(*ccx.tcx.node_types,
                            rslt_node_id as uint, rslt_ty);
        // The result expression of the constructor is now a
        // reference to obj
        let rslt_expr : @ast::expr =
            @{id: rslt_node_id,
              node: ast::expr_path(rslt_path),
              span: ctor.node.body.span};
        let ctor_body_new : ast::blk_ = {stmts: [prologue]
                                             + ctor.node.body.node.stmts,
                                         expr: some(rslt_expr)
                                         with ctor.node.body.node};
        let ctor_body__ : ast::blk = {node: ctor_body_new
                                      with ctor.node.body};
        trans_fn(ccx, *path + [path_name(item.ident)], ctor.node.dec,
                 ctor_body__, llctor_decl, no_self,
                 none, ctor.node.id, some(rslt_expr));
        // TODO: translate methods!
      }
      _ {/* fall through */ }
    }
}

// Translate a module. Doing this amounts to translating the items in the
// module; there ends up being no artifact (aside from linkage names) of
// separate modules in the compiled program.  That's because modules exist
// only as a convenience for humans working with the code, to organize names
// and control visibility.
fn trans_mod(ccx: @crate_ctxt, m: ast::_mod) {
    for item in m.items { trans_item(ccx, *item); }
}

fn compute_ii_method_info(ccx: @crate_ctxt,
                          impl_did: ast::def_id,
                          m: @ast::method,
                          f: fn(ty::t, [ty::param_bounds], ast_map::path)) {
    let {bounds: impl_bnds, ty: impl_ty} =
        ty::lookup_item_type(ccx.tcx, impl_did);
    let m_bounds = *impl_bnds + param_bounds(ccx, m.tps);
    let impl_path = ty::item_path(ccx.tcx, impl_did);
    let m_path = impl_path + [path_name(m.ident)];
    f(impl_ty, m_bounds, m_path);
}

fn get_pair_fn_ty(llpairty: TypeRef) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the pair.
    ret struct_elt(llpairty, 0u);
}

fn register_fn(ccx: @crate_ctxt, sp: span, path: path, flav: str,
               node_id: ast::node_id) -> ValueRef {
    let t = ty::node_id_to_type(ccx.tcx, node_id);
    register_fn_full(ccx, sp, path, flav, node_id, t)
}

fn param_bounds(ccx: @crate_ctxt, tps: [ast::ty_param])
        -> [ty::param_bounds] {
    vec::map(tps) {|tp| ccx.tcx.ty_param_bounds.get(tp.id) }
}

fn register_fn_full(ccx: @crate_ctxt, sp: span, path: path, flav: str,
                    node_id: ast::node_id, node_type: ty::t) -> ValueRef {
    let llfty = type_of_fn_from_ty(ccx, node_type);
    register_fn_fuller(ccx, sp, path, flav, node_id, node_type,
                       lib::llvm::CCallConv, llfty)
}

fn register_fn_fuller(ccx: @crate_ctxt, sp: span, path: path, _flav: str,
                      node_id: ast::node_id, node_type: ty::t,
                      cc: lib::llvm::CallConv, llfty: TypeRef) -> ValueRef {
    let ps: str = mangle_exported_name(ccx, path, node_type);
    let llfn: ValueRef = decl_fn(ccx.llmod, ps, cc, llfty);
    ccx.item_symbols.insert(node_id, ps);

    #debug["register_fn_fuller created fn %s for item %d with path %s",
           val_str(ccx.tn, llfn), node_id, ast_map::path_to_str(path)];

    let is_main = is_main_name(path) && !ccx.sess.building_library;
    if is_main { create_main_wrapper(ccx, sp, llfn, node_type); }
    llfn
}

// Create a _rust_main(args: [str]) function which will be called from the
// runtime rust_start function
fn create_main_wrapper(ccx: @crate_ctxt, sp: span, main_llfn: ValueRef,
                       main_node_type: ty::t) {

    if ccx.main_fn != none::<ValueRef> {
        ccx.sess.span_fatal(sp, "multiple 'main' functions");
    }

    let main_takes_argv =
        // invariant!
        alt ty::get(main_node_type).struct {
          ty::ty_fn({inputs, _}) { inputs.len() != 0u }
          _ { ccx.sess.span_fatal(sp, "main has a non-function type"); }
        };

    let llfn = create_main(ccx, main_llfn, main_takes_argv);
    ccx.main_fn = some(llfn);
    create_entry_fn(ccx, llfn);

    fn create_main(ccx: @crate_ctxt, main_llfn: ValueRef,
                   takes_argv: bool) -> ValueRef {
        let unit_ty = ty::mk_str(ccx.tcx);
        let vecarg_ty: ty::arg =
            {mode: ast::expl(ast::by_val),
             ty: ty::mk_vec(ccx.tcx, {ty: unit_ty, mutbl: ast::m_imm})};
        let nt = ty::mk_nil(ccx.tcx);
        let llfty = type_of_fn(ccx, [vecarg_ty], nt);
        let llfdecl = decl_fn(ccx.llmod, "_rust_main",
                              lib::llvm::CCallConv, llfty);

        let fcx = new_fn_ctxt(ccx, [], llfdecl, none);

        let bcx = top_scope_block(fcx, none);
        let lltop = bcx.llbb;

        let lloutputarg = llvm::LLVMGetParam(llfdecl, 0 as c_uint);
        let llenvarg = llvm::LLVMGetParam(llfdecl, 1 as c_uint);
        let args = [lloutputarg, llenvarg];
        if takes_argv { args += [llvm::LLVMGetParam(llfdecl, 2 as c_uint)]; }
        Call(bcx, main_llfn, args);
        build_return(bcx);

        finish_fn(fcx, lltop);

        ret llfdecl;
    }

    fn create_entry_fn(ccx: @crate_ctxt, rust_main: ValueRef) {
        #[cfg(target_os = "win32")]
        fn main_name() -> str { ret "WinMain@16"; }
        #[cfg(target_os = "macos")]
        fn main_name() -> str { ret "main"; }
        #[cfg(target_os = "linux")]
        fn main_name() -> str { ret "main"; }
        #[cfg(target_os = "freebsd")]
        fn main_name() -> str { ret "main"; }
        let llfty = T_fn([ccx.int_type, ccx.int_type], ccx.int_type);
        let llfn = decl_cdecl_fn(ccx.llmod, main_name(), llfty);
        let llbb = str::as_c_str("top", {|buf|
            llvm::LLVMAppendBasicBlock(llfn, buf)
        });
        let bld = *ccx.builder;
        llvm::LLVMPositionBuilderAtEnd(bld, llbb);
        let crate_map = ccx.crate_map;
        let start_ty = T_fn([val_ty(rust_main), ccx.int_type, ccx.int_type,
                             val_ty(crate_map)], ccx.int_type);
        let start = str::as_c_str("rust_start", {|buf|
            llvm::LLVMAddGlobal(ccx.llmod, start_ty, buf)
        });
        let args = [rust_main, llvm::LLVMGetParam(llfn, 0 as c_uint),
                    llvm::LLVMGetParam(llfn, 1 as c_uint), crate_map];
        let result = unsafe {
            llvm::LLVMBuildCall(bld, start, vec::unsafe::to_ptr(args),
                                args.len() as c_uint, noname())
        };
        llvm::LLVMBuildRet(bld, result);
    }
}

// Create a /real/ closure: this is like create_fn_pair, but creates a
// a fn value on the stack with a specified environment (which need not be
// on the stack).
fn create_real_fn_pair(cx: block, llfnty: TypeRef, llfn: ValueRef,
                       llenvptr: ValueRef) -> ValueRef {
    let pair = alloca(cx, T_fn_pair(cx.ccx(), llfnty));
    fill_fn_pair(cx, pair, llfn, llenvptr);
    ret pair;
}

fn fill_fn_pair(bcx: block, pair: ValueRef, llfn: ValueRef,
                llenvptr: ValueRef) {
    let ccx = bcx.ccx();
    let code_cell = GEPi(bcx, pair, [0, abi::fn_field_code]);
    Store(bcx, llfn, code_cell);
    let env_cell = GEPi(bcx, pair, [0, abi::fn_field_box]);
    let llenvblobptr = PointerCast(bcx, llenvptr, T_opaque_box_ptr(ccx));
    Store(bcx, llenvblobptr, env_cell);
}

fn item_path(ccx: @crate_ctxt, i: @ast::item) -> path {
    *alt check ccx.tcx.items.get(i.id) {
      ast_map::node_item(_, p) { p }
    } + [path_name(i.ident)]
}

fn get_item_val(ccx: @crate_ctxt, id: ast::node_id) -> ValueRef {
    alt ccx.item_vals.find(id) {
      some(v) { v }
      none {
        let val = alt check ccx.tcx.items.get(id) {
          ast_map::node_item(i, pth) {
            let my_path = *pth + [path_name(i.ident)];
            alt check i.node {
              ast::item_const(_, _) {
                let typ = ty::node_id_to_type(ccx.tcx, i.id);
                let s = mangle_exported_name(ccx, my_path, typ);
                let g = str::as_c_str(s, {|buf|
                    llvm::LLVMAddGlobal(ccx.llmod, type_of(ccx, typ), buf)
                });
                ccx.item_symbols.insert(i.id, s);
                g
              }
              ast::item_fn(decl, _, _) {
                let llfn = if decl.purity != ast::crust_fn {
                    register_fn(ccx, i.span, my_path, "fn", i.id)
                } else {
                    native::register_crust_fn(ccx, i.span, my_path, i.id)
                };
                set_inline_hint_if_appr(i.attrs, llfn);
                llfn
              }
              ast::item_res(_, _, _, dtor_id, _) {
                // Note that the destructor is associated with the item's id,
                // not the dtor_id. This is a bit counter-intuitive, but
                // simplifies ty_res, which would have to carry around two
                // def_ids otherwise -- one to identify the type, and one to
                // find the dtor symbol.
                let t = ty::node_id_to_type(ccx.tcx, dtor_id);
                register_fn_full(ccx, i.span, my_path + [path_name("dtor")],
                                 "res_dtor", i.id, t)
              }
            }
          }
          ast_map::node_method(m, impl_id, pth) {
            let mty = ty::node_id_to_type(ccx.tcx, id);
            let pth = *pth + [path_name(int::str(impl_id.node)),
                              path_name(m.ident)];
            let llfn = register_fn_full(ccx, m.span, pth, "impl_method",
                                        id, mty);
            set_inline_hint_if_appr(m.attrs, llfn);
            llfn
          }
          ast_map::node_native_item(ni, _, pth) {
            native::decl_native_fn(ccx, ni, *pth + [path_name(ni.ident)])
          }
          ast_map::node_ctor(i) {
            alt check i.node {
              ast::item_res(_, _, _, _, _) {
                let my_path = item_path(ccx, i);
                let llctor = register_fn(ccx, i.span, my_path, "res_ctor",
                                         id);
                set_inline_hint(llctor);
                llctor
              }
              ast::item_class(_, _, ctor) {
                register_fn(ccx, i.span, item_path(ccx, i), "ctor", id)
              }
            }
          }
          ast_map::node_variant(v, enm, pth) {
            assert v.node.args.len() != 0u;
            let pth = *pth + [path_name(enm.ident), path_name(v.node.name)];
            let llfn = alt check enm.node {
              ast::item_enum(_, _) {
                register_fn(ccx, v.span, pth, "enum", id)
              }
            };
            set_inline_hint(llfn);
            llfn
          }
        };
        ccx.item_vals.insert(id, val);
        val
      }
    }
}

// The constant translation pass.
fn trans_constant(ccx: @crate_ctxt, it: @ast::item) {
    alt it.node {
      ast::item_enum(variants, _) {
        let vi = ty::enum_variants(ccx.tcx, {crate: ast::local_crate,
                                             node: it.id});
        let i = 0, path = item_path(ccx, it);
        for variant in variants {
            let p = path + [path_name(variant.node.name),
                            path_name("discrim")];
            let s = mangle_exported_name(ccx, p, ty::mk_int(ccx.tcx));
            let disr_val = vi[i].disr_val;
            let discrim_gvar = str::as_c_str(s, {|buf|
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
      _ { }
    }
}

fn trans_constants(ccx: @crate_ctxt, crate: @ast::crate) {
    visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
        visit_item: bind trans_constant(ccx, _)
        with *visit::default_simple_visitor()
    }));
}

fn vp2i(cx: block, v: ValueRef) -> ValueRef {
    let ccx = cx.ccx();
    ret PtrToInt(cx, v, ccx.int_type);
}

fn p2i(ccx: @crate_ctxt, v: ValueRef) -> ValueRef {
    ret llvm::LLVMConstPtrToInt(v, ccx.int_type);
}

fn declare_intrinsics(llmod: ModuleRef) -> hashmap<str, ValueRef> {
    let T_memmove32_args: [TypeRef] =
        [T_ptr(T_i8()), T_ptr(T_i8()), T_i32(), T_i32(), T_i1()];
    let T_memmove64_args: [TypeRef] =
        [T_ptr(T_i8()), T_ptr(T_i8()), T_i64(), T_i32(), T_i1()];
    let T_memset32_args: [TypeRef] =
        [T_ptr(T_i8()), T_i8(), T_i32(), T_i32(), T_i1()];
    let T_memset64_args: [TypeRef] =
        [T_ptr(T_i8()), T_i8(), T_i64(), T_i32(), T_i1()];
    let T_trap_args: [TypeRef] = [];
    let gcroot =
        decl_cdecl_fn(llmod, "llvm.gcroot",
                      T_fn([T_ptr(T_ptr(T_i8())), T_ptr(T_i8())], T_void()));
    let gcread =
        decl_cdecl_fn(llmod, "llvm.gcread",
                      T_fn([T_ptr(T_i8()), T_ptr(T_ptr(T_i8()))], T_void()));
    let memmove32 =
        decl_cdecl_fn(llmod, "llvm.memmove.p0i8.p0i8.i32",
                      T_fn(T_memmove32_args, T_void()));
    let memmove64 =
        decl_cdecl_fn(llmod, "llvm.memmove.p0i8.p0i8.i64",
                      T_fn(T_memmove64_args, T_void()));
    let memset32 =
        decl_cdecl_fn(llmod, "llvm.memset.p0i8.i32",
                      T_fn(T_memset32_args, T_void()));
    let memset64 =
        decl_cdecl_fn(llmod, "llvm.memset.p0i8.i64",
                      T_fn(T_memset64_args, T_void()));
    let trap = decl_cdecl_fn(llmod, "llvm.trap", T_fn(T_trap_args, T_void()));
    let intrinsics = str_hash::<ValueRef>();
    intrinsics.insert("llvm.gcroot", gcroot);
    intrinsics.insert("llvm.gcread", gcread);
    intrinsics.insert("llvm.memmove.p0i8.p0i8.i32", memmove32);
    intrinsics.insert("llvm.memmove.p0i8.p0i8.i64", memmove64);
    intrinsics.insert("llvm.memset.p0i8.i32", memset32);
    intrinsics.insert("llvm.memset.p0i8.i64", memset64);
    intrinsics.insert("llvm.trap", trap);
    ret intrinsics;
}

fn declare_dbg_intrinsics(llmod: ModuleRef,
                          intrinsics: hashmap<str, ValueRef>) {
    let declare =
        decl_cdecl_fn(llmod, "llvm.dbg.declare",
                      T_fn([T_metadata(), T_metadata()], T_void()));
    let value =
        decl_cdecl_fn(llmod, "llvm.dbg.value",
                      T_fn([T_metadata(), T_i64(), T_metadata()], T_void()));
    intrinsics.insert("llvm.dbg.declare", declare);
    intrinsics.insert("llvm.dbg.value", value);
}

fn trap(bcx: block) {
    let v: [ValueRef] = [];
    alt bcx.ccx().intrinsics.find("llvm.trap") {
      some(x) { Call(bcx, x, v); }
      _ { bcx.sess().bug("unbound llvm.trap in trap"); }
    }
}

fn create_module_map(ccx: @crate_ctxt) -> ValueRef {
    let elttype = T_struct([ccx.int_type, ccx.int_type]);
    let maptype = T_array(elttype, ccx.module_data.size() + 1u);
    let map = str::as_c_str("_rust_mod_map", {|buf|
        llvm::LLVMAddGlobal(ccx.llmod, maptype, buf)
    });
    lib::llvm::SetLinkage(map, lib::llvm::InternalLinkage);
    let elts: [ValueRef] = [];
    ccx.module_data.items {|key, val|
        let elt = C_struct([p2i(ccx, C_cstr(ccx, key)),
                            p2i(ccx, val)]);
        elts += [elt];
    };
    let term = C_struct([C_int(ccx, 0), C_int(ccx, 0)]);
    elts += [term];
    llvm::LLVMSetInitializer(map, C_array(elttype, elts));
    ret map;
}


fn decl_crate_map(sess: session::session, mapname: str,
                  llmod: ModuleRef) -> ValueRef {
    let targ_cfg = sess.targ_cfg;
    let int_type = T_int(targ_cfg);
    let n_subcrates = 1;
    let cstore = sess.cstore;
    while cstore::have_crate_data(cstore, n_subcrates) { n_subcrates += 1; }
    let mapname = if sess.building_library { mapname } else { "toplevel" };
    let sym_name = "_rust_crate_map_" + mapname;
    let arrtype = T_array(int_type, n_subcrates as uint);
    let maptype = T_struct([int_type, arrtype]);
    let map = str::as_c_str(sym_name, {|buf|
        llvm::LLVMAddGlobal(llmod, maptype, buf)
    });
    lib::llvm::SetLinkage(map, lib::llvm::ExternalLinkage);
    ret map;
}

// FIXME use hashed metadata instead of crate names once we have that
fn fill_crate_map(ccx: @crate_ctxt, map: ValueRef) {
    let subcrates: [ValueRef] = [];
    let i = 1;
    let cstore = ccx.sess.cstore;
    while cstore::have_crate_data(cstore, i) {
        let nm = "_rust_crate_map_" + cstore::get_crate_data(cstore, i).name;
        let cr = str::as_c_str(nm, {|buf|
            llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type, buf)
        });
        subcrates += [p2i(ccx, cr)];
        i += 1;
    }
    subcrates += [C_int(ccx, 0)];
    llvm::LLVMSetInitializer(map, C_struct(
        [p2i(ccx, create_module_map(ccx)),
         C_array(ccx.int_type, subcrates)]));
}

fn write_metadata(cx: @crate_ctxt, crate: @ast::crate) {
    if !cx.sess.building_library { ret; }
    let llmeta = C_bytes(metadata::encoder::encode_metadata(cx, crate));
    let llconst = C_struct([llmeta]);
    let llglobal = str::as_c_str("rust_metadata", {|buf|
        llvm::LLVMAddGlobal(cx.llmod, val_ty(llconst), buf)
    });
    llvm::LLVMSetInitializer(llglobal, llconst);
    str::as_c_str(cx.sess.targ_cfg.target_strs.meta_sect_name, {|buf|
        llvm::LLVMSetSection(llglobal, buf)
    });
    lib::llvm::SetLinkage(llglobal, lib::llvm::InternalLinkage);

    let t_ptr_i8 = T_ptr(T_i8());
    llglobal = llvm::LLVMConstBitCast(llglobal, t_ptr_i8);
    let llvm_used = str::as_c_str("llvm.used", {|buf|
        llvm::LLVMAddGlobal(cx.llmod, T_array(t_ptr_i8, 1u), buf)
    });
    lib::llvm::SetLinkage(llvm_used, lib::llvm::AppendingLinkage);
    llvm::LLVMSetInitializer(llvm_used, C_array(t_ptr_i8, [llglobal]));
}

// Writes the current ABI version into the crate.
fn write_abi_version(ccx: @crate_ctxt) {
    mk_global(ccx, "rust_abi_version", C_uint(ccx, abi::abi_version),
                     false);
}

fn trans_crate(sess: session::session, crate: @ast::crate, tcx: ty::ctxt,
               output: str, emap: resolve::exp_map, maps: maps)
    -> (ModuleRef, link::link_meta) {
    let sha = std::sha1::sha1();
    let link_meta = link::build_link_meta(sess, *crate, output, sha);

    // Append ".rc" to crate name as LLVM module identifier.
    //
    // LLVM code generator emits a ".file filename" directive
    // for ELF backends. Value of the "filename" is set as the
    // LLVM module identifier.  Due to a LLVM MC bug[1], LLVM
    // crashes if the module identifer is same as other symbols
    // such as a function name in the module.
    // 1. http://llvm.org/bugs/show_bug.cgi?id=11479
    let llmod_id = link_meta.name + ".rc";

    let llmod = str::as_c_str(llmod_id, {|buf|
        llvm::LLVMModuleCreateWithNameInContext
            (buf, llvm::LLVMGetGlobalContext())
    });
    let data_layout = sess.targ_cfg.target_strs.data_layout;
    let targ_triple = sess.targ_cfg.target_strs.target_triple;
    let _: () =
        str::as_c_str(data_layout,
                    {|buf| llvm::LLVMSetDataLayout(llmod, buf) });
    let _: () =
        str::as_c_str(targ_triple,
                    {|buf| llvm::LLVMSetTarget(llmod, buf) });
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
    lib::llvm::associate_type(tn, "taskptr", taskptr_type);
    let tydesc_type = T_tydesc(targ_cfg);
    lib::llvm::associate_type(tn, "tydesc", tydesc_type);
    let crate_map = decl_crate_map(sess, link_meta.name, llmod);
    let dbg_cx = if sess.opts.debuginfo {
        option::some(@{llmetadata: map::int_hash(),
                       names: new_namegen()})
    } else {
        option::none
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
          item_symbols: int_hash::<str>(),
          mutable main_fn: none::<ValueRef>,
          link_meta: link_meta,
          enum_sizes: ty::new_ty_hash(),
          discrims: ast_util::new_def_id_hash::<ValueRef>(),
          discrim_symbols: int_hash::<str>(),
          tydescs: ty::new_ty_hash(),
          external: util::common::new_def_hash(),
          monomorphized: map::hashmap(hash_mono_id, {|a, b| a == b}),
          type_use_cache: util::common::new_def_hash(),
          vtables: map::hashmap(hash_mono_id, {|a, b| a == b}),
          module_data: str_hash::<ValueRef>(),
          lltypes: ty::new_ty_hash(),
          names: new_namegen(),
          sha: sha,
          type_sha1s: ty::new_ty_hash(),
          type_short_names: ty::new_ty_hash(),
          tcx: tcx,
          maps: maps,
          stats:
              {mutable n_static_tydescs: 0u,
               mutable n_glues_created: 0u,
               mutable n_null_glues: 0u,
               mutable n_real_glues: 0u,
               fn_times: @mutable []},
          upcalls:
              upcall::declare_upcalls(targ_cfg, tn, tydesc_type,
                                      llmod),
          tydesc_type: tydesc_type,
          int_type: int_type,
          float_type: float_type,
          task_type: task_type,
          opaque_vec_type: T_opaque_vec(targ_cfg),
          builder: BuilderRef_res(llvm::LLVMCreateBuilder()),
          shape_cx: mk_ctxt(llmod),
          crate_map: crate_map,
          dbg_cx: dbg_cx,
          mutable do_not_commit_warning_issued: false};
    trans_constants(ccx, crate);
    trans_mod(ccx, crate.node.module);
    fill_crate_map(ccx, crate_map);
    emit_tydescs(ccx);
    gen_shape_tables(ccx);
    write_abi_version(ccx);

    // Translate the metadata.
    write_metadata(ccx, crate);
    if ccx.sess.opts.stats {
        #error("--- trans stats ---");
        #error("n_static_tydescs: %u", ccx.stats.n_static_tydescs);
        #error("n_glues_created: %u", ccx.stats.n_glues_created);
        #error("n_null_glues: %u", ccx.stats.n_null_glues);
        #error("n_real_glues: %u", ccx.stats.n_real_glues);

        for timing: {ident: str, time: int} in *ccx.stats.fn_times {
            #error("time: %s took %d ms", timing.ident, timing.time);
        }
    }
    ret (llmod, link_meta);
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
