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

import libc::{c_uint, c_ulonglong};
import std::{map, time, list};
import std::map::hashmap;
import std::map::{int_hash, str_hash};
import driver::session;
import session::session;
import syntax::attr;
import back::{link, abi, upcall};
import syntax::{ast, ast_util, codemap, ast_map};
import ast_util::{local_def, path_to_ident};
import syntax::visit;
import syntax::codemap::span;
import syntax::print::pprust::{expr_to_str, stmt_to_str, path_to_str};
import pat_util::*;
import visit::vt;
import util::common::is_main_name;
import lib::llvm::{llvm, mk_target_data, mk_type_names};
import lib::llvm::{ModuleRef, ValueRef, TypeRef, BasicBlockRef};
import lib::llvm::{True, False};
import link::{mangle_internal_name_by_type_only,
              mangle_internal_name_by_seq,
              mangle_internal_name_by_path,
              mangle_internal_name_by_path_and_seq,
              mangle_exported_name};
import metadata::{csearch, cstore, decoder, encoder};
import metadata::common::link_meta;
import util::ppaux;
import util::ppaux::{ty_to_str, ty_to_short_str};
import syntax::diagnostic::expect;

import build::*;
import shape::*;
import type_of::*;
import common::*;
import common::result;
import syntax::ast_map::{path, path_mod, path_name};
import syntax::parse::token::special_idents;

import std::smallintmap;
import option::{is_none, is_some};

// Destinations

// These are passed around by the code generating functions to track the
// destination of a computation's value.

enum dest {
    by_val(@mut ValueRef),
    save_in(ValueRef),
    ignore,
}

fn dest_str(ccx: @crate_ctxt, d: dest) -> ~str {
    match d {
      by_val(v) => fmt!("by_val(%s)", val_str(ccx.tn, *v)),
      save_in(v) => fmt!("save_in(%s)", val_str(ccx.tn, v)),
      ignore => ~"ignore"
    }
}

fn empty_dest_cell() -> @mut ValueRef {
    return @mut llvm::LLVMGetUndef(T_nil());
}

fn dup_for_join(dest: dest) -> dest {
    match dest {
      by_val(_) => by_val(empty_dest_cell()),
      _ => dest
    }
}

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

fn join_returns(parent_cx: block, in_cxs: ~[block],
                in_ds: ~[dest], out_dest: dest) -> block {
    let out = sub_block(parent_cx, ~"join");
    let mut reachable = false, i = 0u, phi = None;
    for vec::each(in_cxs) |cx| {
        if !cx.unreachable {
            Br(cx, out.llbb);
            reachable = true;
            match in_ds[i] {
              by_val(cell) => {
                if option::is_none(phi) {
                    phi = Some(EmptyPhi(out, val_ty(*cell)));
                }
                AddIncomingToPhi(option::get(phi), *cell, cx.llbb);
              }
              _ => ()
            }
        }
        i += 1u;
    }
    if !reachable {
        Unreachable(out);
    } else {
        match out_dest {
          by_val(cell) => *cell = option::get(phi),
          _ => ()
        }
    }
    return out;
}

// Used to put an immediate value in a dest.
fn store_in_dest(bcx: block, val: ValueRef, dest: dest) -> block {
    match dest {
      ignore => (),
      by_val(cell) => *cell = val,
      save_in(addr) => Store(bcx, val, addr)
    }
    return bcx;
}

fn get_dest_addr(dest: dest) -> ValueRef {
    match dest {
       save_in(a) => a,
       _ => fail ~"get_dest_addr: not a save_in"
    }
}

fn log_fn_time(ccx: @crate_ctxt, name: ~str, start: time::timespec,
               end: time::timespec) {
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

fn trans_free(cx: block, v: ValueRef) -> block {
    let _icx = cx.insn_ctxt("trans_free");
    trans_rtcall(cx, ~"free", ~[PointerCast(cx, v, T_ptr(T_i8()))], ignore)
}

fn trans_unique_free(cx: block, v: ValueRef) -> block {
    let _icx = cx.insn_ctxt("trans_unique_free");
    trans_rtcall(cx, ~"exchange_free", ~[PointerCast(cx, v, T_ptr(T_i8()))],
                 ignore)
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

fn alloca(cx: block, t: TypeRef) -> ValueRef {
    alloca_maybe_zeroed(cx, t, false)
}

fn alloca_zeroed(cx: block, t: TypeRef) -> ValueRef {
    alloca_maybe_zeroed(cx, t, true)
}

fn alloca_maybe_zeroed(cx: block, t: TypeRef, zero: bool) -> ValueRef {
    let _icx = cx.insn_ctxt("alloca");
    if cx.unreachable { return llvm::LLVMGetUndef(t); }
    let initcx = raw_block(cx.fcx, false, cx.fcx.llstaticallocas);
    let p = Alloca(initcx, t);
    if zero { memzero(initcx, p, t); }
    return p;
}

fn zero_mem(cx: block, llptr: ValueRef, t: ty::t) -> block {
    let _icx = cx.insn_ctxt("zero_mem");
    let bcx = cx;
    let ccx = cx.ccx();
    let llty = type_of(ccx, t);
    memzero(bcx, llptr, llty);
    return bcx;
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
    let size = IntCast(cx, llsize_of(ccx, llty), ccx.int_type);
    let align = C_i32(1i32);
    let volatile = C_bool(false);
    Call(cx, llintrinsicfn, ~[llptr, llzeroval, size, align, volatile]);
}

fn arrayalloca(cx: block, t: TypeRef, v: ValueRef) -> ValueRef {
    let _icx = cx.insn_ctxt("arrayalloca");
    if cx.unreachable { return llvm::LLVMGetUndef(t); }
    return ArrayAlloca(
        raw_block(cx.fcx, false, cx.fcx.llstaticallocas), t, v);
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
    GEPi(bcx, typed_blobptr, ~[0u, ix])
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
    let bodyptr = GEPi(bcx, boxptr, ~[1u]);
    PointerCast(bcx, bodyptr, T_ptr(type_of(ccx, body_t)))
}

// malloc_raw_dyn: allocates a box to contain a given type, but with a
// potentially dynamic size.
fn malloc_raw_dyn(bcx: block, t: ty::t, heap: heap,
                  size: ValueRef) -> result {
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
    lazily_emit_all_tydesc_glue(ccx, static_ti);

    // Allocate space:
    let tydesc = PointerCast(bcx, static_ti.tydesc, T_ptr(T_i8()));
    let rval = alloca_zeroed(bcx, T_ptr(T_i8()));
    let bcx = trans_rtcall(bcx, rtcall, ~[tydesc, size], save_in(rval));
    let retval = {bcx: bcx, val: PointerCast(bcx, Load(bcx, rval), llty)};
    return retval;
}

// malloc_raw: expects an unboxed type and returns a pointer to
// enough space for a box of that type.  This includes a rust_opaque_box
// header.
fn malloc_raw(bcx: block, t: ty::t, heap: heap) -> result {
    malloc_raw_dyn(bcx, t, heap, llsize_of(bcx.ccx(), type_of(bcx.ccx(), t)))
}

// malloc_general_dyn: usefully wraps malloc_raw_dyn; allocates a box,
// and pulls out the body
fn malloc_general_dyn(bcx: block, t: ty::t, heap: heap, size: ValueRef)
    -> {bcx: block, box: ValueRef, body: ValueRef} {
    let _icx = bcx.insn_ctxt("malloc_general");
    let {bcx: bcx, val: llbox} = malloc_raw_dyn(bcx, t, heap, size);
    let non_gc_box = non_gc_box_cast(bcx, llbox);
    let body = GEPi(bcx, non_gc_box, ~[0u, abi::box_field_body]);
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
        let inf = declare_tydesc(ccx, t);
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

// Chooses the addrspace for newly declared types.
fn declare_tydesc_addrspace(ccx: @crate_ctxt, t: ty::t) -> addrspace {
    if !ty::type_needs_drop(ccx.tcx, t) {
        return default_addrspace;
    } else if ty::type_is_immediate(t) {
        // For immediate types, we don't actually need an addrspace, because
        // e.g. boxed types include pointers to their contents which are
        // already correctly tagged with addrspaces.
        return default_addrspace;
    } else {
        return ccx.next_addrspace();
    }
}

// Generates the declaration for (but doesn't emit) a type descriptor.
fn declare_tydesc(ccx: @crate_ctxt, t: ty::t) -> @tydesc_info {
    let _icx = ccx.insn_ctxt("declare_tydesc");
    // If emit_tydescs already ran, then we shouldn't be creating any new
    // tydescs.
    assert !ccx.finished_tydescs;

    let llty = type_of(ccx, t);

    if ccx.sess.count_type_sizes() {
        io::println(fmt!("%u\t%s",
                         llsize_of_real(ccx, llty),
                         ty_to_str(ccx.tcx, t)));
    }

    let llsize = llsize_of(ccx, llty);
    let llalign = llalign_of(ccx, llty);
    let addrspace = declare_tydesc_addrspace(ccx, t);
    //XXX this triggers duplicate LLVM symbols
    let name = if false /*ccx.sess.opts.debuginfo*/ {
        mangle_internal_name_by_type_only(ccx, t, ~"tydesc")
    } else { mangle_internal_name_by_seq(ccx, ~"tydesc") };
    note_unique_llvm_symbol(ccx, name);
    log(debug, fmt!("+++ declare_tydesc %s %s", ty_to_str(ccx.tcx, t), name));
    let gvar = str::as_c_str(name, |buf| {
        llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type, buf)
    });
    let inf =
        @{ty: t,
          tydesc: gvar,
          size: llsize,
          align: llalign,
          addrspace: addrspace,
          mut take_glue: None,
          mut drop_glue: None,
          mut free_glue: None,
          mut visit_glue: None};
    log(debug, ~"--- declare_tydesc " + ppaux::ty_to_str(ccx.tcx, t));
    return inf;
}

type glue_helper = fn@(block, ValueRef, ty::t);

fn declare_generic_glue(ccx: @crate_ctxt, t: ty::t, llfnty: TypeRef,
                        name: ~str) -> ValueRef {
    let _icx = ccx.insn_ctxt("declare_generic_glue");
    let name = name;
    let mut fn_nm;
    //XXX this triggers duplicate LLVM symbols
    if false /*ccx.sess.opts.debuginfo*/ {
        fn_nm = mangle_internal_name_by_type_only(ccx, t, (~"glue_" + name));
    } else {
        fn_nm = mangle_internal_name_by_seq(ccx, (~"glue_" + name));
    }
    note_unique_llvm_symbol(ccx, fn_nm);
    let llfn = decl_cdecl_fn(ccx.llmod, fn_nm, llfnty);
    set_glue_inlining(llfn, t);
    return llfn;
}

fn make_generic_glue_inner(ccx: @crate_ctxt, t: ty::t,
                           llfn: ValueRef, helper: glue_helper) -> ValueRef {
    let _icx = ccx.insn_ctxt("make_generic_glue_inner");
    let fcx = new_fn_ctxt(ccx, ~[], llfn, None);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    ccx.stats.n_glues_created += 1u;
    // All glue functions take values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.
    //
    // llfn is expected be declared to take a parameter of the appropriate
    // type, so we don't need to explicitly cast the function parameter.

    let bcx = top_scope_block(fcx, None);
    let lltop = bcx.llbb;
    let llrawptr0 = llvm::LLVMGetParam(llfn, 3u as c_uint);
    helper(bcx, llrawptr0, t);
    finish_fn(fcx, lltop);
    return llfn;
}

fn make_generic_glue(ccx: @crate_ctxt, t: ty::t, llfn: ValueRef,
                     helper: glue_helper, name: ~str)
    -> ValueRef {
    let _icx = ccx.insn_ctxt("make_generic_glue");
    if !ccx.sess.trans_stats() {
        return make_generic_glue_inner(ccx, t, llfn, helper);
    }

    let start = time::get_time();
    let llval = make_generic_glue_inner(ccx, t, llfn, helper);
    let end = time::get_time();
    log_fn_time(ccx, ~"glue " + name + ~" " + ty_to_short_str(ccx.tcx, t),
                start, end);
    return llval;
}

fn emit_tydescs(ccx: @crate_ctxt) {
    let _icx = ccx.insn_ctxt("emit_tydescs");
    // As of this point, allow no more tydescs to be created.
    ccx.finished_tydescs = true;
    for ccx.tydescs.each |key, val| {
        let glue_fn_ty = T_ptr(T_generic_glue_fn(ccx));
        let ti = val;

        // Each of the glue functions needs to be cast to a generic type
        // before being put into the tydesc because we only have a singleton
        // tydesc type. Then we'll recast each function to its real type when
        // calling it.
        let take_glue =
            match copy ti.take_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };
        let drop_glue =
            match copy ti.drop_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };
        let free_glue =
            match copy ti.free_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };
        let visit_glue =
            match copy ti.visit_glue {
              None => { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              Some(v) => {
                ccx.stats.n_real_glues += 1u;
                llvm::LLVMConstPointerCast(v, glue_fn_ty)
              }
            };

        let shape = shape_of(ccx, key);
        let shape_tables =
            llvm::LLVMConstPointerCast(ccx.shape_cx.llshapetables,
                                       T_ptr(T_i8()));

        let tydesc =
            C_named_struct(ccx.tydesc_type,
                           ~[ti.size, // size
                             ti.align, // align
                             take_glue, // take_glue
                             drop_glue, // drop_glue
                             free_glue, // free_glue
                             visit_glue, // visit_glue
                             C_shape(ccx, shape), // shape
                             shape_tables]); // shape_tables

        let gvar = ti.tydesc;
        llvm::LLVMSetInitializer(gvar, tydesc);
        llvm::LLVMSetGlobalConstant(gvar, True);
        lib::llvm::SetLinkage(gvar, lib::llvm::InternalLinkage);

        // Index tydesc by addrspace.
        if ti.addrspace > gc_box_addrspace {
            let llty = T_ptr(ccx.tydesc_type);
            let addrspace_name = #fmt("_gc_addrspace_metadata_%u",
                                      ti.addrspace as uint);
            let addrspace_gvar = str::as_c_str(addrspace_name, |buf| {
                llvm::LLVMAddGlobal(ccx.llmod, llty, buf)
            });
            lib::llvm::SetLinkage(addrspace_gvar, lib::llvm::InternalLinkage);
            llvm::LLVMSetInitializer(addrspace_gvar, gvar);
        }
    };
}

fn make_take_glue(bcx: block, v: ValueRef, t: ty::t) {
    let _icx = bcx.insn_ctxt("make_take_glue");
    // NB: v is a *pointer* to type t here, not a direct value.
    let bcx = match ty::get(t).struct {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) => {
        incr_refcnt_of_boxed(bcx, Load(bcx, v)); bcx
      }
      ty::ty_uniq(_) => {
        let {bcx, val} = uniq::duplicate(bcx, Load(bcx, v), t);
        Store(bcx, val, v);
        bcx
      }
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) => {
        let {bcx, val} = tvec::duplicate_uniq(bcx, Load(bcx, v), t);
        Store(bcx, val, v);
        bcx
      }
      ty::ty_evec(_, ty::vstore_slice(_))
      | ty::ty_estr(ty::vstore_slice(_)) => {
        bcx
      }
      ty::ty_fn(_) => {
        closure::make_fn_glue(bcx, v, t, take_ty)
      }
      ty::ty_trait(_, _, _) => {
        let llbox = Load(bcx, GEPi(bcx, v, ~[0u, 1u]));
        incr_refcnt_of_boxed(bcx, llbox);
        bcx
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_take_glue(bcx, ck, v)
      }
      _ if ty::type_is_structural(t) => {
        iter_structural_ty(bcx, v, t, take_ty)
      }
      _ => bcx
    };

    build_return(bcx);
}

fn incr_refcnt_of_boxed(cx: block, box_ptr: ValueRef) {
    let _icx = cx.insn_ctxt("incr_refcnt_of_boxed");
    let ccx = cx.ccx();
    maybe_validate_box(cx, box_ptr);
    let rc_ptr = GEPi(cx, box_ptr, ~[0u, abi::box_field_refcnt]);
    let rc = Load(cx, rc_ptr);
    let rc = Add(cx, rc, C_int(ccx, 1));
    Store(cx, rc, rc_ptr);
}

fn make_visit_glue(bcx: block, v: ValueRef, t: ty::t) {
    let _icx = bcx.insn_ctxt("make_visit_glue");
    let mut bcx = bcx;
    let ty_visitor_name = special_idents::ty_visitor;
    assert bcx.ccx().tcx.intrinsic_defs.contains_key(ty_visitor_name);
    let (trait_id, ty) = bcx.ccx().tcx.intrinsic_defs.get(ty_visitor_name);
    let v = PointerCast(bcx, v, T_ptr(type_of::type_of(bcx.ccx(), ty)));
    bcx = reflect::emit_calls_to_trait_visit_ty(bcx, t, v, trait_id);
    build_return(bcx);
}


fn make_free_glue(bcx: block, v: ValueRef, t: ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = bcx.insn_ctxt("make_free_glue");
    let ccx = bcx.ccx();
    let bcx = match ty::get(t).struct {
      ty::ty_box(body_mt) => {
        let v = Load(bcx, v);
        let body = GEPi(bcx, v, ~[0u, abi::box_field_body]);
        // Cast away the addrspace of the box pointer.
        let body = PointerCast(bcx, body, T_ptr(type_of(ccx, body_mt.ty)));
        let bcx = drop_ty(bcx, body, body_mt.ty);
        trans_free(bcx, v)
      }
      ty::ty_opaque_box => {
        let v = Load(bcx, v);
        let td = Load(bcx, GEPi(bcx, v, ~[0u, abi::box_field_tydesc]));
        let valptr = GEPi(bcx, v, ~[0u, abi::box_field_body]);
        // Generate code that, dynamically, indexes into the
        // tydesc and calls the drop glue that got set dynamically
        call_tydesc_glue_full(bcx, valptr, td, abi::tydesc_field_drop_glue,
                              None);
        trans_free(bcx, v)
      }
      ty::ty_uniq(*) => {
        uniq::make_free_glue(bcx, v, t)
      }
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) |
      ty::ty_evec(_, ty::vstore_box) | ty::ty_estr(ty::vstore_box) => {
        make_free_glue(bcx, v,
                       tvec::expand_boxed_vec_ty(bcx.tcx(), t));
        return;
      }
      ty::ty_fn(_) => {
        closure::make_fn_glue(bcx, v, t, free_ty)
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_free_glue(bcx, ck, v)
      }
      ty::ty_class(did, ref substs) => {
        // Call the dtor if there is one
        do option::map_default(ty::ty_dtor(bcx.tcx(), did), bcx) |dt_id| {
            trans_class_drop(bcx, v, dt_id, did, substs)
        }
      }
      _ => bcx
    };
    build_return(bcx);
}

fn trans_class_drop(bcx: block, v0: ValueRef, dtor_did: ast::def_id,
                    class_did: ast::def_id,
                    substs: &ty::substs) -> block {
  let drop_flag = GEPi(bcx, v0, ~[0u, 0u]);
    do with_cond(bcx, IsNotNull(bcx, Load(bcx, drop_flag))) |cx| {
    let mut bcx = cx;
      // We have to cast v0
     let classptr = GEPi(bcx, v0, ~[0u, 1u]);
     // Find and call the actual destructor
     let dtor_addr = get_res_dtor(bcx.ccx(), dtor_did, class_did, substs.tps);
     // The second argument is the "self" argument for drop
     let params = lib::llvm::fn_ty_param_tys
         (llvm::LLVMGetElementType
          (llvm::LLVMTypeOf(dtor_addr)));
     // Class dtors have no explicit args, so the params should just consist
     // of the output pointer and the environment (self)
     assert(params.len() == 2u);
     let self_arg = PointerCast(bcx, v0, params[1u]);
     let args = ~[bcx.fcx.llretptr, self_arg];
     Call(bcx, dtor_addr, args);
     // Drop the fields
     for vec::eachi(ty::class_items_as_mutable_fields(bcx.tcx(), class_did,
                                                      substs))
         |i, fld| {
        let llfld_a = GEPi(bcx, classptr, ~[0u, i]);
        bcx = drop_ty(bcx, llfld_a, fld.mt.ty);
     }
     Store(bcx, C_u8(0u), drop_flag);
     bcx
  }
}


fn make_drop_glue(bcx: block, v0: ValueRef, t: ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let _icx = bcx.insn_ctxt("make_drop_glue");
    let ccx = bcx.ccx();
    let bcx = match ty::get(t).struct {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_estr(ty::vstore_box) | ty::ty_evec(_, ty::vstore_box) => {
        decr_refcnt_maybe_free(bcx, Load(bcx, v0), t)
      }
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) => {
        free_ty(bcx, v0, t)
      }
      ty::ty_unboxed_vec(_) => {
        tvec::make_drop_glue_unboxed(bcx, v0, t)
      }
      ty::ty_class(did, ref substs) => {
        let tcx = bcx.tcx();
        match ty::ty_dtor(tcx, did) {
          Some(dtor) => {
            trans_class_drop(bcx, v0, dtor, did, substs)
          }
          None => {
            // No dtor? Just the default case
            iter_structural_ty(bcx, v0, t, drop_ty)
          }
        }
      }
      ty::ty_fn(_) => {
        closure::make_fn_glue(bcx, v0, t, drop_ty)
      }
      ty::ty_trait(_, _, _) => {
        let llbox = Load(bcx, GEPi(bcx, v0, ~[0u, 1u]));
        decr_refcnt_maybe_free(bcx, llbox, ty::mk_opaque_box(ccx.tcx))
      }
      ty::ty_opaque_closure_ptr(ck) => {
        closure::make_opaque_cbox_drop_glue(bcx, ck, v0)
      }
      _ => {
        if ty::type_needs_drop(ccx.tcx, t) &&
            ty::type_is_structural(t) {
            iter_structural_ty(bcx, v0, t, drop_ty)
        } else { bcx }
      }
    };
    build_return(bcx);
}

fn get_res_dtor(ccx: @crate_ctxt, did: ast::def_id,
                parent_id: ast::def_id, substs: ~[ty::t])
   -> ValueRef {
    let _icx = ccx.insn_ctxt("trans_res_dtor");
    if (substs.len() > 0u) {
        let did = if did.crate != ast::local_crate {
            maybe_instantiate_inline(ccx, did)
        } else { did };
        assert did.crate == ast::local_crate;
        monomorphic_fn(ccx, did, substs, None, None).val
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

fn maybe_validate_box(_cx: block, _box_ptr: ValueRef) {
    // Uncomment this when debugging annoying use-after-free
    // bugs.  But do not commit with this uncommented!  Big performance hit.

    // let cx = _cx, box_ptr = _box_ptr;
    // let ccx = cx.ccx();
    // warn_not_to_commit(ccx, "validate_box() is uncommented");
    // let raw_box_ptr = PointerCast(cx, box_ptr, T_ptr(T_i8()));
    // Call(cx, ccx.upcalls.validate_box, ~[raw_box_ptr]);
}

fn decr_refcnt_maybe_free(bcx: block, box_ptr: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("decr_refcnt_maybe_free");
    let ccx = bcx.ccx();
    maybe_validate_box(bcx, box_ptr);

    do with_cond(bcx, IsNotNull(bcx, box_ptr)) |bcx| {
        let rc_ptr = GEPi(bcx, box_ptr, ~[0u, abi::box_field_refcnt]);
        let rc = Sub(bcx, Load(bcx, rc_ptr), C_int(ccx, 1));
        Store(bcx, rc, rc_ptr);
        let zero_test = ICmp(bcx, lib::llvm::IntEQ, C_int(ccx, 0), rc);
        with_cond(bcx, zero_test, |bcx| free_ty_immediate(bcx, box_ptr, t))
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
                        t: ty::t, op: ast::binop) -> result {
    let f = |a| compare_scalar_values(cx, lhs, rhs, a, op);

    match ty::get(t).struct {
      ty::ty_nil => return rslt(cx, f(nil_type)),
      ty::ty_bool | ty::ty_ptr(_) => return rslt(cx, f(unsigned_int)),
      ty::ty_int(_) => return rslt(cx, f(signed_int)),
      ty::ty_uint(_) => return rslt(cx, f(unsigned_int)),
      ty::ty_float(_) => return rslt(cx, f(floating_point)),
      ty::ty_type => {
        return rslt(trans_fail(cx, None,
                            ~"attempt to compare values of type type"),
                 C_nil());
      }
      _ => {
        // Should never get here, because t is scalar.
        cx.sess().bug(~"non-scalar type passed to \
                                 compare_scalar_types");
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

fn load_inbounds(cx: block, p: ValueRef, idxs: ~[uint]) -> ValueRef {
    return Load(cx, GEPi(cx, p, idxs));
}

fn store_inbounds(cx: block, v: ValueRef, p: ValueRef,
                  idxs: ~[uint]) {
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
      ty::ty_rec(fields) => {
        for vec::eachi(fields) |i, fld| {
            let llfld_a = GEPi(cx, av, ~[0u, i]);
            cx = f(cx, llfld_a, fld.mt.ty);
        }
      }
      ty::ty_estr(ty::vstore_fixed(_)) |
      ty::ty_evec(_, ty::vstore_fixed(_)) => {
        let (base, len) = tvec::get_base_and_len(cx, av, t);
        cx = tvec::iter_vec_raw(cx, base, t, len, f);
      }
      ty::ty_tup(args) => {
        for vec::eachi(args) |i, arg| {
            let llfld_a = GEPi(cx, av, ~[0u, i]);
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
        let lldiscrim_a_ptr = GEPi(cx, av_enum, ~[0u, 0u]);
        let llunion_a_ptr = GEPi(cx, av_enum, ~[0u, 1u]);
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
      ty::ty_class(did, ref substs) => {
          // Take the drop bit into account
          let classptr = if is_some(ty::ty_dtor(cx.tcx(), did)) {
                  GEPi(cx, av, ~[0u, 1u])
              }
          else { av };
        for vec::eachi(ty::class_items_as_mutable_fields(cx.tcx(), did,
                                                         substs))
            |i, fld| {
               let llfld_a = GEPi(cx, classptr, ~[0u, i]);
               cx = f(cx, llfld_a, fld.mt.ty);
           }
      }
      _ => cx.sess().unimpl(~"type in iter_structural_ty")
    }
    return cx;
}

fn lazily_emit_all_tydesc_glue(ccx: @crate_ctxt,
                               static_ti: @tydesc_info) {
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_take_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_drop_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_free_glue, static_ti);
    lazily_emit_tydesc_glue(ccx, abi::tydesc_field_visit_glue, static_ti);
}

fn lazily_emit_tydesc_glue(ccx: @crate_ctxt, field: uint,
                           ti: @tydesc_info) {
    let _icx = ccx.insn_ctxt("lazily_emit_tydesc_glue");
    let llfnty = type_of_glue_fn(ccx, ti.ty);
    if field == abi::tydesc_field_take_glue {
        match ti.take_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue TAKE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"take");
            ti.take_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_take_glue, ~"take");
            debug!("--- lazily_emit_tydesc_glue TAKE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    } else if field == abi::tydesc_field_drop_glue {
        match ti.drop_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue DROP %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"drop");
            ti.drop_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_drop_glue, ~"drop");
            debug!("--- lazily_emit_tydesc_glue DROP %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    } else if field == abi::tydesc_field_free_glue {
        match ti.free_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue FREE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"free");
            ti.free_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_free_glue, ~"free");
            debug!("--- lazily_emit_tydesc_glue FREE %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    } else if field == abi::tydesc_field_visit_glue {
        match ti.visit_glue {
          Some(_) => (),
          None => {
            debug!("+++ lazily_emit_tydesc_glue VISIT %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
            let glue_fn = declare_generic_glue(ccx, ti.ty, llfnty, ~"visit");
            ti.visit_glue = Some(glue_fn);
            make_generic_glue(ccx, ti.ty, glue_fn, make_visit_glue, ~"visit");
            debug!("--- lazily_emit_tydesc_glue VISIT %s",
                   ppaux::ty_to_str(ccx.tcx, ti.ty));
          }
        }
    }
}

// See [Note-arg-mode]
fn call_tydesc_glue_full(++bcx: block, v: ValueRef, tydesc: ValueRef,
                         field: uint, static_ti: Option<@tydesc_info>) {
    let _icx = bcx.insn_ctxt("call_tydesc_glue_full");
        if bcx.unreachable { return; }
    let ccx = bcx.ccx();

    let static_glue_fn = match static_ti {
      None => None,
      Some(sti) => {
        lazily_emit_tydesc_glue(ccx, field, sti);
        if field == abi::tydesc_field_take_glue {
            sti.take_glue
        } else if field == abi::tydesc_field_drop_glue {
            sti.drop_glue
        } else if field == abi::tydesc_field_free_glue {
            sti.free_glue
        } else if field == abi::tydesc_field_visit_glue {
            sti.visit_glue
        } else {
            None
        }
      }
    };

    // When available, use static type info to give glue the right type.
    let static_glue_fn = match static_ti {
      None => None,
      Some(sti) => {
        match static_glue_fn {
          None => None,
          Some(sgf) => Some(
              PointerCast(bcx, sgf, T_ptr(type_of_glue_fn(ccx, sti.ty))))
        }
      }
    };

    // When static type info is available, avoid casting parameter because the
    // function already has the right type. Otherwise cast to generic pointer.
    let llrawptr = if is_none(static_ti) || is_none(static_glue_fn) {
        PointerCast(bcx, v, T_ptr(T_i8()))
    } else {
        v
    };

    let llfn = {
        match static_glue_fn {
          None => {
            // Select out the glue function to call from the tydesc
            let llfnptr = GEPi(bcx, tydesc, ~[0u, field]);
            Load(bcx, llfnptr)
          }
          Some(sgf) => sgf
        }
    };

    Call(bcx, llfn, ~[C_null(T_ptr(T_nil())), C_null(T_ptr(T_nil())),
                      C_null(T_ptr(T_ptr(bcx.ccx().tydesc_type))), llrawptr]);
}

// See [Note-arg-mode]
fn call_tydesc_glue(++cx: block, v: ValueRef, t: ty::t, field: uint)
    -> block {
    let _icx = cx.insn_ctxt("call_tydesc_glue");
    let ti = get_tydesc(cx.ccx(), t);
    call_tydesc_glue_full(cx, v, ti.tydesc, field, Some(ti));
    return cx;
}

fn call_cmp_glue(bcx: block, lhs: ValueRef, rhs: ValueRef, t: ty::t,
                 llop: ValueRef) -> ValueRef {
    // We can't use call_tydesc_glue_full() and friends here because compare
    // glue has a special signature.
    let _icx = bcx.insn_ctxt("call_cmp_glue");

    let lllhs = spill_if_immediate(bcx, lhs, t);
    let llrhs = spill_if_immediate(bcx, rhs, t);

    let llrawlhsptr = BitCast(bcx, lllhs, T_ptr(T_i8()));
    let llrawrhsptr = BitCast(bcx, llrhs, T_ptr(T_i8()));
    let lltydesc = get_tydesc_simple(bcx.ccx(), t);

    let llfn = bcx.ccx().upcalls.cmp_type;

    let llcmpresultptr = alloca(bcx, T_i1());
    Call(bcx, llfn, ~[llcmpresultptr, lltydesc,
                      llrawlhsptr, llrawrhsptr, llop]);
    return Load(bcx, llcmpresultptr);
}

fn take_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = cx.insn_ctxt("take_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_take_glue);
    }
    return cx;
}

fn drop_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = cx.insn_ctxt("drop_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_drop_glue);
    }
    return cx;
}

fn drop_ty_root(bcx: block, v: ValueRef, rooted: bool, t: ty::t) -> block {
    if rooted {
        // NB: v is a raw ptr to an addrspace'd ptr to the value.
        let v = PointerCast(bcx, Load(bcx, v), T_ptr(type_of(bcx.ccx(), t)));
        drop_ty(bcx, v, t)
    } else {
        drop_ty(bcx, v, t)
    }
}

fn drop_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("drop_ty_immediate");
    match ty::get(t).struct {
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) => {
        free_ty_immediate(bcx, v, t)
      }
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) => {
        decr_refcnt_maybe_free(bcx, v, t)
      }
      _ => bcx.tcx().sess.bug(~"drop_ty_immediate: non-box ty")
    }
}

fn take_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> result {
    let _icx = bcx.insn_ctxt("take_ty_immediate");
    match ty::get(t).struct {
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) => {
        incr_refcnt_of_boxed(bcx, v);
        rslt(bcx, v)
      }
      ty::ty_uniq(_) => {
        uniq::duplicate(bcx, v, t)
      }
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) => {
        tvec::duplicate_uniq(bcx, v, t)
      }
      _ => rslt(bcx, v)
    }
}

fn free_ty(cx: block, v: ValueRef, t: ty::t) -> block {
    // NB: v is an *alias* of type t here, not a direct value.
    let _icx = cx.insn_ctxt("free_ty");
    if ty::type_needs_drop(cx.tcx(), t) {
        return call_tydesc_glue(cx, v, t, abi::tydesc_field_free_glue);
    }
    return cx;
}

fn free_ty_immediate(bcx: block, v: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("free_ty_immediate");
    match ty::get(t).struct {
      ty::ty_uniq(_) |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) |
      ty::ty_box(_) | ty::ty_opaque_box |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) |
      ty::ty_opaque_closure_ptr(_) => {
        let vp = alloca_zeroed(bcx, type_of(bcx.ccx(), t));
        Store(bcx, v, vp);
        free_ty(bcx, vp, t)
      }
      _ => bcx.tcx().sess.bug(~"free_ty_immediate: non-box ty")
    }
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
        let llsz = llsize_of(ccx, type_of(ccx, t));
        call_memmove(bcx, dst, src, llsz);
    } else {
        Store(bcx, Load(bcx, src), dst);
    }
}

enum copy_action { INIT, DROP_EXISTING, }

// These are the types that are passed by pointer.
fn type_is_structural_or_param(t: ty::t) -> bool {
    if ty::type_is_structural(t) { return true; }
    match ty::get(t).struct {
      ty::ty_param(*) => return true,
      _ => return false
    }
}

fn copy_val(cx: block, action: copy_action, dst: ValueRef,
            src: ValueRef, t: ty::t) -> block {
    let _icx = cx.insn_ctxt("copy_val");
    if action == DROP_EXISTING &&
        (type_is_structural_or_param(t) ||
         ty::type_is_unique(t)) {
        let dstcmp = load_if_immediate(cx, dst, t);
        let cast = PointerCast(cx, dstcmp, val_ty(src));
        // Self-copy check
        do with_cond(cx, ICmp(cx, lib::llvm::IntNE, cast, src)) |bcx| {
            copy_val_no_check(bcx, action, dst, src, t)
        }
    } else {
        copy_val_no_check(cx, action, dst, src, t)
    }
}

fn copy_val_no_check(bcx: block, action: copy_action, dst: ValueRef,
                     src: ValueRef, t: ty::t) -> block {
    let _icx = bcx.insn_ctxt("copy_val_no_check");
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    if ty::type_is_scalar(t) || ty::type_is_region_ptr(t) {
        Store(bcx, src, dst);
        return bcx;
    }
    if ty::type_is_nil(t) || ty::type_is_bot(t) { return bcx; }
    if ty::type_is_boxed(t) || ty::type_is_unique(t) {
        if action == DROP_EXISTING { bcx = drop_ty(bcx, dst, t); }
        Store(bcx, src, dst);
        return take_ty(bcx, dst, t);
    }
    if type_is_structural_or_param(t) {
        if action == DROP_EXISTING { bcx = drop_ty(bcx, dst, t); }
        memmove_ty(bcx, dst, src, t);
        return take_ty(bcx, dst, t);
    }
    ccx.sess.bug(~"unexpected type in trans::copy_val_no_check: " +
                     ppaux::ty_to_str(ccx.tcx, t));
}


// This works like copy_val, except that it deinitializes the source.
// Since it needs to zero out the source, src also needs to be an lval.
// FIXME (#839): We always zero out the source. Ideally we would detect the
// case where a variable is always deinitialized by block exit and thus
// doesn't need to be dropped.
fn move_val(cx: block, action: copy_action, dst: ValueRef,
            src: lval_result, t: ty::t) -> block {

    let _icx = cx.insn_ctxt("move_val");
    let mut src_val = src.val;
    let tcx = cx.tcx();
    let mut cx = cx;
    if ty::type_is_scalar(t) || ty::type_is_region_ptr(t) {
        if src.kind == lv_owned { src_val = Load(cx, src_val); }
        Store(cx, src_val, dst);
        return cx;
    } else if ty::type_is_nil(t) || ty::type_is_bot(t) {
        return cx;
    } else if ty::type_is_boxed(t) || ty::type_is_unique(t) {
        if src.kind == lv_owned { src_val = Load(cx, src_val); }
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t); }
        Store(cx, src_val, dst);
        if src.kind == lv_owned { return zero_mem(cx, src.val, t); }
        // If we're here, it must be a temporary.
        revoke_clean(cx, src_val);
        return cx;
    } else if type_is_structural_or_param(t) {
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t); }
        memmove_ty(cx, dst, src_val, t);
        if src.kind == lv_owned { return zero_mem(cx, src_val, t); }
        // If we're here, it must be a temporary.
        revoke_clean(cx, src_val);
        return cx;
    }
    cx.sess().bug(~"unexpected type in trans::move_val: " +
                  ppaux::ty_to_str(tcx, t));
}

fn store_temp_expr(cx: block, action: copy_action, dst: ValueRef,
                   src: lval_result, t: ty::t, last_use: bool)
    -> block {
    let _icx = cx.insn_ctxt("trans_temp_expr");
    // Lvals in memory are not temporaries. Copy them.
    if src.kind != lv_temporary && !last_use {
        let v = if src.kind == lv_owned {
                    load_if_immediate(cx, src.val, t)
                } else {
                    src.val
                };
        return copy_val(cx, action, dst, v, t);
    }
    return move_val(cx, action, dst, src, t);
}

fn trans_lit(cx: block, e: @ast::expr, lit: ast::lit, dest: dest) -> block {
    let _icx = cx.insn_ctxt("trans_lit");
    if dest == ignore { return cx; }
    match lit.node {
        ast::lit_str(s) => tvec::trans_estr(cx, s, None, dest),
        _ => store_in_dest(cx, consts::const_lit(cx.ccx(), e, lit), dest)
    }
}

fn trans_boxed_expr(bcx: block, contents: @ast::expr,
                    t: ty::t, heap: heap,
                    dest: dest) -> block {
    let _icx = bcx.insn_ctxt("trans_boxed_expr");
    let {bcx, box, body} = malloc_general(bcx, t, heap);
    add_clean_free(bcx, box, heap);
    let bcx = trans_expr_save_in(bcx, contents, body);
    revoke_clean(bcx, box);
    return store_in_dest(bcx, box, dest);
}

fn trans_unary(bcx: block, op: ast::unop, e: @ast::expr,
               un_expr: @ast::expr, dest: dest) -> block {
    let _icx = bcx.insn_ctxt("trans_unary");
    // Check for user-defined method call
    match bcx.ccx().maps.method_map.find(un_expr.id) {
      Some(mentry) => {
        let fty = node_id_type(bcx, un_expr.callee_id);
        return trans_call_inner(
            bcx, un_expr.info(), fty,
            expr_ty(bcx, un_expr),
            |bcx| impl::trans_method_callee(bcx, un_expr.callee_id, e,
                                            mentry),
            arg_exprs(~[]), dest);
      }
      _ => ()
    }

    if dest == ignore { return trans_expr(bcx, e, ignore); }
    let e_ty = expr_ty(bcx, e);
    match op {
      ast::not => {
        let {bcx, val} = trans_temp_expr(bcx, e);
        store_in_dest(bcx, Not(bcx, val), dest)
      }
      ast::neg => {
        let {bcx, val} = trans_temp_expr(bcx, e);
        let llneg = if ty::type_is_fp(e_ty) {
            FNeg(bcx, val)
        } else { Neg(bcx, val) };
        store_in_dest(bcx, llneg, dest)
      }
      ast::box(_) => {
        trans_boxed_expr(bcx, e, e_ty, heap_shared, dest)
      }
      ast::uniq(_) => {
        trans_boxed_expr(bcx, e, e_ty, heap_exchange, dest)
      }
      ast::deref => {
        bcx.sess().bug(~"deref expressions should have been \
                               translated using trans_lval(), not \
                               trans_unary()")
      }
    }
}

fn trans_addr_of(cx: block, e: @ast::expr, dest: dest) -> block {
    let _icx = cx.insn_ctxt("trans_addr_of");
    let mut {bcx, val, kind} = trans_temp_lval(cx, e);
    let ety = expr_ty(cx, e);
    let is_immediate = ty::type_is_immediate(ety);
    if (kind == lv_temporary && is_immediate) || kind == lv_owned_imm {
        val = do_spill(bcx, val, ety);
    }
    return store_in_dest(bcx, val, dest);
}

fn trans_compare(cx: block, op: ast::binop, lhs: ValueRef,
                 _lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> result {
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

    let cmpval = call_cmp_glue(cx, lhs, rhs, rhs_t, llop);

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
        trans_fail(bcx, Some(span), text)
    }
}

// Important to get types for both lhs and rhs, because one might be _|_
// and the other not.
fn trans_eager_binop(cx: block, span: span, op: ast::binop, lhs: ValueRef,
                     lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t, dest: dest)
    -> block {
    let mut cx = cx;
    let _icx = cx.insn_ctxt("trans_eager_binop");
    if dest == ignore { return cx; }
    let intype = {
        if ty::type_is_bot(lhs_t) { rhs_t }
        else { lhs_t }
    };
    let is_float = ty::type_is_fp(intype);

    let rhs = cast_shift_expr_rhs(cx, op, lhs, rhs);

    let mut cx = cx;
    let val = match op {
      ast::add => {
        if is_float { FAdd(cx, lhs, rhs) }
        else { Add(cx, lhs, rhs) }
      }
      ast::subtract => {
        if is_float { FSub(cx, lhs, rhs) }
        else { Sub(cx, lhs, rhs) }
      }
      ast::mul => {
        if is_float { FMul(cx, lhs, rhs) }
        else { Mul(cx, lhs, rhs) }
      }
      ast::div => {
        if is_float {
            FDiv(cx, lhs, rhs)
        } else {
            // Only zero-check integers; fp /0 is NaN
            cx = fail_if_zero(cx, span, op, rhs, rhs_t);
            if ty::type_is_signed(intype) {
                SDiv(cx, lhs, rhs)
            } else {
                UDiv(cx, lhs, rhs)
            }
        }
      }
      ast::rem => {
        if is_float {
            FRem(cx, lhs, rhs)
        } else {
            // Only zero-check integers; fp %0 is NaN
            cx = fail_if_zero(cx, span, op, rhs, rhs_t);
            if ty::type_is_signed(intype) {
                SRem(cx, lhs, rhs)
            } else {
                URem(cx, lhs, rhs)
            }
        }
      }
      ast::bitor => Or(cx, lhs, rhs),
      ast::bitand => And(cx, lhs, rhs),
      ast::bitxor => Xor(cx, lhs, rhs),
      ast::shl => Shl(cx, lhs, rhs),
      ast::shr => {
        if ty::type_is_signed(intype) {
            AShr(cx, lhs, rhs)
        } else { LShr(cx, lhs, rhs) }
      }
      _ => {
        let cmpr = trans_compare(cx, op, lhs, lhs_t, rhs, rhs_t);
        cx = cmpr.bcx;
        cmpr.val
      }
    };
    return store_in_dest(cx, val, dest);
}

fn trans_assign_op(bcx: block, ex: @ast::expr, op: ast::binop,
                   dst: @ast::expr, src: @ast::expr) -> block {
    debug!("%s", expr_to_str(ex, bcx.tcx().sess.parse_sess.interner));
    let _icx = bcx.insn_ctxt("trans_assign_op");
    let t = expr_ty(bcx, src);
    let lhs_res = trans_lval(bcx, dst);
    assert (lhs_res.kind == lv_owned);

    // A user-defined operator method
    match bcx.ccx().maps.method_map.find(ex.id) {
      Some(origin) => {
        let bcx = lhs_res.bcx;
        debug!("user-defined method callee_id: %s",
               ast_map::node_id_to_str(bcx.tcx().items, ex.callee_id,
                                       bcx.sess().parse_sess.interner));
        let fty = node_id_type(bcx, ex.callee_id);

        let dty = expr_ty(bcx, dst);
        let target = alloc_ty(bcx, dty);

        let bcx = trans_call_inner(
            bcx, ex.info(), fty,
            expr_ty(bcx, ex),
            |bcx| {
                // FIXME (#2528): provide the already-computed address, not
                // the expr.
                impl::trans_method_callee(bcx, ex.callee_id, dst, origin)
            },
            arg_exprs(~[src]), save_in(target));

        return move_val(bcx, DROP_EXISTING, lhs_res.val,
                     {bcx: bcx, val: target, kind: lv_owned},
                     dty);
      }
      _ => ()
    }

    let {bcx, val: rhs_val} = trans_temp_expr(lhs_res.bcx, src);
    return trans_eager_binop(bcx, ex.span,
                          op, Load(bcx, lhs_res.val), t, rhs_val, t,
                          save_in(lhs_res.val));
}

fn root_value(bcx: block, val: ValueRef, ty: ty::t,
              scope_id: ast::node_id) {
    let _icx = bcx.insn_ctxt("root_value");

    if bcx.sess().trace() {
        trans_trace(
            bcx, None,
            fmt!("preserving until end of scope %d", scope_id));
    }

    let root_loc = alloca_zeroed(bcx, type_of(bcx.ccx(), ty));
    copy_val(bcx, INIT, root_loc, val, ty);
    add_root_cleanup(bcx, scope_id, root_loc, ty);
}

// autoderefs the value `v`, either as many times as we can (if `max ==
// uint::max_value`) or `max` times.
fn autoderef(cx: block, e_id: ast::node_id,
             v: ValueRef, t: ty::t,
             max: uint) -> result_t {
    let _icx = cx.insn_ctxt("autoderef");
    let mut v1: ValueRef = v;
    let mut t1: ty::t = t;
    let ccx = cx.ccx();
    let mut derefs = 0u;
    while derefs < max {
        debug!("autoderef(e_id=%d, v1=%s, t1=%s, derefs=%u)",
               e_id, val_str(ccx.tn, v1), ppaux::ty_to_str(ccx.tcx, t1),
               derefs);

        // root the autoderef'd value, if necessary:
        derefs += 1u;
        match ccx.maps.root_map.find({id:e_id, derefs:derefs}) {
          None => (),
          Some(scope_id) => {
            root_value(cx, v1, t1, scope_id);
          }
        }

        match ty::get(t1).struct {
          ty::ty_box(mt) => {
            let body = GEPi(cx, v1, ~[0u, abi::box_field_body]);
            t1 = mt.ty;

            // Since we're changing levels of box indirection, we may have
            // to cast this pointer, since statically-sized enum types have
            // different types depending on whether they're behind a box
            // or not.
            let llty = type_of(ccx, t1);
            v1 = PointerCast(cx, body, T_ptr(llty));
          }
          ty::ty_uniq(_) => {
            let derefed = uniq::autoderef(cx, v1, t1);
            t1 = derefed.t;
            v1 = derefed.v;
          }
          ty::ty_rptr(_, mt) => {
            t1 = mt.ty;
            v1 = v;
          }
          ty::ty_enum(did, ref substs) => {
            let variants = ty::enum_variants(ccx.tcx, did);
            if (*variants).len() != 1u || variants[0].args.len() != 1u {
                break;
            }
            t1 = ty::subst(ccx.tcx, substs, variants[0].args[0]);
            v1 = PointerCast(cx, v1, T_ptr(type_of(ccx, t1)));
          }
          _ => break
        }
        v1 = load_if_immediate(cx, v1, t1);
    }

    // either we were asked to deref a specific number of times, in which case
    // we should have, or we asked to deref as many times as we can
    assert derefs == max || max == uint::max_value;

    return {bcx: cx, val: v1, ty: t1};
}

// refinement types would obviate the need for this
enum lazy_binop_ty { lazy_and, lazy_or }

fn trans_lazy_binop(bcx: block, op: lazy_binop_ty, a: @ast::expr,
                    b: @ast::expr, dest: dest) -> block {
    let _icx = bcx.insn_ctxt("trans_lazy_binop");
    let {bcx: past_lhs, val: lhs} = {
        do with_scope_result(bcx, a.info(), ~"lhs") |bcx| {
            trans_temp_expr(bcx, a)
        }
    };
    if past_lhs.unreachable { return past_lhs; }
    let join = sub_block(bcx, ~"join"), before_rhs = sub_block(bcx, ~"rhs");

    match op {
      lazy_and => CondBr(past_lhs, lhs, before_rhs.llbb, join.llbb),
      lazy_or => CondBr(past_lhs, lhs, join.llbb, before_rhs.llbb)
    }
    let {bcx: past_rhs, val: rhs} = {
        do with_scope_result(before_rhs, b.info(), ~"rhs") |bcx| {
            trans_temp_expr(bcx, b)
        }
    };

    if past_rhs.unreachable { return store_in_dest(join, lhs, dest); }
    Br(past_rhs, join.llbb);
    let phi =
        Phi(join, T_bool(), ~[lhs, rhs], ~[past_lhs.llbb, past_rhs.llbb]);
    return store_in_dest(join, phi, dest);
}

fn trans_binary(bcx: block, op: ast::binop, lhs: @ast::expr,
                rhs: @ast::expr, dest: dest, ex: @ast::expr) -> block {
    let _icx = bcx.insn_ctxt("trans_binary");
    // User-defined operators
    match bcx.ccx().maps.method_map.find(ex.id) {
      Some(origin) => {
        let fty = node_id_type(bcx, ex.callee_id);
        return trans_call_inner(
            bcx, ex.info(), fty,
            expr_ty(bcx, ex),
            |bcx| {
                impl::trans_method_callee(bcx, ex.callee_id, lhs, origin)
            },
            arg_exprs(~[rhs]), dest);
      }
      _ => ()
    }

    // First couple cases are lazy:
    match op {
      ast::and => {
        return trans_lazy_binop(bcx, lazy_and, lhs, rhs, dest);
      }
      ast::or => {
        return trans_lazy_binop(bcx, lazy_or, lhs, rhs, dest);
      }
      _ => {
        // Remaining cases are eager:
        let lhs_res = trans_temp_expr(bcx, lhs);
        let rhs_res = trans_temp_expr(lhs_res.bcx, rhs);
        return trans_eager_binop(rhs_res.bcx, ex.span,
                              op, lhs_res.val,
                              expr_ty(bcx, lhs), rhs_res.val,
                              expr_ty(bcx, rhs), dest);
      }
    }
}

fn trans_if(cx: block, cond: @ast::expr, thn: ast::blk,
            els: Option<@ast::expr>, dest: dest)
    -> block {
    let _icx = cx.insn_ctxt("trans_if");
    let {bcx, val: cond_val} = trans_temp_expr(cx, cond);

    let then_dest = dup_for_join(dest);
    let else_dest = dup_for_join(dest);
    let then_cx = scope_block(bcx, thn.info(), ~"then");
    let else_cx = scope_block(bcx, els.info(), ~"else");
    CondBr(bcx, cond_val, then_cx.llbb, else_cx.llbb);
    let then_bcx = trans_block(then_cx, thn, then_dest);
    let then_bcx = trans_block_cleanups(then_bcx, block_cleanups(then_cx));
    // Calling trans_block directly instead of trans_expr
    // because trans_expr will create another scope block
    // context for the block, but we've already got the
    // 'else' context
    let else_bcx = match els {
      Some(elexpr) => {
        match elexpr.node {
          ast::expr_if(_, _, _) => {
            let elseif_blk = ast_util::block_from_expr(elexpr);
            trans_block(else_cx, elseif_blk, else_dest)
          }
          ast::expr_block(blk) => {
            trans_block(else_cx, blk, else_dest)
          }
          // would be nice to have a constraint on ifs
          _ => cx.tcx().sess.bug(~"strange alternative in if")
        }
      }
      _ => else_cx
    };
    let else_bcx = trans_block_cleanups(else_bcx, block_cleanups(else_cx));
    return join_returns(cx,
                     ~[then_bcx, else_bcx], ~[then_dest, else_dest], dest);
}

fn trans_while(cx: block, cond: @ast::expr, body: ast::blk)
    -> block {
    let _icx = cx.insn_ctxt("trans_while");
    let next_cx = sub_block(cx, ~"while next");
    let loop_cx = loop_scope_block(cx, next_cx, ~"`while`", body.info());
    let cond_cx = scope_block(loop_cx, cond.info(), ~"while loop cond");
    let body_cx = scope_block(loop_cx, body.info(), ~"while loop body");
    Br(cx, loop_cx.llbb);
    Br(loop_cx, cond_cx.llbb);
    let cond_res = trans_temp_expr(cond_cx, cond);
    let cond_bcx = trans_block_cleanups(cond_res.bcx,
                                        block_cleanups(cond_cx));
    CondBr(cond_bcx, cond_res.val, body_cx.llbb, next_cx.llbb);
    let body_end = trans_block(body_cx, body, ignore);
    cleanup_and_Br(body_end, body_cx, cond_cx.llbb);
    return next_cx;
}

fn trans_loop(cx:block, body: ast::blk) -> block {
    let _icx = cx.insn_ctxt("trans_loop");
    let next_cx = sub_block(cx, ~"next");
    let body_cx = loop_scope_block(cx, next_cx, ~"`loop`", body.info());
    let body_end = trans_block(body_cx, body, ignore);
    cleanup_and_Br(body_end, body_cx, body_cx.llbb);
    Br(cx, body_cx.llbb);
    return next_cx;
}

enum lval_kind {
    lv_temporary, //< Temporary value passed by value if of immediate type
    lv_owned,     //< Non-temporary value passed by pointer
    lv_owned_imm, //< Non-temporary value passed by value
}
type local_var_result = {val: ValueRef, kind: lval_kind};
type lval_result = {bcx: block, val: ValueRef, kind: lval_kind};
enum callee_env {
    null_env,
    is_closure,
    self_env(ValueRef, ty::t, Option<ValueRef>, ast::rmode),
}
type lval_maybe_callee = {bcx: block,
                          val: ValueRef,
                          kind: lval_kind,
                          env: callee_env};

fn null_env_ptr(bcx: block) -> ValueRef {
    C_null(T_opaque_box_ptr(bcx.ccx()))
}

fn lval_from_local_var(bcx: block, r: local_var_result) -> lval_result {
    return { bcx: bcx, val: r.val, kind: r.kind };
}

fn lval_owned(bcx: block, val: ValueRef) -> lval_result {
    return {bcx: bcx, val: val, kind: lv_owned};
}
fn lval_temp(bcx: block, val: ValueRef) -> lval_result {
    return {bcx: bcx, val: val, kind: lv_temporary};
}

fn lval_no_env(bcx: block, val: ValueRef, kind: lval_kind)
    -> lval_maybe_callee {
    return {bcx: bcx, val: val, kind: kind, env: is_closure};
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

fn normalize_for_monomorphization(tcx: ty::ctxt, ty: ty::t) -> Option<ty::t> {
    // FIXME[mono] could do this recursively. is that worthwhile? (#2529)
    match ty::get(ty).struct {
      ty::ty_box(*) => {
        Some(ty::mk_opaque_box(tcx))
      }
      ty::ty_fn(ref fty) => {
        Some(ty::mk_fn(tcx, {purity: ast::impure_fn,
                             proto: fty.proto,
                             bounds: @~[],
                             inputs: ~[],
                             output: ty::mk_nil(tcx),
                             ret_style: ast::return_val}))
      }
      ty::ty_trait(_, _, _) => {
        Some(ty::mk_fn(tcx, {purity: ast::impure_fn,
                             proto: ty::proto_vstore(ty::vstore_box),
                             bounds: @~[],
                             inputs: ~[],
                             output: ty::mk_nil(tcx),
                             ret_style: ast::return_val}))
      }
      ty::ty_ptr(_) => Some(ty::mk_uint(tcx)),
      _ => None
    }
}

fn make_mono_id(ccx: @crate_ctxt, item: ast::def_id, substs: ~[ty::t],
                vtables: Option<typeck::vtable_res>,
                param_uses: Option<~[type_use::type_uses]>) -> mono_id {
    let precise_param_ids = match vtables {
      Some(vts) => {
        let bounds = ty::lookup_item_type(ccx.tcx, item).bounds;
        let mut i = 0u;
        vec::map2(*bounds, substs, |bounds, subst| {
            let mut v = ~[];
            for vec::each(*bounds) |bound| {
                match bound {
                  ty::bound_trait(_) => {
                    vec::push(v, impl::vtable_id(ccx, vts[i]));
                    i += 1u;
                  }
                  _ => ()
                }
            }
            (subst, if v.len() > 0u { Some(v) } else { None })
        })
      }
      None => {
        vec::map(substs, |subst| (subst, None))
      }
    };
    let param_ids = match param_uses {
      Some(uses) => {
        vec::map2(precise_param_ids, uses, |id, uses| {
            match id {
                (a, b@Some(_)) => mono_precise(a, b),
              (subst, None) => {
                if uses == 0u { mono_any }
                else if uses == type_use::use_repr &&
                        !ty::type_needs_drop(ccx.tcx, subst) {
                    let llty = type_of(ccx, subst);
                    let size = shape::llsize_of_real(ccx, llty);
                    let align = shape::llalign_of_pref(ccx, llty);
                    // Special value for nil to prevent problems with undef
                    // return pointers.
                    if size == 1u && ty::type_is_nil(subst) {
                        mono_repr(0u, 0u)
                    } else { mono_repr(size, align) }
                } else { mono_precise(subst, None) }
              }
            }
        })
      }
      None => precise_param_ids.map(|x| { let (a, b) = x;
                mono_precise(a, b) })
    };
    @{def: item, params: param_ids}
}

fn monomorphic_fn(ccx: @crate_ctxt, fn_id: ast::def_id,
                  real_substs: ~[ty::t],
                  vtables: Option<typeck::vtable_res>,
                  ref_id: Option<ast::node_id>)
    -> {val: ValueRef, must_cast: bool} {
    let _icx = ccx.insn_ctxt("monomorphic_fn");
    let mut must_cast = false;
    let substs = vec::map(real_substs, |t| {
        match normalize_for_monomorphization(ccx.tcx, t) {
          Some(t) => { must_cast = true; t }
          None => t
        }
    });

    for real_substs.each() |s| { assert !ty::type_has_params(s); }
    for substs.each() |s| { assert !ty::type_has_params(s); }
    let param_uses = type_use::type_uses_for(ccx, fn_id, substs.len());
    let hash_id = make_mono_id(ccx, fn_id, substs, vtables, Some(param_uses));
    if vec::any(hash_id.params,
                |p| match p { mono_precise(_, _) => false, _ => true }) {
        must_cast = true;
    }

    #debug["monomorphic_fn(fn_id=%? (%s), real_substs=%?, substs=%?, \
           hash_id = %?",
           fn_id, ty::item_path_str(ccx.tcx, fn_id),
           real_substs.map(|s| ty_to_str(ccx.tcx, s)),
           substs.map(|s| ty_to_str(ccx.tcx, s)), hash_id];

    match ccx.monomorphized.find(hash_id) {
      Some(val) => {
        debug!("leaving monomorphic fn %s",
               ty::item_path_str(ccx.tcx, fn_id));
        return {val: val, must_cast: must_cast};
      }
      None => ()
    }

    let tpt = ty::lookup_item_type(ccx.tcx, fn_id);
    let mut llitem_ty = tpt.ty;

    let map_node = session::expect(ccx.sess, ccx.tcx.items.find(fn_id.node),
     || fmt!("While monomorphizing %?, couldn't find it in the item map \
        (may have attempted to monomorphize an item defined in a different \
        crate?)", fn_id));
    // Get the path so that we can create a symbol
    let (pt, name, span) = match map_node {
      ast_map::node_item(i, pt) => (pt, i.ident, i.span),
      ast_map::node_variant(v, enm, pt) => (pt, v.node.name, enm.span),
      ast_map::node_method(m, _, pt) => (pt, m.ident, m.span),
      ast_map::node_foreign_item(i, ast::foreign_abi_rust_intrinsic, pt)
      => (pt, i.ident, i.span),
      ast_map::node_foreign_item(*) => {
        // Foreign externs don't have to be monomorphized.
        return {val: get_item_val(ccx, fn_id.node),
                must_cast: true};
      }
      ast_map::node_ctor(nm, _, ct, _, pt) => (pt, nm, ct.span),
      ast_map::node_dtor(_, dtor, _, pt) =>
          (pt, special_idents::dtor, dtor.span),
      ast_map::node_trait_method(*) => {
        ccx.tcx.sess.bug(~"Can't monomorphize a trait method")
      }
      ast_map::node_expr(*) => {
        ccx.tcx.sess.bug(~"Can't monomorphize an expr")
      }
      ast_map::node_stmt(*) => {
        ccx.tcx.sess.bug(~"Can't monomorphize a stmt")
      }
      ast_map::node_export(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize an export")
      }
      ast_map::node_arg(*) => ccx.tcx.sess.bug(~"Can't monomorphize an arg"),
      ast_map::node_block(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize a block")
      }
      ast_map::node_local(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize a local")
      }
    };
    let mono_ty = ty::subst_tps(ccx.tcx, substs, llitem_ty);
    let llfty = type_of_fn_from_ty(ccx, mono_ty);

    let depth = option::get_default(ccx.monomorphizing.find(fn_id), 0u);
    // Random cut-off -- code that needs to instantiate the same function
    // recursively more than ten times can probably safely be assumed to be
    // causing an infinite expansion.
    if depth > 10u {
        ccx.sess.span_fatal(
            span, ~"overly deep expansion of inlined function");
    }
    ccx.monomorphizing.insert(fn_id, depth + 1u);

    let pt = vec::append(*pt,
                         ~[path_name(ccx.names(ccx.sess.str_of(name)))]);
    let s = mangle_exported_name(ccx, pt, mono_ty);

    let mk_lldecl = || {
        let lldecl = decl_internal_cdecl_fn(ccx.llmod, s, llfty);
        ccx.monomorphized.insert(hash_id, lldecl);
        lldecl
    };

    let psubsts = Some({tys: substs, vtables: vtables, bounds: tpt.bounds});
    let lldecl = match map_node {
      ast_map::node_item(i@@{node: ast::item_fn(decl, _, _, body), _}, _) => {
        let d = mk_lldecl();
        set_inline_hint_if_appr(i.attrs, d);
        trans_fn(ccx, pt, decl, body, d, no_self, psubsts, fn_id.node);
        d
      }
      ast_map::node_item(*) => {
          ccx.tcx.sess.bug(~"Can't monomorphize this kind of item")
      }
      ast_map::node_foreign_item(i, _, _) => {
          let d = mk_lldecl();
          foreign::trans_intrinsic(ccx, d, i, pt, option::get(psubsts),
                                ref_id);
          d
      }
      ast_map::node_variant(v, enum_item, _) => {
        let tvs = ty::enum_variants(ccx.tcx, local_def(enum_item.id));
        let this_tv = option::get(vec::find(*tvs, |tv| {
            tv.id.node == fn_id.node}));
        let d = mk_lldecl();
        set_inline_hint(d);
        match v.node.kind {
            ast::tuple_variant_kind(args) => {
                trans_enum_variant(ccx, enum_item.id, v, args,
                                   this_tv.disr_val, (*tvs).len() == 1u,
                                   psubsts, d);
            }
            ast::struct_variant_kind(_) =>
                ccx.tcx.sess.bug(~"can't monomorphize struct variants"),
            ast::enum_variant_kind(_) =>
                ccx.tcx.sess.bug(~"can't monomorphize enum variants")
        }
        d
      }
      ast_map::node_method(mth, _, _) => {
        let d = mk_lldecl();
        set_inline_hint_if_appr(mth.attrs, d);
        impl::trans_method(ccx, pt, mth, psubsts, d);
        d
      }
      ast_map::node_ctor(_, tps, ctor, parent_id, _) => {
        // ctors don't have attrs, at least not right now
        let d = mk_lldecl();
        let tp_tys = ty::ty_params_to_tys(ccx.tcx, tps);
        trans_class_ctor(ccx, pt, ctor.node.dec, ctor.node.body, d,
               option::get_default(psubsts,
                        {tys:tp_tys, vtables: None, bounds: @~[]}),
                         fn_id.node, parent_id, ctor.span);
        d
      }
      ast_map::node_dtor(_, dtor, _, pt) => {
        let parent_id = match ty::ty_to_def_id(ty::node_id_to_type(ccx.tcx,
                                              dtor.node.self_id)) {
                Some(did) => did,
                None      => ccx.sess.span_bug(dtor.span, ~"Bad self ty in \
                                                            dtor")
        };
        trans_class_dtor(ccx, *pt, dtor.node.body,
          dtor.node.id, psubsts, Some(hash_id), parent_id)
      }
      // Ugh -- but this ensures any new variants won't be forgotten
      ast_map::node_expr(*) |
      ast_map::node_stmt(*) |
      ast_map::node_trait_method(*) |
      ast_map::node_export(*) |
      ast_map::node_arg(*) |
      ast_map::node_block(*) |
      ast_map::node_local(*) => {
        ccx.tcx.sess.bug(fmt!("Can't monomorphize a %?", map_node))
      }
    };
    ccx.monomorphizing.insert(fn_id, depth);

    debug!("leaving monomorphic fn %s", ty::item_path_str(ccx.tcx, fn_id));
    {val: lldecl, must_cast: must_cast}
}

fn maybe_instantiate_inline(ccx: @crate_ctxt, fn_id: ast::def_id)
    -> ast::def_id {
    let _icx = ccx.insn_ctxt("maybe_instantiate_inline");
    match ccx.external.find(fn_id) {
      Some(Some(node_id)) => {
        // Already inline
        debug!("maybe_instantiate_inline(%s): already inline as node id %d",
               ty::item_path_str(ccx.tcx, fn_id), node_id);
        local_def(node_id)
      }
      Some(None) => fn_id, // Not inlinable
      None => { // Not seen yet
        match csearch::maybe_get_item_ast(
            ccx.tcx, fn_id,
            |a,b,c,d| {
                astencode::decode_inlined_item(a, b, ccx.maps, c, d)
            }) {

          csearch::not_found => {
            ccx.external.insert(fn_id, None);
            fn_id
          }
          csearch::found(ast::ii_item(item)) => {
            ccx.external.insert(fn_id, Some(item.id));
            trans_item(ccx, *item);
            local_def(item.id)
          }
          csearch::found(ast::ii_ctor(ctor, _, tps, _)) => {
            ccx.external.insert(fn_id, Some(ctor.node.id));
            local_def(ctor.node.id)
          }
          csearch::found(ast::ii_foreign(item)) => {
            ccx.external.insert(fn_id, Some(item.id));
            local_def(item.id)
          }
          csearch::found_parent(parent_id, ast::ii_item(item)) => {
            ccx.external.insert(parent_id, Some(item.id));
            let mut my_id = 0;
            match item.node {
              ast::item_enum(_, _) => {
                let vs_here = ty::enum_variants(ccx.tcx, local_def(item.id));
                let vs_there = ty::enum_variants(ccx.tcx, parent_id);
                do vec::iter2(*vs_here, *vs_there) |here, there| {
                    if there.id == fn_id { my_id = here.id.node; }
                    ccx.external.insert(there.id, Some(here.id.node));
                }
              }
              _ => ccx.sess.bug(~"maybe_instantiate_inline: item has a \
                    non-enum parent")
            }
            trans_item(ccx, *item);
            local_def(my_id)
          }
          csearch::found_parent(_, _) => {
              ccx.sess.bug(~"maybe_get_item_ast returned a found_parent \
               with a non-item parent");
          }
          csearch::found(ast::ii_method(impl_did, mth)) => {
            ccx.external.insert(fn_id, Some(mth.id));
            let {bounds: impl_bnds, region_param: _, ty: impl_ty} =
                ty::lookup_item_type(ccx.tcx, impl_did);
            if (*impl_bnds).len() + mth.tps.len() == 0u {
                let llfn = get_item_val(ccx, mth.id);
                let path = vec::append(
                    ty::item_path(ccx.tcx, impl_did),
                    ~[path_name(mth.ident)]);
                trans_fn(ccx, path, mth.decl, mth.body,
                         llfn, impl_self(impl_ty), None, mth.id);
            }
            local_def(mth.id)
          }
          csearch::found(ast::ii_dtor(dtor, _, tps, _)) => {
              ccx.external.insert(fn_id, Some(dtor.node.id));
              local_def(dtor.node.id)
          }
        }
      }
    }
}

fn lval_static_fn(bcx: block, fn_id: ast::def_id, id: ast::node_id)
    -> lval_maybe_callee {
    let _icx = bcx.insn_ctxt("lval_static_fn");
    let vts = option::map(bcx.ccx().maps.vtable_map.find(id), |vts| {
        impl::resolve_vtables_in_fn_ctxt(bcx.fcx, vts)
    });
    lval_static_fn_inner(bcx, fn_id, id, node_id_type_params(bcx, id), vts)
}

fn lval_static_fn_inner(bcx: block, fn_id: ast::def_id, id: ast::node_id,
                        tys: ~[ty::t], vtables: Option<typeck::vtable_res>)
    -> lval_maybe_callee {
    let _icx = bcx.insn_ctxt("lval_static_fn_inner");
    let ccx = bcx.ccx(), tcx = ccx.tcx;
    let tpt = ty::lookup_item_type(tcx, fn_id);

    // Check whether this fn has an inlined copy and, if so, redirect fn_id to
    // the local id of the inlined copy.
    let fn_id = if fn_id.crate != ast::local_crate {
        maybe_instantiate_inline(ccx, fn_id)
    } else { fn_id };

    if fn_id.crate == ast::local_crate && tys.len() > 0u {
        let mut {val, must_cast} =
            monomorphic_fn(ccx, fn_id, tys, vtables, Some(id));
        if must_cast {
            val = PointerCast(bcx, val, T_ptr(type_of_fn_from_ty(
                ccx, node_id_type(bcx, id))));
        }
        return {bcx: bcx, val: val, kind: lv_owned, env: null_env};
    }

    let mut val = if fn_id.crate == ast::local_crate {
        // Internal reference.
        get_item_val(ccx, fn_id.node)
    } else {
        // External reference.
        trans_external_path(ccx, fn_id, tpt.ty)
    };
    if tys.len() > 0u {
        val = PointerCast(bcx, val, T_ptr(type_of_fn_from_ty(
            ccx, node_id_type(bcx, id))));
    }

    match ty::get(tpt.ty).struct {
      ty::ty_fn(fn_ty) => {
        match fn_ty.purity {
          ast::extern_fn => {
            // Extern functions are just opaque pointers
            let val = PointerCast(bcx, val, T_ptr(T_i8()));
            return lval_no_env(bcx, val, lv_owned_imm);
          }
          _ => { /* fall through */ }
        }
      }
      _ => { /* fall through */ }
    }

    return {bcx: bcx, val: val, kind: lv_owned, env: null_env};
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

// This shouldn't exist. We should cast self *once*, but right now this
// conflicts with default methods.
fn cast_self(cx: block, slf: val_self_data) -> ValueRef {
    PointerCast(cx, slf.v, T_ptr(type_of(cx.ccx(), slf.t)))
}

fn trans_local_var(cx: block, def: ast::def) -> local_var_result {
    let _icx = cx.insn_ctxt("trans_local_var");
    fn take_local(table: hashmap<ast::node_id, local_val>,
                  id: ast::node_id) -> local_var_result {
        match table.find(id) {
          Some(local_mem(v)) => {val: v, kind: lv_owned},
          Some(local_imm(v)) => {val: v, kind: lv_owned_imm},
          None => fail(fmt!("take_local: internal error, \
                             found no entry for %?", id))
        }
    }
    match def {
      ast::def_upvar(nid, _, _, _) => {
        assert (cx.fcx.llupvars.contains_key(nid));
        return { val: cx.fcx.llupvars.get(nid), kind: lv_owned };
      }
      ast::def_arg(nid, _) => {
        assert (cx.fcx.llargs.contains_key(nid));
        return take_local(cx.fcx.llargs, nid);
      }
      ast::def_local(nid, _) | ast::def_binding(nid, _) => {
        assert (cx.fcx.lllocals.contains_key(nid));
        return take_local(cx.fcx.lllocals, nid);
      }
      ast::def_self(_) => {
        let slf = match copy cx.fcx.llself {
          Some(s) => cast_self(cx, s),
          None => cx.sess().bug(~"trans_local_var: reference to self \
                                 out of context")
        };
        return {val: slf, kind: lv_owned};
      }
      _ => {
        cx.sess().unimpl(fmt!("unsupported def type in trans_local_var: %?",
                              def));
      }
    }
}

fn trans_path(cx: block, id: ast::node_id)
    -> lval_maybe_callee {
    let _icx = cx.insn_ctxt("trans_path");
    match cx.tcx().def_map.find(id) {
      None => cx.sess().bug(~"trans_path: unbound node ID"),
      Some(df) => {
          return trans_var(cx, df, id);
      }
    }
}

fn trans_var(cx: block, def: ast::def, id: ast::node_id)-> lval_maybe_callee {
    let _icx = cx.insn_ctxt("trans_var");
    let ccx = cx.ccx();
    match def {
      ast::def_fn(did, _) => {
        return lval_static_fn(cx, did, id);
      }
      ast::def_static_method(did, _) => {
        return impl::trans_static_method_callee(cx, did, id);
      }
      ast::def_variant(tid, vid) => {
        if ty::enum_variant_with_id(ccx.tcx, tid, vid).args.len() > 0u {
            // N-ary variant.
            return lval_static_fn(cx, vid, id);
        } else {
            // Nullary variant.
            let enum_ty = node_id_type(cx, id);
            let llenumptr = alloc_ty(cx, enum_ty);
            let lldiscrimptr = GEPi(cx, llenumptr, ~[0u, 0u]);
            let lldiscrim_gv = lookup_discriminant(ccx, vid);
            let lldiscrim = Load(cx, lldiscrim_gv);
            Store(cx, lldiscrim, lldiscrimptr);
            return lval_no_env(cx, llenumptr, lv_temporary);
        }
      }
      ast::def_const(did) => {
        if did.crate == ast::local_crate {
            return lval_no_env(cx, get_item_val(ccx, did.node), lv_owned);
        } else {
            let tp = node_id_type(cx, id);
            let val = trans_external_path(ccx, did, tp);
            return lval_no_env(cx, load_if_immediate(cx, val, tp),
                            lv_owned_imm);
        }
      }
      _ => {
        let loc = trans_local_var(cx, def);
        return lval_no_env(cx, loc.val, loc.kind);
      }
    }
}

fn trans_rec_field(bcx: block, base: @ast::expr,
                   field: ast::ident) -> lval_result {
    let _icx = bcx.insn_ctxt("trans_rec_field");
    let {bcx, val} = trans_temp_expr(bcx, base);
    let {bcx, val, ty} =
        autoderef(bcx, base.id, val, expr_ty(bcx, base),
                  uint::max_value);
    trans_rec_field_inner(bcx, val, ty, field, base.span)
}

fn trans_rec_field_inner(bcx: block, val: ValueRef, ty: ty::t,
                         field: ast::ident, sp: span) -> lval_result {
    let mut llderef = false;
    let fields = match ty::get(ty).struct {
       ty::ty_rec(fs) => fs,
       ty::ty_class(did, ref substs) => {
         if option::is_some(ty::ty_dtor(bcx.tcx(), did)) {
           llderef = true;
         }
         ty::class_items_as_mutable_fields(bcx.tcx(), did, substs)
       }
       // Constraint?
       _ => bcx.tcx().sess.span_bug(sp, ~"trans_rec_field:\
                 base expr has non-record type")
    };
    // seems wrong? Doesn't take into account the field
    // sizes

    let ix = field_idx_strict(bcx.tcx(), sp, field, fields);

    debug!("val = %s ix = %u", bcx.val_str(val), ix);

    /* self is a class with a dtor, which means we
       have to select out the object itself
       (If any other code does the same thing, that's
       a bug */
    let val = if llderef {
        GEPi(bcx, GEPi(bcx, val, ~[0u, 1u]), ~[0u, ix])
    }
    else { GEPi(bcx, val, ~[0u, ix]) };

    return {bcx: bcx, val: val, kind: lv_owned};
}


fn trans_index(cx: block, ex: @ast::expr, base: @ast::expr,
               idx: @ast::expr) -> lval_result {
    let _icx = cx.insn_ctxt("trans_index");
    let base_ty = expr_ty(cx, base);
    let exp = trans_temp_expr(cx, base);
    let lv = autoderef(exp.bcx, base.id, exp.val, base_ty, uint::max_value);
    let ix = trans_temp_expr(lv.bcx, idx);
    let v = lv.val;
    let bcx = ix.bcx;
    let ccx = cx.ccx();

    // Cast to an LLVM integer. Rust is less strict than LLVM in this regard.
    let ix_size = llsize_of_real(cx.ccx(), val_ty(ix.val));
    let int_size = llsize_of_real(cx.ccx(), ccx.int_type);
    let ix_val = if ix_size < int_size {
        if ty::type_is_signed(expr_ty(cx, idx)) {
            SExt(bcx, ix.val, ccx.int_type)
        } else { ZExt(bcx, ix.val, ccx.int_type) }
    } else if ix_size > int_size {
        Trunc(bcx, ix.val, ccx.int_type)
    } else {
        ix.val
    };

    let unit_ty = node_id_type(cx, ex.id);
    let llunitty = type_of(ccx, unit_ty);
    let unit_sz = llsize_of(ccx, llunitty);
    maybe_name_value(cx.ccx(), unit_sz, ~"unit_sz");
    let scaled_ix = Mul(bcx, ix_val, unit_sz);
    maybe_name_value(cx.ccx(), scaled_ix, ~"scaled_ix");

    let mut (base, len) = tvec::get_base_and_len(bcx, v, base_ty);

    if ty::type_is_str(base_ty) {
        len = Sub(bcx, len, C_uint(bcx.ccx(), 1u));
    }

    debug!("trans_index: base %s", val_str(bcx.ccx().tn, base));
    debug!("trans_index: len %s", val_str(bcx.ccx().tn, len));

    let bounds_check = ICmp(bcx, lib::llvm::IntUGE, scaled_ix, len);
    let bcx = do with_cond(bcx, bounds_check) |bcx| {
        // fail: bad bounds check.
        trans_fail(bcx, Some(ex.span), ~"bounds check")
    };
    let elt = InBoundsGEP(bcx, base, ~[ix_val]);
    return lval_owned(bcx, PointerCast(bcx, elt, T_ptr(llunitty)));
}

fn expr_is_borrowed(bcx: block, e: @ast::expr) -> bool {
    bcx.tcx().borrowings.contains_key(e.id)
}

fn expr_is_lval(bcx: block, e: @ast::expr) -> bool {
    let ccx = bcx.ccx();
    ty::expr_is_lval(ccx.maps.method_map, e)
}

fn trans_callee(bcx: block, e: @ast::expr) -> lval_maybe_callee {
    let _icx = bcx.insn_ctxt("trans_callee");
    match e.node {
      ast::expr_path(_) => return trans_path(bcx, e.id),
      ast::expr_field(base, _, _) => {
        // Lval means this is a record field, so not a method
        if !expr_is_lval(bcx, e) {
            match bcx.ccx().maps.method_map.find(e.id) {
              Some(origin) => { // An impl method
                return impl::trans_method_callee(bcx, e.id, base, origin);
              }
              _ => {
                bcx.ccx().sess.span_bug(e.span, ~"trans_callee: weird expr");
              }
            }
        }
      }
      _ => ()
    }
    let lv = trans_temp_lval(bcx, e);
    return lval_no_env(lv.bcx, lv.val, lv.kind);
}

// Use this when you know you are compiling an lval.
// The additional bool returned indicates whether it's mem (that is
// represented as an alloca or heap, hence needs a 'load' to be used as an
// immediate).
fn trans_lval(cx: block, e: @ast::expr) -> lval_result {
    return match cx.ccx().maps.root_map.find({id:e.id, derefs:0u}) {
      // No need to root this lvalue.
      None => unrooted(cx, e),

      // Lvalue must remain rooted until exit of `scope_id`.  See
      // add_root_cleanup() for comments on why this works the way it does.
      Some(scope_id) => {
        let lv = unrooted(cx, e);

        if !cx.sess().no_asm_comments() {
            add_comment(cx, fmt!("preserving until end of scope %d",
                                 scope_id));
        }

        let _icx = lv.bcx.insn_ctxt("root_value_lval");
        let ty = expr_ty(lv.bcx, e);
        let root_loc = alloca_zeroed(lv.bcx, type_of(cx.ccx(), ty));
        let bcx = store_temp_expr(lv.bcx, INIT, root_loc, lv, ty, false);
        add_root_cleanup(bcx, scope_id, root_loc, ty);
        {bcx: bcx with lv}
      }
    };

    fn unrooted(cx: block, e: @ast::expr) -> lval_result {
        let _icx = cx.insn_ctxt("trans_lval");
        match e.node {
          ast::expr_path(_) => {
            let v = trans_path(cx, e.id);
            return lval_maybe_callee_to_lval(v, e.span);
          }
          ast::expr_field(base, ident, _) => {
            return trans_rec_field(cx, base, ident);
          }
          ast::expr_index(base, idx) => {
            return trans_index(cx, e, base, idx);
          }
          ast::expr_unary(ast::deref, base) => {
            let ccx = cx.ccx();
            let sub = trans_temp_expr(cx, base);
            let t = expr_ty(cx, base);
            let val = match ty::get(t).struct {
              ty::ty_box(_) => {
                let non_gc_val = non_gc_box_cast(sub.bcx, sub.val);
                GEPi(sub.bcx, non_gc_val, ~[0u, abi::box_field_body])
              }
              ty::ty_uniq(_) => {
                let non_gc_val = non_gc_box_cast(sub.bcx, sub.val);
                GEPi(sub.bcx, non_gc_val, ~[0u, abi::box_field_body])
              }
              ty::ty_enum(_, _) => {
                let ety = expr_ty(cx, e);
                let ellty = T_ptr(type_of(ccx, ety));
                PointerCast(sub.bcx, sub.val, ellty)
              }
              ty::ty_ptr(_) | ty::ty_rptr(_,_) => sub.val,
              _ => cx.sess().impossible_case(e.span, #fmt("unary operand \
                may not have type %s", cx.ty_to_str(t)))
            };
            return lval_owned(sub.bcx, val);
          }
          _ => cx.sess().span_bug(e.span, ~"non-lval in trans_lval")
        }
    }
}

/**
 * Get the type of a box in the default address space.
 *
 * Shared box pointers live in address space 1 so the GC strategy can find
 * them. Before taking a pointer to the inside of a box it should be cast into
 * address space 0. Otherwise the resulting (non-box) pointer will be in the
 * wrong address space and thus be the wrong type.
 */
fn non_gc_box_cast(cx: block, val: ValueRef) -> ValueRef {
    debug!("non_gc_box_cast");
    add_comment(cx, ~"non_gc_box_cast");
    assert(llvm::LLVMGetPointerAddressSpace(val_ty(val)) == gc_box_addrspace);
    let non_gc_t = T_ptr(llvm::LLVMGetElementType(val_ty(val)));
    PointerCast(cx, val, non_gc_t)
}

fn lval_maybe_callee_to_lval(c: lval_maybe_callee, sp: span) -> lval_result {
    match c.env {
      self_env(*) => {
        c.bcx.sess().span_bug(sp, ~"implicitly binding method call");
      }
      is_closure => { {bcx: c.bcx, val: c.val, kind: c.kind} }
      null_env => {
        let llfnty = llvm::LLVMGetElementType(val_ty(c.val));
        let llfn = create_real_fn_pair(c.bcx, llfnty, c.val,
                                       null_env_ptr(c.bcx));
        {bcx: c.bcx, val: llfn, kind: lv_temporary}
      }
    }
}

fn int_cast(bcx: block, lldsttype: TypeRef, llsrctype: TypeRef,
            llsrc: ValueRef, signed: bool) -> ValueRef {
    let _icx = bcx.insn_ctxt("int_cast");
    let srcsz = llvm::LLVMGetIntTypeWidth(llsrctype);
    let dstsz = llvm::LLVMGetIntTypeWidth(lldsttype);
    return if dstsz == srcsz {
        BitCast(bcx, llsrc, lldsttype)
    } else if srcsz > dstsz {
        TruncOrBitCast(bcx, llsrc, lldsttype)
    } else if signed {
        SExtOrBitCast(bcx, llsrc, lldsttype)
    } else { ZExtOrBitCast(bcx, llsrc, lldsttype) };
}

fn float_cast(bcx: block, lldsttype: TypeRef, llsrctype: TypeRef,
              llsrc: ValueRef) -> ValueRef {
    let _icx = bcx.insn_ctxt("float_cast");
    let srcsz = lib::llvm::float_width(llsrctype);
    let dstsz = lib::llvm::float_width(lldsttype);
    return if dstsz > srcsz {
        FPExt(bcx, llsrc, lldsttype)
    } else if srcsz > dstsz {
        FPTrunc(bcx, llsrc, lldsttype)
    } else { llsrc };
}

enum cast_kind { cast_pointer, cast_integral, cast_float,
                cast_enum, cast_other, }
fn cast_type_kind(t: ty::t) -> cast_kind {
    match ty::get(t).struct {
      ty::ty_float(*)   => cast_float,
      ty::ty_ptr(*)     => cast_pointer,
      ty::ty_rptr(*)    => cast_pointer,
      ty::ty_int(*)     => cast_integral,
      ty::ty_uint(*)    => cast_integral,
      ty::ty_bool       => cast_integral,
      ty::ty_enum(*)    => cast_enum,
      _                 => cast_other
    }
}


fn trans_cast(cx: block, e: @ast::expr, id: ast::node_id,
              dest: dest) -> block {
    let _icx = cx.insn_ctxt("trans_cast");
    let ccx = cx.ccx();
    let t_out = node_id_type(cx, id);
    match ty::get(t_out).struct {
      ty::ty_trait(_, _, _) => return impl::trans_cast(cx, e, id, dest),
      _ => ()
    }
    let e_res = trans_temp_expr(cx, e);
    let ll_t_in = val_ty(e_res.val);
    let t_in = expr_ty(cx, e);
    let ll_t_out = type_of(ccx, t_out);

    let k_in = cast_type_kind(t_in);
    let k_out = cast_type_kind(t_out);
    let s_in = k_in == cast_integral && ty::type_is_signed(t_in);

    let newval =
        match {in: k_in, out: k_out} {
          {in: cast_integral, out: cast_integral} => {
            int_cast(e_res.bcx, ll_t_out, ll_t_in, e_res.val, s_in)
          }
          {in: cast_float, out: cast_float} => {
            float_cast(e_res.bcx, ll_t_out, ll_t_in, e_res.val)
          }
          {in: cast_integral, out: cast_float} => {
            if s_in {
                SIToFP(e_res.bcx, e_res.val, ll_t_out)
            } else { UIToFP(e_res.bcx, e_res.val, ll_t_out) }
          }
          {in: cast_float, out: cast_integral} => {
            if ty::type_is_signed(t_out) {
                FPToSI(e_res.bcx, e_res.val, ll_t_out)
            } else { FPToUI(e_res.bcx, e_res.val, ll_t_out) }
          }
          {in: cast_integral, out: cast_pointer} => {
            IntToPtr(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: cast_pointer, out: cast_integral} => {
            PtrToInt(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: cast_pointer, out: cast_pointer} => {
            PointerCast(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: cast_enum, out: cast_integral} |
          {in: cast_enum, out: cast_float} => {
            let cx = e_res.bcx;
            let llenumty = T_opaque_enum_ptr(ccx);
            let av_enum = PointerCast(cx, e_res.val, llenumty);
            let lldiscrim_a_ptr = GEPi(cx, av_enum, ~[0u, 0u]);
            let lldiscrim_a = Load(cx, lldiscrim_a_ptr);
            match k_out {
              cast_integral => int_cast(e_res.bcx, ll_t_out,
                                        val_ty(lldiscrim_a),
                                        lldiscrim_a, true),
              cast_float => SIToFP(e_res.bcx, lldiscrim_a, ll_t_out),
              _ => ccx.sess.bug(~"translating unsupported cast.")
            }
          }
          _ => ccx.sess.bug(~"translating unsupported cast.")
        };
    return store_in_dest(e_res.bcx, newval, dest);
}

fn trans_loop_body(bcx: block, id: ast::node_id,
                   decl: ast::fn_decl, body: ast::blk,
                   proto: ty::fn_proto, cap: ast::capture_clause,
                   ret_flag: Option<ValueRef>,
                   dest: dest) -> block {
    closure::trans_expr_fn(bcx, proto, decl, body, id,
                           cap, Some(ret_flag), dest)
}

// temp_cleanups: cleanups that should run only if failure occurs before the
// call takes place:
fn trans_arg_expr(cx: block, arg: ty::arg, lldestty: TypeRef, e: @ast::expr,
                  &temp_cleanups: ~[ValueRef], ret_flag: Option<ValueRef>,
                  derefs: uint)
    -> result {
    let _icx = cx.insn_ctxt("trans_arg_expr");
    let ccx = cx.ccx();
    debug!("+++ trans_arg_expr on %s", expr_to_str(e, ccx.sess.intr()));
    let e_ty = expr_ty(cx, e);
    let is_bot = ty::type_is_bot(e_ty);

    // translate the arg expr as an lvalue
    let lv = match ret_flag {
      // If there is a ret_flag, this *must* be a loop body
      Some(_) => match e.node {
          ast::expr_loop_body(blk@@{node:
                  ast::expr_fn_block(decl, body, cap),_}) => {
            let scratch = alloc_ty(cx, expr_ty(cx, blk));
            let proto = match ty::get(expr_ty(cx, e)).struct {
                ty::ty_fn({proto, _}) => proto,
                _ => cx.sess().impossible_case(e.span, ~"Loop body has \
                       non-fn ty")
            };
            let bcx = trans_loop_body(cx, blk.id, decl, body, proto,
                                      cap, ret_flag, save_in(scratch));
            {bcx: bcx, val: scratch, kind: lv_temporary}
        }
        _ => cx.sess().impossible_case(e.span, ~"ret_flag with non-loop-\
              body expr")
      },
      None => {
        trans_temp_lval(cx, e)
      }
    };

    // auto-deref value as required (this only applies to method
    // call receivers) of method
    debug!("   pre-deref value: %s", val_str(lv.bcx.ccx().tn, lv.val));
    let {lv, e_ty} = if derefs == 0u {
      {lv: lv, e_ty: e_ty}
    } else {
      let {bcx, val} = lval_result_to_result(lv, e_ty);
      let {bcx, val, ty: e_ty} =
          autoderef(bcx, e.id, val, e_ty, derefs);
      {lv: {bcx: bcx, val: val, kind: lv_temporary},
       e_ty: e_ty}
    };

    // borrow value (convert from @T to &T and so forth)
    debug!("   pre-adaptation value: %s", val_str(lv.bcx.ccx().tn, lv.val));
    let {lv, ty: e_ty} = adapt_borrowed_value(lv, e, e_ty);
    let mut bcx = lv.bcx;
    let mut val = lv.val;
    debug!("   adapted value: %s", val_str(bcx.ccx().tn, val));

    // finally, deal with the various modes
    let arg_mode = ty::resolved_mode(ccx.tcx, arg.mode);
    if is_bot {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty (the callee's expected type).
        val = llvm::LLVMGetUndef(lldestty);
    } else {
        match arg_mode {
          ast::by_ref | ast::by_mutbl_ref => {
            // Ensure that the value is spilled into memory:
            if lv.kind != lv_owned && ty::type_is_immediate(e_ty) {
                val = do_spill_noroot(bcx, val);
            }
          }

          ast::by_val => {
            // Ensure that the value is not spilled into memory:
            if lv.kind == lv_owned || !ty::type_is_immediate(e_ty) {
                val = Load(bcx, val);
            }
          }

          ast::by_copy | ast::by_move => {
            // Ensure that an owned copy of the value is in memory:
            let alloc = alloc_ty(bcx, arg.ty);
            let move_out = arg_mode == ast::by_move ||
                ccx.maps.last_use_map.contains_key(e.id);
            if lv.kind == lv_temporary { revoke_clean(bcx, val); }
            if lv.kind == lv_owned || !ty::type_is_immediate(arg.ty) {
                memmove_ty(bcx, alloc, val, arg.ty);
                if move_out && ty::type_needs_drop(ccx.tcx, arg.ty) {
                    bcx = zero_mem(bcx, val, arg.ty);
                }
            } else { Store(bcx, val, alloc); }
            val = alloc;
            if lv.kind != lv_temporary && !move_out {
                bcx = take_ty(bcx, val, arg.ty);
            }

            // In the event that failure occurs before the call actually
            // happens, have to cleanup this copy:
            add_clean_temp_mem(bcx, val, arg.ty);
            vec::push(temp_cleanups, val);
          }
        }
    }

    if !is_bot && arg.ty != e_ty || ty::type_has_params(arg.ty) {
        debug!("   casting from %s", val_str(bcx.ccx().tn, val));
        val = PointerCast(bcx, val, lldestty);
    }

    debug!("--- trans_arg_expr passing %s", val_str(bcx.ccx().tn, val));
    return rslt(bcx, val);
}

// when invoking a method, an argument of type @T or ~T can be implicltly
// converted to an argument of type &T. Similarly, ~[T] can be converted to
// &[T] and so on.  If such a conversion (called borrowing) is necessary,
// then the borrowings table will have an appropriate entry inserted.  This
// routine consults this table and performs these adaptations.  It returns a
// new location for the borrowed result as well as a new type for the argument
// that reflects the borrowed value and not the original.
fn adapt_borrowed_value(lv: lval_result,
                        e: @ast::expr,
                        e_ty: ty::t) -> {lv: lval_result,
                                         ty: ty::t} {
    let bcx = lv.bcx;
    if !expr_is_borrowed(bcx, e) {
        return {lv:lv, ty:e_ty};
    }

    match ty::get(e_ty).struct {
      ty::ty_uniq(mt) | ty::ty_box(mt) => {
        let box_ptr = load_value_from_lval_result(lv, e_ty);
        let body_ptr = GEPi(bcx, box_ptr, ~[0u, abi::box_field_body]);
        let rptr_ty = ty::mk_rptr(bcx.tcx(), ty::re_static, mt);
        return {lv: lval_temp(bcx, body_ptr), ty: rptr_ty};
      }

      ty::ty_estr(_) | ty::ty_evec(_, _) => {
        let ccx = bcx.ccx();
        let val = match lv.kind {
          lv_temporary => lv.val,
          lv_owned => load_if_immediate(bcx, lv.val, e_ty),
          lv_owned_imm => lv.val
        };

        let unit_ty = ty::sequence_element_type(ccx.tcx, e_ty);
        let llunit_ty = type_of(ccx, unit_ty);
        let (base, len) = tvec::get_base_and_len(bcx, val, e_ty);
        let p = alloca(bcx, T_struct(~[T_ptr(llunit_ty), ccx.int_type]));

        debug!("adapt_borrowed_value: adapting %s to %s",
               val_str(bcx.ccx().tn, val),
               val_str(bcx.ccx().tn, p));

        Store(bcx, base, GEPi(bcx, p, ~[0u, abi::slice_elt_base]));
        Store(bcx, len, GEPi(bcx, p, ~[0u, abi::slice_elt_len]));

        // this isn't necessarily the type that rust would assign but it's
        // close enough for trans purposes, as it will have the same runtime
        // representation
        let slice_ty = ty::mk_evec(bcx.tcx(),
                                   {ty: unit_ty, mutbl: ast::m_imm},
                                   ty::vstore_slice(ty::re_static));

        return {lv: lval_temp(bcx, p), ty: slice_ty};
      }

      _ => {
        // Just take a reference. This is basically like trans_addr_of.
        let mut {bcx, val, kind} = trans_temp_lval(bcx, e);
        let is_immediate = ty::type_is_immediate(e_ty);
        if (kind == lv_temporary && is_immediate) || kind == lv_owned_imm {
            val = do_spill(bcx, val, e_ty);
        }
        return {lv: {bcx: bcx, val: val, kind: lv_temporary},
                ty: ty::mk_rptr(bcx.tcx(), ty::re_static,
                                {ty: e_ty, mutbl: ast::m_imm})};
      }
    }
}

enum call_args {
    arg_exprs(~[@ast::expr]),
    arg_vals(~[ValueRef])
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn trans_args(cx: block, llenv: ValueRef, args: call_args, fn_ty: ty::t,
              dest: dest, ret_flag: Option<ValueRef>)
    -> {bcx: block, args: ~[ValueRef], retslot: ValueRef} {
    let _icx = cx.insn_ctxt("trans_args");
    let mut temp_cleanups = ~[];
    let arg_tys = ty::ty_fn_args(fn_ty);
    let mut llargs: ~[ValueRef] = ~[];

    let ccx = cx.ccx();
    let mut bcx = cx;

    let retty = ty::ty_fn_ret(fn_ty);
    // Arg 0: Output pointer.
    let llretslot = match dest {
      ignore => {
        if ty::type_is_nil(retty) {
            llvm::LLVMGetUndef(T_ptr(T_nil()))
        } else { alloc_ty(bcx, retty) }
      }
      save_in(dst) => dst,
      by_val(_) => alloc_ty(bcx, retty)
    };

    vec::push(llargs, llretslot);

    // Arg 1: Env (closure-bindings / self value)
    vec::push(llargs, llenv);

    // ... then explicit args.

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    match args {
      arg_exprs(es) => {
        let llarg_tys = type_of_explicit_args(ccx, arg_tys);
        let last = es.len() - 1u;
        do vec::iteri(es) |i, e| {
            let r = trans_arg_expr(bcx, arg_tys[i], llarg_tys[i],
                                   e, temp_cleanups, if i == last { ret_flag }
                                   else { None }, 0u);
            bcx = r.bcx;
            vec::push(llargs, r.val);
        }
      }
      arg_vals(vs) => {
        vec::push_all(llargs, vs);
      }
    }

    // now that all arguments have been successfully built, we can revoke any
    // temporary cleanups, as they are only needed if argument construction
    // should fail (for example, cleanup of copy mode args).
    do vec::iter(temp_cleanups) |c| {
        revoke_clean(bcx, c)
    }

    return {bcx: bcx,
         args: llargs,
         retslot: llretslot};
}

fn trans_call(in_cx: block, call_ex: @ast::expr, f: @ast::expr,
              args: call_args, id: ast::node_id, dest: dest)
    -> block {
    let _icx = in_cx.insn_ctxt("trans_call");
    trans_call_inner(
        in_cx, call_ex.info(), expr_ty(in_cx, f), node_id_type(in_cx, id),
        |cx| trans_callee(cx, f), args, dest)
}

fn body_contains_ret(body: ast::blk) -> bool {
    let cx = {mut found: false};
    visit::visit_block(body, cx, visit::mk_vt(@{
        visit_item: |_i, _cx, _v| { },
        visit_expr: |e: @ast::expr, cx: {mut found: bool}, v| {
            if !cx.found {
                match e.node {
                  ast::expr_ret(_) => cx.found = true,
                  _ => visit::visit_expr(e, cx, v),
                }
            }
        } with *visit::default_visitor()
    }));
    cx.found
}

// See [Note-arg-mode]
fn trans_call_inner(
    ++in_cx: block,
    call_info: Option<node_info>,
    fn_expr_ty: ty::t,
    ret_ty: ty::t,
    get_callee: fn(block) -> lval_maybe_callee,
    args: call_args,
    dest: dest) -> block {

    do with_scope(in_cx, call_info, ~"call") |cx| {
        let ret_in_loop = match args {
          arg_exprs(args) => {
            args.len() > 0u && match vec::last(args).node {
              ast::expr_loop_body(@{
                node: ast::expr_fn_block(_, body, _),
                _
              }) =>  body_contains_ret(body),
              _ => false
            }
          }
          _ => false
        };

        let f_res = get_callee(cx);
        let mut bcx = f_res.bcx;
        let ccx = cx.ccx();
        let ret_flag = if ret_in_loop {
            let flag = alloca(bcx, T_bool());
            Store(bcx, C_bool(false), flag);
            Some(flag)
        } else { None };

        let mut faddr = f_res.val;
        let llenv = match f_res.env {
          null_env => {
            llvm::LLVMGetUndef(T_opaque_box_ptr(ccx))
          }
          self_env(e, _, _, _) => {
            PointerCast(bcx, e, T_opaque_box_ptr(ccx))
          }
          is_closure => {
            // It's a closure. Have to fetch the elements
            if f_res.kind == lv_owned {
                faddr = load_if_immediate(bcx, faddr, fn_expr_ty);
            }
            let pair = faddr;
            faddr = GEPi(bcx, pair, ~[0u, abi::fn_field_code]);
            faddr = Load(bcx, faddr);
            let llclosure = GEPi(bcx, pair, ~[0u, abi::fn_field_box]);
            Load(bcx, llclosure)
          }
        };

        let args_res = {
            trans_args(bcx, llenv, args, fn_expr_ty, dest, ret_flag)
        };
        bcx = args_res.bcx;
        let mut llargs = args_res.args;

        let llretslot = args_res.retslot;

        // Now that the arguments have finished evaluating, we need to revoke
        // the cleanup for the self argument, if it exists
        match f_res.env {
          self_env(e, _, _, ast::by_copy) => revoke_clean(bcx, e),
          _ => (),
        }

        /* If the block is terminated,
        then one or more of the args has
        type _|_. Since that means it diverges, the code
        for the call itself is unreachable. */
        bcx = invoke(bcx, faddr, llargs);
        match dest {
          ignore => {
            if llvm::LLVMIsUndef(llretslot) != lib::llvm::True {
                bcx = drop_ty(bcx, llretslot, ret_ty);
            }
          }
          save_in(_) => { } // Already saved by callee
          by_val(cell) => {
            *cell = Load(bcx, llretslot);
          }
        }
        if ty::type_is_bot(ret_ty) {
            Unreachable(bcx);
        } else if ret_in_loop {
            bcx = do with_cond(bcx, Load(bcx, option::get(ret_flag))) |bcx| {
                do option::iter(copy bcx.fcx.loop_ret) |lret| {
                    Store(bcx, C_bool(true), lret.flagptr);
                    Store(bcx, C_bool(false), bcx.fcx.llretptr);
                }
                cleanup_and_leave(bcx, None, Some(bcx.fcx.llreturn));
                Unreachable(bcx);
                bcx
            }
        }
        bcx
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

fn trans_tup(bcx: block, elts: ~[@ast::expr], dest: dest) -> block {
    let _icx = bcx.insn_ctxt("trans_tup");
    let mut bcx = bcx;
    let addr = match dest {
      ignore => {
        for vec::each(elts) |ex| { bcx = trans_expr(bcx, ex, ignore); }
        return bcx;
      }
      save_in(pos) => pos,
      _ => bcx.tcx().sess.bug(~"trans_tup: weird dest")
    };
    let mut temp_cleanups = ~[];
    for vec::eachi(elts) |i, e| {
        let dst = GEPi(bcx, addr, ~[0u, i]);
        let e_ty = expr_ty(bcx, e);
        bcx = trans_expr_save_in(bcx, e, dst);
        add_clean_temp_mem(bcx, dst, e_ty);
        vec::push(temp_cleanups, dst);
    }
    for vec::each(temp_cleanups) |cleanup| { revoke_clean(bcx, cleanup); }
    return bcx;
}

fn trans_rec(bcx: block, fields: ~[ast::field],
             base: Option<@ast::expr>, id: ast::node_id,
             // none = ignore; some(x) = save_in(x)
             dest: Option<ValueRef>) -> block {
    let _icx = bcx.insn_ctxt("trans_rec");
    let t = node_id_type(bcx, id);
    let mut bcx = bcx;
    let addr = match dest {
      None => {
        for vec::each(fields) |fld| {
            bcx = trans_expr(bcx, fld.node.expr, ignore);
        }
        return bcx;
      }
      Some(pos) => pos
    };

    let ty_fields = match ty::get(t).struct {
        ty::ty_rec(f) => f,
        _ => bcx.sess().bug(~"trans_rec: record has non-record type")
    };

    let mut temp_cleanups = ~[];
    for fields.each |fld| {
        let ix = option::get(vec::position(ty_fields,
            |ft| ft.ident == fld.node.ident));
        let dst = GEPi(bcx, addr, ~[0u, ix]);
        bcx = trans_expr_save_in(bcx, fld.node.expr, dst);
        add_clean_temp_mem(bcx, dst, ty_fields[ix].mt.ty);
        vec::push(temp_cleanups, dst);
    }
    match base {
      Some(bexp) => {
        let {bcx: cx, val: base_val} = trans_temp_expr(bcx, bexp);
        bcx = cx;
        // Copy over inherited fields
        for ty_fields.eachi |i, tf| {
            if !vec::any(fields, |f| f.node.ident == tf.ident) {
                let dst = GEPi(bcx, addr, ~[0u, i]);
                let base = GEPi(bcx, base_val, ~[0u, i]);
                let val = load_if_immediate(bcx, base, tf.mt.ty);
                bcx = copy_val(bcx, INIT, dst, val, tf.mt.ty);
            }
        }
      }
      None => ()
    };

    // Now revoke the cleanups as we pass responsibility for the data
    // structure on to the caller
    for temp_cleanups.each |cleanup| { revoke_clean(bcx, cleanup); }
    return bcx;
}

// If the class has a destructor, our GEP is a little more
// complicated.
fn get_struct_field(block_context: block, dest_address: ValueRef,
                    class_id: ast::def_id, index: uint) -> ValueRef {
    if ty::ty_dtor(block_context.tcx(), class_id).is_some() {
        return GEPi(block_context,
                    GEPi(block_context, dest_address, ~[0, 1]),
                    ~[0, index]);
    }
    return GEPi(block_context, dest_address, ~[0, index]);
}

fn trans_struct(block_context: block, span: span, fields: ~[ast::field],
                base: Option<@ast::expr>, id: ast::node_id, dest: dest)
             -> block {

    let _instruction_context = block_context.insn_ctxt("trans_struct");
    let mut block_context = block_context;
    let type_context = block_context.ccx().tcx;

    let struct_type = node_id_type(block_context, id);

    // Get the address to store the structure into. If there is no address,
    // just translate each field and be done with it.
    let dest_address;
    match dest {
        ignore => {
            for fields.each |field| {
                block_context = trans_expr(block_context,
                                           field.node.expr,
                                           ignore);
            }
            return block_context;
        }
        save_in(destination_address) => {
            dest_address = destination_address;
        }
        by_val(_) => {
            type_context.sess.span_bug(span, ~"didn't expect by_val");
        }
    }

    // Get the class ID and its fields.
    let class_fields, class_id, substitutions;
    match ty::get(struct_type).struct {
        ty::ty_class(existing_class_id, ref existing_substitutions) => {
            class_id = existing_class_id;
            substitutions = existing_substitutions;
            class_fields = ty::lookup_class_fields(type_context, class_id);
        }
        _ => {
            type_context.sess.span_bug(span, ~"didn't resolve to a struct");
        }
    }

    // Add the drop flag if necessary.
    if ty::ty_dtor(block_context.tcx(), class_id).is_some() {
        let llflagptr = GEPi(block_context, dest_address, ~[0, 0]);
        Store(block_context, C_u8(1), llflagptr);
    }

    // Now translate each field.
    let mut temp_cleanups = ~[];
    for fields.each |field| {
        let mut found = None;
        for class_fields.eachi |i, class_field| {
            if class_field.ident == field.node.ident {
                found = Some((i, class_field.id));
                break;
            }
        }

        let index, field_id;
        match found {
            Some((found_index, found_field_id)) => {
                index = found_index;
                field_id = found_field_id;
            }
            None => {
                type_context.sess.span_bug(span, ~"unknown field");
            }
        }

        let dest = get_struct_field(block_context, dest_address, class_id,
                                    index);

        block_context = trans_expr_save_in(block_context,
                                           field.node.expr,
                                           dest);

        let field_type = ty::lookup_field_type(type_context, class_id,
                                               field_id, substitutions);
        add_clean_temp_mem(block_context, dest, field_type);
        vec::push(temp_cleanups, dest);
    }

    match base {
        Some(base_expr) => {
            let { bcx: bcx, val: llbasevalue } =
                trans_temp_expr(block_context, base_expr);
            block_context = bcx;

            // Copy over inherited fields.
            for class_fields.eachi |i, class_field| {
                let exists = do vec::any(fields) |provided_field| {
                   provided_field.node.ident == class_field.ident
                };
                if exists {
                    again;
                }
                let lldestfieldvalue = get_struct_field(block_context,
                                                        dest_address,
                                                        class_id,
                                                        i);
                let llbasefieldvalue = GEPi(block_context,
                                            llbasevalue,
                                            ~[0, i]);
                let field_type = ty::lookup_field_type(block_context.tcx(),
                                                       class_id,
                                                       class_field.id,
                                                       substitutions);
                let llbasefieldvalue = load_if_immediate(block_context,
                                                         llbasefieldvalue,
                                                         field_type);
                block_context = copy_val(block_context, INIT,
                                         lldestfieldvalue, llbasefieldvalue,
                                         field_type);
            }
        }
        None => ()
    }

    // Now revoke the cleanups, as we pass responsibility for the data
    // structure onto the caller.
    for temp_cleanups.each |temp_cleanup| {
        revoke_clean(block_context, temp_cleanup);
    }

    block_context
}

// Store the result of an expression in the given memory location, ensuring
// that nil or bot expressions get ignore rather than save_in as destination.
fn trans_expr_save_in(bcx: block, e: @ast::expr, dest: ValueRef)
    -> block {
    let t = expr_ty(bcx, e);
    let do_ignore = ty::type_is_bot(t) || ty::type_is_nil(t);
    return trans_expr(bcx, e, if do_ignore { ignore } else { save_in(dest) });
}

// Call this to compile an expression that you need as an intermediate value,
// and you want to know whether you're dealing with an lval or not (the kind
// field in the returned struct). For non-intermediates, use trans_expr or
// trans_expr_save_in. For intermediates where you don't care about lval-ness,
// use trans_temp_expr.
fn trans_temp_lval(bcx: block, e: @ast::expr) -> lval_result {
    let _icx = bcx.insn_ctxt("trans_temp_lval");
    let mut bcx = bcx;
    if expr_is_lval(bcx, e) {
        return trans_lval(bcx, e);
    } else {
        let ty = expr_ty(bcx, e);
        if ty::type_is_nil(ty) || ty::type_is_bot(ty) {
            bcx = trans_expr(bcx, e, ignore);
            return {bcx: bcx, val: C_nil(), kind: lv_temporary};
        } else if ty::type_is_immediate(ty) {
            let cell = empty_dest_cell();
            bcx = trans_expr(bcx, e, by_val(cell));
            add_clean_temp_immediate(bcx, *cell, ty);
            return {bcx: bcx, val: *cell, kind: lv_temporary};
        } else {
            let scratch = alloc_ty(bcx, ty);
            let bcx = trans_expr_save_in(bcx, e, scratch);
            add_clean_temp_mem(bcx, scratch, ty);
            return {bcx: bcx, val: scratch, kind: lv_temporary};
        }
    }
}

// Use only for intermediate values. See trans_expr and trans_expr_save_in for
// expressions that must 'end up somewhere' (or get ignored).
fn trans_temp_expr(bcx: block, e: @ast::expr) -> result {
    let _icx = bcx.insn_ctxt("trans_temp_expr");
    lval_result_to_result(trans_temp_lval(bcx, e), expr_ty(bcx, e))
}

fn load_value_from_lval_result(lv: lval_result, ty: ty::t) -> ValueRef {
    match lv.kind {
      lv_temporary => lv.val,
      lv_owned => load_if_immediate(lv.bcx, lv.val, ty),
      lv_owned_imm => lv.val
    }
}

fn lval_result_to_result(lv: lval_result, ty: ty::t) -> result {
    let val = load_value_from_lval_result(lv, ty);
    {bcx: lv.bcx, val: val}
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

// Translate an expression, with the dest argument deciding what happens with
// the result. Invariants:
// - exprs returning nil or bot always get dest=ignore
// - exprs with non-immediate type never get dest=by_val
fn trans_expr(bcx: block, e: @ast::expr, dest: dest) -> block {
    let _icx = bcx.insn_ctxt("trans_expr");
    debuginfo::update_source_pos(bcx, e.span);

    if expr_is_lval(bcx, e) {
        return lval_to_dps(bcx, e, dest);
    }

    return match bcx.ccx().maps.root_map.find({id:e.id, derefs:0u}) {
      None => unrooted(bcx, e, dest),
      Some(scope_id) => {
        debug!("expression %d found in root map with scope %d",
               e.id, scope_id);

        let ty = expr_ty(bcx, e);
        let root_loc = alloca_zeroed(bcx, type_of(bcx.ccx(), ty));
        let bcx = unrooted(bcx, e, save_in(root_loc));

        if !bcx.sess().no_asm_comments() {
            add_comment(bcx, fmt!("preserving until end of scope %d",
                                  scope_id));
        }

        let _icx = bcx.insn_ctxt("root_value_expr");
        add_root_cleanup(bcx, scope_id, root_loc, ty);
        let lv = {bcx: bcx, val: root_loc, kind: lv_owned};
        lval_result_to_dps(lv, ty, false, dest)
      }
    };

    fn unrooted(bcx: block, e: @ast::expr, dest: dest) -> block {
        let tcx = bcx.tcx();
        match e.node {
          ast::expr_if(cond, thn, els) => {
            return trans_if(bcx, cond, thn, els, dest);
          }
          ast::expr_match(expr, arms) => {
            return alt::trans_alt(bcx, e, expr, arms, dest);
          }
          ast::expr_block(blk) => {
            return do with_scope(bcx, blk.info(), ~"block-expr body") |bcx| {
                trans_block(bcx, blk, dest)
            };
          }
          ast::expr_rec(args, base) => {
              let d = match dest {
                  ignore => None,
                  save_in(p) => Some(p),
                  _ => bcx.sess().impossible_case(e.span,
                        "trans_expr::unrooted: can't pass a record by val")
              };
            return trans_rec(bcx, args, base, e.id, d);
          }
          ast::expr_struct(_, fields, base) => {
            return trans_struct(bcx, e.span, fields, base, e.id, dest);
          }
          ast::expr_tup(args) => { return trans_tup(bcx, args, dest); }
          ast::expr_vstore(e, v) => {
            return tvec::trans_vstore(bcx, e, v, dest);
          }
          ast::expr_lit(lit) => return trans_lit(bcx, e, *lit, dest),
          ast::expr_vec(args, _) => {
            return tvec::trans_evec(bcx, tvec::individual_evec(args),
                                    ast::vstore_fixed(None), e.id, dest);
          }
          ast::expr_repeat(element, count_expr, _) => {
            let count = ty::eval_repeat_count(bcx.tcx(), count_expr, e.span);
            return tvec::trans_evec(bcx, tvec::repeating_evec(element, count),
                                    ast::vstore_fixed(None), e.id, dest);
          }
          ast::expr_binary(op, lhs, rhs) => {
            return trans_binary(bcx, op, lhs, rhs, dest, e);
          }
          ast::expr_unary(op, x) => {
            assert op != ast::deref; // lvals are handled above
            return trans_unary(bcx, op, x, e, dest);
          }
          ast::expr_addr_of(_, x) => { return trans_addr_of(bcx, x, dest); }
          ast::expr_fn(proto, decl, body, cap_clause) => {
            // Don't use this function for anything real. Use the one in
            // astconv instead.
            fn ast_proto_to_proto_simple(ast_proto: ast::proto)
                                      -> ty::fn_proto {
                match ast_proto {
                    ast::proto_bare =>
                        ty::proto_bare,
                    ast::proto_uniq =>
                        ty::proto_vstore(ty::vstore_uniq),
                    ast::proto_box =>
                        ty::proto_vstore(ty::vstore_box),
                    ast::proto_block =>
                        ty::proto_vstore(ty::vstore_slice(ty::re_static))
                }
            }

            // XXX: This syntax should be reworked a bit (in the parser I
            // guess?); @fn() { ... } won't work.
            return closure::trans_expr_fn(bcx,
                                          ast_proto_to_proto_simple(proto),
                                          decl, body, e.id, cap_clause, None,
                                          dest);
          }
          ast::expr_fn_block(decl, body, cap_clause) => {
            match ty::get(expr_ty(bcx, e)).struct {
              ty::ty_fn({proto, _}) => {
                debug!("translating fn_block %s with type %s",
                       expr_to_str(e, tcx.sess.intr()),
                       ppaux::ty_to_str(tcx, expr_ty(bcx, e)));
                return closure::trans_expr_fn(bcx, proto, decl, body,
                                           e.id, cap_clause, None, dest);
              }
              _ =>  bcx.sess().impossible_case(e.span, "fn_block has \
                         body with a non-fn type")
            }
          }
          ast::expr_loop_body(blk) => {
              match ty::get(expr_ty(bcx, e)).struct {
                  ty::ty_fn({proto, _}) => {
                      match blk.node {
                          ast::expr_fn_block(decl, body, cap) =>
                            return trans_loop_body(bcx, blk.id, decl, body,
                                                   proto, cap, None, dest),
                          _ => bcx.sess().impossible_case(e.span, "loop_body \
                                 has the wrong kind of contents")
                      }

                  }
                  _ => bcx.sess().impossible_case(e.span, "loop_body has \
                         body with a non-fn type")
              }
          }
          ast::expr_do_body(blk) => {
            return trans_expr(bcx, blk, dest);
          }
          ast::expr_copy(a) | ast::expr_unary_move(a) => {
            if !expr_is_lval(bcx, a) {
                return trans_expr(bcx, a, dest);
            }
            else { return lval_to_dps(bcx, a, dest); }
          }
          ast::expr_cast(val, _) => return trans_cast(bcx, val, e.id, dest),
          ast::expr_call(f, args, _) => {
            return trans_call(bcx, e, f, arg_exprs(args), e.id, dest);
          }
          ast::expr_field(base, _, _) => {
            if dest == ignore { return trans_expr(bcx, base, ignore); }
            let callee = trans_callee(bcx, e), ty = expr_ty(bcx, e);
            let lv = lval_maybe_callee_to_lval(callee, e.span);
            revoke_clean(lv.bcx, lv.val);
            memmove_ty(lv.bcx, get_dest_addr(dest), lv.val, ty);
            return lv.bcx;
          }
          ast::expr_index(base, idx) => {
            // If it is here, it's not an lval, so this is a user-defined
            // index op
            let origin = bcx.ccx().maps.method_map.get(e.id);
            let fty = node_id_type(bcx, e.callee_id);
            return trans_call_inner(
                bcx, e.info(), fty,
                expr_ty(bcx, e),
                |bcx| impl::trans_method_callee(bcx, e.callee_id, base,
                                                origin),
                arg_exprs(~[idx]), dest);
          }

          // These return nothing
          ast::expr_break(label_opt) => {
            assert dest == ignore;
            if label_opt.is_some() {
                bcx.tcx().sess.span_unimpl(e.span, ~"labeled break");
            }
            return trans_break(bcx);
          }
          ast::expr_again(label_opt) => {
            assert dest == ignore;
            if label_opt.is_some() {
                bcx.tcx().sess.span_unimpl(e.span, ~"labeled again");
            }
            return trans_cont(bcx);
          }
          ast::expr_ret(ex) => {
            assert dest == ignore;
            return trans_ret(bcx, ex);
          }
          ast::expr_fail(expr) => {
            assert dest == ignore;
            return trans_fail_expr(bcx, Some(e.span), expr);
          }
          ast::expr_log(_, lvl, a) => {
            assert dest == ignore;
            return trans_log(e, lvl, bcx, a);
          }
          ast::expr_assert(a) => {
            assert dest == ignore;
            return trans_check_expr(bcx, e, a, ~"Assertion");
          }
          ast::expr_while(cond, body) => {
            assert dest == ignore;
            return trans_while(bcx, cond, body);
          }
          ast::expr_loop(body, _) => {
            assert dest == ignore;
            return trans_loop(bcx, body);
          }
          ast::expr_assign(dst, src) => {
            assert dest == ignore;
            let src_r = trans_temp_lval(bcx, src);
            let {bcx, val: addr, kind} = trans_lval(src_r.bcx, dst);
            assert kind == lv_owned;
            let is_last_use =
                bcx.ccx().maps.last_use_map.contains_key(src.id);
            return store_temp_expr(bcx, DROP_EXISTING, addr, src_r,
                                expr_ty(bcx, src), is_last_use);
          }
          ast::expr_move(dst, src) => {
            // FIXME: calculate copy init-ness in typestate. (#839)
            assert dest == ignore;
            let src_r = trans_temp_lval(bcx, src);
            let {bcx, val: addr, kind} = trans_lval(src_r.bcx, dst);
            assert kind == lv_owned;
            return move_val(bcx, DROP_EXISTING, addr, src_r,
                         expr_ty(bcx, src));
          }
          ast::expr_swap(dst, src) => {
            assert dest == ignore;
            let lhs_res = trans_lval(bcx, dst);
            assert lhs_res.kind == lv_owned;
            let rhs_res = trans_lval(lhs_res.bcx, src);
            let t = expr_ty(bcx, src);
            let tmp_alloc = alloc_ty(rhs_res.bcx, t);
            // Swap through a temporary.
            let bcx = move_val(rhs_res.bcx, INIT, tmp_alloc, lhs_res, t);
            let bcx = move_val(bcx, INIT, lhs_res.val, rhs_res, t);
            return move_val(bcx, INIT, rhs_res.val,
                         lval_owned(bcx, tmp_alloc), t);
          }
          ast::expr_assign_op(op, dst, src) => {
            assert dest == ignore;
            return trans_assign_op(bcx, e, op, dst, src);
          }
          _ => {
            bcx.tcx().sess.span_bug(e.span, ~"trans_expr reached \
                                             fall-through case");
          }
        }
    }
}

fn lval_to_dps(bcx: block, e: @ast::expr, dest: dest) -> block {
    let last_use_map = bcx.ccx().maps.last_use_map;
    let ty = expr_ty(bcx, e);
    let lv = trans_lval(bcx, e);
    let last_use = (lv.kind == lv_owned && last_use_map.contains_key(e.id));
    debug!("is last use (%s) = %b, %d", expr_to_str(e, bcx.ccx().sess.intr()),
           last_use, lv.kind as int);
    lval_result_to_dps(lv, ty, last_use, dest)
}

fn lval_result_to_dps(lv: lval_result, ty: ty::t,
                      last_use: bool, dest: dest) -> block {
    let mut {bcx, val, kind} = lv;
    let ccx = bcx.ccx();
    match dest {
      by_val(cell) => {
        if kind == lv_temporary {
            revoke_clean(bcx, val);
            *cell = val;
        } else if last_use {
            *cell = Load(bcx, val);
            if ty::type_needs_drop(ccx.tcx, ty) {
                bcx = zero_mem(bcx, val, ty);
            }
        } else {
            if kind == lv_owned { val = Load(bcx, val); }
            let {bcx: cx, val} = take_ty_immediate(bcx, val, ty);
            *cell = val;
            bcx = cx;
        }
      }
      save_in(loc) => {
        bcx = store_temp_expr(bcx, INIT, loc, lv, ty, last_use);
      }
      ignore => ()
    }
    return bcx;
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

fn trans_log(log_ex: @ast::expr, lvl: @ast::expr,
             bcx: block, e: @ast::expr) -> block {
    let _icx = bcx.insn_ctxt("trans_log");
    let ccx = bcx.ccx();
    if ty::type_is_bot(expr_ty(bcx, lvl)) {
       return trans_expr(bcx, lvl, ignore);
    }

    let modpath = vec::append(
        ~[path_mod(ccx.sess.ident_of(ccx.link_meta.name))],
        vec::filter(bcx.fcx.path, |e|
            match e { path_mod(_) => true, _ => false }
        ));
    let modname = path_str(ccx.sess, modpath);

    let global = if ccx.module_data.contains_key(modname) {
        ccx.module_data.get(modname)
    } else {
        let s = link::mangle_internal_name_by_path_and_seq(
            ccx, modpath, ~"loglevel");
        let global = str::as_c_str(s, |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, T_i32(), buf)
        });
        llvm::LLVMSetGlobalConstant(global, False);
        llvm::LLVMSetInitializer(global, C_null(T_i32()));
        lib::llvm::SetLinkage(global, lib::llvm::InternalLinkage);
        ccx.module_data.insert(modname, global);
        global
    };
    let current_level = Load(bcx, global);
    let {bcx, val: level} = {
        do with_scope_result(bcx, lvl.info(), ~"level") |bcx| {
            trans_temp_expr(bcx, lvl)
        }
    };

    do with_cond(bcx, ICmp(bcx, lib::llvm::IntUGE, current_level, level))
        |bcx| {
        do with_scope(bcx, log_ex.info(), ~"log") |bcx| {
            let {bcx, val, _} = trans_temp_expr(bcx, e);
            let e_ty = expr_ty(bcx, e);
            let tydesc = get_tydesc_simple(ccx, e_ty);
            // Call the polymorphic log function.
            let val = spill_if_immediate(bcx, val, e_ty);
            let val = PointerCast(bcx, val, T_ptr(T_i8()));
            Call(bcx, ccx.upcalls.log_type, ~[tydesc, val, level]);
            bcx
        }
    }
}

fn trans_check_expr(bcx: block, chk_expr: @ast::expr,
                    pred_expr: @ast::expr, s: ~str) -> block {
    let _icx = bcx.insn_ctxt("trans_check_expr");
    let expr_str = s + ~" " + expr_to_str(pred_expr, bcx.ccx().sess.intr())
        + ~" failed";
    let {bcx, val} = {
        do with_scope_result(bcx, chk_expr.info(), ~"check") |bcx| {
            trans_temp_expr(bcx, pred_expr)
        }
    };
    do with_cond(bcx, Not(bcx, val)) |bcx| {
        trans_fail(bcx, Some(pred_expr.span), expr_str)
    }
}

fn trans_fail_expr(bcx: block, sp_opt: Option<span>,
                   fail_expr: Option<@ast::expr>) -> block {
    let _icx = bcx.insn_ctxt("trans_fail_expr");
    let mut bcx = bcx;
    match fail_expr {
      Some(expr) => {
        let ccx = bcx.ccx(), tcx = ccx.tcx;
        let expr_res = trans_temp_expr(bcx, expr);
        let e_ty = expr_ty(bcx, expr);
        bcx = expr_res.bcx;

        if ty::type_is_str(e_ty) {
            let body = tvec::get_bodyptr(bcx, expr_res.val);
            let data = tvec::get_dataptr(bcx, body);
            return trans_fail_value(bcx, sp_opt, data);
        } else if bcx.unreachable || ty::type_is_bot(e_ty) {
            return bcx;
        } else {
            bcx.sess().span_bug(
                expr.span, ~"fail called with unsupported type " +
                ppaux::ty_to_str(tcx, e_ty));
        }
      }
      _ => return trans_fail(bcx, sp_opt, ~"explicit failure")
    }
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

fn trans_fail(bcx: block, sp_opt: Option<span>, fail_str: ~str) ->
    block {
    let _icx = bcx.insn_ctxt("trans_fail");
    let V_fail_str = C_cstr(bcx.ccx(), fail_str);
    return trans_fail_value(bcx, sp_opt, V_fail_str);
}

fn trans_fail_value(bcx: block, sp_opt: Option<span>,
                    V_fail_str: ValueRef) -> block {
    let _icx = bcx.insn_ctxt("trans_fail_value");
    let ccx = bcx.ccx();
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
    let V_str = PointerCast(bcx, V_fail_str, T_ptr(T_i8()));
    let V_filename = PointerCast(bcx, V_filename, T_ptr(T_i8()));
    let args = ~[V_str, V_filename, C_int(ccx, V_line)];
    let bcx = trans_rtcall(bcx, ~"fail", args, ignore);
    Unreachable(bcx);
    return bcx;
}

fn trans_rtcall(bcx: block, name: ~str, args: ~[ValueRef], dest: dest)
    -> block {
    let did = bcx.ccx().rtcalls[name];
    let fty = if did.crate == ast::local_crate {
        ty::node_id_to_type(bcx.ccx().tcx, did.node)
    } else {
        csearch::get_type(bcx.ccx().tcx, did).ty
    };
    let rty = ty::ty_fn_ret(fty);
    return trans_call_inner(
        bcx, None, fty, rty,
        |bcx| lval_static_fn_inner(bcx, did, 0, ~[], None),
        arg_vals(args), dest);
}

fn trans_break_cont(bcx: block, to_end: bool)
    -> block {
    let _icx = bcx.insn_ctxt("trans_break_cont");
    // Locate closest loop block, outputting cleanup as we go.
    let mut unwind = bcx;
    let mut target;
    loop {
        match unwind.kind {
          block_scope({loop_break: Some(brk), _}) => {
            target = if to_end {
                brk
            } else {
                unwind
            };
            break;
          }
          _ => ()
        }
        unwind = match unwind.parent {
          Some(cx) => cx,
          // This is a return from a loop body block
          None => {
            Store(bcx, C_bool(!to_end), bcx.fcx.llretptr);
            cleanup_and_leave(bcx, None, Some(bcx.fcx.llreturn));
            Unreachable(bcx);
            return bcx;
          }
        };
    }
    cleanup_and_Br(bcx, unwind, target.llbb);
    Unreachable(bcx);
    return bcx;
}

fn trans_break(cx: block) -> block {
    return trans_break_cont(cx, true);
}

fn trans_cont(cx: block) -> block {
    return trans_break_cont(cx, false);
}

fn trans_ret(bcx: block, e: Option<@ast::expr>) -> block {
    let _icx = bcx.insn_ctxt("trans_ret");
    let mut bcx = bcx;
    let retptr = match copy bcx.fcx.loop_ret {
      Some({flagptr, retptr}) => {
        // This is a loop body return. Must set continue flag (our retptr)
        // to false, return flag to true, and then store the value in the
        // parent's retptr.
        Store(bcx, C_bool(true), flagptr);
        Store(bcx, C_bool(false), bcx.fcx.llretptr);
        match e {
          Some(x) => PointerCast(bcx, retptr,
                                 T_ptr(type_of(bcx.ccx(), expr_ty(bcx, x)))),
          None => retptr
        }
      }
      None => bcx.fcx.llretptr
    };
    match e {
      Some(x) => {
        bcx = trans_expr_save_in(bcx, x, retptr);
      }
      _ => ()
    }
    cleanup_and_leave(bcx, None, Some(bcx.fcx.llreturn));
    Unreachable(bcx);
    return bcx;
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
    let _icx = bcx.insn_ctxt("init_local");
    let ty = node_id_type(bcx, local.node.id);

    if ignore_lhs(bcx, local) {
        // Handle let _ = e; just like e;
        match local.node.init {
            Some(init) => {
              return trans_expr(bcx, init.expr, ignore);
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
        if init.op == ast::init_assign || !expr_is_lval(bcx, init.expr) {
            bcx = trans_expr_save_in(bcx, init.expr, llptr);
        } else { // This is a move from an lval, must perform an actual move
            let sub = trans_lval(bcx, init.expr);
            bcx = move_val(sub.bcx, INIT, llptr, sub, ty);
        }
      }
      _ => bcx = zero_mem(bcx, llptr, ty),
    }
    // Make a note to drop this slot on the way out.
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
        bcx = trans_expr(cx, e, ignore);
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
    if bcx.unreachable { return bcx; }
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
    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    leave_block(f(scope_cx), scope_cx)
}

fn with_scope_result(bcx: block, opt_node_info: Option<node_info>,
                     name: ~str, f: fn(block) -> result)
    -> result {
    let _icx = bcx.insn_ctxt("with_scope_result");
    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    let {bcx, val} = f(scope_cx);
    {bcx: leave_block(bcx, scope_cx), val: val}
}

fn with_cond(bcx: block, val: ValueRef, f: fn(block) -> block) -> block {
    let _icx = bcx.insn_ctxt("with_cond");
    let next_cx = sub_block(bcx, ~"next"), cond_cx = sub_block(bcx, ~"cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb);
    let after_cx = f(cond_cx);
    if !after_cx.terminated { Br(after_cx, next_cx.llbb); }
    next_cx
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

fn alloc_ty(bcx: block, t: ty::t) -> ValueRef {
    let _icx = bcx.insn_ctxt("alloc_ty");
    let ccx = bcx.ccx();
    let llty = type_of(ccx, t);
    if ty::type_has_params(t) { log(error, ppaux::ty_to_str(ccx.tcx, t)); }
    assert !ty::type_has_params(t);
    let val = alloca(bcx, llty);
    return val;
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

fn trans_block(bcx: block, b: ast::blk, dest: dest)
    -> block {
    let _icx = bcx.insn_ctxt("trans_block");
    let mut bcx = bcx;
    do block_locals(b) |local| { bcx = alloc_local(bcx, local); };
    for vec::each(b.node.stmts) |s| {
        debuginfo::update_source_pos(bcx, b.span);
        bcx = trans_stmt(bcx, *s);
    }
    match b.node.expr {
      Some(e) => {
        let bt = ty::type_is_bot(expr_ty(bcx, e));
        debuginfo::update_source_pos(bcx, e.span);
        bcx = trans_expr(bcx, e, if bt { ignore } else { dest });
      }
      _ => assert dest == ignore || bcx.unreachable
    }
    return bcx;
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
        cx.llself = Some({v: cx.llenv, t: tt, is_owned: false});
      }
      impl_owned_self(tt) => {
        cx.llself = Some({v: cx.llenv, t: tt, is_owned: true});
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
        // We really should do this regardless of whether self is owned,
        // but it doesn't work right with default method impls yet.
        if slf.is_owned {
            let self_val = PointerCast(bcx, slf.v,
                                       T_ptr(type_of(bcx.ccx(), slf.t)));
            fcx.llself = Some({v: self_val with slf});
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
        bcx = trans_block(bcx, body, ignore);
    } else {
        bcx = trans_block(bcx, body, save_in(fcx.llretptr));
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
        let lldiscrimptr = GEPi(bcx, llenumptr, ~[0u, 0u]);
        Store(bcx, C_int(ccx, disr), lldiscrimptr);
        GEPi(bcx, llenumptr, ~[0u, 1u])
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
  bcx_top = copy_args_to_allocas(fcx, bcx_top, decl.inputs,
              ty::ty_fn_args(node_id_type(bcx_top, ctor_id)));

  // We *don't* want self to be passed to the ctor -- that
  // wouldn't make sense
  // So we initialize it here

  let selfptr = alloc_ty(bcx_top, rslt_ty);
  // If we have a dtor, we have a two-word representation with a drop
  // flag, then a pointer to the class itself
  let valptr = if option::is_some(ty::ty_dtor(bcx_top.tcx(),
                                  parent_id)) {
    // Initialize the drop flag
    let one = C_u8(1u);
    let flag = GEPi(bcx_top, selfptr, ~[0u, 0u]);
    Store(bcx_top, one, flag);
    // Select the pointer to the class itself
    GEPi(bcx_top, selfptr, ~[0u, 1u])
  }
  else { selfptr };

  // initialize fields to zero
  let dsubsts = dummy_substs(psubsts.tys);
  let fields = ty::class_items_as_mutable_fields(bcx_top.tcx(), parent_id,
                                                 &dsubsts);
  let mut bcx = bcx_top;
  // Initialize fields to zero so init assignments can validly
  // drop their LHS
    for fields.each |field| {
     let ix = field_idx_strict(bcx.tcx(), sp, field.ident, fields);
     bcx = zero_mem(bcx, GEPi(bcx, valptr, ~[0u, ix]), field.mt.ty);
  }

  // note we don't want to take *or* drop self.
  fcx.llself = Some({v: selfptr, t: rslt_ty, is_owned: false});

  // Translate the body of the ctor
  bcx = trans_block(bcx_top, body, ignore);
  let lval_res = {bcx: bcx, val: selfptr, kind: lv_owned};
  // Generate the return expression
  bcx = store_temp_expr(bcx, INIT, fcx.llretptr, lval_res,
                        rslt_ty, true);
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

// Create a /real/ closure: this is like create_fn_pair, but creates a
// a fn value on the stack with a specified environment (which need not be
// on the stack).
fn create_real_fn_pair(cx: block, llfnty: TypeRef, llfn: ValueRef,
                       llenvptr: ValueRef) -> ValueRef {
    let pair = alloca(cx, T_fn_pair(cx.ccx(), llfnty));
    fill_fn_pair(cx, pair, llfn, llenvptr);
    return pair;
}

fn fill_fn_pair(bcx: block, pair: ValueRef, llfn: ValueRef,
                llenvptr: ValueRef) {
    let ccx = bcx.ccx();
    let code_cell = GEPi(bcx, pair, ~[0u, abi::fn_field_code]);
    Store(bcx, llfn, code_cell);
    let env_cell = GEPi(bcx, pair, ~[0u, abi::fn_field_box]);
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
              ast::item_fn(decl, purity, _, _) => {
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
                    ccx.item_symbols.insert(ni.id, copy *ident);
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
        visit_item: |a| trans_constant(ccx, a)
        with *visit::default_simple_visitor()
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
          ast::item_fn(decl, _, _, _) => {
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
        }
        with *visit::default_simple_visitor()
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
               emap: resolve3::ExportMap,
               emap2: resolve3::ExportMap2,
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
    emit_tydescs(ccx);
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
