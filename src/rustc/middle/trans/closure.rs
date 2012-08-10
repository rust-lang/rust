import libc::c_uint;
import syntax::ast;
import syntax::ast_util;
import lib::llvm::llvm;
import lib::llvm::{ValueRef, TypeRef};
import common::*;
import build::*;
import base::*;
import type_of::*;
import back::abi;
import syntax::codemap::span;
import syntax::print::pprust::expr_to_str;
import back::link::{
    mangle_internal_name_by_path,
    mangle_internal_name_by_path_and_seq};
import util::ppaux::ty_to_str;
import syntax::ast_map::{path, path_mod, path_name};
import driver::session::session;
import std::map::hashmap;

// ___Good to know (tm)__________________________________________________
//
// The layout of a closure environment in memory is
// roughly as follows:
//
// struct rust_opaque_box {         // see rust_internal.h
//   unsigned ref_count;            // only used for fn@()
//   type_desc *tydesc;             // describes closure_data struct
//   rust_opaque_box *prev;         // (used internally by memory alloc)
//   rust_opaque_box *next;         // (used internally by memory alloc)
//   struct closure_data {
//       type_desc *bound_tdescs[]; // bound descriptors
//       struct {
//         upvar1_t upvar1;
//         ...
//         upvarN_t upvarN;
//       } bound_data;
//    }
// };
//
// Note that the closure is itself a rust_opaque_box.  This is true
// even for fn~ and fn&, because we wish to keep binary compatibility
// between all kinds of closures.  The allocation strategy for this
// closure depends on the closure type.  For a sendfn, the closure
// (and the referenced type descriptors) will be allocated in the
// exchange heap.  For a fn, the closure is allocated in the task heap
// and is reference counted.  For a block, the closure is allocated on
// the stack.
//
// ## Opaque closures and the embedded type descriptor ##
//
// One interesting part of closures is that they encapsulate the data
// that they close over.  So when I have a ptr to a closure, I do not
// know how many type descriptors it contains nor what upvars are
// captured within.  That means I do not know precisely how big it is
// nor where its fields are located.  This is called an "opaque
// closure".
//
// Typically an opaque closure suffices because we only manipulate it
// by ptr.  The routine common::T_opaque_box_ptr() returns an
// appropriate type for such an opaque closure; it allows access to
// the box fields, but not the closure_data itself.
//
// But sometimes, such as when cloning or freeing a closure, we need
// to know the full information.  That is where the type descriptor
// that defines the closure comes in handy.  We can use its take and
// drop glue functions to allocate/free data as needed.
//
// ## Subtleties concerning alignment ##
//
// It is important that we be able to locate the closure data *without
// knowing the kind of data that is being bound*.  This can be tricky
// because the alignment requirements of the bound data affects the
// alignment requires of the closure_data struct as a whole.  However,
// right now this is a non-issue in any case, because the size of the
// rust_opaque_box header is always a mutiple of 16-bytes, which is
// the maximum alignment requirement we ever have to worry about.
//
// The only reason alignment matters is that, in order to learn what data
// is bound, we would normally first load the type descriptors: but their
// location is ultimately depend on their content!  There is, however, a
// workaround.  We can load the tydesc from the rust_opaque_box, which
// describes the closure_data struct and has self-contained derived type
// descriptors, and read the alignment from there.   It's just annoying to
// do.  Hopefully should this ever become an issue we'll have monomorphized
// and type descriptors will all be a bad dream.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum environment_value {
    // Copy the value from this llvm ValueRef into the environment.
    env_copy(ValueRef, ty::t, lval_kind),

    // Move the value from this llvm ValueRef into the environment.
    env_move(ValueRef, ty::t, lval_kind),

    // Access by reference (used for blocks).
    env_ref(ValueRef, ty::t, lval_kind),
}

fn ev_to_str(ccx: @crate_ctxt, ev: environment_value) -> ~str {
    match ev {
      env_copy(v, t, lk) => fmt!{"copy(%s,%s)", val_str(ccx.tn, v),
                                ty_to_str(ccx.tcx, t)},
      env_move(v, t, lk) => fmt!{"move(%s,%s)", val_str(ccx.tn, v),
                                ty_to_str(ccx.tcx, t)},
      env_ref(v, t, lk) => fmt!{"ref(%s,%s)", val_str(ccx.tn, v),
                                ty_to_str(ccx.tcx, t)}
    }
}

fn mk_tuplified_uniq_cbox_ty(tcx: ty::ctxt, cdata_ty: ty::t) -> ty::t {
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    return ty::mk_imm_uniq(tcx, cbox_ty);
}

// Given a closure ty, emits a corresponding tuple ty
fn mk_closure_tys(tcx: ty::ctxt,
                  bound_values: ~[environment_value])
    -> ty::t {
    let mut bound_tys = ~[];

    // Compute the closed over data
    for vec::each(bound_values) |bv| {
        vec::push(bound_tys, match bv {
            env_copy(_, t, _) => t,
            env_move(_, t, _) => t,
            env_ref(_, t, _) => t
        });
    }
    let cdata_ty = ty::mk_tup(tcx, bound_tys);
    debug!{"cdata_ty=%s", ty_to_str(tcx, cdata_ty)};
    return cdata_ty;
}

fn allocate_cbox(bcx: block,
                 ck: ty::closure_kind,
                 cdata_ty: ty::t)
    -> result {
    let _icx = bcx.insn_ctxt(~"closure::allocate_cbox");
    let ccx = bcx.ccx(), tcx = ccx.tcx;

    fn nuke_ref_count(bcx: block, llbox: ValueRef) {
        let _icx = bcx.insn_ctxt(~"closure::nuke_ref_count");
        // Initialize ref count to arbitrary value for debugging:
        let ccx = bcx.ccx();
        let llbox = PointerCast(bcx, llbox, T_opaque_box_ptr(ccx));
        let ref_cnt = GEPi(bcx, llbox, ~[0u, abi::box_field_refcnt]);
        let rc = C_int(ccx, 0x12345678);
        Store(bcx, rc, ref_cnt);
    }

    // Allocate and initialize the box:
    let {bcx, val} = match ck {
      ty::ck_box => malloc_raw(bcx, cdata_ty, heap_shared),
      ty::ck_uniq => malloc_raw(bcx, cdata_ty, heap_exchange),
      ty::ck_block => {
        let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
        let llbox = base::alloc_ty(bcx, cbox_ty);
        nuke_ref_count(bcx, llbox);
        {bcx: bcx, val: llbox}
      }
    };

    return {bcx: bcx, val: val};
}

type closure_result = {
    llbox: ValueRef,     // llvalue of ptr to closure
    cdata_ty: ty::t,      // type of the closure data
    bcx: block     // final bcx
};

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
fn store_environment(bcx: block,
                     bound_values: ~[environment_value],
                     ck: ty::closure_kind) -> closure_result {
    let _icx = bcx.insn_ctxt(~"closure::store_environment");
    let ccx = bcx.ccx(), tcx = ccx.tcx;

    // compute the shape of the closure
    let cdata_ty = mk_closure_tys(tcx, bound_values);

    // allocate closure in the heap
    let {bcx: bcx, val: llbox} = allocate_cbox(bcx, ck, cdata_ty);
    let mut temp_cleanups = ~[];

    // cbox_ty has the form of a tuple: (a, b, c) we want a ptr to a
    // tuple.  This could be a ptr in uniq or a box or on stack,
    // whatever.
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    let cboxptr_ty = ty::mk_ptr(tcx, {ty:cbox_ty, mutbl:ast::m_imm});

    let llbox = PointerCast(bcx, llbox, type_of(ccx, cboxptr_ty));
    debug!{"tuplify_box_ty = %s", ty_to_str(tcx, cbox_ty)};

    // Copy expr values into boxed bindings.
    let mut bcx = bcx;
    do vec::iteri(bound_values) |i, bv| {
        debug!{"Copy %s into closure", ev_to_str(ccx, bv)};

        if !ccx.sess.no_asm_comments() {
            add_comment(bcx, fmt!{"Copy %s into closure",
                                  ev_to_str(ccx, bv)});
        }

        let bound_data = GEPi(bcx, llbox,
             ~[0u, abi::box_field_body, i]);
        match bv {
          env_copy(val, ty, lv_owned) => {
            let val1 = load_if_immediate(bcx, val, ty);
            bcx = base::copy_val(bcx, INIT, bound_data, val1, ty);
          }
          env_copy(val, ty, lv_owned_imm) => {
            bcx = base::copy_val(bcx, INIT, bound_data, val, ty);
          }
          env_copy(_, _, lv_temporary) => {
            fail ~"cannot capture temporary upvar";
          }
          env_move(val, ty, kind) => {
            let src = {bcx:bcx, val:val, kind:kind};
            bcx = move_val(bcx, INIT, bound_data, src, ty);
          }
          env_ref(val, ty, lv_owned) => {
            debug!{"> storing %s into %s",
                   val_str(bcx.ccx().tn, val),
                   val_str(bcx.ccx().tn, bound_data)};
            Store(bcx, val, bound_data);
          }
          env_ref(val, ty, lv_owned_imm) => {
            let addr = do_spill_noroot(bcx, val);
            Store(bcx, addr, bound_data);
          }
          env_ref(_, _, lv_temporary) => {
            fail ~"cannot capture temporary upvar";
          }
        }
    }
    for vec::each(temp_cleanups) |cleanup| { revoke_clean(bcx, cleanup); }

    return {llbox: llbox, cdata_ty: cdata_ty, bcx: bcx};
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
fn build_closure(bcx0: block,
                 cap_vars: ~[capture::capture_var],
                 ck: ty::closure_kind,
                 id: ast::node_id,
                 include_ret_handle: option<ValueRef>) -> closure_result {
    let _icx = bcx0.insn_ctxt(~"closure::build_closure");
    // If we need to, package up the iterator body to call
    let mut env_vals = ~[];
    let mut bcx = bcx0;
    let ccx = bcx.ccx(), tcx = ccx.tcx;

    // Package up the captured upvars
    do vec::iter(cap_vars) |cap_var| {
        debug!{"Building closure: captured variable %?", cap_var};
        let lv = trans_local_var(bcx, cap_var.def);
        let nid = ast_util::def_id_of_def(cap_var.def).node;
        debug!{"Node id is %s",
               syntax::ast_map::node_id_to_str(bcx.ccx().tcx.items, nid)};
        let mut ty = node_id_type(bcx, nid);
        match cap_var.mode {
          capture::cap_ref => {
            assert ck == ty::ck_block;
            ty = ty::mk_mut_ptr(tcx, ty);
            vec::push(env_vals, env_ref(lv.val, ty, lv.kind));
          }
          capture::cap_copy => {
            let mv = match check ccx.maps.last_use_map.find(id) {
              none => false,
              some(vars) => (*vars).contains(nid)
            };
            if mv { vec::push(env_vals, env_move(lv.val, ty, lv.kind)); }
            else { vec::push(env_vals, env_copy(lv.val, ty, lv.kind)); }
          }
          capture::cap_move => {
            vec::push(env_vals, env_move(lv.val, ty, lv.kind));
          }
          capture::cap_drop => {
            assert lv.kind == lv_owned;
            bcx = drop_ty(bcx, lv.val, ty);
            bcx = zero_mem(bcx, lv.val, ty);
          }
        }
    }
    do option::iter(include_ret_handle) |flagptr| {
        let our_ret = match bcx.fcx.loop_ret {
          some({retptr, _}) => retptr,
          none => bcx.fcx.llretptr
        };
        let nil_ret = PointerCast(bcx, our_ret, T_ptr(T_nil()));
        vec::push(env_vals,
                  env_ref(flagptr,
                          ty::mk_mut_ptr(tcx, ty::mk_bool(tcx)), lv_owned));
        vec::push(env_vals,
                  env_ref(nil_ret, ty::mk_nil_ptr(tcx), lv_owned));
    }
    return store_environment(bcx, env_vals, ck);
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment(fcx: fn_ctxt,
                    cdata_ty: ty::t,
                    cap_vars: ~[capture::capture_var],
                    load_ret_handle: bool,
                    ck: ty::closure_kind) {
    let _icx = fcx.insn_ctxt(~"closure::load_environment");
    let bcx = raw_block(fcx, false, fcx.llloadenv);

    // Load a pointer to the closure data, skipping over the box header:
    let llcdata = base::opaque_box_body(bcx, cdata_ty, fcx.llenv);

    // Populate the upvars from the environment.
    let mut i = 0u;
    do vec::iter(cap_vars) |cap_var| {
        match cap_var.mode {
          capture::cap_drop => { /* ignore */ }
          _ => {
            let mut upvarptr =
                GEPi(bcx, llcdata, ~[0u, i]);
            match ck {
              ty::ck_block => { upvarptr = Load(bcx, upvarptr); }
              ty::ck_uniq | ty::ck_box => ()
            }
            let def_id = ast_util::def_id_of_def(cap_var.def);
            fcx.llupvars.insert(def_id.node, upvarptr);
            i += 1u;
          }
        }
    }
    if load_ret_handle {
        let flagptr = Load(bcx, GEPi(bcx, llcdata,
                                     ~[0u, i]));
        let retptr = Load(bcx,
                          GEPi(bcx, llcdata,
                               ~[0u, i+1u]));
        fcx.loop_ret = some({flagptr: flagptr, retptr: retptr});
    }
}

fn trans_expr_fn(bcx: block,
                 proto: ast::proto,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 id: ast::node_id,
                 cap_clause: ast::capture_clause,
                 is_loop_body: option<option<ValueRef>>,
                 dest: dest) -> block {
    let _icx = bcx.insn_ctxt(~"closure::trans_expr_fn");
    if dest == ignore { return bcx; }
    let ccx = bcx.ccx();
    let fty = node_id_type(bcx, id);
    let llfnty = type_of_fn_from_ty(ccx, fty);
    let sub_path = vec::append_one(bcx.fcx.path, path_name(@~"anon"));
    let s = mangle_internal_name_by_path(ccx, sub_path);
    let llfn = decl_internal_cdecl_fn(ccx.llmod, s, llfnty);

    let trans_closure_env = fn@(ck: ty::closure_kind) -> result {
        let cap_vars = capture::compute_capture_vars(
            ccx.tcx, id, proto, cap_clause);
        let ret_handle = match is_loop_body { some(x) => x, none => none };
        let {llbox, cdata_ty, bcx} = build_closure(bcx, cap_vars, ck, id,
                                                   ret_handle);
        trans_closure(ccx, sub_path, decl, body, llfn, no_self,
                      bcx.fcx.param_substs, id, |fcx| {
            load_environment(fcx, cdata_ty, cap_vars,
                             option::is_some(ret_handle), ck);
                      }, |bcx| {
            if option::is_some(is_loop_body) {
                Store(bcx, C_bool(true), bcx.fcx.llretptr);
            }
        });
        {bcx: bcx, val: llbox}
    };

    let {bcx: bcx, val: closure} = match proto {
      ast::proto_block => trans_closure_env(ty::ck_block),
      ast::proto_box => trans_closure_env(ty::ck_box),
      ast::proto_uniq => trans_closure_env(ty::ck_uniq),
      ast::proto_bare => {
        trans_closure(ccx, sub_path, decl, body, llfn, no_self, none,
                      id, |_fcx| { }, |_bcx| { });
        {bcx: bcx, val: C_null(T_opaque_box_ptr(ccx))}
      }
    };
    fill_fn_pair(bcx, get_dest_addr(dest), llfn, closure);

    return bcx;
}

fn make_fn_glue(
    cx: block,
    v: ValueRef,
    t: ty::t,
    glue_fn: fn@(block, v: ValueRef, t: ty::t) -> block)
    -> block {
    let _icx = cx.insn_ctxt(~"closure::make_fn_glue");
    let bcx = cx;
    let tcx = cx.tcx();

    let fn_env = fn@(ck: ty::closure_kind) -> block {
        let box_cell_v = GEPi(cx, v, ~[0u, abi::fn_field_box]);
        let box_ptr_v = Load(cx, box_cell_v);
        do with_cond(cx, IsNotNull(cx, box_ptr_v)) |bcx| {
            let closure_ty = ty::mk_opaque_closure_ptr(tcx, ck);
            glue_fn(bcx, box_cell_v, closure_ty)
        }
    };

    return match ty::get(t).struct {
      ty::ty_fn({proto: ast::proto_bare, _}) |
      ty::ty_fn({proto: ast::proto_block, _}) => bcx,
      ty::ty_fn({proto: ast::proto_uniq, _}) => fn_env(ty::ck_uniq),
      ty::ty_fn({proto: ast::proto_box, _}) => fn_env(ty::ck_box),
      _ => fail ~"make_fn_glue invoked on non-function type"
    };
}

fn make_opaque_cbox_take_glue(
    bcx: block,
    ck: ty::closure_kind,
    cboxptr: ValueRef)     // ptr to ptr to the opaque closure
    -> block {
    // Easy cases:
    let _icx = bcx.insn_ctxt(~"closure::make_opaque_cbox_take_glue");
    match ck {
      ty::ck_block => return bcx,
      ty::ck_box => {
        incr_refcnt_of_boxed(bcx, Load(bcx, cboxptr));
        return bcx;
      }
      ty::ck_uniq => { /* hard case: */ }
    }

    // Hard case, a deep copy:
    let ccx = bcx.ccx(), tcx = ccx.tcx;
    let llopaquecboxty = T_opaque_box_ptr(ccx);
    let cbox_in = Load(bcx, cboxptr);
    do with_cond(bcx, IsNotNull(bcx, cbox_in)) |bcx| {
        // Load the size from the type descr found in the cbox
        let cbox_in = PointerCast(bcx, cbox_in, llopaquecboxty);
        let tydescptr = GEPi(bcx, cbox_in, ~[0u, abi::box_field_tydesc]);
        let tydesc = Load(bcx, tydescptr);
        let tydesc = PointerCast(bcx, tydesc, T_ptr(ccx.tydesc_type));
        let sz = Load(bcx, GEPi(bcx, tydesc, ~[0u, abi::tydesc_field_size]));

        // Adjust sz to account for the rust_opaque_box header fields
        let sz = Add(bcx, sz, shape::llsize_of(ccx, T_box_header(ccx)));

        // Allocate memory, update original ptr, and copy existing data
        let malloc = ~"exchange_malloc";
        let opaque_tydesc = PointerCast(bcx, tydesc, T_ptr(T_i8()));
        let rval = alloca_zeroed(bcx, T_ptr(T_i8()));
        let bcx = trans_rtcall(bcx, malloc, ~[opaque_tydesc, sz],
                               save_in(rval));
        let cbox_out = PointerCast(bcx, Load(bcx, rval), llopaquecboxty);
        call_memmove(bcx, cbox_out, cbox_in, sz);
        Store(bcx, cbox_out, cboxptr);

        // Take the (deeply cloned) type descriptor
        let tydesc_out = GEPi(bcx, cbox_out, ~[0u, abi::box_field_tydesc]);
        let bcx = take_ty(bcx, tydesc_out, ty::mk_type(tcx));

        // Take the data in the tuple
        let cdata_out = GEPi(bcx, cbox_out, ~[0u, abi::box_field_body]);
        call_tydesc_glue_full(bcx, cdata_out, tydesc,
                              abi::tydesc_field_take_glue, none);
        bcx
    }
}

fn make_opaque_cbox_drop_glue(
    bcx: block,
    ck: ty::closure_kind,
    cboxptr: ValueRef)     // ptr to the opaque closure
    -> block {
    let _icx = bcx.insn_ctxt(~"closure::make_opaque_cbox_drop_glue");
    match ck {
      ty::ck_block => bcx,
      ty::ck_box => {
        decr_refcnt_maybe_free(bcx, Load(bcx, cboxptr),
                               ty::mk_opaque_closure_ptr(bcx.tcx(), ck))
      }
      ty::ck_uniq => {
        free_ty(bcx, cboxptr,
                ty::mk_opaque_closure_ptr(bcx.tcx(), ck))
      }
    }
}

fn make_opaque_cbox_free_glue(
    bcx: block,
    ck: ty::closure_kind,
    cbox: ValueRef)     // ptr to ptr to the opaque closure
    -> block {
    let _icx = bcx.insn_ctxt(~"closure::make_opaque_cbox_free_glue");
    match ck {
      ty::ck_block => return bcx,
      ty::ck_box | ty::ck_uniq => { /* hard cases: */ }
    }

    let ccx = bcx.ccx();
    do with_cond(bcx, IsNotNull(bcx, cbox)) |bcx| {
        // Load the type descr found in the cbox
        let lltydescty = T_ptr(ccx.tydesc_type);
        let cbox = Load(bcx, cbox);
        let tydescptr = GEPi(bcx, cbox, ~[0u, abi::box_field_tydesc]);
        let tydesc = Load(bcx, tydescptr);
        let tydesc = PointerCast(bcx, tydesc, lltydescty);

        // Drop the tuple data then free the descriptor
        let cdata = GEPi(bcx, cbox, ~[0u, abi::box_field_body]);
        call_tydesc_glue_full(bcx, cdata, tydesc,
                              abi::tydesc_field_drop_glue, none);

        // Free the ty descr (if necc) and the box itself
        match ck {
          ty::ck_block => fail ~"Impossible",
          ty::ck_box => trans_free(bcx, cbox),
          ty::ck_uniq => trans_unique_free(bcx, cbox)
        }
    }
}

