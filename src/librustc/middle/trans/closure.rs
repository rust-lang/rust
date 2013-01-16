// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use back::abi;
use back::link::{mangle_internal_name_by_path_and_seq};
use back::link::{mangle_internal_name_by_path};
use lib::llvm::llvm;
use lib::llvm::{ValueRef, TypeRef};
use middle::capture;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::datum::{Datum, INIT, ByRef, ByValue, FromLvalue};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::type_of::*;
use util::ppaux::ty_to_str;

use core::libc::c_uint;
use std::map::HashMap;
use syntax::ast;
use syntax::ast_map::{path, path_mod, path_name};
use syntax::ast_util;
use syntax::codemap::span;
use syntax::print::pprust::expr_to_str;

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

enum EnvAction {
    /// Copy the value from this llvm ValueRef into the environment.
    EnvStore,

    /// Move the value from this llvm ValueRef into the environment.
    EnvMove,

    /// Access by reference (used for stack closures).
    EnvRef
}

struct EnvValue {
    action: EnvAction,
    datum: Datum
}

impl EnvAction {
    fn to_str() -> ~str {
        match self {
            EnvStore => ~"EnvStore",
            EnvMove => ~"EnvMove",
            EnvRef => ~"EnvRef"
        }
    }
}

impl EnvValue {
    fn to_str(ccx: @crate_ctxt) -> ~str {
        fmt!("%s(%s)", self.action.to_str(), self.datum.to_str(ccx))
    }
}

fn mk_tuplified_uniq_cbox_ty(tcx: ty::ctxt, cdata_ty: ty::t) -> ty::t {
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    return ty::mk_imm_uniq(tcx, cbox_ty);
}

// Given a closure ty, emits a corresponding tuple ty
fn mk_closure_tys(tcx: ty::ctxt,
                  bound_values: ~[EnvValue])
    -> ty::t {
    // determine the types of the values in the env.  Note that this
    // is the actual types that will be stored in the map, not the
    // logical types as the user sees them, so by-ref upvars must be
    // converted to ptrs.
    let bound_tys = bound_values.map(|bv| {
        match bv.action {
            EnvStore | EnvMove => bv.datum.ty,
            EnvRef => ty::mk_mut_ptr(tcx, bv.datum.ty)
        }
    });
    let cdata_ty = ty::mk_tup(tcx, bound_tys);
    debug!("cdata_ty=%s", ty_to_str(tcx, cdata_ty));
    return cdata_ty;
}

fn allocate_cbox(bcx: block, proto: ast::Proto, cdata_ty: ty::t)
    -> Result
{
    let _icx = bcx.insn_ctxt("closure::allocate_cbox");
    let ccx = bcx.ccx(), tcx = ccx.tcx;

    fn nuke_ref_count(bcx: block, llbox: ValueRef) {
        let _icx = bcx.insn_ctxt("closure::nuke_ref_count");
        // Initialize ref count to arbitrary value for debugging:
        let ccx = bcx.ccx();
        let llbox = PointerCast(bcx, llbox, T_opaque_box_ptr(ccx));
        let ref_cnt = GEPi(bcx, llbox, [0u, abi::box_field_refcnt]);
        let rc = C_int(ccx, 0x12345678);
        Store(bcx, rc, ref_cnt);
    }

    // Allocate and initialize the box:
    match proto {
        ast::ProtoBox => {
            malloc_raw(bcx, cdata_ty, heap_shared)
        }
        ast::ProtoUniq => {
            malloc_raw(bcx, cdata_ty, heap_exchange)
        }
        ast::ProtoBorrowed => {
            let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
            let llbox = base::alloc_ty(bcx, cbox_ty);
            nuke_ref_count(bcx, llbox);
            rslt(bcx, llbox)
        }
        ast::ProtoBare => {
            let cdata_llty = type_of(bcx.ccx(), cdata_ty);
            rslt(bcx, C_null(cdata_llty))
        }
    }
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
                     bound_values: ~[EnvValue],
                     proto: ast::Proto) -> closure_result {
    let _icx = bcx.insn_ctxt("closure::store_environment");
    let ccx = bcx.ccx(), tcx = ccx.tcx;

    // compute the shape of the closure
    // XXX: Bad copy.
    let cdata_ty = mk_closure_tys(tcx, copy bound_values);

    // allocate closure in the heap
    let Result {bcx: bcx, val: llbox} = allocate_cbox(bcx, proto, cdata_ty);
    let mut temp_cleanups = ~[];

    // cbox_ty has the form of a tuple: (a, b, c) we want a ptr to a
    // tuple.  This could be a ptr in uniq or a box or on stack,
    // whatever.
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    let cboxptr_ty = ty::mk_ptr(tcx, ty::mt {ty:cbox_ty, mutbl:ast::m_imm});

    let llbox = PointerCast(bcx, llbox, type_of(ccx, cboxptr_ty));
    debug!("tuplify_box_ty = %s", ty_to_str(tcx, cbox_ty));

    // Copy expr values into boxed bindings.
    let mut bcx = bcx;
    for vec::eachi(bound_values) |i, bv| {
        debug!("Copy %s into closure", bv.to_str(ccx));

        if !ccx.sess.no_asm_comments() {
            add_comment(bcx, fmt!("Copy %s into closure",
                                  bv.to_str(ccx)));
        }

        let bound_data = GEPi(bcx, llbox, [0u, abi::box_field_body, i]);

        match bv.action {
            EnvStore => {
                bcx = bv.datum.store_to(bcx, INIT, bound_data);
            }
            EnvMove => {
                bcx = bv.datum.move_to(bcx, INIT, bound_data);
            }
            EnvRef => {
                Store(bcx, bv.datum.to_ref_llval(bcx), bound_data);
            }
        }

    }
    for vec::each(temp_cleanups) |cleanup| {
        revoke_clean(bcx, *cleanup);
    }

    return {llbox: llbox, cdata_ty: cdata_ty, bcx: bcx};
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
fn build_closure(bcx0: block,
                 cap_vars: ~[capture::capture_var],
                 proto: ast::Proto,
                 include_ret_handle: Option<ValueRef>) -> closure_result {
    let _icx = bcx0.insn_ctxt("closure::build_closure");
    // If we need to, package up the iterator body to call
    let mut bcx = bcx0;;
    let ccx = bcx.ccx(), tcx = ccx.tcx;

    // Package up the captured upvars
    let mut env_vals = ~[];
    for vec::each(cap_vars) |cap_var| {
        debug!("Building closure: captured variable %?", *cap_var);
        let datum = expr::trans_local_var(bcx, cap_var.def, None);
        match cap_var.mode {
            capture::cap_ref => {
                assert proto == ast::ProtoBorrowed;
                env_vals.push(EnvValue {action: EnvRef,
                                        datum: datum});
            }
            capture::cap_copy => {
                env_vals.push(EnvValue {action: EnvStore,
                                        datum: datum});
            }
            capture::cap_move => {
                env_vals.push(EnvValue {action: EnvMove,
                                        datum: datum});
            }
            capture::cap_drop => {
                bcx = datum.drop_val(bcx);
                datum.cancel_clean(bcx);
            }
        }
    }

    // If this is a `for` loop body, add two special environment
    // variables:
    do option::iter(&include_ret_handle) |flagptr| {
        // Flag indicating we have returned (a by-ref bool):
        let flag_datum = Datum {val: *flagptr, ty: ty::mk_bool(tcx),
                                mode: ByRef, source: FromLvalue};
        env_vals.push(EnvValue {action: EnvRef,
                                datum: flag_datum});

        // Return value (we just pass a by-ref () and cast it later to
        // the right thing):
        let ret_true = match bcx.fcx.loop_ret {
            Some({retptr, _}) => retptr,
            None => bcx.fcx.llretptr
        };
        let ret_casted = PointerCast(bcx, ret_true, T_ptr(T_nil()));
        let ret_datum = Datum {val: ret_casted, ty: ty::mk_nil(tcx),
                               mode: ByRef, source: FromLvalue};
        env_vals.push(EnvValue {action: EnvRef,
                                datum: ret_datum});
    }

    return store_environment(bcx, env_vals, proto);
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment(fcx: fn_ctxt,
                    cdata_ty: ty::t,
                    cap_vars: ~[capture::capture_var],
                    load_ret_handle: bool,
                    proto: ast::Proto) {
    let _icx = fcx.insn_ctxt("closure::load_environment");

    let llloadenv = match fcx.llloadenv {
        Some(ll) => ll,
        None => {
            let ll =
                str::as_c_str(~"load_env",
                              |buf|
                              unsafe {
                                llvm::LLVMAppendBasicBlock(fcx.llfn, buf)
                              });
            fcx.llloadenv = Some(ll);
            ll
        }
    };

    let bcx = raw_block(fcx, false, llloadenv);

    // Load a pointer to the closure data, skipping over the box header:
    let llcdata = base::opaque_box_body(bcx, cdata_ty, fcx.llenv);

    // Populate the upvars from the environment.
    let mut i = 0u;
    for vec::each(cap_vars) |cap_var| {
        match cap_var.mode {
          capture::cap_drop => { /* ignore */ }
          _ => {
            let mut upvarptr = GEPi(bcx, llcdata, [0u, i]);
            match proto {
                ast::ProtoBorrowed => { upvarptr = Load(bcx, upvarptr); }
                ast::ProtoBox | ast::ProtoUniq | ast::ProtoBare => {}
            }
            let def_id = ast_util::def_id_of_def(cap_var.def);
            fcx.llupvars.insert(def_id.node, upvarptr);
            i += 1u;
          }
        }
    }
    if load_ret_handle {
        let flagptr = Load(bcx, GEPi(bcx, llcdata, [0u, i]));
        let retptr = Load(bcx,
                          GEPi(bcx, llcdata, [0u, i+1u]));
        fcx.loop_ret = Some({flagptr: flagptr, retptr: retptr});
    }
}

fn trans_expr_fn(bcx: block,
                 proto: ast::Proto,
                 +decl: ast::fn_decl,
                 +body: ast::blk,
                 outer_id: ast::node_id,
                 user_id: ast::node_id,
                 cap_clause: ast::capture_clause,
                 is_loop_body: Option<Option<ValueRef>>,
                 dest: expr::Dest) -> block
{
    /*!
     *
     * Translates the body of a closure expression.
     *
     * - `proto`
     * - `decl`
     * - `body`
     * - `outer_id`: The id of the closure expression with the correct
     *   type.  This is usually the same as as `user_id`, but in the
     *   case of a `for` loop, the `outer_id` will have the return
     *   type of boolean, and the `user_id` will have the return type
     *   of `nil`.
     * - `user_id`: The id of the closure as the user expressed it.
         Generally the same as `outer_id`
     * - `cap_clause`: information about captured variables, if any.
     * - `is_loop_body`: `Some()` if this is part of a `for` loop.
     * - `dest`: where to write the closure value, which must be a
         (fn ptr, env) pair
     */

    let _icx = bcx.insn_ctxt("closure::trans_expr_fn");

    let dest_addr = match dest {
        expr::SaveIn(p) => p,
        expr::Ignore => {
            return bcx; // closure construction is non-side-effecting
        }
    };

    let ccx = bcx.ccx();
    let fty = node_id_type(bcx, outer_id);
    let llfnty = type_of_fn_from_ty(ccx, fty);
    let sub_path = vec::append_one(/*bad*/copy bcx.fcx.path,
                                   path_name(special_idents::anon));
    // XXX: Bad copy.
    let s = mangle_internal_name_by_path_and_seq(ccx,
                                                 copy sub_path,
                                                 ~"expr_fn");
    let llfn = decl_internal_cdecl_fn(ccx.llmod, s, llfnty);

    let trans_closure_env: &fn(ast::Proto) -> Result = |proto| {
        let cap_vars = capture::compute_capture_vars(ccx.tcx, user_id, proto,
                                                     cap_clause);
        let ret_handle = match is_loop_body { Some(x) => x, None => None };
        // XXX: Bad copy.
        let {llbox, cdata_ty, bcx} = build_closure(bcx, copy cap_vars, proto,
                                                   ret_handle);
        trans_closure(ccx, /*bad*/copy sub_path, decl, /*bad*/copy body,
                      llfn, no_self, /*bad*/copy bcx.fcx.param_substs,
                      user_id, None, |fcx| {
                          load_environment(fcx, cdata_ty, copy cap_vars,
                                           ret_handle.is_some(), proto);
                      }, |bcx| {
                          if is_loop_body.is_some() {
                              Store(bcx, C_bool(true), bcx.fcx.llretptr);
                          }
                      });
        rslt(bcx, llbox)
    };

    let Result {bcx: bcx, val: closure} = match proto {
        ast::ProtoBorrowed | ast::ProtoBox | ast::ProtoUniq => {
            trans_closure_env(proto)
        }
        ast::ProtoBare => {
            trans_closure(ccx, sub_path, decl, body, llfn, no_self, None,
                          user_id, None, |_fcx| { }, |_bcx| { });
            rslt(bcx, C_null(T_opaque_box_ptr(ccx)))
        }
    };
    fill_fn_pair(bcx, dest_addr, llfn, closure);

    return bcx;
}

fn make_fn_glue(
    cx: block,
    v: ValueRef,
    t: ty::t,
    glue_fn: fn@(block, v: ValueRef, t: ty::t) -> block)
    -> block
{
    let _icx = cx.insn_ctxt("closure::make_fn_glue");
    let bcx = cx;
    let tcx = cx.tcx();

    let proto = ty::ty_fn_proto(t);
    match proto {
        ast::ProtoBare | ast::ProtoBorrowed => bcx,
        ast::ProtoUniq | ast::ProtoBox => {
            let box_cell_v = GEPi(cx, v, [0u, abi::fn_field_box]);
            let box_ptr_v = Load(cx, box_cell_v);
            do with_cond(cx, IsNotNull(cx, box_ptr_v)) |bcx| {
                let closure_ty = ty::mk_opaque_closure_ptr(tcx, proto);
                glue_fn(bcx, box_cell_v, closure_ty)
            }
        }
    }
}

fn make_opaque_cbox_take_glue(
    bcx: block,
    proto: ast::Proto,
    cboxptr: ValueRef)     // ptr to ptr to the opaque closure
    -> block
{
    // Easy cases:
    let _icx = bcx.insn_ctxt("closure::make_opaque_cbox_take_glue");
    match proto {
        ast::ProtoBare | ast::ProtoBorrowed => {
            return bcx;
        }
        ast::ProtoBox => {
            glue::incr_refcnt_of_boxed(bcx, Load(bcx, cboxptr));
            return bcx;
        }
        ast::ProtoUniq => {
            /* hard case: fallthrough to code below */
        }
    }

    // fn~ requires a deep copy.
    let ccx = bcx.ccx(), tcx = ccx.tcx;
    let llopaquecboxty = T_opaque_box_ptr(ccx);
    let cbox_in = Load(bcx, cboxptr);
    do with_cond(bcx, IsNotNull(bcx, cbox_in)) |bcx| {
        // Load the size from the type descr found in the cbox
        let cbox_in = PointerCast(bcx, cbox_in, llopaquecboxty);
        let tydescptr = GEPi(bcx, cbox_in, [0u, abi::box_field_tydesc]);
        let tydesc = Load(bcx, tydescptr);
        let tydesc = PointerCast(bcx, tydesc, T_ptr(ccx.tydesc_type));
        let sz = Load(bcx, GEPi(bcx, tydesc, [0u, abi::tydesc_field_size]));

        // Adjust sz to account for the rust_opaque_box header fields
        let sz = Add(bcx, sz, shape::llsize_of(ccx, T_box_header(ccx)));

        // Allocate memory, update original ptr, and copy existing data
        let opaque_tydesc = PointerCast(bcx, tydesc, T_ptr(T_i8()));
        let rval = alloca_zeroed(bcx, T_ptr(T_i8()));
        let bcx = callee::trans_rtcall_or_lang_call(
            bcx,
            bcx.tcx().lang_items.exchange_malloc_fn(),
            ~[opaque_tydesc, sz],
            expr::SaveIn(rval));
        let cbox_out = PointerCast(bcx, Load(bcx, rval), llopaquecboxty);
        call_memcpy(bcx, cbox_out, cbox_in, sz);
        Store(bcx, cbox_out, cboxptr);

        // Take the (deeply cloned) type descriptor
        let tydesc_out = GEPi(bcx, cbox_out, [0u, abi::box_field_tydesc]);
        let bcx = glue::take_ty(bcx, tydesc_out, ty::mk_type(tcx));

        // Take the data in the tuple
        let cdata_out = GEPi(bcx, cbox_out, [0u, abi::box_field_body]);
        glue::call_tydesc_glue_full(bcx, cdata_out, tydesc,
                                    abi::tydesc_field_take_glue, None);
        bcx
    }
}

fn make_opaque_cbox_drop_glue(
    bcx: block,
    proto: ast::Proto,
    cboxptr: ValueRef)     // ptr to the opaque closure
    -> block {
    let _icx = bcx.insn_ctxt("closure::make_opaque_cbox_drop_glue");
    match proto {
        ast::ProtoBare | ast::ProtoBorrowed => bcx,
        ast::ProtoBox => {
            glue::decr_refcnt_maybe_free(
                bcx, Load(bcx, cboxptr),
                ty::mk_opaque_closure_ptr(bcx.tcx(), proto))
        }
        ast::ProtoUniq => {
            glue::free_ty(
                bcx, cboxptr,
                ty::mk_opaque_closure_ptr(bcx.tcx(), proto))
        }
    }
}

fn make_opaque_cbox_free_glue(
    bcx: block,
    proto: ast::Proto,
    cbox: ValueRef)     // ptr to ptr to the opaque closure
    -> block {
    let _icx = bcx.insn_ctxt("closure::make_opaque_cbox_free_glue");
    match proto {
        ast::ProtoBare | ast::ProtoBorrowed => {
            return bcx;
        }
        ast::ProtoBox | ast::ProtoUniq => {
            /* hard cases: fallthrough to code below */
        }
    }

    let ccx = bcx.ccx();
    do with_cond(bcx, IsNotNull(bcx, cbox)) |bcx| {
        // Load the type descr found in the cbox
        let lltydescty = T_ptr(ccx.tydesc_type);
        let cbox = Load(bcx, cbox);
        let tydescptr = GEPi(bcx, cbox, [0u, abi::box_field_tydesc]);
        let tydesc = Load(bcx, tydescptr);
        let tydesc = PointerCast(bcx, tydesc, lltydescty);

        // Drop the tuple data then free the descriptor
        let cdata = GEPi(bcx, cbox, [0u, abi::box_field_body]);
        glue::call_tydesc_glue_full(bcx, cdata, tydesc,
                                    abi::tydesc_field_drop_glue, None);

        // Free the ty descr (if necc) and the box itself
        match proto {
            ast::ProtoBox => glue::trans_free(bcx, cbox),
            ast::ProtoUniq => glue::trans_unique_free(bcx, cbox),
            ast::ProtoBare | ast::ProtoBorrowed => {
                bcx.sess().bug(~"impossible")
            }
        }
    }
}

