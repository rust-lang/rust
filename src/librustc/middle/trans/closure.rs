// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use back::link::{mangle_internal_name_by_path_and_seq};
use lib::llvm::ValueRef;
use middle::moves;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::datum::{Datum, INIT, ByRef, ZeroMem};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::type_of::*;
use middle::ty;
use util::ppaux::ty_to_str;

use middle::trans::type_::Type;

use std::vec;
use syntax::ast;
use syntax::ast_map::path_name;
use syntax::ast_util;
use syntax::parse::token::special_idents;

// ___Good to know (tm)__________________________________________________
//
// The layout of a closure environment in memory is
// roughly as follows:
//
// struct rust_opaque_box {         // see rust_internal.h
//   unsigned ref_count;            // only used for @fn()
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
// even for ~fn and &fn, because we wish to keep binary compatibility
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
// by ptr.  The routine Type::opaque_box().ptr_to() returns an
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

pub enum EnvAction {
    /// Copy the value from this llvm ValueRef into the environment.
    EnvCopy,

    /// Move the value from this llvm ValueRef into the environment.
    EnvMove,

    /// Access by reference (used for stack closures).
    EnvRef
}

pub struct EnvValue {
    action: EnvAction,
    datum: Datum
}

impl EnvAction {
    pub fn to_str(&self) -> ~str {
        match *self {
            EnvCopy => ~"EnvCopy",
            EnvMove => ~"EnvMove",
            EnvRef => ~"EnvRef"
        }
    }
}

impl EnvValue {
    pub fn to_str(&self, ccx: &CrateContext) -> ~str {
        fmt!("%s(%s)", self.action.to_str(), self.datum.to_str(ccx))
    }
}

pub fn mk_tuplified_uniq_cbox_ty(tcx: ty::ctxt, cdata_ty: ty::t) -> ty::t {
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    return ty::mk_imm_uniq(tcx, cbox_ty);
}

// Given a closure ty, emits a corresponding tuple ty
pub fn mk_closure_tys(tcx: ty::ctxt,
                      bound_values: &[EnvValue])
                   -> ty::t {
    // determine the types of the values in the env.  Note that this
    // is the actual types that will be stored in the map, not the
    // logical types as the user sees them, so by-ref upvars must be
    // converted to ptrs.
    let bound_tys = bound_values.map(|bv| {
        match bv.action {
            EnvCopy | EnvMove => bv.datum.ty,
            EnvRef => ty::mk_mut_ptr(tcx, bv.datum.ty)
        }
    });
    let cdata_ty = ty::mk_tup(tcx, bound_tys);
    debug!("cdata_ty=%s", ty_to_str(tcx, cdata_ty));
    return cdata_ty;
}

fn heap_for_unique_closure(bcx: block, t: ty::t) -> heap {
    if ty::type_contents(bcx.tcx(), t).contains_managed() {
        heap_managed_unique
    } else {
        heap_exchange_closure
    }
}

pub fn allocate_cbox(bcx: block, sigil: ast::Sigil, cdata_ty: ty::t)
                  -> Result {
    let _icx = push_ctxt("closure::allocate_cbox");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    // Allocate and initialize the box:
    match sigil {
        ast::ManagedSigil => {
            malloc_raw(bcx, cdata_ty, heap_managed)
        }
        ast::OwnedSigil => {
            malloc_raw(bcx, cdata_ty, heap_for_unique_closure(bcx, cdata_ty))
        }
        ast::BorrowedSigil => {
            let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
            let llbox = alloc_ty(bcx, cbox_ty, "__closure");
            rslt(bcx, llbox)
        }
    }
}

pub struct ClosureResult {
    llbox: ValueRef, // llvalue of ptr to closure
    cdata_ty: ty::t, // type of the closure data
    bcx: block       // final bcx
}

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
pub fn store_environment(bcx: block,
                         bound_values: ~[EnvValue],
                         sigil: ast::Sigil) -> ClosureResult {
    let _icx = push_ctxt("closure::store_environment");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    // compute the type of the closure
    let cdata_ty = mk_closure_tys(tcx, bound_values);

    // cbox_ty has the form of a tuple: (a, b, c) we want a ptr to a
    // tuple.  This could be a ptr in uniq or a box or on stack,
    // whatever.
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    let cboxptr_ty = ty::mk_ptr(tcx, ty::mt {ty:cbox_ty, mutbl:ast::m_imm});
    let llboxptr_ty = type_of(ccx, cboxptr_ty);

    // If there are no bound values, no point in allocating anything.
    if bound_values.is_empty() {
        return ClosureResult {llbox: C_null(llboxptr_ty),
                              cdata_ty: cdata_ty,
                              bcx: bcx};
    }

    // allocate closure in the heap
    let Result {bcx: bcx, val: llbox} = allocate_cbox(bcx, sigil, cdata_ty);

    let llbox = PointerCast(bcx, llbox, llboxptr_ty);
    debug!("tuplify_box_ty = %s", ty_to_str(tcx, cbox_ty));

    // Copy expr values into boxed bindings.
    let mut bcx = bcx;
    for bound_values.iter().enumerate().advance |(i, bv)| {
        debug!("Copy %s into closure", bv.to_str(ccx));

        if ccx.sess.asm_comments() {
            add_comment(bcx, fmt!("Copy %s into closure",
                                  bv.to_str(ccx)));
        }

        let bound_data = GEPi(bcx, llbox, [0u, abi::box_field_body, i]);

        match bv.action {
            EnvCopy => {
                bcx = bv.datum.copy_to(bcx, INIT, bound_data);
            }
            EnvMove => {
                bcx = bv.datum.move_to(bcx, INIT, bound_data);
            }
            EnvRef => {
                Store(bcx, bv.datum.to_ref_llval(bcx), bound_data);
            }
        }

    }

    ClosureResult { llbox: llbox, cdata_ty: cdata_ty, bcx: bcx }
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
pub fn build_closure(bcx0: block,
                     cap_vars: &[moves::CaptureVar],
                     sigil: ast::Sigil,
                     include_ret_handle: Option<ValueRef>) -> ClosureResult {
    let _icx = push_ctxt("closure::build_closure");

    // If we need to, package up the iterator body to call
    let bcx = bcx0;

    // Package up the captured upvars
    let mut env_vals = ~[];
    for cap_vars.iter().advance |cap_var| {
        debug!("Building closure: captured variable %?", *cap_var);
        let datum = expr::trans_local_var(bcx, cap_var.def);
        match cap_var.mode {
            moves::CapRef => {
                assert_eq!(sigil, ast::BorrowedSigil);
                env_vals.push(EnvValue {action: EnvRef,
                                        datum: datum});
            }
            moves::CapCopy => {
                env_vals.push(EnvValue {action: EnvCopy,
                                        datum: datum});
            }
            moves::CapMove => {
                env_vals.push(EnvValue {action: EnvMove,
                                        datum: datum});
            }
        }
    }

    // If this is a `for` loop body, add two special environment
    // variables:
    for include_ret_handle.iter().advance |flagptr| {
        // Flag indicating we have returned (a by-ref bool):
        let flag_datum = Datum {val: *flagptr, ty: ty::mk_bool(),
                                mode: ByRef(ZeroMem)};
        env_vals.push(EnvValue {action: EnvRef,
                                datum: flag_datum});

        // Return value (we just pass a by-ref () and cast it later to
        // the right thing):
        let ret_true = match bcx.fcx.loop_ret {
            Some((_, retptr)) => retptr,
            None => match bcx.fcx.llretptr {
                None => C_null(Type::nil().ptr_to()),
                Some(retptr) => PointerCast(bcx, retptr, Type::nil().ptr_to()),
            }
        };
        let ret_datum = Datum {val: ret_true, ty: ty::mk_nil(),
                               mode: ByRef(ZeroMem)};
        env_vals.push(EnvValue {action: EnvRef,
                                datum: ret_datum});
    }

    return store_environment(bcx, env_vals, sigil);
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
pub fn load_environment(fcx: fn_ctxt,
                        cdata_ty: ty::t,
                        cap_vars: &[moves::CaptureVar],
                        load_ret_handle: bool,
                        sigil: ast::Sigil) {
    let _icx = push_ctxt("closure::load_environment");

    // Don't bother to create the block if there's nothing to load
    if cap_vars.len() == 0 && !load_ret_handle {
        return;
    }

    let bcx = fcx.entry_bcx.get();

    // Load a pointer to the closure data, skipping over the box header:
    let llcdata = opaque_box_body(bcx, cdata_ty, fcx.llenv);

    // Populate the upvars from the environment.
    let mut i = 0u;
    for cap_vars.iter().advance |cap_var| {
        let mut upvarptr = GEPi(bcx, llcdata, [0u, i]);
        match sigil {
            ast::BorrowedSigil => { upvarptr = Load(bcx, upvarptr); }
            ast::ManagedSigil | ast::OwnedSigil => {}
        }
        let def_id = ast_util::def_id_of_def(cap_var.def);
        fcx.llupvars.insert(def_id.node, upvarptr);
        i += 1u;
    }
    if load_ret_handle {
        let flagptr = Load(bcx, GEPi(bcx, llcdata, [0u, i]));
        let retptr = Load(bcx,
                          GEPi(bcx, llcdata, [0u, i+1u]));
        fcx.loop_ret = Some((flagptr, retptr));
    }
}

pub fn trans_expr_fn(bcx: block,
                     sigil: ast::Sigil,
                     decl: &ast::fn_decl,
                     body: &ast::Block,
                     outer_id: ast::node_id,
                     user_id: ast::node_id,
                     is_loop_body: Option<Option<ValueRef>>,
                     dest: expr::Dest) -> block {
    /*!
     *
     * Translates the body of a closure expression.
     *
     * - `sigil`
     * - `decl`
     * - `body`
     * - `outer_id`: The id of the closure expression with the correct
     *   type.  This is usually the same as `user_id`, but in the
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

    let _icx = push_ctxt("closure::trans_expr_fn");

    let dest_addr = match dest {
        expr::SaveIn(p) => p,
        expr::Ignore => {
            return bcx; // closure construction is non-side-effecting
        }
    };

    let ccx = bcx.ccx();
    let fty = node_id_type(bcx, outer_id);

    let llfnty = type_of_fn_from_ty(ccx, fty);

    let sub_path = vec::append_one(bcx.fcx.path.clone(),
                                   path_name(special_idents::anon));
    // XXX: Bad copy.
    let s = mangle_internal_name_by_path_and_seq(ccx,
                                                 sub_path.clone(),
                                                 "expr_fn");
    let llfn = decl_internal_cdecl_fn(ccx.llmod, s, llfnty);

    // Always mark inline if this is a loop body. This is important for
    // performance on many programs with tight loops.
    if is_loop_body.is_some() {
        set_always_inline(llfn);
    } else {
        // Can't hurt.
        set_inline_hint(llfn);
    }

    let real_return_type = if is_loop_body.is_some() {
        ty::mk_bool()
    } else {
        ty::ty_fn_ret(fty)
    };

    let Result {bcx: bcx, val: closure} = match sigil {
        ast::BorrowedSigil | ast::ManagedSigil | ast::OwnedSigil => {
            let cap_vars = ccx.maps.capture_map.get_copy(&user_id);
            let ret_handle = match is_loop_body {Some(x) => x,
                                                 None => None};
            let ClosureResult {llbox, cdata_ty, bcx}
                = build_closure(bcx, cap_vars, sigil, ret_handle);
            trans_closure(ccx,
                          sub_path,
                          decl,
                          body,
                          llfn,
                          no_self,
                          bcx.fcx.param_substs,
                          user_id,
                          [],
                          real_return_type,
                          |fcx| load_environment(fcx, cdata_ty, cap_vars,
                                                 ret_handle.is_some(), sigil),
                          |bcx| {
                              if is_loop_body.is_some() {
                                  Store(bcx,
                                        C_bool(true),
                                        bcx.fcx.llretptr.get());
                              }
                          });
            rslt(bcx, llbox)
        }
    };
    fill_fn_pair(bcx, dest_addr, llfn, closure);

    return bcx;
}

pub fn make_closure_glue(
        cx: block,
        v: ValueRef,
        t: ty::t,
        glue_fn: &fn(block, v: ValueRef, t: ty::t) -> block) -> block {
    let _icx = push_ctxt("closure::make_closure_glue");
    let bcx = cx;
    let tcx = cx.tcx();

    let sigil = ty::ty_closure_sigil(t);
    match sigil {
        ast::BorrowedSigil => bcx,
        ast::OwnedSigil | ast::ManagedSigil => {
            let box_cell_v = GEPi(cx, v, [0u, abi::fn_field_box]);
            let box_ptr_v = Load(cx, box_cell_v);
            do with_cond(cx, IsNotNull(cx, box_ptr_v)) |bcx| {
                let closure_ty = ty::mk_opaque_closure_ptr(tcx, sigil);
                glue_fn(bcx, box_cell_v, closure_ty)
            }
        }
    }
}

pub fn make_opaque_cbox_take_glue(
    bcx: block,
    sigil: ast::Sigil,
    cboxptr: ValueRef)     // ptr to ptr to the opaque closure
    -> block {
    // Easy cases:
    let _icx = push_ctxt("closure::make_opaque_cbox_take_glue");
    match sigil {
        ast::BorrowedSigil => {
            return bcx;
        }
        ast::ManagedSigil => {
            glue::incr_refcnt_of_boxed(bcx, Load(bcx, cboxptr));
            return bcx;
        }
        ast::OwnedSigil => {
            fail!("unique closures are not copyable")
        }
    }
}

pub fn make_opaque_cbox_drop_glue(
    bcx: block,
    sigil: ast::Sigil,
    cboxptr: ValueRef)     // ptr to the opaque closure
    -> block {
    let _icx = push_ctxt("closure::make_opaque_cbox_drop_glue");
    match sigil {
        ast::BorrowedSigil => bcx,
        ast::ManagedSigil => {
            glue::decr_refcnt_maybe_free(
                bcx, Load(bcx, cboxptr), Some(cboxptr),
                ty::mk_opaque_closure_ptr(bcx.tcx(), sigil))
        }
        ast::OwnedSigil => {
            glue::free_ty(
                bcx, cboxptr,
                ty::mk_opaque_closure_ptr(bcx.tcx(), sigil))
        }
    }
}

pub fn make_opaque_cbox_free_glue(
    bcx: block,
    sigil: ast::Sigil,
    cbox: ValueRef)     // ptr to ptr to the opaque closure
    -> block {
    let _icx = push_ctxt("closure::make_opaque_cbox_free_glue");
    match sigil {
        ast::BorrowedSigil => {
            return bcx;
        }
        ast::ManagedSigil | ast::OwnedSigil => {
            /* hard cases: fallthrough to code below */
        }
    }

    let ccx = bcx.ccx();
    do with_cond(bcx, IsNotNull(bcx, cbox)) |bcx| {
        // Load the type descr found in the cbox
        let lltydescty = ccx.tydesc_type.ptr_to();
        let cbox = Load(bcx, cbox);
        let tydescptr = GEPi(bcx, cbox, [0u, abi::box_field_tydesc]);
        let tydesc = Load(bcx, tydescptr);
        let tydesc = PointerCast(bcx, tydesc, lltydescty);

        // Drop the tuple data then free the descriptor
        let cdata = GEPi(bcx, cbox, [0u, abi::box_field_body]);
        glue::call_tydesc_glue_full(bcx, cdata, tydesc,
                                    abi::tydesc_field_drop_glue, None);

        // Free the ty descr (if necc) and the box itself
        match sigil {
            ast::ManagedSigil => glue::trans_free(bcx, cbox),
            ast::OwnedSigil => glue::trans_exchange_free(bcx, cbox),
            ast::BorrowedSigil => {
                bcx.sess().bug("impossible")
            }
        }
    }
}
