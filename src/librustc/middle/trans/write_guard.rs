// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Logic relating to rooting and write guards for managed values
//! (`@` and `@mut`). This code is primarily for use by datum;
//! it exists in its own module both to keep datum.rs bite-sized
//! and for each in debugging (e.g., so you can use
//! `RUST_LOG=rustc::middle::trans::write_guard`).

use lib::llvm::ValueRef;
use middle::borrowck::{RootInfo, root_map_key, DynaImm, DynaMut};
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr;
use middle::ty;
use driver::session;
use syntax::codemap::span;
use syntax::ast;

pub fn root_and_write_guard(datum: &Datum,
                            mut bcx: block,
                            span: span,
                            expr_id: ast::node_id,
                            derefs: uint) -> block {
    let key = root_map_key { id: expr_id, derefs: derefs };
    debug!("write_guard::root_and_write_guard(key=%?)", key);

    // root the autoderef'd value, if necessary:
    //
    // (Note: root'd values are always boxes)
    let ccx = bcx.ccx();
    bcx = match ccx.maps.root_map.find(&key) {
        None => bcx,
        Some(&root_info) => root(datum, bcx, span, key, root_info)
    };

    // Perform the write guard, if necessary.
    //
    // (Note: write-guarded values are always boxes)
    if ccx.maps.write_guard_map.contains(&key) {
        perform_write_guard(datum, bcx, span)
    } else {
        bcx
    }
}

pub fn return_to_mut(mut bcx: block,
                     root_key: root_map_key,
                     frozen_val_ref: ValueRef,
                     bits_val_ref: ValueRef,
                     filename_val: ValueRef,
                     line_val: ValueRef) -> block {
    debug!("write_guard::return_to_mut(root_key=%?, %s, %s, %s)",
           root_key,
           bcx.to_str(),
           val_str(bcx.ccx().tn, frozen_val_ref),
           val_str(bcx.ccx().tn, bits_val_ref));

    let box_ptr =
        Load(bcx, PointerCast(bcx,
                              frozen_val_ref,
                              T_ptr(T_ptr(T_i8()))));

    let bits_val =
        Load(bcx, bits_val_ref);

    if bcx.tcx().sess.debug_borrows() {
        bcx = callee::trans_lang_call(
            bcx,
            bcx.tcx().lang_items.unrecord_borrow_fn(),
            [
                box_ptr,
                bits_val,
                filename_val,
                line_val
            ],
            expr::Ignore);
    }

    callee::trans_lang_call(
        bcx,
        bcx.tcx().lang_items.return_to_mut_fn(),
        [
            box_ptr,
            bits_val,
            filename_val,
            line_val
        ],
        expr::Ignore
    )
}

fn root(datum: &Datum,
        mut bcx: block,
        span: span,
        root_key: root_map_key,
        root_info: RootInfo) -> block {
    //! In some cases, borrowck will decide that an @T/@[]/@str
    //! value must be rooted for the program to be safe.  In that
    //! case, we will call this function, which will stash a copy
    //! away until we exit the scope `scope_id`.

    debug!("write_guard::root(root_key=%?, root_info=%?, datum=%?)",
           root_key, root_info, datum.to_str(bcx.ccx()));

    if bcx.sess().trace() {
        trans_trace(
            bcx, None,
            @fmt!("preserving until end of scope %d",
                  root_info.scope));
    }

    // First, root the datum. Note that we must zero this value,
    // because sometimes we root on one path but not another.
    // See e.g. #4904.
    let scratch = scratch_datum(bcx, datum.ty, true);
    datum.copy_to_datum(bcx, INIT, scratch);
    let cleanup_bcx = find_bcx_for_scope(bcx, root_info.scope);
    add_clean_temp_mem(cleanup_bcx, scratch.val, scratch.ty);

    // Now, consider also freezing it.
    match root_info.freeze {
        None => {}
        Some(freeze_kind) => {
            let (filename, line) = filename_and_line_num_from_span(bcx, span);

            // in this case, we don't have to zero, because
            // scratch.val will be NULL should the cleanup get
            // called without the freezing actually occurring, and
            // return_to_mut checks for this condition.
            let scratch_bits = scratch_datum(bcx, ty::mk_uint(), false);

            let freeze_did = match freeze_kind {
                DynaImm => bcx.tcx().lang_items.borrow_as_imm_fn(),
                DynaMut => bcx.tcx().lang_items.borrow_as_mut_fn(),
            };

            let box_ptr = Load(bcx,
                               PointerCast(bcx,
                                           scratch.val,
                                           T_ptr(T_ptr(T_i8()))));

            bcx = callee::trans_lang_call(
                bcx,
                freeze_did,
                [
                    box_ptr,
                    filename,
                    line
                ],
                expr::SaveIn(scratch_bits.val));

            if bcx.tcx().sess.debug_borrows() {
                bcx = callee::trans_lang_call(
                    bcx,
                    bcx.tcx().lang_items.record_borrow_fn(),
                    [
                        box_ptr,
                        Load(bcx, scratch_bits.val),
                        filename,
                        line
                    ],
                    expr::Ignore);
            }

            add_clean_return_to_mut(
                cleanup_bcx, root_key, scratch.val, scratch_bits.val,
                filename, line);
        }
    }

    bcx
}

fn perform_write_guard(datum: &Datum,
                       bcx: block,
                       span: span) -> block {
    debug!("perform_write_guard");

    let llval = datum.to_value_llval(bcx);
    let (filename, line) = filename_and_line_num_from_span(bcx, span);

    callee::trans_lang_call(
        bcx,
        bcx.tcx().lang_items.check_not_borrowed_fn(),
        [PointerCast(bcx, llval, T_ptr(T_i8())), filename, line],
        expr::Ignore)
}
