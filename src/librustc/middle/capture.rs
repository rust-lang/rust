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

use middle::freevars;
use middle::ty;

use core::option;
use core::vec;
use std::map::HashMap;
use std::map;
use syntax::codemap::span;
use syntax::{ast, ast_util};

export capture_mode;
export capture_var;
export capture_map;
export check_capture_clause;
export compute_capture_vars;
export cap_copy;
export cap_move;
export cap_drop;
export cap_ref;

enum capture_mode {
    cap_copy, // Copy the value into the closure.
    cap_move, // Move the value into the closure.
    cap_drop, // Drop value after creating closure.
    cap_ref,  // Reference directly from parent stack frame (block fn).
}

type capture_var = {
    def: ast::def,                       // Variable being accessed free
    span: span,                          // Location of access or cap item
    cap_item: Option<ast::capture_item>, // Capture item, if any
    mode: capture_mode                   // How variable is being accessed
};

type capture_map = map::HashMap<ast::def_id, capture_var>;

// checks the capture clause for a fn_expr() and issues warnings or
// errors for any irregularities which we identify.
fn check_capture_clause(tcx: ty::ctxt,
                        fn_expr_id: ast::node_id,
                        cap_clause: ast::capture_clause) {
    let freevars = freevars::get_freevars(tcx, fn_expr_id);
    let seen_defs = map::HashMap();

    for (*cap_clause).each |cap_item| {
        let cap_def = tcx.def_map.get(cap_item.id);
        if !vec::any(*freevars, |fv| fv.def == cap_def ) {
            tcx.sess.span_warn(
                cap_item.span,
                fmt!("captured variable `%s` not used in closure",
                     tcx.sess.str_of(cap_item.name)));
        }

        let cap_def_id = ast_util::def_id_of_def(cap_def).node;
        if !seen_defs.insert(cap_def_id, ()) {
            tcx.sess.span_err(
                cap_item.span,
                fmt!("variable `%s` captured more than once",
                     tcx.sess.str_of(cap_item.name)));
        }
    }
}

fn compute_capture_vars(tcx: ty::ctxt,
                        fn_expr_id: ast::node_id,
                        fn_proto: ast::Proto,
                        cap_clause: ast::capture_clause) -> ~[capture_var] {
    let freevars = freevars::get_freevars(tcx, fn_expr_id);
    let cap_map = map::HashMap();

    // first add entries for anything explicitly named in the cap clause

    for (*cap_clause).each |cap_item| {
        debug!("Doing capture var: %s (%?)",
               tcx.sess.str_of(cap_item.name), cap_item.id);

        let cap_def = tcx.def_map.get(cap_item.id);
        let cap_def_id = ast_util::def_id_of_def(cap_def).node;
        if cap_item.is_move {
            // if we are moving the value in, but it's not actually used,
            // must drop it.
            if vec::any(*freevars, |fv| fv.def == cap_def ) {
                cap_map.insert(cap_def_id, {def:cap_def,
                                            span: cap_item.span,
                                            cap_item: Some(*cap_item),
                                            mode:cap_move});
            } else {
                cap_map.insert(cap_def_id, {def:cap_def,
                                            span: cap_item.span,
                                            cap_item: Some(*cap_item),
                                            mode:cap_drop});
            }
        } else {
            // if we are copying the value in, but it's not actually used,
            // just ignore it.
            if vec::any(*freevars, |fv| fv.def == cap_def ) {
                cap_map.insert(cap_def_id, {def:cap_def,
                                            span: cap_item.span,
                                            cap_item: Some(*cap_item),
                                            mode:cap_copy});
            }
        }
    }

    // now go through anything that is referenced but was not explicitly
    // named and add that

    let implicit_mode_is_by_ref = fn_proto == ast::ProtoBorrowed;
    for vec::each(*freevars) |fvar| {
        let fvar_def_id = ast_util::def_id_of_def(fvar.def).node;
        match cap_map.find(fvar_def_id) {
            option::Some(_) => { /* was explicitly named, do nothing */ }
            option::None => {
                // Move if this type implicitly moves; copy otherwise.
                let mode;
                if implicit_mode_is_by_ref {
                    mode = cap_ref;
                } else {
                    let fvar_ty = ty::node_id_to_type(tcx, fvar_def_id);
                    if ty::type_implicitly_moves(tcx, fvar_ty) {
                        mode = cap_move;
                    } else {
                        mode = cap_copy;
                    }
                };

                cap_map.insert(fvar_def_id, {def:fvar.def,
                                             span: fvar.span,
                                             cap_item: None,
                                             mode:mode});
            }
        }
    }

    let mut result = ~[];
    for cap_map.each_value |cap_var| { result.push(cap_var); }
    return result;
}
