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
use util::common::indenter;
use util::ppaux::ty_to_str;

pub enum CaptureMode {
    CapCopy, // Copy the value into the closure.
    CapMove, // Move the value into the closure.
    CapRef,  // Reference directly from parent stack frame (used by `&fn()`).
}

pub struct CaptureVar {
    def: ast::def,    // Variable being accessed free
    span: span,       // Location of an access to this variable
    mode: CaptureMode // How variable is being accessed
}

pub type CaptureMap = map::HashMap<ast::def_id, CaptureVar>;

pub fn compute_capture_vars(tcx: ty::ctxt,
                            fn_expr_id: ast::node_id,
                            fn_proto: ast::Proto) -> ~[CaptureVar] {
    debug!("compute_capture_vars(fn_expr_id=%?, fn_proto=%?)",
           fn_expr_id, fn_proto);
    let _indenter = indenter();

    let freevars = freevars::get_freevars(tcx, fn_expr_id);
    return if fn_proto == ast::ProtoBorrowed {
        // &fn() captures everything by ref
        do freevars.map |fvar| {
            CaptureVar {def: fvar.def, span: fvar.span, mode: CapRef}
        }
    } else {
        // @fn() and ~fn() capture by copy or by move depending on type
        do freevars.map |fvar| {
            let fvar_def_id = ast_util::def_id_of_def(fvar.def).node;
            let fvar_ty = ty::node_id_to_type(tcx, fvar_def_id);
            debug!("fvar_def_id=%? fvar_ty=%s",
                  fvar_def_id, ty_to_str(tcx, fvar_ty));
            let mode = if ty::type_implicitly_moves(tcx, fvar_ty) {
                CapMove
            } else {
                CapCopy
            };
            CaptureVar {def: fvar.def, span: fvar.span, mode:mode}
        }
    };
}
