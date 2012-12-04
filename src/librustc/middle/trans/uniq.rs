// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use lib::llvm::ValueRef;
use common::*;
use build::*;
use base::*;
use datum::immediate_rvalue;

export make_free_glue, autoderef, duplicate;

fn make_free_glue(bcx: block, vptrptr: ValueRef, box_ty: ty::t)
    -> block {
    let _icx = bcx.insn_ctxt("uniq::make_free_glue");
    let box_datum = immediate_rvalue(Load(bcx, vptrptr), box_ty);

    let not_null = IsNotNull(bcx, box_datum.val);
    do with_cond(bcx, not_null) |bcx| {
        let body_datum = box_datum.box_body(bcx);
        let bcx = glue::drop_ty(bcx, body_datum.to_ref_llval(bcx),
                                body_datum.ty);
        glue::trans_unique_free(bcx, box_datum.val)
    }
}

fn duplicate(bcx: block, src_box: ValueRef, src_ty: ty::t) -> Result {
    let _icx = bcx.insn_ctxt("uniq::duplicate");

    // Load the body of the source (*src)
    let src_datum = immediate_rvalue(src_box, src_ty);
    let body_datum = src_datum.box_body(bcx);

    // Malloc space in exchange heap and copy src into it
    let {bcx: bcx, box: dst_box, body: dst_body} =
        malloc_unique(bcx, body_datum.ty);
    body_datum.copy_to(bcx, datum::INIT, dst_body);

    // Copy the type descriptor
    let src_tydesc_ptr = GEPi(bcx, src_box,
                              [0u, back::abi::box_field_tydesc]);
    let dst_tydesc_ptr = GEPi(bcx, dst_box,
                              [0u, back::abi::box_field_tydesc]);
    let td = Load(bcx, src_tydesc_ptr);
    Store(bcx, td, dst_tydesc_ptr);

    return rslt(bcx, dst_box);
}
